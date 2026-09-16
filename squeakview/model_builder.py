from __future__ import annotations

"""Project-scoped YOLO26 pose builder used by the pre-acquisition GUI.

Heavy CUDA, TensorRT, PyTorch, ONNX, and Ultralytics imports deliberately live
inside :func:`build_model_package`.  The project/setup GUI imports this module
only for source discovery and never creates a CUDA context of its own.
"""

import hashlib
import json
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml

from squeakview.model_build import (
    cleanup_staging_package,
    cleanup_stale_staging_packages,
    create_staging_package,
    promote_model_package,
    run_trtexec_validation,
)
from squeakview.model_package import validate_model_package
from squeakview.project import AppPaths, Project, ProjectSession, set_default_model


BuildEventEmitter = Callable[[str, str], None]


@dataclass(frozen=True, slots=True)
class ModelSource:
    name: str
    root: Path
    checkpoint: Path
    data_yaml: Path


@dataclass(frozen=True, slots=True)
class BuildSpec:
    project_root: Path
    source_name: str
    model_name: str
    overwrite: bool = False
    set_as_default: bool = True
    precision: str = "fp16"
    batch_size: int = 1
    image_size: int = 640
    device: int = 0
    tensorrt_workspace_gib: float = 1.0
    confidence_threshold: float = 0.25
    keypoint_threshold: float = 0.50


@dataclass(frozen=True, slots=True)
class BuildResult:
    project_root: Path
    package_root: Path
    config: Path
    model_name: str


def _one_component(value: str, *, label: str) -> str:
    candidate = value.strip()
    if not candidate or Path(candidate).name != candidate or candidate in {".", ".."}:
        raise ValueError(f"{label} must be one non-empty path component")
    return candidate


def discover_model_sources(project: Project) -> tuple[ModelSource, ...]:
    """Return complete direct-child checkpoint/YAML pairs without following links."""

    choices: list[ModelSource] = []
    try:
        candidates = sorted(
            (
                path
                for path in project.paths.model_sources.iterdir()
                if path.is_dir()
                and not path.is_symlink()
                and not path.name.startswith(".")
            ),
            key=lambda path: path.name.casefold(),
        )
    except OSError:
        return ()
    for candidate in candidates[:256]:
        try:
            checkpoints = sorted(
                path
                for path in candidate.iterdir()
                if path.is_file() and not path.is_symlink() and path.suffix == ".pt"
            )
            yamls = sorted(
                path
                for path in candidate.iterdir()
                if path.is_file()
                and not path.is_symlink()
                and path.suffix.lower() in {".yaml", ".yml"}
            )
        except OSError:
            continue
        if len(checkpoints) == 1 and len(yamls) == 1:
            choices.append(
                ModelSource(
                    name=candidate.name,
                    root=candidate.resolve(),
                    checkpoint=checkpoints[0].resolve(),
                    data_yaml=yamls[0].resolve(),
                )
            )
    return tuple(choices)


def import_model_source(
    project: Project,
    *,
    name: str,
    checkpoint: Path,
    data_yaml: Path,
) -> ModelSource:
    """Transactionally copy one external checkpoint/YAML pair into a project.

    The caller must hold the project's writer lock. Existing source packages
    are never replaced.
    """

    source_name = _one_component(name, label="model source name")
    checkpoint_input = Path(checkpoint).expanduser()
    data_yaml_input = Path(data_yaml).expanduser()
    if checkpoint_input.is_symlink() or data_yaml_input.is_symlink():
        raise ValueError("model source inputs may not be symbolic links")
    checkpoint = checkpoint_input.resolve(strict=True)
    data_yaml = data_yaml_input.resolve(strict=True)
    if not checkpoint.is_file() or checkpoint.suffix != ".pt":
        raise ValueError("model checkpoint must be a regular .pt file")
    if (
        not data_yaml.is_file()
        or data_yaml.suffix.lower() not in {".yaml", ".yml"}
    ):
        raise ValueError("model data file must be a regular YAML file")
    destination = project.paths.model_sources / source_name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"model source already exists: {destination}")
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{source_name}.import-",
            dir=project.paths.model_sources,
        )
    )
    published = False
    try:
        staging.chmod(0o700)
        copied_checkpoint = staging / checkpoint.name
        copied_yaml = staging / data_yaml.name
        shutil.copyfile(checkpoint, copied_checkpoint)
        shutil.copyfile(data_yaml, copied_yaml)
        copied_checkpoint.chmod(0o600)
        copied_yaml.chmod(0o600)
        staging.rename(destination)
        published = True
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)
    return _source_for(project, source_name)


def _source_for(project: Project, source_name: str) -> ModelSource:
    source_name = _one_component(source_name, label="model source")
    source = next(
        (item for item in discover_model_sources(project) if item.name == source_name),
        None,
    )
    if source is None:
        raise ValueError(
            f"model source '{source_name}' must contain exactly one .pt and one YAML file"
        )
    project.paths.resolve_path(source.checkpoint, within=project.paths.model_sources, must_exist=True)
    project.paths.resolve_path(source.data_yaml, within=project.paths.model_sources, must_exist=True)
    return source


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _ordered_labels(value: object) -> list[str]:
    if isinstance(value, dict):
        def sort_key(key: object) -> tuple[int, int | str]:
            return (0, int(key)) if str(key).isdigit() else (1, str(key))

        return [str(value[key]) for key in sorted(value, key=sort_key)]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return []


def _class_value(mapping: object, class_id: int, class_name: str) -> object:
    if not isinstance(mapping, dict):
        return None
    for key in (class_id, str(class_id), class_name):
        if key in mapping:
            return mapping[key]
    return None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_model_package(
    spec: BuildSpec,
    *,
    emit: BuildEventEmitter | None = None,
) -> BuildResult:
    """Build, validate, and atomically publish one project-owned model package."""

    notify = emit or (lambda _stage, _message: None)
    model_name = _one_component(spec.model_name, label="model name")
    _require(spec.precision in {"fp16", "fp32"}, "precision must be fp16 or fp32")
    _require(spec.batch_size > 0, "batch size must be positive")
    _require(spec.image_size > 0, "image size must be positive")
    _require(
        spec.tensorrt_workspace_gib > 0,
        "TensorRT workspace limit must be positive",
    )
    app = AppPaths.discover()
    staging_package: Path | None = None

    notify("locking", "Opening the project exclusively for model construction")
    with ProjectSession.open(Path(spec.project_root)) as session:
        project = session.project
        app.validate_for_project(project.paths)
        source = _source_for(project, spec.source_name)
        parser_library = (
            app.native / "nvdsinfer_custom_impl_yolo/libnvdsinfer_custom_impl_Yolo.so"
        )
        if not parser_library.is_file():
            raise FileNotFoundError(
                f"DeepStream parser is not built: {parser_library}. "
                "Run bash scripts/build_native.sh first."
            )
        final_package = project.paths.models / model_name
        if final_package.exists() and not spec.overwrite:
            raise FileExistsError(
                f"model package already exists and was preserved: {final_package}"
            )

        notify("recovering", "Removing an incomplete prior staging build, if present")
        cleanup_stale_staging_packages(project.paths.models, model_name)
        staging_package = create_staging_package(project.paths.models, model_name)
        try:
            notify("loading", "Loading CUDA, TensorRT, and the selected checkpoint")
            import onnx
            import tensorrt as trt
            import torch
            import ultralytics
            from ultralytics import YOLO

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required for TensorRT export")
            data_config = yaml.safe_load(source.data_yaml.read_text(encoding="utf-8")) or {}
            model = YOLO(str(source.checkpoint))
            head = model.model.model[-1]
            checkpoint_classes = [str(model.names[index]) for index in sorted(model.names)]
            class_names = _ordered_labels(data_config.get("names")) or checkpoint_classes
            checkpoint_kpt_shape = [int(value) for value in head.kpt_shape]
            yaml_kpt_shape = data_config.get("kpt_shape")
            keypoint_count, keypoint_dims = map(
                int, yaml_kpt_shape or checkpoint_kpt_shape
            )
            _require(model.task == "pose", f"expected a pose checkpoint, found {model.task}")
            one2one_head = getattr(head, "one2one", None)
            _require(
                isinstance(one2one_head, dict) and bool(one2one_head),
                "expected a YOLO26 pose checkpoint with a selectable one-to-one head",
            )
            _require(
                checkpoint_classes == class_names,
                f"checkpoint/YAML classes differ: {checkpoint_classes!r} != {class_names!r}",
            )
            _require(
                checkpoint_kpt_shape == [keypoint_count, keypoint_dims],
                "checkpoint/YAML keypoint shapes differ",
            )
            _require(keypoint_dims == 3, "pose keypoints must use x/y/confidence")

            keypoint_names: list[str] | None = None
            keypoint_name_source = "generated"
            for source_label, value in (
                ("yaml.kp_names", data_config.get("kp_names")),
                ("yaml.keypoint_names", data_config.get("keypoint_names")),
                ("yaml.kpt_names", data_config.get("kpt_names")),
                ("checkpoint.kpt_names", getattr(model, "kpt_names", None)),
            ):
                candidates: list[list[str]] = []
                if isinstance(value, (list, tuple)):
                    candidates = [[str(item) for item in value]]
                elif isinstance(value, dict):
                    for class_id, class_name in enumerate(class_names):
                        candidate = _class_value(value, class_id, class_name)
                        if isinstance(candidate, (list, tuple)):
                            candidates.append([str(item) for item in candidate])
                valid = [item for item in candidates if len(item) == keypoint_count]
                if valid and all(item == valid[0] for item in valid):
                    if source_label.startswith("checkpoint") and all(
                        name.isdigit() for name in valid[0]
                    ):
                        continue
                    keypoint_names = valid[0]
                    keypoint_name_source = source_label
                    break
            if keypoint_names is None:
                keypoint_names = [f"keypoint_{index}" for index in range(keypoint_count)]
            _require(len(set(keypoint_names)) == keypoint_count, "keypoint names must be unique")

            pose_classes = [
                {
                    "id": class_id,
                    "name": class_name,
                    "threshold": spec.confidence_threshold,
                    "track": class_id == 0,
                    "keypoint_indices": list(range(keypoint_count)),
                }
                for class_id, class_name in enumerate(class_names)
            ]
            expected_output = [spec.batch_size, 300, 6 + keypoint_count * keypoint_dims]

            weights_dir = staging_package / "weights"
            onnx_dir = staging_package / "onnx"
            engines_dir = staging_package / "engines"
            labels_dir = staging_package / "labels"
            lib_dir = staging_package / "lib"
            configs_dir = staging_package / "configs"
            validation_dir = staging_package / "validation"
            for directory in (
                weights_dir,
                onnx_dir,
                engines_dir,
                labels_dir,
                lib_dir,
                configs_dir,
                validation_dir,
            ):
                directory.mkdir(parents=True, exist_ok=True)

            artifact_stem = f"{source.checkpoint.stem}_{spec.precision}_b{spec.batch_size}"
            packaged_model = weights_dir / source.checkpoint.name
            packaged_parser = lib_dir / parser_library.name
            onnx_path = onnx_dir / f"{artifact_stem}.onnx"
            engine_path = engines_dir / f"{artifact_stem}.engine"
            shutil.copy2(source.checkpoint, packaged_model)
            shutil.copy2(parser_library, packaged_parser)

            notify(
                "exporting",
                "Building the TensorRT engine; this is the longest stage and has no reliable percentage",
            )
            export_model = YOLO(str(packaged_model))
            ultralytics_engine = Path(
                export_model.export(
                    format="engine",
                    device=spec.device,
                    imgsz=spec.image_size,
                    batch=spec.batch_size,
                    dynamic=False,
                    quantize=16 if spec.precision == "fp16" else 32,
                    simplify=True,
                    nms=False,
                    # Jetson memory is shared by the CPU and GPU. Leaving this
                    # unset lets TensorRT consider tactics whose workspace can
                    # exhaust an 8 GiB Orin Nano during engine construction.
                    workspace=spec.tensorrt_workspace_gib,
                    data=str(source.data_yaml),
                )
            ).resolve()
            exported_onnx = packaged_model.with_suffix(".onnx")
            _require(ultralytics_engine.is_file(), "Ultralytics did not produce an engine")
            _require(exported_onnx.is_file(), "Ultralytics did not retain the ONNX graph")
            shutil.move(str(exported_onnx), onnx_path)

            with ultralytics_engine.open("rb") as engine_file:
                metadata_size = int.from_bytes(engine_file.read(4), "little", signed=True)
                _require(
                    0 < metadata_size < 16 * 1024 * 1024,
                    "Ultralytics engine metadata prefix is invalid",
                )
                engine_metadata = json.loads(engine_file.read(metadata_size))
                raw_plan = engine_file.read()
            _require(engine_metadata.get("end2end") is True, "engine is not end-to-end")
            _require(
                engine_metadata.get("args", {}).get("nms") is False,
                "engine unexpectedly contains Ultralytics NMS",
            )
            _require(bool(raw_plan), "the exported TensorRT plan is empty")
            engine_path.write_bytes(raw_plan)
            ultralytics_engine.unlink()

            notify("validating", "Validating ONNX structure and TensorRT execution")
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)

            def tensor_shape(value_info: object) -> list[int | str]:
                return [
                    int(dim.dim_value) if dim.dim_value else dim.dim_param
                    for dim in value_info.type.tensor_type.shape.dim
                ]

            input_shape = tensor_shape(onnx_model.graph.input[0])
            output_shapes = [tensor_shape(output) for output in onnx_model.graph.output]
            _require(
                input_shape == [spec.batch_size, 3, spec.image_size, spec.image_size],
                f"unexpected ONNX input shape: {input_shape}",
            )
            _require(output_shapes == [expected_output], f"unexpected ONNX outputs: {output_shapes}")
            runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
            _require(
                runtime.deserialize_cuda_engine(raw_plan) is not None,
                "TensorRT could not deserialize the exported engine",
            )

            classes_path = labels_dir / "classes.txt"
            keypoints_path = labels_dir / "labels.txt"
            classes_path.write_text("\n".join(class_names) + "\n", encoding="utf-8")
            keypoints_path.write_text("\n".join(keypoint_names) + "\n", encoding="utf-8")
            config_path = configs_dir / f"{model_name}.txt"
            network_mode = 2 if spec.precision == "fp16" else 0
            config_path.write_text(
                "\n".join(
                    [
                        "[property]",
                        "gpu-id=0",
                        "net-scale-factor=0.00392156862745098",
                        "model-color-format=0",
                        f"onnx-file=../onnx/{onnx_path.name}",
                        f"model-engine-file=../engines/{engine_path.name}",
                        f"network-mode={network_mode}",
                        "network-type=0",
                        f"infer-dims=3;{spec.image_size};{spec.image_size}",
                        f"batch-size={spec.batch_size}",
                        "output-tensor-meta=1",
                        f"num-detected-classes={len(class_names)}",
                        "labelfile-path=../labels/classes.txt",
                        "parse-bbox-func-name=NvDsInferParseYolo26Pose",
                        f"custom-lib-path=../lib/{packaged_parser.name}",
                        "cluster-mode=4",
                        "maintain-aspect-ratio=1",
                        "symmetric-padding=1",
                        "gie-unique-id=1",
                        "interval=0",
                        "process-mode=1",
                        "",
                        "[class-attrs-all]",
                        f"pre-cluster-threshold={spec.confidence_threshold}",
                        "topk=300",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            output_layer = onnx_model.graph.output[0].name
            pose_schema = {
                "schema_version": 2,
                "task": "pose",
                "postprocess": "pyservicemaker_yolo26_pose_v1",
                "output_layer": output_layer,
                "input_width": spec.image_size,
                "input_height": spec.image_size,
                "letterbox": "symmetric",
                "end2end": True,
                "keypoint_labels_path": "../labels/labels.txt",
                "keypoint_count": keypoint_count,
                "keypoint_dims": keypoint_dims,
                "keypoint_threshold": spec.keypoint_threshold,
                "classes": pose_classes,
            }
            pose_path = configs_dir / f"{model_name}.pose.json"
            pose_path.write_text(json.dumps(pose_schema, indent=2) + "\n", encoding="utf-8")
            data_reference = project.paths.portable_path(source.data_yaml)
            dataset_sha256 = _file_sha256(source.data_yaml)

            def artifact_identity(path: Path) -> dict[str, str]:
                return {
                    "path": path.relative_to(staging_package).as_posix(),
                    "sha256": _file_sha256(path),
                }

            engine_execution = run_trtexec_validation(engine_path)
            _require(
                bool(engine_execution["passed"]),
                f"TensorRT execution validation failed: {engine_execution}",
            )
            build_environment = {
                "tensorrt_version": str(trt.__version__),
                "cuda_version": str(torch.version.cuda),
                "gpu_name": str(torch.cuda.get_device_name(spec.device)),
                "compute_capability": list(torch.cuda.get_device_capability(spec.device)),
                "device_model": Path("/proc/device-tree/model").read_text(
                    encoding="utf-8", errors="replace"
                ).replace("\x00", "").strip(),
                "jetson_linux_release": Path("/etc/nv_tegra_release").read_text(
                    encoding="utf-8", errors="replace"
                ).strip(),
                "ultralytics_version": str(ultralytics.__version__),
                "torch_version": str(torch.__version__),
                "tensorrt_workspace_gib": spec.tensorrt_workspace_gib,
                "keypoint_name_source": keypoint_name_source,
            }
            report_path = validation_dir / "import_report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "onnx_input_shape": input_shape,
                        "onnx_output_shapes": output_shapes,
                        "output_layer": output_layer,
                        "ultralytics_engine_metadata": engine_metadata,
                        "build_environment": build_environment,
                        "checks": {
                            "onnx": True,
                            "raw_engine": True,
                            "yaml_labels": True,
                            "schema_v2": True,
                            "engine_identity": True,
                            "engine_execution": True,
                        },
                        "engine_execution": engine_execution,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            artifact_identities = {
                "config": artifact_identity(config_path),
                "pose_schema": artifact_identity(pose_path),
                "onnx": artifact_identity(onnx_path),
                "engine": artifact_identity(engine_path),
                "class_labels": artifact_identity(classes_path),
                "keypoint_labels": artifact_identity(keypoints_path),
                "custom_parser": artifact_identity(packaged_parser),
                "import_report": artifact_identity(report_path),
            }
            manifest_path = staging_package / "model.yaml"
            manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 3,
                        "name": model_name,
                        "framework": "yolo26",
                        "task": "pose",
                        "precision": spec.precision,
                        "batch_size": spec.batch_size,
                        "classes": class_names,
                        "keypoints": keypoint_names,
                        "artifacts": artifact_identities,
                        "export": {
                            "builder": "ultralytics",
                            "tensorrt_workspace_gib": spec.tensorrt_workspace_gib,
                            "data": data_reference,
                            "data_sha256": dataset_sha256,
                            "end2end": True,
                        },
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            validate_model_package(config_path)
            notify("publishing", "Publishing the validated package atomically")
            published_config = promote_model_package(
                staging_package,
                final_package,
                config_name=config_path.name,
                overwrite=spec.overwrite,
            )
            if spec.set_as_default:
                project = set_default_model(project, model_name)
            notify("complete", f"Model package is ready: {final_package}")
            return BuildResult(
                project_root=project.paths.root,
                package_root=final_package,
                config=published_config,
                model_name=model_name,
            )
        finally:
            if staging_package is not None:
                cleanup_staging_package(staging_package)


__all__ = [
    "BuildResult",
    "BuildSpec",
    "ModelSource",
    "build_model_package",
    "discover_model_sources",
    "import_model_source",
]
