from __future__ import annotations

"""Validation and identity helpers for SqueakView model packages."""

import argparse
import configparser
import hashlib
import json
import math
import os
import re
import stat
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from squeakview.common.bounded_input import read_json_object, read_stable_regular_file


_MAX_CONFIG_BYTES = 1024 * 1024
_MAX_METADATA_BYTES = 4 * 1024 * 1024
_MAX_LABEL_BYTES = 1024 * 1024
_MAX_MODEL_ARTIFACT_BYTES = 4 * 1024 * 1024 * 1024


class ModelPackageError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ModelPackageInfo:
    name: str
    root: Path
    config: Path
    manifest: Path
    pose_sidecar: Path
    onnx: Path
    engine: Path
    classes: Path
    keypoint_labels: Path
    parser_library: Path
    import_report: Path
    model_manifest_sha256: str
    pose_sidecar_sha256: str
    onnx_sha256: str
    config_sha256: str
    engine_sha256: str
    model_manifest_schema: int
    engine_build_identity: dict[str, object] | None

    def manifest_snapshot(self) -> dict[str, object]:
        payload = asdict(self)
        return {key: str(value) if isinstance(value, Path) else value for key, value in payload.items()}


def _sha256(path: Path, *, max_bytes: int, label: str) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ModelPackageError(f"{label} must be a regular file: {path}")
            if before.st_size > max_bytes:
                raise ModelPackageError(
                    f"{label} exceeds {max_bytes} byte limit: {path}"
                )
            remaining = before.st_size
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                digest.update(chunk)
                remaining -= len(chunk)
            grew = bool(handle.read(1))
            after = os.fstat(handle.fileno())
        current = path.stat()
    except OSError as exc:
        raise ModelPackageError(f"Could not hash {label} {path}: {exc}") from exc
    if (
        remaining
        or grew
        or after.st_dev != before.st_dev
        or after.st_ino != before.st_ino
        or after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
        or current.st_dev != before.st_dev
        or current.st_ino != before.st_ino
        or current.st_size != before.st_size
        or current.st_mtime_ns != before.st_mtime_ns
    ):
        raise ModelPackageError(f"{label} changed while being hashed: {path}")
    return digest.hexdigest()


def _resolve_config_path(raw: str, config_dir: Path) -> Path:
    path = Path(raw.strip().strip('"')).expanduser()
    return path.resolve() if path.is_absolute() else (config_dir / path).resolve()


def _load_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = read_json_object(path, max_bytes=_MAX_METADATA_BYTES, label=label)
    except ValueError as exc:
        raise ModelPackageError(f"Could not read {label} {path}: {exc}") from exc
    return value


def _runtime_engine_identity() -> dict[str, object]:
    """Return the runtime identity that constrains a generated TensorRT plan."""

    try:
        import tensorrt as trt
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        compute_capability = list(torch.cuda.get_device_capability(0))
        gpu_name = torch.cuda.get_device_name(0)
    except Exception as exc:
        raise ModelPackageError(
            f"TensorRT engine runtime identity could not be verified: {exc}"
        ) from exc

    def read_optional(path: Path) -> str | None:
        try:
            return path.read_text(errors="replace").replace("\x00", "").strip() or None
        except OSError:
            return None

    return {
        "tensorrt_version": str(trt.__version__),
        "cuda_version": str(torch.version.cuda),
        "gpu_name": str(gpu_name),
        "compute_capability": compute_capability,
        "device_model": read_optional(Path("/proc/device-tree/model")),
        "jetson_linux_release": read_optional(Path("/etc/nv_tegra_release")),
    }


def _validate_engine_build_identity(
    recorded: object, runtime: dict[str, object]
) -> dict[str, object]:
    if not isinstance(recorded, dict):
        raise ModelPackageError(
            "Schema-3 model package is missing TensorRT engine build_environment"
        )
    required = (
        "tensorrt_version",
        "cuda_version",
        "gpu_name",
        "compute_capability",
        "device_model",
        "jetson_linux_release",
    )
    missing = [name for name in required if recorded.get(name) in (None, "", [])]
    if missing:
        raise ModelPackageError(
            "TensorRT engine build identity is incomplete: " + ", ".join(missing)
        )
    mismatches = [
        f"{name} built={recorded.get(name)!r} runtime={runtime.get(name)!r}"
        for name in required
        if recorded.get(name) != runtime.get(name)
    ]
    if mismatches:
        raise ModelPackageError(
            "TensorRT engine was built for a different runtime/device: "
            + "; ".join(mismatches)
        )
    return dict(recorded)


def _validate_schema_three_artifacts(
    recorded: object,
    root: Path,
    expected: dict[str, Path],
) -> None:
    """Bind every runtime package artifact to its path and digest."""

    if not isinstance(recorded, dict):
        raise ModelPackageError("Schema-3 model package is missing artifact identities")
    limits = {
        "config": _MAX_CONFIG_BYTES,
        "pose_schema": _MAX_METADATA_BYTES,
        "onnx": _MAX_MODEL_ARTIFACT_BYTES,
        "engine": _MAX_MODEL_ARTIFACT_BYTES,
        "class_labels": _MAX_LABEL_BYTES,
        "keypoint_labels": _MAX_LABEL_BYTES,
        "custom_parser": _MAX_MODEL_ARTIFACT_BYTES,
        "import_report": _MAX_METADATA_BYTES,
    }
    for name, expected_path in expected.items():
        item = recorded.get(name)
        if not isinstance(item, dict):
            raise ModelPackageError(f"Schema-3 artifact identity {name!r} is missing")
        relative = item.get("path")
        digest = item.get("sha256")
        if not isinstance(relative, str) or not relative.strip():
            raise ModelPackageError(f"Schema-3 artifact {name!r} path is invalid")
        resolved = (root / relative).resolve()
        if resolved != expected_path.resolve() or root not in resolved.parents:
            raise ModelPackageError(
                f"Schema-3 artifact {name!r} path does not match the runtime package"
            )
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ModelPackageError(f"Schema-3 artifact {name!r} SHA-256 is invalid")
        if _sha256(
            resolved, max_bytes=limits[name], label=f"Schema-3 artifact {name}"
        ) != digest:
            raise ModelPackageError(
                f"Schema-3 artifact {name!r} does not match its recorded SHA-256"
            )


def _read_labels(path: Path, label: str) -> list[str]:
    try:
        raw = read_stable_regular_file(
            path, max_bytes=_MAX_LABEL_BYTES, label=label
        )
        labels = [
            line.strip()
            for line in raw.decode("utf-8").splitlines()
            if line.strip()
        ]
    except (ValueError, UnicodeDecodeError) as exc:
        raise ModelPackageError(f"Could not read {label} {path}: {exc}") from exc
    if not labels or len(labels) != len(set(labels)):
        raise ModelPackageError(f"{label.capitalize()} must contain unique, non-empty labels: {path}")
    return labels


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ModelPackageError(f"{label} must be an integer")
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ModelPackageError(f"{label} must be an integer")
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()) is None:
        raise ModelPackageError(f"{label} must be an integer")
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ModelPackageError(f"{label} must be an integer") from exc


def _probability(value: object, label: str) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ModelPackageError(f"{label} must be numeric") from exc
    if not 0.0 <= number <= 1.0:
        raise ModelPackageError(f"{label} must be between 0 and 1")
    return number


def _validate_pose_schema(
    sidecar: dict[str, object],
    *,
    class_names: list[str],
    keypoint_names: list[str],
    properties: configparser.SectionProxy,
) -> tuple[int, int, int]:
    version = _integer(sidecar.get("schema_version"), "Pose sidecar schema_version")
    if version != 2:
        raise ModelPackageError(
            f"Pose sidecar schema_version must be 2; rebuild this legacy schema-{version} package"
        )
    required_values = {
        "task": "pose",
        "postprocess": "pyservicemaker_yolo26_pose_v1",
        "letterbox": "symmetric",
        "end2end": True,
    }
    for key, expected in required_values.items():
        if sidecar.get(key) != expected:
            raise ModelPackageError(f"Pose sidecar field {key!r} must be {expected!r}")
    if not str(sidecar.get("output_layer", "")).strip():
        raise ModelPackageError("Pose sidecar output_layer must not be empty")

    width = _integer(sidecar.get("input_width"), "Pose sidecar input_width")
    height = _integer(sidecar.get("input_height"), "Pose sidecar input_height")
    count = _integer(sidecar.get("keypoint_count"), "Pose sidecar keypoint_count")
    dimensions = _integer(sidecar.get("keypoint_dims"), "Pose sidecar keypoint_dims")
    if width <= 0 or height <= 0:
        raise ModelPackageError("Pose sidecar input dimensions must be positive")
    if count != len(keypoint_names):
        raise ModelPackageError(
            f"Pose sidecar declares {count} keypoints but labels contain {len(keypoint_names)}"
        )
    if dimensions != 3:
        raise ModelPackageError("YOLO26 pose packages require keypoint_dims=3")
    _probability(sidecar.get("keypoint_threshold"), "Pose sidecar keypoint_threshold")

    infer_dims = [part.strip() for part in properties.get("infer-dims", "").split(";")]
    if infer_dims != ["3", str(height), str(width)]:
        raise ModelPackageError("DeepStream infer-dims does not match the pose sidecar")
    if properties.getint("num-detected-classes", fallback=-1) != len(class_names):
        raise ModelPackageError("DeepStream num-detected-classes does not match class labels")

    raw_classes = sidecar.get("classes")
    if not isinstance(raw_classes, list) or len(raw_classes) != len(class_names):
        raise ModelPackageError("Pose sidecar classes must contain one entry per class label")
    seen_ids: set[int] = set()
    covered_keypoints: set[int] = set()
    for entry in raw_classes:
        if not isinstance(entry, dict):
            raise ModelPackageError("Every pose sidecar class entry must be an object")
        class_id = _integer(entry.get("id"), "Pose class id")
        if class_id < 0 or class_id >= len(class_names) or class_id in seen_ids:
            raise ModelPackageError(f"Pose sidecar has invalid or duplicate class id {class_id}")
        seen_ids.add(class_id)
        if entry.get("name") != class_names[class_id]:
            raise ModelPackageError(
                f"Pose sidecar class {class_id} must be named {class_names[class_id]!r}"
            )
        if not isinstance(entry.get("track"), bool):
            raise ModelPackageError(f"Pose sidecar class {class_id} track must be boolean")
        _probability(entry.get("threshold"), f"Pose sidecar class {class_id} threshold")
        raw_indices = entry.get("keypoint_indices")
        if not isinstance(raw_indices, list):
            raise ModelPackageError(f"Pose sidecar class {class_id} keypoint_indices must be a list")
        indices = [_integer(value, "Pose keypoint index") for value in raw_indices]
        if len(indices) != len(set(indices)):
            raise ModelPackageError(f"Pose sidecar class {class_id} has duplicate keypoint indices")
        if any(index < 0 or index >= count for index in indices):
            raise ModelPackageError(f"Pose sidecar class {class_id} has an out-of-range keypoint index")
        covered_keypoints.update(indices)
    if seen_ids != set(range(len(class_names))):
        raise ModelPackageError("Pose sidecar class ids must be contiguous from zero")
    if covered_keypoints != set(range(count)):
        missing = sorted(set(range(count)) - covered_keypoints)
        raise ModelPackageError(f"Pose sidecar does not assign keypoint indices {missing} to a class")

    return width, height, count


def validate_model_package(config_path: str | Path) -> ModelPackageInfo:
    config = Path(config_path).expanduser().resolve()
    if not config.is_file():
        raise ModelPackageError(f"DeepStream model config does not exist: {config}")
    if config.parent.name != "configs":
        raise ModelPackageError(f"Model config must be inside a model package configs/ directory: {config}")

    root = config.parent.parent
    parser = configparser.ConfigParser(interpolation=None, strict=True)
    try:
        config_text = read_stable_regular_file(
            config, max_bytes=_MAX_CONFIG_BYTES, label="model config"
        ).decode("utf-8")
        parser.read_string(config_text, source=str(config))
    except (ValueError, UnicodeDecodeError, configparser.Error) as exc:
        raise ModelPackageError(f"Could not read DeepStream model config {config}: {exc}") from exc
    if not parser.has_section("property"):
        raise ModelPackageError(f"DeepStream model config is missing [property]: {config}")

    properties = parser["property"]
    required_keys = (
        "onnx-file", "model-engine-file", "labelfile-path", "custom-lib-path",
        "infer-dims", "batch-size", "num-detected-classes", "parse-bbox-func-name",
    )
    missing_keys = [key for key in required_keys if not properties.get(key, "").strip()]
    if missing_keys:
        raise ModelPackageError(f"Model config is missing required keys: {', '.join(missing_keys)}")

    pose_sidecar = config.with_name(f"{config.stem}.pose.json")
    manifest = root / "model.yaml"
    onnx = _resolve_config_path(properties["onnx-file"], config.parent)
    engine = _resolve_config_path(properties["model-engine-file"], config.parent)
    classes = _resolve_config_path(properties["labelfile-path"], config.parent)
    parser_library = _resolve_config_path(properties["custom-lib-path"], config.parent)
    keypoint_labels = root / "labels" / "labels.txt"
    import_report = root / "validation" / "import_report.json"

    sidecar_data: dict[str, object] = {}
    if pose_sidecar.is_file():
        sidecar_data = _load_json_object(pose_sidecar, "pose sidecar")
        raw_labels = sidecar_data.get("keypoint_labels_path")
        if raw_labels:
            keypoint_labels = _resolve_config_path(str(raw_labels), pose_sidecar.parent)

    required_files = {
        "model manifest": manifest,
        "pose sidecar": pose_sidecar,
        "ONNX model": onnx,
        "TensorRT engine": engine,
        "class labels": classes,
        "keypoint labels": keypoint_labels,
        "custom parser library": parser_library,
        "import report": import_report,
    }
    missing_files = [f"{label}: {path}" for label, path in required_files.items() if not path.is_file()]
    if missing_files:
        raise ModelPackageError("Model package is incomplete:\n" + "\n".join(missing_files))

    for label, path in required_files.items():
        if label == "custom parser library":
            continue
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ModelPackageError(
                f"{label.capitalize()} must remain inside model package {root}: {path}"
            ) from exc

    if properties.get("parse-bbox-func-name") != "NvDsInferParseYolo26Pose":
        raise ModelPackageError("DeepStream parser must be NvDsInferParseYolo26Pose")
    for key, expected in {
        "output-tensor-meta": "1", "cluster-mode": "4",
        "maintain-aspect-ratio": "1", "symmetric-padding": "1",
    }.items():
        if properties.get(key, "").strip() != expected:
            raise ModelPackageError(f"DeepStream property {key!r} must be {expected!r}")

    class_names = _read_labels(classes, "class labels")
    keypoint_names = _read_labels(keypoint_labels, "keypoint labels")
    width, height, keypoint_count = _validate_pose_schema(
        sidecar_data,
        class_names=class_names,
        keypoint_names=keypoint_names,
        properties=properties,
    )

    class _UniqueKeyLoader(yaml.SafeLoader):
        pass

    def _construct_unique_mapping(loader, node, deep=False):
        mapping = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in mapping:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping

    _UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
    )
    try:
        manifest_text = read_stable_regular_file(
            manifest, max_bytes=_MAX_METADATA_BYTES, label="model manifest"
        ).decode("utf-8")
        manifest_data = yaml.load(manifest_text, Loader=_UniqueKeyLoader)
    except (ValueError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ModelPackageError(f"Could not read model manifest {manifest}: {exc}") from exc
    if not isinstance(manifest_data, dict):
        raise ModelPackageError("Model manifest must contain a YAML mapping")
    manifest_schema = _integer(manifest_data.get("schema_version"), "Model manifest schema_version")
    if manifest_schema not in {2, 3}:
        raise ModelPackageError(
            f"Model manifest schema_version must be 2 or 3 (found {manifest_schema})"
        )
    expected_manifest = {
        "name": root.name, "framework": "yolo26", "task": "pose",
        "classes": class_names, "keypoints": keypoint_names,
    }
    for key, expected in expected_manifest.items():
        if manifest_data.get(key) != expected:
            raise ModelPackageError(f"Model manifest field {key!r} must be {expected!r}")
    batch_size = properties.getint("batch-size", fallback=-1)
    if batch_size <= 0 or manifest_data.get("batch_size") != batch_size:
        raise ModelPackageError("Model manifest batch_size must match the DeepStream config")
    export_data = manifest_data.get("export")
    if not isinstance(export_data, dict) or export_data.get("builder") != "ultralytics" or export_data.get("end2end") is not True:
        raise ModelPackageError("Model manifest must identify an end-to-end Ultralytics export")

    report = _load_json_object(import_report, "import report")
    checks = report.get("checks")
    if not isinstance(checks, dict) or any(checks.get(key) is not True for key in ("onnx", "raw_engine", "yaml_labels")):
        raise ModelPackageError("Import report must pass ONNX, engine, and YAML label checks")
    expected_input = [batch_size, 3, height, width]
    expected_output = [[batch_size, 300, 6 + 3 * keypoint_count]]
    if report.get("onnx_input_shape") != expected_input or report.get("onnx_output_shapes") != expected_output:
        raise ModelPackageError("Import report tensor shapes do not match the schema-v2 contract")
    engine_build_identity = None
    if manifest_schema >= 3:
        if checks.get("engine_execution") is not True:
            raise ModelPackageError(
                "Schema-3 import report must pass bounded TensorRT engine execution"
            )
        execution = report.get("engine_execution")
        if not isinstance(execution, dict):
            raise ModelPackageError("Schema-3 import report is missing engine execution evidence")
        command = execution.get("command")
        max_output_bytes = execution.get("max_output_bytes")
        output = execution.get("output")
        if (
            execution.get("passed") is not True
            or execution.get("timed_out") is not False
            or isinstance(execution.get("returncode"), bool)
            or execution.get("returncode") != 0
            or not isinstance(command, list)
            or not 1 <= len(command) <= 16
            or any(not isinstance(part, str) or len(part) > 4096 for part in command)
            or isinstance(max_output_bytes, bool)
            or not isinstance(max_output_bytes, int)
            or not 1 <= max_output_bytes <= 64 * 1024
            or not isinstance(output, str)
            or len(output.encode("utf-8")) > max_output_bytes
            or not isinstance(execution.get("output_truncated"), bool)
        ):
            raise ModelPackageError(
                "Schema-3 TensorRT engine execution evidence is invalid or unbounded"
            )
        _validate_schema_three_artifacts(
            manifest_data.get("artifacts"),
            root,
            {
                "config": config,
                "pose_schema": pose_sidecar,
                "onnx": onnx,
                "engine": engine,
                "class_labels": classes,
                "keypoint_labels": keypoint_labels,
                "custom_parser": parser_library,
                "import_report": import_report,
            },
        )
        engine_build_identity = _validate_engine_build_identity(
            report.get("build_environment"), _runtime_engine_identity()
        )

    return ModelPackageInfo(
        name=root.name,
        root=root,
        config=config,
        manifest=manifest,
        pose_sidecar=pose_sidecar,
        onnx=onnx,
        engine=engine,
        classes=classes,
        keypoint_labels=keypoint_labels,
        parser_library=parser_library,
        import_report=import_report,
        model_manifest_sha256=_sha256(
            manifest, max_bytes=_MAX_METADATA_BYTES, label="model manifest"
        ),
        pose_sidecar_sha256=_sha256(
            pose_sidecar, max_bytes=_MAX_METADATA_BYTES, label="pose sidecar"
        ),
        onnx_sha256=_sha256(
            onnx, max_bytes=_MAX_MODEL_ARTIFACT_BYTES, label="ONNX model"
        ),
        config_sha256=_sha256(
            config, max_bytes=_MAX_CONFIG_BYTES, label="model config"
        ),
        engine_sha256=_sha256(
            engine, max_bytes=_MAX_MODEL_ARTIFACT_BYTES, label="TensorRT engine"
        ),
        model_manifest_schema=manifest_schema,
        engine_build_identity=engine_build_identity,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a SqueakView model package")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--require-engine-identity",
        action="store_true",
        help="reject migration-only schema-2 packages without runtime engine identity",
    )
    args = parser.parse_args()
    try:
        info = validate_model_package(args.config)
        if args.require_engine_identity and info.engine_build_identity is None:
            raise ModelPackageError(
                "Model package uses migration-only schema 2; rebuild it from "
                "Project Setup before scientific acquisition"
            )
    except ModelPackageError as exc:
        print(f"[FAIL] {exc}")
        return 2
    print(f"[PASS] Model package '{info.name}' is complete")
    print(json.dumps(info.manifest_snapshot(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
