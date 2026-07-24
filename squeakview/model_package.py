from __future__ import annotations

"""Validation and identity helpers for SqueakView model packages."""

import argparse
import configparser
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


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
    config_sha256: str
    engine_sha256: str

    def manifest_snapshot(self) -> dict[str, str]:
        payload = asdict(self)
        return {key: str(value) if isinstance(value, Path) else value for key, value in payload.items()}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_config_path(raw: str, config_dir: Path) -> Path:
    path = Path(raw.strip().strip('"')).expanduser()
    return path.resolve() if path.is_absolute() else (config_dir / path).resolve()


def _load_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelPackageError(f"Could not read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ModelPackageError(f"{label.capitalize()} must contain a JSON object: {path}")
    return value


def _read_labels(path: Path, label: str) -> list[str]:
    try:
        labels = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    except OSError as exc:
        raise ModelPackageError(f"Could not read {label} {path}: {exc}") from exc
    if not labels or len(labels) != len(set(labels)):
        raise ModelPackageError(f"{label.capitalize()} must contain unique, non-empty labels: {path}")
    return labels


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ModelPackageError(f"{label} must be an integer")
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
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
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    try:
        with config.open() as handle:
            parser.read_file(handle)
    except (OSError, configparser.Error) as exc:
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

    try:
        manifest_data = yaml.safe_load(manifest.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ModelPackageError(f"Could not read model manifest {manifest}: {exc}") from exc
    expected_manifest = {
        "schema_version": 2, "name": root.name, "framework": "yolo26", "task": "pose",
        "classes": class_names, "keypoints": keypoint_names,
    }
    if not isinstance(manifest_data, dict):
        raise ModelPackageError("Model manifest must contain a YAML mapping")
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
        config_sha256=_sha256(config),
        engine_sha256=_sha256(engine),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a SqueakView model package")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    try:
        info = validate_model_package(args.config)
    except ModelPackageError as exc:
        print(f"[FAIL] {exc}")
        return 2
    print(f"[PASS] Model package '{info.name}' is complete")
    print(json.dumps(info.manifest_snapshot(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
