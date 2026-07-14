from __future__ import annotations

"""Validation and identity helpers for SqueakView model packages."""

import argparse
import configparser
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


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
    required_keys = ("onnx-file", "model-engine-file", "labelfile-path", "custom-lib-path")
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

    if pose_sidecar.is_file():
        try:
            sidecar_data = json.loads(pose_sidecar.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelPackageError(f"Could not read pose sidecar {pose_sidecar}: {exc}") from exc
        raw_labels = sidecar_data.get("keypoint_labels_path") if isinstance(sidecar_data, dict) else None
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
    }
    missing_files = [f"{label}: {path}" for label, path in required_files.items() if not path.is_file()]
    if missing_files:
        raise ModelPackageError("Model package is incomplete:\n" + "\n".join(missing_files))

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
