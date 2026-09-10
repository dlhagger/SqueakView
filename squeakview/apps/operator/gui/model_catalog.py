"""Qt-free discovery and production eligibility policy for model packages."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from squeakview import model_package


MAX_MODEL_PACKAGES = 256
MAX_CONFIGS_PER_PACKAGE = 32


@dataclass(frozen=True, slots=True)
class ModelChoice:
    name: str
    config: Path
    eligible: bool
    detail: str


def validate_production_model(
    config: Path,
    *,
    validator: Callable[[Path], object] = model_package.validate_model_package,
) -> object:
    """Require the schema-3, runtime-matched package used for acquisition."""

    info = validator(Path(config))
    if getattr(info, "model_manifest_schema", None) != 3:
        raise model_package.ModelPackageError(
            "Model package uses migration-only schema 2; rebuild it on this device "
            "before scientific acquisition"
        )
    if getattr(info, "engine_build_identity", None) is None:
        raise model_package.ModelPackageError(
            "Schema-3 model package has no verified device-local engine identity"
        )
    return info


def enumerate_model_configs(
    models_root: Path,
    *,
    validator: Callable[[Path], object] = model_package.validate_model_package,
) -> tuple[ModelChoice, ...]:
    """Enumerate bounded, direct, non-hidden package/config children."""

    root = Path(models_root)
    try:
        packages = sorted(
            (
                path
                for path in root.iterdir()
                if path.is_dir() and not path.is_symlink() and not path.name.startswith(".")
            ),
            key=lambda path: path.name.casefold(),
        )[:MAX_MODEL_PACKAGES]
    except OSError:
        return ()

    choices: list[ModelChoice] = []
    for package in packages:
        configs_dir = package / "configs"
        try:
            configs = sorted(
                (
                    path
                    for path in configs_dir.iterdir()
                    if path.is_file()
                    and not path.name.startswith(".")
                    and path.suffix.lower() in {".txt", ".cfg"}
                ),
                key=lambda path: path.name.casefold(),
            )[:MAX_CONFIGS_PER_PACKAGE]
        except OSError:
            continue
        for config in configs:
            try:
                info = validate_production_model(config, validator=validator)
                choices.append(
                    ModelChoice(
                        str(getattr(info, "name", package.name)),
                        config.resolve(),
                        True,
                        "Schema 3; engine identity matches this device",
                    )
                )
            except (model_package.ModelPackageError, OSError, ValueError) as exc:
                detail = " ".join(str(exc).split())[:300] or "validation failed"
                choices.append(
                    ModelChoice(package.name, config.resolve(), False, detail)
                )
    return tuple(
        sorted(choices, key=lambda item: (not item.eligible, item.name.casefold(), item.config.name))
    )


__all__ = ["ModelChoice", "enumerate_model_configs", "validate_production_model"]
