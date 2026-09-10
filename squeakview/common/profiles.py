from __future__ import annotations

"""Profile storage for experiments and subjects."""

import json
import re
from itertools import islice
from dataclasses import asdict, dataclass, field
from pathlib import Path

from squeakview import config as squeakview_config
from squeakview.common.bounded_input import read_json_object
from squeakview.common.run_context import atomic_write_json


MAX_PROFILE_BYTES = 1024 * 1024
MAX_PROFILE_FILES = 1024


def slugify(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    text = re.sub(r"_+", "_", text).strip("._-")
    return text or "profile"


@dataclass(slots=True)
class ExperimentProfile:
    name: str
    slug: str
    config: dict[str, object] = field(default_factory=dict)
    subject_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SubjectProfile:
    name: str
    subject_id: str
    default_experiment: str | None = None


class ProfileStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else squeakview_config.ensure_profiles_dir()
        self.experiments_dir = self.root / "experiments"
        self.subjects_dir = self.root / "subjects"
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.subjects_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _assert_save_capacity(directory: Path, target: Path) -> None:
        if target.exists():
            return
        count = sum(1 for _path in islice(directory.glob("*.json"), MAX_PROFILE_FILES))
        if count >= MAX_PROFILE_FILES:
            raise ValueError(
                f"profile store already contains the {MAX_PROFILE_FILES} file limit"
            )

    def list_experiments(self) -> list[ExperimentProfile]:
        profiles: list[ExperimentProfile] = []
        paths = sorted(islice(self.experiments_dir.glob("*.json"), MAX_PROFILE_FILES))
        for path in paths:
            try:
                data = read_json_object(
                    path, max_bytes=MAX_PROFILE_BYTES, label="experiment profile"
                )
                profiles.append(
                    ExperimentProfile(
                        name=str(data.get("name") or path.stem),
                        slug=str(data.get("slug") or path.stem),
                        config=(dict(data.get("config") or {})),
                        subject_ids=[str(item) for item in (data.get("subject_ids") or []) if str(item).strip()],
                    )
                )
            except Exception:
                continue
        return profiles

    def list_subjects(self) -> list[SubjectProfile]:
        profiles: list[SubjectProfile] = []
        paths = sorted(islice(self.subjects_dir.glob("*.json"), MAX_PROFILE_FILES))
        for path in paths:
            try:
                data = read_json_object(
                    path, max_bytes=MAX_PROFILE_BYTES, label="subject profile"
                )
                profiles.append(
                    SubjectProfile(
                        name=str(data.get("name") or path.stem),
                        subject_id=str(data.get("subject_id") or data.get("name") or path.stem),
                        default_experiment=(str(data["default_experiment"]) if data.get("default_experiment") else None),
                    )
                )
            except Exception:
                continue
        return profiles

    def save_experiment(self, profile: ExperimentProfile) -> Path:
        slug = slugify(profile.slug or profile.name)
        payload = asdict(profile)
        payload["slug"] = slug
        payload["config"] = self._normalize_config(payload.get("config") or {})
        path = self.experiments_dir / f"{slug}.json"
        self._assert_save_capacity(self.experiments_dir, path)
        encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        if len(encoded) > MAX_PROFILE_BYTES:
            raise ValueError(f"experiment profile exceeds {MAX_PROFILE_BYTES} byte limit")
        return atomic_write_json(path, payload)

    def save_subject(self, profile: SubjectProfile) -> Path:
        slug = slugify(profile.subject_id or profile.name)
        payload = asdict(profile)
        path = self.subjects_dir / f"{slug}.json"
        self._assert_save_capacity(self.subjects_dir, path)
        encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        if len(encoded) > MAX_PROFILE_BYTES:
            raise ValueError(f"subject profile exceeds {MAX_PROFILE_BYTES} byte limit")
        return atomic_write_json(path, payload)

    def delete_experiment(self, slug: str) -> None:
        path = self.experiments_dir / f"{slugify(slug)}.json"
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    def delete_subject(self, subject_id: str) -> None:
        path = self.subjects_dir / f"{slugify(subject_id)}.json"
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    @staticmethod
    def _normalize_config(config: dict[str, object]) -> dict[str, object]:
        normalized: dict[str, object] = {}
        for key, value in config.items():
            if isinstance(value, Path):
                if key in {"ds_cfg", "task_cfg"}:
                    normalized[key] = squeakview_config.portable_workspace_path(value)
                else:
                    normalized[key] = str(value)
            elif key in {"ds_cfg", "task_cfg"} and isinstance(value, str):
                normalized[key] = squeakview_config.portable_workspace_path(value)
            else:
                normalized[key] = value
        return normalized
