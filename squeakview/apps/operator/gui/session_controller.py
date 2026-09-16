from __future__ import annotations

"""Session/profile coordination for the operator GUI.

This module owns profile selection and modal-dialog orchestration.  It keeps the
top-level window focused on presenting run state instead of also acting as a
profile repository and configuration mapper.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Type

from PySide6 import QtWidgets

from squeakview.apps.operator.backend import process
from squeakview.apps.operator.gui.config_dialog import ConfigDialog
from squeakview.apps.operator.gui.session_dialog import (
    CreateExperimentDialog,
    CreateSubjectDialog,
    SessionLauncherDialog,
)
from squeakview.common.profiles import (
    ExperimentProfile,
    ProfileStore,
    SubjectProfile,
    slugify,
)
from squeakview.project import Project


ConfigCommit = Callable[[dict], None]
LogEmitter = Callable[[str], None]


def default_config_data(project: Project) -> dict[str, object]:
    """Return the initial GUI configuration without requiring a widget."""

    defaults = process.LaunchConfig()
    default_task_name = project.metadata.default_task
    task_cfg = (
        project.paths.resolve_path(
            Path("tasks") / default_task_name,
            within=project.paths.tasks,
        )
        if default_task_name
        else None
    )
    default_model_name = project.metadata.default_model
    default_model_cfg = (
        project.paths.models
        / default_model_name
        / "configs"
        / f"{default_model_name}.txt"
        if default_model_name
        else None
    )
    return {
        "width": int(defaults.width or 1440),
        "height": int(defaults.height or 1080),
        "fps": int(defaults.fps or 30),
        "pixel_format": defaults.pixel_format or "Mono8",
        "capture_backend": defaults.capture_backend,
        "trigger_on": defaults.trigger_on,
        "exposure_us": int(defaults.exposure_us or 10000),
        "arduino_fps": defaults.arduino_fps,
        "serial_enabled": defaults.serial_enabled,
        "serial_port": defaults.serial_port,
        "serial_baud": defaults.serial_baud,
        "ds_cfg": str(default_model_cfg) if default_model_cfg else "",
        # A fresh project deliberately starts without a deployable TensorRT
        # package. Project Setup enables the normal inference default only
        # after a validated package has been built and selected.
        "inference_enabled": bool(default_model_cfg) and defaults.inference_enabled,
        "task_cfg": str(task_cfg) if task_cfg else "",
        "num_cameras": max(1, defaults.num_cameras),
        "bitrate": defaults.bitrate,
        "mouse_id": "",
        "experiment_name": "",
        "experiment_mode": "sandbox",
    }


def merge_profile_selection(
    config: dict | None,
    experiment: ExperimentProfile | None,
    subject: SubjectProfile | None,
    *,
    project: Project,
) -> dict:
    """Overlay a selected experiment and subject on a copied config."""

    data = dict(config or default_config_data(project))
    if experiment is not None:
        data.update(dict(experiment.config or {}))
        # Identity comes from the selected profile, never from a stale value
        # embedded in its reusable capture defaults.
        data["experiment_name"] = experiment.slug
    else:
        data["experiment_name"] = ""
    data["mouse_id"] = subject.subject_id if subject is not None else ""
    return data


@dataclass(frozen=True, slots=True)
class ResolvedConfig:
    data: dict
    ds_cfg: Path | None
    task_cfg: Path | None


def resolve_config_paths(config: dict, *, project: Project) -> ResolvedConfig:
    """Resolve and category-bound project-relative paths on a copied config."""

    data = dict(config)
    ds_cfg = (
        project.paths.resolve_path(
            data["ds_cfg"],
            within=project.paths.models,
        )
        if data.get("ds_cfg")
        else None
    )
    task_cfg = (
        project.paths.resolve_path(
            data["task_cfg"],
            within=project.paths.tasks,
        )
        if data.get("task_cfg")
        else None
    )
    if ds_cfg is not None:
        data["ds_cfg"] = str(ds_cfg)
    if task_cfg is not None:
        data["task_cfg"] = str(task_cfg)
    return ResolvedConfig(data=data, ds_cfg=ds_cfg, task_cfg=task_cfg)


def build_launch_config(
    config: dict | None,
    *,
    bottles: dict[str, object],
    preview_window_id: int | None,
    project: Project,
    environment: dict[str, str] | None = None,
) -> process.LaunchConfig:
    """Translate GUI configuration into an immutable backend request."""

    if not config:
        raise RuntimeError("Configuration not set")
    resolved = resolve_config_paths(config, project=project)
    data = resolved.data
    env = os.environ if environment is None else environment
    failure_plan_value = env.get("SQUEAKVIEW_FAILURE_PLAN")
    failure_plan = (
        project.paths.resolve_path(
            failure_plan_value,
            within=project.paths.qualification,
        )
        if failure_plan_value
        else None
    )
    controller_protocol = env.get("SQUEAKVIEW_CONTROLLER_PROTOCOL", "legacy").strip()
    try:
        watchdog_lease_ms = int(
            env.get("SQUEAKVIEW_CONTROLLER_WATCHDOG_LEASE_MS", "1500")
        )
    except ValueError as exc:
        raise RuntimeError(
            "SQUEAKVIEW_CONTROLLER_WATCHDOG_LEASE_MS must be an integer"
        ) from exc
    preview_disabled = env.get("SQUEAKVIEW_DISABLE_PREVIEW", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    return process.LaunchConfig(
        capture_backend=str(data.get("capture_backend", "flir_direct")),
        width=data["width"],
        height=data["height"],
        fps=data["fps"],
        pixel_format=data["pixel_format"],
        trigger_on=data["trigger_on"],
        exposure_us=data.get("exposure_us", 10000),
        ds_cfg=resolved.ds_cfg,
        inference_enabled=data.get("inference_enabled", True),
        num_cameras=int(data.get("num_cameras", 1)),
        bitrate=data["bitrate"],
        serial_enabled=data["serial_enabled"],
        serial_port=data["serial_port"],
        serial_baud=data["serial_baud"],
        controller_protocol=controller_protocol,
        controller_watchdog_lease_ms=watchdog_lease_ms,
        arduino_fps=data["arduino_fps"],
        mouse_id=data.get("mouse_id", ""),
        experiment_name=data.get("experiment_name", ""),
        task_cfg=resolved.task_cfg,
        failure_plan=failure_plan,
        bottles=bottles,
        preview_window_id=preview_window_id,
        preview_enabled=not preview_disabled,
    )


class SessionConfigController:
    """Coordinate profile selectors and session/configuration dialogs."""

    def __init__(
        self,
        parent: QtWidgets.QWidget,
        experiment_combo: QtWidgets.QComboBox,
        subject_combo: QtWidgets.QComboBox,
        *,
        project: Project,
        store: ProfileStore,
        commit: ConfigCommit,
        emit: LogEmitter,
        launcher_dialog: Type[SessionLauncherDialog] = SessionLauncherDialog,
        config_dialog: Type[ConfigDialog] = ConfigDialog,
        experiment_dialog: Type[CreateExperimentDialog] = CreateExperimentDialog,
        subject_dialog: Type[CreateSubjectDialog] = CreateSubjectDialog,
    ) -> None:
        self.parent = parent
        self.project = project
        self.experiment_combo = experiment_combo
        self.subject_combo = subject_combo
        self.store = store
        self.commit = commit
        self.emit = emit
        self.launcher_dialog = launcher_dialog
        self.config_dialog = config_dialog
        self.experiment_dialog = experiment_dialog
        self.subject_dialog = subject_dialog
        self.experiments: list[ExperimentProfile] = []
        self.subjects: list[SubjectProfile] = []
        self.selection_updating = False

    def current_experiment_slug(self) -> str:
        return str(self.experiment_combo.currentData() or "")

    def current_subject_id(self) -> str:
        return str(self.subject_combo.currentData() or "")

    def find_experiment(self, slug: str) -> ExperimentProfile | None:
        return next((item for item in self.experiments if item.slug == slug), None)

    def find_subject(self, subject_id: str) -> SubjectProfile | None:
        return next(
            (item for item in self.subjects if item.subject_id == subject_id),
            None,
        )

    def reload(self) -> None:
        self.experiments = self.store.list_experiments()
        self.subjects = self.store.list_subjects()
        self.refresh_selectors()

    def refresh_selectors(self) -> None:
        selected_exp = self.current_experiment_slug()
        selected_subject = self.current_subject_id()
        self.selection_updating = True
        try:
            self.experiment_combo.clear()
            self.experiment_combo.addItem("No experiment", "")
            for profile in self.experiments:
                self.experiment_combo.addItem(profile.name, profile.slug)
            self.experiment_combo.setCurrentIndex(
                max(0, self.experiment_combo.findData(selected_exp))
            )

            self.subject_combo.clear()
            self.subject_combo.addItem("No subject", "")
            for profile in self.subjects:
                self.subject_combo.addItem(profile.name, profile.subject_id)
            self.subject_combo.setCurrentIndex(
                max(0, self.subject_combo.findData(selected_subject))
            )
        finally:
            self.selection_updating = False

    def apply_defaults(self, config: dict | None) -> dict:
        if self.subjects and not self.current_subject_id():
            index = self.subject_combo.findData(self.subjects[0].subject_id)
            if index >= 0:
                self.subject_combo.setCurrentIndex(index)
        if self.experiments and not self.current_experiment_slug():
            index = self.experiment_combo.findData(self.experiments[0].slug)
            if index >= 0:
                self.experiment_combo.setCurrentIndex(index)
        return self.apply_selection(config)

    def apply_selection(self, config: dict | None) -> dict:
        result = merge_profile_selection(
            config,
            self.find_experiment(self.current_experiment_slug()),
            self.find_subject(self.current_subject_id()),
            project=self.project,
        )
        self.commit(result)
        return result

    def experiment_selected(self, config: dict | None) -> dict | None:
        if self.selection_updating:
            return None
        return self.apply_selection(config)

    def subject_selected(self, config: dict | None) -> dict | None:
        if self.selection_updating:
            return None
        subject = self.find_subject(self.current_subject_id())
        if subject and subject.default_experiment:
            index = self.experiment_combo.findData(subject.default_experiment)
            if index >= 0 and index != self.experiment_combo.currentIndex():
                self.selection_updating = True
                try:
                    self.experiment_combo.setCurrentIndex(index)
                finally:
                    self.selection_updating = False
        return self.apply_selection(config)

    def create_experiment(self, config: dict | None) -> None:
        dialog = self.experiment_dialog(self.parent)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        data = config or default_config_data(self.project)
        experiment_slug = slugify(dialog.experiment_name)
        profile_config = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in data.items()
        }
        profile_config["experiment_name"] = experiment_slug
        profile_config["mouse_id"] = ""
        profile = ExperimentProfile(
            name=dialog.experiment_name,
            slug=experiment_slug,
            config=profile_config,
        )
        path = self.store.save_experiment(profile)
        self.emit(f"[GUI] experiment profile saved → {path}")
        self.reload()
        index = self.experiment_combo.findData(path.stem)
        if index >= 0:
            self.experiment_combo.setCurrentIndex(index)

    def create_subject(self) -> None:
        dialog = self.subject_dialog(self.parent)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        profile = SubjectProfile(
            name=dialog.subject_name,
            subject_id=dialog.subject_id,
            default_experiment=self.current_experiment_slug() or None,
        )
        path = self.store.save_subject(profile)
        self.emit(f"[GUI] subject profile saved → {path}")
        self.reload()
        index = self.subject_combo.findData(profile.subject_id)
        if index >= 0:
            self.subject_combo.setCurrentIndex(index)

    def show_launcher(self, config: dict | None) -> dict | None:
        dialog = self.launcher_dialog(
            self.parent,
            project=self.project,
            base_config=config,
            profile_store=self.store,
        )
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        return dict(dialog.result_config) if dialog.result_config else None

    def show_config(self, config: dict | None) -> dict | None:
        dialog = self.config_dialog(
            self.parent,
            config=config,
            profile_store=self.store,
            project=self.project,
            show_session_setup=False,
        )
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        if not dialog.result_config:
            return None
        result = dict(dialog.result_config)
        if config and "experiment_name" not in result:
            result["experiment_name"] = config.get("experiment_name", "")
        return result


__all__ = [
    "ResolvedConfig",
    "SessionConfigController",
    "build_launch_config",
    "default_config_data",
    "merge_profile_selection",
    "resolve_config_paths",
]
