from __future__ import annotations

"""Choose or create the explicit project owned by a SqueakView session."""

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Sequence

from PySide6 import QtCore, QtGui, QtWidgets

from squeakview.apps.model_builder import EVENT_PREFIX
from squeakview.apps.operator.gui.dialog_style import (
    DARK_DIALOG_STYLE,
    apply_dark_combo_popups,
)
from squeakview.model_builder import discover_model_sources, import_model_source
from squeakview.project import (
    AppPaths,
    Project,
    ProjectCatalog,
    ProjectSession,
    UserPaths,
    set_default_model,
)


class ProjectChooser(QtWidgets.QDialog):
    def __init__(self, catalog: ProjectCatalog) -> None:
        super().__init__()
        self.catalog = catalog
        self.selected_project: Project | None = None
        self.setWindowTitle("Open SqueakView Project")
        self.setModal(True)
        self.setMinimumSize(620, 400)
        self.setStyleSheet(DARK_DIALOG_STYLE)

        layout = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel("Choose a scientific project")
        title.setStyleSheet("font-size: 22px; font-weight: 700;")
        layout.addWidget(title)
        description = QtWidgets.QLabel(
            "Runs, models, tasks, profiles, and qualification records stay in "
            "the selected project and are not changed when SqueakView is updated."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        self.projects = QtWidgets.QListWidget(self)
        for project in catalog.recent():
            item = QtWidgets.QListWidgetItem(
                f"{project.metadata.name}\n{project.paths.root}"
            )
            item.setData(QtCore.Qt.ItemDataRole.UserRole, str(project.paths.root))
            self.projects.addItem(item)
        if self.projects.count():
            # The catalog is ordered most-recent-first.  Always present the
            # chooser, but make the common reopen path a single confirmation.
            self.projects.setCurrentRow(0)
        self.projects.itemDoubleClicked.connect(lambda _item: self._open_recent())
        layout.addWidget(self.projects, 1)

        buttons = QtWidgets.QHBoxLayout()
        create_button = QtWidgets.QPushButton("Create Project…", self)
        browse_button = QtWidgets.QPushButton("Open Other…", self)
        self.open_button = QtWidgets.QPushButton("Open Project", self)
        self.open_button.setDefault(True)
        self.open_button.setEnabled(self.projects.currentItem() is not None)
        cancel_button = QtWidgets.QPushButton("Cancel", self)
        buttons.addWidget(create_button)
        buttons.addWidget(browse_button)
        buttons.addStretch(1)
        buttons.addWidget(cancel_button)
        buttons.addWidget(self.open_button)
        layout.addLayout(buttons)

        self.projects.currentItemChanged.connect(
            lambda current, _previous: self.open_button.setEnabled(current is not None)
        )
        create_button.clicked.connect(self._create)
        browse_button.clicked.connect(self._browse)
        self.open_button.clicked.connect(self._open_recent)
        cancel_button.clicked.connect(self.reject)
        apply_dark_combo_popups(self)

    def _accept_project(self, root: Path) -> None:
        try:
            self.selected_project = self.catalog.open(root)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Project Could Not Open", str(exc))
            return
        self.accept()

    def _open_recent(self) -> None:
        item = self.projects.currentItem()
        if item is not None:
            self._accept_project(Path(str(item.data(QtCore.Qt.ItemDataRole.UserRole))))

    def _browse(self) -> None:
        root = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Open SqueakView Project",
            str(self.catalog.user.projects_parent),
        )
        if root:
            self._accept_project(Path(root))

    def _create(self) -> None:
        name, accepted = QtWidgets.QInputDialog.getText(
            self,
            "Create SqueakView Project",
            "Project name:",
        )
        if not accepted or not name.strip():
            return
        parent = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Choose Parent Folder",
            str(self.catalog.user.projects_parent),
        )
        if not parent:
            return
        try:
            project = self.catalog.create(name=name, parent=Path(parent))
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Project Could Not Be Created", str(exc))
            return
        self.selected_project = project
        self.accept()


class ProjectSetupDialog(QtWidgets.QDialog):
    """Prepare/select a model before the acquisition supervisor is launched."""

    def __init__(self, catalog: ProjectCatalog, project: Project) -> None:
        super().__init__()
        self.catalog = catalog
        self.project = project
        self._process: QtCore.QProcess | None = None
        self._stdout_buffer = ""
        self._build_succeeded = False
        self._build_stage = ""
        self._suggested_package_name = ""
        self._build_log_file = None
        self._build_log_path: Path | None = None
        self._build_log_bytes = 0
        self.setWindowTitle("SqueakView Project Setup")
        self.setModal(True)
        self.setMinimumSize(760, 620)
        self.resize(900, 720)
        self.setStyleSheet(
            DARK_DIALOG_STYLE
            + """
            QProgressBar {
                background-color: #12172a;
                border: 1px solid #333a55;
                border-radius: 5px;
                min-height: 10px;
                max-height: 10px;
            }
            QProgressBar::chunk {
                background-color: #5c6df5;
                border-radius: 4px;
            }
            """
        )

        layout = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel("Prepare project models", self)
        title.setStyleSheet("font-size: 22px; font-weight: 700;")
        layout.addWidget(title)
        project_label = QtWidgets.QLabel(
            f"<b>{project.metadata.name}</b><br>{project.paths.root}", self
        )
        project_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(project_label)

        explanation = QtWidgets.QLabel(
            "Build the device-local TensorRT package before defining experiments "
            "and subjects. The build runs in an isolated process; recording does "
            "not start until that process has exited and released its GPU memory.",
            self,
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        form = QtWidgets.QFormLayout()
        self.source_combo = QtWidgets.QComboBox(self)
        source_row = QtWidgets.QWidget(self)
        source_layout = QtWidgets.QHBoxLayout(source_row)
        source_layout.setContentsMargins(0, 0, 0, 0)
        self.import_source_button = QtWidgets.QPushButton("Import Source…", source_row)
        source_layout.addWidget(self.source_combo, 1)
        source_layout.addWidget(self.import_source_button)
        self.package_name = QtWidgets.QLineEdit(self)
        self.package_name.setPlaceholderText("Project-local model package name")
        self.overwrite_check = QtWidgets.QCheckBox(
            "Replace an existing package with this name after validation", self
        )
        form.addRow("Model source:", source_row)
        form.addRow("Package name:", self.package_name)
        form.addRow("", self.overwrite_check)
        layout.addLayout(form)

        build_actions = QtWidgets.QHBoxLayout()
        self.build_button = QtWidgets.QPushButton("Build Model", self)
        self.cancel_build_button = QtWidgets.QPushButton("Cancel Build", self)
        self.cancel_build_button.setEnabled(False)
        self.copy_log_button = QtWidgets.QPushButton("Copy Log", self)
        build_actions.addWidget(self.build_button)
        build_actions.addWidget(self.cancel_build_button)
        build_actions.addWidget(self.copy_log_button)
        build_actions.addStretch(1)
        layout.addLayout(build_actions)

        self.progress = QtWidgets.QProgressBar(self)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        layout.addWidget(self.progress)
        self.status_label = QtWidgets.QLabel("", self)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        package_group = QtWidgets.QGroupBox("Published project models", self)
        package_layout = QtWidgets.QHBoxLayout(package_group)
        self.models_combo = QtWidgets.QComboBox(package_group)
        self.default_button = QtWidgets.QPushButton("Use as Project Default", package_group)
        package_layout.addWidget(self.models_combo, 1)
        package_layout.addWidget(self.default_button)
        layout.addWidget(package_group)

        self.log = QtWidgets.QPlainTextEdit(self)
        self.log.setReadOnly(True)
        self.log.document().setMaximumBlockCount(2500)
        self.log.setPlaceholderText("Model build output will appear here.")
        layout.addWidget(self.log, 1)

        footer = QtWidgets.QHBoxLayout()
        self.back_button = QtWidgets.QPushButton("Back", self)
        self.continue_button = QtWidgets.QPushButton(self)
        self.continue_button.setDefault(True)
        footer.addWidget(self.back_button)
        footer.addStretch(1)
        footer.addWidget(self.continue_button)
        layout.addLayout(footer)

        self.source_combo.currentIndexChanged.connect(self._source_changed)
        self.import_source_button.clicked.connect(self._import_source)
        self.package_name.textChanged.connect(self._refresh_build_button)
        self.build_button.clicked.connect(self._start_build)
        self.cancel_build_button.clicked.connect(self._cancel_build)
        self.copy_log_button.clicked.connect(self._copy_log)
        self.default_button.clicked.connect(self._set_selected_default)
        self.models_combo.currentIndexChanged.connect(self._refresh_default_button)
        self.back_button.clicked.connect(self.reject)
        self.continue_button.clicked.connect(self.accept)

        self._refresh_sources()
        self._refresh_project()
        apply_dark_combo_popups(self)

    def _refresh_sources(self) -> None:
        current = str(self.source_combo.currentData() or "")
        self.source_combo.clear()
        for source in discover_model_sources(self.project):
            self.source_combo.addItem(source.name, source.name)
        preferred = self.source_combo.findData(current or "mousehouse_v2")
        if preferred >= 0:
            self.source_combo.setCurrentIndex(preferred)
        self._source_changed(self.source_combo.currentIndex())

    def _source_changed(self, _index: int) -> None:
        source_name = str(self.source_combo.currentData() or "")
        current_name = self.package_name.text().strip()
        if source_name and (
            not current_name or current_name == self._suggested_package_name
        ):
            self.package_name.setText(source_name)
        self._suggested_package_name = source_name
        self._refresh_build_button()

    def _import_source(self) -> None:
        if self._process is not None:
            return
        checkpoint, _selected_filter = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Choose YOLO Pose Checkpoint",
            str(Path.home()),
            "PyTorch checkpoint (*.pt)",
        )
        if not checkpoint:
            return
        data_yaml, _selected_filter = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Choose Dataset YAML",
            str(Path(checkpoint).parent),
            "YAML files (*.yaml *.yml)",
        )
        if not data_yaml:
            return
        name, accepted = QtWidgets.QInputDialog.getText(
            self,
            "Import Model Source",
            "Project source name:",
            text=Path(checkpoint).stem,
        )
        if not accepted or not name.strip():
            return
        try:
            with ProjectSession.open(self.project.paths.root) as session:
                imported = import_model_source(
                    session.project,
                    name=name.strip(),
                    checkpoint=Path(checkpoint),
                    data_yaml=Path(data_yaml),
                )
            self._refresh_sources()
            index = self.source_combo.findData(imported.name)
            if index >= 0:
                self.source_combo.setCurrentIndex(index)
            self.status_label.setText(f"Imported project model source: {imported.name}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Model Source Could Not Be Imported",
                str(exc),
            )

    def _published_models(self) -> tuple[str, ...]:
        try:
            return tuple(
                path.name
                for path in sorted(
                    self.project.paths.models.iterdir(), key=lambda item: item.name.casefold()
                )
                if path.is_dir()
                and not path.is_symlink()
                and not path.name.startswith(".")
                and (path / "configs" / f"{path.name}.txt").is_file()
                and (path / "model.yaml").is_file()
            )
        except OSError:
            return ()

    def _refresh_project(self) -> None:
        self.project = self.catalog.open(self.project.paths.root)
        selected = str(self.models_combo.currentData() or "")
        self.models_combo.blockSignals(True)
        try:
            self.models_combo.clear()
            for name in self._published_models():
                label = name
                if name == self.project.metadata.default_model:
                    label += " — project default"
                self.models_combo.addItem(label, name)
            index = self.models_combo.findData(
                selected or self.project.metadata.default_model or ""
            )
            if index >= 0:
                self.models_combo.setCurrentIndex(index)
        finally:
            self.models_combo.blockSignals(False)
        default = self.project.metadata.default_model
        self.continue_button.setText(
            "Continue to SqueakView" if default else "Continue Without Inference"
        )
        self.continue_button.setToolTip(
            f"Experiments will initially use the project default model '{default}'."
            if default
            else "No model is selected; experiments will initially have inference disabled."
        )
        self._refresh_default_button()
        self._refresh_build_button()

    def _refresh_build_button(self) -> None:
        idle = self._process is None
        name = self.package_name.text().strip()
        valid_name = bool(name) and Path(name).name == name and name not in {".", ".."}
        self.build_button.setEnabled(
            idle and valid_name and self.source_combo.currentData() is not None
        )

    def _refresh_default_button(self, _index: int = -1) -> None:
        selected = str(self.models_combo.currentData() or "")
        self.default_button.setEnabled(
            self._process is None
            and bool(selected)
            and selected != self.project.metadata.default_model
        )

    def _set_selected_default(self) -> None:
        selected = str(self.models_combo.currentData() or "")
        if not selected or self._process is not None:
            return
        try:
            with ProjectSession.open(self.project.paths.root) as session:
                set_default_model(session.project, selected)
            self._refresh_project()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Default Model Could Not Be Changed",
                str(exc),
            )

    def _set_controls_building(self, building: bool) -> None:
        self.source_combo.setEnabled(not building)
        self.import_source_button.setEnabled(not building)
        self.package_name.setEnabled(not building)
        self.overwrite_check.setEnabled(not building)
        self.models_combo.setEnabled(not building)
        self.default_button.setEnabled(False)
        self.cancel_build_button.setEnabled(building)
        self.continue_button.setEnabled(not building)
        self.back_button.setEnabled(not building)
        if building:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 1)
            self.progress.setValue(1 if self._build_succeeded else 0)
        self._refresh_build_button()
        if not building:
            self._refresh_default_button()

    def _copy_log(self) -> None:
        QtWidgets.QApplication.clipboard().setText(self.log.toPlainText())

    def _open_build_log(self) -> None:
        self.catalog.user.ensure()
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        suffix = os.urandom(3).hex()
        path = self.catalog.user.launch_logs / f"model_build_{timestamp}_{suffix}.log"
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self._build_log_file = os.fdopen(descriptor, "wb")
        self._build_log_path = path
        self._build_log_bytes = 0

    def _append_build_log(self, text: str) -> None:
        if not text:
            return
        self.log.appendPlainText(text)
        handle = self._build_log_file
        if handle is None:
            return
        encoded = (text + "\n").encode("utf-8", errors="replace")
        remaining = 16 * 1024 * 1024 - self._build_log_bytes
        if remaining <= 0:
            return
        chunk = encoded[:remaining]
        handle.write(chunk)
        handle.flush()
        self._build_log_bytes += len(chunk)

    def _close_build_log(self) -> None:
        handle = self._build_log_file
        self._build_log_file = None
        if handle is None:
            return
        try:
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            handle.close()

    def _start_build(self) -> None:
        if self._process is not None:
            return
        source = str(self.source_combo.currentData() or "")
        model_name = self.package_name.text().strip()
        if not source or not model_name:
            return
        existing = model_name in self._published_models()
        if existing and not self.overwrite_check.isChecked():
            QtWidgets.QMessageBox.warning(
                self,
                "Model Package Exists",
                "That package already exists. Select the replacement option only "
                "when you intentionally want to replace it after validation.",
            )
            return
        self.log.clear()
        self._close_build_log()
        try:
            self._open_build_log()
        except OSError as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Build Log Could Not Be Created",
                str(exc),
            )
            return
        self._stdout_buffer = ""
        self._build_succeeded = False
        self._build_stage = "starting"
        self.status_label.setText("Starting isolated model builder…")
        process = QtCore.QProcess(self)
        process.setProgram(sys.executable)
        arguments = [
            "-m",
            "squeakview.apps.model_builder",
            "--project",
            str(self.project.paths.root),
            "--source",
            source,
            "--model-name",
            model_name,
        ]
        if self.overwrite_check.isChecked():
            arguments.append("--overwrite")
        process.setArguments(arguments)
        process.setWorkingDirectory(str(self.catalog.app.root))
        unix_parameters = QtCore.QProcess.UnixProcessParameters()
        unix_parameters.flags = (
            QtCore.QProcess.UnixProcessFlag.CreateNewSession
            | QtCore.QProcess.UnixProcessFlag.ResetSignalHandlers
        )
        process.setUnixProcessParameters(unix_parameters)
        process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.SeparateChannels)
        process.readyReadStandardOutput.connect(self._read_stdout)
        process.readyReadStandardError.connect(self._read_stderr)
        process.finished.connect(self._build_finished)
        process.errorOccurred.connect(self._build_process_error)
        self._process = process
        self._set_controls_building(True)
        process.start()

    def _consume_stdout_line(self, line: str) -> None:
        if line.startswith(EVENT_PREFIX):
            try:
                payload = json.loads(line[len(EVENT_PREFIX):])
            except (json.JSONDecodeError, TypeError):
                self._append_build_log(line)
                return
            stage = str(payload.get("stage", ""))
            message = str(payload.get("message", ""))
            if stage:
                self._build_stage = stage
            if message:
                self.status_label.setText(message)
                self._append_build_log(f"[{stage.upper() or 'BUILD'}] {message}")
            if stage == "result":
                self._build_succeeded = True
            return
        if line:
            self._append_build_log(line)

    def _read_stdout(self) -> None:
        if self._process is None:
            return
        chunk = bytes(self._process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        self._stdout_buffer += chunk
        while "\n" in self._stdout_buffer:
            line, self._stdout_buffer = self._stdout_buffer.split("\n", 1)
            self._consume_stdout_line(line.rstrip("\r"))

    def _read_stderr(self) -> None:
        if self._process is None:
            return
        chunk = bytes(self._process.readAllStandardError()).decode(
            "utf-8", errors="replace"
        )
        if chunk:
            self._append_build_log(chunk.rstrip())

    def _build_process_error(self, error: QtCore.QProcess.ProcessError) -> None:
        if error == QtCore.QProcess.ProcessError.Crashed:
            return
        self.status_label.setText(f"Model builder could not start: {self._process.errorString() if self._process else error}")
        if (
            self._process is not None
            and self._process.state() == QtCore.QProcess.ProcessState.NotRunning
        ):
            QtCore.QTimer.singleShot(0, lambda: self._finish_release(False))

    def _build_finished(
        self,
        exit_code: int,
        status: QtCore.QProcess.ExitStatus,
    ) -> None:
        self._read_stdout()
        self._read_stderr()
        if self._stdout_buffer:
            self._consume_stdout_line(self._stdout_buffer.rstrip("\r"))
            self._stdout_buffer = ""
        success = exit_code == 0 and self._build_succeeded
        failure = ""
        if not success and status == QtCore.QProcess.ExitStatus.CrashExit:
            if self._build_stage == "exporting":
                failure = (
                    "The operating system terminated TensorRT during engine construction "
                    f"(exit code {exit_code}). On an 8 GiB Jetson this normally means "
                    "memory pressure. The published model was preserved. Close other "
                    "memory-heavy applications or reboot, then retry."
                )
            else:
                failure = (
                    "The model builder was terminated by the operating system "
                    f"(exit code {exit_code}) during {self._build_stage or 'startup'}. "
                    "The published model was preserved."
                )
            self._append_build_log(f"[FAILED] {failure}")
        if self._build_log_path is not None:
            self._append_build_log(f"[LOG] Saved to {self._build_log_path}")
        self._close_build_log()
        if success:
            self.status_label.setText("Build complete. Releasing GPU resources…")
        elif failure:
            self.status_label.setText(failure)
        else:
            self.status_label.setText(
                "Model build did not complete. The previous published model was preserved."
            )
        QtCore.QTimer.singleShot(750, lambda: self._finish_release(success))

    def _finish_release(self, success: bool) -> None:
        self._close_build_log()
        process = self._process
        self._process = None
        if process is not None:
            process.deleteLater()
        self._build_succeeded = success
        self._set_controls_building(False)
        if success:
            try:
                self._refresh_project()
            except Exception as exc:
                self.status_label.setText(f"Model built, but project refresh failed: {exc}")
                return
            self.status_label.setText(
                "Model ready. The TensorRT worker has exited and released its GPU context."
            )
        else:
            self._refresh_default_button()

    def _cancel_build(self) -> None:
        process = self._process
        if process is None:
            return
        self.status_label.setText("Cancelling model build…")
        process_id = int(process.processId())
        if os.name == "posix" and process_id > 0:
            try:
                os.killpg(process_id, signal.SIGTERM)
            except ProcessLookupError:
                process.terminate()
            except PermissionError:
                process.terminate()
        else:
            process.terminate()
        QtCore.QTimer.singleShot(5000, self._force_kill_build)

    def _force_kill_build(self) -> None:
        process = self._process
        if process is None or process.state() == QtCore.QProcess.ProcessState.NotRunning:
            return
        process_id = int(process.processId())
        if os.name == "posix" and process_id > 0:
            try:
                os.killpg(process_id, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                process.kill()
        else:
            process.kill()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        if self._process is not None:
            QtWidgets.QMessageBox.information(
                self,
                "Model Build In Progress",
                "Cancel the model build and wait for the worker to exit before closing project setup.",
            )
            event.ignore()
            return
        self._close_build_log()
        super().closeEvent(event)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project",
        type=Path,
        help="validate and select this project without showing the chooser",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        user = UserPaths.discover()
        catalog = ProjectCatalog(app=AppPaths.discover(), user=user)
        catalog.ensure_projects_parent()
    except Exception as exc:
        print(f"SqueakView path configuration is invalid: {exc}", file=sys.stderr)
        return 1
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setApplicationName("SqueakView")
    explicit_project: Project | None = None
    if args.project is not None:
        try:
            explicit_project = catalog.open(args.project)
        except Exception as exc:
            print(f"SqueakView project could not be opened: {exc}", file=sys.stderr)
            return 1

    while True:
        if explicit_project is None:
            chooser = ProjectChooser(catalog)
            if chooser.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                return 2
            if chooser.selected_project is None:
                return 1
            project = chooser.selected_project
        else:
            project = explicit_project
        setup = ProjectSetupDialog(catalog, project)
        if setup.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            print(setup.project.paths.root)
            return 0
        if explicit_project is not None:
            return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ProjectChooser", "ProjectSetupDialog", "main"]
