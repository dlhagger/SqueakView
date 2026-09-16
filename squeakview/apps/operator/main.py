"""Entry point for the consolidated operator GUI."""
from __future__ import annotations

from pathlib import Path
import signal
from typing import Callable

from PySide6 import QtCore, QtGui, QtWidgets

from squeakview.project import AppPaths
from .gui.main_window import MainWindow


def _install_close_signal_handlers(
    app: QtWidgets.QApplication,
    window_provider: Callable[[], MainWindow | None],
) -> None:
    """Route terminal/session signals through the guarded Qt close lifecycle."""

    def request_close(_signum: int, _frame: object) -> None:
        window = window_provider()
        callback = window.close if window is not None else app.quit
        QtCore.QTimer.singleShot(0, callback)

    handled = (signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGHUP", None))
    for signum in handled:
        if signum is not None:
            signal.signal(signum, request_close)


def _launch_main_window(
    app: QtWidgets.QApplication,
    window_holder: dict[str, MainWindow],
    *,
    splash: QtWidgets.QSplashScreen | None,
    app_icon: QtGui.QIcon | None,
    window_factory: Callable[[], MainWindow] = MainWindow,
) -> bool:
    """Construct the required GUI or terminate the lease process visibly."""

    if splash is not None:
        splash.close()
        app.processEvents()
    try:
        window = window_factory()
    except Exception as exc:
        print(
            "[FATAL] SqueakView GUI could not connect to its durable supervisor: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        app.exit(1)
        return False
    if app_icon is not None and not app_icon.isNull():
        window.setWindowIcon(app_icon)
    window_holder["win"] = window
    window.showMaximized()
    return True


def main() -> int:
    app = QtWidgets.QApplication([])
    app.setStyle("Fusion")
    app.setApplicationName("SqueakView")
    app.setOrganizationName("SqueakView")

    splash = None
    app_icon = None
    logo_path = AppPaths.discover().logo
    if logo_path.exists():
        app_icon = QtGui.QIcon(str(logo_path))
        if not app_icon.isNull():
            app.setWindowIcon(app_icon)
            app.setDesktopFileName("squeakview")
        pixmap = QtGui.QPixmap(str(logo_path))
        if not pixmap.isNull():
            splash = QtWidgets.QSplashScreen(pixmap)
            splash.setWindowFlag(QtCore.Qt.WindowType.FramelessWindowHint)
            splash.setEnabled(False)
            splash.show()
            app.processEvents()

    window_holder: dict[str, MainWindow] = {}
    _install_close_signal_handlers(app, lambda: window_holder.get("win"))

    def launch_main_window() -> None:
        _launch_main_window(
            app,
            window_holder,
            splash=splash,
            app_icon=app_icon,
        )

    if splash is not None:
        QtCore.QTimer.singleShot(3000, launch_main_window)
    else:
        QtCore.QTimer.singleShot(0, launch_main_window)

    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
