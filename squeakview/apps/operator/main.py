"""Entry point for the consolidated operator GUI."""
from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from squeakview import config as squeakview_config
from .gui.main_window import MainWindow


def main() -> None:
    app = QtWidgets.QApplication([])
    app.setStyle("Fusion")
    app.setApplicationName("SqueakView")
    app.setOrganizationName("SqueakView")

    splash = None
    app_icon = None
    logo_path = squeakview_config.WORKSPACE / "SqueakView_logo.png"
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

    def launch_main_window() -> None:
        if splash is not None:
            splash.close()
            app.processEvents()
        win = MainWindow()
        if app_icon is not None and not app_icon.isNull():
            win.setWindowIcon(app_icon)
        window_holder["win"] = win
        win.showMaximized()

    if splash is not None:
        QtCore.QTimer.singleShot(3000, launch_main_window)
    else:
        launch_main_window()

    app.exec()


if __name__ == "__main__":
    main()
