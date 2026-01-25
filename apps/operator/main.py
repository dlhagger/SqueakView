"""Entry point for the consolidated operator GUI."""
from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from .gui.main_window import MainWindow


def main() -> None:
    app = QtWidgets.QApplication([])

    splash = None
    logo_path = Path(__file__).resolve().parents[2] / "SqueakView_logo.png"
    if logo_path.exists():
        pixmap = QtGui.QPixmap(str(logo_path))
        if not pixmap.isNull():
            splash = QtWidgets.QSplashScreen(pixmap)
            splash.setWindowFlag(QtCore.Qt.WindowType.FramelessWindowHint)
            splash.setEnabled(False)
            splash.show()
            app.processEvents()

    window_holder: dict[str, MainWindow] = {}

    def launch_main_window() -> None:
        win = MainWindow()
        window_holder["win"] = win
        win.show()
        if splash is not None:
            splash.finish(win)

    if splash is not None:
        QtCore.QTimer.singleShot(3000, launch_main_window)
    else:
        launch_main_window()

    app.exec()


if __name__ == "__main__":
    main()
