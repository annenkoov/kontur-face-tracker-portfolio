"""Bootstrap the local PySide6 + Qt Quick interface."""

from __future__ import annotations

from pathlib import Path
import sys

from PySide6.QtCore import QUrl
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuickControls2 import QQuickStyle

from .controller import TrackerController


def main() -> int:
    """Start the QML dashboard with its Python camera/session controller."""
    QQuickStyle.setStyle("Basic")
    app = QGuiApplication(sys.argv)
    app.setApplicationName("Контур")
    app.setOrganizationName("Kontur")

    engine = QQmlApplicationEngine()
    controller = TrackerController()
    engine.addImageProvider("camera", controller.image_provider)
    app.aboutToQuit.connect(controller.close)

    qml_file = Path(__file__).resolve().parent / "qml" / "Main.qml"
    engine.load(QUrl.fromLocalFile(str(qml_file)))
    if not engine.rootObjects():
        controller.close()
        return 1
    # Main.qml owns a local `backend` property so components can also be
    # previewed without Python. Set it after load to avoid the local default
    # shadowing an identically named context property.
    engine.rootObjects()[0].setProperty("backend", controller)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
