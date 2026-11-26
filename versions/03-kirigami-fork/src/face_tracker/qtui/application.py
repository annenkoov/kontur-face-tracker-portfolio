"""Bootstrap the local PySide6 + Qt Quick interface."""

from __future__ import annotations

from pathlib import Path
import sys

from PySide6.QtCore import QLibraryInfo, QUrl
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuickControls2 import QQuickStyle

from .controller import TrackerController


def system_reduced_motion() -> bool:
    """Respect Windows client-area animation setting; prefer stillness on failure."""
    if sys.platform != "win32":
        return True
    try:
        import ctypes
        enabled = ctypes.c_int()
        ok = ctypes.windll.user32.SystemParametersInfoW(0x1042, 0, ctypes.byref(enabled), 0)
        return not bool(enabled.value) if ok else True
    except (AttributeError, OSError):
        return True


def main() -> int:
    """Start the QML dashboard with its Python camera/session controller."""
    QQuickStyle.setStyle("Basic")
    app = QGuiApplication(sys.argv)
    app.setApplicationName("Контур")
    app.setOrganizationName("Kontur")

    engine = QQmlApplicationEngine()
    # В Craft QML-модули (в том числе Kirigami) лежат рядом с Qt runtime,
    # поэтому добавляем путь из той же сборки, что загрузила PySide6.
    engine.addImportPath(QLibraryInfo.path(QLibraryInfo.LibraryPath.QmlImportsPath))
    controller = TrackerController()
    engine.addImageProvider("camera", controller.image_provider)
    app.aboutToQuit.connect(controller.close)

    qml_file = Path(__file__).resolve().parent / "qml" / "Main.qml"
    reduced_motion = system_reduced_motion()
    engine.load(QUrl.fromLocalFile(str(qml_file)))
    if not engine.rootObjects():
        controller.close()
        return 1
    # Main.qml owns a local `backend` property so components can also be
    # previewed without Python. Set it after load to avoid the local default
    # shadowing an identically named context property.
    engine.rootObjects()[0].setProperty("reducedMotion", reduced_motion)
    engine.rootObjects()[0].setProperty("backend", controller)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
