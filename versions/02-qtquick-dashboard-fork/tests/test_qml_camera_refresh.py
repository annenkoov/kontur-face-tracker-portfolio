"""Regression coverage for the Qt Quick camera preview refresh state."""

from __future__ import annotations

import os
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QUrl, QSize
from PySide6.QtGui import QGuiApplication, QImage, QColor
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuick import QQuickImageProvider, QQuickItem
from PySide6.QtTest import QTest
from PySide6.QtQuickControls2 import QQuickStyle


class DelayedCameraProvider(QQuickImageProvider):
    """Small synthetic source that makes asynchronous loading observable."""

    def __init__(self) -> None:
        super().__init__(QQuickImageProvider.Image)

    def requestImage(self, image_id: str, size: QSize, requested_size: QSize) -> QImage:
        del image_id, requested_size
        time.sleep(0.025)
        image = QImage(64, 64, QImage.Format.Format_ARGB32)
        image.fill(QColor("#728cff"))
        size.setWidth(image.width())
        size.setHeight(image.height())
        return image


class CameraRefreshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        QQuickStyle.setStyle("Basic")
        cls.app = QGuiApplication.instance() or QGuiApplication([])

    def setUp(self) -> None:
        self.engine = QQmlApplicationEngine()
        self.engine.addImageProvider("camera", DelayedCameraProvider())
        qml_path = Path(__file__).parents[1] / "src/face_tracker/qtui/qml/Main.qml"
        self.engine.load(QUrl.fromLocalFile(str(qml_path.resolve())))
        self.assertTrue(self.engine.rootObjects(), "Main.qml should load for camera state test")
        self.window = self.engine.rootObjects()[0]
        self.window.show()
        self.panel = next(
            item for item in self.window.findChildren(QQuickItem)
            if item.metaObject().indexOfProperty("cameraState") >= 0
        )
        self.image = next(
            item for item in self.panel.findChildren(QQuickItem)
            if item.metaObject().className().startswith("QQuickImage")
        )
        self.landmark_overlay = next(
            item for item in self.panel.findChildren(QQuickItem)
            if item.metaObject().className().startswith("QQuickCanvasItem")
        )
        self.wait_label = next(
            item for item in self.panel.findChildren(QQuickItem)
            if item.property("text") in {"ОЖИДАНИЕ КАДРА", "КАМЕРА ОСТАНОВЛЕНА"}
        )

    def tearDown(self) -> None:
        self.window.close()
        self.engine.deleteLater()
        self.app.processEvents()

    def test_wait_message_stays_hidden_while_preview_urls_refresh(self) -> None:
        self.panel.setProperty("cameraState", "running")
        self.panel.setProperty("landmarkPoints", [{"x": 0.5, "y": 0.5}])
        self.panel.setProperty("frameRevision", 1)
        deadline = time.monotonic() + 2
        while not self.image.isVisible() and time.monotonic() < deadline:
            QTest.qWait(10)
        self.assertTrue(self.image.isVisible(), "first synthetic frame should load")
        self.assertFalse(self.wait_label.isVisible(), "wait label should hide after first frame")
        self.assertTrue(self.landmark_overlay.isVisible(), "valid landmarks should show over the first frame")

        for revision in range(2, 10):
            self.panel.setProperty("frameRevision", revision)
            QTest.qWait(5)  # observe the asynchronous loading interval before the 25 ms response
            self.assertFalse(
                self.wait_label.isVisible(),
                "refreshing a valid frame must not show the first-frame placeholder",
            )
            self.assertTrue(
                self.landmark_overlay.isVisible(),
                "valid landmarks must remain visible while the retained frame is refreshed",
            )
            QTest.qWait(35)
            QTest.qWait(50)

        self.panel.setProperty("cameraState", "stopped")
        QTest.qWait(50)
        self.assertTrue(self.wait_label.isVisible(), "stopping the source restores the empty state")


if __name__ == "__main__":
    unittest.main()
