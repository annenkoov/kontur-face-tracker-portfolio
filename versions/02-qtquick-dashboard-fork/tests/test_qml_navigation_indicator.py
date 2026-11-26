"""Regression coverage for the animated active tab indicator geometry."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, QUrl
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine, QQmlComponent
from PySide6.QtQuick import QQuickItem
from PySide6.QtTest import QTest
from PySide6.QtQuickControls2 import QQuickStyle


class NavigationIndicatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        QQuickStyle.setStyle("Basic")
        cls.app = QGuiApplication.instance() or QGuiApplication([])

    def test_active_pill_overlays_selected_button_at_supported_widths(self) -> None:
        qml_root = Path(__file__).parents[1] / "src/face_tracker/qtui/qml"
        engine = QQmlApplicationEngine()
        engine.load(QUrl.fromLocalFile(str((qml_root / "Main.qml").resolve())))
        self.assertTrue(engine.rootObjects(), "Main.qml should load")
        root = engine.rootObjects()[0]
        header = next(
            item for item in root.findChildren(QObject)
            if "HeaderBar_" in item.metaObject().className()
        )
        self.assertIsInstance(header, QQuickItem)

        for width in (1080, 1584):
            header.setWidth(width)
            self.app.processEvents()
            items = list(self._visual_descendants(header))
            pill = next(item for item in items if item.objectName() == "navActivePill")
            buttons = sorted(
                (item for item in items if item.objectName().startswith("navigationButton_")),
                key=lambda item: int(item.objectName().rsplit("_", 1)[1]),
            )
            self.assertEqual(len(buttons), 3, "navigation should contain exactly three buttons")

            for index, tab_name in enumerate(("Главное", "История", "Подробно")):
                header.setProperty("activeTab", tab_name)
                QTest.qWait(220)
                target = buttons[index]
                pill_left = pill.mapToItem(header, 0, 0).x()
                button_left = target.mapToItem(header, 0, 0).x()
                self.assertAlmostEqual(
                    pill_left, button_left, delta=1.5,
                    msg=f"active pill should overlay tab {index} at width {width}",
                )
                self.assertAlmostEqual(
                    float(pill.width()), float(target.width()), delta=1.5,
                    msg=f"active pill width should match tab {index} at width {width}",
                )
        root.deleteLater()
        engine.deleteLater()
        self.app.processEvents()

    @staticmethod
    def _visual_descendants(item: QQuickItem):
        for child in item.childItems():
            yield child
            yield from NavigationIndicatorTests._visual_descendants(child)


if __name__ == "__main__":
    unittest.main()
