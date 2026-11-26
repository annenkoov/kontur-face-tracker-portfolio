"""Проверки импорта Craft runtime без запуска веб-камеры."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
CRAFT_ROOT = Path(os.environ.get("KONTUR_CRAFT_ROOT", ROOT / "work/craft-root"))
CRAFT_PYTHON = CRAFT_ROOT / "bin" / "python.exe"


@unittest.skipUnless(CRAFT_PYTHON.is_file(), "Craft runtime is not installed")
class CraftRuntimeImportTests(unittest.TestCase):
    def run_craft(self, code: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["KONTUR_CRAFT_ROOT"] = str(CRAFT_ROOT.resolve())
        return subprocess.run(
            [str(CRAFT_PYTHON), "-I", "-c", code], cwd=ROOT, env=env,
            capture_output=True, text=True, timeout=90, check=False,
        )

    def test_qt_qml_kirigami_and_analysis_imports(self):
        code = (
            "import cv2, mediapipe, numpy, onnx, onnxruntime, PIL; "
            "import PySide6.QtCore, PySide6.QtGui, PySide6.QtQml, "
            "PySide6.QtQuick, PySide6.QtQuickControls2; "
            "from hsemotion_onnx.facial_emotions import HSEmotionRecognizer; "
            "print('imports-ok')"
        )
        result = self.run_craft(code)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("imports-ok", result.stdout)

    def test_controller_modules_import_without_opening_camera(self):
        code = (
            "import sys; sys.path.insert(0, 'src'); "
            "from face_tracker.qtui.controller import TrackerController; "
            "from face_tracker.worker import AnalysisWorker; "
            "from face_tracker.camera import CameraSource; print('controller-ok')"
        )
        result = self.run_craft(code)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("controller-ok", result.stdout)

    def test_minimal_application_qml_load_and_quit_without_camera(self):
        code = (
            "import sys; from pathlib import Path; "
            "from PySide6.QtCore import QLibraryInfo,QTimer,QUrl; "
            "from PySide6.QtGui import QGuiApplication; "
            "from PySide6.QtQml import QQmlApplicationEngine; "
            "sys.path.insert(0,'src'); "
            "from face_tracker.qtui.controller import TrackerController; "
            "app=QGuiApplication([]); engine=QQmlApplicationEngine(); "
            "engine.addImportPath(QLibraryInfo.path(QLibraryInfo.LibraryPath.QmlImportsPath)); "
            "controller=TrackerController(); engine.addImageProvider('camera',controller.image_provider); "
            "engine.load(QUrl.fromLocalFile(str(Path('src/face_tracker/qtui/qml/Main.qml').resolve()))); "
            "assert engine.rootObjects(), 'QML failed to load'; "
            "engine.rootObjects()[0].setProperty('backend',controller); "
            "QTimer.singleShot(250,app.quit); result=app.exec(); controller.close(); "
            "print('qml-launch-quit-ok'); raise SystemExit(result)"
        )
        env = os.environ.copy()
        env["KONTUR_CRAFT_ROOT"] = str(CRAFT_ROOT.resolve())
        env["QT_QPA_PLATFORM"] = "offscreen"
        env["PATH"] = str(CRAFT_ROOT / "bin") + os.pathsep + env.get("PATH", "")
        env["QT_PLUGIN_PATH"] = str(CRAFT_ROOT / "plugins")
        env["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(CRAFT_ROOT / "plugins" / "platforms")
        env["QML2_IMPORT_PATH"] = os.pathsep.join((str(CRAFT_ROOT / "qml"), str(CRAFT_ROOT / "bin")))
        result = subprocess.run(
            [str(CRAFT_PYTHON), "-I", "-c", code], cwd=ROOT, env=env,
            capture_output=True, text=True, timeout=30, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("qml-launch-quit-ok", result.stdout)

    def test_launcher_bootstrap_defines_isolated_paths_before_qt_import(self):
        # The launcher is covered by a static contract: its subprocess starts
        # isolated Python, then the bootstrap sets DLL/QML paths before import.
        bootstrap = (ROOT / "tools/bootstrap_kirigami.py").read_text(encoding="utf-8")
        launcher = (ROOT / "tools/run_kirigami.ps1").read_text(encoding="utf-8")
        self.assertLess(bootstrap.index("os.add_dll_directory"), bootstrap.index("from face_tracker.qtui.application"))
        self.assertIn("QML2_IMPORT_PATH", bootstrap)
        self.assertIn("-I", launcher)
        self.assertIn("QT_PLUGIN_PATH", launcher)
        self.assertIn("QT_QPA_PLATFORM_PLUGIN_PATH", launcher)
        self.assertIn("PYSIDE_DISABLE_INTERNAL_QT_CONF", launcher)
        self.assertNotIn("$envFile", launcher)
        self.assertIn("SetEnvironmentVariable($name, $previousEnvironment[$name], 'Process')", launcher)


if __name__ == "__main__":
    unittest.main()
