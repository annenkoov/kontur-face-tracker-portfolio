"""The integration gate is opt-in; ordinary tests must never accept pip Qt as Craft."""
import json
from contextlib import closing
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import sysconfig
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "tools/kirigami_smoke/main.py"
SETUP = ROOT / "tools/craft_setup.ps1"
SPEC = spec_from_file_location("kirigami_runtime_gate", ENTRY)
GATE = module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


class RuntimeGuardTests(unittest.TestCase):
    def test_smoke_entry_exists(self):
        self.assertTrue(ENTRY.is_file(), "Compatibility entry point is missing")

    def test_probe_rejects_non_craft_interpreter(self):
        result = subprocess.run([sys.executable, str(ENTRY), "--probe",
                                 "--craft-root", str(ROOT / "work/not-a-craft-root")],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 2, result.stderr)
        evidence = json.loads(result.stdout)
        self.assertEqual(evidence["status"], "BLOCKED")
        self.assertTrue(any("interpreter" in error for error in evidence["errors"]))
        self.assertEqual(evidence["python_abi"]["ext_suffix"], sysconfig.get_config_var("EXT_SUFFIX"))
        self.assertEqual(evidence["python_abi"]["cache_tag"], sys.implementation.cache_tag)

    @unittest.skipUnless(sys.platform == "win32", "Windows DLL loader evidence")
    def test_enumeration_lists_the_running_python_module(self):
        # Windows venv Python is a launcher, which starts the base interpreter.
        self.assertIn(Path(sys._base_executable).resolve(), GATE.loaded_windows_modules())

    def test_checks_craft_file_ownership_and_hash_for_loaded_dlls(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "etc/blueprints/install.db"
            db.parent.mkdir(parents=True)
            owned = root / "bin/Qt6Core.dll"
            owned.parent.mkdir()
            owned.write_bytes(b"Craft")
            wheel = root / "bin/Lib/site-packages/PySide6/Qt6Core.dll"
            wheel.parent.mkdir(parents=True)
            wheel.write_bytes(b"same version wheel")
            with closing(sqlite3.connect(db)) as connection, connection:
                connection.executescript("""
                    CREATE TABLE packageList (packageId INTEGER, packagePath TEXT,
                        version TEXT, revision TEXT, cacheVersion TEXT);
                    CREATE TABLE fileList (packageId INTEGER, filename TEXT, fileHash TEXT);
                    INSERT INTO packageList VALUES (1, 'libs/qt6/qtbase', '6.12.0-1', 'source-rev', NULL);
                """)
                connection.execute("INSERT INTO fileList VALUES (1, ?, ?)",
                                   ("bin/Qt6Core.dll", "[SHA256]" + hashlib.sha256(b"Craft").hexdigest()))
            inventory, files = GATE.craft_install_inventory(root)
            self.assertEqual(inventory[0]["patch_level"], 1)
            self.assertEqual(inventory[0]["revision"], "source-rev")
            evidence, errors = GATE.verify_craft_binary_files(root, [owned], files)
            self.assertEqual(errors, [])
            self.assertEqual(evidence[0]["package"], "libs/qt6/qtbase")
            self.assertTrue(GATE.verify_craft_binary_files(root, [wheel], files)[1])
            owned.write_bytes(b"replaced in place")
            self.assertTrue(GATE.verify_craft_binary_files(root, [owned], files)[1])

    def test_historic_download_and_archive_do_not_establish_installed_origin(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = root / "etc/CraftSettings.ini"
            settings.parent.mkdir(parents=True)
            settings.write_text("ABI = windows-cl-msvc2022-x86_64\n", encoding="utf-8")
            packages = [
                {"package": "libs/qt6/qtbase", "version": "6.12.0", "cache_version": "26.10"},
                {"package": "libs/qt6/qtdeclarative", "cache_version": "26.10"},
            ]
            for package in ("python-modules/pyside6", "kde/frameworks/tier1/kirigami"):
                compiler = (root / "build" / package / "work/build/CMakeFiles/3.31"
                            / "CMakeCXXCompiler.cmake")
                compiler.parent.mkdir(parents=True)
                compiler.write_text(
                    'set(CMAKE_CXX_COMPILER "cl.exe")\n'
                    'set(CMAKE_CXX_COMPILER_ID "MSVC")\n'
                    'set(CMAKE_CXX_COMPILER_VERSION "19.44")\n',
                    encoding="utf-8",
                )
            package = "libs/qt6/qtbase"
            relative = ("26.10/windows/cl/msvc2022/x86_64/RelWithDebInfo/"
                        + package + "/qtbase-6.12.0-162-20261022T104712-windows-cl-msvc2022-x86_64.7z")
            url = "https://files.kde.org/craft/Qt6/" + relative
            archive = root / "download/cache" / relative
            archive.parent.mkdir(parents=True)
            archive.write_bytes(b"historic archive")
            log = root / "logs" / package / "qtbase.log"
            log.parent.mkdir(parents=True)
            log.write_text(
                f"Command ['curl.exe', '-o', '{archive}', '{url}'] succeeded with exit code 0\n"
                f"Path = {archive}\n",
                encoding="utf-8",
            )

            evidence = GATE.cmake_compilers(root, packages)
            self.assertEqual(evidence[package]["source"], "unverified")
            self.assertEqual(evidence[package]["abi"], "windows-cl-msvc2022-x86_64")
            self.assertEqual(evidence[package]["cache_version"], "26.10")
            self.assertNotIn("archive_url", evidence[package])
            self.assertNotIn("archive_sha256", evidence[package])
            self.assertEqual(evidence["python-modules/pyside6"]["source"], "local CMake build")

    @unittest.skipUnless(os.environ.get("KONTUR_CRAFT_ROOT"), "Requires isolated Craft runtime")
    def test_real_kirigami_window_and_provider_pixels(self):
        root = Path(os.environ["KONTUR_CRAFT_ROOT"])
        evidence_path = ROOT / "work/kirigami-runtime.json"
        result = subprocess.run([str(root / "bin/python.exe"), "-I", str(ENTRY),
                                 "--craft-root", str(root), "--evidence", str(evidence_path)],
                                capture_output=True, text=True, timeout=40)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        self.assertEqual(evidence["status"], "PASS")
        self.assertTrue(evidence["image_ready"])
        self.assertGreater(evidence["provider_requests"], 0)
        self.assertEqual(evidence["rendered_center"], "#26a69a")
        self.assertTrue(Path(evidence["screenshot"]).is_file())
        self.assertTrue(evidence["loaded_runtime_binaries"])
        for binding in evidence["modules"].values():
            self.assertTrue(binding["filename"].endswith(".pyd"))


@unittest.skipUnless(sys.platform == "win32" and shutil.which("pwsh"), "PowerShell on Windows")
class CraftSetupGuardTests(unittest.TestCase):
    def test_rejects_junction_before_overwriting_receipt(self):
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "project"
            tools_dir = workspace / "tools"
            tools_dir.mkdir(parents=True)
            entry = Path(shutil.copy(SETUP, tools_dir))
            outside = Path(tmp) / "outside"
            outside.mkdir()
            sentinel = outside / "craft-preflight.json"
            sentinel.write_text("keep", encoding="utf-8")
            subprocess.run(["pwsh", "-NoProfile", "-Command",
                            "New-Item -ItemType Junction -Path $env:TEST_LINK -Target $env:TEST_TARGET | Out-Null"],
                           env={**os.environ, "TEST_LINK": str(workspace / "work"), "TEST_TARGET": str(outside)},
                           check=True, capture_output=True, timeout=30)
            result = subprocess.run(["pwsh", "-NoProfile", "-File", str(entry)],
                                    capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("reparse point", result.stderr.lower())
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_build_rejects_junctions_in_short_drive_and_ecm_paths(self):
        for relative in ("short-drive-resume", "share/ECM/kde-modules"):
            with self.subTest(relative=relative), TemporaryDirectory() as tmp:
                workspace = Path(tmp) / "project"
                tools_dir = workspace / "tools"
                tools_dir.mkdir(parents=True)
                entry = Path(shutil.copy(SETUP, tools_dir))
                outside = Path(tmp) / "outside"
                outside.mkdir()
                module = outside / "KDEClangFormat.cmake"
                module.write_text("keep", encoding="utf-8")
                root = workspace / "work/craft-root"
                link = root / relative
                link.parent.mkdir(parents=True)
                subprocess.run(["pwsh", "-NoProfile", "-Command",
                                "New-Item -ItemType Junction -Path $env:TEST_LINK -Target $env:TEST_TARGET | Out-Null"],
                               env={**os.environ, "TEST_LINK": str(link), "TEST_TARGET": str(outside)},
                               check=True, capture_output=True, timeout=30)
                result = subprocess.run(["pwsh", "-NoProfile", "-File", str(entry),
                                         "-Mode", "Build", "-CraftRoot", str(root)],
                                        capture_output=True, text=True, timeout=30)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("reparse point", result.stderr.lower())
                self.assertEqual(module.read_text(encoding="utf-8"), "keep")
                self.assertFalse((workspace / "work/craft-preflight.json").exists())

    def test_build_rejects_unpinned_input_before_package_execution(self):
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            tools_dir = workspace / "tools"
            tools_dir.mkdir()
            entry = Path(shutil.copy(SETUP, tools_dir))
            root = workspace / "work/craft-root"
            root.mkdir(parents=True)
            result = subprocess.run(["pwsh", "-NoProfile", "-File", str(entry),
                                     "-Mode", "Build", "-CraftRoot", str(root)],
                                    capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("pin check failed", result.stderr.lower())
            self.assertFalse((root / "etc/blueprints/install.db").exists())
