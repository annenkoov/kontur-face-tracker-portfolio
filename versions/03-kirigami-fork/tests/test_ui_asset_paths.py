"""Ensure local QML SVG references resolve and remain self-contained."""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
QML = ROOT / "src/face_tracker/qtui/qml"


class UiAssetPathTests(unittest.TestCase):
    def test_every_local_svg_reference_resolves_and_has_24px_viewbox(self):
        refs = []
        for source in QML.rglob("*.qml"):
            for relative in re.findall(r'source:\s*"([^"]+\.svg)"', source.read_text(encoding="utf-8")):
                refs.append((source, (source.parent / relative).resolve()))
        self.assertTrue(refs, "expected at least one local SVG asset")
        for source, asset in refs:
            with self.subTest(source=source.name, asset=asset.name):
                self.assertTrue(asset.is_file(), f"missing SVG referenced by {source}: {asset}")
                self.assertIn('viewBox="0 0 24 24"', asset.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
