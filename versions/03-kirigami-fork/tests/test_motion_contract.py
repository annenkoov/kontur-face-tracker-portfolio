from pathlib import Path
import unittest
import os
import subprocess
ROOT = Path(__file__).resolve().parents[1]
class MotionContract(unittest.TestCase):
 def test_motion_is_opt_in_and_excludes_camera(self):
  main=(ROOT/'src/face_tracker/qtui/qml/Main.qml').read_text(encoding='utf-8')
  self.assertIn('property bool reducedMotion: true',main)
  self.assertIn('objectName: "historyPage"',main)
  self.assertIn('enabled: !window.reducedMotion',main)
  self.assertNotIn('Behavior on opacity', main[main.index('CameraPanel {'):main.index('ActionBar {')])
 def test_buttons_expose_focus_and_motion(self):
  text=(ROOT/'src/face_tracker/qtui/qml/components/ActionBar.qml').read_text(encoding='utf-8')
  self.assertIn('property bool reducedMotion: true', text)
  self.assertIn('Behavior on color',text)
  self.assertIn('activeFocus',text)
 def test_qml_motion_keeps_camera_and_button_geometry(self):
  craft = Path(os.environ.get("KONTUR_CRAFT_ROOT", ROOT / "work/craft-root")).resolve()
  python = craft / "bin/python.exe"
  if not python.is_file(): self.skipTest("Craft runtime unavailable")
  code = r'''
from pathlib import Path
import sys, ctypes
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0, "src")
from face_tracker.qtui.application import system_reduced_motion
with patch.object(ctypes, "windll", SimpleNamespace(user32=SimpleNamespace(SystemParametersInfoW=lambda *args: 0))):
 assert system_reduced_motion() is True
from PySide6.QtCore import QPoint, QUrl, Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuick import QQuickItem
from PySide6.QtTest import QTest
app=QGuiApplication([]); engine=QQmlApplicationEngine(); engine.load(QUrl.fromLocalFile(str(Path("src/face_tracker/qtui/qml/Main.qml").resolve())))
assert engine.rootObjects(); w=engine.rootObjects()[0]
for _ in range(5): app.processEvents()
b=w.findChild(QQuickItem,"startButton"); camera=w.findChild(QQuickItem,"cameraPanel"); history=w.findChild(QQuickItem,"historyPage")
assert b and camera and history
w.setProperty("reducedMotion",False); app.processEvents()
before=(b.x(),b.y(),b.width(),b.height()); b.forceActiveFocus(); app.processEvents(); assert b.hasActiveFocus()
p=b.mapToScene(b.boundingRect().center()); QTest.mousePress(w,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,QPoint(int(p.x()),int(p.y())))
QTest.qWait(45); assert b.property("down"), "press did not reach button"
assert (b.x(),b.y(),b.width(),b.height())==before
QTest.mouseRelease(w,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,QPoint(int(p.x()),int(p.y())))
assert not b.property("down")
def camera_chain():
 result=[]; node=camera
 while node:
  origin=node.mapToScene(QPointF(0,0)); unit=node.mapToScene(QPointF(1,1))
  result.append((node.opacity(),node.scale(),node.rotation(),origin.x(),origin.y(),unit.x(),unit.y()))
  node=node.parentItem()
 return result
from PySide6.QtCore import QPointF
w.setProperty("reducedMotion",False); w.chooseTab("Главное"); QTest.qWait(240)
w.chooseTab("История"); QTest.qWait(240); w.chooseTab("Главное"); QTest.qWait(240)
expected_camera=camera_chain()
w.chooseTab("История"); QTest.qWait(55)
mid_opacity=history.opacity(); mid_y=history.mapToScene(QPointF(0,0)).y()
assert history.isVisible() and 0 < mid_opacity < 1, ("intermediate opacity",mid_opacity)
assert camera_chain()==expected_camera, "camera or ancestor transform/opacity changed"
QTest.qWait(180); final_y=history.mapToScene(QPointF(0,0)).y()
assert abs(history.opacity()-1)<0.01 and mid_y>final_y+0.1, (mid_y,final_y)
assert camera_chain()==expected_camera
w.setProperty("reducedMotion",True); w.chooseTab("Главное"); app.processEvents(); w.chooseTab("История"); app.processEvents()
assert history.opacity()==1 and abs(history.mapToScene(QPointF(0,0)).y()-final_y)<0.1
assert camera_chain()==expected_camera
print("intermediate-opacity",mid_opacity,"slide",mid_y-final_y,"pressed",True)
print("motion-ok")
'''
  env=os.environ.copy(); env['QT_QPA_PLATFORM']='offscreen'; env['PATH']=str(craft/'bin')+os.pathsep+env.get('PATH',''); env['QT_PLUGIN_PATH']=str(craft/'plugins'); env['QML2_IMPORT_PATH']=str(craft/'qml')
  result=subprocess.run([str(python),'-I','-c',code],cwd=ROOT,env=env,capture_output=True,text=True,timeout=30)
  self.assertEqual(result.returncode,0,result.stdout+result.stderr); self.assertIn('motion-ok',result.stdout)
  print(result.stdout.strip())
if __name__=='__main__': unittest.main()
