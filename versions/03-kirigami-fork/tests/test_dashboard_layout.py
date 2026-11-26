"""Layout checks for the real QML dashboard without opening a camera."""
import os
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
CRAFT = Path(os.environ.get('KONTUR_CRAFT_ROOT', ROOT / 'work/craft-root')).resolve()
PYTHON = CRAFT / 'bin/python.exe'

@unittest.skipUnless(PYTHON.is_file(), 'Craft runtime unavailable')
class DashboardLayoutTests(unittest.TestCase):
    def test_trend_labels_distinguish_idle_calibration_and_real_score(self):
        code = '''
from pathlib import Path
from PySide6.QtCore import QUrl
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuick import QQuickItem
app=QGuiApplication([]); e=QQmlApplicationEngine()
e.load(QUrl.fromLocalFile(str(Path("src/face_tracker/qtui/qml/Main.qml").resolve())))
assert e.rootObjects(), "dashboard failed to load"
w=e.rootObjects()[0]; app.processEvents()
panel=w.findChild(QQuickItem,"trendPanel"); assert panel is not None
label=w.findChild(QQuickItem,"trendValueLabel"); assert label is not None
hint=w.findChild(QQuickItem,"trendScaleHint"); assert hint is not None
assert hint.property("font").pixelSize() >= 13, hint.property("font").pixelSize()
for width,height in ((1584,990),(1080,700)):
 w.resize(width,height)
 for _ in range(5): app.processEvents()
 assert hint.property("truncated") is False, (width,height,hint.property("width"),hint.property("truncated"))
 hintText=str(hint.property("text"))
 assert "мало изменений" in hintText and "сильное отклонение" in hintText and "личной базы" in hintText, hintText
assert label.property("text")=="Нет данных", label.property("text")
panel.setProperty("collectingBaseline",True); app.processEvents()
assert label.property("text")=="Собираем базу", label.property("text")
panel.setProperty("hasScore",True); panel.setProperty("scoreValue",24); app.processEvents()
assert label.property("text")=="24 / 100", label.property("text")
panel.setProperty("scoreHistory",[{"timestamp":1,"value":22},None,{"timestamp":3,"value":26}])
assert panel.property("scoreHistory")[1] is None, "null gap marker must be preserved"
print("trend-states-ok")
'''
        env = os.environ.copy()
        env['QT_QPA_PLATFORM'] = 'offscreen'
        env['PATH'] = str(CRAFT / 'bin') + os.pathsep + env.get('PATH', '')
        env['QT_PLUGIN_PATH'] = str(CRAFT / 'plugins')
        env['QML2_IMPORT_PATH'] = str(CRAFT / 'qml')
        result = subprocess.run([str(PYTHON), '-I', '-c', code], cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('trend-states-ok', result.stdout)

    def test_camera_frame_retains_current_texture_while_next_frame_loads(self):
        code = '''
from pathlib import Path
from PySide6.QtCore import QUrl
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuick import QQuickItem
app=QGuiApplication([])
e=QQmlApplicationEngine()
e.load(QUrl.fromLocalFile(str(Path("src/face_tracker/qtui/qml/Main.qml").resolve())))
assert e.rootObjects(), "dashboard failed to load"
w=e.rootObjects()[0]
app.processEvents()
image=w.findChild(QQuickItem, "cameraFrame")
assert image is not None, "camera image item is not exposed for validation"
assert image.property("retainWhileLoading") is True, "new source may discard the last camera texture"
print("camera-frame-retention-ok")
'''
        env = os.environ.copy()
        env['QT_QPA_PLATFORM'] = 'offscreen'
        env['PATH'] = str(CRAFT / 'bin') + os.pathsep + env.get('PATH', '')
        env['QT_PLUGIN_PATH'] = str(CRAFT / 'plugins')
        env['QML2_IMPORT_PATH'] = str(CRAFT / 'qml')
        result = subprocess.run([str(PYTHON), '-I', '-c', code], cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('camera-frame-retention-ok', result.stdout)

    def test_panels_fit_at_reference_and_minimum_sizes(self):
        code = '''
from pathlib import Path
from PySide6.QtCore import QObject, QPointF, QUrl
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuick import QQuickItem
app = QGuiApplication([])
e = QQmlApplicationEngine()
e.load(QUrl.fromLocalFile(str(Path("src/face_tracker/qtui/qml/Main.qml").resolve())))
assert e.rootObjects(), "dashboard failed to load"
w=e.rootObjects()[0]
for width,height in ((1584,990),(1080,700)):
 w.resize(width,height)
 for _ in range(5): app.processEvents()
 def item(name):
  result=w.findChild(QQuickItem,name)
  assert result is not None, name
  return result
 names=("header","cameraPanel","actionBar","currentStatus","statusGauge","trendPanel","observationPanel")
 for name in names:
  obj=item(name); rect=(obj.width(),obj.height())
  assert rect[0]>0 and rect[1]>0, (name,rect)
  top=obj.mapToScene(QPointF(0,0)); bottom=obj.mapToScene(QPointF(obj.width(),obj.height()))
  assert top.x()>=-1 and top.y()>=-1 and bottom.x()<=width+1 and bottom.y()<=height+1, (name,top,bottom)
 assert item("trendPanel").height()>100, item("trendPanel").height()
 assert item("currentStatus").mapToScene(QPointF(0,item("currentStatus").height())).y() <= item("statusGauge").mapToScene(QPointF(0,0)).y()+1
 assert item("statusGauge").mapToScene(QPointF(0,item("statusGauge").height())).y() <= item("trendPanel").mapToScene(QPointF(0,0)).y()+1
 nav=item("navTabs"); privacy=item("privacyBadge"); camera=item("cameraPanel"); actions=item("actionBar"); observation=item("observationPanel")
 navCenter=nav.mapToScene(QPointF(nav.width()/2,0)).x()
 assert abs(navCenter-width/2)<3, (width,navCenter)
 assert nav.mapToScene(QPointF(nav.width(),0)).x() < privacy.mapToScene(QPointF(0,0)).x()
 assert abs(camera.mapToScene(QPointF(0,camera.height())).y()-item("trendPanel").mapToScene(QPointF(0,item("trendPanel").height())).y())<8
 assert abs(actions.mapToScene(QPointF(0,0)).y()-observation.mapToScene(QPointF(0,0)).y())<8
 for name in names[1:]: assert item(name).property("radius")==18, name
 assert item("currentStatus").property("lightColor").name()=="#8b6fff"
 assert item("statusGauge").property("lightColor").name()=="#51c9ec"
 assert item("observationPanel").property("lightColor").name()=="#886aff"
image=w.grabWindow(); dpr=image.devicePixelRatio()
for name in names[1:]:
 obj=item(name); pos=obj.mapToScene(QPointF(0,0)); right=obj.mapToScene(QPointF(obj.width(),0))
 def pixel(x,y): return image.pixelColor(round(x*dpr),round(y*dpr))
 for x,outside in ((pos.x()+2,pos.x()-3),(right.x()-2,right.x()+3)):
  a=pixel(x,pos.y()+1); b=pixel(outside,pos.y()+1)
  assert max(abs(a.red()-b.red()),abs(a.green()-b.green()),abs(a.blue()-b.blue()))<24,(name,a.name(),b.name())
for name,channel in (("currentStatus","red"),("statusGauge","green"),("observationPanel","red")):
 obj=item(name)
 lp=obj.mapToScene(QPointF(obj.width()*0.2,8)); rp=obj.mapToScene(QPointF(obj.width()*0.9,8))
 a=pixel(lp.x(),lp.y()); b=pixel(rp.x(),rp.y())
 assert getattr(b,channel)()-getattr(a,channel)()>4,(name,a.name(),b.name())
print("layout-ok")
'''
        env = os.environ.copy()
        env['QT_QPA_PLATFORM'] = 'offscreen'
        env['PATH'] = str(CRAFT / 'bin') + os.pathsep + env.get('PATH', '')
        env['QT_PLUGIN_PATH'] = str(CRAFT / 'plugins')
        env['QML2_IMPORT_PATH'] = str(CRAFT / 'qml')
        result = subprocess.run([str(PYTHON), '-I', '-c', code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('layout-ok', result.stdout)



    def test_live_surface_masks_corners_using_synthetic_image(self):
        code = '''
from pathlib import Path
from PySide6.QtCore import QEventLoop, QPointF, QTimer, QUrl
from PySide6.QtGui import QColor, QGuiApplication, QImage
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuick import QQuickImageProvider, QQuickItem
app=QGuiApplication([]); e=QQmlApplicationEngine()
size=[900,800]
class Provider(QQuickImageProvider):
 def __init__(self): super().__init__(QQuickImageProvider.Image)
 def requestImage(self, image_id, outsize, requested):
  im=QImage(size[0],size[1],QImage.Format_RGB32); im.fill(QColor("#ff0000"))
  if outsize is not None: outsize.setWidth(size[0]); outsize.setHeight(size[1])
  return im
provider=Provider(); e.addImageProvider("camera",provider)
e.load(QUrl.fromLocalFile(str(Path("src/face_tracker/qtui/qml/Main.qml").resolve())))
assert e.rootObjects(), "load failed"
w=e.rootObjects()[0]; w.resize(1080,700)
for _ in range(5): app.processEvents()
camera=w.findChild(QQuickItem,"cameraPanel"); size[:]=[round(camera.width()),round(camera.height())]
camera.setProperty("cameraState","active")
loop=QEventLoop(); QTimer.singleShot(1000,loop.quit); loop.exec()
image=w.grabWindow(); assert not image.isNull()
top=camera.mapToScene(QPointF(0,0)); center=camera.mapToScene(QPointF(camera.width()/2,camera.height()/2))
dpr=image.devicePixelRatio()
def color(x,y): return image.pixelColor(round(x*dpr),round(y*dpr))
inside=color(center.x(),center.y())
assert inside.red()>220 and inside.green()<30, inside.name()
for x,y,outside in ((2,2,-3),(camera.width()-2,2,camera.width()+3),(2,camera.height()-2,-3),(camera.width()-2,camera.height()-2,camera.width()+3)):
 p=camera.mapToScene(QPointF(x,y)); q=camera.mapToScene(QPointF(outside,y))
 corner=color(p.x(),p.y()); bg=color(q.x(),q.y())
 assert not (corner.red()>220 and corner.green()<30), corner.name()
 assert max(abs(corner.red()-bg.red()),abs(corner.green()-bg.green()),abs(corner.blue()-bg.blue()))<24,(corner.name(),bg.name())
print("corners-ok")
'''
        env = os.environ.copy()
        env['QT_QPA_PLATFORM'] = 'offscreen'
        env['PATH'] = str(CRAFT / 'bin') + os.pathsep + env.get('PATH', '')
        env['QT_PLUGIN_PATH'] = str(CRAFT / 'plugins')
        env['QML2_IMPORT_PATH'] = str(CRAFT / 'qml')
        result = subprocess.run([str(PYTHON), '-I', '-c', code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('corners-ok', result.stdout)

    def test_timestamped_chart_area_reaches_baseline(self):
        code = '''
from pathlib import Path
from PySide6.QtCore import QEventLoop, QPointF, QTimer, QUrl
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuick import QQuickItem
app=QGuiApplication([]); e=QQmlApplicationEngine()
e.load(QUrl.fromLocalFile(str(Path("src/face_tracker/qtui/qml/Main.qml").resolve())))
assert e.rootObjects()
w=e.rootObjects()[0]; w.resize(1584,900)
def wait():
 loop=QEventLoop(); QTimer.singleShot(500,loop.quit); loop.exec()
wait(); panel=w.findChild(QQuickItem,"trendPanel"); canvas=w.findChild(QQuickItem,"trendCanvas")
before=w.grabWindow(); assert not before.isNull()
panel.setProperty("scoreHistory",[{"timestamp":0,"value":70},{"timestamp":20,"value":76},{"timestamp":40,"value":55},{"timestamp":60,"value":70}])
wait(); after=w.grabWindow(); assert not after.isNull()
left=38; right=canvas.width()-10; top=12; bottom=canvas.height()-26
# Every sample is between the line and baseline, including the left part
# that an erroneous diagonal close would leave unfilled.
for fraction in (0.16,0.32,0.48):
 point=canvas.mapToScene(QPointF(left+(right-left)*fraction, top+(bottom-top)*0.76))
 dpr=after.devicePixelRatio(); x=round(point.x()*dpr); y=round(point.y()*dpr)
 a=after.pixelColor(x,y); b=before.pixelColor(x,y)
 delta=abs(a.red()-b.red())+abs(a.green()-b.green())+abs(a.blue()-b.blue())
 assert delta>=4, (fraction,a.name(),b.name(),delta)
print("area-ok")
'''
        env = os.environ.copy()
        env['QT_QPA_PLATFORM'] = 'offscreen'
        env['PATH'] = str(CRAFT / 'bin') + os.pathsep + env.get('PATH', '')
        env['QT_PLUGIN_PATH'] = str(CRAFT / 'plugins')
        env['QML2_IMPORT_PATH'] = str(CRAFT / 'qml')
        result = subprocess.run([str(PYTHON), '-I', '-c', code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('area-ok', result.stdout)

    def test_timestamped_chart_keeps_null_gap_in_line(self):
        code = '''
from pathlib import Path
from PySide6.QtCore import QEventLoop, QPointF, QTimer, QUrl
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuick import QQuickItem
app=QGuiApplication([]); e=QQmlApplicationEngine()
e.load(QUrl.fromLocalFile(str(Path("src/face_tracker/qtui/qml/Main.qml").resolve())))
assert e.rootObjects()
w=e.rootObjects()[0]; w.resize(1584,900)
loop=QEventLoop(); QTimer.singleShot(400,loop.quit); loop.exec()
panel=w.findChild(QQuickItem,"trendPanel"); canvas=w.findChild(QQuickItem,"trendCanvas")
# Две точки по краям окна с пропуском между ними: линия не должна пересекать середину.
panel.setProperty("scoreHistory",[{"timestamp":0,"value":60},None,{"timestamp":60,"value":60}])
loop=QEventLoop(); QTimer.singleShot(400,loop.quit); loop.exec()
image=w.grabWindow(); assert not image.isNull()
mid=canvas.mapToScene(QPointF(canvas.width()/2, 12+(canvas.height()-38)*0.4))
dpr=image.devicePixelRatio(); cx=round(mid.x()*dpr); cy=round(mid.y()*dpr)
# Если gap удалён, соединительная кривая проходит здесь (около уровня 60).
colors=[image.pixelColor(x,y) for x in range(cx-3,cx+4) for y in range(cy-3,cy+4)]
assert not any(c.blue()>170 and c.red()>80 and c.green()<190 for c in colors), [c.name() for c in colors]
print("gap-ok")
'''
        env = os.environ.copy()
        env['QT_QPA_PLATFORM'] = 'offscreen'
        env['PATH'] = str(CRAFT / 'bin') + os.pathsep + env.get('PATH', '')
        env['QT_PLUGIN_PATH'] = str(CRAFT / 'plugins')
        env['QML2_IMPORT_PATH'] = str(CRAFT / 'qml')
        result = subprocess.run([str(PYTHON), '-I', '-c', code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('gap-ok', result.stdout)
