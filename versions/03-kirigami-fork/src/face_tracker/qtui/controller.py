"""Qt/QML adapter for the existing local face-analysis and session services.

The controller deliberately owns no camera device. ``AnalysisWorker`` keeps
its capture/analyzer threads and generation fencing; this object only consumes
its bounded message queue on the Qt GUI thread.

Porting status: camera/calibration/live score/session/basic history/rating,
manual labeled-frame capture, local model comparison, and validation-gated
personal emotion adaptation are exposed. The legacy UI's blind-rating flow,
per-session episode/rating detail, privacy dialog, AU/pattern panels, and
sample review/relabeling are not yet wired to QML. The face-only index and
technical image quality remain separate; the index is not a clinical stress
measure.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import QObject, Property, QSettings, QTimer, Signal, Slot, QSize
from PySide6.QtGui import QImage
from PySide6.QtQuick import QQuickImageProvider

from ..camera import discover_camera_indices
from ..events import EventTracker, TrackerState
from ..labels import LABELS, LabeledSampleStore, observation_is_fresh
from ..personalization import default_adapter_path, load_adapter
from ..scoring import CalibrationProfile, CalibrationSampler, Scorer
from ..session import TrackingSession
from ..storage import JournalStore
from ..temporal import ExpressionSmoother
from ..emotion_evaluation import top_two_summary
from ..worker import AnalysisWorker, WorkerMessage
from ..quality_states import QualityStatusDebouncer
from ..vision import EMOTION_MODEL_VERSION


EXPRESSION_NAMES = {
    "Anger": "Злость", "Contempt": "Презрение", "Disgust": "Отвращение",
    "Fear": "Страх", "Happiness": "Радость", "Neutral": "Нейтральное",
    "Sadness": "Грусть", "Surprise": "Удивление",
}
LIVE_MAX_AGE_SECONDS = 1.8
LANDMARK_MAX_AGE_SECONDS = 0.20


class _CameraScanSignals(QObject):
    """Передаёт результат фонового поиска устройств в главный поток Qt."""

    finished = Signal(object, str)


class FaceImageProvider(QQuickImageProvider):
    """Потокобезопасный последний кадр для URL ``image://camera/latest``."""

    def __init__(self) -> None:
        super().__init__(QQuickImageProvider.Image)
        self._lock = threading.Lock()
        self._image = QImage()

    def set_frame(self, frame: np.ndarray | None) -> None:
        if frame is None:
            with self._lock:
                self._image = QImage()
            return
        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            return
        try:
            # Скругляем сам видеослой, а не только внешнюю карточку QML:
            # иначе прямоугольные углы изображения остаются видны внутри неё.
            rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            height, width, channels = rgba.shape
            radius = max(2, min(width, height) // 28)
            alpha = np.zeros((height, width), dtype=np.uint8)
            cv2.rectangle(alpha, (radius, 0), (width - radius - 1, height - 1), 255, -1, cv2.LINE_AA)
            cv2.rectangle(alpha, (0, radius), (width - 1, height - radius - 1), 255, -1, cv2.LINE_AA)
            for center in ((radius, radius), (width - radius - 1, radius),
                           (radius, height - radius - 1), (width - radius - 1, height - radius - 1)):
                cv2.circle(alpha, center, radius, 255, -1, cv2.LINE_AA)
            rgba[:, :, 3] = alpha
            # copy() отделяет QImage от буфера NumPy, который может быть
            # освобождён или переиспользован после обработки сообщения.
            image = QImage(rgba.data, width, height, channels * width,
                           QImage.Format.Format_RGBA8888).copy()
        except (cv2.error, ValueError, TypeError):
            return
        with self._lock:
            self._image = image

    def requestImage(self, image_id: str, size: QSize, requested_size: QSize) -> QImage:
        del image_id
        with self._lock:
            image = self._image.copy()
        if not image.isNull() and requested_size.isValid():
            image = image.scaled(requested_size)
        if size is not None:
            size.setWidth(image.width())
            size.setHeight(image.height())
        # QQuickImageProvider в PySide6 требует вернуть сам QImage. Размер
        # исходника передаётся через аргумент size; кортеж здесь ломает Image.
        return image


class TrackerController(QObject):
    """QML-facing properties and slots around existing tracking domain code."""

    cameraStateChanged = Signal()
    cameraHintChanged = Signal()
    stateTitleChanged = Signal()
    stateDescriptionChanged = Signal()
    hasScoreChanged = Signal()
    scoreValueChanged = Signal()
    confidenceTextChanged = Signal()
    qualityTextChanged = Signal()
    expressionTextChanged = Signal()
    observationTextChanged = Signal()
    frameRevisionChanged = Signal()
    scoreHistoryChanged = Signal()
    landmarkPointsChanged = Signal()
    historyEntriesChanged = Signal()
    selectedTabChanged = Signal()
    cameraDevicesChanged = Signal()
    cameraIndexChanged = Signal()
    cameraScanStateChanged = Signal()
    emotionSampleCountsChanged = Signal()
    emotionStatusChanged = Signal()
    emotionModelStatusChanged = Signal()
    emotionTaskRunningChanged = Signal()
    errorOccurred = Signal(str)

    def __init__(self, store: JournalStore | None = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.store = store or JournalStore()
        self.sample_store = LabeledSampleStore()
        self._project_root = Path(__file__).resolve().parents[3]
        self.report_directory = self.sample_store.root.parent
        self.benchmark_report_path = self.report_directory / "benchmark-report.json"
        self.personalization_report_path = self.report_directory / "personalization-report.json"
        self.adapter_path = default_adapter_path()
        self.personal_adapter = load_adapter(
            self.adapter_path, expected_model="enet_b0_8_best_vgaf", expected_preprocessing="tight")
        self._emotion_sample_counts = self._sample_counts_text()
        self._emotion_status = "Примеры хранятся локально; камера не записывается автоматически."
        self._emotion_task_running = False
        self._emotion_task_kind: str | None = None
        self._emotion_task_messages: queue.Queue[tuple[str, int, str, str]] = queue.Queue()
        self._emotion_task_cancel = threading.Event()
        self._emotion_process: subprocess.Popen | None = None
        self._settings = QSettings("Kontur", "Kontur")
        try:
            stored_camera_index = int(self._settings.value("camera/index", 0))
        except (TypeError, ValueError):
            stored_camera_index = 0
        self._selected_camera_index = min(31, max(0, stored_camera_index))
        self._camera_devices: list[dict[str, Any]] = [
            self._camera_item(self._selected_camera_index)
        ]
        self._camera_scan_state = "idle"
        self._camera_scan_running = False
        self._camera_scan_cancel = threading.Event()
        self._camera_scan_signals = _CameraScanSignals(self)
        self._camera_scan_signals.finished.connect(self._finish_camera_scan)
        self.image_provider = FaceImageProvider()
        self.messages: queue.Queue[WorkerMessage] = queue.Queue(maxsize=2)
        self.worker: AnalysisWorker | None = None
        self.worker_generation = 0
        self.profile: CalibrationProfile | None = None
        self.calibration_sampler = CalibrationSampler()
        self.calibration_observations = []
        self.calibrating = False
        self.tracking = False
        self.scorer: Scorer | None = None
        self.event_tracker = EventTracker()
        self.session: TrackingSession | None = None
        self.expression_smoother = ExpressionSmoother()
        self.latest_observation = None
        self.latest_score = None
        self.latest_frame = None
        self.latest_analyzed_frame = None
        self._state = "stopped"
        self._hint = "Создай личную базу, чтобы начать."
        self._title = "Ожидание лица"
        self._description = "Сначала откалибруй личную базу"
        self._has_score = False
        self._score = 0.0
        self._confidence = "Модель не сообщает точность оценки состояния"
        self._quality = "Ожидаем пригодный кадр"
        self._expression = "Недостаточно данных"
        self._observation = "Создай личную базу, чтобы начать."
        self._frame_revision = 0
        # Храним около пяти минут истории при редкой выборке 2 Гц, а не
        # сотни почти одинаковых точек на каждый кадр модели.
        self._score_history: deque[dict[str, Any]] = deque(maxlen=620)
        self._last_score_sample_at = 0.0
        self._landmarks: list[dict[str, float]] = []
        self._history_entries: list[dict[str, Any]] = []
        self._selected_tab = "Главная"
        self._last_preview_monotonic = 0.0
        self._landmark_monotonic = 0.0
        self._last_tense = False
        self._quality_status = QualityStatusDebouncer()
        self._closed = False
        self._timer = QTimer(self)
        self._timer.setInterval(80)
        self._timer.timeout.connect(self._poll_messages)
        self._timer.start()
        self.refreshHistory()

    cameraState = Property(str, lambda self: self._state, notify=cameraStateChanged)
    cameraHint = Property(str, lambda self: self._hint, notify=cameraHintChanged)
    stateTitle = Property(str, lambda self: self._title, notify=stateTitleChanged)
    stateDescription = Property(str, lambda self: self._description, notify=stateDescriptionChanged)
    hasScore = Property(bool, lambda self: self._has_score, notify=hasScoreChanged)
    scoreValue = Property(float, lambda self: self._score, notify=scoreValueChanged)
    confidenceText = Property(str, lambda self: self._confidence, notify=confidenceTextChanged)
    qualityText = Property(str, lambda self: self._quality, notify=qualityTextChanged)
    expressionText = Property(str, lambda self: self._expression, notify=expressionTextChanged)
    observationText = Property(str, lambda self: self._observation, notify=observationTextChanged)
    frameRevision = Property(int, lambda self: self._frame_revision, notify=frameRevisionChanged)
    scoreHistory = Property("QVariantList", lambda self: list(self._score_history), notify=scoreHistoryChanged)
    landmarkPoints = Property("QVariantList", lambda self: list(self._landmarks), notify=landmarkPointsChanged)
    historyEntries = Property("QVariantList", lambda self: list(self._history_entries), notify=historyEntriesChanged)
    selectedTab = Property(str, lambda self: self._selected_tab, notify=selectedTabChanged)
    cameraDevices = Property("QVariantList", lambda self: list(self._camera_devices),
                             notify=cameraDevicesChanged)
    selectedCameraIndex = Property(int, lambda self: self._selected_camera_index,
                                   notify=cameraIndexChanged)
    cameraScanState = Property(str, lambda self: self._camera_scan_state,
                               notify=cameraScanStateChanged)
    emotionLabels = Property("QVariantList", lambda self: [
        {"value": label, "title": EXPRESSION_NAMES.get(label, label)} for label in LABELS
    ])
    emotionSampleCounts = Property(str, lambda self: self._emotion_sample_counts,
                                   notify=emotionSampleCountsChanged)
    emotionStatus = Property(str, lambda self: self._emotion_status, notify=emotionStatusChanged)
    emotionModelStatus = Property(str, lambda self: self._emotion_model_status_text(),
                                  notify=emotionModelStatusChanged)
    emotionTaskRunning = Property(bool, lambda self: self._emotion_task_running,
                                  notify=emotionTaskRunningChanged)

    def _sample_counts_text(self) -> str:
        counts = self.sample_store.counts()
        nonzero = [f"{EXPRESSION_NAMES.get(label, label)}: {count}"
                   for label, count in counts.items() if count]
        return " · ".join(nonzero) if nonzero else "Пока нет сохранённых примеров."

    def _emotion_model_status_text(self) -> str:
        if self.personal_adapter is None:
            return "Общая модель HSEmotion · 8 выражений"
        return (f"Персональная поправка проверена · macro-F1 "
                f"{self.personal_adapter.baseline_macro_f1:.2f} → "
                f"{self.personal_adapter.adapted_macro_f1:.2f}")

    def _set_emotion_status(self, value: str) -> None:
        if self._emotion_status != value:
            self._emotion_status = value
            self.emotionStatusChanged.emit()

    def _refresh_emotion_counts(self) -> None:
        value = self._sample_counts_text()
        if value != self._emotion_sample_counts:
            self._emotion_sample_counts = value
            self.emotionSampleCountsChanged.emit()

    def _invalidate_personal_emotion_model(self) -> None:
        # Новая или исправленная метка делает прежнюю оценку и адаптер устаревшими.
        for path in (self.adapter_path, self.benchmark_report_path,
                     self.benchmark_report_path.with_suffix(".csv"),
                     self.personalization_report_path):
            path.unlink(missing_ok=True)
        self.personal_adapter = None
        self.emotionModelStatusChanged.emit()

    @Slot(str)
    def saveEmotionSample(self, label: str) -> None:
        """Сохраняет кадр только по явному нажатию и указанной метке пользователя."""
        observation = self.latest_observation
        frame = self.latest_analyzed_frame
        if (label not in LABELS or not self.tracking or self.session is None or
                frame is None or observation is None or not observation.present or
                observation.quality < 0.45 or not observation.face_box or
                not observation_is_fresh(observation.timestamp, time.time(),
                                         monotonic_timestamp=observation.monotonic_timestamp,
                                         monotonic_now=time.monotonic())):
            self._set_emotion_status("Нужен свежий кадр лица хорошего качества во время активной сессии.")
            return
        try:
            sample = self.sample_store.save_crop(
                frame, observation.face_box, label, observation.quality,
                EMOTION_MODEL_VERSION, "mediapipe-mesh-bbox / BGR-JPEG92",
                self.session.capture_id or "unknown-session")
        except (ValueError, OSError, RuntimeError) as exc:
            self._set_emotion_status(f"Не удалось сохранить пример: {exc}")
            return
        self._invalidate_personal_emotion_model()
        self._refresh_emotion_counts()
        self._set_emotion_status(
            f"Сохранён пример «{EXPRESSION_NAMES.get(label, label)}» · качество {sample.quality:.0%}. "
            "Сравни примеры после остановки камеры.")

    def _launch_emotion_tool(self, task: str, script: str, arguments: list[str], status: str) -> None:
        if self._emotion_task_running:
            return
        if self.tracking or self.calibrating or (self.worker and self.worker.running):
            self._set_emotion_status("Останови камеру перед сравнением или обучением.")
            return
        self._emotion_task_running = True
        self._emotion_task_kind = task
        self._emotion_task_cancel.clear()
        self.emotionTaskRunningChanged.emit()
        self._set_emotion_status(status)
        script_path = self._project_root / "tools" / script

        def run() -> None:
            process = None
            try:
                environment = os.environ.copy()
                source_path = str(self._project_root / "src")
                environment["PYTHONPATH"] = source_path + (
                    os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
                environment["PYTHONUTF8"] = "1"
                process = subprocess.Popen(
                    [sys.executable, str(script_path), *arguments], cwd=self._project_root,
                    env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, encoding="utf-8", errors="replace")
                self._emotion_process = process
                while True:
                    try:
                        stdout, stderr = process.communicate(timeout=0.25)
                        break
                    except subprocess.TimeoutExpired:
                        if self._emotion_task_cancel.is_set():
                            process.terminate()
                            try:
                                stdout, stderr = process.communicate(timeout=2)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                stdout, stderr = process.communicate(timeout=2)
                            break
                self._emotion_task_messages.put((task, process.returncode or 0, stdout, stderr))
            except Exception as exc:
                self._emotion_task_messages.put((task, 1, "", str(exc)))
            finally:
                self._emotion_process = None

        threading.Thread(target=run, name=f"kontur-{task}", daemon=True).start()

    @Slot()
    def compareEmotionModels(self) -> None:
        if not self.sample_store.list_samples():
            self._set_emotion_status("Сначала сохрани несколько примеров с выбранными выражениями.")
            return
        self._launch_emotion_tool(
            "benchmark", "benchmark_models.py",
            ["--samples", str(self.sample_store.root), "--output", str(self.benchmark_report_path),
             "--models", "enet_b0_8_best_vgaf"],
            "Сравниваю модель с твоими примерами локально…")

    @Slot()
    def evaluateEmotionAdaptation(self) -> None:
        if not self.benchmark_report_path.is_file():
            self._set_emotion_status("Сначала запусти сравнение на своих размеченных примерах.")
            return
        self._launch_emotion_tool(
            "personalization", "train_personal_adapter.py",
            [str(self.benchmark_report_path), "--report", str(self.personalization_report_path),
             "--adapter", str(self.adapter_path)],
            "Проверяю поправку на отдельных сессиях…")

    @Slot()
    def resetEmotionAdaptation(self) -> None:
        self.adapter_path.unlink(missing_ok=True)
        self.personal_adapter = None
        self.expression_smoother.reset()
        self.emotionModelStatusChanged.emit()
        self._set_emotion_status("Персональную поправку сброшено; используется общая модель.")

    @Slot()
    def deleteEmotionSamples(self) -> None:
        if self.tracking or self.calibrating or self._emotion_task_running:
            self._set_emotion_status("Останови камеру и дождись локальной обработки перед удалением.")
            return
        count = self.sample_store.delete_all()
        self._invalidate_personal_emotion_model()
        self._refresh_emotion_counts()
        self._set_emotion_status(f"Локальные примеры удалены: {count}.")

    def _drain_emotion_tasks(self) -> None:
        while True:
            try:
                task, return_code, stdout, stderr = self._emotion_task_messages.get_nowait()
            except queue.Empty:
                return
            self._emotion_task_running = False
            self._emotion_task_kind = None
            self.emotionTaskRunningChanged.emit()
            if self._closed:
                continue
            if return_code != 0:
                detail = (stderr or stdout or "Локальная обработка завершилась с ошибкой.")[-800:]
                self._set_emotion_status(f"Ошибка: {detail}")
                continue
            if task == "benchmark":
                try:
                    report = json.loads(self.benchmark_report_path.read_text(encoding="utf-8"))
                    rows = report.get("comparisons", [])
                    best = max(rows, key=lambda row: row.get("macro_f1", 0)) if rows else None
                    if best:
                        sadness = best.get("per_class", {}).get("Sadness", {})
                        self._set_emotion_status(
                            f"Проверка на твоих примерах · macro-F1 {best['macro_f1']:.2f} · "
                            f"грусть: recall {sadness.get('recall', 0):.0%}. "
                            "Это личная предварительная проверка, не независимая точность модели.")
                    else:
                        self._set_emotion_status("Отчёт готов, но пока недостаточно примеров для сравнения.")
                except (OSError, ValueError, TypeError, KeyError) as exc:
                    self._set_emotion_status(f"Не удалось прочитать отчёт сравнения: {exc}")
            elif task == "personalization":
                try:
                    result = json.loads(self.personalization_report_path.read_text(encoding="utf-8"))
                    if result.get("activated"):
                        self.personal_adapter = load_adapter(
                            self.adapter_path, expected_model="enet_b0_8_best_vgaf",
                            expected_preprocessing="tight")
                    self.emotionModelStatusChanged.emit()
                    self._set_emotion_status(result.get("reason", "Проверка адаптации завершена."))
                except (OSError, ValueError, TypeError) as exc:
                    self._set_emotion_status(f"Не удалось прочитать отчёт адаптации: {exc}")

    @staticmethod
    def _camera_item(index: int) -> dict[str, Any]:
        return {"index": index, "label": f"Камера {index + 1}"}

    def _set_camera_scan_state(self, value: str) -> None:
        if self._camera_scan_state != value:
            self._camera_scan_state = value
            self.cameraScanStateChanged.emit()

    @Slot()
    def refreshCameras(self) -> None:
        """Проверяет первые восемь индексов, не блокируя интерфейс Qt."""
        if self._closed:
            return
        if self.worker and self.worker.running:
            self.errorOccurred.emit("Останови обработку, чтобы искать камеры.")
            return
        if self._camera_scan_running:
            return
        self._camera_scan_running = True
        self._camera_scan_cancel.clear()
        self._set_camera_scan_state("scanning")

        def scan() -> None:
            try:
                indices = discover_camera_indices(cancel_event=self._camera_scan_cancel)
                error = ""
            except Exception as exc:  # драйверы иногда возвращают нестандартные ошибки
                indices, error = [], str(exc)
            try:
                self._camera_scan_signals.finished.emit(indices, error)
            except RuntimeError:
                # Qt мог уже уничтожить контроллер, если драйвер завис до выхода.
                pass

        # Некоторые драйверы зависают на открытии отсутствующего индекса.
        # Daemon гарантирует, что такой драйвер не удержит закрытие приложения.
        threading.Thread(target=scan, name="camera-discovery", daemon=True).start()

    @Slot(int)
    def selectCamera(self, index: int) -> None:
        """Запоминает выбранную камеру; она применяется при следующем запуске."""
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index <= 31:
            return
        if self.worker and self.worker.running:
            self.errorOccurred.emit("Останови обработку перед сменой камеры.")
            return
        if index not in {item["index"] for item in self._camera_devices}:
            return
        if index == self._selected_camera_index:
            return
        self._selected_camera_index = index
        self._settings.setValue("camera/index", index)
        self.cameraIndexChanged.emit()

    @Slot(object, str)
    def _finish_camera_scan(self, indices: object, error: str) -> None:
        self._camera_scan_running = False
        if self._closed:
            return
        try:
            valid_indices = sorted({int(index) for index in indices if 0 <= int(index) <= 31})
        except (TypeError, ValueError):
            valid_indices = []
        self._camera_devices = [self._camera_item(index) for index in valid_indices]
        self.cameraDevicesChanged.emit()
        if error:
            self._set_camera_scan_state("error")
            self.errorOccurred.emit(f"Не удалось проверить камеры: {error}")
        elif valid_indices:
            self._set_camera_scan_state("ready")
        else:
            self._set_camera_scan_state("empty")

    def _selected_camera_was_not_found(self) -> bool:
        return (self._camera_scan_state in {"ready", "empty"} and
                self._selected_camera_index not in {item["index"] for item in self._camera_devices})

    def _set(self, key: str, value: Any) -> None:
        attr, signal = {
            "state": ("_state", self.cameraStateChanged),
            "hint": ("_hint", self.cameraHintChanged),
            "title": ("_title", self.stateTitleChanged),
            "description": ("_description", self.stateDescriptionChanged),
            "has_score": ("_has_score", self.hasScoreChanged),
            "score": ("_score", self.scoreValueChanged),
            "confidence": ("_confidence", self.confidenceTextChanged),
            "quality": ("_quality", self.qualityTextChanged),
            "expression": ("_expression", self.expressionTextChanged),
            "observation": ("_observation", self.observationTextChanged),
        }[key]
        if getattr(self, attr) != value:
            setattr(self, attr, value)
            signal.emit()

    def _clear_score(self, reason: str) -> None:
        self.latest_score = None
        self._set("has_score", False)
        self._set("confidence", "Модель не сообщает точность оценки состояния")
        self._set("quality", reason)

    @Slot()
    def createBaseline(self) -> None:
        self.calibrate()

    @Slot()
    def calibrate(self) -> None:
        if self._camera_scan_running:
            self._set("hint", "Подожди, пока закончится поиск камер.")
            return
        if self._selected_camera_was_not_found():
            self._set("hint", "Выбери найденную камеру или обнови список устройств.")
            return
        if self.tracking or (self.worker and self.worker.running):
            self._set("hint", "Сначала дождись остановки текущей обработки.")
            return
        self.worker_generation += 1
        self.calibration_observations.clear()
        self.calibration_sampler.start(time.monotonic())
        self.calibrating = True
        self._clear_score("Собираем личную базу…")
        self._set("state", "starting")
        self._set("title", "Калибровка личной базы")
        self._set("description", "Смотри прямо при обычном выражении и хорошем освещении.")
        self._set("hint", "Сбор занимает не менее пяти секунд и требует устойчивого лица.")
        self._set("observation", "Собираю личную базу мимики.")
        self._start_worker()

    @Slot()
    def startSession(self) -> None:
        if self._camera_scan_running:
            self._set("hint", "Подожди, пока закончится поиск камер.")
            return
        if self._selected_camera_was_not_found():
            self._set("hint", "Выбери найденную камеру или обнови список устройств.")
            return
        if self.profile is None:
            self._set("hint", "Сначала создай личную базу.")
            return
        if self.worker and self.worker.running:
            self._set("hint", "Дождись остановки камеры и попробуй снова.")
            return
        self.scorer = Scorer(self.profile)
        self.event_tracker = EventTracker()
        self.expression_smoother.reset()
        self.session = TrackingSession(self.store, self.event_tracker)
        self.session.start(datetime.now().timestamp())
        self.tracking = True
        self._last_tense = False
        self._score_history.clear()
        self._last_score_sample_at = 0.0
        self.scoreHistoryChanged.emit()
        self._clear_score("Ожидаем пригодный кадр")
        self._set("state", "starting")
        self._set("title", "Подключение камеры")
        self._set("description", "Индекс появится после получения свежего качественного кадра.")
        self._set("observation", "Сигнал меняется плавно.")
        self._set("hint", "Индекс описывает изменение мимики, а не измеряет стресс.")
        self.refreshHistory()
        self._start_worker()

    @Slot()
    def pauseSession(self) -> None:
        self._end_session(interrupted=True, title="Сессия прервана",
                          description="Можно начать новую сессию.")

    @Slot()
    def finishSession(self) -> None:
        self._end_session(interrupted=False, title="Сессия завершена",
                          description="Сводка сессии сохранена локально.")

    def _end_session(self, *, interrupted: bool, title: str, description: str) -> None:
        if not self.tracking:
            return
        self.tracking = False
        if self.session:
            self.session.finish(datetime.now().timestamp(), interrupted=interrupted)
            self.session = None
        self.worker_generation += 1
        self._stop_worker()
        self._clear_score("Нет активного кадра")
        self._set("state", "stopped")
        self._set("title", title)
        self._set("description", description)
        self._set("observation", description)
        self._set("hint", "Данные хранятся локально на этом компьютере.")
        self._landmarks = []
        self.landmarkPointsChanged.emit()
        self.refreshHistory()

    @Slot(str)
    def selectTab(self, name: str) -> None:
        if name in {"Главная", "История", "Подробно"} and name != self._selected_tab:
            self._selected_tab = name
            self.selectedTabChanged.emit()
            if name == "История":
                self.refreshHistory()

    @Slot()
    def refreshHistory(self) -> None:
        entries = []
        for item in self.store.list_sessions(100):
            duration = max(0.0, (item.ended_at or datetime.now().timestamp()) - item.started_at)
            started = datetime.fromtimestamp(item.started_at).strftime("%d.%m.%Y · %H:%M")
            state = ("Прервана" if item.interrupted else
                     "Идёт" if item.ended_at is None else "Завершена")
            entries.append(f"{started} · {duration / 60:.1f} мин · "
                           f"{item.episode_count} эпизодов · {item.rating_count} отметок · {state}")
        self._history_entries = entries
        self.historyEntriesChanged.emit()

    @Slot(int)
    def saveRating(self, value: int) -> None:
        observation = self.latest_observation
        score = self.latest_score
        captured_monotonic = (getattr(observation, "monotonic_timestamp", None)
                              if observation is not None else None)
        try:
            fresh = (captured_monotonic is not None and
                     0.0 <= time.monotonic() - float(captured_monotonic) <= LIVE_MAX_AGE_SECONDS)
        except (TypeError, ValueError, OverflowError):
            fresh = False
        if (not self.tracking or self.session is None or observation is None or score is None or
                not score.usable or score.index is None or not fresh):
            self.errorOccurred.emit("Дождись свежего пригодного сигнала перед сохранением отметки.")
            return
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 10:
            self.errorOccurred.emit("Оценка должна быть целым числом от 0 до 10.")
            return
        try:
            self.store.record_self_rating(
                self.session.session_id, datetime.now().timestamp(), observation.timestamp,
                int(value), float(score.index), float(observation.quality), "visible")
        except (ValueError, TypeError) as exc:
            self.errorOccurred.emit(str(exc))
            return
        self.refreshHistory()

    def _start_worker(self) -> None:
        if self.worker is not None and self.worker.running:
            return
        self.worker_generation += 1
        self.worker = AnalysisWorker(self.messages, camera_index=self._selected_camera_index,
                                     generation=self.worker_generation)
        self.worker.start()
        self._set("state", "starting")
        self._set("hint", "Открываем камеру и загружаем локальные модели…")

    def _stop_worker(self) -> None:
        if self.worker is not None:
            # timeout=0 задаёт остановку, не блокируя цикл событий Qt.
            self.worker.stop(timeout=0.0)

    @Slot()
    def _poll_messages(self) -> None:
        if self._closed:
            return
        self._drain_emotion_tasks()
        while True:
            try:
                message = self.messages.get_nowait()
            except queue.Empty:
                break
            self._handle_message(message)
        self._refresh_latest_preview()
        if self.worker and not self.worker.running:
            self.worker = None
            if self.calibrating:
                self._start_worker()
        if self.latest_observation is not None:
            captured = self.latest_observation.monotonic_timestamp or 0.0
            if captured and time.monotonic() - captured > LIVE_MAX_AGE_SECONDS:
                self._expire_stale_frame()

    def _refresh_latest_preview(self) -> None:
        """Refresh at QTimer cadence even if model inference is slower than video."""
        worker = self.worker
        if worker is None:
            return
        preview = worker.latest_preview
        if preview is None:
            return
        captured = preview.captured_monotonic
        if captured is None:
            captured = preview.captured_at
        try:
            captured = float(captured)
        except (TypeError, ValueError, OverflowError):
            return
        if not math.isfinite(captured) or captured <= self._last_preview_monotonic:
            return
        self._last_preview_monotonic = captured
        self.latest_frame = preview.frame
        self.image_provider.set_frame(preview.frame)
        self._frame_revision += 1
        self.frameRevisionChanged.emit()
        # Ключевые точки относятся к кадру модели, а не к более свежему кадру
        # видеопотока. Скрываем их, если разница во времени заметна.
        if self._landmarks and abs(captured - self._landmark_monotonic) > LANDMARK_MAX_AGE_SECONDS:
            self._landmarks = []
            self.landmarkPointsChanged.emit()

    def _handle_message(self, message: WorkerMessage) -> None:
        if message.generation != self.worker_generation:
            return
        if message.kind == "camera_ready":
            # QML camera panel renders the live stream in this state; model
            # initialization may still be in progress in the analysis thread.
            self._set("state", "running")
            self._set("hint", "Камера открыта; подготавливаем анализатор.")
            return
        if message.kind == "ready":
            self._set("state", "active")
            self._set("hint", "Камера работает локально.")
            return
        if message.kind == "error":
            self.calibrating = False
            self.tracking = False
            if self.session:
                self.session.finish(datetime.now().timestamp(), interrupted=True)
                self.session = None
            self._clear_score("Нет активного кадра")
            self._set("state", "error")
            self._set("title", "Ошибка камеры или модели")
            self._set("description", message.error or "Не удалось запустить обработку.")
            self._set("hint", "Проверь подключение камеры и попробуй ещё раз.")
            self.errorOccurred.emit(message.error or "Не удалось запустить обработку.")
            self.refreshHistory()
            return
        if message.kind != "frame" or message.frame is None or message.observation is None:
            return
        observation = message.observation
        self.latest_observation = observation
        self.latest_analyzed_frame = message.frame
        if (self._last_preview_monotonic and observation.monotonic_timestamp is not None and
                abs(self._last_preview_monotonic - observation.monotonic_timestamp) <= LANDMARK_MAX_AGE_SECONDS):
            self._landmark_monotonic = observation.monotonic_timestamp
            self._landmarks = [{"x": x, "y": y} for x, y in observation.landmarks]
        else:
            self._landmarks = []
        self.landmarkPointsChanged.emit()
        quality = float(observation.quality) if observation.present else 0.0
        captured_at = observation.monotonic_timestamp
        frame_fresh = (captured_at is not None and math.isfinite(float(captured_at))
                       and 0.0 <= time.monotonic() - float(captured_at) <= LIVE_MAX_AGE_SECONDS)
        user_reason = observation.quality_reason or ("Низкое качество кадра" if quality < 0.45 else None)
        debounced_status = self._quality_status.update(
            time.monotonic(), observation.present, frame_fresh, user_reason)
        if quality >= 0.75:
            quality_text = "Высокое"
        elif quality >= 0.45:
            quality_text = "Среднее"
        else:
            quality_text = debounced_status
        self._set("quality", quality_text)
        self._set("confidence", "Уверенность модели: не измеряется")
        self._set("expression", self._stable_expression(observation))

        timestamp = observation.monotonic_timestamp or time.monotonic()
        if self.calibrating:
            if observation.present and quality >= 0.45 and observation.features:
                accepted = self.calibration_sampler.accept(timestamp)
                if self.calibration_sampler.continuity_broken:
                    self.calibration_observations.clear()
                if accepted:
                    self.calibration_observations.append(observation)
                count = self.calibration_sampler.sample_count
                self._set("title", "Калибровка личной базы")
                self._set("description", f"Набрано пригодных кадров: {count}/50")
                self._set("observation", "Оставайся перед камерой и сохраняй обычное выражение.")
                if self.calibration_sampler.ready(timestamp, 50):
                    self._finish_calibration()
            else:
                self.calibration_observations.clear()
                self.calibration_sampler.start(timestamp)
                self._set("description", quality_text if quality_text != "Камера работает локально"
                           else "Поставь лицо по центру камеры.")
            return

        if not self.tracking or self.scorer is None or self.session is None:
            return
        score = self.scorer.update(observation, timestamp)
        self.latest_score = score
        self.session.update(score, timestamp, observation.timestamp)
        if not score.usable or score.index is None:
            self._set("has_score", False)
            # Закрытие глаз уже прошло свой 0.4-секундный фильтр в анализаторе;
            # повторно задерживать это сообщение не нужно.
            needs_debounce = not observation.present or not frame_fresh
            score_message = (quality_text if needs_debounce else
                             score.reason or quality_text)
            self._set("quality", score_message)
            status_confirmed = score_message != "Камера работает локально"
            self._set("title", "Недостаточно данных" if status_confirmed else "Обновляем сигнал")
            self._set("description", score_message if status_confirmed
                       else "Короткая пауза — проверяем следующий кадр.")
            self._set("observation", "Оценка приостановлена." if status_confirmed
                       else "Ждём подтверждения, что лицо действительно пропало.")
            self._append_score(None, observation.timestamp)
            return

        value = float(score.index)
        self._set("has_score", True)
        self._set("score", value)
        tense = self.event_tracker.state == TrackerState.TENSION
        self._set("title", "Устойчивое изменение лица" if tense else "Близко к личной базе")
        self._set("description", "Сигнал сравнивается с твоей личной мимической базой.")
        self._set("observation", "Изменение лица сохраняется." if tense else "Сигнал меняется плавно.")
        self._append_score(value, observation.timestamp)
        if tense != self._last_tense:
            self._last_tense = tense
            self.refreshHistory()

    def _stable_expression(self, observation) -> str:
        probabilities = observation.expressions
        if self.personal_adapter is not None and probabilities:
            try:
                probabilities = self.personal_adapter.apply(probabilities)
            except (ValueError, TypeError, KeyError):
                probabilities = observation.expressions
        timestamp = observation.monotonic_timestamp or time.monotonic()
        state = self.expression_smoother.update(timestamp, observation.quality,
                                                probabilities)
        if state.ready and state.label:
            return EXPRESSION_NAMES.get(state.label, state.label)
        if state.probabilities:
            top, second, ambiguous = top_two_summary(state.probabilities)
            if ambiguous:
                return f"Неясно: {EXPRESSION_NAMES.get(top, top)} / {EXPRESSION_NAMES.get(second, second)}"
        return state.reason or "Недостаточно данных"

    def _append_score(self, value: float | None, timestamp: float) -> None:
        timestamp = float(timestamp)
        last_value = self._score_history[-1]["value"] if self._score_history else None
        if value is None:
            # Единственная отметка разрыва сохраняет честную паузу на графике,
            # но не раздувает историю повторяющимися пустыми кадрами.
            if last_value is None:
                return
        elif timestamp - self._last_score_sample_at < 0.48:
            return
        self._score_history.append({"timestamp": timestamp, "value": value})
        self._last_score_sample_at = timestamp
        self.scoreHistoryChanged.emit()

    def _finish_calibration(self) -> None:
        self.calibrating = False
        try:
            # Личная база строится только из принятых по качеству кадров.
            self.profile = CalibrationProfile.from_observations(self.calibration_observations)
        except ValueError as exc:
            self._set("title", "Базу не удалось создать")
            self._set("description", str(exc))
            self._set("hint", "Проверь качество кадра и попробуй снова.")
            self._set("state", "stopped")
            self._stop_worker()
            return
        qualities = [float(item.quality) for item in self.calibration_observations]
        first = self.calibration_sampler.first_sample_at
        last = self.calibration_sampler.last_sample_at
        span = max(0.0, last - first) if first is not None and last is not None else 0.0
        self._set("state", "stopped")
        self._set("title", "Личная база готова")
        self._set("description", f"{len(qualities)} кадров · {span:.1f} с · качество {np.median(qualities):.0%}")
        self._set("hint", "Это личный ориентир мимики и условий съёмки, не норма стресса.")
        self._set("observation", "Теперь можно начать сессию.")
        self.worker_generation += 1
        self._stop_worker()

    def _expire_stale_frame(self) -> None:
        if self.tracking and self.session is not None:
            self.session.interrupt_observation_gap()
        if self.scorer:
            self.scorer.reset()
        self.expression_smoother.reset()
        self.latest_frame = None
        self.latest_observation = None
        self.image_provider.set_frame(None)
        self._landmarks = []
        self.landmarkPointsChanged.emit()
        self._frame_revision += 1
        self.frameRevisionChanged.emit()
        self._clear_score("Нет свежего кадра")
        self._set("title", "Ожидание свежего кадра")
        self._set("description", "Ключевые точки и оценка временно скрыты.")
        self._set("expression", "Нет свежего предсказания")

    @Slot()
    def close(self) -> None:
        self._closed = True
        self._camera_scan_cancel.set()
        self._emotion_task_cancel.set()
        if self._emotion_process is not None and self._emotion_process.poll() is None:
            try:
                self._emotion_process.terminate()
            except OSError:
                pass
        self._timer.stop()
        self.worker_generation += 1
        if self.session:
            self.session.finish(datetime.now().timestamp(), interrupted=True)
            self.session = None
        if self.worker:
            self.worker.stop(timeout=3.0)
            self.worker = None
        self.sample_store.close()
        self.store.close()

