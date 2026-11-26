"""Qt/QML adapter for the existing local face-analysis and session services.

The controller deliberately owns no camera device. ``AnalysisWorker`` keeps
its capture/analyzer threads and generation fencing; this object only consumes
its bounded message queue on the Qt GUI thread.

Porting status: camera/calibration/live score/session/basic history/rating are
exposed. The legacy UI's blind-rating flow, per-session episode/rating detail,
privacy dialog, action-unit and facial-pattern panels, labeled-frame capture
and review, model comparison/benchmark, and personalization training are not
yet wired to QML; add explicit Qt properties/slots for these before claiming
full feature parity. The face-only index and technical image quality remain
separate; this adapter never labels the index as a clinical stress measure.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime
import math
import queue
import threading
import time
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import QObject, Property, QTimer, Signal, Slot, QSize
from PySide6.QtGui import QImage
from PySide6.QtQuick import QQuickImageProvider

from ..camera import discover_cameras
from ..events import EventTracker, TrackerState
from ..scoring import CalibrationProfile, CalibrationSampler, Scorer
from ..session import TrackingSession
from ..storage import JournalStore
from ..temporal import ExpressionSmoother
from ..worker import AnalysisWorker, WorkerMessage
from ..vision import consensus_expression_probabilities


EXPRESSION_NAMES = {
    "Anger": "Злость", "Contempt": "Презрение", "Disgust": "Отвращение",
    "Fear": "Страх", "Happiness": "Радость", "Neutral": "Нейтральное",
    "Sadness": "Грусть", "Surprise": "Удивление",
}
LIVE_MAX_AGE_SECONDS = 1.8
LANDMARK_MAX_AGE_SECONDS = 0.20


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
    expressionScoresTextChanged = Signal()
    eyeStatusTextChanged = Signal()
    observationTextChanged = Signal()
    frameRevisionChanged = Signal()
    scoreHistoryChanged = Signal()
    landmarkPointsChanged = Signal()
    historyEntriesChanged = Signal()
    selectedTabChanged = Signal()
    cameraDevicesChanged = Signal()
    selectedCameraIndexChanged = Signal()
    cameraScanRunningChanged = Signal()
    _cameraScanCompleted = Signal(object, str)
    errorOccurred = Signal(str)

    def __init__(self, store: JournalStore | None = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.store = store or JournalStore()
        self.image_provider = FaceImageProvider()
        self.messages: queue.Queue[WorkerMessage] = queue.Queue(maxsize=2)
        self.worker: AnalysisWorker | None = None
        self.worker_generation = 0
        self._camera_devices: list[dict[str, object]] = []
        self._selected_camera_index = 0
        self._camera_scan_running = False
        self._preview_ready = False
        self._analysis_ready = False
        self._cameraScanCompleted.connect(self._finish_camera_scan)
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
        self._state = "stopped"
        self._hint = "Создай личную базу, чтобы начать."
        self._title = "Ожидание лица"
        self._description = "Сначала откалибруй личную базу"
        self._has_score = False
        self._score = 0.0
        self._confidence = "Модель не сообщает точность оценки состояния"
        self._quality = "Ожидаем пригодный кадр"
        self._expression = "Недостаточно данных"
        self._expression_scores = "Оценки выражения появятся при стабильном кадре"
        self._eye_status = "Глаза: ждём лицо"
        self._eye_closed = False
        self._eye_candidate_count = 0
        self._eye_open_reference: float | None = None
        self._eye_last_valid_at = 0.0
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
        self._closed = False
        self._timer = QTimer(self)
        self._timer.setInterval(80)
        self._timer.timeout.connect(self._poll_messages)
        self._timer.start()
        self.refreshHistory()
        QTimer.singleShot(500, self.refreshCameras)

    cameraState = Property(str, lambda self: self._state, notify=cameraStateChanged)
    cameraHint = Property(str, lambda self: self._hint, notify=cameraHintChanged)
    stateTitle = Property(str, lambda self: self._title, notify=stateTitleChanged)
    stateDescription = Property(str, lambda self: self._description, notify=stateDescriptionChanged)
    hasScore = Property(bool, lambda self: self._has_score, notify=hasScoreChanged)
    scoreValue = Property(float, lambda self: self._score, notify=scoreValueChanged)
    confidenceText = Property(str, lambda self: self._confidence, notify=confidenceTextChanged)
    qualityText = Property(str, lambda self: self._quality, notify=qualityTextChanged)
    expressionText = Property(str, lambda self: self._expression, notify=expressionTextChanged)
    expressionScoresText = Property(str, lambda self: self._expression_scores,
                                   notify=expressionScoresTextChanged)
    eyeStatusText = Property(str, lambda self: self._eye_status, notify=eyeStatusTextChanged)
    observationText = Property(str, lambda self: self._observation, notify=observationTextChanged)
    frameRevision = Property(int, lambda self: self._frame_revision, notify=frameRevisionChanged)
    scoreHistory = Property("QVariantList", lambda self: list(self._score_history), notify=scoreHistoryChanged)
    landmarkPoints = Property("QVariantList", lambda self: list(self._landmarks), notify=landmarkPointsChanged)
    historyEntries = Property("QVariantList", lambda self: list(self._history_entries), notify=historyEntriesChanged)
    selectedTab = Property(str, lambda self: self._selected_tab, notify=selectedTabChanged)
    cameraDevices = Property("QVariantList", lambda self: list(self._camera_devices),
                             notify=cameraDevicesChanged)
    selectedCameraIndex = Property(int, lambda self: self._selected_camera_index,
                                   notify=selectedCameraIndexChanged)
    cameraScanRunning = Property(bool, lambda self: self._camera_scan_running,
                                notify=cameraScanRunningChanged)

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
            "expression_scores": ("_expression_scores", self.expressionScoresTextChanged),
            "eye_status": ("_eye_status", self.eyeStatusTextChanged),
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
            self._set("hint", "Подожди, пока завершится поиск камер.")
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
            self._set("hint", "Подожди, пока завершится поиск камер.")
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
    def refreshCameras(self) -> None:
        """Scan available camera indices in a worker thread, never in the UI loop."""
        if self._camera_scan_running:
            return
        if self.worker is not None and self.worker.running:
            self._set("hint", "Останови камеру перед повторным поиском устройств.")
            return
        self._camera_scan_running = True
        self.cameraScanRunningChanged.emit()

        def scan() -> None:
            try:
                devices = discover_cameras()
                error = ""
            except Exception as exc:
                devices = []
                error = str(exc)
            self._cameraScanCompleted.emit(devices, error)

        threading.Thread(target=scan, name="camera-discovery", daemon=True).start()

    @Slot(object, str)
    def _finish_camera_scan(self, devices: object, error: str) -> None:
        self._camera_scan_running = False
        self.cameraScanRunningChanged.emit()
        if error:
            self._set("hint", f"Не удалось найти камеры: {error}")
            return
        self._camera_devices = list(devices or [])
        available_indices = [int(item["index"]) for item in self._camera_devices]
        if available_indices and self._selected_camera_index not in available_indices:
            self._selected_camera_index = available_indices[0]
            self.selectedCameraIndexChanged.emit()
        self.cameraDevicesChanged.emit()
        if self._camera_devices:
            selected = next((item for item in self._camera_devices
                             if int(item["index"]) == self._selected_camera_index), self._camera_devices[0])
            self._set("hint", f"Найдено камер: {len(self._camera_devices)} · выбрана {selected['label']}")
        else:
            self._set("hint", "Камеры не найдены. Подключи устройство и нажми «Обновить список».")

    @Slot(int)
    def selectCamera(self, index: int) -> None:
        if self.worker is not None and self.worker.running:
            self._set("hint", "Останови текущую обработку, чтобы переключить камеру.")
            return
        index = int(index)
        if not any(int(item["index"]) == index for item in self._camera_devices):
            self._set("hint", "Эта камера больше не доступна. Обнови список устройств.")
            return
        if index != self._selected_camera_index:
            self._selected_camera_index = index
            self.selectedCameraIndexChanged.emit()
        selected = next(item for item in self._camera_devices if int(item["index"]) == index)
        self._set("hint", f"Выбрана {selected['label']}.")

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
        self._preview_ready = False
        self._analysis_ready = False
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
        if not self._preview_ready:
            self._preview_ready = True
            # Состояние ожидания не объявляется активным до получения кадра.
            self._set("state", "active" if self._analysis_ready else "running")
            self._set("hint", "Камера и первый кадр готовы." if self._analysis_ready
                      else "Камера показывает видео; готовим анализатор.")
        # Ключевые точки относятся к кадру модели, а не к более свежему кадру
        # видеопотока. Скрываем их, если разница во времени заметна.
        if self._landmarks and abs(captured - self._landmark_monotonic) > LANDMARK_MAX_AGE_SECONDS:
            self._landmarks = []
            self.landmarkPointsChanged.emit()

    def _handle_message(self, message: WorkerMessage) -> None:
        if message.generation != self.worker_generation:
            return
        if message.kind == "camera_ready":
            # Камера открылась, но до первого кадра не сообщаем UI о готовом видео.
            self._set("hint", "Камера открыта; ждём первый кадр.")
            return
        if message.kind == "ready":
            self._analysis_ready = True
            if self._preview_ready:
                self._set("state", "active")
                self._set("hint", "Камера и локальный анализ готовы.")
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
        self._update_eye_status(observation)
        if (self._last_preview_monotonic and observation.monotonic_timestamp is not None and
                abs(self._last_preview_monotonic - observation.monotonic_timestamp) <= LANDMARK_MAX_AGE_SECONDS):
            self._landmark_monotonic = observation.monotonic_timestamp
            self._landmarks = [{"x": x, "y": y} for x, y in observation.landmarks]
        else:
            self._landmarks = []
        self.landmarkPointsChanged.emit()
        quality = float(observation.quality) if observation.present else 0.0
        if quality >= 0.75:
            quality_text = "Высокое"
        elif quality >= 0.45:
            quality_text = "Среднее"
        else:
            quality_text = observation.quality_reason or "Низкое"
        self._set("quality", quality_text)
        self._set("confidence", "Уверенность модели: не измеряется")
        expression_scores = consensus_expression_probabilities(observation)
        ranked_expressions = sorted(expression_scores.items(), key=lambda item: item[1], reverse=True)
        if observation.present and ranked_expressions and quality >= 0.45:
            top = " · ".join(f"{EXPRESSION_NAMES.get(name, name)} {value:.0%}"
                             for name, value in ranked_expressions[:2])
            self._set("expression_scores", f"Оценки выражения: {top}")
        else:
            self._set("expression_scores", "Оценки выражения появятся при стабильном кадре")
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
                self._set("description", observation.quality_reason or "Поставь лицо по центру камеры.")
            return

        if not self.tracking or self.scorer is None or self.session is None:
            return
        score = self.scorer.update(observation, timestamp)
        self.latest_score = score
        self.session.update(score, timestamp, observation.timestamp)
        if not score.usable or score.index is None:
            self._set("has_score", False)
            self._set("quality", score.reason or quality_text)
            self._set("title", "Недостаточно данных")
            self._set("description", score.reason or "Ожидаем надёжный лицевой сигнал.")
            self._set("observation", "Оценка временно приостановлена.")
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
        timestamp = observation.monotonic_timestamp or time.monotonic()
        state = self.expression_smoother.update(timestamp, observation.quality,
                                                consensus_expression_probabilities(observation))
        if state.ready and state.label:
            return EXPRESSION_NAMES.get(state.label, state.label)
        return state.reason or "Недостаточно данных"

    def _update_eye_status(self, observation) -> None:
        """Показывать закрытие глаз отдельно от классификации эмоций."""
        ratio = getattr(observation, "eye_aspect_ratio", None)
        timestamp = getattr(observation, "monotonic_timestamp", None) or time.monotonic()
        try:
            ratio = float(ratio)
        except (TypeError, ValueError, OverflowError):
            ratio = math.nan
        if not observation.present or not math.isfinite(ratio) or observation.quality < 0.45:
            # Кандидатные кадры должны быть последовательными, без пауз/плохого качества.
            self._eye_candidate_count = 0
            if self._eye_last_valid_at and timestamp - self._eye_last_valid_at > 0.7:
                self._eye_closed = False
                self._set("eye_status", "Глаза: нет надёжного сигнала")
            return

        self._eye_last_valid_at = timestamp
        reference = self._eye_open_reference if self._eye_open_reference and self._eye_open_reference > 0 else 0.29
        closed_threshold = max(0.12, min(0.19, reference * 0.60))
        open_threshold = max(closed_threshold + 0.045, min(0.27, reference * 0.78))
        desired = None
        if ratio <= closed_threshold:
            desired = True
        elif ratio >= open_threshold:
            desired = False
        if desired is None:
            self._eye_candidate_count = 0
            self._set("eye_status", f"Глаза: наблюдение · EAR {ratio:.2f}")
            return

        if desired == self._eye_closed:
            self._eye_candidate_count = 0
        else:
            self._eye_candidate_count += 1
            required = 3 if desired else 2
            if self._eye_candidate_count >= required:
                self._eye_closed = desired
                self._eye_candidate_count = 0
        label = "закрыты" if self._eye_closed else "открыты"
        self._set("eye_status", f"Глаза {label} · EAR {ratio:.2f}")

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
        open_eye_ratios = [float(item.eye_aspect_ratio) for item in self.calibration_observations
                           if item.eye_aspect_ratio is not None and
                           math.isfinite(float(item.eye_aspect_ratio)) and float(item.eye_aspect_ratio) >= 0.20]
        if open_eye_ratios:
            self._eye_open_reference = float(np.median(open_eye_ratios))
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
        self._set("eye_status", "Глаза: нет свежего сигнала")
        self._set("expression_scores", "Оценки выражения появятся при стабильном кадре")
        self.latest_frame = None
        self.latest_observation = None
        self._landmarks = []
        self.landmarkPointsChanged.emit()
        # Сохраняем живое видео; только анализируемое наблюдение устарело.
        self._clear_score("Нет свежего кадра")
        self._set("title", "Ожидание свежего кадра")
        self._set("description", "Ключевые точки и оценка временно скрыты.")
        self._set("expression", "Нет свежего предсказания")

    @Slot()
    def close(self) -> None:
        self._closed = True
        self._timer.stop()
        self.worker_generation += 1
        if self.session:
            self.session.finish(datetime.now().timestamp(), interrupted=True)
            self.session = None
        if self.worker:
            self.worker.stop(timeout=3.0)
            self.worker = None
        self.store.close()

