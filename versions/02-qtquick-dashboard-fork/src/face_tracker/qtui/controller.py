"""Qt/QML adapter for the existing local face-analysis and session services.

The controller deliberately owns no camera device. ``AnalysisWorker`` keeps
its capture/analyzer threads and generation fencing; this object only consumes
its bounded message queue on the Qt GUI thread.

Porting status: camera/calibration/live score/session/basic history/rating are
exposed. The legacy UI's blind-rating flow, per-session episode/rating detail,
privacy dialog, full action-unit and facial-pattern diagnostics, and model
comparison are not yet wired to QML. The face-only index and technical image quality remain
separate; this adapter never labels the index as a clinical stress measure.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
import math
import queue
import sqlite3
import threading
import time
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
from PySide6.QtCore import QObject, Property, QSize, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QImage
from PySide6.QtQuick import QQuickImageProvider

from ..events import EventTracker, TrackerState
from ..labels import LABELS, MIN_SAVE_QUALITY, EncodedFaceCrop, LabeledSampleStore, encode_face_crop
from ..personal_states import (
    FEATURE_NAMES, MIN_FRAMES_PER_SESSION_LABEL, MIN_TRAIN_SESSIONS, MIN_VALIDATION_SESSIONS,
    PERSONAL_LABELS, feature_vector, fit_personal_model,
)
from ..patterns import FacialPatternTracker
from ..scoring import CalibrationProfile, CalibrationSampler, Scorer
from ..session import TrackingSession
from ..storage import JournalStore
from ..temporal import ExpressionSmoother
from ..vision import EMOTION_MODEL_VERSION, FaceObservation
from ..worker import AnalysisWorker, WorkerMessage


EXPRESSION_NAMES = {
    "Anger": "Злость", "Contempt": "Презрение", "Disgust": "Отвращение",
    "Fear": "Страх", "Happiness": "Радость", "Neutral": "Нейтральное",
    "Sadness": "Грусть", "Surprise": "Удивление",
}
EXPRESSION_AGREEMENT = {
    "collecting": "Формируется",
    "switching": "Формируется",
    "ambiguous": "Слабое",
    "stable": "Устойчивое",
}
LIVE_MAX_AGE_SECONDS = 1.8
LANDMARK_MAX_AGE_SECONDS = 0.20
PERSONAL_SAVE_INTERVAL_SECONDS = 1.0
EXPRESSION_DRAFT_SECONDS = 30.0
EXPRESSION_DRAFT_PREFIX = "expression-draft/"
FACE_SIGNALS_MIN_QUALITY = 0.45
EYE_LABELS = {
    "open": ("Открыты", "Открыт"),
    "partial": ("Частично закрыты", "Частично закрыт"),
    "closed": ("Закрыты", "Закрыт"),
    "blink": ("Моргание", "Моргание"),
    "unknown": ("Не уверен", "Не уверен"),
}
VISIBLE_SIGNAL_LABELS = {
    "smile": "Улыбка",
    "eye_squint": "Сужение глаз",
    "eye_wide": "Расширение глаз",
    "brow_down": "Сведение бровей",
    "jaw_open": "Открытие рта",
}


@dataclass(frozen=True)
class ExpressionDraft:
    token: str
    crop: EncodedFaceCrop
    capture_id: str
    captured_at: float
    quality: float
    expires_at: float


class FaceImageProvider(QQuickImageProvider):
    """Provide the live camera frame and an explicitly captured expression draft."""

    def __init__(self) -> None:
        super().__init__(QQuickImageProvider.Image)
        self._lock = threading.Lock()
        self._image = QImage()
        self._draft_image = ("", QImage())

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

    def set_expression_draft(self, token: str, image: QImage) -> None:
        with self._lock:
            self._draft_image = (EXPRESSION_DRAFT_PREFIX + token, image) if token else ("", QImage())

    def requestImage(self, image_id: str, size: QSize, requested_size: QSize) -> QImage:
        with self._lock:
            draft_id, draft = self._draft_image
            if image_id.startswith(EXPRESSION_DRAFT_PREFIX):
                image = draft.copy() if image_id == draft_id else QImage()
            else:
                image = self._image.copy()
        if not image.isNull() and requested_size.isValid():
            image = image.scaled(requested_size)
        if size is not None:
            size.setWidth(image.width())
            size.setHeight(image.height())
        # QQuickImageProvider в PySide6 требует вернуть сам QImage. Размер
        # исходника передаётся через аргумент size; кортеж здесь ломает Image.
        return image


_DEFAULT_SAMPLE_STORE = object()


class TrackerController(QObject):
    """QML-facing properties and slots around existing tracking domain code."""

    cameraStateChanged = Signal()
    cameraHintChanged = Signal()
    stateTitleChanged = Signal()
    stateDescriptionChanged = Signal()
    hasScoreChanged = Signal()
    scoreValueChanged = Signal()
    expressionAgreementTextChanged = Signal()
    qualityTextChanged = Signal()
    expressionTextChanged = Signal()
    observationTextChanged = Signal()
    eyeStateTextChanged = Signal()
    leftEyeTextChanged = Signal()
    rightEyeTextChanged = Signal()
    eyeStateReasonChanged = Signal()
    eyeClosurePercentChanged = Signal()
    blinkCountChanged = Signal()
    visibleSignalsChanged = Signal()
    frameRevisionChanged = Signal()
    scoreHistoryChanged = Signal()
    landmarkPointsChanged = Signal()
    historyEntriesChanged = Signal()
    selectedTabChanged = Signal()
    sampleFrameChanged = Signal()
    personalSummaryChanged = Signal()
    expressionSamplesChanged = Signal()
    sampleFeedbackChanged = Signal()
    expressionDraftChanged = Signal()
    errorOccurred = Signal(str)

    def __init__(self, store: JournalStore | None = None, parent: QObject | None = None,
                 sample_store: LabeledSampleStore | None | object = _DEFAULT_SAMPLE_STORE) -> None:
        super().__init__(parent)
        self.store = store or JournalStore()
        try:
            # Встроенный запуск получает стандартное локальное хранилище;
            # при внедрённом журнале хранилище примеров задаётся явно.
            self.sample_store = (LabeledSampleStore() if store is None else None
                                 ) if sample_store is _DEFAULT_SAMPLE_STORE else sample_store
        except (OSError, sqlite3.Error):
            self.sample_store = None
        self._sample_candidate: WorkerMessage | None = None
        self._expression_draft: ExpressionDraft | None = None
        self._last_sample_times: dict[str, float] = {}
        self._sample_frame: dict[str, Any] = {}
        self._personal_summary: dict[str, Any] = {}
        self._expression_samples: list[dict[str, Any]] = []
        self._sample_feedback = {"personal": "", "expression": ""}
        self._personal_fit = fit_personal_model(())
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
        self.expression_smoother = ExpressionSmoother(expected_labels=EXPRESSION_NAMES)
        self.latest_observation = None
        self.latest_score = None
        self.latest_frame = None
        self._state = "stopped"
        self._hint = "Создай личную базу, чтобы начать."
        self._title = "Ожидание лица"
        self._description = "Сначала откалибруй личную базу"
        self._has_score = False
        self._score = 0.0
        self._expression_agreement = "—"
        self._quality = "Ожидаем пригодный кадр"
        self._expression = "Ожидание лица"
        self._observation = "Создай личную базу, чтобы начать."
        self._eye_state_text = "Не уверен"
        self._left_eye_text = "Не уверен"
        self._right_eye_text = "Не уверен"
        self._eye_state_reason = "Камера не запущена."
        self._eye_closure_percent: int | None = None
        self._blink_count: int | None = None
        self._visible_signals: list[str] = []
        self._face_signals_at: float | None = None
        self._visible_pattern_tracker: FacialPatternTracker | None = None
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
        self._refresh_personal_samples()
        self._refresh_expression_samples()
        self._refresh_sample_frame()
        self._timer.start()
        self.refreshHistory()

    cameraState = Property(str, lambda self: self._state, notify=cameraStateChanged)
    cameraHint = Property(str, lambda self: self._hint, notify=cameraHintChanged)
    stateTitle = Property(str, lambda self: self._title, notify=stateTitleChanged)
    stateDescription = Property(str, lambda self: self._description, notify=stateDescriptionChanged)
    hasScore = Property(bool, lambda self: self._has_score, notify=hasScoreChanged)
    scoreValue = Property(float, lambda self: self._score, notify=scoreValueChanged)
    expressionAgreementText = Property(str, lambda self: self._expression_agreement,
                                       notify=expressionAgreementTextChanged)
    qualityText = Property(str, lambda self: self._quality, notify=qualityTextChanged)
    expressionText = Property(str, lambda self: self._expression, notify=expressionTextChanged)
    observationText = Property(str, lambda self: self._observation, notify=observationTextChanged)
    eyeStateText = Property(str, lambda self: self._eye_state_text, notify=eyeStateTextChanged)
    leftEyeText = Property(str, lambda self: self._left_eye_text, notify=leftEyeTextChanged)
    rightEyeText = Property(str, lambda self: self._right_eye_text, notify=rightEyeTextChanged)
    eyeStateReason = Property(str, lambda self: self._eye_state_reason, notify=eyeStateReasonChanged)
    eyeClosurePercent = Property("QVariant", lambda self: self._eye_closure_percent,
                                 notify=eyeClosurePercentChanged)
    blinkCount = Property("QVariant", lambda self: self._blink_count, notify=blinkCountChanged)
    visibleSignals = Property("QStringList", lambda self: self._visible_signals, notify=visibleSignalsChanged)
    frameRevision = Property(int, lambda self: self._frame_revision, notify=frameRevisionChanged)
    scoreHistory = Property("QVariantList", lambda self: list(self._score_history), notify=scoreHistoryChanged)
    landmarkPoints = Property("QVariantList", lambda self: list(self._landmarks), notify=landmarkPointsChanged)
    historyEntries = Property("QVariantList", lambda self: list(self._history_entries), notify=historyEntriesChanged)
    selectedTab = Property(str, lambda self: self._selected_tab, notify=selectedTabChanged)

    sampleFrame = Property("QVariantMap", lambda self: self._sample_frame, notify=sampleFrameChanged)
    personalSummary = Property("QVariantMap", lambda self: self._personal_summary, notify=personalSummaryChanged)
    expressionSamples = Property("QVariantList", lambda self: self._expression_samples, notify=expressionSamplesChanged)
    expressionLabels = Property("QVariantList", lambda self: [
        {"key": label, "title": EXPRESSION_NAMES[label]} for label in LABELS
    ], constant=True)
    sampleFeedback = Property("QVariantMap", lambda self: self._sample_feedback, notify=sampleFeedbackChanged)

    @Property(str, notify=expressionDraftChanged)
    def expressionDraftUrl(self) -> str:
        draft = self._expression_draft
        return f"image://camera/{EXPRESSION_DRAFT_PREFIX}{draft.token}" if draft else ""

    def _set(self, key: str, value: Any) -> None:
        attr, signal = {
            "state": ("_state", self.cameraStateChanged),
            "hint": ("_hint", self.cameraHintChanged),
            "title": ("_title", self.stateTitleChanged),
            "description": ("_description", self.stateDescriptionChanged),
            "has_score": ("_has_score", self.hasScoreChanged),
            "score": ("_score", self.scoreValueChanged),
            "expression_agreement": ("_expression_agreement", self.expressionAgreementTextChanged),
            "quality": ("_quality", self.qualityTextChanged),
            "expression": ("_expression", self.expressionTextChanged),
            "observation": ("_observation", self.observationTextChanged),
            "eye_state_text": ("_eye_state_text", self.eyeStateTextChanged),
            "left_eye_text": ("_left_eye_text", self.leftEyeTextChanged),
            "right_eye_text": ("_right_eye_text", self.rightEyeTextChanged),
            "eye_state_reason": ("_eye_state_reason", self.eyeStateReasonChanged),
            "eye_closure_percent": ("_eye_closure_percent", self.eyeClosurePercentChanged),
            "blink_count": ("_blink_count", self.blinkCountChanged),
            "visible_signals": ("_visible_signals", self.visibleSignalsChanged),
        }[key]
        if getattr(self, attr) != value:
            setattr(self, attr, value)
            signal.emit()

    def _clear_score(self, reason: str) -> None:
        self.latest_score = None
        self._set("has_score", False)
        self._set("quality", reason)

    @Slot()
    def createBaseline(self) -> None:
        self.calibrate()

    @Slot()
    def calibrate(self) -> None:
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
        if self.calibrating:
            count = self.calibration_sampler.sample_count
            self._set("title", "Калибровка ещё идёт")
            self._set("description", f"Подожди завершения сбора личной базы: {count}/50 кадров.")
            self._set("hint", "Сессия начнётся после завершения калибровки.")
            self._set("observation", "Дождись готовности личной базы, затем нажми «Начать сессию».")
            return
        if self.profile is None:
            self._set("title", "Сначала создай личную базу")
            self._set("description", "Сбор занимает не менее пяти секунд; дождись сообщения о готовности базы.")
            self._set("hint", "Нажми «Создать базу» и дождись завершения калибровки.")
            self._set("observation", "Для старта сессии нужна готовая личная база.")
            return
        if self.worker and self.worker.running:
            self._set("title", "Камера ещё занята")
            self._set("description", "Дождись завершения предыдущей обработки и повтори запуск.")
            self._set("hint", "Камера ещё работает; дождись её остановки.")
            self._set("observation", "Новая сессия пока не запущена.")
            return
        self.scorer = Scorer(self.profile)
        self.event_tracker = EventTracker()
        self.session = TrackingSession(self.store, self.event_tracker)
        self.session.start(datetime.now().timestamp())
        self._visible_pattern_tracker = FacialPatternTracker(self.profile)
        self.tracking = True
        self._last_tense = False
        self._score_history.clear()
        self._last_score_sample_at = 0.0
        self.scoreHistoryChanged.emit()
        self._clear_score("Ожидаем пригодный кадр")
        self._set("state", "starting")
        self._set("title", "Подключение камеры")
        self._set("description", "Индекс появится после получения свежего качественного кадра.")
        self._set("observation", "Ожидаем пригодный кадр для сравнения.")
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
        self._refresh_personal_samples()
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

    def _set_sample_feedback(self, kind: str, message: str) -> None:
        if message != self._sample_feedback[kind]:
            self._sample_feedback[kind] = message
            self.sampleFeedbackChanged.emit()

    def _refresh_personal_samples(self, *, refit: bool = True) -> None:
        store = self.sample_store
        status_override = ""
        try:
            counts = store.personal_counts() if store is not None else {}
            if refit:
                self._personal_fit = fit_personal_model(
                    store.list_personal_samples() if store is not None else (),
                    active_session=self.session.capture_id if self.session else None,
                )
            if store is None:
                status_override = "Локальное хранилище недоступно"
        except (OSError, sqlite3.Error, ValueError, TypeError):
            self._personal_fit = fit_personal_model(())
            counts = {}
            status_override = "Не удалось прочитать личные примеры"
        fit = self._personal_fit
        self._personal_summary = {
            "status": status_override or fit.status,
            "reason": "" if status_override else fit.reason,
            "sessions": fit.sessions,
            "storeAvailable": store is not None,
            "requirements": (f"Для каждой метки сохрани от {MIN_FRAMES_PER_SESSION_LABEL} кадров за сессию. "
                             f"Нужно от {MIN_TRAIN_SESSIONS} сессий для обучения и от "
                             f"{MIN_VALIDATION_SESSIONS} других для проверки. "
                             "Распределение происходит автоматически после завершения сессии."),
            "labels": [
                {"label": label, "count": counts.get(label, 0),
                 "train": fit.coverage[label][0], "validation": fit.coverage[label][1]}
                for label in PERSONAL_LABELS
            ],
        }
        self.personalSummaryChanged.emit()

    def _refresh_expression_samples(self) -> None:
        store = self.sample_store
        rows = []
        if store is not None:
            try:
                for sample in reversed(store.list_samples()):
                    try:
                        date = datetime.fromisoformat(sample.created_at).astimezone().strftime("%d.%m.%Y %H:%M:%S")
                    except ValueError:
                        date = sample.created_at
                    rows.append({
                        "id": sample.sample_id, "label": sample.label,
                        "description": f"{EXPRESSION_NAMES.get(sample.label, sample.label)} · {date}",
                    })
            except (OSError, sqlite3.Error, ValueError, TypeError):
                self._set_sample_feedback("expression", "Не удалось прочитать примеры выражений.")
        self._expression_samples = rows
        self.expressionSamplesChanged.emit()

    @Slot(str, result=QUrl)
    def expressionPreview(self, sample_id: str) -> QUrl:
        store = self.sample_store
        if store is not None:
            try:
                path = store.expression_preview_path(sample_id)
                if path is not None:
                    return QUrl.fromLocalFile(str(path))
            except (OSError, sqlite3.Error):
                pass
        return QUrl()

    def _sample_frame_issue(self) -> str:
        if self.sample_store is None:
            return "Локальное хранилище недоступно."
        if not self.tracking or self.session is None or self.session.capture_id is None:
            return "Начни сессию, чтобы сохранить кадр."
        if self.worker is None or not self.worker.running:
            return "Камера остановлена."
        if self._sample_candidate is None:
            return "Ожидаем свежий кадр с лицом."
        observation = self._sample_candidate.observation
        captured = observation.monotonic_timestamp
        if captured is None or not 0 <= time.monotonic() - captured <= LIVE_MAX_AGE_SECONDS:
            self._sample_candidate = None
            return "Нет свежего кадра."
        if not observation.present or observation.face_box is None:
            return observation.quality_reason or "Лицо не найдено."
        if not MIN_SAVE_QUALITY <= observation.quality <= 1:
            return observation.quality_reason or "Низкое качество кадра."
        return ""

    def _label_issue(self, kind: str, timestamp: float) -> str:
        interval = PERSONAL_SAVE_INTERVAL_SECONDS if kind == "personal" else 0.0
        if timestamp - self._last_sample_times.get(kind, -math.inf) <= interval:
            return "Дождись следующего кадра." if not interval else "Подожди секунду между личными примерами."
        return ""

    def _personal_features(self, observation: FaceObservation) -> dict[str, float]:
        return {key: observation.patterns[key] for key in FEATURE_NAMES if key in observation.patterns}

    def _refresh_sample_frame(self) -> None:
        issue = self._sample_frame_issue()
        result = {"frameText": issue, "personalIssue": issue, "expressionIssue": issue,
                  "prediction": "", "predictionReason": issue}
        if self.tracking and issue and self._personal_fit.centers is not None:
            result["prediction"] = "Не уверен"
            result["predictionReason"] = issue
        if not issue:
            observation = self._sample_candidate.observation
            timestamp = observation.monotonic_timestamp
            result["frameText"] = f"Свежий кадр · качество {observation.quality:.0%}"
            result["personalIssue"] = self._label_issue("personal", timestamp)
            result["expressionIssue"] = self._label_issue("expression", timestamp)
            try:
                vector = feature_vector(self._personal_features(observation))
            except ValueError:
                result["personalIssue"] = "Нет полных данных о движениях лица."
                result["predictionReason"] = result["personalIssue"]
                if self.tracking and self._personal_fit.centers is not None:
                    result["prediction"] = "Не уверен"
            else:
                if self._personal_fit.centers is not None:
                    label = self._personal_fit.predict(vector)
                    result["prediction"] = label or "Не уверен"
                    if label is None:
                        result["predictionReason"] = "Кадр слабо похож на сохранённые примеры."
        if result != self._sample_frame:
            self._sample_frame = result
            self.sampleFrameChanged.emit()

    @Slot(str)
    def savePersonalSample(self, label: str) -> None:
        issue = self._sample_frame_issue()
        if issue:
            self._set_sample_feedback("personal", issue)
            return
        candidate = self._sample_candidate
        observation = candidate.observation
        timestamp = observation.monotonic_timestamp
        issue = self._label_issue("personal", timestamp)
        if issue:
            self._set_sample_feedback("personal", issue)
            return
        try:
            self.sample_store.save_personal_sample(
                candidate.frame, observation.face_box, label, observation.quality,
                self.session.capture_id, self._personal_features(observation),
            )
        except (OSError, sqlite3.Error, ValueError, TypeError, RuntimeError, cv2.error):
            self._set_sample_feedback("personal", "Не удалось сохранить кадр.")
            return
        self._last_sample_times["personal"] = timestamp
        self._set_sample_feedback("personal", f"Кадр сохранён · {label}")
        self._refresh_personal_samples(refit=False)
        self._refresh_sample_frame()

    @Slot(result=bool)
    def prepareExpressionSample(self) -> bool:
        self.discardExpressionDraft()
        issue = self._sample_frame_issue()
        if issue:
            self._set_sample_feedback("expression", issue)
            return False
        candidate = self._sample_candidate
        observation = candidate.observation
        timestamp = observation.monotonic_timestamp
        issue = self._label_issue("expression", timestamp)
        if issue:
            self._set_sample_feedback("expression", issue)
            return False
        try:
            crop = encode_face_crop(candidate.frame, observation.face_box)
            image = QImage.fromData(crop.jpeg)
        except (ValueError, TypeError, RuntimeError, cv2.error):
            image = QImage()
        if image.isNull():
            self._set_sample_feedback("expression", "Не удалось показать кадр.")
            return False
        draft = ExpressionDraft(uuid4().hex, crop, self.session.capture_id, timestamp,
                                observation.quality, time.monotonic() + EXPRESSION_DRAFT_SECONDS)
        self.image_provider.set_expression_draft(draft.token, image)
        self._expression_draft = draft
        self._set_sample_feedback("expression", "")
        self.expressionDraftChanged.emit()
        return True

    @Slot()
    def discardExpressionDraft(self) -> None:
        if self._expression_draft is not None:
            self._expression_draft = None
            self.image_provider.set_expression_draft("", QImage())
            self.expressionDraftChanged.emit()

    def _expire_expression_draft(self) -> None:
        draft = self._expression_draft
        if draft is None:
            return
        if (not self.tracking or self.session is None or self.session.capture_id != draft.capture_id or
                self.worker is None or not self.worker.running):
            issue = "Сессия закончилась. Кадр не сохранён."
        elif time.monotonic() >= draft.expires_at:
            issue = "Время просмотра истекло. Открой новый кадр."
        else:
            return
        self.discardExpressionDraft()
        self._set_sample_feedback("expression", issue)

    @Slot(str, result=bool)
    def confirmExpressionSample(self, label: str) -> bool:
        self._expire_expression_draft()
        draft = self._expression_draft
        if draft is None:
            return False
        try:
            self.sample_store.save_expression_crop(
                draft.crop, label, draft.quality, EMOTION_MODEL_VERSION,
                "mediapipe-mesh-bbox / BGR-JPEG92", draft.capture_id,
            )
        except (OSError, sqlite3.Error, ValueError, RuntimeError):
            self._set_sample_feedback("expression", "Не удалось сохранить кадр.")
            return False
        self.discardExpressionDraft()
        self._last_sample_times["expression"] = draft.captured_at
        self._set_sample_feedback("expression", f"Кадр сохранён · {EXPRESSION_NAMES[label]}")
        self._refresh_expression_samples()
        self._refresh_sample_frame()
        return True

    @Slot()
    def clearPersonalSamples(self) -> None:
        if self.sample_store is None:
            return
        try:
            self.sample_store.delete_all_personal()
        except (OSError, sqlite3.Error, RuntimeError):
            self._set_sample_feedback("personal", "Не удалось удалить все личные примеры.")
        else:
            self._set_sample_feedback("personal", "Личные примеры удалены.")
        self._refresh_personal_samples()
        self._refresh_sample_frame()

    @Slot(str, str)
    def updateExpressionSample(self, sample_id: str, label: str) -> None:
        if self.sample_store is None:
            return
        try:
            changed = self.sample_store.update_label(sample_id, label)
        except (OSError, sqlite3.Error, ValueError):
            self._set_sample_feedback("expression", "Не удалось исправить метку.")
        else:
            self._set_sample_feedback("expression", "Метка исправлена." if changed else "Пример уже удалён.")
        self._refresh_expression_samples()

    @Slot(str)
    def deleteExpressionSample(self, sample_id: str) -> None:
        if self.sample_store is None:
            return
        try:
            deleted = self.sample_store.delete_sample(sample_id)
        except (OSError, sqlite3.Error, RuntimeError):
            self._set_sample_feedback("expression", "Не удалось удалить пример.")
        else:
            self._set_sample_feedback("expression", "Пример удалён." if deleted else "Пример уже удалён.")
        self._refresh_expression_samples()

    def _start_worker(self) -> None:
        if self.worker is not None and self.worker.running:
            return
        self._sample_candidate = None
        self._last_sample_times.clear()
        self._clear_face_signals("Ожидаем кадр с лицом.")
        self._clear_expression("Ожидание лица")
        self.worker_generation += 1
        self.worker = AnalysisWorker(self.messages, generation=self.worker_generation)
        self.worker.start()
        self._set("state", "starting")
        self._set("hint", "Открываем камеру и загружаем локальные модели…")
        self._refresh_sample_frame()

    def _stop_worker(self) -> None:
        self._sample_candidate = None
        self._expire_expression_draft()
        self._refresh_sample_frame()
        self._clear_face_signals("Камера не запущена.")
        self._clear_expression("Камера остановлена")
        self._visible_pattern_tracker = None
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
        if (self._face_signals_at is not None and
                time.monotonic() - self._face_signals_at > LIVE_MAX_AGE_SECONDS):
            self._clear_face_signals("Нет свежего кадра.")
        if self.latest_observation is not None:
            captured = self.latest_observation.monotonic_timestamp or 0.0
            if captured and time.monotonic() - captured > LIVE_MAX_AGE_SECONDS:
                self._expire_stale_frame()
        self._expire_expression_draft()
        self._refresh_sample_frame()

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
            self._sample_candidate = None
            self.calibrating = False
            self.tracking = False
            self._clear_face_signals("Обработка кадра остановлена из-за ошибки.")
            self._clear_expression("Ошибка обработки")
            if self.session:
                self.session.finish(datetime.now().timestamp(), interrupted=True)
                self.session = None
                self._refresh_personal_samples()
            self._clear_score("Нет активного кадра")
            self._set("state", "error")
            self._set("title", "Ошибка камеры или модели")
            self._set("description", message.error or "Не удалось запустить обработку.")
            self._set("hint", "Проверь подключение камеры и попробуй ещё раз.")
            self.errorOccurred.emit(message.error or "Не удалось запустить обработку.")
            self.refreshHistory()
            return
        if message.kind == "stopped":
            self._sample_candidate = None
            if self._state != "error":
                self._clear_face_signals("Камера не запущена.")
                self._clear_expression("Камера остановлена")
            return
        if message.kind != "frame":
            return
        if message.frame is None or message.observation is None:
            self._sample_candidate = None
            self._clear_face_signals("Не удалось обработать кадр.")
            self._clear_expression("Нет данных кадра")
            return
        # Save against the same image used for the observation, not the newer preview.
        self._sample_candidate = message if self.tracking else None
        observation = message.observation
        self.latest_observation = observation
        if self.calibrating or self.tracking:
            self._update_face_signals(observation)
        if (self._last_preview_monotonic and observation.monotonic_timestamp is not None and
                abs(self._last_preview_monotonic - observation.monotonic_timestamp) <= LANDMARK_MAX_AGE_SECONDS):
            self._landmark_monotonic = observation.monotonic_timestamp
            self._landmarks = [{"x": x, "y": y} for x, y in observation.landmarks]
        else:
            self._landmarks = []
        self.landmarkPointsChanged.emit()
        quality = float(observation.quality) if observation.present else 0.0
        if not 0 <= quality <= 1:
            quality_text = "Некорректное качество"
        elif quality >= 0.75:
            quality_text = "Высокое"
        elif quality >= 0.45:
            quality_text = "Среднее"
        else:
            quality_text = observation.quality_reason or "Низкое"
        self._set("quality", quality_text)
        self._update_expression(observation)

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
        self._set("observation", "Изменение мимики сохраняется." if tense else "Мимика близка к личной базе.")
        self._append_score(value, observation.timestamp)
        if tense != self._last_tense:
            self._last_tense = tense
            self.refreshHistory()

    def _set_eye_signals(self, *, overall: str = "unknown", left: str = "unknown",
                         right: str = "unknown", reason: str = "", closure: int | None = None,
                         blinks: int | None = None) -> None:
        self._set("eye_state_text", EYE_LABELS[overall][0])
        self._set("left_eye_text", EYE_LABELS[left][1])
        self._set("right_eye_text", EYE_LABELS[right][1])
        self._set("eye_state_reason", reason)
        self._set("eye_closure_percent", closure)
        self._set("blink_count", blinks)

    def _clear_face_signals(self, reason: str) -> None:
        self._face_signals_at = None
        if self._visible_pattern_tracker is not None:
            self._visible_pattern_tracker.reset()
        self._set_eye_signals(reason=reason)
        self._set("visible_signals", [])

    def _update_face_signals(self, observation: FaceObservation) -> None:
        timestamp = observation.monotonic_timestamp
        if timestamp is None or not 0.0 <= time.monotonic() - timestamp <= LIVE_MAX_AGE_SECONDS:
            self._clear_face_signals("Нет свежего кадра.")
            return
        self._face_signals_at = timestamp
        eye = observation.eye_state
        if eye is None:
            self._set_eye_signals(reason=observation.quality_reason or "Не удалось определить состояние глаз.")
        elif eye.overall == "unknown":
            eye_reason = eye.reason
            if not eye_reason or eye_reason == "Недостаточное качество кадра для оценки глаз":
                eye_reason = observation.quality_reason or eye_reason
            self._set_eye_signals(left=eye.left, right=eye.right,
                                  reason=eye_reason or "Не удалось определить состояние глаз.")
        else:
            closure = round(eye.closure * 100) if eye.closure is not None else None
            self._set_eye_signals(overall=eye.overall, left=eye.left, right=eye.right,
                                  closure=closure, blinks=eye.blink_count)

        # Глаза вычисляются быстрым MediaPipe-путём отдельно от эмоций. Не скрываем
        # пригодную оценку век только потому, что HSEmotion не вернул вероятности.
        if not observation.present or not FACE_SIGNALS_MIN_QUALITY <= observation.quality <= 1.0:
            if self._visible_pattern_tracker is not None:
                self._visible_pattern_tracker.reset()
            self._set("visible_signals", [])
            self._face_signals_at = timestamp
            return

        visible_signals = []
        tracker = self._visible_pattern_tracker
        if tracker is not None:
            # Не удерживаем старое движение, когда его больше нечем проверить.
            patterns = {key: observation.patterns[key] for key in VISIBLE_SIGNAL_LABELS
                        if key in observation.patterns and key in tracker.profile.pattern_medians}
            summary = tracker.update(timestamp, observation.quality, patterns)
            if summary.ready:
                visible_signals = [label for key, label in VISIBLE_SIGNAL_LABELS.items()
                                   if key in patterns and summary.signals[key].above_baseline]
        self._set("visible_signals", visible_signals)

    def _clear_expression(self, reason: str) -> None:
        self.expression_smoother.reset()
        self._set("expression", reason)
        self._set("expression_agreement", "—")

    def _update_expression(self, observation: FaceObservation) -> None:
        now = time.monotonic()
        timestamp = observation.monotonic_timestamp
        if timestamp is None:
            timestamp = now
        if not 0 <= now - timestamp <= LIVE_MAX_AGE_SECONDS:
            self._clear_expression("Нет свежего кадра")
            return
        if not observation.present:
            self._clear_expression("Лицо не найдено")
            return
        state = self.expression_smoother.update(
            timestamp, observation.quality, observation.expressions
        )
        label = EXPRESSION_NAMES.get(state.label) if state.ready and state.label else None
        reason = state.reason
        if not observation.expressions and state.ready_reason in {"low_quality", "missing_probabilities"}:
            reason = observation.quality_reason or reason
        self._set("expression", label or reason or "Недостаточно данных")
        self._set("expression_agreement", EXPRESSION_AGREEMENT.get(state.ready_reason, "—"))

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
        self._clear_expression("Нет свежего кадра")
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

    @Slot()
    def close(self) -> None:
        self.discardExpressionDraft()
        self._closed = True
        self._timer.stop()
        self.worker_generation += 1
        if self.session:
            self.session.finish(datetime.now().timestamp(), interrupted=True)
            self.session = None
        if self.worker:
            self.worker.stop(timeout=3.0)
            self.worker = None
        if self.sample_store is not None:
            self.sample_store.close()
        self.store.close()

