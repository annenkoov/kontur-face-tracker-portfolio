"""Small, deterministic states for separating eye closure from optical blur."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping


# MediaPipe blendshape activations are in [0, 1]. Mid-level activations indicate
# a blink; stronger bilateral activations are treated as sustained closure.
BLINK_SCORE_THRESHOLD = 0.55
CLOSED_SCORE_THRESHOLD = 0.82
QUALITY_STATUS_CONFIRMATION_SECONDS = 0.4
PERSISTENT_EYE_CLOSURE_SECONDS = 0.4
DEFAULT_QUALITY_STATUS = "Камера работает локально"


@dataclass(frozen=True)
class EyeState:
    state: str
    valid: bool
    left_score: float | None = None
    right_score: float | None = None

    @property
    def bilateral_blink(self) -> bool:
        """True only for a temporary, bilateral blink (never persistent closure)."""
        return (
            self.valid
            and self.state == "blink"
            and self.left_score is not None
            and self.right_score is not None
            and self.left_score >= BLINK_SCORE_THRESHOLD
            and self.right_score >= BLINK_SCORE_THRESHOLD
        )


def classify_eye_state(blendshapes: Mapping[str, float]) -> EyeState:
    """Classify both eyes, returning an invalid ``unknown`` on missing/bad data."""
    try:
        left = float(blendshapes["eyeBlinkLeft"])
        right = float(blendshapes["eyeBlinkRight"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return EyeState("unknown", False)
    if (not math.isfinite(left) or not math.isfinite(right)
            or not 0.0 <= left <= 1.0 or not 0.0 <= right <= 1.0):
        return EyeState("unknown", False)
    if left >= CLOSED_SCORE_THRESHOLD and right >= CLOSED_SCORE_THRESHOLD:
        state = "closed"
    elif left >= BLINK_SCORE_THRESHOLD or right >= BLINK_SCORE_THRESHOLD:
        state = "blink"
    else:
        state = "open"
    return EyeState(state, True, left, right)


class EyeStateStabilizer:
    """Treat high bilateral blink activations as a blink until closure persists."""

    def __init__(self, confirmation_seconds: float = PERSISTENT_EYE_CLOSURE_SECONDS) -> None:
        interval = float(confirmation_seconds)
        if not math.isfinite(interval) or interval < 0:
            raise ValueError("Время подтверждения закрытых глаз некорректно")
        self.confirmation_seconds = interval
        self._closed_since: float | None = None

    def reset(self) -> None:
        self._closed_since = None

    def update(self, now: float, eye_state: EyeState) -> EyeState:
        timestamp = float(now)
        if not math.isfinite(timestamp):
            self.reset()
            return EyeState("unknown", False)
        if not eye_state.valid or eye_state.state != "closed":
            self.reset()
            return eye_state
        if self._closed_since is None or timestamp < self._closed_since:
            self._closed_since = timestamp
        if timestamp - self._closed_since + 1e-9 >= self.confirmation_seconds:
            return eye_state
        # A high-score bilateral closure is a normal blink until it is sustained.
        return EyeState("blink", True, eye_state.left_score, eye_state.right_score)


class QualityStatusDebouncer:
    """Debounce user-facing frame/face status, never scoring validity."""

    def __init__(self, confirmation_seconds: float = QUALITY_STATUS_CONFIRMATION_SECONDS) -> None:
        interval = float(confirmation_seconds)
        if not math.isfinite(interval) or interval < 0:
            raise ValueError("Время подтверждения статуса должно быть конечным и неотрицательным")
        self.confirmation_seconds = interval
        self._stable_status = DEFAULT_QUALITY_STATUS
        self._pending_status: str | None = None
        self._pending_since: float | None = None

    def update(self, now: float, face_present: bool, frame_fresh: bool,
               reason: str | None) -> str:
        """Return a debounced display status; callers must still score each raw frame."""
        timestamp = float(now)
        if not math.isfinite(timestamp):
            raise ValueError("Время статуса должно быть конечным")
        candidate = None if face_present and frame_fresh else (
            str(reason).strip() if reason and str(reason).strip() else
            ("Лицо не найдено" if not face_present else "Нет свежего кадра")
        )
        desired = candidate or DEFAULT_QUALITY_STATUS
        # A fresh valid frame clears a confirmed warning immediately; only loss
        # transitions need 0.4 s confirmation, preventing tiny camera/model gaps
        # from flashing while keeping the display truthful on recovery.
        if desired == DEFAULT_QUALITY_STATUS:
            self._stable_status = DEFAULT_QUALITY_STATUS
            self._pending_status = None
            self._pending_since = None
            return self._stable_status
        if desired == self._stable_status:
            self._pending_status = None
            self._pending_since = None
            return self._stable_status
        if desired != self._pending_status:
            self._pending_status = desired
            self._pending_since = timestamp
            return self._stable_status
        if (self._pending_since is not None and
                timestamp - self._pending_since + 1e-9 >= self.confirmation_seconds):
            self._stable_status = desired
            self._pending_status = None
            self._pending_since = None
        return self._stable_status
