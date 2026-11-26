"""OpenCV camera access, isolated from the Tkinter event loop."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time

import cv2


class CameraError(RuntimeError):
    """Raised when a camera cannot be opened or stops returning frames."""


def discover_camera_indices(max_index: int = 8,
                            cancel_event: threading.Event | None = None) -> list[int]:
    """Return OpenCV camera indices that the configured Windows backend can open.

    Probing can block on some camera drivers, so callers should run this helper
    outside the UI thread. Every successful or failed probe releases its handle.
    """
    found: list[int] = []
    for index in range(max(0, int(max_index))):
        if cancel_event is not None and cancel_event.is_set():
            break
        capture = None
        try:
            capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if (capture is not None and capture.isOpened() and
                    not (cancel_event is not None and cancel_event.is_set())):
                ok, frame = capture.read()
                if (ok and frame is not None and
                        not (cancel_event is not None and cancel_event.is_set())):
                    found.append(index)
        except (cv2.error, OSError):
            # A driver can reject one index without invalidating the rest.
            continue
        finally:
            if capture is not None:
                capture.release()
    return found


@dataclass(frozen=True)
class FrameResult:
    frame: object
    captured_at: float
    captured_monotonic: float | None = None


class CameraSource:
    def __init__(self, index: int = 0) -> None:
        self.index = index
        self._capture = None

    def start(self) -> None:
        if self._capture is not None and self._capture.isOpened():
            return
        self._capture = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not self._capture.isOpened():
            self._capture.release()
            self._capture = None
            raise CameraError(f"Не удалось открыть камеру {self.index}")
        # Best-effort latency hint; OpenCV backends may ignore this property.
        try:
            self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except (AttributeError, cv2.error):
            pass

    def read(self) -> FrameResult:
        if self._capture is None or not self._capture.isOpened():
            raise CameraError("Камера не запущена")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise CameraError("Камера не вернула кадр")
        return FrameResult(frame=frame, captured_at=time.time(), captured_monotonic=time.monotonic())

    def stop(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
