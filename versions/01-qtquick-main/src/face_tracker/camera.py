"""OpenCV camera access, isolated from the Tkinter event loop."""

from __future__ import annotations

from dataclasses import dataclass
import os
import time

import cv2


class CameraError(RuntimeError):
    """Raised when a camera cannot be opened or stops returning frames."""


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


def discover_cameras(max_index: int = 8) -> list[dict[str, object]]:
    """Probe common OpenCV camera indices; callers should run this off the UI thread."""
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    devices: list[dict[str, object]] = []
    for index in range(max(1, int(max_index))):
        capture = cv2.VideoCapture(index, backend)
        try:
            if not capture.isOpened():
                continue
            ok, frame = capture.read()
            if ok and frame is not None:
                devices.append({"index": index, "label": f"Камера {len(devices) + 1}"})
        except cv2.error:
            continue
        finally:
            capture.release()
    return devices
