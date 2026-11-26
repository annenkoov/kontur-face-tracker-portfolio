"""Background camera inference; never touches Tk widgets."""

from __future__ import annotations

from dataclasses import dataclass
import queue
import threading
import time

from .camera import CameraSource, FrameResult
from .vision import FaceAnalyzer, FaceObservation


class LatestFrameSlot:
    """One-slot handoff: new camera frames replace any frame not yet analyzed."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._sequence = 0
        self._frame_result = None
        self._error: BaseException | None = None
        self._closed = False

    def publish(self, frame_result) -> None:
        with self._condition:
            if self._closed:
                return
            self._sequence += 1
            self._frame_result = frame_result
            self._condition.notify_all()

    def close(self, error: BaseException | None = None) -> None:
        with self._condition:
            self._closed = True
            if error is not None:
                self._error = error
            self._condition.notify_all()

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed

    def next_after(self, sequence: int, timeout: float = 0.25):
        with self._condition:
            self._condition.wait_for(
                lambda: self._sequence > sequence or self._closed, timeout=max(0.0, timeout)
            )
            if self._error is not None:
                raise self._error
            if self._sequence > sequence:
                return self._sequence, self._frame_result
            return None


@dataclass(frozen=True)
class WorkerMessage:
    kind: str
    generation: int = 0
    frame: object | None = None
    observation: FaceObservation | None = None
    error: str | None = None


class AnalysisWorker:
    def __init__(self, messages: queue.Queue, camera_index: int = 0,
                 camera_factory=CameraSource, analyzer_factory=FaceAnalyzer, generation: int = 0):
        self.messages = messages
        self.camera_index = camera_index
        self.camera_factory = camera_factory
        self.analyzer_factory = analyzer_factory
        self.generation = generation
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._capture_thread: threading.Thread | None = None
        self._preview_lock = threading.Lock()
        self._latest_preview: FrameResult | None = None

    @property
    def latest_preview(self) -> FrameResult | None:
        """Последний захваченный кадр, доступный UI независимо от анализа модели."""
        with self._preview_lock:
            return self._latest_preview

    @property
    def running(self) -> bool:
        return bool((self._thread is not None and self._thread.is_alive()) or
                    (self._capture_thread is not None and self._capture_thread.is_alive()))

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._thread = None
        self._capture_thread = None
        self._thread = threading.Thread(target=self._run, name="face-analysis", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> bool:
        self._stop.set()
        deadline = time.monotonic() + max(0.0, timeout)
        if self._thread is not None:
            self._thread.join(max(0.0, deadline - time.monotonic()))
            if self._thread.is_alive():
                return False
            self._thread = None
        if self._capture_thread is not None:
            self._capture_thread.join(max(0.0, deadline - time.monotonic()))
            if self._capture_thread.is_alive():
                return False
            self._capture_thread = None
        return True

    def _publish(self, message: WorkerMessage) -> None:
        try:
            self.messages.put_nowait(message)
        except queue.Full:
            try:
                self.messages.get_nowait()
            except queue.Empty:
                pass
            try:
                self.messages.put_nowait(message)
            except queue.Full:
                pass

    def _run(self) -> None:
        camera = None
        analyzer = None
        capture_thread = None
        latest_frame = LatestFrameSlot()
        try:
            camera = self.camera_factory(self.camera_index)
            camera.start()
            # UI может сразу сообщить, что камера открылась, даже если загрузка
            # модели или первый инференс займут заметное время.
            self._publish(WorkerMessage("camera_ready", generation=self.generation))

            def capture_loop() -> None:
                capture_error = None
                try:
                    while not self._stop.is_set():
                        frame_result = camera.read()
                        # Предпросмотр всегда получает самый свежий кадр прямо
                        # от потока захвата и не ждёт очереди обработки модели.
                        with self._preview_lock:
                            self._latest_preview = frame_result
                        latest_frame.publish(frame_result)
                except Exception as exc:
                    capture_error = exc
                try:
                    # The reader thread owns release, so release never races with read().
                    camera.stop()
                except Exception as exc:
                    capture_error = RuntimeError(f"Не удалось корректно освободить камеру: {exc}")
                latest_frame.close(capture_error)

            capture_thread = threading.Thread(target=capture_loop, name="camera-capture", daemon=True)
            self._capture_thread = capture_thread
            capture_thread.start()
            analyzer = self.analyzer_factory()
            self._publish(WorkerMessage("ready", generation=self.generation))
            sequence = 0
            while not self._stop.is_set():
                next_frame = latest_frame.next_after(sequence)
                if next_frame is None:
                    if latest_frame.closed:
                        break
                    continue
                sequence, frame_result = next_frame
                observation = analyzer.analyze(frame_result.frame, frame_result.captured_at,
                                               frame_result.captured_monotonic)
                self._publish(WorkerMessage("frame", self.generation, frame_result.frame, observation))
        except Exception as exc:
            self._publish(WorkerMessage("error", generation=self.generation, error=str(exc)))
        finally:
            self._stop.set()
            latest_frame.close()
            if capture_thread is not None:
                capture_thread.join(timeout=1.0)
            elif camera is not None:
                try:
                    camera.stop()
                except Exception as exc:
                    self._publish(WorkerMessage("error", generation=self.generation,
                                                error=f"Не удалось корректно освободить камеру: {exc}"))
            if analyzer is not None:
                try:
                    analyzer.close()
                except Exception:
                    pass
            if capture_thread is not None and capture_thread.is_alive():
                self._publish(WorkerMessage(
                    "error", generation=self.generation,
                    error="Чтение камеры ещё не завершилось; повторный запуск останется заблокирован до освобождения камеры",
                ))
            else:
                self._publish(WorkerMessage("stopped", generation=self.generation))
