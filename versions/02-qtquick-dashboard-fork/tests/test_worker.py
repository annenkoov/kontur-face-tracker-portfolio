import queue
import threading
import time
import unittest

from face_tracker.camera import FrameResult
from face_tracker.worker import AnalysisWorker, LatestFrameSlot


class FakeCamera:
    def __init__(self, _index):
        self.stopped = False

    def start(self):
        pass

    def read(self):
        time.sleep(0.001)
        return FrameResult(object(), time.time(), time.monotonic())

    def stop(self):
        self.stopped = True


class StopFailCamera(FakeCamera):
    def stop(self):
        self.stopped = True
        raise RuntimeError("release failed")


class BlockingReadCamera:
    def __init__(self, entered, allow_return):
        self.entered = entered
        self.allow_return = allow_return
        self.stop_thread = None
        self.frame = object()

    def start(self):
        pass

    def read(self):
        self.entered.set()
        self.allow_return.wait()
        return FrameResult(self.frame, time.time(), time.monotonic())

    def stop(self):
        self.stop_thread = threading.current_thread().name


class BlockingAnalyzer:
    def __init__(self, release_event, entered_event=None):
        self.release_event = release_event
        self.entered_event = entered_event

    def analyze(self, _frame, _timestamp, _monotonic_timestamp=None):
        if self.entered_event:
            self.entered_event.set()
        self.release_event.wait(2)
        return None

    def close(self):
        pass


class FailingAnalyzer:
    def analyze(self, _frame, _timestamp, _monotonic_timestamp=None):
        raise RuntimeError("analysis failed")

    def close(self):
        pass


class WorkerTests(unittest.TestCase):
    def test_latest_frame_slot_discards_unprocessed_frames(self):
        slot = LatestFrameSlot()
        slot.publish("old")
        slot.publish("new")
        sequence, frame = slot.next_after(0)
        self.assertEqual(sequence, 2)
        self.assertEqual(frame, "new")

    def test_latest_frame_slot_propagates_capture_failure(self):
        slot = LatestFrameSlot()
        slot.close(RuntimeError("camera failed"))
        with self.assertRaisesRegex(RuntimeError, "camera failed"):
            slot.next_after(0)

    def test_blocked_capture_does_not_report_stopped_or_release_during_read(self):
        entered = threading.Event()
        allow_return = threading.Event()
        camera = BlockingReadCamera(entered, allow_return)
        messages = queue.Queue(maxsize=8)
        worker = AnalysisWorker(messages, camera_factory=lambda _index: camera,
                                analyzer_factory=lambda: BlockingAnalyzer(threading.Event()))
        worker.start()
        self.assertTrue(entered.wait(1))
        self.assertFalse(worker.stop(timeout=2))
        self.assertTrue(worker.running)
        self.assertIsNone(camera.stop_thread)
        received = []
        while not messages.empty():
            received.append(messages.get_nowait())
        self.assertFalse(any(message.kind == "stopped" for message in received))

        allow_return.set()
        deadline = time.monotonic() + 2
        while worker.running and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertFalse(worker.running)
        self.assertEqual(camera.stop_thread, "camera-capture")

    def test_stop_does_not_report_completion_while_worker_is_still_alive(self):
        release_event = threading.Event()
        worker = AnalysisWorker(queue.Queue(maxsize=2), camera_factory=FakeCamera,
                                analyzer_factory=lambda: BlockingAnalyzer(release_event))
        worker.start()
        time.sleep(0.05)
        self.assertFalse(worker.stop(timeout=0.01))
        self.assertTrue(worker.running)
        release_event.set()
        self.assertTrue(worker.stop(timeout=1))
        self.assertFalse(worker.running)

    def test_latest_preview_keeps_up_while_model_analysis_is_blocked(self):
        entered = threading.Event()
        release = threading.Event()
        worker = AnalysisWorker(
            queue.Queue(maxsize=8), camera_factory=FakeCamera,
            analyzer_factory=lambda: BlockingAnalyzer(release, entered),
        )
        worker.start()
        try:
            self.assertTrue(entered.wait(1))
            deadline = time.monotonic() + 1
            preview = None
            while preview is None and time.monotonic() < deadline:
                preview = worker.latest_preview
                time.sleep(0.005)
            self.assertIsNotNone(preview)
            # Захват работает отдельно: за время блокировки модели слот успевает
            # обновиться новыми кадрами, а не хранит только первый кадр.
            first_captured = preview.captured_monotonic
            time.sleep(0.03)
            self.assertGreater(worker.latest_preview.captured_monotonic, first_captured)
        finally:
            release.set()
            worker.stop(timeout=2)

    def test_camera_cleanup_error_does_not_skip_stopped_message(self):
        messages = queue.Queue(maxsize=8)
        worker = AnalysisWorker(messages, camera_factory=StopFailCamera,
                                analyzer_factory=FailingAnalyzer)
        worker.start()
        deadline = time.time() + 2
        received = []
        while time.time() < deadline and worker.running:
            try:
                received.append(messages.get(timeout=0.05))
            except queue.Empty:
                pass
        while not messages.empty():
            received.append(messages.get_nowait())
        self.assertTrue(any(message.kind == "stopped" for message in received))


if __name__ == "__main__":
    unittest.main()
