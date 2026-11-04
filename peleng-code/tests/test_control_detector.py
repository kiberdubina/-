import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.control_run import CameraClip, ControlConfig, camera_blobs, detector_blobs
from app.geom import CameraPose
from app.object_detector import Box


class FakeTruckModel:
    """Заглушка YOLO: одна рамка «truck» в долях поданного кадра."""

    device = "cpu"

    def __init__(self):
        self.calls = 0

    def predict(self, image, conf, classes):
        self.calls += 1
        height, width = image.shape[:2]
        return [Box(0.25 * width, 0.5 * height, 0.75 * width, 0.75 * height, 0.9, "truck")]


def _pose(width: int, height: int) -> CameraPose:
    K = np.array([[100.0, 0, width / 2], [0, 100.0, height / 2], [0, 0, 1]])
    return CameraPose("cam-1", K, np.zeros(5), np.zeros((3, 1)), np.array([[0.0], [0.0], [10.0]]), width, height)


class ControlDetectorTest(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.folder)
        self.video = self.folder / "cam.mp4"
        sink = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
        for _ in range(3):
            sink.write(np.full((48, 64, 3), 120, np.uint8))
        sink.release()
        self.weights = self.folder / "truck.pt"
        self.weights.write_bytes(b"weights")
        pose = _pose(64, 48)
        self.clip = CameraClip("cam-1", self.video, 64, 48, 3, 30.0, lambda frame: pose)
        self.config = ControlConfig(detector_weights=str(self.weights), detector_mode="detector")

    def test_boxes_become_blobs_and_are_cached(self):
        model = FakeTruckModel()
        with mock.patch("tempfile.gettempdir", return_value=str(self.folder)), \
                mock.patch("app.object_detector.load_model", return_value=model):
            first = detector_blobs(self.clip, self.config)
            second = detector_blobs(self.clip, self.config)
        self.assertEqual(sorted(first), [0, 1, 2])
        blob = first[0][0]
        self.assertEqual((blob.x, blob.y, blob.w, blob.h), (16.0, 24.0, 32.0, 12.0))
        self.assertEqual(model.calls, 3)  # второй вызов — из кэша
        self.assertEqual([(b.x, b.y, b.w, b.h) for b in second[2]], [(16.0, 24.0, 32.0, 12.0)])

    def test_modes(self):
        with mock.patch("tempfile.gettempdir", return_value=str(self.folder)), \
                mock.patch("app.object_detector.load_model", return_value=FakeTruckModel()):
            only = camera_blobs(self.clip, self.config, model=None)
            self.config.detector_mode = "both"
            both = camera_blobs(self.clip, self.config, model=None)
        self.assertEqual(len(only[1]), 1)
        self.assertGreaterEqual(len(both[1]), 1)
        self.assertIn((16.0, 24.0, 32.0, 12.0), [(b.x, b.y, b.w, b.h) for b in both[1]])

    def test_apply_detector(self):
        from app.control_run import apply_detector

        config = ControlConfig()
        self.assertIn("не задан", apply_detector(config, {}))
        self.assertIsNone(config.detector_weights)
        self.assertIn("нет весов", apply_detector(config, {"detector": str(self.folder / "нет.pt")}))
        self.assertIsNone(config.detector_weights)
        note = apply_detector(config, {"detector": str(self.weights)})
        self.assertEqual(config.detector_weights, str(self.weights))
        self.assertEqual(config.detector_mode, "both")
        self.assertIn("truck.pt", note)
        apply_detector(config, {"detector": str(self.weights)}, mode="detector")
        self.assertEqual(config.detector_mode, "detector")

    def test_moving_camera_uses_detector_only(self):
        from app.control_run import clip_translates

        def moving(frame):
            pose = _pose(64, 48)
            # Центр камеры уезжает на 0,1 м за кадр: tvec = −R·C, R = I.
            return CameraPose("cam-1", pose.K, pose.dist, pose.rvec, np.array([[-0.1 * frame], [0.0], [10.0]]), 64, 48)

        self.assertFalse(clip_translates(self.clip))
        clip = CameraClip("cam-1", self.video, 64, 48, 30, 30.0, moving)
        self.assertTrue(clip_translates(clip))
        self.config.detector_mode = "both"
        with mock.patch("app.control_run.detector_blobs", return_value={0: []}) as detected, \
                mock.patch("app.control_run.RecordingBackground") as background:
            result = camera_blobs(clip, self.config, model=None)
        detected.assert_called_once()
        background.assert_not_called()
        self.assertEqual(result, {0: []})

    def test_without_weights_background_only(self):
        config = ControlConfig()
        with mock.patch("app.control_run.detector_blobs") as called:
            camera_blobs(self.clip, config, model=None)
        called.assert_not_called()


if __name__ == "__main__":
    unittest.main()
