"""Пакетный detect_batch: те же рамки и то же состояние детекторов, что у detect по камерам."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.object_detector import Box, ObjectDetector, detect_batch, parse_spec
from app.profile import TargetProfile


def _profile() -> TargetProfile:
    return TargetProfile("t", 0.0, 1e12, 0.0, 1e9, 0.0, 1e9, 0.0, 1e9, None, 0.5, None)


class BatchModel:
    """Заглушка с predict и predict_many: рамка зависит от яркости кадра, чтобы камеры различались."""

    device = "cpu"

    def __init__(self):
        self.single = 0
        self.batches: list[int] = []

    def _boxes(self, image):
        height, width = image.shape[:2]
        level = float(image[0, 0, 0]) / 255.0
        return [Box(0.1 * width, level * 0.5 * height, 0.4 * width, 0.9 * height, 0.8, "truck")]

    def predict(self, image, conf, classes):
        self.single += 1
        return self._boxes(image)

    def predict_many(self, images, conf, classes):
        self.batches.append(len(images))
        return [self._boxes(image) for image in images]


def _spec(**extra):
    raw = {"mode": "object", "class": "truck", "conf": 0.25, "det_every": 1, "det_width": 640, "roi_scale": 0}
    raw.update(extra)
    return parse_spec(raw)


def _frames(levels, shape=(1080, 1920)):
    return [np.full((*shape, 3), level, np.uint8) for level in levels]


class DetectBatchTest(unittest.TestCase):
    def test_batch_matches_per_camera_detect(self):
        frames = _frames([20, 120, 220])
        model = BatchModel()
        alone = [ObjectDetector(_profile(), _spec(), model=model).detect(frame) for frame in frames]
        together = detect_batch([(ObjectDetector(_profile(), _spec(), model=model), frame) for frame in frames])
        self.assertEqual(together, alone)
        self.assertEqual(model.batches, [3])

    def test_single_item_uses_plain_predict(self):
        model = BatchModel()
        detector = ObjectDetector(_profile(), _spec(), model=model)
        detect_batch([(detector, _frames([50])[0])])
        self.assertEqual((model.single, model.batches), (1, []))

    def test_different_frame_sizes_are_not_mixed(self):
        # Разный размер — разный letterbox у ultralytics: такие кадры идут отдельными прогонами.
        model = BatchModel()
        frames = _frames([20, 120]) + _frames([200], shape=(720, 1280))
        detectors = [ObjectDetector(_profile(), _spec(), model=model) for _ in frames]
        found = detect_batch(list(zip(detectors, frames)))
        self.assertEqual(model.batches, [2])
        self.assertEqual(model.single, 1)
        self.assertTrue(all(len(item) == 1 for item in found))

    def test_skipped_frames_return_empty_and_keep_ticks(self):
        model = BatchModel()
        detectors = [ObjectDetector(_profile(), _spec(det_every=2), model=model) for _ in range(2)]
        frames = _frames([60, 160])
        first = detect_batch(list(zip(detectors, frames)))
        second = detect_batch(list(zip(detectors, frames)))
        self.assertTrue(all(first) and not any(second))
        self.assertTrue(all(item.skipped for item in detectors))
        self.assertEqual(model.batches, [2])

    def test_state_is_per_detector(self):
        model = BatchModel()
        detectors = [ObjectDetector(_profile(), _spec(), model=model) for _ in range(2)]
        detect_batch(list(zip(detectors, _frames([20, 220]))))
        tops = [item.labels[0].detection.bbox[1] for item in detectors]
        self.assertLess(tops[0], tops[1])
        self.assertTrue(all(item.calls == 1 and item.ms >= 0.0 for item in detectors))

    def test_roi_miss_falls_back_to_full_frame(self):
        model = BatchModel()
        detector = ObjectDetector(_profile(), _spec(roi_scale=2.5), model=model)
        detector.model = type("Empty", (), {"device": "cpu", "predict": lambda self, *a: []})()
        detector.set_hint((100.0, 100.0, 50.0, 50.0))
        self.assertEqual(detect_batch([(detector, _frames([90])[0])]), [[]])
        self.assertTrue(detector._state.roi_failed)
        detector.model = model
        self.assertEqual(len(detector.detect(_frames([90])[0])), 1)
        self.assertFalse(detector._state.roi_failed)


if __name__ == "__main__":
    unittest.main()
