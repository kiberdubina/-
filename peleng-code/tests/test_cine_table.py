import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.cine_table import pose_at, relative_layout, track_meters
from app.geom import camera_center


def _look(pose) -> np.ndarray:
    """Ось взгляда в мире: OpenCV смотрит по +Z, это R.T @ [0, 0, 1]."""
    rotation, _ = cv2.Rodrigues(pose.rvec)
    return (rotation.T @ np.array([0.0, 0.0, 1.0])).ravel()


class CineTableTest(unittest.TestCase):
    def test_track_has_1002_finite_points(self):
        track = track_meters(ROOT)
        self.assertEqual(track.shape, (1002, 3))
        self.assertEqual(track.dtype, np.float64)
        self.assertTrue(np.isfinite(track).all())

    def test_track_span_is_about_407_m(self):
        track = track_meters(ROOT)
        span = float(np.linalg.norm(track[-1] - track[0]))
        self.assertAlmostEqual(span, 407.0, delta=5.0)

    def test_static_cameras_keep_centers(self):
        first = camera_center(pose_at(ROOT, "cam-1", 0, 1920, 1080))
        later = camera_center(pose_at(ROOT, "cam-1", 500, 1920, 1080))
        self.assertLess(float(np.linalg.norm(first - later)), 1e-3)
        second_now = camera_center(pose_at(ROOT, "cam-2", 0, 1920, 1080))
        second_later = camera_center(pose_at(ROOT, "cam-2", 500, 1920, 1080))
        self.assertLess(float(np.linalg.norm(second_now - second_later)), 1e-3)
        distance = float(np.linalg.norm(first - second_now))
        self.assertAlmostEqual(distance, 257.0, delta=2.0)
        layout = relative_layout(ROOT)
        self.assertAlmostEqual(layout["distance_m"]["cam-1|cam-2"], distance, delta=1e-6)

    def test_camera_five_turns_in_place(self):
        start = pose_at(ROOT, "cam-5", 0, 1920, 1080)
        end = pose_at(ROOT, "cam-5", 1001, 1920, 1080)
        self.assertLess(
            float(np.linalg.norm(camera_center(start) - camera_center(end))),
            1e-3,
        )
        # Рысканье не двигает R @ [0,0,1]: у cam-5 постоянны крен и тангаж.
        # Взгляд в мире — третья строка R.
        gap = float(np.linalg.norm(_look(start) - _look(end)))
        self.assertGreater(gap, 0.2)

    def test_focal_length_in_pixels(self):
        pose = pose_at(ROOT, "cam-1", 0, 1920, 1080)
        fx = 35.0 / 23.76 * 1920.0
        fy = 35.0 / 13.365 * 1080.0
        self.assertLess(abs(float(pose.K[0, 0]) - fx) / fx, 0.01)
        self.assertLess(abs(float(pose.K[1, 1]) - fy) / fy, 0.01)
        self.assertAlmostEqual(float(pose.K[0, 2]), 960.0)
        self.assertAlmostEqual(float(pose.K[1, 2]), 540.0)

    def test_track_is_in_front_of_camera_one(self):
        track = track_meters(ROOT)
        pose = pose_at(ROOT, "cam-1", 0, 1920, 1080)
        rotation, _ = cv2.Rodrigues(pose.rvec)
        camera = rotation @ track[0].reshape(3, 1) + pose.tvec.reshape(3, 1)
        self.assertGreater(float(camera[2, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
