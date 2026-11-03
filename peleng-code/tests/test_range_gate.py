import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.geom import CameraPose, camera_center, look_at, range_from_cameras
from app.track3d import Track3D
from app.profile import load_patterns


class RangeGateTest(unittest.TestCase):
    def test_range_is_to_nearest_camera_not_world_origin(self):
        K = np.eye(3, dtype=np.float64) * 700
        K[0, 2] = 320
        K[1, 2] = 240
        center = np.array([200.0, 0.0, 10.0])
        rvec, tvec = look_at(center, np.array([250.0, 0.0, 5.0]))
        pose = CameraPose("cam", K, np.zeros(5), rvec, tvec, 640, 480)
        xyz = np.array([250.0, 0.0, 5.0])
        # От нуля мира ~250 м, до камеры ~50 м.
        self.assertGreater(float(np.linalg.norm(xyz)), 100.0)
        self.assertLess(range_from_cameras(xyz, [pose]), 60.0)
        self.assertAlmostEqual(
            range_from_cameras(xyz, [pose]),
            float(np.linalg.norm(xyz - camera_center(pose))),
            places=5,
        )

    def test_way1_profile_accepts_far_point_near_camera(self):
        profile = load_patterns(ROOT / "config.yaml")["way1"]
        track = Track3D(profile)
        K = np.eye(3, dtype=np.float64) * 900
        K[0, 2] = 960
        K[1, 2] = 540
        cam = np.array([-247.0, -66.0, 12.0])
        target = cam + np.array([40.0, 10.0, -5.0])
        rvec, tvec = look_at(cam, target)
        pose = CameraPose("cam-1", K, np.zeros(5), rvec, tvec, 1920, 1080)
        view = track.update(target, 0.0, {"cam-1": pose})
        self.assertTrue(view.active)
        self.assertTrue(view.fresh)


if __name__ == "__main__":
    unittest.main()
