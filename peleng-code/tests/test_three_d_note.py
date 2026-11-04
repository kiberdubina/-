"""Подпись способа 3D совпадает с развилкой Session.step: ящик машины, триангуляция или 2D."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.geom import CameraPose, look_at
from app.main import three_d_note, truck_detector_note
from app.profile import load_patterns
from app.session import Session
from app.source import BlankSource

PATTERNS = load_patterns(ROOT / "config.yaml")


def _pose(camera_id: str, x: float) -> CameraPose:
    rvec, tvec = look_at(np.array([x, -50.0, 10.0]), np.zeros(3))
    matrix = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]])
    return CameraPose(camera_id, matrix, np.zeros(5), rvec, tvec, 1920, 1080)


def _session(profile_name: str, cameras: int) -> Session:
    session = Session(PATTERNS[profile_name], BlankSource())
    session.poses = {f"cam-{i + 1}": _pose(f"cam-{i + 1}", 20.0 * i) for i in range(cameras)}
    return session


class ThreeDNoteTest(unittest.TestCase):
    def test_way1_is_box_fit_not_triangulation(self):
        session = _session("way1", 5)
        self.assertTrue(session.box_mode)
        note = three_d_note(session)
        self.assertIn("ящик машины", note)
        self.assertIn("5 камер", note)
        self.assertIn("пятна фона и рамки детектора", note)
        self.assertNotIn("триангуляция", note)

    def test_truck_detector_profile_uses_detector_boxes_only(self):
        session = _session("truck_detector", 2)
        self.assertTrue(session.box_mode)
        self.assertIn("(рамки детектора грузовика)", three_d_note(session))

    def test_point_target_is_triangulated(self):
        session = _session("green_ball", 2)
        self.assertFalse(session.box_mode)
        self.assertIn("триангуляция", three_d_note(session))

    def test_one_pose_is_2d_even_for_vehicle(self):
        session = _session("way1", 1)
        self.assertFalse(session.box_mode)
        self.assertIn("Только 2D", three_d_note(session))

    def test_truck_detector_note_reports_the_detector_that_runs(self):
        session = _session("way1", 2)
        self.assertIsNone(truck_detector_note(session))
        spec = SimpleNamespace(model="/x/models/truck_world.pt", conf=0.25, det_width=960)
        session._truck_detectors = {
            "cam-1": SimpleNamespace(spec=spec, device="mps", ms=24.6),
            "cam-2": SimpleNamespace(spec=spec, device="mps", ms=31.2),
            "cam-3": False,
        }
        note = truck_detector_note(session)
        self.assertIn("truck_world.pt", note)
        self.assertIn("mps", note)
        self.assertIn("31 мс", note)
        self.assertIn("порог 0.25", note)


if __name__ == "__main__":
    unittest.main()
