import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.cine_table import pose_at
from app.pipeline import Step
from app.profile import load_profile
from app.session import Session
from app.source import Frame, SyntheticSource
from app.track2d import TrackView


def _look(pose) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(pose.rvec)
    return (rotation.T @ np.array([0.0, 0.0, 1.0])).ravel()


class _IndexedSource:
    def __init__(self, index: int, camera_id: str = "cam-5"):
        self.index = index
        self.camera_id = camera_id


class _PipelineStub:
    def __init__(self, source: _IndexedSource):
        self.source = source


def _empty_track() -> TrackView:
    return TrackView(
        active=False,
        cx=0.0,
        cy=0.0,
        vx=0.0,
        vy=0.0,
        speed_px_s=0.0,
        state="поиск",
        in_gate=False,
        measured=False,
        pred_cx=0.0,
        pred_cy=0.0,
        bbox=None,
        pred_bbox=None,
        area=None,
    )


class SessionPoseTest(unittest.TestCase):
    def test_set_sources_clears_pose_root(self):
        profile = load_profile(ROOT / "config.yaml")
        session = Session(profile, SyntheticSource())
        session.set_pose_source(ROOT)
        session.set_sources([SyntheticSource()])
        self.assertIsNone(session.pose_root)

    def test_cam5_pose_follows_frame_index(self):
        profile = load_profile(ROOT / "config.yaml")
        session = Session(profile, SyntheticSource())
        session.poses["cam-5"] = pose_at(ROOT, "cam-5", 0, 1920, 1080)
        session.set_pose_source(ROOT)

        image = np.zeros((1080, 1920, 3), np.uint8)
        step = Step(Frame(image, 0.0, "cam-5"), None, _empty_track())

        session._refresh_moving_pose(_PipelineStub(_IndexedSource(1)), step)
        look_start = _look(session.poses["cam-5"])

        session._refresh_moving_pose(_PipelineStub(_IndexedSource(501)), step)
        look_mid = _look(session.poses["cam-5"])
        expected_mid = _look(pose_at(ROOT, "cam-5", 500, 1920, 1080))
        self.assertLess(float(np.linalg.norm(look_mid - expected_mid)), 1e-6)

        session._refresh_moving_pose(_PipelineStub(_IndexedSource(1002)), step)
        look_end = _look(session.poses["cam-5"])

        expected_start = _look(pose_at(ROOT, "cam-5", 0, 1920, 1080))
        expected_end = _look(pose_at(ROOT, "cam-5", 1001, 1920, 1080))
        self.assertLess(float(np.linalg.norm(look_start - expected_start)), 1e-6)
        self.assertLess(float(np.linalg.norm(look_end - expected_end)), 1e-6)
        self.assertGreater(float(np.linalg.norm(look_start - look_end)), 0.2)

    def test_static_camera_id_not_refreshed(self):
        profile = load_profile(ROOT / "config.yaml")
        session = Session(profile, SyntheticSource())
        fixed = pose_at(ROOT, "cam-1", 0, 1920, 1080)
        session.poses["cam-1"] = fixed
        session.set_pose_source(ROOT)

        image = np.zeros((1080, 1920, 3), np.uint8)
        step = Step(Frame(image, 0.0, "cam-1"), None, _empty_track())
        session._refresh_moving_pose(_PipelineStub(_IndexedSource(501)), step)
        self.assertIs(session.poses["cam-1"], fixed)


if __name__ == "__main__":
    unittest.main()
