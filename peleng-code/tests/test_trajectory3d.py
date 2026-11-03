import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.motion import Detection
from app.pipeline import Step
from app.profile import load_patterns
from app.rig import Rig
from app.session import Session
from app.source import Frame, SyntheticSource
from app.track2d import TrackView


def _rig_session():
    profile = load_patterns(ROOT / "config.yaml")["green_ball"]
    rig = Rig()
    session = Session(profile, SyntheticSource())
    session.set_rig(rig)
    return profile, rig, session


def _measured_step(camera_id: str, time_s: float, u: float, v: float) -> Step:
    frame = Frame(np.zeros((8, 8, 3), np.uint8), time_s, camera_id)
    detection = Detection(u, v, 1000.0, (0, 0, 10, 10))
    track = TrackView(
        True,
        u,
        v,
        0.0,
        0.0,
        0.0,
        "сопровождение",
        True,
        True,
        u,
        v,
        (0, 0, 10, 10),
        None,
        1000.0,
    )
    return Step(frame, detection, track)


def _pair(rig: Rig, point: np.ndarray, times: dict[str, float]) -> list[Step]:
    steps = []
    for camera_id, pose in rig.poses.items():
        projected, _ = cv2.projectPoints(
            np.asarray(point, np.float64).reshape(1, 1, 3),
            pose.rvec,
            pose.tvec,
            pose.K,
            pose.dist,
        )
        steps.append(
            _measured_step(
                camera_id,
                times[camera_id],
                float(projected[0, 0, 0]),
                float(projected[0, 0, 1]),
            )
        )
    return steps


def _frame_time(tick) -> float:
    return max(step.frame.time_s for step in tick.steps)


class _SilentSource:
    """Drops reads so one pipeline yields no step. Does not change Rig."""

    def __init__(self, inner, remain: int):
        self.inner = inner
        self.remain = remain
        self.rewound = False

    def read(self):
        self.rewound = False
        if self.remain > 0:
            self.remain -= 1
            return None
        frame = self.inner.read()
        self.rewound = bool(getattr(self.inner, "rewound", False))
        return frame


class Trajectory3DTest(unittest.TestCase):
    def test_mean_trajectory_error_stays_under_2cm(self):
        _profile, rig, session = _rig_session()
        errors = []
        for _ in range(40):
            tick = session.step()
            self.assertIsNotNone(tick)
            if not tick.world.fresh or tick.world.xyz is None:
                continue
            errors.append(float(np.linalg.norm(tick.world.xyz - rig.point_at(_frame_time(tick)))))
        self.assertGreaterEqual(len(errors), 30)
        mean = float(np.mean(errors))
        within = sum(error <= 0.05 for error in errors) / len(errors)
        self.assertLess(mean, 0.02, f"mean={mean:.4f} m")
        self.assertGreaterEqual(within, 0.90, f"within 5 cm: {within:.0%}")

    def test_silent_camera_does_not_jump_the_trail(self):
        _profile, rig, session = _rig_session()
        for _ in range(15):
            tick = session.step()
            self.assertIsNotNone(tick)
        self.assertGreaterEqual(len(session.trail3d), 10)
        held = len(session.trail3d)
        anchor = np.array(session.trail3d[-1], dtype=np.float64)

        follower = session.pipelines[1]
        self.assertEqual(follower.source.camera_id, "B")
        self.assertFalse(follower.source.leader)
        follower.source = _SilentSource(follower.source, 8)

        for _ in range(8):
            tick = session.step()
            self.assertIsNotNone(tick)
            self.assertEqual(len(tick.steps), 1)
            self.assertFalse(tick.world.fresh)
            self.assertEqual(len(session.trail3d), held)

        errors = []
        gaps = []
        previous = anchor
        for _ in range(20):
            tick = session.step()
            self.assertIsNotNone(tick)
            self.assertEqual(len(tick.steps), 2)
            if not tick.world.fresh or tick.world.xyz is None:
                continue
            errors.append(float(np.linalg.norm(tick.world.xyz - rig.point_at(_frame_time(tick)))))
            gaps.append(float(np.linalg.norm(tick.world.xyz - previous)))
            previous = np.array(tick.world.xyz, dtype=np.float64)
        self.assertGreaterEqual(len(errors), 10)
        self.assertEqual(len(session.trail3d) - held, len(errors))
        median = float(np.median(errors))
        self.assertLess(median, 0.02, f"median after recovery={median:.4f} m")
        # 8 missed ticks plus the next frame is 0.36 s. Peak rig speed is ~0.18 m/s,
        # so the trail may resume about 0.07 m ahead. 0.12 m still rejects a bad fix.
        self.assertLess(max(gaps), 0.12, f"trail step={max(gaps):.4f} m")

    def test_forecast_error_stays_under_5cm(self):
        profile, rig, session = _rig_session()
        # Horizon is 0.5 s. The rig path peaks near 0.18 m/s and 0.22 m/s², so a
        # constant-velocity forecast misses by about 0.5*a*T^2 <= 0.03 m. A frozen
        # forecast (zero velocity) misses by about v*T, typically ~0.06 m, and a
        # backwards one by about twice that. Median 0.05 m accepts the model error
        # and still fails those two.
        profile.predict_s = 0.5
        errors = []
        for index in range(40):
            tick = session.step()
            self.assertIsNotNone(tick)
            if index < 10 or tick.world.pred is None:
                continue
            future = rig.point_at(_frame_time(tick) + profile.predict_s)
            errors.append(float(np.linalg.norm(tick.world.pred - future)))
        self.assertGreaterEqual(len(errors), 20)
        median = float(np.median(errors))
        self.assertLess(median, 0.05, f"median forecast error={median:.4f} m")

    def test_views_200ms_apart_are_not_fresh(self):
        _profile, rig, session = _rig_session()
        point = rig.point_at(0.0)
        world, rays = session._fuse(_pair(rig, point, {"A": 0.0, "B": 0.08}))
        self.assertTrue(world.fresh)
        self.assertIsNotNone(world.xyz)
        self.assertEqual(len(rays), 2)

        world, rays = session._fuse(_pair(rig, point, {"A": 1.0, "B": 1.2}))
        self.assertFalse(world.fresh)
        self.assertEqual(rays, [])


if __name__ == "__main__":
    unittest.main()
