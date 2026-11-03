import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.geom import CameraPose, default_intrinsics, look_at
from app.vehicle import (
    OMEGA,
    PSI,
    STATE_DIM,
    THETA,
    V,
    BoxObservation,
    SideNoise,
    VehicleModel,
    VehicleTracker,
    anchor_point,
    box_center,
    clipped_bbox,
    ctrv_step,
    flipped,
    predicted_sides,
    robust_update,
    side_weight,
    state_from_center,
    wrap_angle,
)

MODEL = VehicleModel(front_m=4.0, rear_m=-1.0, width_m=2.0, height_m=1.6, anchor_m=(4.5, 0.0, 0.8))


def state(x=0.0, y=0.0, z=0.0, psi=0.0, theta=0.0, v=0.0, omega=0.0) -> np.ndarray:
    return np.array([x, y, z, psi, theta, v, omega], np.float64)


def camera(camera_id: str, center, target=(0.0, 0.0, 0.8), size=(1280, 720)) -> CameraPose:
    matrix, dist = default_intrinsics(*size)
    rvec, tvec = look_at(np.asarray(center, float), np.asarray(target, float))
    return CameraPose(camera_id, matrix, dist, rvec, tvec, size[0], size[1])


RIG = [
    camera("a", (40.0, 0.0, 6.0)),
    camera("b", (0.0, 40.0, 6.0)),
    camera("c", (-30.0, -30.0, 8.0)),
]


def observe(truth: np.ndarray, poses=RIG, noise_px: float = 0.0, rng=None) -> list[BoxObservation]:
    observations = []
    for pose in poses:
        sides = predicted_sides(truth, MODEL, pose)
        if sides is None:
            continue
        if noise_px:
            sides = sides + rng.normal(0.0, noise_px, 4)
        observations.append(BoxObservation(pose, (sides[0], sides[1], sides[2] - sides[0], sides[3] - sides[1])))
    return observations


class GeometryTest(unittest.TestCase):
    def test_anchor_and_center_follow_heading(self) -> None:
        current = state(1.0, 2.0, 0.5, psi=np.pi / 2)
        np.testing.assert_allclose(anchor_point(current, MODEL), [1.0, 6.5, 1.3], atol=1e-12)
        np.testing.assert_allclose(box_center(current, MODEL), [1.0, 3.5, 1.3], atol=1e-12)

    def test_pitch_lifts_the_front(self) -> None:
        climbing = state(theta=np.radians(15.0))
        anchor = anchor_point(climbing, MODEL)
        expected_z = 4.5 * np.sin(np.radians(15.0)) + 0.8 * np.cos(np.radians(15.0))
        self.assertAlmostEqual(float(anchor[2]), expected_z, places=12)

    def test_state_from_center_inverts_box_center(self) -> None:
        center = np.array([10.0, -3.0, 2.0])
        current = state_from_center(center, 0.7, MODEL, theta=0.1)
        np.testing.assert_allclose(box_center(current, MODEL), center, atol=1e-12)

    def test_flip_keeps_box_and_is_involution(self) -> None:
        current = state(3.0, -1.0, 0.2, psi=0.4, theta=0.05, v=5.0, omega=0.1)
        turned = flipped(current, MODEL)
        np.testing.assert_allclose(box_center(turned, MODEL), box_center(current, MODEL), atol=1e-12)
        self.assertAlmostEqual(abs(wrap_angle(turned[PSI] - current[PSI])), np.pi, places=12)
        self.assertAlmostEqual(turned[THETA], -0.05)
        self.assertAlmostEqual(turned[V], -5.0)
        self.assertAlmostEqual(turned[OMEGA], 0.1)
        corners = np.sort(np.round(MODEL.local_corners(), 9), axis=0)
        self.assertEqual(corners.shape, (8, 3))
        back = flipped(turned, MODEL)
        np.testing.assert_allclose(back[:3], current[:3], atol=1e-12)
        self.assertAlmostEqual(wrap_angle(back[PSI] - current[PSI]), 0.0, places=12)
        self.assertAlmostEqual(back[THETA], current[THETA], places=12)

    def test_clipped_bbox_cuts_at_frame(self) -> None:
        pixels = np.array([[-50.0, 10.0], [100.0, 10.0], [100.0, 900.0], [-50.0, 900.0]])
        np.testing.assert_allclose(clipped_bbox(pixels, 640, 480), [0.0, 10.0, 100.0, 480.0], atol=1e-3)
        self.assertIsNone(clipped_bbox(pixels - 2000.0, 640, 480))

    def test_box_behind_camera_is_not_predicted(self) -> None:
        pose = camera("front", (0.0, -10.0, 1.0), (0.0, -20.0, 1.0))
        self.assertIsNone(predicted_sides(np.zeros(STATE_DIM), MODEL, pose))


class MotionTest(unittest.TestCase):
    def test_straight_line(self) -> None:
        np.testing.assert_allclose(ctrv_step(state(v=10.0), 2.0)[:3], [20.0, 0.0, 0.0], atol=1e-12)

    def test_climb_follows_pitch(self) -> None:
        moved = ctrv_step(state(theta=0.2, v=10.0), 1.0)
        np.testing.assert_allclose(moved[:3], [10 * np.cos(0.2), 0.0, 10 * np.sin(0.2)], atol=1e-12)

    def test_turn_keeps_radius(self) -> None:
        # Без затухания CTRV идёт по окружности радиуса v / omega.
        moved = state(v=10.0, omega=0.5)
        for _ in range(100):
            moved = ctrv_step(moved, 0.05, yaw_tau_s=0.0)
        center = np.array([0.0, 20.0])
        self.assertAlmostEqual(float(np.linalg.norm(moved[:2] - center)), 20.0, places=9)
        self.assertAlmostEqual(wrap_angle(moved[PSI] - 2.5), 0.0, places=9)


PRIOR = np.diag([4.0, 4.0, 4.0, 0.2, 0.05, 1.0, 0.1])


class UpdateTest(unittest.TestCase):
    def test_robust_update_recovers_pose_from_three_views(self) -> None:
        truth = state(2.0, -1.5, 0.0, psi=0.6)
        start = truth + state(1.2, -0.8, 0.4, psi=0.15, theta=0.05)
        result = robust_update(start, PRIOR, MODEL, observe(truth), iterations=30)
        self.assertIsNotNone(result)
        self.assertLess(np.linalg.norm(anchor_point(result.state, MODEL) - anchor_point(truth, MODEL)), 0.02)
        self.assertLess(result.rms_px, 0.1)

    def test_pitch_is_recovered_on_a_slope(self) -> None:
        truth = state(1.0, 0.5, 1.0, psi=-0.4, theta=np.radians(12.0))
        start = truth.copy()
        start[THETA] = 0.0
        result = robust_update(start, PRIOR, MODEL, observe(truth), iterations=30)
        self.assertAlmostEqual(float(result.state[THETA]), np.radians(12.0), delta=np.radians(1.0))
        self.assertLess(np.linalg.norm(anchor_point(result.state, MODEL) - anchor_point(truth, MODEL)), 0.05)

    def test_influence_inward_vanishes_outward_stays(self) -> None:
        # Влияние стороны = вес × отклонение. «Внутрь» (окклюзия) — убывает к нулю,
        # «наружу» (силуэт шире ящика) — не опускается ниже порога Хьюбера.
        noise = SideNoise()
        inward = [side_weight(-r, noise) * r for r in (20.0, 100.0, 1000.0)]
        outward = [side_weight(r, noise) * r for r in (20.0, 100.0, 1000.0)]
        self.assertGreater(inward[0], inward[1])
        self.assertGreater(inward[1], inward[2])
        self.assertLess(inward[2], 1.0)
        for value in outward:
            self.assertAlmostEqual(value, noise.huber_px)

    def test_heavy_occlusion_keeps_pose(self) -> None:
        truth = state(psi=0.3)
        observations = observe(truth)
        # Дерево закрыло 90% силуэта в первой камере: левая сторона ушла «внутрь» на ~100 px.
        x, y, w, h = observations[0].bbox
        observations[0] = BoxObservation(observations[0].pose, (x + 0.9 * w, y, 0.1 * w, h))
        prior = np.diag([1.0, 1.0, 1.0, 0.05, 0.01, 1.0, 0.1])
        result = robust_update(truth, prior, MODEL, observations, iterations=30)
        self.assertLess(np.linalg.norm(anchor_point(result.state, MODEL) - anchor_point(truth, MODEL)), 0.3)


def drive(frames: int, dt: float, start: np.ndarray) -> list[np.ndarray]:
    states = [start]
    for _ in range(frames - 1):
        states.append(ctrv_step(states[-1], dt, yaw_tau_s=0.0))
    return states


class TrackerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dt = 1.0 / 30.0
        self.truth = drive(90, self.dt, state(-6.0, -3.0, 0.0, psi=0.3, v=6.0, omega=0.15))
        self.rng = np.random.default_rng(7)

    def run_tracker(self, start: np.ndarray) -> VehicleTracker:
        tracker = VehicleTracker(MODEL)
        tracker.start(start, np.diag([1.0, 1.0, 1.0, 0.1, 0.01, 25.0, 0.25]), 0.0)
        for index, truth in enumerate(self.truth[1:], start=1):
            tracker.step(index * self.dt, observe(truth, noise_px=1.0, rng=self.rng))
        return tracker

    def test_tracks_and_smoother_is_not_worse(self) -> None:
        start = self.truth[0].copy()
        start[V] = 0.0
        start[OMEGA] = 0.0
        tracker = self.run_tracker(start)
        filtered = [anchor_point(r.filtered, MODEL) for r in tracker.history]
        smoothed = [anchor_point(s, MODEL) for s in tracker.smoothed()]
        truth = [anchor_point(s, MODEL) for s in self.truth]
        err_f = np.array([np.linalg.norm(a - b) for a, b in zip(filtered, truth)])[30:]
        err_s = np.array([np.linalg.norm(a - b) for a, b in zip(smoothed, truth)])[30:]
        self.assertLess(float(np.median(err_f)), 0.15)
        self.assertLessEqual(float(np.sqrt(np.mean(err_s**2))), float(np.sqrt(np.mean(err_f**2))) + 1e-9)
        self.assertAlmostEqual(float(tracker.state[V]), 6.0, delta=0.5)

    def test_backwards_start_is_flipped_by_motion(self) -> None:
        start = flipped(self.truth[0], MODEL)
        start[V] = 0.0
        start[OMEGA] = 0.0
        tracker = self.run_tracker(start)
        self.assertEqual(tracker.flips, 1)
        smoothed = tracker.smoothed()
        error = np.linalg.norm(anchor_point(smoothed[0], MODEL) - anchor_point(self.truth[0], MODEL))
        self.assertLess(error, 0.3)
        self.assertGreater(float(tracker.state[V]), 0.0)


if __name__ == "__main__":
    unittest.main()
