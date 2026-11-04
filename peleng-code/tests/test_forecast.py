import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.profile import load_patterns
from app.session import vehicle_forecast
from app.track3d import FORECAST_POINTS, Track3D
from app.vehicle import MotionNoise, VehicleModel, anchor_point, ctrv_step


def _model() -> VehicleModel:
    return VehicleModel.from_config(load_patterns(ROOT / "config.yaml")["way1"].vehicle)


class VehicleForecastTest(unittest.TestCase):
    def test_straight_motion_is_a_line_and_ends_at_old_forecast(self):
        model, motion = _model(), MotionNoise()
        state = np.array([10.0, 5.0, 3.0, 0.3, 0.0, 14.0, 0.0])
        path = vehicle_forecast(state, model, motion, 1.0)
        self.assertEqual(path.shape, (FORECAST_POINTS, 3))
        expected = anchor_point(ctrv_step(state, 1.0, motion.yaw_tau_s, motion.pitch_tau_s), model)
        self.assertTrue(np.allclose(path[-1], expected))
        # Без поворота все точки на одной прямой: отклонение от хорды ~0.
        start = anchor_point(state, model)
        chord = (path[-1] - start) / np.linalg.norm(path[-1] - start)
        offsets = [np.linalg.norm(np.cross(point - start, chord)) for point in path]
        self.assertLess(max(offsets), 1e-6)
        # Шаг вдоль пути ≈ v·dt.
        self.assertAlmostEqual(np.linalg.norm(path[0] - start), 14.0 * 1.0 / FORECAST_POINTS, places=2)

    def test_turning_motion_bends(self):
        model, motion = _model(), MotionNoise()
        state = np.array([0.0, 0.0, 3.0, 0.0, 0.0, 14.0, 0.5])
        path = vehicle_forecast(state, model, motion, 1.0)
        start = anchor_point(state, model)
        chord = (path[-1] - start) / np.linalg.norm(path[-1] - start)
        offsets = [np.linalg.norm(np.cross(point - start, chord)) for point in path]
        self.assertGreater(max(offsets), 0.05)


class PointForecastTest(unittest.TestCase):
    def test_constant_velocity_path(self):
        profile = load_patterns(ROOT / "config.yaml")["motion"]
        profile.predict_s = 0.5
        track = Track3D(profile)
        track.update(np.array([0.0, 0.0, 1.0]), 0.0, {})
        track.update(np.array([0.1, 0.0, 1.0]), 0.1, {})
        world = track.update(np.array([0.2, 0.0, 1.0]), 0.2, {})
        self.assertEqual(world.path.shape, (FORECAST_POINTS, 3))
        self.assertTrue(np.allclose(world.path[-1], world.pred))
        steps = np.diff(np.vstack([world.xyz, world.path]), axis=0)
        self.assertTrue(np.allclose(steps, steps[0]))


if __name__ == "__main__":
    unittest.main()
