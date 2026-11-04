"""Разворот ящика, который долго едет задом (перепутан перед), в итоговой траектории."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.control_run import orient_forward
from app.vehicle import PSI, V, VehicleModel, box_center, state_from_center

MODEL = VehicleModel(front_m=14.0, rear_m=-5.5, width_m=5.0, height_m=6.4, anchor_m=(15.4, 0.0, 3.0))


def _state(x: float, psi: float, v: float) -> np.ndarray:
    return state_from_center(np.array([x, 0.0, 3.2]), psi, MODEL, v=v)


class OrientForwardTest(unittest.TestCase):
    def test_long_reverse_run_is_flipped(self):
        # Ящик смотрит на запад, а едет на восток: v < 0 весь участок.
        states = {frame: _state(0.5 * frame, np.pi, -15.0) for frame in range(30)}
        fixed = orient_forward(states, MODEL, 30.0)
        for frame in range(30):
            self.assertAlmostEqual(abs(np.cos(fixed[frame][PSI])), 1.0, places=6)
            self.assertGreater(np.cos(fixed[frame][PSI]), 0.0)
            self.assertGreater(fixed[frame][V], 0.0)
            np.testing.assert_allclose(box_center(fixed[frame], MODEL), box_center(states[frame], MODEL), atol=1e-9)

    def test_short_blip_and_forward_stay(self):
        states = {frame: _state(0.5 * frame, 0.0, 15.0) for frame in range(30)}
        for frame in range(10, 14):  # 4 кадра < 0,3 с при 30 к/с
            states[frame] = _state(0.5 * frame, 0.0, -3.0)
        fixed = orient_forward(states, MODEL, 30.0)
        for frame in range(30):
            np.testing.assert_allclose(fixed[frame], states[frame])


if __name__ == "__main__":
    unittest.main()
