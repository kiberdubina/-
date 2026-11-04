import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import session as live
from app.profile import load_patterns
from app.session import Session
from app.source import BlankSource


class LiveTrailTest(unittest.TestCase):
    """След живого режима машины: что попадает в отчёт погрешности по кадрам."""

    def setUp(self):
        self.session = Session(load_patterns(ROOT / "config.yaml")["way1"], BlankSource())
        self.session._trail_reset()

    def accept(self, key, x, fresh=True, cameras=2):
        return self.session._trail_accept(key, np.array([x, 0.0, 0.0]), fresh, cameras)

    def warm(self):
        for key in range(live.TRAIL_WARMUP):
            self.assertFalse(self.accept(key, 0.5 * key))

    def test_warmup_points_are_skipped(self):
        self.warm()
        self.assertTrue(self.accept(live.TRAIL_WARMUP, 0.5 * live.TRAIL_WARMUP))

    def test_one_camera_point_is_not_written(self):
        self.warm()
        self.assertFalse(self.accept(100, 50.0, cameras=1))
        self.assertTrue(self.accept(101, 50.5, cameras=2))

    def test_jump_is_outlier_until_it_repeats(self):
        self.warm()
        self.assertTrue(self.accept(10, 5.0))
        jumps = [self.accept(11 + index, 40.0 + 0.5 * index) for index in range(live.TRAIL_REJECT_LIMIT)]
        # Скачок отброшен, пока не повторится TRAIL_REJECT_LIMIT раз: тогда это исправление трека.
        self.assertEqual(jumps, [False] * (live.TRAIL_REJECT_LIMIT - 1) + [True])

    def test_short_gap_written_from_prediction(self):
        self.warm()
        self.assertTrue(self.accept(10, 5.0))
        coast = [self.accept(11 + index, 5.5 + 0.5 * index, fresh=False, cameras=0) for index in range(live.TRAIL_COAST + 1)]
        self.assertEqual(coast, [True] * live.TRAIL_COAST + [False])


if __name__ == "__main__":
    unittest.main()
