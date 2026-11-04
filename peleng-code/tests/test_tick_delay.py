"""Пауза между кадрами окна: время работы кадра вычитается из периода, а не прибавляется к нему."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.main import MIN_TICK_GAP_MS, next_tick_delay


class TickDelayTest(unittest.TestCase):
    def test_fast_frame_waits_rest_of_period(self):
        self.assertEqual(next_tick_delay(33, 10.0), 23)

    def test_idle_frame_waits_full_period(self):
        self.assertEqual(next_tick_delay(33, 0.0), 33)

    def test_slow_frame_does_not_add_full_period(self):
        # Детекция 5 камер ~130 мс: раньше к ней добавлялись ещё 33 мс.
        self.assertEqual(next_tick_delay(33, 130.0), MIN_TICK_GAP_MS)

    def test_gap_never_below_minimum(self):
        self.assertEqual(next_tick_delay(30, 29.9), MIN_TICK_GAP_MS)
        self.assertGreater(MIN_TICK_GAP_MS, 0)


if __name__ == "__main__":
    unittest.main()
