import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.motion import Detection
from app.profile import load_patterns
from app.track2d import Track2D


def _spot(x: float, y: float) -> Detection:
    return Detection(x, y, 2000.0, (int(x) - 5, int(y) - 5, 10, 10))


def _profile():
    profile = load_patterns(ROOT / "config.yaml")["green_ball"]
    profile.predict_s = 0.5
    return profile


def _distance_to_polyline(point: tuple[float, float], corners: list[tuple[float, float]]) -> float:
    best = float("inf")
    px, py = point
    for (x0, y0), (x1, y1) in zip(corners, corners[1:]):
        vx, vy = x1 - x0, y1 - y0
        length = vx * vx + vy * vy
        if length <= 1e-9:
            dist = math.hypot(px - x0, py - y0)
        else:
            t = max(0.0, min(1.0, ((px - x0) * vx + (py - y0) * vy) / length))
            dist = math.hypot(px - (x0 + vx * t), py - (y0 + vy * t))
        best = min(best, dist)
    return best


class Track2DTest(unittest.TestCase):
    def test_trail_follows_the_corner(self):
        tracker = Track2D(_profile())
        corners = [(0.0, 100.0), (200.0, 100.0), (200.0, 300.0)]
        points = [(x, 100.0) for x in range(0, 201, 10)]
        points += [(200.0, y) for y in range(110, 301, 10)]
        last = None
        for index, (x, y) in enumerate(points):
            last = tracker.update(_spot(x, y), index * 0.05)
        self.assertIsNotNone(last)
        distances = [_distance_to_polyline(point, corners) for point in last.trail]
        ordered = sorted(distances)
        median = ordered[len(ordered) // 2]
        self.assertLess(median, 8.0)
        self.assertGreaterEqual(len(last.trail), 10)
        true = (200.0, 300.0 + 200.0 * 0.5)
        self.assertLess(math.hypot(last.pred_cx - true[0], last.pred_cy - true[1]), 40.0)

    def test_jitter_under_3px_is_not_a_new_trail_point(self):
        tracker = Track2D(_profile())
        tracker.update(_spot(50, 50), 0.0)
        for index, (x, y) in enumerate(((51, 50), (50.5, 51), (51, 51)), start=1):
            view = tracker.update(_spot(x, y), index * 0.04)
        self.assertEqual(len(view.trail), 1)

    def test_four_hundred_pixel_jump_stays_out_of_the_trail(self):
        tracker = Track2D(_profile())
        for index in range(10):
            tracker.update(_spot(100 + index * 8, 200), index * 0.04)
        before = tracker.update(_spot(100 + 10 * 8, 200), 10 * 0.04)
        jumped = tracker.update(_spot(before.cx, before.cy + 400), 11 * 0.04)
        self.assertLess(abs(jumped.cy - before.cy), 140)
        self.assertTrue(all(abs(y - 200) < 140 for _x, y in jumped.trail))


if __name__ == "__main__":
    unittest.main()
