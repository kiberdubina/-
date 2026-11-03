import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.motion import Detection
from app.profile import load_patterns
from app.track2d import MultiTrack2D, Track2D


def _spot(x: float, y: float, area: float = 2000.0) -> Detection:
    return Detection(x, y, area, (int(x) - 5, int(y) - 5, 10, 10))


def _profile():
    profile = load_patterns(ROOT / "config.yaml")["green_ball"]
    profile.predict_s = 0.5
    return profile


class MultiTrack2DTest(unittest.TestCase):
    def test_far_blob_does_not_hijack_primary(self):
        multi = MultiTrack2D(_profile())
        for index in range(8):
            multi.update(_spot(100 + index * 10, 200), index * 0.04)
        before = multi.update(_spot(180, 200), 8 * 0.04)
        near = _spot(188, 200, area=22000)
        far = _spot(530, 80, area=80000)
        view = multi.update([far, near], 9 * 0.04)
        self.assertIs(multi.accepted, near)
        self.assertTrue(view.measured)
        self.assertLess(abs(view.cx - near.cx), 5.0)

        jumped = multi.update([_spot(before.cx + 400, 200, area=50000)], 10 * 0.04)
        self.assertIsNone(multi.accepted)
        self.assertFalse(jumped.measured)
        self.assertLess(abs(jumped.cy - before.cy), 140)

    def test_primary_claims_gate_before_secondary(self):
        multi = MultiTrack2D(_profile())
        multi.update(_spot(100, 100), 0.0)
        for index in range(1, 6):
            multi.update(_spot(100 + index * 8, 100), index * 0.04)
        primary_x = multi.update(_spot(140, 100), 6 * 0.04).cx
        # Второе пятно рядом с главным — не должно стать основным треком.
        view = multi.update([_spot(148, 100), _spot(400, 400, area=5000)], 7 * 0.04)
        self.assertLess(abs(view.cx - primary_x), 20.0)

    def test_two_separate_targets_get_two_tracks(self):
        multi = MultiTrack2D(_profile())
        multi.update([_spot(80, 80)], 0.0)
        for index in range(1, 5):
            multi.update([_spot(80 + index * 10, 80), _spot(300, 300)], index * 0.04)
        views = multi.tracks
        self.assertGreaterEqual(len(views), 2)
        xs = sorted(view.cx for view in views[:2])
        self.assertLess(xs[0], 150.0)
        self.assertGreater(xs[1], 250.0)

    def test_single_track_matches_track2d_gate(self):
        profile = _profile()
        single = Track2D(profile)
        multi = MultiTrack2D(profile)
        for index in range(10):
            spot = _spot(100 + index * 8, 200)
            t = index * 0.04
            v1 = single.update(spot, t)
            v2 = multi.update(spot, t)
        before_single = v1.cx
        before_multi = v2.cx
        far_only = [_spot(before_single, before_single + 400, area=50000)]
        jumped_single = single.update(far_only, 10 * 0.04)
        jumped_multi = multi.update(far_only, 10 * 0.04)
        self.assertFalse(jumped_single.measured)
        self.assertFalse(jumped_multi.measured)
        self.assertLess(abs(jumped_single.cy - before_single), 140)
        self.assertLess(abs(jumped_multi.cy - before_multi), 140)


if __name__ == "__main__":
    unittest.main()
