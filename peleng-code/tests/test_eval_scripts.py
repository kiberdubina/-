"""Проверки check_trail.py и _eval_hold_trail.py работают без исходных роликов участника."""

import contextlib
import hashlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import _eval_hold_trail as evaluation
import _synthetic_clips as clips
import check_trail

REPORT = ROOT / "_eval_hold_trail_report.json"


def _digest(path: Path) -> str | None:
    return hashlib.sha1(path.read_bytes()).hexdigest() if path.is_file() else None


class SyntheticClipTest(unittest.TestCase):
    def test_truth_matches_the_colour_ground_truth(self):
        import cv2

        with tempfile.TemporaryDirectory() as folder:
            for kind, make in clips.KINDS.items():
                clip = make(Path(folder), frames=12)
                capture = cv2.VideoCapture(str(clip.path))
                for index in range(12):
                    ok, image = capture.read()
                    self.assertTrue(ok)
                    found = evaluation.GT_FN[kind](image)
                    self.assertIsNotNone(found, f"{kind}: кадр {index}")
                    self.assertLess(abs(found[0] - clip.truth[index][0]) + abs(found[1] - clip.truth[index][1]), 6.0)
                capture.release()


class CheckTrailTest(unittest.TestCase):
    def test_gate_does_not_depend_on_active_profile(self):
        check_trail.check_gate()

    def test_runs_without_the_original_video(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            check_trail.main([])
        self.assertIn("след без выброса", out.getvalue())


class EvalHoldTrailTest(unittest.TestCase):
    def test_synthetic_scenes_pass_with_their_profiles(self):
        with tempfile.TemporaryDirectory() as folder:
            for kind, profile, frames in (("green", "green_ball", 190), ("sky_dark", "sky_uav", 120),
                                          ("white_drone", "white_drone", 120)):
                clip = clips.KINDS[kind](Path(folder), frames=frames)
                row = evaluation.evaluate(str(clip.path), profile, kind)
                self.assertTrue(row["PASS"], row)
                self.assertLess(row["median_err_px"], 3.0, row)

    def test_no_videos_is_an_error_and_keeps_the_report(self):
        if any((ROOT / case["video"]).is_file() for case in evaluation.CASES):
            self.skipTest("исходные ролики на месте")
        before = _digest(REPORT)
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                evaluation.main(["--no-synthetic"])
        self.assertEqual(caught.exception.code, 2)
        self.assertEqual(_digest(REPORT), before)


if __name__ == "__main__":
    unittest.main()
