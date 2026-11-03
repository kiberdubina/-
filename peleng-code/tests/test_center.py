import math
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.motion import MotionDetector
from app.profile import load_patterns


class CenterTest(unittest.TestCase):
    def test_growing_rectangle_reports_its_box_center(self):
        profile = load_patterns(ROOT / "config.yaml")["green_ball"]
        profile.s_min = 10
        detector = MotionDetector(profile)
        for index in range(12):
            cx = 70.0 + index * 5.0
            cy = 48.0 + index * 3.0
            rect_w = 16 + index * 4
            rect_h = 12 + index * 2
            x = int(cx - rect_w / 2)
            y = int(cy - rect_h / 2)
            mask = np.zeros((180, 280), np.uint8)
            cv2.rectangle(
                mask,
                (x, y),
                (x + rect_w - 1, y + rect_h - 1),
                255,
                thickness=-1,
            )
            spots = detector._spots(mask)
            self.assertEqual(len(spots), 1)
            error = math.hypot(spots[0].cx - cx, spots[0].cy - cy)
            self.assertLessEqual(error, 2.0)


if __name__ == "__main__":
    unittest.main()
