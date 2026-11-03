import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.motion import MotionDetector
from app.profile import load_patterns


class HandMotionTest(unittest.TestCase):
    def test_moving_rectangle_becomes_a_spot_within_eight_frames(self):
        profile = load_patterns(ROOT / "config.yaml")["motion"]
        self.assertIsNone(profile.color)
        self.assertFalse(profile.require_motion)
        self.assertFalse(profile.moving_camera)
        detector = MotionDetector(profile)
        height, width = 360, 640
        background = np.full((height, width, 3), 128, np.uint8)
        x = 40
        found = None
        for index in range(8):
            frame = background.copy()
            if index >= 2:
                cv2.rectangle(frame, (x, 140), (x + 80, 220), (32, 32, 32), -1)
                x += 70
            spots = detector.detect(frame)
            if not spots:
                continue
            found = index
            self.assertGreater(max(spot.area for spot in spots), 400)
            break
        self.assertIsNotNone(found)
        self.assertLessEqual(found, 7)


if __name__ == "__main__":
    unittest.main()
