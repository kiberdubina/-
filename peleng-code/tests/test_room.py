import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.main import TILE_H, TILE_W, advance_zoom, camera_layout, fit_camera_tile, grid_shape
from app.motion import MotionDetector
from app.profile import load_patterns
from app.track2d import Track2D


def _scene(index: int) -> tuple[np.ndarray, tuple[float, float]]:
    """Комната: светлый идёт, тёмный диван стоит, чёрная футболка идёт."""
    image = np.full((360, 640, 3), (190, 185, 180), np.uint8)
    cv2.rectangle(image, (40, 200), (180, 330), (25, 25, 25), -1)
    light_x = 220 + index * 14
    cv2.rectangle(image, (light_x, 80), (light_x + 36, 170), (230, 225, 220), -1)
    dark_x = 80 + index * 16
    dark_y = 60
    cv2.rectangle(image, (dark_x, dark_y), (dark_x + 40, dark_y + 90), (20, 20, 20), -1)
    return image, (dark_x + 20.0, dark_y + 45.0)


class RoomTest(unittest.TestCase):
    def test_only_the_black_shirt_is_followed(self):
        profile = load_patterns(ROOT / "config.yaml")["black_shirt"]
        profile.predict_s = 0.5
        detector = MotionDetector(profile)
        tracker = Track2D(profile)
        hits = []
        for index in range(16):
            image, truth = _scene(index)
            spots = detector.detect(image)
            view = tracker.update(spots, index * 0.04)
            if index < 6 or not view.active or view.area is None:
                continue
            hits.append((view.cx, view.cy, truth))
            self.assertLess(abs(view.cx - truth[0]), 30.0)
            self.assertLess(abs(view.cy - truth[1]), 30.0)
        self.assertGreaterEqual(len(hits), 6)
        last = tracker.update(detector.detect(_scene(16)[0]), 16 * 0.04)
        self.assertGreater(last.pred_cx, last.cx)

    def test_zoom_follows_and_holds_through_a_short_gap(self):
        state = {"cx": 320.0, "cy": 180.0, "zoom": 1.0, "miss": 0.0}
        for _ in range(8):
            advance_zoom(state, True, 100.0, 80.0, 80.0, 640.0)
        self.assertGreater(state["zoom"], 1.5)
        self.assertLess(state["cx"], 200.0)
        held = state["zoom"]
        advance_zoom(state, False, state["cx"], state["cy"], 0.0, 640.0)
        self.assertAlmostEqual(state["zoom"], held)
        for _ in range(10):
            advance_zoom(state, False, state["cx"], state["cy"], 0.0, 640.0)
        self.assertLess(state["zoom"], held)

    def test_two_cameras_sit_side_by_side(self):
        self.assertEqual(grid_shape(1), (1, 1))
        self.assertEqual(grid_shape(2), (2, 1))
        self.assertEqual(grid_shape(4), (2, 2))
        cols, rows = grid_shape(6)
        self.assertEqual(cols, 3)
        self.assertGreaterEqual(cols * rows, 6)

    def test_camera_tile_keeps_one_size(self):
        wide = np.zeros((360, 640, 3), np.uint8)
        tall = np.zeros((800, 400, 3), np.uint8)
        small = np.zeros((100, 120, 3), np.uint8)
        for image in (wide, tall, small):
            tile = fit_camera_tile(image)
            self.assertEqual(tile.shape, (TILE_H, TILE_W, 3))

    def test_wider_trajectory_shrinks_cameras(self):
        wide_w, wide_h, _gap = camera_layout(2, 1400, 700)
        narrow_w, narrow_h, _gap = camera_layout(2, 700, 700)
        self.assertGreater(wide_w, TILE_W)
        self.assertLess(narrow_w, TILE_W)
        self.assertLess(narrow_w, wide_w)
        self.assertLess(narrow_h, wide_h)

    def test_smaller_trajectory_leaves_cameras_some_slack(self):
        area_w, area_h = 1400, 700
        tile_w, tile_h, gap = camera_layout(2, area_w, area_h)
        cols, rows = grid_shape(2)
        grid_w = cols * tile_w + cols * 2 * gap
        grid_h = rows * tile_h + rows * 2 * gap
        self.assertGreater(tile_w, TILE_W)
        self.assertLess(grid_w, area_w * 0.92)
        self.assertLess(grid_h, area_h * 0.92)
        self.assertGreater(gap, 8)


if __name__ == "__main__":
    unittest.main()
