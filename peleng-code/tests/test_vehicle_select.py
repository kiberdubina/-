import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.geom import CameraPose
from app.main import tile_to_image
from app.motion import Detection
from app.pipeline import Step
from app.profile import load_patterns
from app.source import Frame
from app.track2d import MultiTrack2D


def _spot(cx: float, cy: float, size: float = 60.0) -> Detection:
    return Detection(cx, cy, size * size, (int(cx - size / 2), int(cy - size / 2), int(size), int(size)))


class TileToImageTest(unittest.TestCase):
    def test_letterbox_and_zoom(self):
        # Кадр 1920×1080 в окне 480×360: масштаб 0,25, картинка 480×270, сверху поле 45 px.
        self.assertEqual(tile_to_image(240, 180, 1920, 1080, 480, 360), (960.0, 540.0))
        self.assertEqual(tile_to_image(0, 45, 1920, 1080, 480, 360), (0.0, 0.0))
        self.assertIsNone(tile_to_image(240, 10, 1920, 1080, 480, 360))
        # Приближение: видна середина кадра 960×540 начиная с (480, 270).
        x, y = tile_to_image(240, 180, 1920, 1080, 480, 360, crop=(480, 270, 1440, 810))
        self.assertAlmostEqual(x, 960.0)
        self.assertAlmostEqual(y, 540.0)


class SelectTrackTest(unittest.TestCase):
    def setUp(self):
        self.tracker = MultiTrack2D(load_patterns(ROOT / "config.yaml")["traffic"])

    def _feed(self, frames, cars, start=0):
        view = None
        for index in range(frames):
            time_s = (start + index) / 30.0
            spots = [_spot(x + 4 * (start + index), y) for x, y in cars]
            view = self.tracker.update(spots, time_s)
        return view

    def test_selected_car_stays_primary(self):
        self._feed(3, [(300, 400), (900, 400)])
        self.assertTrue(self.tracker.select_at(912, 400))
        view = self._feed(5, [(300, 400), (900, 400)], start=3)
        self.assertTrue(self.tracker.has_target)
        self.assertAlmostEqual(view.cx, 900 + 4 * 7, delta=6)
        self.assertGreater(view.cx, 600)

    def test_click_on_empty_place_selects_nothing(self):
        self._feed(3, [(300, 400)])
        self.assertFalse(self.tracker.select_at(1500, 100))
        self.assertFalse(self.tracker.locked)

    def test_lost_car_is_not_replaced_by_neighbour_and_relocks(self):
        self._feed(3, [(300, 400), (900, 400)])
        self.tracker.select_at(912, 400)
        # Выбранная пропала (заехала за дом), соседняя осталась.
        view = None
        for index in range(20):
            view = self.tracker.update([_spot(300 + 4 * (3 + index), 400)], (3 + index) / 30.0)
        self.assertTrue(self.tracker.locked)
        self.assertFalse(self.tracker.has_target)
        self.assertFalse(view.active)
        self.assertIsNone(self.tracker.accepted)
        # Вернулась недалеко от места, где пропала, — снова она.
        view = self.tracker.update([_spot(400, 400), _spot(1000, 410)], 24 / 30.0)
        view = self.tracker.update([_spot(404, 400), _spot(1004, 410)], 25 / 30.0)
        self.assertTrue(self.tracker.has_target)
        self.assertGreater(view.cx, 900)

    def test_clear_selection_restores_old_behaviour(self):
        self._feed(3, [(300, 400), (900, 400)])
        self.tracker.select_at(912, 400)
        self.tracker.clear_selection()
        self.assertFalse(self.tracker.locked)
        self.assertTrue(self._feed(2, [(300, 400), (900, 400)], start=3).active)


def _pose(camera_id: str, center, target) -> CameraPose:
    """Камера в center смотрит на target, Z мира вверх."""
    center = np.asarray(center, np.float64)
    forward = np.asarray(target, np.float64) - center
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    rvec, _ = cv2.Rodrigues(rotation)
    K = np.array([[1000.0, 0, 960], [0, 1000.0, 540], [0, 0, 1]])
    return CameraPose(camera_id, K, np.zeros(5), rvec, (-rotation @ center).reshape(3, 1), 1920, 1080)


def _pixel(pose: CameraPose, xyz) -> tuple[float, float]:
    uv, _ = cv2.projectPoints(np.asarray(xyz, np.float64).reshape(1, 1, 3), pose.rvec, pose.tvec, pose.K, np.zeros(5))
    return float(uv[0, 0, 0]), float(uv[0, 0, 1])


class CrossCameraMatchTest(unittest.TestCase):
    def test_same_car_found_on_second_camera_by_geometry(self):
        from app.session import Session
        from app.source import SyntheticSource

        profile = load_patterns(ROOT / "config.yaml")["traffic"]
        session = Session(load_patterns(ROOT / "config.yaml")["motion"], SyntheticSource())
        session.profile = profile
        poses = {"A": _pose("A", (0, -40, 6), (0, 0, 0)), "B": _pose("B", (40, 0, 6), (0, 0, 0))}
        session.poses = dict(poses)
        trackers = {name: MultiTrack2D(profile) for name in poses}
        session.pipelines = [
            SimpleNamespace(
                source=SimpleNamespace(camera_id=name),
                tracker=tracker,
                select_at=tracker.select_at,
                clear_selection=tracker.clear_selection,
            )
            for name, tracker in trackers.items()
        ]
        # Две машины: выбранная в (0, 0) и другая в (3, 5). На камере B другая ближе
        # к центру кадра, но с выбранной сходится по геометрии только первая.
        cars = [np.array([0.0, 0.0, 0.8]), np.array([3.0, 5.0, 0.8])]

        def tick(index):
            steps = []
            for name, tracker in trackers.items():
                spots = [_spot(*_pixel(poses[name], car + [0.3 * index, 0, 0]), size=50) for car in cars]
                view = tracker.update(spots, index / 30.0)
                frame = Frame(np.zeros((4, 4, 3), np.uint8), index / 30.0, name, index)
                steps.append(Step(frame, tracker.accepted, view, list(tracker.tracks), None, tracker.has_target))
            return steps

        for index in range(3):
            tick(index)
        chosen = _pixel(poses["A"], cars[0] + [0.6, 0, 0])
        self.assertTrue(session.select_target("A", *chosen))
        # Другая машина почти на луче выбранной (невязка ~3 px): выбор не сразу, а после
        # накопления невязки по кадрам — и именно нужная машина.
        for index in range(3, 15):
            steps = tick(index)
            session._match_selection(steps)
        steps = tick(15)
        by_id = {step.frame.camera_id: step for step in steps}
        self.assertTrue(by_id["B"].selected)
        expected = _pixel(poses["B"], cars[0] + [4.5, 0, 0])
        self.assertLess(np.hypot(by_id["B"].track.cx - expected[0], by_id["B"].track.cy - expected[1]), 15)
        self.assertEqual(sorted(session.selection_cameras()), ["A", "B"])
        session.clear_target()
        self.assertEqual(session.selection_cameras(), [])


if __name__ == "__main__":
    unittest.main()
