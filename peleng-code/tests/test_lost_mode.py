"""Режим LOST: ушедшая за кадр цель не подменяется мелким пятном (облако на truck_new_env)."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.motion import Detection
from app.profile import load_patterns
from app.track2d import LOST_S, LOST_STATE, MultiTrack2D

DT = 1.0 / 30.0
TRUCK_AREA = 20000.0


def _spot(x: float, y: float, area: float) -> Detection:
    side = int(area ** 0.5)
    return Detection(x, y, area, (int(x) - side // 2, int(y) - side // 2, side, side))


def _lost_truck() -> tuple[MultiTrack2D, int]:
    """Фура едет вправо и уходит за кадр; кадры без пятен, пока трек не отпустит её."""
    multi = MultiTrack2D(load_patterns(ROOT / "config.yaml")["way1"])
    for frame in range(30):
        view = multi.update(_spot(400 + frame * 20, 600, TRUCK_AREA), frame * DT)
    assert view.active
    frame = 30
    while multi.update([], frame * DT).active:
        frame += 1
    return multi, frame + 1


class LostModeTest(unittest.TestCase):
    def test_small_blob_does_not_replace_lost_target(self):
        multi, frame = _lost_truck()
        end = frame + int((LOST_S + 3.0) / DT)
        states = set()
        for index in range(frame, end):
            view = multi.update([_spot(1500 - (index - frame) * 0.5, 4, 200.0)], index * DT)
            self.assertFalse(view.active, f"облако стало целью на кадре {index}")
            states.add(view.state)
        self.assertEqual(states, {LOST_STATE, "поиск"})

    def test_similar_size_target_is_reacquired_while_lost(self):
        multi, frame = _lost_truck()
        views = [
            multi.update([_spot(300 + step * 20, 600, TRUCK_AREA * 0.8)], (frame + step) * DT)
            for step in range(5)
        ]
        self.assertTrue(views[-1].active)

    def test_large_target_is_acquired_after_lost(self):
        multi, frame = _lost_truck()
        start = frame + int((LOST_S + 0.5) / DT)
        self.assertEqual(multi.update([], start * DT).state, "поиск")
        views = [
            multi.update([_spot(300 + step * 20, 600, TRUCK_AREA * 0.15)], (start + 1 + step) * DT)
            for step in range(5)
        ]
        self.assertTrue(views[-1].active)

    def test_without_history_small_blob_is_acquired(self):
        multi = MultiTrack2D(load_patterns(ROOT / "config.yaml")["way1"])
        views = [multi.update([_spot(500 + step * 5, 4, 200.0)], step * DT) for step in range(5)]
        self.assertTrue(views[-1].active)


if __name__ == "__main__":
    unittest.main()
