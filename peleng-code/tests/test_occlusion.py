"""Сопровождение машины при окклюзии: полная видимость, частичная, потеря всеми камерами.

Пять камер вокруг дороги, силуэт — проекция 3D-ящика (как пятно фона). Проверяется:
положение и единый трек, вес частичного силуэта, исключение пустых камер,
неопределённость (σ) и её рост без измерений.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.background import Blob
from app.vehicle import anchor_point, ctrv_step, predicted_sides
from app.vehicle_fusion import CameraView, FusionConfig, VehicleFusion
from test_vehicle import MODEL, camera  # noqa: E402

FIVE = [
    camera("cam-1", (40.0, 0.0, 6.0)),
    camera("cam-2", (0.0, 40.0, 6.0)),
    camera("cam-3", (-30.0, -30.0, 8.0)),
    camera("cam-4", (-40.0, 10.0, 5.0)),
    camera("cam-5", (20.0, -35.0, 7.0)),
]
DT = 1.0 / 30.0
START = np.array([-6.0, 1.0, 0.0, -0.2, 0.0, 6.0, 0.05])


def silhouette(state: np.ndarray, pose, visible: float = 1.0) -> list[Blob]:
    """Силуэт ящика; visible < 1 — правая часть закрыта деревом."""
    sides = predicted_sides(state, MODEL, pose)
    if sides is None:
        return []
    x0, y0, x1, y1 = sides
    x1 = x0 + (x1 - x0) * visible
    return [Blob(x0, y0, x1 - x0, y1 - y0, 0.7 * (x1 - x0) * (y1 - y0))]


def run(frames: int, scene) -> tuple[VehicleFusion, list, list]:
    """scene(frame, state, pose) -> пятна камеры. Возвращает фильтр, виды и истину."""
    fusion = VehicleFusion(MODEL, FusionConfig(init_min_area_px=50.0, range_m=(1.0, 500.0)))
    state = START.copy()
    views, truth = [], []
    for frame in range(frames):
        truth.append(anchor_point(state, MODEL))
        cameras = {pose.camera_id: CameraView(pose, scene(frame, state, pose)) for pose in FIVE}
        views.append(fusion.step(frame * DT, cameras, key=frame))
        state = ctrv_step(state, DT, yaw_tau_s=0.0)
    return fusion, views, truth


def error(view, truth) -> float:
    return float(np.linalg.norm(view.anchor - truth))


class FullVisibilityTest(unittest.TestCase):
    def test_all_five_cameras_fused(self):
        fusion, views, truth = run(90, lambda frame, state, pose: silhouette(state, pose))
        tail = views[30:]
        self.assertTrue(all(view.active and view.fresh for view in tail))
        self.assertTrue(all(len(view.observed) == 5 for view in tail))
        self.assertLess(max(error(view, truth[30 + index]) for index, view in enumerate(tail)), 0.3)
        self.assertLess(max(view.sigma_m for view in tail), 0.2)
        self.assertEqual(len(fusion.segments), 1)


class PartialOcclusionTest(unittest.TestCase):
    def test_two_cameras_lost_one_half_hidden(self):
        # С кадра 40: cam-4 и cam-5 ничего не видят, у cam-1 правая половина силуэта за деревом.
        def scene(frame, state, pose):
            if frame < 40:
                return silhouette(state, pose)
            if pose.camera_id in ("cam-4", "cam-5"):
                return []
            return silhouette(state, pose, 0.5 if pose.camera_id == "cam-1" else 1.0)

        fusion, views, truth = run(100, scene)
        occluded = views[50:]
        self.assertTrue(all(view.active and view.fresh for view in occluded))
        self.assertTrue(all(set(view.observed) <= {"cam-1", "cam-2", "cam-3"} for view in occluded))
        # Частичный силуэт не тянет ящик: положение держится по оставшимся камерам.
        self.assertLess(max(error(view, truth[50 + index]) for index, view in enumerate(occluded)), 0.5)
        # Меньше камер — шире неопределённость, но трек тот же.
        self.assertGreater(np.median([v.sigma_m for v in occluded]), np.median([v.sigma_m for v in views[20:40]]))
        self.assertEqual(len(fusion.segments), 1)


class TotalLossTest(unittest.TestCase):
    def test_coasts_with_growing_uncertainty_and_keeps_identity(self):
        # Кадры 50–69: ни одна камера не видит цель. Потом цель снова видна.
        def scene(frame, state, pose):
            return [] if 50 <= frame < 70 else silhouette(state, pose)

        fusion, views, truth = run(100, scene)
        lost = views[50:70]
        self.assertTrue(all(view.active and not view.fresh for view in lost))
        self.assertTrue(all(not view.observed for view in lost))
        sigmas = [view.sigma_m for view in lost]
        self.assertTrue(all(later > earlier for earlier, later in zip(sigmas, sigmas[1:])))
        self.assertGreater(sigmas[-1], 3.0 * views[49].sigma_m)
        # Прогноз CTRV за 20 кадров без измерений уходит недалеко.
        self.assertLess(error(lost[-1], truth[69]), 1.0)
        back = views[75:]
        self.assertTrue(all(view.fresh for view in back))
        self.assertLess(max(error(view, truth[75 + index]) for index, view in enumerate(back)), 0.3)
        # Тот же трек, без перезапуска.
        self.assertEqual(len(fusion.segments), 1)


class SingleCameraTest(unittest.TestCase):
    def test_one_camera_does_not_teach_speed(self):
        # С кадра 40 цель видит только cam-1, и силуэт обрезан деревом: по такому измерению
        # дальность не задана. С hold_motion_single скорость остаётся прежней.
        def scene(frame, state, pose):
            if frame < 40:
                return silhouette(state, pose)
            return silhouette(state, pose, 0.6) if pose.camera_id == "cam-1" else []

        def speeds(hold: bool) -> list[float]:
            fusion = VehicleFusion(
                MODEL, FusionConfig(init_min_area_px=50.0, range_m=(1.0, 500.0), hold_motion_single=hold)
            )
            state = START.copy()
            out = []
            for frame in range(70):
                cameras = {pose.camera_id: CameraView(pose, scene(frame, state, pose)) for pose in FIVE}
                view = fusion.step(frame * DT, cameras, key=frame)
                if frame >= 45:
                    self.assertEqual(sorted(view.observed), ["cam-1"])
                    out.append(view.speed_m_s)
                state = ctrv_step(state, DT, yaw_tau_s=0.0)
            return out

        held = speeds(True)
        self.assertLess(max(abs(speed - START[5]) for speed in held), 0.3)


if __name__ == "__main__":
    unittest.main()
