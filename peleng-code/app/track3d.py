from dataclasses import dataclass, field

import cv2
import numpy as np

from app.geom import CameraPose, range_from_cameras, velocity_in_camera
from app.profile import TargetProfile


FORECAST_POINTS = 10


@dataclass
class WorldView:
    active: bool
    xyz: np.ndarray | None
    speed_m_s: float
    pred: np.ndarray | None
    relative: dict[str, np.ndarray] = field(default_factory=dict)
    fresh: bool = False
    # Прогноз траектории: FORECAST_POINTS точек от «сейчас» до горизонта профиля (последняя = pred).
    path: np.ndarray | None = None


class Track3D:
    """Положение и скорость в метрах. Прогноз — то же состояние на горизонт профиля."""

    def __init__(self, profile: TargetProfile):
        self.profile = profile
        self.kf = cv2.KalmanFilter(6, 3)
        self.kf.measurementMatrix = np.hstack(
            [np.eye(3, dtype=np.float32), np.zeros((3, 3), np.float32)]
        )
        # На дальней трассе измерения шумнее — доверяем им меньше, чем на столе.
        meas = 0.002 if self.profile.v_max <= 20.0 else 2.0
        self.kf.processNoiseCov = np.diag([1e-4, 1e-4, 1e-4, 0.05, 0.05, 0.05]).astype(
            np.float32
        )
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * float(meas)
        self.kf.errorCovPost = np.eye(6, dtype=np.float32)
        self.ready = False
        self.miss = 0
        self.last_t = 0.0

    def reset(self) -> None:
        self.ready = False
        self.miss = 0
        self.kf.errorCovPost = np.eye(6, dtype=np.float32)

    def update(
        self,
        xyz: np.ndarray | None,
        time_s: float,
        poses: dict[str, CameraPose],
    ) -> WorldView:
        if not self.ready:
            if xyz is None:
                return WorldView(False, None, 0.0, None)
            self.kf.statePost = np.array(
                [[xyz[0]], [xyz[1]], [xyz[2]], [0], [0], [0]], np.float32
            )
            self.ready = True
            self.last_t = time_s
            self.miss = 0
            return self._view(True, poses)

        dt = max(1e-3, time_s - self.last_t)
        self.last_t = time_s
        transition = np.eye(6, dtype=np.float32)
        transition[0, 3] = dt
        transition[1, 4] = dt
        transition[2, 5] = dt
        self.kf.transitionMatrix = transition
        self.kf.predict()
        fresh = False
        if xyz is not None and self._accept(xyz, dt, poses):
            self.kf.correct(np.array(xyz, np.float32).reshape(3, 1))
            self.miss = 0
            fresh = True
        else:
            self.miss += 1
            # Долгий пропуск: трек «засыпает», но состояние и след снаружи не стираем.
            if self.miss > 90:
                self.ready = False
                return WorldView(False, None, 0.0, None, {}, False)
        return self._view(fresh, poses)

    def _accept(
        self,
        xyz: np.ndarray,
        dt: float,
        poses: dict[str, CameraPose] | None = None,
    ) -> bool:
        predicted = self.kf.statePre[:3, 0]
        jump = float(np.linalg.norm(xyz - predicted))
        # После пропуска ворота шире: прогноз уехал, шум триангуляции на дальних камерах больше.
        jump_cap = max(
            0.15,
            self.profile.v_max * dt * (1.0 + float(self.miss)),
            0.05 * self.profile.v_max,
        )
        if jump > jump_cap:
            return False
        distance = range_from_cameras(xyz, poses or {})
        return self.profile.r_min <= distance <= self.profile.r_max

    def _view(self, fresh: bool, poses: dict[str, CameraPose] | None = None) -> WorldView:
        state = self.kf.statePost.ravel()
        xyz = state[:3].astype(np.float64)
        velocity = state[3:].astype(np.float64)
        speed = float(np.linalg.norm(velocity))
        pred = xyz + velocity * self.profile.predict_s
        steps = np.linspace(1.0, FORECAST_POINTS, FORECAST_POINTS) / FORECAST_POINTS * self.profile.predict_s
        path = xyz + steps[:, None] * velocity
        relative = {}
        if poses:
            for camera_id, pose in poses.items():
                relative[camera_id] = velocity_in_camera(velocity, pose)
        return WorldView(True, xyz, speed, pred, relative, fresh, path)
