from dataclasses import dataclass, field

import cv2
import numpy as np

from app.geom import CameraPose, velocity_in_camera
from app.profile import TargetProfile


@dataclass
class WorldView:
    active: bool
    xyz: np.ndarray | None
    speed_m_s: float
    pred: np.ndarray | None
    relative: dict[str, np.ndarray] = field(default_factory=dict)
    fresh: bool = False


class Track3D:
    """Положение и скорость в метрах. Прогноз — то же состояние на горизонт профиля."""

    def __init__(self, profile: TargetProfile):
        self.profile = profile
        self.kf = cv2.KalmanFilter(6, 3)
        self.kf.measurementMatrix = np.hstack(
            [np.eye(3, dtype=np.float32), np.zeros((3, 3), np.float32)]
        )
        self.kf.processNoiseCov = np.diag([1e-4, 1e-4, 1e-4, 0.05, 0.05, 0.05]).astype(
            np.float32
        )
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 0.002
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
            return self._view(True)

        dt = max(1e-3, time_s - self.last_t)
        self.last_t = time_s
        transition = np.eye(6, dtype=np.float32)
        transition[0, 3] = dt
        transition[1, 4] = dt
        transition[2, 5] = dt
        self.kf.transitionMatrix = transition
        self.kf.predict()
        fresh = False
        if xyz is not None and self._accept(xyz, dt):
            self.kf.correct(np.array(xyz, np.float32).reshape(3, 1))
            self.miss = 0
            fresh = True
        else:
            self.miss += 1
            if self.miss > 12:
                self.reset()
                return WorldView(False, None, 0.0, None)
        return self._view(fresh, poses)

    def _accept(self, xyz: np.ndarray, dt: float) -> bool:
        predicted = self.kf.statePre[:3, 0]
        jump = float(np.linalg.norm(xyz - predicted))
        if jump > max(0.15, self.profile.v_max * dt):
            return False
        distance = float(np.linalg.norm(xyz))
        return self.profile.r_min <= distance <= self.profile.r_max

    def _view(self, fresh: bool, poses: dict[str, CameraPose] | None = None) -> WorldView:
        state = self.kf.statePost.ravel()
        xyz = state[:3].astype(np.float64)
        velocity = state[3:].astype(np.float64)
        speed = float(np.linalg.norm(velocity))
        pred = xyz + velocity * self.profile.predict_s
        relative = {}
        if poses:
            for camera_id, pose in poses.items():
                relative[camera_id] = velocity_in_camera(velocity, pose)
        return WorldView(True, xyz, speed, pred, relative, fresh)
