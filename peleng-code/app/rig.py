import cv2
import numpy as np

from app.geom import CameraPose, look_at
from app.source import Frame


class Rig:
    """Две камеры и мяч с известной траекторией. Проверка метров без телефонов."""

    def __init__(self, width: int = 640, height: int = 480, fps: float = 25):
        self.width = width
        self.height = height
        self.fps = fps
        self.index = 0
        self.time_s = 0.0
        self.truth: list[np.ndarray] = []
        focal = 700.0
        self.K = np.array(
            [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
            np.float64,
        )
        self.dist = np.zeros(5, np.float64)
        target = np.array([0.0, 0.0, 0.05])
        centers = {
            "A": np.array([-0.18, -0.55, 0.38]),
            "B": np.array([0.22, -0.55, 0.34]),
        }
        self.poses = {}
        for camera_id, center in centers.items():
            rvec, tvec = look_at(center, target)
            self.poses[camera_id] = CameraPose(
                camera_id, self.K.copy(), self.dist.copy(), rvec, tvec, width, height
            )
        rng = np.random.default_rng(1)
        self.background = rng.integers(28, 48, (height, width, 3), dtype=np.uint8)
        self.latest: dict[str, Frame] = {}

    @property
    def camera_ids(self) -> list[str]:
        return list(self.poses)

    def advance(self) -> None:
        self.time_s = self.index / self.fps
        self.index += 1
        point = self.point_at(self.time_s)
        self.truth.append(point.copy())
        for camera_id, pose in self.poses.items():
            self.latest[camera_id] = Frame(self._render(pose, point), self.time_s, camera_id)

    def point_at(self, time_s: float) -> np.ndarray:
        return np.array(
            [0.12 * np.sin(time_s * 1.3), 0.08 * np.cos(time_s * 1.1), 0.05],
            np.float64,
        )

    def _render(self, pose: CameraPose, point: np.ndarray) -> np.ndarray:
        image = self.background.copy()
        projected, _ = cv2.projectPoints(
            point.reshape(1, 1, 3), pose.rvec, pose.tvec, pose.K, pose.dist
        )
        u = int(round(float(projected[0, 0, 0])))
        v = int(round(float(projected[0, 0, 1])))
        rotation, _ = cv2.Rodrigues(pose.rvec)
        depth = float((rotation @ point.reshape(3, 1) + pose.tvec.reshape(3, 1))[2, 0])
        radius = 8 if depth <= 1e-3 else int(max(20, min(90, pose.K[0, 0] * 0.055 / depth)))
        if 0 <= u < self.width and 0 <= v < self.height:
            cv2.circle(image, (u, v), radius, (40, 180, 40), -1)
        return image


class RigCamera:
    def __init__(self, rig: Rig, camera_id: str, leader: bool):
        self.rig = rig
        self.camera_id = camera_id
        self.leader = leader
        self.rewound = False

    def read(self) -> Frame:
        if self.leader or self.camera_id not in self.rig.latest:
            self.rig.advance()
        return self.rig.latest[self.camera_id]
