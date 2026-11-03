from dataclasses import dataclass

import cv2
import numpy as np

from app.geom import CameraPose, default_intrinsics
from app.profile import BoardSpec


@dataclass
class BoardView:
    object_points: np.ndarray
    image_points: np.ndarray
    width: int
    height: int


class BoardDetector:
    def __init__(self, spec: BoardSpec):
        self.spec = spec
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.board = cv2.aruco.CharucoBoard(
            (spec.squares_x, spec.squares_y),
            spec.square_m,
            spec.marker_m,
            dictionary,
        )
        self.detector = cv2.aruco.CharucoDetector(self.board)

    def image(self) -> np.ndarray:
        width = self.spec.squares_x * 180
        height = self.spec.squares_y * 180
        board = self.board.generateImage((width, height))
        if board.ndim == 2:
            board = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)
        return board

    def detect(self, image: np.ndarray) -> BoardView | None:
        corners, ids, _, _ = self.detector.detectBoard(image)
        if ids is None or len(ids) < 6:
            return None
        if self.board.checkCharucoCornersCollinear(ids):
            return None
        object_points, image_points = self.board.matchImagePoints(corners, ids)
        if object_points is None or len(object_points) < 6:
            return None
        height, width = image.shape[:2]
        return BoardView(
            object_points.astype(np.float32),
            image_points.astype(np.float32),
            width,
            height,
        )


def pose_from_view(
    camera_id: str,
    view: BoardView,
    matrix: np.ndarray | None = None,
    dist: np.ndarray | None = None,
) -> CameraPose | None:
    if matrix is None or dist is None:
        matrix, dist = default_intrinsics(view.width, view.height)
    ok, rvec, tvec = cv2.solvePnP(
        view.object_points,
        view.image_points,
        matrix,
        dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    return CameraPose(
        camera_id,
        matrix.astype(np.float64),
        dist.astype(np.float64).reshape(-1),
        rvec.astype(np.float64),
        tvec.astype(np.float64),
        view.width,
        view.height,
    )


def intrinsics_from_views(
    views: list[BoardView],
) -> tuple[np.ndarray, np.ndarray] | None:
    if len(views) < 4:
        return None
    size = (views[-1].width, views[-1].height)
    rms, matrix, dist, _, _ = cv2.calibrateCamera(
        [view.object_points for view in views],
        [view.image_points for view in views],
        size,
        None,
        None,
        flags=cv2.CALIB_FIX_K3,
    )
    if not np.isfinite(rms):
        return None
    return matrix.astype(np.float64), dist.astype(np.float64).reshape(-1)
