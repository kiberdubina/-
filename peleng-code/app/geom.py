from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CameraPose:
    camera_id: str
    K: np.ndarray
    dist: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    width: int
    height: int


def default_intrinsics(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    focal = float(max(width, height))
    matrix = np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        np.float64,
    )
    return matrix, np.zeros(5, np.float64)


def look_at(center: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray(center, np.float64)
    target = np.asarray(target, np.float64)
    forward = target - center
    forward = forward / np.linalg.norm(forward)
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(forward, up))) > 0.92:
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    down = down / np.linalg.norm(down)
    rotation = np.stack([right, down, forward], axis=0)
    translation = (-rotation @ center).reshape(3, 1)
    rvec, _ = cv2.Rodrigues(rotation)
    return rvec, translation


def camera_center(pose: CameraPose) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(pose.rvec)
    return (-rotation.T @ pose.tvec.reshape(3)).ravel()


def frustum_edges(pose: CameraPose, depth: float = 0.15) -> list[tuple[np.ndarray, np.ndarray]]:
    rotation, _ = cv2.Rodrigues(pose.rvec)
    center = camera_center(pose)
    right = rotation.T @ np.array([1.0, 0.0, 0.0])
    down = rotation.T @ np.array([0.0, 1.0, 0.0])
    forward = rotation.T @ np.array([0.0, 0.0, 1.0])
    fx = float(pose.K[0, 0])
    fy = float(pose.K[1, 1])
    cx = float(pose.K[0, 2])
    cy = float(pose.K[1, 2])
    corners = []
    for u, v in (
        (0.0, 0.0),
        (float(pose.width), 0.0),
        (float(pose.width), float(pose.height)),
        (0.0, float(pose.height)),
    ):
        corners.append(
            center
            + forward * depth
            + right * ((u - cx) / fx * depth)
            + down * ((v - cy) / fy * depth)
        )
    edges = [(center, corner) for corner in corners]
    for index in range(4):
        edges.append((corners[index], corners[(index + 1) % 4]))
    return edges


def _projection(pose: CameraPose) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(pose.rvec)
    return pose.K @ np.hstack([rotation, pose.tvec.reshape(3, 1)])


def _undistort(pose: CameraPose, u: float, v: float) -> tuple[float, float] | None:
    raw = np.array([[[float(u), float(v)]]], np.float64)
    refined = cv2.undistortPoints(raw, pose.K, pose.dist, P=pose.K)
    if refined is None or not np.isfinite(refined).all():
        return None
    return float(refined[0, 0, 0]), float(refined[0, 0, 1])


def _triangulate_pair(
    first: np.ndarray,
    second: np.ndarray,
    pixel_a: tuple[float, float],
    pixel_b: tuple[float, float],
) -> np.ndarray | None:
    """Пересечение двух лучей. Точки уже без дисторсии, в пикселях матрицы P."""
    left = np.array([[pixel_a[0]], [pixel_a[1]]], np.float64)
    right = np.array([[pixel_b[0]], [pixel_b[1]]], np.float64)
    homogeneous = cv2.triangulatePoints(
        np.asarray(first, np.float64),
        np.asarray(second, np.float64),
        left,
        right,
    )
    if homogeneous is None or not np.isfinite(homogeneous).all():
        return None
    weight = float(homogeneous[3, 0])
    if abs(weight) < 1e-9:
        return None
    return (homogeneous[:3, 0] / weight).astype(np.float64)


def _depth(pose: CameraPose, xyz: np.ndarray) -> float:
    rotation, _ = cv2.Rodrigues(pose.rvec)
    camera = rotation @ np.asarray(xyz, np.float64).reshape(3, 1) + pose.tvec.reshape(3, 1)
    return float(camera[2, 0])


def _reprojection_error(
    poses: list[CameraPose], pixels: list[tuple[float, float]], xyz: np.ndarray
) -> float | None:
    errors = []
    for pose, pixel in zip(poses, pixels):
        image, _ = cv2.projectPoints(
            np.asarray(xyz, np.float64).reshape(1, 1, 3),
            pose.rvec,
            pose.tvec,
            pose.K,
            np.zeros(5),
        )
        if image is None or not np.isfinite(image).all():
            return None
        errors.append(
            float(np.hypot(image[0, 0, 0] - pixel[0], image[0, 0, 1] - pixel[1]))
        )
    if not errors:
        return None
    return float(np.mean(errors))


def triangulate(poses: list[CameraPose], pixels: list[tuple[float, float]]) -> np.ndarray | None:
    """Метры из лучей камер. Два вида считает cv2.triangulatePoints, дальше — лучшая пара."""
    if len(poses) < 2 or len(poses) != len(pixels):
        return None
    clean: list[tuple[float, float]] = []
    projections = []
    for pose, (u, v) in zip(poses, pixels):
        point = _undistort(pose, u, v)
        if point is None:
            return None
        clean.append(point)
        projections.append(_projection(pose))
    candidates = []
    pairs = [(0, 1)] if len(poses) == 2 else [
        (i, j) for i in range(len(poses)) for j in range(i + 1, len(poses))
    ]
    for i, j in pairs:
        xyz = _triangulate_pair(projections[i], projections[j], clean[i], clean[j])
        if xyz is not None:
            candidates.append(xyz)
    best = None
    best_error = None
    for xyz in candidates:
        if any(_depth(pose, xyz) <= 1e-4 for pose in poses):
            continue
        error = _reprojection_error(poses, clean, xyz)
        if error is None or error > 25.0:
            continue
        if best_error is None or error < best_error:
            best = xyz
            best_error = error
    return None if best is None else best.astype(np.float64)


def sight_rays(
    poses: list[CameraPose], xyz: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Отрезки лучей от центров камер до общей точки."""
    point = np.asarray(xyz, np.float64).reshape(3)
    return [(camera_center(pose), point.copy()) for pose in poses]


def velocity_in_camera(velocity: np.ndarray, pose: CameraPose) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(pose.rvec)
    return (rotation @ np.asarray(velocity, np.float64).reshape(3)).ravel()
