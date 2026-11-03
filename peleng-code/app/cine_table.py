import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.geom import CameraPose

# Разбор CSV один раз на процесс: pose_at не открывает файл заново.
_CACHE: dict[tuple[str, str], object] = {}

_TRACK_NAME = "object_transform_SM_AA_track.csv"
_CAMERA_FILES = {
    "cam-1": "object_transform_CineCameraActor1.csv",
    "cam-2": "object_transform_CineCameraActor2.csv",
    "cam-3": "object_transform_CineCameraActor3.csv",
    "cam-4": "object_transform_CineCameraActor4.csv",
    "cam-5": "object_transform_CineCameraActor5.csv",
}
_FOCAL_MM = 35.0
_SENSOR_W_MM = 23.76
_SENSOR_H_MM = 13.365
_SENSOR = re.compile(r"(\d+[,.]\d+)\s*\*\s*(\d+[,.]\d+)")


@dataclass
class _Clip:
    centers: np.ndarray
    rvecs: np.ndarray
    tvecs: np.ndarray
    focal_mm: float
    sensor_w_mm: float
    sensor_h_mm: float


def track_meters(root: Path) -> np.ndarray:
    """Траектория цели, метры, кадр 0..N-1. shape (N, 3)."""
    track = _cached(root, _TRACK_NAME, _read_track)
    return np.array(track, np.float64, copy=True)


def pose_at(root: Path, camera_id: str, frame: int, width: int, height: int) -> CameraPose:
    """Поза кинокамеры на кадре. Вне диапазона — крайний кадр."""
    filename = _CAMERA_FILES.get(camera_id)
    if filename is None:
        raise ValueError(f"неизвестная камера: {camera_id}")
    clip = _cached(root, filename, _read_camera)
    frame = _clamp(frame, int(clip.centers.shape[0]))
    matrix = _intrinsics(clip.focal_mm, clip.sensor_w_mm, clip.sensor_h_mm, width, height)
    return CameraPose(
        camera_id=camera_id,
        K=matrix,
        dist=np.zeros(5, np.float64),
        rvec=clip.rvecs[frame].reshape(3, 1).copy(),
        tvec=clip.tvecs[frame].reshape(3, 1).copy(),
        width=int(width),
        height=int(height),
    )


def relative_layout(root: Path) -> dict:
    """Центры на кадре 0 и попарные расстояния, метры."""
    centers = {}
    for camera_id in _CAMERA_FILES:
        pose = pose_at(root, camera_id, 0, 1, 1)
        rotation, _ = cv2.Rodrigues(pose.rvec)
        center = (-rotation.T @ pose.tvec.reshape(3)).ravel()
        centers[camera_id] = [float(center[0]), float(center[1]), float(center[2])]
    distance = {}
    ids = list(_CAMERA_FILES)
    for i, left in enumerate(ids):
        a = np.asarray(centers[left], np.float64)
        for right in ids[i + 1 :]:
            b = np.asarray(centers[right], np.float64)
            distance[f"{left}|{right}"] = float(np.linalg.norm(a - b))
    return {"centers_m": centers, "distance_m": distance}


def _cached(root: Path, name: str, loader):
    key = (str(Path(root).resolve()), name)
    hit = _CACHE.get(key)
    if hit is None:
        hit = loader(Path(root) / name)
        _CACHE[key] = hit
    return hit


def _decode(path: Path) -> str:
    data = path.read_bytes()
    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig")
    try:
        return data.decode("cp1251")
    except UnicodeDecodeError:
        return data.decode("utf-8-sig")


def _lines(path: Path) -> list[str]:
    return [line.strip() for line in _decode(path).splitlines() if line.strip()]


def _number(text: str) -> float:
    # В хвосте CSV десятичная запятая: «23,76».
    cleaned = text.strip().replace(" ", "")
    if "," in cleaned and "." not in cleaned:
        cleaned = cleaned.replace(",", ".")
    return float(cleaned)


def _maybe_number(text: str) -> float:
    try:
        return _number(text)
    except ValueError:
        return float("nan")


def _fill_gaps(values: np.ndarray) -> np.ndarray:
    # Excel записал пару координат как даты. Берём соседние кадры.
    filled = np.array(values, np.float64, copy=True)
    index = np.arange(filled.shape[0], dtype=np.float64)
    for axis in range(filled.shape[1]):
        column = filled[:, axis]
        good = np.isfinite(column)
        if bool(good.all()) or not bool(good.any()):
            continue
        filled[:, axis] = np.interp(index, index[good], column[good])
    return filled


def _focal_mm(header: str) -> float:
    cells = header.split(";")
    for index, cell in enumerate(cells):
        if cell.strip().lower() != "focal length":
            continue
        if index + 1 < len(cells) and cells[index + 1].strip():
            return _number(cells[index + 1])
    return _FOCAL_MM


def _sensor_mm(line: str) -> tuple[float, float]:
    match = _SENSOR.search(line)
    if match is None:
        return _SENSOR_W_MM, _SENSOR_H_MM
    return _number(match.group(1)), _number(match.group(2))


def _by_frame(frames: list[int], values: np.ndarray) -> np.ndarray:
    count = max(frames) + 1
    packed = np.zeros((count,) + values.shape[1:], np.float64)
    for index, frame in enumerate(frames):
        packed[frame] = values[index]
    return packed


def _read_track(path: Path) -> np.ndarray:
    frames = []
    points = []
    for line in _lines(path)[1:]:
        cells = line.split(";")
        if len(cells) < 4:
            continue
        frames.append(int(float(cells[0])))
        # Unreal: см, X вперёд, Y вправо, Z вверх → метры, Y со знаком минус.
        points.append(
            (
                _maybe_number(cells[1]) / 100.0,
                -_maybe_number(cells[2]) / 100.0,
                _maybe_number(cells[3]) / 100.0,
            )
        )
    return _fill_gaps(_by_frame(frames, np.asarray(points, np.float64)))


def _unreal_axes(roll_deg: float, pitch_deg: float, yaw_deg: float):
    """FRotator: оси камеры в мире Unreal — вперёд, вправо, вверх."""
    roll, pitch, yaw = np.deg2rad([roll_deg, pitch_deg, yaw_deg])
    sr, cr = np.sin(roll), np.cos(roll)
    sp, cp = np.sin(pitch), np.cos(pitch)
    sy, cy = np.sin(yaw), np.cos(yaw)
    forward = np.array([cp * cy, cp * sy, sp], np.float64)
    right = np.array([sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, -sr * cp], np.float64)
    up = np.array([-(cr * sp * cy + sr * sy), cy * sr - cr * sp * sy, cr * cp], np.float64)
    return forward, right, up


def _opencv_rotation(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    forward, right, up = _unreal_axes(roll_deg, pitch_deg, yaw_deg)
    # Та же смена оси, что у точки: мир программы правый, Z вверх.
    flip = np.array([1.0, -1.0, 1.0], np.float64)
    forward = forward * flip
    right = right * flip
    up = up * flip
    forward = forward / np.linalg.norm(forward)
    right = right - forward * float(np.dot(right, forward))
    right = right / np.linalg.norm(right)
    # OpenCV: X вправо, Y вниз, Z по лучу. down = forward × right.
    down = np.cross(forward, right)
    down = down / np.linalg.norm(down)
    return np.stack([right, down, forward], axis=0)


def _read_camera(path: Path) -> _Clip:
    lines = _lines(path)
    focal = _focal_mm(lines[0])
    sensor_w, sensor_h = _sensor_mm(lines[1]) if len(lines) > 1 else (_SENSOR_W_MM, _SENSOR_H_MM)
    frames = []
    centers = []
    rvecs = []
    tvecs = []
    for line in lines[1:]:
        cells = line.split(";")
        if len(cells) < 7:
            continue
        frames.append(int(float(cells[0])))
        center = np.array(
            [
                _number(cells[1]) / 100.0,
                -_number(cells[2]) / 100.0,
                _number(cells[3]) / 100.0,
            ],
            np.float64,
        )
        rotation = _opencv_rotation(_number(cells[4]), _number(cells[5]), _number(cells[6]))
        rvec, _ = cv2.Rodrigues(rotation)
        tvec = -rotation @ center
        centers.append(center)
        rvecs.append(rvec.reshape(3))
        tvecs.append(tvec.reshape(3))
    return _Clip(
        centers=_by_frame(frames, np.asarray(centers, np.float64)),
        rvecs=_by_frame(frames, np.asarray(rvecs, np.float64)),
        tvecs=_by_frame(frames, np.asarray(tvecs, np.float64)),
        focal_mm=float(focal),
        sensor_w_mm=float(sensor_w),
        sensor_h_mm=float(sensor_h),
    )


def _intrinsics(focal_mm: float, sensor_w: float, sensor_h: float, width: int, height: int) -> np.ndarray:
    fx = focal_mm / sensor_w * float(width)
    fy = focal_mm / sensor_h * float(height)
    return np.array(
        [[fx, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]],
        np.float64,
    )


def _clamp(frame: int, count: int) -> int:
    frame = int(frame)
    if frame < 0:
        return 0
    if frame >= count:
        return count - 1
    return frame
