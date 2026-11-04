import json
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.geom import CameraPose
from app.way1_files import find_file

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
    # Кадры таблицы, где поза была записана, и число строк, которые не разобрались.
    first_frame: int = 0
    last_frame: int = 0
    bad_rows: int = 0


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
        hit = loader(find_file(root, name))
        _CACHE[key] = hit
    return hit


# ---------------------------------------------------------------- таблицы по пути
# Ролики и таблицы любой записи: имя файла любое, кадр таблицы — абсолютный номер Frame.


def _cached_path(path: Path, loader):
    path = Path(path).resolve()
    key = (str(path), f"{loader.__name__}:{path.stat().st_mtime_ns}")
    hit = _CACHE.get(key)
    if hit is None:
        hit = loader(path)
        _CACHE[key] = hit
    return hit


def camera_table(path: Path) -> _Clip:
    """Таблица позы камеры (формат object_transform) по пути. Кадры 0..последний."""
    return _cached_path(path, _read_camera)


def track_table(path: Path) -> np.ndarray:
    """Путь цели по пути таблицы, метры, индекс — номер кадра таблицы. shape (N, 3)."""
    return np.array(_cached_path(path, _read_track), np.float64, copy=True)


def pose_in_table(path: Path, frame: int, width: int, height: int, camera_id: str = "") -> CameraPose:
    """Поза камеры на кадре таблицы. Вне диапазона — крайний кадр."""
    clip = camera_table(path)
    frame = _clamp(frame, int(clip.centers.shape[0]))
    matrix = _intrinsics(clip.focal_mm, clip.sensor_w_mm, clip.sensor_h_mm, width, height)
    return CameraPose(
        camera_id=camera_id or Path(path).stem,
        K=matrix,
        dist=np.zeros(5, np.float64),
        rvec=clip.rvecs[frame].reshape(3, 1).copy(),
        tvec=clip.tvecs[frame].reshape(3, 1).copy(),
        width=int(width),
        height=int(height),
    )


def camera_motion(path: Path) -> tuple[float, float]:
    """Насколько камера сдвигается и поворачивается за запись: метры и градусы от первого кадра."""
    clip = camera_table(path)
    start = clip.first_frame
    centers = clip.centers[start:]
    shift = float(np.linalg.norm(centers - centers[0], axis=1).max()) if len(centers) else 0.0
    first, _ = cv2.Rodrigues(clip.rvecs[start].reshape(3, 1))
    turn = 0.0
    for rvec in clip.rvecs[start:]:
        rotation, _ = cv2.Rodrigues(rvec.reshape(3, 1))
        cosine = (float(np.trace(first.T @ rotation)) - 1.0) / 2.0
        turn = max(turn, float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))))
    return shift, turn


def table_kind(path: Path) -> str | None:
    """«camera» — поза камеры (есть углы), «track» — путь цели (только координаты), иначе None."""
    try:
        if _calibration_json(Path(path)) is not None:
            return "camera"
        lines = _lines(Path(path))
    except Exception:
        return None
    if not lines:
        return None
    header = lines[0].lower()
    if "loc" in header and "rot" in header:
        return "camera"
    if "loc" in header:
        return "track"
    # Таблица без шапки: по числу чисел в первой строке. Ровно 7 — поза, ровно 4 — путь;
    # разметка MOT (10 чисел: кадр, id, рамка…) — ни то ни другое.
    cells = [cell for cell in lines[0].split(_delimiter(lines[0])) if cell.strip()]
    try:
        [_number(cell) for cell in cells]
    except ValueError:
        return None
    if len(cells) == 7:
        return "camera"
    if len(cells) == 4:
        return "track"
    return None


def _calibration_json(path: Path) -> dict | None:
    """Калибровка камеры CARLA (Synthehicle): JSON с intrinsic_matrix и extrinsic_matrix."""
    if path.suffix.lower() not in (".txt", ".json") or path.stat().st_size > 100_000:
        return None
    text = path.read_text(encoding="utf-8", errors="replace").lstrip()
    if not text.startswith("{"):
        return None
    try:
        data = json.loads(text)
    except ValueError:
        return None
    if "intrinsic_matrix" in data and "extrinsic_matrix" in data:
        return data
    return None


def _read_calibration(data: dict) -> _Clip:
    """Неподвижная камера CARLA. extrinsic — мир→камера в осях Unreal (вперёд, вправо, вверх),
    метры. Мир программы — тот же Unreal со сменой знака Y (как у таблиц Way1), камера — OpenCV."""
    K = np.asarray(data["intrinsic_matrix"], np.float64)
    E = np.asarray(data["extrinsic_matrix"], np.float64)
    flip = np.diag([1.0, -1.0, 1.0])
    # Камера Unreal (вперёд, вправо, вверх) → OpenCV (вправо, вниз, вперёд).
    to_cv = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    rotation = to_cv @ E[:3, :3] @ flip
    tvec = to_cv @ E[:3, 3]
    rvec, _ = cv2.Rodrigues(rotation)
    center = -rotation.T @ tvec
    # Матрица K задана для кадра 2·cx × 2·cy: фокус в тех же единицах, что ширина матрицы.
    return _Clip(
        centers=center.reshape(1, 3),
        rvecs=rvec.reshape(1, 3),
        tvecs=tvec.reshape(1, 3),
        focal_mm=float(K[0, 0]),
        sensor_w_mm=float(2.0 * K[0, 2]),
        sensor_h_mm=float(2.0 * K[1, 2]) * float(K[0, 0]) / float(K[1, 1]),
    )


def _delimiter(header: str) -> str:
    if ";" in header:
        return ";"
    if "\t" in header:
        return "\t"
    return ","


def _decode(path: Path) -> str:
    data = path.read_bytes()
    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig")
    try:
        return data.decode("cp1251")
    except UnicodeDecodeError:
        return data.decode("utf-8-sig")


def _lines(path: Path) -> list[str]:
    if Path(path).suffix.lower() in (".xlsx", ".xlsm"):
        return _excel_lines(Path(path))
    return [line.strip() for line in _decode(path).splitlines() if line.strip()]


def _excel_lines(path: Path) -> list[str]:
    """Первый лист книги Excel как строки CSV через «;». Ячейка-дата — битая, её заполнят соседи."""
    from openpyxl import load_workbook

    book = load_workbook(path, read_only=True, data_only=True)
    try:
        sheet = book.worksheets[0]
        lines = []
        for row in sheet.iter_rows(values_only=True):
            cells = ["" if value is None else (repr(value) if isinstance(value, float) else str(value)) for value in row]
            line = ";".join(cells).strip()
            if line.strip(";"):
                lines.append(line)
        return lines
    finally:
        book.close()


def _number(text: str) -> float:
    # В хвосте CSV десятичная запятая: «23,76».
    cleaned = text.strip().replace(" ", "").replace("\xa0", "")
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
    cells = header.split(_delimiter(header))
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


def _by_frame(frames: list[int], values: np.ndarray, fill: float = 0.0) -> np.ndarray:
    count = max(frames) + 1
    packed = np.full((count,) + values.shape[1:], fill, np.float64)
    for index, frame in enumerate(frames):
        packed[frame] = values[index]
    return packed


def _frame_number(text: str) -> int | None:
    """Номер кадра из первой ячейки. Шапка или мусор — None."""
    try:
        value = float(text.strip().replace("\xa0", "").replace(",", "."))
    except ValueError:
        return None
    if not np.isfinite(value) or value < 0:
        return None
    return int(round(value))


def _read_track(path: Path) -> np.ndarray:
    lines = _lines(path)
    if not lines:
        raise ValueError(f"{Path(path).name}: пустая таблица")
    delimiter = _delimiter(lines[0])
    frames = []
    points = []
    for line in lines:
        cells = line.split(delimiter)
        if len(cells) < 4:
            continue
        frame = _frame_number(cells[0])
        if frame is None:
            continue
        frames.append(frame)
        # Unreal: см, X вперёд, Y вправо, Z вверх → метры, Y со знаком минус.
        points.append(
            (
                _maybe_number(cells[1]) / 100.0,
                -_maybe_number(cells[2]) / 100.0,
                _maybe_number(cells[3]) / 100.0,
            )
        )
    if not frames:
        raise ValueError(f"{Path(path).name}: нет ни одной строки с кадром")
    # Пропущенный кадр и битая ячейка — NaN, их заполняют соседние кадры.
    return _fill_gaps(_by_frame(frames, np.asarray(points, np.float64), fill=np.nan))


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
    calibration = _calibration_json(Path(path))
    if calibration is not None:
        return _read_calibration(calibration)
    lines = _lines(path)
    if not lines:
        raise ValueError(f"{Path(path).name}: пустая таблица")
    delimiter = _delimiter(lines[0])
    focal = _focal_mm(lines[0])
    sensor_w, sensor_h = _sensor_mm(lines[1]) if len(lines) > 1 else (_SENSOR_W_MM, _SENSOR_H_MM)
    # Кадр → (центр в метрах, углы Unreal). Битая строка (Excel записал число датой) пропускается.
    rows: dict[int, tuple[np.ndarray, tuple[float, float, float]]] = {}
    bad = 0
    for line in lines:
        cells = line.split(delimiter)
        frame = _frame_number(cells[0]) if cells else None
        if frame is None:
            continue
        try:
            if len(cells) < 7:
                raise ValueError("мало ячеек")
            center = np.array(
                [
                    _number(cells[1]) / 100.0,
                    -_number(cells[2]) / 100.0,
                    _number(cells[3]) / 100.0,
                ],
                np.float64,
            )
            angles = (_number(cells[4]), _number(cells[5]), _number(cells[6]))
        except ValueError:
            bad += 1
            continue
        if not (np.isfinite(center).all() and np.isfinite(angles).all()):
            bad += 1
            continue
        rows[frame] = (center, angles)
    if not rows:
        raise ValueError(f"{Path(path).name}: нет ни одной строки позы")
    known = np.array(sorted(rows), np.int64)
    count = int(known[-1]) + 1
    index = np.arange(count)
    # Пропуск: центр — по соседним кадрам, поворот — ближайшего записанного кадра.
    known_centers = np.stack([rows[int(frame)][0] for frame in known])
    centers = np.stack([np.interp(index, known, known_centers[:, axis]) for axis in range(3)], axis=1)
    after = np.clip(np.searchsorted(known, index), 0, len(known) - 1)
    before = np.clip(after - 1, 0, len(known) - 1)
    nearest = np.where(np.abs(known[before] - index) <= np.abs(known[after] - index), known[before], known[after])
    rvecs = np.zeros((count, 3), np.float64)
    tvecs = np.zeros((count, 3), np.float64)
    cache: dict[tuple[float, float, float], tuple[np.ndarray, np.ndarray]] = {}
    for frame in range(count):
        angles = rows[int(nearest[frame])][1]
        if angles not in cache:
            rotation = _opencv_rotation(*angles)
            rvec, _ = cv2.Rodrigues(rotation)
            cache[angles] = (rotation, rvec.reshape(3))
        rotation, rvec = cache[angles]
        rvecs[frame] = rvec
        tvecs[frame] = -rotation @ centers[frame]
    return _Clip(
        centers=centers,
        rvecs=rvecs,
        tvecs=tvecs,
        focal_mm=float(focal),
        sensor_w_mm=float(sensor_w),
        sensor_h_mm=float(sensor_h),
        first_frame=int(known[0]),
        last_frame=int(known[-1]),
        bad_rows=bad,
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
