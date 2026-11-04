"""Папка с роликами любой записи: камеры, таблицы поз, таблица цели.

Ролики могут называться как угодно, камер — сколько угодно. Видео — файлы с расширением
видео в папке и её подпапках (один уровень). Таблица позы камеры (формат Unreal
object_transform: Frame;Loc_X;Loc_Y;Loc_Z;Rot_Roll;Rot_Pitch;Rot_Yaw…; .csv, .txt или .xlsx)
сопоставляется с роликом:
  1. имя таблицы без «object_transform_» входит в имя ролика целым словом;
  2. иначе — по последнему числу в имени (Cam_03.mp4 ↔ …Actor3.csv);
  3. иначе, если роликов и таблиц поровну, — по порядку имён (с предупреждением).
Таблица без углов поворота — путь цели: только для погрешности, в расчёт не входит.

Кадр 0 роликов — первый кадр таблиц (frame_offset), если таблицы начинаются не с нуля.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from app.cine_table import camera_motion, camera_table, pose_in_table, table_kind, track_table
from app.geom import CameraPose

VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg", ".wmv")
TABLE_SUFFIXES = (".csv", ".txt", ".xlsx", ".xlsm")
# Свои результаты и служебные папки не считаются роликами записи.
_SKIP_DIRS = {"reports", "models", "__pycache__", ".git", ".venv", "venv"}
_SKIP_PREFIXES = (".", "~$", "control_", "xyz")
# Камера «едет», если центр сдвинулся больше этого, и «поворачивается» — если угол больше.
MOVE_M = 0.5
TURN_DEG = 0.5
_TARGET_WORDS = ("track", "truck", "target", "цель", "sm_")


@dataclass
class SceneCamera:
    camera_id: str
    video: Path
    table: Path | None = None
    width: int = 0
    height: int = 0
    frames: int = 0
    fps: float = 30.0
    shift_m: float = 0.0
    turn_deg: float = 0.0

    @property
    def moving(self) -> bool:
        return self.shift_m > MOVE_M

    @property
    def turning(self) -> bool:
        return self.turn_deg > TURN_DEG


@dataclass
class SceneFolder:
    root: Path
    cameras: list[SceneCamera]
    target: Path | None = None
    frame_offset: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def has_poses(self) -> bool:
        return bool(self.cameras) and all(camera.table is not None for camera in self.cameras)

    @property
    def frames(self) -> int:
        return min((camera.frames for camera in self.cameras), default=0)


@dataclass(frozen=True)
class FilePose:
    """Поза камеры по таблице на кадре ролика. Переносится в другой процесс."""

    table: str
    offset: int
    width: int
    height: int
    camera_id: str = ""

    def __call__(self, frame: int) -> CameraPose:
        return pose_in_table(Path(self.table), self.offset + int(frame), self.width, self.height, self.camera_id)


def scan_folder(root: Path) -> SceneFolder:
    """Ролики, их таблицы поз и таблица цели в папке. Ничего нет — пустой список камер и причина в notes."""
    root = Path(root).resolve()
    notes: list[str] = []
    if not root.is_dir():
        return SceneFolder(root, [], notes=[f"нет папки {root}"])
    videos, tables = _files(root)
    cameras: list[SceneCamera] = []
    for path in videos:
        info = _probe(path)
        if info is None:
            notes.append(f"ролик не открылся: {path.name}")
            continue
        width, height, frames, fps = info
        cameras.append(SceneCamera("", path, None, width, height, frames, fps))
    if not cameras:
        notes.append("в папке нет роликов (mp4, avi, mov, mkv…)")
        return SceneFolder(root, [], notes=notes)

    camera_tables, track_tables = [], []
    for path in tables:
        kind = table_kind(path)
        if kind == "camera":
            camera_tables.append(path)
        elif kind == "track":
            track_tables.append(path)
    _match(cameras, camera_tables, notes, root)
    _name(cameras, root)

    offsets = []
    for camera in cameras:
        if camera.table is None:
            continue
        try:
            clip = camera_table(camera.table)
            camera.shift_m, camera.turn_deg = camera_motion(camera.table)
        except Exception as error:
            notes.append(f"таблица {camera.table.name} не прочиталась ({error}) — у {camera.camera_id} нет позы")
            camera.table = None
            continue
        offsets.append(clip.first_frame)
        if clip.bad_rows:
            notes.append(f"{camera.table.name}: битых строк {clip.bad_rows}, заполнены соседними кадрами")
    offset = min(offsets) if offsets else 0
    if offsets and max(offsets) != offset:
        notes.append(f"таблицы начинаются с разных кадров ({sorted(set(offsets))}), кадр 0 роликов = {offset}")
    elif offset:
        notes.append(f"таблицы начинаются с кадра {offset}: кадр 0 роликов = кадр {offset} таблиц")

    target = _target(track_tables, notes)
    scene = SceneFolder(root, cameras, target, offset, notes)
    missing = [camera.camera_id for camera in cameras if camera.table is None]
    if missing and len(missing) < len(cameras):
        notes.append(f"нет таблиц поз у {', '.join(missing)} — они без метров")
    if not any(camera.table is not None for camera in cameras):
        notes.append("таблиц поз нет: только 2D. Для метров — калибровка по листу ChArUco")
    for camera in cameras:
        if camera.moving:
            notes.append(f"{camera.camera_id} едет (сдвиг до {camera.shift_m:.1f} м)")
        elif camera.turning:
            notes.append(f"{camera.camera_id} поворачивается (до {camera.turn_deg:.0f}°)")
    return scene


def scene_truth(scene: SceneFolder) -> np.ndarray | None:
    """Путь цели по кадрам роликов (кадр 0 = frame_offset таблицы). Нет таблицы — None."""
    if scene.target is None:
        return None
    try:
        track = track_table(scene.target)
    except Exception:
        return None
    if scene.frame_offset >= len(track):
        return None
    return track[scene.frame_offset :]


def scene_poses(scene: SceneFolder) -> dict[str, FilePose]:
    """Поза каждой камеры с таблицей на любой кадр ролика."""
    return {
        camera.camera_id: FilePose(str(camera.table), scene.frame_offset, camera.width, camera.height, camera.camera_id)
        for camera in scene.cameras
        if camera.table is not None
    }


def safe_name(name: str) -> str:
    """Имя папки отчёта без символов, которые не любит Windows."""
    cleaned = "".join("_" if char in '<>:"/\\|?* ' else char for char in name).strip("._")
    return cleaned or "scene"


def describe(scene: SceneFolder) -> str:
    """Короткая сводка для строки статуса и окна сообщения."""
    with_pose = sum(camera.table is not None for camera in scene.cameras)
    parts = [f"{scene.root.name}: роликов {len(scene.cameras)}, с позами {with_pose}, кадров {scene.frames}"]
    parts.append("эталон цели есть" if scene.target is not None else "эталона цели нет")
    return "; ".join(parts) + ("\n" + "\n".join(scene.notes) if scene.notes else "")


def video_motion_px(path: Path, pairs: int = 12, width: int = 640) -> float:
    """Медианный сдвиг картинки за кадр (px при ширине 640): камера в руке или едет — больше ~1,5."""
    capture = cv2.VideoCapture(str(path))
    shifts: list[float] = []
    previous = None
    try:
        for _ in range(pairs + 1):
            ok, image = capture.read()
            if not ok or image is None:
                break
            scale = width / float(image.shape[1])
            small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if previous is not None:
                points = cv2.goodFeaturesToTrack(previous, 200, 0.01, 8)
                if points is not None and len(points) >= 12:
                    moved, status, _ = cv2.calcOpticalFlowPyrLK(previous, gray, points, None)
                    good = status.reshape(-1).astype(bool)
                    if good.sum() >= 12:
                        steps = np.linalg.norm((moved - points)[good].reshape(-1, 2), axis=1)
                        shifts.append(float(np.median(steps)))
            previous = gray
    finally:
        capture.release()
    return float(np.median(shifts)) if shifts else 0.0


# ---------------------------------------------------------------- разбор папки


def _files(root: Path, depth: int = 3) -> tuple[list[Path], list[Path]]:
    """Видео и таблицы в папке и подпапках до depth уровней (Synthehicle: C01/out_rgb/video.mp4)."""
    videos, tables = [], []
    folders = [(root, 0)]
    while folders:
        folder, level = folders.pop(0)
        for path in sorted(folder.iterdir()):
            name = path.name.lower()
            if path.is_dir():
                if level < depth and name not in _SKIP_DIRS and not name.startswith("."):
                    folders.append((path, level + 1))
                continue
            if not path.is_file() or name.startswith(_SKIP_PREFIXES):
                continue
            suffix = path.suffix.lower()
            if suffix in VIDEO_SUFFIXES:
                videos.append(path)
            elif suffix in TABLE_SUFFIXES:
                tables.append(path)
    tables += _synthehicle_calibration(root)
    return videos, tables


def _synthehicle_calibration(root: Path) -> list[Path]:
    """Сцена Synthehicle (Town04-O-day): калибровки лежат отдельно — calibration/overlapping/Town04."""
    match = re.match(r"(Town\w+?)-([ON])-", root.name)
    if match is None:
        return []
    kind = "overlapping" if match.group(2) == "O" else "non_overlapping"
    for folder in [root, *root.parents]:
        found = folder / "calibration" / kind / match.group(1) / "camera_info"
        if found.is_dir():
            return sorted(found.glob("camera_*.txt"))
    return []


def _label(camera: SceneCamera, root: Path | None = None) -> str:
    """Имя ролика для сопоставления: путь от папки записи без расширения (C01/out_rgb/video)."""
    path = camera.video.with_suffix("")
    if root is not None:
        try:
            path = path.relative_to(root)
        except ValueError:
            pass
    return path.as_posix() if root is not None else camera.video.stem


def _probe(path: Path) -> tuple[int, int, int, float] | None:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return None
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
        if frames <= 0:
            # Контейнер без числа кадров: считаем сами.
            frames = 0
            while capture.grab():
                frames += 1
        if frames <= 0 or width <= 0 or height <= 0:
            return None
        return width, height, frames, fps if 1.0 <= fps <= 240.0 else 30.0
    finally:
        capture.release()


def _key(table: Path) -> str:
    stem = table.stem
    lowered = stem.lower()
    for prefix in ("object_transform_", "object_transform"):
        if lowered.startswith(prefix):
            return stem[len(prefix):]
    return stem


def _has_token(key: str, name: str) -> bool:
    if not key:
        return False
    return re.search(rf"(?<![0-9a-zа-я]){re.escape(key.lower())}(?![0-9a-zа-я])", name.lower()) is not None


def _tail_number(text: str) -> int | None:
    found = re.findall(r"\d+", text)
    return int(found[-1]) if found else None


def _match(cameras: list[SceneCamera], tables: list[Path], notes: list[str], root: Path | None = None) -> None:
    free = list(tables)
    # 1. Имя таблицы целым словом в имени ролика; длинное имя важнее (Actor10 раньше Actor1).
    for camera in cameras:
        hits = [table for table in free if _has_token(_key(table), _label(camera, root))]
        if hits:
            best = max(hits, key=lambda table: len(_key(table)))
            camera.table = best
            free.remove(best)
    # 2. По последнему числу в имени.
    left = [camera for camera in cameras if camera.table is None]
    for camera in left:
        number = _tail_number(_label(camera, root))
        if number is None:
            continue
        hits = [table for table in free if _tail_number(_key(table)) == number]
        if len(hits) == 1:
            camera.table = hits[0]
            free.remove(hits[0])
    # 3. Поровну — по порядку имён.
    left = [camera for camera in cameras if camera.table is None]
    if left and len(left) == len(free):
        for camera, table in zip(left, sorted(free)):
            camera.table = table
            notes.append(f"{camera.video.name} ↔ {table.name}: по порядку имён — проверьте")
        free = []
    for table in free:
        notes.append(f"таблица {table.name} без ролика")


def _name(cameras: list[SceneCamera], root: Path | None = None) -> None:
    """cam-N по числу в имени таблицы или ролика; совпало — по порядку."""
    names = []
    for camera in cameras:
        source = _key(camera.table) if camera.table is not None else _label(camera, root)
        number = _tail_number(source)
        names.append(f"cam-{number}" if number is not None else None)
    if None in names or len(set(names)) != len(names):
        names = [f"cam-{index + 1}" for index in range(len(cameras))]
    for camera, name in zip(cameras, names):
        camera.camera_id = name


def _target(tables: list[Path], notes: list[str]) -> Path | None:
    if not tables:
        return None
    named = [table for table in tables if any(word in table.stem.lower() for word in _TARGET_WORDS)]
    pick = sorted(named or tables)[0]
    if len(tables) > 1:
        notes.append(f"таблиц пути цели {len(tables)}, взята {pick.name}")
    return pick
