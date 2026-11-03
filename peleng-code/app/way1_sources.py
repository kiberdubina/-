from pathlib import Path

import cv2

from app.source import VideoSource
from app.way1_files import find_file

# Номер в имени файла Way1.FinalImage.CineCameraActorN.mp4 → cam-N.
_CAMERAS = tuple(f"cam-{number}" for number in range(1, 6))


def _clip_name(camera_id: str) -> str:
    number = camera_id.rsplit("-", 1)[1]
    return f"Way1.FinalImage.CineCameraActor{number}.mp4"


def video_paths(root: Path) -> dict[str, Path]:
    """Ключи cam-1…cam-5. Файл ищется в корне и в подпапках (видео/ и др.).
    Нет файла — KeyError с его именем."""
    paths: dict[str, Path] = {}
    for camera_id in _CAMERAS:
        name = _clip_name(camera_id)
        try:
            paths[camera_id] = find_file(Path(root), name)
        except FileNotFoundError:
            raise KeyError(name) from None
    return paths


def video_info(root: Path) -> dict[str, dict]:
    """Ширина, высота, fps и число кадров через CAP_PROP, без чтения кадров."""
    info: dict[str, dict] = {}
    for camera_id, path in video_paths(root).items():
        capture = cv2.VideoCapture(str(path))
        try:
            info[camera_id] = {
                "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": float(capture.get(cv2.CAP_PROP_FPS)),
                "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            }
        finally:
            capture.release()
    return info


def open_sources(root: Path) -> list:
    """Уже открытые VideoSource в порядке cam-1…cam-5."""
    opened: list = []
    try:
        for camera_id, path in video_paths(root).items():
            source = VideoSource(str(path), camera_id)
            if not source.is_open():
                source.close()
                raise RuntimeError(path.name)
            opened.append(source)
    except Exception:
        for source in opened:
            source.close()
        raise
    return opened
