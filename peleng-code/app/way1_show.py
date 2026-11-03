"""Пять роликов Way1: источники, позы камер и точка цели на текущем кадре."""

from pathlib import Path

import numpy as np

from app.cine_table import pose_at
from app.way1_sources import open_sources, video_info


def load_way1(root: Path) -> tuple[list, dict]:
    """Источники видео и позы на кадре 0. Весь путь заранее не отдаёт."""
    root = Path(root)
    info = video_info(root)
    sources = open_sources(root)
    try:
        poses: dict = {}
        for source in sources:
            camera_id = source.camera_id
            meta = info[camera_id]
            poses[camera_id] = pose_at(
                root,
                camera_id,
                0,
                int(meta["width"]),
                int(meta["height"]),
            )
    except Exception:
        _close_sources(sources)
        raise
    return sources, poses


def points_until(
    track: np.ndarray,
    shown: int,
    frame: int,
) -> tuple[list[np.ndarray], int, bool]:
    """Точки, которые цель уже прошла к этому кадру.

    Кадры впереди не возвращаются. Если ролик пошёл сначала, restart=True
    и в списке только положение на новом кадре.
    """
    points = np.asarray(track, np.float64).reshape(-1, 3)
    count = int(points.shape[0])
    if count == 0:
        return [], shown, False
    frame = int(min(max(int(frame), 0), count - 1))
    if shown >= count:
        shown = count - 1
    if frame < shown:
        return [points[frame].copy()], frame, True
    if frame == shown:
        return [], shown, False
    chunk = [points[index].copy() for index in range(shown + 1, frame + 1)]
    return chunk, frame, False


def _close_sources(sources: list) -> None:
    for source in sources:
        close = getattr(source, "close", None)
        if close is None:
            continue
        try:
            close()
        except Exception:
            continue
