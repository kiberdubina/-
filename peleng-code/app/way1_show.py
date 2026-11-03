"""Пять роликов Way1: источники и позы камер. Путь цели из таблицы не рисуется."""

from pathlib import Path

from app.cine_table import pose_at
from app.way1_sources import open_sources, video_info


def load_way1(root: Path) -> tuple[list, dict]:
    """Источники видео и позы на кадре 0."""
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


def _close_sources(sources: list) -> None:
    for source in sources:
        close = getattr(source, "close", None)
        if close is None:
            continue
        try:
            close()
        except Exception:
            continue
