"""Где лежат ролики и таблицы Way1: корень проекта или его подпапки."""

from pathlib import Path

SUBDIRS = ("", "видео", "video", "videos", "execl", "excel", "tables", "data")


def find_file(root: Path, name: str) -> Path:
    """Путь к файлу name в root или его известной подпапке. Нет — FileNotFoundError."""
    root = Path(root)
    for sub in SUBDIRS:
        candidate = root / sub / name if sub else root / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(str(root / name))
