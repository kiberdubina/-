"""Экспорт контрольного результата: видео с bbox, CSV и графики XYZ."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np


def write_control_video(
    frames_bgr: list[np.ndarray], path: Path, fps: float = 30.0
) -> Path:
    """Сохраняет последовательность BGR-кадров в MP4 или AVI."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames_bgr:
        raise ValueError("write_control_video: нет кадров")
    height, width = frames_bgr[0].shape[:2]
    size = (width, height)
    attempts: list[tuple[Path, str]] = [
        (path if path.suffix else path.with_suffix(".mp4"), "mp4v"),
        (path.with_suffix(".avi") if path.suffix.lower() != ".avi" else path, "XVID"),
    ]
    seen: set[Path] = set()
    for out_path, fourcc_str in attempts:
        if out_path in seen:
            continue
        seen.add(out_path)
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*fourcc_str),
            float(fps),
            size,
        )
        if not writer.isOpened():
            continue
        for frame in frames_bgr:
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)
            writer.write(frame)
        writer.release()
        return out_path
    raise RuntimeError("write_control_video: кодек не открылся")


def write_xyz_csv(
    times: list[float], xyz: list[np.ndarray], path: Path
) -> Path:
    """CSV с колонками t, x, y, z (метры)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t", "x", "y", "z"])
        for time_s, point in zip(times, xyz):
            vector = np.asarray(point, dtype=float).reshape(3)
            writer.writerow(
                [
                    float(time_s),
                    float(vector[0]),
                    float(vector[1]),
                    float(vector[2]),
                ]
            )
    return path


def write_xyz_charts(
    times: list[float], xyz: list[np.ndarray], path_prefix: Path
) -> list[Path]:
    """Графики X/Y/Z по времени: path_prefix_x.png и т.д."""
    path_prefix = Path(path_prefix)
    path_prefix.parent.mkdir(parents=True, exist_ok=True)
    if not times or not xyz:
        return []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        write_xyz_csv(times, xyz, path_prefix.with_name(path_prefix.name + ".csv"))
        return []

    t = np.asarray(times, dtype=float)
    arr = np.stack([np.asarray(p, dtype=float).reshape(3) for p in xyz], axis=0)
    labels = ("x", "y", "z")
    titles = ("X, м", "Y, м", "Z, м")
    out_paths: list[Path] = []
    for index, (axis, title) in enumerate(zip(labels, titles)):
        fig, ax = plt.subplots(figsize=(6, 3), dpi=100)
        ax.plot(t, arr[:, index], color="#2ecc71", linewidth=1.5)
        ax.set_xlabel("t, с")
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = path_prefix.parent / f"{path_prefix.name}_{axis}.png"
        fig.savefig(out)
        plt.close(fig)
        out_paths.append(out)
    return out_paths


def _annotate_bbox(image: np.ndarray, track) -> np.ndarray:
    out = image.copy()
    bbox = getattr(track, "bbox", None)
    if bbox is not None:
        x, y, w, h = bbox
        color = (80, 220, 80) if getattr(track, "in_gate", True) else (0, 200, 255)
        cv2.rectangle(out, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
    return out


def export_session_sample(session, out_dir: Path, max_frames: int = 120) -> dict:
    """Офлайн-прогон session.step(): overlay-видео, xyz.csv, графики."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_bgr: list[np.ndarray] = []
    times: list[float] = []
    xyz: list[np.ndarray] = []
    for _ in range(max(1, int(max_frames))):
        tick = session.step()
        if tick is None:
            break
        step = tick.steps[0] if tick.steps else None
        if step is not None:
            frames_bgr.append(_annotate_bbox(step.frame.image, step.track))
        world = tick.world
        if world.xyz is not None and step is not None:
            times.append(float(step.frame.time_s))
            xyz.append(np.asarray(world.xyz, dtype=float).reshape(3))

    result: dict[str, object] = {"frames": len(frames_bgr), "points": len(xyz)}
    if frames_bgr:
        result["video"] = write_control_video(
            frames_bgr, out_dir / "control_overlay.mp4", fps=30.0
        )
    if times and xyz:
        result["csv"] = write_xyz_csv(times, xyz, out_dir / "xyz.csv")
        result["charts"] = write_xyz_charts(times, xyz, out_dir / "xyz")
    return result
