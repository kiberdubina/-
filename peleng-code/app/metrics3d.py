"""Погрешность 3D-трассы относительно object_transform_SM_AA_track.csv."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.cine_table import track_meters

# Доля кадров с евклидовой ошибкой не больше порога, м.
HIT_THRESHOLD_M = 10.0

_METRIC_LABELS = {
    "frames": "Кадров с оценкой",
    "coverage": "Покрытие записи",
    "rmse_m": "RMSE, м",
    "mean_m": "Средняя ошибка, м",
    "median_m": "Медиана ошибки, м",
    "p95_m": "95-й перцентиль, м",
    "max_m": "Максимум ошибки, м",
    "rmse_x": "RMSE X, м",
    "rmse_y": "RMSE Y, м",
    "rmse_z": "RMSE Z, м",
    "hit_ratio": "Доля попаданий",
}


def load_truth_meters(root: Path) -> np.ndarray:
    """Эталонная трасса цели в метрах, shape (N, 3)."""
    return track_meters(root)


def _as_track(array: np.ndarray) -> np.ndarray:
    data = np.asarray(array, dtype=np.float64)
    if data.size == 0:
        return data.reshape(0, 3)
    if data.ndim == 1:
        if data.shape[0] != 3:
            raise ValueError("ожидается (N, 3) или пустой массив")
        return data.reshape(1, 3)
    if data.shape[1] != 3:
        raise ValueError("ожидается (N, 3)")
    return data


def _stats_from_diff(diff: np.ndarray) -> dict:
    err = np.linalg.norm(diff, axis=1)
    return {
        "rmse_m": float(np.sqrt(np.mean(err**2))),
        "median_m": float(np.median(err)),
        "max_m": float(np.max(err)),
        "rmse_x": float(np.sqrt(np.mean(diff[:, 0] ** 2))),
        "rmse_y": float(np.sqrt(np.mean(diff[:, 1] ** 2))),
        "rmse_z": float(np.sqrt(np.mean(diff[:, 2] ** 2))),
        "hit_ratio": float(np.mean(err <= HIT_THRESHOLD_M)),
    }


def compare_tracks(est: np.ndarray, truth: np.ndarray) -> dict:
    """Сравнение по min(len): построчная евклидова ошибка."""
    est = _as_track(est)
    truth = _as_track(truth)
    n = min(est.shape[0], truth.shape[0])
    if n == 0:
        return {
            "rmse_m": float("nan"),
            "median_m": float("nan"),
            "max_m": float("nan"),
            "rmse_x": float("nan"),
            "rmse_y": float("nan"),
            "rmse_z": float("nan"),
            "hit_ratio": 0.0,
        }
    return _stats_from_diff(est[:n] - truth[:n])


def compare_tracks_nearest(est: np.ndarray, truth: np.ndarray) -> dict:
    """Каждая оценка к ближайшей точке эталона. Для редкого следа на длинной трассе."""
    est = _as_track(est)
    truth = _as_track(truth)
    if est.shape[0] == 0 or truth.shape[0] == 0:
        return {
            "rmse_m": float("nan"),
            "median_m": float("nan"),
            "max_m": float("nan"),
            "rmse_x": float("nan"),
            "rmse_y": float("nan"),
            "rmse_z": float("nan"),
            "hit_ratio": 0.0,
        }
    diff = np.empty_like(est)
    for index, point in enumerate(est):
        nearest = int(np.argmin(np.linalg.norm(truth - point.reshape(1, 3), axis=1)))
        diff[index] = point - truth[nearest]
    return _stats_from_diff(diff)


def compare_by_frame(
    frames: list[int] | np.ndarray, est: np.ndarray, truth: np.ndarray, total_frames: int | None = None
) -> dict:
    """Погрешность по номерам кадров: est[i] сравнивается с truth[frames[i]].

    Кадры вне таблицы отбрасываются. coverage — доля кадров записи, для которых есть
    оценка (знаменатель — total_frames или длина таблицы).
    """
    frames = np.asarray(frames, dtype=np.int64).reshape(-1)
    est = _as_track(est)
    truth = _as_track(truth)
    if frames.size != est.shape[0]:
        raise ValueError("frames и est разной длины")
    inside = (frames >= 0) & (frames < truth.shape[0])
    frames = frames[inside]
    est = est[inside]
    stats = compare_tracks(est, truth[frames]) if frames.size else compare_tracks(est[:0], truth[:0])
    total = int(total_frames) if total_frames is not None else int(truth.shape[0])
    stats["frames"] = int(frames.size)
    stats["coverage"] = float(np.unique(frames).size / total) if total > 0 else 0.0
    if frames.size:
        err = np.linalg.norm(est - truth[frames], axis=1)
        stats["p95_m"] = float(np.percentile(err, 95))
        stats["mean_m"] = float(np.mean(err))
    else:
        stats["p95_m"] = float("nan")
        stats["mean_m"] = float("nan")
    return stats


def frame_errors(frames: list[int] | np.ndarray, est: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Евклидова ошибка на каждом кадре (NaN, если кадра нет в таблице)."""
    frames = np.asarray(frames, dtype=np.int64).reshape(-1)
    est = _as_track(est)
    truth = _as_track(truth)
    out = np.full(frames.size, np.nan)
    inside = (frames >= 0) & (frames < truth.shape[0])
    out[inside] = np.linalg.norm(est[inside] - truth[frames[inside]], axis=1)
    return out


def write_metrics_report(stats: dict, path: Path) -> Path:
    """Текстовый отчёт UTF-8 с ключевыми метриками."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Метрики 3D-трассы", f"Порог попадания: {HIT_THRESHOLD_M:g} м", ""]
    for key, label in _METRIC_LABELS.items():
        if key not in stats and key in ("frames", "coverage", "mean_m", "p95_m"):
            continue
        value = stats.get(key, float("nan"))
        if key == "frames":
            lines.append(f"{label}: {int(value)}")
        elif key in ("hit_ratio", "coverage"):
            lines.append(f"{label}: {float(value):.4f}")
        elif np.isfinite(value):
            lines.append(f"{label}: {float(value):.6f}")
        else:
            lines.append(f"{label}: —")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
