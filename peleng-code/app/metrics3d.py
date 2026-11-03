"""Погрешность 3D-трассы относительно object_transform_SM_AA_track.csv."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.cine_table import track_meters

# Доля кадров с евклидовой ошибкой не больше порога, м.
HIT_THRESHOLD_M = 10.0

_METRIC_LABELS = {
    "rmse_m": "RMSE, м",
    "median_m": "Медиана ошибки, м",
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


def write_metrics_report(stats: dict, path: Path) -> Path:
    """Текстовый отчёт UTF-8 с ключевыми метриками."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Метрики 3D-трассы", f"Порог попадания: {HIT_THRESHOLD_M:g} м", ""]
    for key, label in _METRIC_LABELS.items():
        value = stats.get(key, float("nan"))
        if key == "hit_ratio":
            lines.append(f"{label}: {float(value):.4f}")
        elif np.isfinite(value):
            lines.append(f"{label}: {float(value):.6f}")
        else:
            lines.append(f"{label}: —")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
