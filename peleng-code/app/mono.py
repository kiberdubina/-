"""Одна камера без позы: примерный путь цели в поле 3D.

Дальность — по видимой высоте рамки: Z = f · H / h_px, где H — известная высота цели
(vehicle.height_m профиля, иначе d_m). Высота почти не зависит от разворота машины
к камере, в отличие от ширины. Фокус f неизвестен: берётся из условного угла обзора
MONO_HFOV_DEG. Поэтому метры здесь «примерно»: ошибка угла обзора и высоты цели
масштабирует весь путь, но форма пути (куда и как ехала) сохраняется.

Вычитание фона часто находит не всю машину, а кусок (кромку кабины, половину кузова).
Кусок почти всегда ниже машины и почти никогда не выше, поэтому высота цели — наибольшая
за последние HEIGHT_WINDOW_S, а не последняя и не медиана.

Камера условно стоит в нуле мира и смотрит вдоль +Y, Z — вверх.
"""

from __future__ import annotations

import math

import numpy as np

from app.geom import CameraPose, look_at

# Условный горизонтальный угол обзора камеры, градусы.
MONO_HFOV_DEG = 60.0
# Рамка ниже этого по высоте — дальность уже шум.
MIN_BOX_H_PX = 8
# Высота цели — наибольшая высота целой рамки за столько секунд.
HEIGHT_WINDOW_S = 1.0
# Пока путь моложе этого, его точки пересчитываются по уточнённой дальности:
# в первые кадры фон ещё учится и видна лишь часть машины.
SETTLE_S = 1.0
# Пропуск дольше этого или время назад (перемотка) — новый путь.
GAP_S = 1.5
# Сколько точек пути держим.
MAX_POINTS = 2000
# Рамка ближе этого к краю кадра — цель обрезана.
EDGE_PX = 3
# Дальность меняется не быстрее чем во столько раз за секунду.
DEPTH_RATE_PER_S = 1.5


def _touches_edge(bbox, width: int, height: int) -> bool:
    x, y, w, h = bbox
    return x <= EDGE_PX or y <= EDGE_PX or x + w >= width - EDGE_PX or y + h >= height - EDGE_PX


def target_height_m(profile) -> float | None:
    vehicle = getattr(profile, "vehicle", None) or {}
    height = vehicle.get("height_m")
    if height:
        return float(height)
    size = getattr(profile, "d_m", None)
    return float(size) if size else None


class MonoPath:
    def __init__(self, height_m: float, hfov_deg: float = MONO_HFOV_DEG):
        self.height_m = float(height_m)
        self.hfov_deg = float(hfov_deg)
        self.trail: list[np.ndarray] = []
        self.current: np.ndarray | None = None
        self.pred: np.ndarray | None = None
        self.depth_m: float | None = None
        # (время, высота целой рамки, px) за HEIGHT_WINDOW_S.
        self._heights: list[tuple[float, float]] = []
        # Пиксели точек пути, пока он моложе SETTLE_S: (время, u, v).
        self._fresh: list[tuple[float, float, float]] = []
        self._start_t: float | None = None
        self._last_t: float | None = None
        self._depth_t: float | None = None
        self._pose: CameraPose | None = None

    def _focal(self, width: int) -> float:
        return (width / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)

    def pose(self, width: int, height: int) -> CameraPose:
        """Условная камера для поля 3D: в нуле, смотрит вдоль +Y."""
        pose = self._pose
        if pose is None or pose.width != width or pose.height != height:
            focal = self._focal(width)
            matrix = np.array(
                [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
                np.float64,
            )
            rvec, tvec = look_at(np.zeros(3), np.array([0.0, 1.0, 0.0]))
            pose = CameraPose("камера (примерно)", matrix, np.zeros(5), rvec, tvec, width, height)
            self._pose = pose
        return pose

    def _world(self, u: float, v: float, depth: float, width: int, height: int) -> np.ndarray:
        """Пиксель на глубине depth → точка мира: x вправо, y вперёд от камеры, z вверх."""
        focal = self._focal(width)
        right = (u - width / 2.0) / focal * depth
        down = (v - height / 2.0) / focal * depth
        return np.array([right, depth, -down], np.float64)

    def update(self, track, width: int, height: int, time_s: float) -> None:
        """Цель пропала — пройденный путь остаётся на экране до новой цели или перемотки."""
        time_s = float(time_s)
        if self._last_t is not None and time_s < self._last_t:
            self.reset()
        bbox = getattr(track, "bbox", None)
        if not getattr(track, "active", False) or bbox is None or bbox[3] < MIN_BOX_H_PX:
            self.current = None
            self.pred = None
            return
        if self._last_t is not None and time_s - self._last_t > GAP_S:
            self.reset()
        self._last_t = time_s
        # Рамка у края кадра — цель видна не целиком: её высота в дальность не идёт.
        if getattr(track, "measured", True) and not _touches_edge(bbox, width, height):
            self._heights.append((time_s, float(bbox[3])))
        self._heights = [item for item in self._heights if time_s - item[0] <= HEIGHT_WINDOW_S]
        if not self._heights:
            # Целой рамки ещё не было (или давно не было) — дальность прежняя или её нет.
            if self.depth_m is None:
                self.current = None
                self.pred = None
                return
        else:
            tallest = max(height_px for _, height_px in self._heights)
            depth = self._focal(width) * self.height_m / tallest
            settled = self._start_t is not None and time_s - self._start_t > SETTLE_S
            if settled and self.depth_m is not None and self._depth_t is not None:
                # Дальность не прыгает: обрывок машины у края — не «уехала на 70 м за кадр».
                # В первую секунду — свободно: фон ещё учится, первые рамки — куски машины.
                limit = DEPTH_RATE_PER_S ** max(time_s - self._depth_t, 0.0)
                depth = min(max(depth, self.depth_m / limit), self.depth_m * limit)
            self.depth_m = depth
            self._depth_t = time_s
        depth = self.depth_m
        if self._start_t is None:
            self._start_t = time_s
        self.current = self._world(track.cx, track.cy, depth, width, height)
        self.pred = self._world(track.pred_cx, track.pred_cy, depth, width, height)
        # В путь — только измеренные целые рамки. Пропуск (трек дорисован по прогнозу,
        # иногда за край кадра) и обрезанная у края рамка давали «крючок» в конце пути.
        if not getattr(track, "measured", True) or _touches_edge(bbox, width, height):
            return
        if time_s - self._start_t <= SETTLE_S:
            # Путь молод: все его точки — по текущей, более точной дальности.
            self._fresh.append((time_s, float(track.cx), float(track.cy)))
            self.trail = [self._world(u, v, depth, width, height) for _, u, v in self._fresh]
        else:
            self._fresh.clear()
            self.trail.append(self.current.copy())
        del self.trail[:-MAX_POINTS]

    def reset(self) -> None:
        self.trail = []
        self.current = None
        self.pred = None
        self.depth_m = None
        self._heights.clear()
        self._fresh.clear()
        self._start_t = None
        self._last_t = None
        self._depth_t = None
