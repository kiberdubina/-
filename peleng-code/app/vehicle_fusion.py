"""Слияние камер для наземной машины: пятна → рамки → итерационный EKF по 3D-ящику.

Один шаг:
  1. Прогноз состояния на момент кадра.
  2. В каждой камере — рамка прогнозного ящика. Пятна рядом с ней собираются жадно:
     пятно входит, если увеличивает IoU объединённой рамки с прогнозной. Так силуэт,
     разрезанный деревом, склеивается, а «призрак» или листва рядом — нет.
  3. Обновление EKF по сторонам рамок всех камер сразу.

Старт без прогноза: пары крупных пятен из двух камер триангулируются (центр рамки ≈
луч на тело машины), затем курс перебирается по кругу и ящик подгоняется к рамкам
симметричной робастной функцией. Перед/зад решает движение (VehicleTracker).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.background import Blob
from app.geom import CameraPose, range_from_cameras, triangulate
from app.vehicle import (
    OMEGA,
    PSI,
    V,
    BoxObservation,
    MotionNoise,
    SideNoise,
    UpdateResult,
    VehicleModel,
    VehicleTracker,
    anchor_point,
    box_corners,
    numeric_jacobian,
    predicted_sides,
    robust_update,
    state_from_center,
)


@dataclass
class FusionConfig:
    gate_share: float = 0.25        # окно поиска пятен: доля размера прогнозной рамки
    gate_px: float = 15.0           # плюс постоянный запас, px
    blob_share: float = 0.01        # пятно меньше этой доли прогнозной рамки — шум
    blob_min_px: float = 30.0       # и меньше этой площади — тоже
    min_iou: float = 0.15           # объединённая рамка беднее — камера без измерения,
    contained_share: float = 0.85   # если только она не лежит внутри прогноза (окклюзия)
    contained_area: float = 0.02    # и не мельче этой доли прогнозной рамки
    init_min_area_px: float = 300.0 # пятна для старта — не мельче
    init_blobs_per_camera: int = 3
    init_headings: int = 12
    init_rms_px: float = 12.0       # подгонка хуже — старта нет
    # σ старта: x, y, z, курс, тангаж, скорость, поворот.
    init_spread: tuple[float, ...] = (100.0, 100.0, 100.0, 3.2, 0.15, 10.0, 0.5)
    inside_share: float = 0.5       # пятно должно лежать в окне поиска хотя бы этой долей
    miss_limit: int = 60            # столько шагов без измерений — трек закрыт
    range_m: tuple[float, float] = (1.0, 2000.0)
    support_iou: float = 0.5        # камера «подтверждает» ящик: силуэт совпал с проекцией
    reacquire_below: int = 2        # опора трека ниже — ищем цель заново
    reacquire_every: int = 10       # не чаще, чем раз в столько шагов
    reacquire_jump_m: float = 5.0   # новая гипотеза дальше этого от трека — это перезахват
    # Старт по направляющей траектории (итерация сглаживания): σ x, y, z, курс, тангаж, v, ω.
    guide_spread: tuple[float, ...] = (2.0, 2.0, 2.0, 0.1, 0.05, 3.0, 0.2)
    # Подтверждение трека «M из N»: новая гипотеза пробная, пока опора ≥ reacquire_below
    # не наберётся confirm_hits раз за confirm_window шагов. Иначе гипотеза сбрасывается.
    confirm_hits: int = 4
    confirm_window: int = 8


@dataclass
class CameraView:
    pose: CameraPose
    blobs: list[Blob]


@dataclass
class VehicleView:
    active: bool
    fresh: bool = False
    state: np.ndarray | None = None
    anchor: np.ndarray | None = None
    corners: np.ndarray | None = None
    speed_m_s: float = 0.0
    heading_rad: float = 0.0
    observed: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)
    rms_px: float | None = None
    # Неопределённость опорной точки, м: σ по худшему направлению из ковариации фильтра.
    # Мала — геометрия камер задаёт положение; велика — по одной камере дальность не измерена.
    sigma_m: float | None = None


def anchor_sigma(state: np.ndarray, covariance: np.ndarray, model: VehicleModel) -> float:
    """σ опорной точки по худшему направлению: J·P·Jᵀ, J — якобиан опорной точки по состоянию."""
    jacobian = numeric_jacobian(lambda candidate: anchor_point(candidate, model), state)
    spread = jacobian @ covariance @ jacobian.T
    return float(np.sqrt(max(float(np.linalg.eigvalsh(0.5 * (spread + spread.T)).max()), 0.0)))


@dataclass
class Segment:
    """Непрерывный кусок трека: ключи шагов (номера кадров) и фильтр с историей."""

    keys: list
    tracker: VehicleTracker
    confirmed: bool = False
    hits: int = 0
    steps: int = 0


def _iou(a: tuple[float, float, float, float], b: np.ndarray) -> float:
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(0.0, ix) * max(0.0, iy)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def gather_box(
    predicted: np.ndarray, blobs: list[Blob], config: FusionConfig
) -> tuple[float, float, float, float] | None:
    """Рамка силуэта цели из пятен у прогнозной рамки (x0, y0, x1, y1) или None."""
    width = float(predicted[2] - predicted[0])
    height = float(predicted[3] - predicted[1])
    if width < 2.0 or height < 2.0:
        return None
    pad_x = config.gate_share * width + config.gate_px
    pad_y = config.gate_share * height + config.gate_px
    gate = (predicted[0] - pad_x, predicted[1] - pad_y, predicted[2] + pad_x, predicted[3] + pad_y)
    floor = max(config.blob_min_px, config.blob_share * width * height)
    near = []
    for blob in blobs:
        if blob.area < floor:
            continue
        x0, y0, x1, y1 = blob.rect
        overlap_w = min(x1, gate[2]) - max(x0, gate[0])
        overlap_h = min(y1, gate[3]) - max(y0, gate[1])
        if overlap_w <= 0 or overlap_h <= 0:
            continue
        # Пятно, которое в основном вне окна, — не силуэт цели, а её склейка с чужим
        # (артефакт фона, соседний объект): целиком его не берём.
        if overlap_w * overlap_h < config.inside_share * max(blob.w * blob.h, 1e-9):
            continue
        near.append(blob)
    rect = None
    best = 0.0
    for blob in sorted(near, key=lambda item: -item.area):
        x0, y0, x1, y1 = blob.rect
        grown = (x0, y0, x1, y1) if rect is None else (
            min(rect[0], x0), min(rect[1], y0), max(rect[2], x1), max(rect[3], y1)
        )
        score = _iou(grown, predicted)
        if score > best:
            rect, best = grown, score
    if rect is None:
        return None
    if best >= config.min_iou:
        return rect
    # Цель закрыта склоном или деревом: видна лишь часть силуэта. Такая рамка мала
    # по IoU, но лежит внутри прогноза — её стороны «внутрь» погасит функция Коши,
    # а видимые стороны остаются полноценным измерением.
    area = (rect[2] - rect[0]) * (rect[3] - rect[1])
    inside_w = min(rect[2], predicted[2] + config.gate_px) - max(rect[0], predicted[0] - config.gate_px)
    inside_h = min(rect[3], predicted[3] + config.gate_px) - max(rect[1], predicted[1] - config.gate_px)
    inside = max(0.0, inside_w) * max(0.0, inside_h)
    if area >= config.contained_area * width * height and inside >= config.contained_share * area:
        return rect
    return None


def _touches_border(blob: Blob, pose: CameraPose, edge: float = 2.0) -> bool:
    x0, y0, x1, y1 = blob.rect
    return x0 <= edge or y0 <= edge or x1 >= pose.width - edge or y1 >= pose.height - edge


class VehicleFusion:
    def __init__(
        self,
        model: VehicleModel,
        config: FusionConfig | None = None,
        motion: MotionNoise | None = None,
        sides: SideNoise | None = None,
    ):
        self.model = model
        self.config = config or FusionConfig()
        self.motion = motion or MotionNoise()
        self.sides = sides or SideNoise()
        self.tracker = self._new_tracker()
        self.segments: list[Segment] = []
        self.support = 0
        self.reacquired = 0
        self._since_check = 0
        self.guide: dict | None = None

    def set_guide(self, states: dict | None) -> None:
        """Направляющая траектория: ключ шага → состояние (сглаженное прошлой итерацией).

        С ней пятна сопоставляются не с прогнозом прямого фильтра, а с оценкой, которая
        уже учла всю запись, — так сопоставление на трудных участках не наследует увод
        прямого фильтра. Это итерационное сглаживание: каждая итерация уточняет и
        сопоставление, и траекторию.
        """
        self.guide = dict(states) if states else None

    def _new_tracker(self) -> VehicleTracker:
        return VehicleTracker(self.model, motion=self.motion, sides=self.sides)

    def reset(self) -> None:
        self.tracker = self._new_tracker()
        self.segments = []
        self.support = 0
        self.reacquired = 0
        self._since_check = 0

    # ------------------------------------------------------------ шаг

    def step(self, time_s: float, views: dict[str, CameraView], key=None) -> VehicleView:
        tracker = self.tracker
        guided = self.guide is not None and key in self.guide
        if not tracker.ready:
            if guided:
                return self._start_from_guide(time_s, views, key)
            return self._start(time_s, views, key)

        cfg = self.config
        predicted, _, _ = tracker.predict_to(time_s)
        reference = self.guide[key] if guided else predicted
        observations, used, support = self._associate(reference, views)
        self._since_check += 1
        # С направляющей траекторией сопоставление ведёт она, а кадры без неё (дыры
        # прошлой итерации) фильтр проходит по прогнозу: перезахват там только навредит.
        searching = self.guide is None
        if searching and support < cfg.reacquire_below and self._since_check >= cfg.reacquire_every:
            # Трек держится меньше чем на двух камерах: возможно, он ушёл с цели на мусор.
            # Ищем цель заново по всем пятнам; гипотеза с большей опорой вдали — перезахват.
            self._since_check = 0
            found = self._initialize(views)
            if found is not None and found[3] > support:
                jump = float(np.linalg.norm(anchor_point(found[0], self.model) - anchor_point(predicted, self.model)))
                if jump > cfg.reacquire_jump_m:
                    self.reacquired += 1
                    return self._start(time_s, views, key, found)
        result = tracker.step(time_s, observations)
        self.support = support
        segment = self.segments[-1] if self.segments and self.segments[-1].tracker is tracker else None
        if segment is not None:
            segment.keys.append(key)
            if not segment.confirmed:
                segment.steps += 1
                segment.hits += int(support >= cfg.reacquire_below)
                if segment.hits >= cfg.confirm_hits:
                    segment.confirmed = True
                elif segment.steps >= cfg.confirm_window:
                    # Пробная гипотеза не подтвердилась — это была случайная пара пятен.
                    self.segments.pop()
                    tracker.stop()
                    return VehicleView(False)
        if tracker.misses > cfg.miss_limit:
            tracker.stop()
            return VehicleView(False)
        if segment is not None and not segment.confirmed:
            return VehicleView(False)
        return self._view(result is not None, used, None if result is None else result.rms_px)

    def _start_from_guide(self, time_s: float, views: dict[str, CameraView], key) -> VehicleView:
        assert self.guide is not None
        state = np.asarray(self.guide[key], np.float64).copy()
        covariance = np.diag(np.asarray(self.config.guide_spread, np.float64) ** 2)
        tracker = self._new_tracker()
        tracker.start(state, covariance, time_s)
        self.tracker = tracker
        _, used, self.support = self._associate(state, views)
        # Направляющая — уже подтверждённая траектория прошлой итерации.
        self.segments.append(Segment([key], tracker, confirmed=True))
        return self._view(True, used, None)

    def _start(self, time_s: float, views: dict[str, CameraView], key, found=None) -> VehicleView:
        found = found if found is not None else self._initialize(views)
        if found is None:
            return VehicleView(False)
        state, covariance, used, support = found
        tracker = self._new_tracker()
        tracker.start(state, covariance, time_s)
        self.tracker = tracker
        self.support = support
        self._since_check = 0
        segment = Segment([key], tracker, hits=int(support >= self.config.reacquire_below))
        segment.confirmed = segment.hits >= self.config.confirm_hits
        self.segments.append(segment)
        if not segment.confirmed:
            return VehicleView(False)
        return self._view(True, used, None)

    def _view(self, fresh: bool, used: dict, rms: float | None) -> VehicleView:
        state = self.tracker.state
        assert state is not None
        covariance = self.tracker.covariance
        sigma = None if covariance is None else anchor_sigma(state, covariance, self.model)
        return VehicleView(
            active=True,
            fresh=fresh,
            state=state.copy(),
            anchor=anchor_point(state, self.model),
            corners=box_corners(state, self.model),
            speed_m_s=abs(float(state[V])),
            heading_rad=float(state[PSI]),
            observed=used,
            rms_px=rms,
            sigma_m=sigma,
        )

    def _associate(
        self, state: np.ndarray, views: dict[str, CameraView]
    ) -> tuple[list[BoxObservation], dict, int]:
        """Рамки силуэтов у проекции ящика по всем камерам и опора: сколько камер
        дали силуэт, совпавший с проекцией не хуже support_iou."""
        observations = []
        used = {}
        support = 0
        for camera_id, view in views.items():
            predicted = predicted_sides(state, self.model, view.pose)
            if predicted is None:
                continue
            rect = gather_box(predicted, view.blobs, self.config)
            if rect is None:
                continue
            if _iou(rect, predicted) >= self.config.support_iou:
                support += 1
            bbox = (rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1])
            observations.append(BoxObservation(view.pose, bbox))
            used[camera_id] = bbox
        return observations, used, support

    # ------------------------------------------------------------ старт

    def _initialize(self, views: dict[str, CameraView]):
        cfg = self.config
        candidates: dict[str, list[Blob]] = {}
        for camera_id, view in views.items():
            fit = [
                blob
                for blob in view.blobs
                if blob.area >= cfg.init_min_area_px and not _touches_border(blob, view.pose)
            ]
            fit.sort(key=lambda blob: -blob.area)
            if fit:
                candidates[camera_id] = fit[: cfg.init_blobs_per_camera]
        ids = list(candidates)
        # Гипотеза: ящик, подогнанный к паре пятен двух камер. Лучшая — с наибольшей
        # опорой по всем камерам, при равной опоре — с меньшей невязкой пары.
        best: tuple[int, float, UpdateResult, dict] | None = None
        for i, first in enumerate(ids):
            for second in ids[i + 1 :]:
                for blob_a in candidates[first]:
                    for blob_b in candidates[second]:
                        result = self._try_pair(views[first].pose, blob_a, views[second].pose, blob_b)
                        if result is None:
                            continue
                        _, used, support = self._associate(result.state, views)
                        if best is None or (support, -result.rms_px) > (best[0], -best[1]):
                            best = (support, result.rms_px, result, used)
        if best is None:
            return None
        support, _, result, used = best
        covariance = result.covariance.copy()
        spread = np.asarray(cfg.init_spread, np.float64)
        # Скорость и поворот рамки одного кадра не измеряют.
        for index in (V, OMEGA):
            covariance[index, :] = 0.0
            covariance[:, index] = 0.0
            covariance[index, index] = spread[index] ** 2
        return result.state, covariance, used, support

    def _try_pair(
        self, pose_a: CameraPose, blob_a: Blob, pose_b: CameraPose, blob_b: Blob
    ) -> UpdateResult | None:
        cfg = self.config
        pixels = [
            (blob_a.x + 0.5 * blob_a.w, blob_a.y + 0.5 * blob_a.h),
            (blob_b.x + 0.5 * blob_b.w, blob_b.y + 0.5 * blob_b.h),
        ]
        center = triangulate([pose_a, pose_b], pixels)
        if center is None:
            return None
        distance = range_from_cameras(center, [pose_a, pose_b])
        if not (cfg.range_m[0] <= distance <= cfg.range_m[1]):
            return None
        observations = [
            BoxObservation(pose_a, (blob_a.x, blob_a.y, blob_a.w, blob_a.h)),
            BoxObservation(pose_b, (blob_b.x, blob_b.y, blob_b.w, blob_b.h)),
        ]
        prior = np.diag(np.asarray(cfg.init_spread, np.float64) ** 2)
        symmetric = SideNoise(sigma_px=self.sides.sigma_px, huber_px=self.sides.huber_px, symmetric=True)
        best = None
        for psi in np.linspace(-np.pi, np.pi, cfg.init_headings, endpoint=False):
            start = state_from_center(center, float(psi), self.model)
            result = robust_update(start, prior, self.model, observations, symmetric, iterations=30)
            if result is not None and (best is None or result.rms_px < best.rms_px):
                best = result
        if best is None or best.rms_px > cfg.init_rms_px:
            return None
        return best

    # ------------------------------------------------------------ итог по записи

    def smoothed_states(self) -> dict:
        """Ключ шага → состояние после сглаживания RTS (по каждому сегменту отдельно)."""
        result = {}
        for segment in self.segments:
            if not segment.confirmed:
                continue
            for key, state in zip(segment.keys, segment.tracker.smoothed()):
                result[key] = state
        return result

    def filtered_states(self) -> dict:
        result = {}
        for segment in self.segments:
            if not segment.confirmed:
                continue
            for key, record in zip(segment.keys, segment.tracker.history):
                result[key] = record.filtered.copy()
        return result
