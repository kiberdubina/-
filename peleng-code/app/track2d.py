import math
from dataclasses import dataclass

import cv2
import numpy as np

from app.motion import Detection
from app.profile import TargetProfile

# Соседний кадр честного хода на ролике мяча — около 100 px.
# Слипшийся контур уводит центр сразу примерно на 180 px и в фильтр не входит.
# Допуск не растёт на пропуске: иначе прогноз уезжает за кадр и возврат мяча
# рисует скачок в сотни пикселей. На живом измерении допуск может вырасти
# со скоростью, но для обычного хода база остаётся 140 px, и скачок
# в сотни пикселей по-прежнему выбрасывается.
# После серии пропусков трек отпускает цель.
MEASUREMENT_GATE_PX = 140.0
# Штраф за площадь далеко от середины диапазона, в пикселях допуска.
AREA_COST_PX = 40.0
# Дрожание центра в слабом свете: ближе этого порога след не удлиняется.
TRAIL_MIN_PX = 3.0
# Скорость линии прогноза — по последним принятым измерениям.
VELOCITY_WINDOW = 4
# Потолок живого допуска: даже на большой скорости сотни пикселей не проходят.
GATE_CAP_PX = 280.0
# Новый трек — только после стольких измерений подряд. Вспышка листвы, тени или шума
# на кадр-другой не рисует рамку, след и прогноз и не идёт в 3D.
CONFIRM_HITS = 3


@dataclass
class TrackView:
    active: bool
    cx: float
    cy: float
    vx: float
    vy: float
    speed_px_s: float
    state: str
    in_gate: bool
    measured: bool
    pred_cx: float
    pred_cy: float
    bbox: tuple[int, int, int, int] | None
    pred_bbox: tuple[int, int, int, int] | None
    area: float | None
    trail: tuple[tuple[float, float], ...] = ()
    horizon_s: float = 0.5

    @property
    def fx(self) -> float:
        return self.pred_cx

    @property
    def fy(self) -> float:
        return self.pred_cy


def _shift_bbox(
    bbox: tuple[int, int, int, int], dx: float, dy: float
) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    return (int(round(x + dx)), int(round(y + dy)), int(w), int(h))


class Track2D:
    """Фильтр Калмана OpenCV: положение и скорость пятна по матрице.

    Измерение дальше прогноза выбрасывается: фильтр его не вносит,
    и в след такая точка не попадает. След — сами принятые измерения,
    без точек ближе 3 px к предыдущей. Линия прогноза на horizon_s идёт
    по скорости последних четырёх измерений, когда окно уже набрано.
    """

    def __init__(self, profile: TargetProfile):
        self.profile = profile
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], np.float32
        )
        # Шум процесса в пикселях и пикс/с. Малое значение держит точку на старом месте,
        # когда предмет ведут рукой и он меняет направление за один кадр.
        self.kf.processNoiseCov = np.diag([25, 25, 400, 400]).astype(np.float32)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 4
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 100
        mode = (getattr(profile, "mode", "auto") or "auto").lower()
        # Небо/дрон: реже плотные измерения, держим рамку дольше и шире по допуску.
        if mode in {"sky_dark", "white_blob"}:
            self.gate_px = 220.0
            self.miss_limit = 36
        else:
            self.gate_px = MEASUREMENT_GATE_PX
            self.miss_limit = 12
        self.ready = False
        self.miss = 0
        self.hits = 0
        self.last_t = 0.0
        self.accepted: Detection | None = None
        self.last_bbox: tuple[int, int, int, int] | None = None
        self.last_cx = 0.0
        self.last_cy = 0.0
        self.points: list[tuple[float, float]] = []
        # (x, y, время) принятых измерений. Окно короткое: после поворота
        # скорость не тащит за собой старое направление.
        self.samples: list[tuple[float, float, float]] = []

    @property
    def horizon_s(self) -> float:
        return float(self.profile.predict_s)

    @horizon_s.setter
    def horizon_s(self, value: float) -> None:
        self.profile.predict_s = float(value)

    def reset(self) -> None:
        self.ready = False
        self.miss = 0
        self.hits = 0
        self.last_t = 0.0
        self.accepted = None
        self.last_bbox = None
        self.last_cx = 0.0
        self.last_cy = 0.0
        self.points.clear()
        self.samples.clear()
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 100

    def update(self, spots: list[Detection] | Detection | None, time_s: float) -> TrackView:
        self.accepted = None
        found = self._as_spots(spots)
        if not self.ready:
            spot = self._pick_initial(found)
            if spot is None:
                return self._idle()
            self.kf.statePost = np.array(
                [[np.float32(spot.cx)], [np.float32(spot.cy)], [0], [0]], np.float32
            )
            self.ready = True
            self.last_t = time_s
            self.miss = 0
            self.hits = 1
            self._remember(spot, time_s)
            return self._view("захват", True, spot)

        dt = max(1e-3, time_s - self.last_t)
        self.last_t = time_s
        self.kf.transitionMatrix = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]],
            np.float32,
        )
        self.kf.predict()
        spot = self._pick_gated(found, self._gate_radius(dt))
        if spot is None:
            self.miss += 1
            # Неподтверждённый трек пропуска не переживает: это была вспышка, не цель.
            if self.miss > self.miss_limit or self.hits < CONFIRM_HITS:
                self.ready = False
                self.hits = 0
                self.last_bbox = None
                return self._idle()
            return self._view(None, False, None)

        measure = np.array([[np.float32(spot.cx)], [np.float32(spot.cy)]])
        self.kf.correct(measure)
        self.miss = 0
        self.hits += 1
        self._remember(spot, time_s)
        self._snap_filter(spot)
        return self._view(None, True, spot)

    def _gate_radius(self, dt: float) -> float:
        # На пропуске радиус остаётся базовым и не растёт вместе с прогнозом.
        if self.miss > 0:
            return self.gate_px
        vx, vy = self._measure_velocity()
        step_dt = dt
        if len(self.samples) >= 2:
            span = self.samples[-1][2] - self.samples[-2][2]
            if span > 1e-3:
                step_dt = span
        grown = math.hypot(vx, vy) * step_dt * 3.0
        return max(self.gate_px, min(grown, GATE_CAP_PX))

    def _measure_velocity(self) -> tuple[float, float]:
        """Скорость по последним четырём измерениям. Иначе — скорость фильтра.

        Медиана скоростей по всем парам точек (оценка Тейла–Сена): один скачок центра на
        соседнее пятно её не раздувает, и прогноз не улетает. Прямая МНК — раздувала.
        """
        if len(self.samples) >= VELOCITY_WINDOW:
            window = self.samples[-VELOCITY_WINDOW:]
            slopes_x = []
            slopes_y = []
            for first in range(len(window)):
                for second in range(first + 1, len(window)):
                    span = window[second][2] - window[first][2]
                    if span > 1e-4:
                        slopes_x.append((window[second][0] - window[first][0]) / span)
                        slopes_y.append((window[second][1] - window[first][1]) / span)
            if slopes_x:
                return float(np.median(slopes_x)), float(np.median(slopes_y))
        return float(self.kf.statePost[2, 0]), float(self.kf.statePost[3, 0])

    def _snap_filter(self, spot: Detection) -> None:
        # correct() уже внёс измерение. Положение дальше — само пятно, скорость —
        # по окну, чтобы следующий прогноз не срезал угол.
        vx, vy = self._measure_velocity()
        self.kf.statePost[0, 0] = np.float32(spot.cx)
        self.kf.statePost[1, 0] = np.float32(spot.cy)
        self.kf.statePost[2, 0] = np.float32(vx)
        self.kf.statePost[3, 0] = np.float32(vy)

    def _area_cost(self, area: float) -> float:
        mid = math.sqrt(max(self.profile.s_min, 1.0) * max(self.profile.s_max, 1.0))
        if area <= 0:
            return 0.0
        return abs(math.log(area / mid))

    def _as_spots(self, spots: list[Detection] | Detection | None) -> list[Detection]:
        return _as_spots(spots)

    def _pick_initial(self, spots: list[Detection]) -> Detection | None:
        if not spots:
            return None
        return max(spots, key=lambda spot: spot.area)

    def _pick_gated(self, spots: list[Detection], gate: float) -> Detection | None:
        px = float(self.kf.statePre[0, 0])
        py = float(self.kf.statePre[1, 0])
        best: Detection | None = None
        best_score: float | None = None
        for spot in spots:
            dist = math.hypot(spot.cx - px, spot.cy - py)
            if dist > gate:
                continue
            score = dist + self._area_cost(spot.area) * AREA_COST_PX
            if best_score is None or score < best_score:
                best = spot
                best_score = score
        return best

    def _remember(self, spot: Detection, time_s: float) -> None:
        self.accepted = spot
        self.last_bbox = spot.bbox
        self.last_cx = float(spot.cx)
        self.last_cy = float(spot.cy)
        self.samples.append((float(spot.cx), float(spot.cy), float(time_s)))
        if len(self.samples) > VELOCITY_WINDOW:
            del self.samples[:-VELOCITY_WINDOW]
        point = (float(spot.cx), float(spot.cy))
        if self.points:
            last_x, last_y = self.points[-1]
            if math.hypot(point[0] - last_x, point[1] - last_y) < TRAIL_MIN_PX:
                return
        self.points.append(point)

    def _idle(self) -> TrackView:
        self.points.clear()
        self.samples.clear()
        return self._pack(False, 0, 0, 0, 0, "поиск", False, False, None, None)

    def _view(self, state: str | None, measured: bool, spot: Detection | None) -> TrackView:
        vx, vy = self._measure_velocity()
        if measured and spot is not None:
            cx = float(spot.cx)
            cy = float(spot.cy)
        elif self.samples:
            x, y, stamp = self.samples[-1]
            elapsed = max(0.0, self.last_t - stamp)
            cx = x + vx * elapsed
            cy = y + vy * elapsed
        else:
            cx = float(self.kf.statePost[0, 0])
            cy = float(self.kf.statePost[1, 0])
        speed = float(math.hypot(vx, vy))
        horizon = max(0.0, self.horizon_s)
        in_gate = self.profile.u_min <= speed <= self.profile.u_max
        confirmed = self.hits >= CONFIRM_HITS
        if not confirmed:
            state = "проверка"
        elif state is None:
            state = "сопровождение" if in_gate else "вне порога скорости"
        if spot is not None:
            bbox = spot.bbox
        elif self.last_bbox is not None:
            bbox = _shift_bbox(self.last_bbox, cx - self.last_cx, cy - self.last_cy)
        else:
            bbox = None
        pred_bbox = None if bbox is None else _shift_bbox(bbox, vx * horizon, vy * horizon)
        area = None if spot is None else spot.area
        if not confirmed:
            # Ещё не цель: без рамки, следа и прогноза на экране и без измерения для 3D.
            return self._pack(False, cx, cy, vx, vy, state, False, False, None, None, area, horizon)
        return self._pack(
            True, cx, cy, vx, vy, state, in_gate, measured, bbox, pred_bbox, area, horizon
        )

    def _pack(
        self,
        active: bool,
        cx: float,
        cy: float,
        vx: float,
        vy: float,
        state: str,
        in_gate: bool,
        measured: bool,
        bbox: tuple[int, int, int, int] | None,
        pred_bbox: tuple[int, int, int, int] | None,
        area: float | None = None,
        horizon: float | None = None,
    ) -> TrackView:
        horizon = self.horizon_s if horizon is None else horizon
        return TrackView(
            active,
            cx,
            cy,
            vx,
            vy,
            float(math.hypot(vx, vy)),
            state,
            in_gate,
            measured,
            cx + vx * horizon,
            cy + vy * horizon,
            bbox,
            pred_bbox,
            area,
            tuple(self.points),
            horizon,
        )


def _as_spots(spots: list[Detection] | Detection | None) -> list[Detection]:
    if spots is None:
        return []
    if isinstance(spots, Detection):
        return [spots]
    return list(spots)


# Сколько одновременных 2D-треков держим на кадре.
MAX_TRACKS_2D = 8
# Новый трек только если пятно не слишком далеко от уже ведущихся целей.
SPAWN_NEAR_TRACK_PX = 420.0
# Выбранная машина: клик попал в рамку или ближе этого к центру пятна.
SELECT_NEAR_PX = 40.0
# Выбранная потерялась: трек ближе этого к её последнему месту и не позже — она же.
RELOCK_PX = 150.0
RELOCK_S = 3.0


class MultiTrack2D:
    """Несколько Track2D: пятна раздаются по очереди, ворота не дают перехватить чужим."""

    def __init__(self, profile: TargetProfile):
        self._profile = profile
        self._tracks: list[Track2D] = []
        self._last_views: list[TrackView] = []
        # Выбор машины оператором: главный трек закреплён за ней и не уходит на соседнюю.
        self._locked = False
        self._selected: Track2D | None = None
        self._lost: tuple[float, float, float] | None = None  # x, y, время потери
        self._last_spots: list[Detection] = []
        self._last_time = 0.0

    @property
    def locked(self) -> bool:
        """Оператор выбрал машину (даже если она сейчас потеряна)."""
        return self._locked

    @property
    def has_target(self) -> bool:
        """Выбранная машина сейчас ведётся."""
        return self._locked and self._selected is not None

    def select_at(self, x: float, y: float) -> bool:
        """Выбрать машину под точкой кадра: трек с рамкой здесь или пятно последнего кадра."""
        hit = None
        for track in self._tracks:
            if track.ready and _inside(track.last_bbox, x, y, track.last_cx, track.last_cy):
                hit = track
                break
        if hit is None:
            spot = next(
                (spot for spot in self._last_spots if _inside(spot.bbox, x, y, spot.cx, spot.cy)),
                None,
            )
            if spot is None:
                return False
            hit = Track2D(self._profile)
            hit.update([spot], self._last_time)
            if not hit.ready:
                return False
            self._tracks.insert(0, hit)
            self._last_views.insert(0, hit.update([spot], self._last_time))
        self._take(hit)
        return True

    def select_index(self, index: int) -> bool:
        """Выбрать трек по номеру в tracks (для той же машины на другой камере)."""
        ready = [track for track in self._tracks if track.ready]
        if not 0 <= index < len(ready):
            return False
        self._take(ready[index])
        return True

    def clear_selection(self) -> None:
        self._locked = False
        self._selected = None
        self._lost = None

    def _take(self, track: Track2D) -> None:
        index = self._tracks.index(track)
        self._tracks.insert(0, self._tracks.pop(index))
        if index < len(self._last_views):
            self._last_views.insert(0, self._last_views.pop(index))
        self._locked = True
        self._selected = track
        self._lost = None

    @property
    def profile(self) -> TargetProfile:
        return self._profile

    @profile.setter
    def profile(self, value: TargetProfile) -> None:
        self._profile = value
        for track in self._tracks:
            track.profile = value

    @property
    def horizon_s(self) -> float:
        return float(self._profile.predict_s)

    @horizon_s.setter
    def horizon_s(self, value: float) -> None:
        self._profile.predict_s = float(value)

    @property
    def accepted(self) -> Detection | None:
        primary = self._primary_track()
        return None if primary is None else primary.accepted

    def reset(self) -> None:
        self._tracks.clear()
        self._last_views.clear()
        self.clear_selection()
        self._last_spots = []

    def update(self, spots: list[Detection] | Detection | None, time_s: float) -> TrackView:
        self._last_spots = list(_as_spots(spots))
        self._last_time = float(time_s)
        views = self._update_all(spots, time_s)
        if self._locked:
            self._follow_selected(time_s)
            # Выбранная потеряна — главного трека нет: соседняя машина в 3D не попадёт.
            if self._selected is None:
                return self._idle_view()
            return self._last_views[0]
        return views[0] if views else self._idle_view()

    def _follow_selected(self, time_s: float) -> None:
        if self._selected is not None and self._selected not in self._tracks:
            self._lost = (self._selected.last_cx, self._selected.last_cy, float(time_s))
            self._selected = None
        if self._selected is None and self._lost is not None:
            x, y, since = self._lost
            if time_s - since <= RELOCK_S:
                near = [
                    track
                    for track in self._tracks
                    if track.ready and math.hypot(track.last_cx - x, track.last_cy - y) <= RELOCK_PX
                ]
                if near:
                    best = min(near, key=lambda track: math.hypot(track.last_cx - x, track.last_cy - y))
                    self._take(best)
                    return
        if self._selected is not None and self._tracks and self._tracks[0] is not self._selected:
            self._take(self._selected)

    @property
    def tracks(self) -> tuple[TrackView, ...]:
        return tuple(self._last_views)

    def _idle_view(self) -> TrackView:
        return TrackView(
            False,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            "поиск",
            False,
            False,
            0.0,
            0.0,
            None,
            None,
            None,
            (),
            self.horizon_s,
        )

    def _primary_track(self) -> Track2D | None:
        if self._locked:
            return self._selected
        for track in self._tracks:
            if track.ready:
                return track
        return None

    def _update_all(
        self, spots: list[Detection] | Detection | None, time_s: float
    ) -> list[TrackView]:
        remaining = _as_spots(spots)
        self._last_views = []

        if not self._tracks:
            starter = Track2D(self._profile)
            view = starter.update(remaining, time_s)
            if starter.ready:
                self._tracks.append(starter)
                self._last_views.append(view)
                self._drop_spot(remaining, starter.accepted)
            self._spawn_secondary(remaining, time_s)
            return self._finish_views()

        # Главный трек (индекс 0) первым забирает ближайшее пятно в воротах.
        order = list(range(len(self._tracks)))
        order.sort(key=lambda index: (0 if index == 0 else 1, index))
        kept: list[tuple[int, Track2D, TrackView]] = []
        for index in order:
            track = self._tracks[index]
            view = track.update(remaining, time_s)
            if track.ready:
                kept.append((index, track, view))
            self._drop_spot(remaining, track.accepted)

        kept.sort(key=lambda item: item[0])
        self._tracks = [track for _, track, _ in kept]
        self._last_views = [view for _, _, view in kept]
        self._spawn_secondary(remaining, time_s)
        return self._finish_views()

    @staticmethod
    def _drop_spot(remaining: list[Detection], spot: Detection | None) -> None:
        if spot is not None and spot in remaining:
            remaining.remove(spot)

    def _spawn_secondary(self, remaining: list[Detection], time_s: float) -> None:
        for spot in list(remaining):
            if len(self._tracks) >= MAX_TRACKS_2D:
                break
            if not self._should_spawn(spot):
                continue
            track = Track2D(self._profile)
            view = track.update([spot], time_s)
            if not track.ready:
                continue
            self._tracks.append(track)
            self._last_views.append(view)
            self._drop_spot(remaining, spot)

    def _should_spawn(self, spot: Detection) -> bool:
        if self._locked:
            # Поток машин: треки по всему кадру — выбранная могла вернуться где угодно.
            return True
        ready = [track for track in self._tracks if track.ready]
        if not ready:
            return True
        nearest = min(
            math.hypot(spot.cx - track.last_cx, spot.cy - track.last_cy) for track in ready
        )
        return nearest <= SPAWN_NEAR_TRACK_PX

    def _finish_views(self) -> list[TrackView]:
        if not self._last_views:
            idle = self._idle_view()
            self._last_views = [idle]
            return [idle]
        return list(self._last_views)


def _inside(bbox, x: float, y: float, cx: float, cy: float) -> bool:
    """Точка в рамке или ближе SELECT_NEAR_PX к центру."""
    if bbox is not None:
        left, top, width, height = bbox
        if left <= x <= left + width and top <= y <= top + height:
            return True
    return math.hypot(x - cx, y - cy) <= SELECT_NEAR_PX
