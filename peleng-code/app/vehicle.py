"""Наземная машина как 3D-ящик: модель движения CTRV и итерационный EKF по рамкам камер.

Состояние s = [x, y, z, psi, theta, v, omega]:
  x, y, z — точка задней оси на земле (м, мир программы, Z вверх);
  psi     — курс (рад, от оси X к оси Y);
  theta   — тангаж (рад, нос вверх > 0): машина повторяет уклон дороги;
  v       — скорость вдоль продольной оси корпуса (м/с);
  omega   — скорость поворота курса (рад/с).

Задняя ось выбрана опорой потому, что у машины без бокового скольжения её скорость
направлена строго по продольной оси. Для любой другой точки корпуса это не так: перед
в повороте заносит наружу на угол atan(d / R), где d — расстояние до задней оси,
R — радиус поворота. На подъёме та же длина d поднимает перед на d·sin(theta):
у фуры с d ≈ 15 м уклон 15° — это ±4 м по высоте опорной точки.

Измерение одной камеры — рамка силуэта (u0, v0, u1, v1). Прогноз — рамка выпуклой
оболочки восьми углов ящика после обрезки прямоугольником кадра. Сторона силуэта,
упёртая в край кадра, ничего не измеряет и выбрасывается. Окклюзия (дерево, склон)
только уменьшает видимый силуэт. Поэтому отклонение стороны «внутрь» прогноза идёт
через убывающую функцию Коши (влияние большой окклюзии стремится к нулю), а «наружу» —
через Хьюбера (силуэт шире ящика — сильное свидетельство, влияние не гаснет).

Рамки не отличают перед ящика от зада. Перед определяется движением: машина едет
передом, и устойчивая отрицательная скорость разворачивает всю гипотезу на пол-оборота.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

STATE_DIM = 7
X, Y, Z, PSI, THETA, V, OMEGA = range(STATE_DIM)

# Порядок углов VehicleModel.local_corners: индекс = 4*ix + 2*iy + iz
# (ix: зад/перед, iy: правый/левый борт, iz: низ/верх).
BOX_EDGES = (
    (0, 1), (2, 3), (4, 5), (6, 7),
    (0, 2), (1, 3), (4, 6), (5, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


@dataclass(frozen=True)
class VehicleModel:
    """Габарит относительно задней оси на земле: вперёд, влево, вверх (м)."""

    front_m: float
    rear_m: float
    width_m: float
    height_m: float
    # Точка, которой отвечают внешние координаты (pivot 3D-модели), в той же системе.
    anchor_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def length_m(self) -> float:
        return self.front_m - self.rear_m

    @property
    def middle_m(self) -> float:
        return 0.5 * (self.front_m + self.rear_m)

    def local_corners(self) -> np.ndarray:
        half = 0.5 * self.width_m
        return np.array(
            [
                (x, y, z)
                for x in (self.rear_m, self.front_m)
                for y in (-half, half)
                for z in (0.0, self.height_m)
            ],
            np.float64,
        )

    @classmethod
    def from_config(cls, data: dict) -> "VehicleModel":
        anchor = data.get("anchor_m", (0.0, 0.0, 0.0))
        return cls(
            front_m=float(data["front_m"]),
            rear_m=float(data["rear_m"]),
            width_m=float(data["width_m"]),
            height_m=float(data["height_m"]),
            anchor_m=(float(anchor[0]), float(anchor[1]), float(anchor[2])),
        )


# ---------------------------------------------------------------- геометрия ящика


def wrap_angle(angle: float) -> float:
    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


def body_axes(psi: float, theta: float = 0.0) -> np.ndarray:
    """Строки: вперёд, влево, вверх корпуса в мире. Курс psi, затем тангаж theta."""
    cp, sp = float(np.cos(psi)), float(np.sin(psi))
    ct, st = float(np.cos(theta)), float(np.sin(theta))
    forward = [ct * cp, ct * sp, st]
    left = [-sp, cp, 0.0]
    up = [-st * cp, -st * sp, ct]
    return np.array([forward, left, up], np.float64)


def state_axes(state: np.ndarray) -> np.ndarray:
    return body_axes(float(state[PSI]), float(state[THETA]))


def to_world(state: np.ndarray, local: np.ndarray) -> np.ndarray:
    axes = state_axes(state)
    return np.asarray(state[:3], np.float64) + np.asarray(local, np.float64).reshape(-1, 3) @ axes


def box_corners(state: np.ndarray, model: VehicleModel) -> np.ndarray:
    return to_world(state, model.local_corners())


def anchor_point(state: np.ndarray, model: VehicleModel) -> np.ndarray:
    return to_world(state, np.asarray(model.anchor_m, np.float64))[0]


def box_center(state: np.ndarray, model: VehicleModel) -> np.ndarray:
    local = np.array([model.middle_m, 0.0, 0.5 * model.height_m], np.float64)
    return to_world(state, local)[0]


def state_from_center(
    center: np.ndarray,
    psi: float,
    model: VehicleModel,
    v: float = 0.0,
    omega: float = 0.0,
    theta: float = 0.0,
) -> np.ndarray:
    """Состояние, у которого центр ящика стоит в center при курсе psi и тангаже theta."""
    axes = body_axes(psi, theta)
    local = np.array([model.middle_m, 0.0, 0.5 * model.height_m], np.float64)
    base = np.asarray(center, np.float64).reshape(3) - local @ axes
    state = np.zeros(STATE_DIM, np.float64)
    state[:3] = base
    state[PSI] = wrap_angle(psi)
    state[THETA] = theta
    state[V] = v
    state[OMEGA] = omega
    return state


def flipped(state: np.ndarray, model: VehicleModel) -> np.ndarray:
    """Тот же ящик, развёрнутый на пол-оборота вокруг вертикали через центр: перед ↔ зад.

    Нос вверх становится носом вниз (тангаж меняет знак), езда передом — ездой задом.
    """
    center = box_center(state, model)
    psi = wrap_angle(float(state[PSI]) + np.pi)
    return state_from_center(
        center, psi, model, v=-float(state[V]), omega=float(state[OMEGA]), theta=-float(state[THETA])
    )


# ---------------------------------------------------------------- движение CTRV


def ctrv_step(
    state: np.ndarray, dt: float, yaw_tau_s: float = 2.0, pitch_tau_s: float = 0.0
) -> np.ndarray:
    """CTRV на шаг dt с уклоном: по плану — дуга со скоростью v·cos(theta),
    по высоте — подъём v·sin(theta)·dt. Скорость поворота затухает к нулю:
    руль возвращается к прямой. Тангаж (процесс Орнштейна–Уленбека) возвращается
    к нулю с постоянной pitch_tau_s: дороги в среднем горизонтальны (0 — без возврата)."""
    x, y, z, psi, theta, v, omega = (float(value) for value in state)
    ground = v * np.cos(theta)
    if abs(omega) > 1e-6:
        x += ground / omega * (np.sin(psi + omega * dt) - np.sin(psi))
        y += ground / omega * (np.cos(psi) - np.cos(psi + omega * dt))
    else:
        x += ground * np.cos(psi) * dt
        y += ground * np.sin(psi) * dt
    z += v * np.sin(theta) * dt
    psi = wrap_angle(psi + omega * dt)
    if yaw_tau_s > 0.0:
        omega *= float(np.exp(-dt / yaw_tau_s))
    if pitch_tau_s > 0.0:
        theta *= float(np.exp(-dt / pitch_tau_s))
    return np.array([x, y, z, psi, theta, v, omega], np.float64)


_STATE_STEPS = np.array([1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-3, 1e-4], np.float64)


def numeric_jacobian(function, point: np.ndarray, steps: np.ndarray = _STATE_STEPS) -> np.ndarray:
    """Центральные разности. У отображений состояния → состояние курс в разности
    приводится к (-π, π]."""
    point = np.asarray(point, np.float64)
    base = np.asarray(function(point), np.float64)
    jacobian = np.zeros((base.size, point.size), np.float64)
    for index in range(point.size):
        delta = np.zeros_like(point)
        delta[index] = steps[index]
        diff = np.asarray(function(point + delta), np.float64) - np.asarray(function(point - delta), np.float64)
        if base.size == STATE_DIM:
            diff[PSI] = wrap_angle(diff[PSI])
        jacobian[:, index] = diff / (2.0 * steps[index])
    return jacobian


@dataclass
class MotionNoise:
    accel: float = 2.0         # м/с², разгон и торможение
    yaw_accel: float = 0.25    # рад/с², смена кривизны (грузовик)
    pitch_sigma: float = 0.12  # рад, стационарный разброс уклона дороги (≈7°)
    pitch_tau_s: float = 3.0   # с, за сколько уклон «забывается»
    climb: float = 0.1         # м/√с, неровности под колёсами сверх уклона
    yaw_tau_s: float = 2.0     # постоянная затухания скорости поворота


def process_noise(state: np.ndarray, dt: float, noise: MotionNoise) -> np.ndarray:
    """Шум модели: ускорение по оси корпуса, угловое ускорение курса,
    тангаж Орнштейна–Уленбека и неровность высоты — проинтегрированные за dt.

    Дисперсия приращения тангажа за dt у процесса О–У со стационарной σ и временем τ:
    σ²·(1 − exp(−2dt/τ)).
    """
    axes = state_axes(state)
    forward = axes[0]
    gain = np.zeros((STATE_DIM, 4), np.float64)
    gain[X, 0] = 0.5 * dt * dt * forward[0]
    gain[Y, 0] = 0.5 * dt * dt * forward[1]
    gain[Z, 0] = 0.5 * dt * dt * forward[2]
    gain[V, 0] = dt
    gain[PSI, 1] = 0.5 * dt * dt
    gain[OMEGA, 1] = dt
    gain[THETA, 2] = 1.0
    gain[Z, 3] = 1.0
    if noise.pitch_tau_s > 0.0:
        pitch_var = noise.pitch_sigma**2 * (1.0 - float(np.exp(-2.0 * dt / noise.pitch_tau_s)))
    else:
        pitch_var = noise.pitch_sigma**2 * dt
    spectral = np.diag([noise.accel**2, noise.yaw_accel**2, pitch_var, noise.climb**2 * dt])
    return gain @ spectral @ gain.T


# ---------------------------------------------------------------- измерение рамкой


@dataclass
class BoxObservation:
    """Рамка силуэта цели в одной камере, пиксели полного кадра."""

    pose: object  # app.geom.CameraPose
    bbox: tuple[float, float, float, float]  # x, y, w, h
    border_px: float = 2.0

    def sides(self) -> np.ndarray:
        x, y, w, h = (float(value) for value in self.bbox)
        return np.array([x, y, x + w, y + h], np.float64)

    def usable(self) -> np.ndarray:
        """Стороны, не упёртые в край кадра."""
        x0, y0, x1, y1 = self.sides()
        width = float(self.pose.width)
        height = float(self.pose.height)
        edge = self.border_px
        return np.array([x0 > edge, y0 > edge, x1 < width - edge, y1 < height - edge], bool)


def project_points(pose, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Пиксели (с дисторсией камеры) и глубины точек мира."""
    points = np.asarray(points, np.float64).reshape(-1, 3)
    rotation, _ = cv2.Rodrigues(pose.rvec)
    depth = (points @ rotation.T + pose.tvec.reshape(1, 3))[:, 2]
    pixels, _ = cv2.projectPoints(points.reshape(-1, 1, 3), pose.rvec, pose.tvec, pose.K, pose.dist)
    return pixels.reshape(-1, 2), depth


def clipped_bbox(pixels: np.ndarray, width: float, height: float) -> np.ndarray | None:
    """Рамка выпуклой оболочки точек после обрезки прямоугольником кадра."""
    hull = cv2.convexHull(np.asarray(pixels, np.float32).reshape(-1, 1, 2))
    frame = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32)
    area, polygon = cv2.intersectConvexConvex(hull, frame)
    if polygon is None or area <= 0.0:
        return None
    points = polygon.reshape(-1, 2).astype(np.float64)
    return np.array(
        [points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()],
        np.float64,
    )


def predicted_sides(state: np.ndarray, model: VehicleModel, pose) -> np.ndarray | None:
    """Рамка, которую дал бы силуэт ящика: проекция углов, оболочка, обрезка кадром.

    Ящик, хоть одним углом ушедший за плоскость камеры, не прогнозируется:
    перспективная проекция там не ограничена.
    """
    pixels, depth = project_points(pose, box_corners(state, model))
    if float(depth.min()) <= 0.5:
        return None
    return clipped_bbox(pixels, float(pose.width), float(pose.height))


# Знак «наружу» для сторон (u0, v0, u1, v1): у минимумов наружу — меньше.
_OUTWARD = np.array([-1.0, -1.0, 1.0, 1.0], np.float64)


@dataclass
class SideNoise:
    sigma_px: float = 2.5    # шум стороны рамки без окклюзии
    huber_px: float = 6.0    # где «наружу» переходит с квадрата на модуль
    cauchy_px: float = 20.0  # масштаб Коши «внутрь»: влияние максимально здесь, дальше гаснет
    # Без надёжного прогноза (старт) окклюзию не допускаем: иначе ящик раздувается
    # и накрывает силуэты с малой невязкой.
    symmetric: bool = False


def side_weight(outward: float, noise: SideNoise) -> float:
    """Вес IRLS стороны по её отклонению «наружу» (+) или «внутрь» (−), px."""
    if outward >= 0.0 or noise.symmetric:
        size = abs(outward)
        return 1.0 if size <= noise.huber_px else noise.huber_px / size
    return 1.0 / (1.0 + (outward / noise.cauchy_px) ** 2)


@dataclass
class _Stack:
    residual: np.ndarray
    jacobian: np.ndarray
    weights: np.ndarray


def _stack(
    state: np.ndarray,
    model: VehicleModel,
    observations: list[BoxObservation],
    noise: SideNoise,
) -> _Stack | None:
    residuals: list[float] = []
    jacobians: list[np.ndarray] = []
    weights: list[float] = []
    for observation in observations:
        mask = observation.usable()
        if not mask.any():
            continue
        pose = observation.pose
        predicted = predicted_sides(state, model, pose)
        if predicted is None:
            continue

        def sides_of(candidate: np.ndarray, pose=pose, fallback=predicted) -> np.ndarray:
            value = predicted_sides(candidate, model, pose)
            return fallback if value is None else value

        jacobian = numeric_jacobian(sides_of, state)
        measured = observation.sides()
        for index in np.flatnonzero(mask):
            residual = float(measured[index] - predicted[index])
            residuals.append(residual)
            jacobians.append(jacobian[index])
            weights.append(side_weight(residual * float(_OUTWARD[index]), noise))
    if not residuals:
        return None
    return _Stack(np.array(residuals), np.array(jacobians), np.array(weights))


@dataclass
class UpdateResult:
    state: np.ndarray
    covariance: np.ndarray
    sides: int
    rms_px: float


def robust_update(
    state: np.ndarray,
    covariance: np.ndarray,
    model: VehicleModel,
    observations: list[BoxObservation],
    noise: SideNoise | None = None,
    iterations: int = 6,
) -> UpdateResult | None:
    """Итерационный EKF: Гаусс–Ньютон на апостериорном максимуме с весами IRLS.

    Минимизирует (s − s⁻)ᵀ P⁻¹ (s − s⁻) + Σ wᵢ rᵢ(s)² / σ². Ковариация после шага —
    обратный гессиан в найденной точке.
    """
    noise = noise or SideNoise()
    prior = np.asarray(state, np.float64)
    information = np.linalg.inv(covariance)
    current = prior.copy()
    for _ in range(iterations):
        stack = _stack(current, model, observations, noise)
        if stack is None:
            return None
        weight = stack.weights / noise.sigma_px**2
        jtw = stack.jacobian.T * weight
        delta_prior = current - prior
        delta_prior[PSI] = wrap_angle(delta_prior[PSI])
        hessian = information + jtw @ stack.jacobian
        gradient = jtw @ stack.residual - information @ delta_prior
        step = np.linalg.solve(hessian, gradient)
        current = current + step
        current[PSI] = wrap_angle(current[PSI])
        if float(np.abs(step[:3]).max()) < 1e-3 and abs(float(step[PSI])) < 1e-4:
            break
    stack = _stack(current, model, observations, noise)
    if stack is None:
        return None
    weight = stack.weights / noise.sigma_px**2
    posterior = np.linalg.inv(information + (stack.jacobian.T * weight) @ stack.jacobian)
    posterior = 0.5 * (posterior + posterior.T)
    rms = float(np.sqrt(np.average(stack.residual**2, weights=np.maximum(stack.weights, 1e-6))))
    return UpdateResult(current, posterior, int(stack.residual.size), rms)


# ---------------------------------------------------------------- фильтр и сглаживание


@dataclass
class TrackRecord:
    time_s: float
    predicted: np.ndarray
    predicted_cov: np.ndarray
    filtered: np.ndarray
    filtered_cov: np.ndarray
    transition: np.ndarray
    measured: bool


@dataclass
class VehicleTracker:
    model: VehicleModel
    motion: MotionNoise = field(default_factory=MotionNoise)
    sides: SideNoise = field(default_factory=SideNoise)
    # Задним ходом быстрее этого не ездим: такая скорость — признак перепутанного переда.
    reverse_m_s: float = 1.0
    # Сколько секунд подряд нужна уверенная езда «задом», чтобы развернуть гипотезу.
    reverse_hold_s: float = 0.3
    state: np.ndarray | None = None
    covariance: np.ndarray | None = None
    time_s: float | None = None
    history: list[TrackRecord] = field(default_factory=list)
    misses: int = 0
    flips: int = 0
    _reverse_time_s: float = 0.0

    @property
    def ready(self) -> bool:
        return self.state is not None

    def start(self, state: np.ndarray, covariance: np.ndarray, time_s: float) -> None:
        self.state = np.asarray(state, np.float64).copy()
        self.covariance = np.asarray(covariance, np.float64).copy()
        self.time_s = float(time_s)
        self.misses = 0
        self.flips = 0
        self._reverse_time_s = 0.0
        self.history = [
            TrackRecord(
                float(time_s), self.state.copy(), self.covariance.copy(), self.state.copy(),
                self.covariance.copy(), np.eye(STATE_DIM), True,
            )
        ]

    def stop(self) -> None:
        self.state = None
        self.covariance = None
        self.time_s = None

    def predict_to(self, time_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Прогноз состояния и ковариации на момент time_s, якобиан перехода."""
        assert self.state is not None and self.covariance is not None and self.time_s is not None
        dt = max(0.0, float(time_s) - float(self.time_s))
        if dt == 0.0:
            return self.state.copy(), self.covariance.copy(), np.eye(STATE_DIM)
        yaw_tau = self.motion.yaw_tau_s
        pitch_tau = self.motion.pitch_tau_s

        def advance(candidate: np.ndarray) -> np.ndarray:
            return ctrv_step(candidate, dt, yaw_tau, pitch_tau)

        transition = numeric_jacobian(advance, self.state)
        predicted = advance(self.state)
        covariance = transition @ self.covariance @ transition.T + process_noise(
            self.state, dt, self.motion
        )
        return predicted, 0.5 * (covariance + covariance.T), transition

    def step(self, time_s: float, observations: list[BoxObservation]) -> UpdateResult | None:
        assert self.time_s is not None
        dt = max(0.0, float(time_s) - float(self.time_s))
        predicted, predicted_cov, transition = self.predict_to(time_s)
        result = robust_update(predicted, predicted_cov, self.model, observations, self.sides)
        if result is None:
            self.state, self.covariance = predicted, predicted_cov
            self.misses += 1
        else:
            self.state, self.covariance = result.state, result.covariance
            self.misses = 0
        self.time_s = float(time_s)
        self.history.append(
            TrackRecord(
                float(time_s), predicted, predicted_cov, self.state.copy(),
                self.covariance.copy(), transition, result is not None,
            )
        )
        self._resolve_front(dt)
        return result

    def _resolve_front(self, dt: float) -> None:
        """Устойчивое v < 0 — перед перепутан с самого старта: разворачиваем весь трек."""
        assert self.state is not None and self.covariance is not None
        speed = float(self.state[V])
        spread = float(np.sqrt(max(float(self.covariance[V, V]), 0.0)))
        if speed >= -max(self.reverse_m_s, 3.0 * spread):
            self._reverse_time_s = 0.0
            return
        self._reverse_time_s += dt
        if self._reverse_time_s < self.reverse_hold_s:
            return
        self._reverse_time_s = 0.0
        self._flip_history()
        last = self.history[-1]
        self.state = last.filtered.copy()
        self.covariance = last.filtered_cov.copy()
        self.flips += 1

    def _flip_jacobian(self, state: np.ndarray) -> np.ndarray:
        return numeric_jacobian(lambda candidate: flipped(candidate, self.model), state)

    def _flip_history(self) -> None:
        """Отображение разворота — инволюция: J⁻¹(s) = J(flip(s)). F' = J₊ F J⁻¹."""
        previous_inverse = None
        for record in self.history:
            j_pred = self._flip_jacobian(record.predicted)
            j_filt = self._flip_jacobian(record.filtered)
            predicted = flipped(record.predicted, self.model)
            filtered = flipped(record.filtered, self.model)
            if previous_inverse is not None:
                record.transition = j_pred @ record.transition @ previous_inverse
            record.predicted_cov = j_pred @ record.predicted_cov @ j_pred.T
            record.filtered_cov = j_filt @ record.filtered_cov @ j_filt.T
            record.predicted = predicted
            record.filtered = filtered
            previous_inverse = self._flip_jacobian(filtered)

    def smoothed(self) -> list[np.ndarray]:
        """Сглаживатель Рауха–Тунга–Штрибеля по всей истории фильтра."""
        if not self.history:
            return []
        states = [record.filtered.copy() for record in self.history]
        covs = [record.filtered_cov.copy() for record in self.history]
        for index in range(len(self.history) - 2, -1, -1):
            nxt = self.history[index + 1]
            gain = covs[index] @ nxt.transition.T @ np.linalg.inv(nxt.predicted_cov)
            innovation = states[index + 1] - nxt.predicted
            innovation[PSI] = wrap_angle(innovation[PSI])
            states[index] = states[index] + gain @ innovation
            states[index][PSI] = wrap_angle(states[index][PSI])
            covs[index] = covs[index] + gain @ (covs[index + 1] - nxt.predicted_cov) @ gain.T
        return states
