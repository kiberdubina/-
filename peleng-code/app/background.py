"""Фон камеры и пятна переднего плана.

Три источника фона на одном классификаторе:
  StaticBackground    — неподвижная камера, онлайн (гауссиана на пиксель);
  RotatingBackground  — камера только вращается вокруг центра, поза на кадре известна,
                        онлайн (гауссиана на направление луча в мире);
  RecordingBackground — запись целиком: робастная медиана по кадрам
                        (для вращающейся камеры — медиана в ячейках панорамы).

Для камеры, которая вращается без сдвига центра, направление луча в мире
d = Rᵀ K⁻¹ [u, v, 1]ᵀ задаёт одну и ту же точку фона при любой глубине сцены.
Поэтому фон хранится в сферической панораме (азимут, угол места): соответствие
кадра и фона точное, без оценки движения по картинке.

Классификатор: отличие от фона больше max(min_diff, k·σ), где σ — разброс фона
в этой ячейке (качающаяся листва даёт большую σ и сама себя гасит); минус тень —
яркость упала в долю [low, high], а соотношение каналов сохранилось.
Перед сравнением фон приводится к экспозиции кадра: усиление по каналу —
медиана отношения кадр/фон (цель занимает малую долю кадра).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class BackgroundConfig:
    work_width: int = 960          # рабочая ширина кадра, px
    learn_rate: float = 0.03       # обновление фона там, где фон
    foreground_share: float = 0.1  # доля скорости обучения под передним планом
    min_diff: float = 14.0         # нижний порог отличия от фона, уровни яркости
    sigma_k: float = 3.0           # порог в сигмах фона
    init_sigma: float = 6.0        # сигма фона на старте и у медианы записи
    shadow_low: float = 0.40       # тень: доля яркости фона снизу
    shadow_high: float = 0.92      # и сверху
    shadow_chroma: float = 0.07    # разброс отношения по каналам у тени (гладкий фон)
    texture_window: int = 7        # окно локальной корреляции кадра и фона, px
    texture_sigma: float = 3.0     # фон с такой σ яркости в окне считается текстурным
    texture_ncc: float = 0.5       # тень на текстурном фоне сохраняет его рисунок
    shadow_uniform: float = 0.06   # σ отношения кадр/фон в окне у тени (затемнение однородно)
    ghost_frames: int = 45         # неподвижный «передний план» дольше — уходит в фон
    ghost_still: float = 6.0       # изменение пикселя, при котором он ещё «неподвижен»
    warmup_frames: int = 3         # первые кадры только учат фон
    exposure: bool = True          # выравнивать экспозицию кадра и фона


@dataclass
class Blob:
    """Связное пятно переднего плана в пикселях полного кадра."""

    x: float
    y: float
    w: float
    h: float
    area: float

    @property
    def rect(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)


def work_size(width: int, height: int, work_width: int) -> tuple[float, tuple[int, int]]:
    scale = min(1.0, float(work_width) / float(width))
    return scale, (max(1, int(round(width * scale))), max(1, int(round(height * scale))))


def classify(
    image: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray | float,
    valid: np.ndarray,
    config: BackgroundConfig,
) -> np.ndarray:
    """Маска переднего плана без теней. Кадр и фон уже на одной сетке, float32."""
    if config.exposure:
        # Медиана отношения по прореженной сетке: та же оценка, в 16 раз меньше работы.
        thin = valid[::4, ::4]
        if thin.any():
            ratio_thin = (image[::4, ::4][thin] + 1.0) / (mean[::4, ::4][thin] + 1.0)
            gain = np.median(ratio_thin, axis=0).astype(np.float32)
            mean = mean * gain.reshape(1, 1, 3)
    diff = cv2.absdiff(image, mean)
    blue, green, red = cv2.split(diff)
    largest = cv2.max(cv2.max(blue, green), red)
    if np.isscalar(var) or np.ndim(var) == 0:
        threshold = max(config.min_diff, config.sigma_k * float(np.sqrt(var)))
    else:
        threshold = np.maximum(config.min_diff, config.sigma_k * cv2.sqrt(np.asarray(var, np.float32)))
    moving = valid & (largest > threshold)
    ratio = cv2.divide(image + 1.0, mean + 1.0)
    blue, green, red = cv2.split(ratio)
    level = (blue + green + red) * (1.0 / 3.0)
    darker = (level > config.shadow_low) & (level < config.shadow_high)
    # Где фон текстурный, тень узнаётся по сохранённому рисунку фона (корреляция окна),
    # цвет не проверяется: тень на снегу синеет от неба. Гладкий борт машины рисунок
    # закрывает — его корреляция с фоном около нуля. Где фон гладкий, рисунка нет —
    # остаётся признак «потемнело, а соотношение каналов прежнее».
    textured, correlation, uniform = _texture_match(image, mean, level, config)
    chroma = cv2.max(cv2.max(cv2.absdiff(blue, level), cv2.absdiff(green, level)), cv2.absdiff(red, level))
    flat_shadow = ~textured & (chroma < config.shadow_chroma)
    textured_shadow = textured & uniform & (correlation >= config.texture_ncc)
    shadow = darker & (flat_shadow | textured_shadow)
    return moving & ~shadow


def _texture_match(
    image: np.ndarray, mean: np.ndarray, level: np.ndarray, config: BackgroundConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Текстурный ли фон, локальная корреляция кадра с фоном и однородность затемнения.

    Тень умножает яркость всего окна на одну константу: отношение кадр/фон в окне почти
    постоянно. У края объекта в окно попадают и объект, и фон — отношение скачет, и высокая
    корреляция там (от текстуры фона) тенью не считается.
    """
    size = (config.texture_window, config.texture_window)
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    back = cv2.cvtColor(mean.astype(np.float32), cv2.COLOR_BGR2GRAY)
    mu_f = cv2.blur(frame, size)
    mu_b = cv2.blur(back, size)
    var_f = cv2.blur(frame * frame, size) - mu_f * mu_f
    var_b = cv2.blur(back * back, size) - mu_b * mu_b
    cov = cv2.blur(frame * back, size) - mu_f * mu_b
    correlation = cov / np.sqrt(np.maximum(var_f * var_b, 1e-6))
    mu_l = cv2.blur(level, size)
    spread = cv2.blur(level * level, size) - mu_l * mu_l
    uniform = spread < config.shadow_uniform**2
    return var_b >= config.texture_sigma**2, correlation, uniform


def masked_median(stack: np.ndarray) -> np.ndarray:
    """Медиана по оси 0 без NaN (пропуск = исключённый отсчёт). Нет ни одного — NaN.

    Сортировка уводит NaN в конец, медиана берётся по числу валидных отсчётов n:
    среднее элементов (n−1)//2 и n//2. Точно как np.nanmedian, но в разы быстрее.
    """
    ordered = np.sort(stack, axis=0)
    count = np.sum(~np.isnan(stack), axis=0)
    low = np.clip((count - 1) // 2, 0, stack.shape[0] - 1)
    high = np.clip(count // 2, 0, stack.shape[0] - 1)
    lower = np.take_along_axis(ordered, low[None], axis=0)[0]
    upper = np.take_along_axis(ordered, high[None], axis=0)[0]
    median = 0.5 * (lower + upper)
    median[count == 0] = np.nan
    return median


def foreground_blobs(mask: np.ndarray, scale: float, min_area_px: float = 12.0) -> list[Blob]:
    """Пятна маски рабочего размера → прямоугольники полного кадра.

    Открытие 2×2 снимает одиночные пиксели, закрытие эллипсом 5×5 сшивает
    силуэт, разрезанный тонкой веткой или столбом.
    """
    work = (mask.astype(np.uint8) * 255) if mask.dtype == bool else mask
    work = cv2.morphologyEx(work, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    count, _, stats, _ = cv2.connectedComponentsWithStats(work)
    inverse = 1.0 / scale
    blobs = []
    for index in range(1, count):
        x, y, w, h, area = (int(value) for value in stats[index])
        if area < min_area_px:
            continue
        blobs.append(
            Blob(x * inverse, y * inverse, w * inverse, h * inverse, area * inverse * inverse)
        )
    return blobs


class _GaussianModel:
    """Среднее и дисперсия на ячейку. Ячейки — пиксели кадра или панорамы."""

    def __init__(self, shape: tuple[int, int], config: BackgroundConfig):
        self.config = config
        self.mean = np.zeros(shape + (3,), np.float32)
        self.var = np.full(shape, config.init_sigma**2, np.float32)
        self.seen = np.zeros(shape, bool)
        self.age = np.zeros(shape, np.uint16)
        self.last = np.zeros(shape + (3,), np.float32)

    @staticmethod
    def learn_arrays(
        config: BackgroundConfig,
        mean: np.ndarray,
        var: np.ndarray,
        seen: np.ndarray,
        age: np.ndarray,
        last: np.ndarray,
        image: np.ndarray,
        foreground: np.ndarray,
        valid: np.ndarray,
    ) -> None:
        """Обновляет переданные массивы на месте."""
        fresh = valid & ~seen
        mean[fresh] = image[fresh]
        seen[fresh] = True

        # Призрак: «передний план», который не меняется дольше ghost_frames, — это фон,
        # открывшийся после ухода цели (или цель, вставшая надолго).
        still = np.abs(image - last).max(axis=2) < config.ghost_still
        grown = np.minimum(age.astype(np.int32) + 1, 65535).astype(np.uint16)
        age[:] = np.where(valid & foreground & still, grown, 0)
        ghost = age > config.ghost_frames
        mean[ghost] = image[ghost]
        var[ghost] = config.init_sigma**2
        age[ghost] = 0

        # Фон учится везде, под передним планом — медленно. Иначе качающаяся листва,
        # однажды помеченная целью, никогда не выучит свою дисперсию.
        update = valid & seen & ~fresh
        rate = np.where(foreground, config.learn_rate * config.foreground_share, config.learn_rate)
        rate = rate.astype(np.float32)[update]
        delta = image - mean
        mean[update] += rate[:, None] * delta[update]
        energy = (delta**2).max(axis=2)
        var[update] = (1.0 - rate) * var[update] + rate * energy[update]
        last[valid] = image[valid]


class StaticBackground:
    """Неподвижная камера, фон учится на ходу."""

    def __init__(self, width: int, height: int, config: BackgroundConfig | None = None):
        self.config = config or BackgroundConfig()
        self.scale, self.size = work_size(width, height, self.config.work_width)
        self.model = _GaussianModel((self.size[1], self.size[0]), self.config)
        self.frames = 0

    def apply(self, image: np.ndarray, pose=None) -> np.ndarray:
        work = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA).astype(np.float32)
        valid = np.ones(work.shape[:2], bool)
        model = self.model
        if self.frames < self.config.warmup_frames:
            foreground = np.zeros_like(valid)
        else:
            foreground = classify(work, model.mean, model.var, valid & model.seen, self.config)
        _GaussianModel.learn_arrays(
            self.config, model.mean, model.var, model.seen, model.age, model.last,
            work, foreground, valid,
        )
        self.frames += 1
        return foreground


class PanoramaGrid:
    """Сферическая панорама направлений для камеры, вращающейся вокруг своего центра.

    Ячейка (row, col) ↔ азимут az0 + col·step, угол места top − row·step;
    step — угловой размер пикселя рабочего кадра в центре (1 / f).
    Шов по азимуту — напротив стартового направления камеры.
    """

    def __init__(self, width: int, height: int, work_width: int, pose, margin_rad: float = 0.15):
        self.scale, self.size = work_size(width, height, work_width)
        matrix = np.array(pose.K, np.float64).copy()
        matrix[:2] *= self.scale
        self.K = matrix
        w, h = self.size
        u, v = np.meshgrid(np.arange(w, dtype=np.float64) + 0.5, np.arange(h, dtype=np.float64) + 0.5)
        pixels = np.stack([u, v, np.ones_like(u)], axis=-1)
        rays = pixels @ np.linalg.inv(matrix).T
        self.rays = rays / np.linalg.norm(rays, axis=-1, keepdims=True)
        self.step = 1.0 / float(matrix[0, 0])
        rotation, _ = cv2.Rodrigues(pose.rvec)
        world = self.rays @ rotation
        azimuth = np.arctan2(world[..., 1], world[..., 0])
        elevation = np.arcsin(np.clip(world[..., 2], -1.0, 1.0))
        self.azimuth0 = float(np.arctan2(np.sin(azimuth).mean(), np.cos(azimuth).mean())) - np.pi
        self.top = float(elevation.max()) + margin_rad
        bottom = float(elevation.min()) - margin_rad
        self.rows = int(np.ceil((self.top - bottom) / self.step))
        self.cols = int(np.ceil(2.0 * np.pi / self.step))

    def lookup(self, pose) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Для каждого пикселя рабочего кадра — дробные (col, row) панорамы и маска «внутри»."""
        rotation, _ = cv2.Rodrigues(pose.rvec)
        world = self.rays @ rotation
        azimuth = np.arctan2(world[..., 1], world[..., 0])
        elevation = np.arcsin(np.clip(world[..., 2], -1.0, 1.0))
        col = (((azimuth - self.azimuth0) % (2.0 * np.pi)) / self.step).astype(np.float32)
        row = ((self.top - elevation) / self.step).astype(np.float32)
        inside = (row >= 0) & (row <= self.rows - 1)
        return col, row, inside

    def window(self, col: np.ndarray, row: np.ndarray) -> tuple[int, int, int, int] | None:
        """Прямоугольник панорамы, накрывающий текущий обзор: r0, r1, c0, c1 (c — без модуля)."""
        c0, c1 = int(np.floor(col.min())), int(np.ceil(col.max())) + 1
        r0, r1 = max(0, int(np.floor(row.min()))), min(self.rows, int(np.ceil(row.max())) + 1)
        if c1 - c0 >= self.cols or r1 <= r0:
            return None
        return r0, r1, c0, c1

    def to_frame(self, pose, r0: int, r1: int, c0: int, c1: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Для ячеек окна панорамы — (u, v) рабочего кадра и маска «видно в кадре»."""
        rotation, _ = cv2.Rodrigues(pose.rvec)
        grid_c, grid_r = np.meshgrid(
            np.arange(c0, c1, dtype=np.float64), np.arange(r0, r1, dtype=np.float64)
        )
        azimuth = self.azimuth0 + grid_c * self.step
        elevation = self.top - grid_r * self.step
        world = np.stack(
            [np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)],
            axis=-1,
        )
        camera = world @ rotation.T
        depth = camera[..., 2]
        ahead = depth > 1e-6
        safe = np.where(ahead, depth, 1.0)
        u = (self.K[0, 0] * camera[..., 0] / safe + self.K[0, 2] - 0.5).astype(np.float32)
        v = (self.K[1, 1] * camera[..., 1] / safe + self.K[1, 2] - 0.5).astype(np.float32)
        w, h = self.size
        covered = ahead & (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
        return u, v, covered


class RotatingBackground:
    """Камера вращается вокруг неподвижного центра; поза на кадре известна. Онлайн."""

    def __init__(self, width: int, height: int, config: BackgroundConfig | None = None):
        self.config = config or BackgroundConfig()
        self.width = width
        self.height = height
        self.scale, self.size = work_size(width, height, self.config.work_width)
        self.grid: PanoramaGrid | None = None
        self.model: _GaussianModel | None = None
        self.frames = 0

    def apply(self, image: np.ndarray, pose) -> np.ndarray:
        if self.grid is None:
            self.grid = PanoramaGrid(self.width, self.height, self.config.work_width, pose)
            self.model = _GaussianModel((self.grid.rows, self.grid.cols), self.config)
        grid, model = self.grid, self.model
        assert model is not None
        work = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA).astype(np.float32)

        col, row, inside = grid.lookup(pose)
        mean = cv2.remap(model.mean, col, row, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        var = cv2.remap(model.var, col, row, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        seen = cv2.remap(model.seen.astype(np.uint8), col, row, cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP) > 0
        valid = inside & seen
        if self.frames < self.config.warmup_frames:
            foreground = np.zeros(work.shape[:2], bool)
        else:
            foreground = classify(work, mean, var, valid, self.config)

        window = grid.window(col, row)
        if window is not None:
            r0, r1, c0, c1 = window
            u, v, covered = grid.to_frame(pose, r0, r1, c0, c1)
            patch = cv2.remap(work, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            fg_patch = cv2.remap(
                foreground.astype(np.uint8), u, v, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT
            ) > 0
            # Окно может пересечь шов: работаем со столбцами по модулю.
            columns = np.arange(c0, c1) % grid.cols
            arrays = [model.mean, model.var, model.seen, model.age, model.last]
            parts = [array[r0:r1][:, columns].copy() for array in arrays]
            _GaussianModel.learn_arrays(self.config, *parts, patch, fg_patch, covered)
            for array, part in zip(arrays, parts):
                array[r0:r1, columns] = part
        self.frames += 1
        return foreground


class RecordingBackground:
    """Фон записи целиком: медиана по выборке кадров.

    Неподвижная камера — медиана в пикселях. Камера, которая поворачивается, —
    медиана в ячейках панорамы по тем кадрам, где ячейка была видна. Пиксели цели
    по предыдущему проходу можно исключить из выборки (exclude): камера,
    сопровождающая цель, видит её почти всё время в одних и тех же направлениях.
    """

    def __init__(self, width: int, height: int, config: BackgroundConfig | None = None):
        self.config = config or BackgroundConfig()
        self.width = width
        self.height = height
        self.scale, self.size = work_size(width, height, self.config.work_width)
        self.samples: list[tuple[np.ndarray, np.ndarray, object]] = []
        self.rotating = False
        self.grid: PanoramaGrid | None = None
        self.image: np.ndarray | None = None
        self.panorama: np.ndarray | None = None
        self.panorama_seen: np.ndarray | None = None
        self._first_pose = None

    def add(self, image: np.ndarray, pose, exclude: np.ndarray | None = None) -> None:
        work = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)
        keep = np.ones(work.shape[:2], bool)
        if exclude is not None:
            small = cv2.resize(exclude.astype(np.uint8), self.size, interpolation=cv2.INTER_NEAREST)
            keep &= small == 0
        if self._first_pose is None:
            self._first_pose = pose
        elif pose is not None and not _same_rotation(self._first_pose, pose):
            self.rotating = True
        self.samples.append((work, keep, pose))

    def build(self) -> None:
        if not self.samples:
            raise ValueError("RecordingBackground: нет кадров")
        if self.rotating:
            self._build_panorama()
        else:
            self._build_image()
        self.samples = []

    def _build_image(self) -> None:
        height, width = self.samples[0][0].shape[:2]
        image = np.zeros((height, width, 3), np.float32)
        # Полосами строк: стопка всех кадров целиком заняла бы гигабайты.
        for top in range(0, height, 64):
            bottom = min(height, top + 64)
            plain = np.stack([sample[0][top:bottom] for sample in self.samples]).astype(np.float32)
            keep = np.stack([sample[1][top:bottom] for sample in self.samples])
            stack = plain.copy()
            stack[~keep] = np.nan
            median = masked_median(stack)
            missing = ~np.isfinite(median).all(axis=2)
            if missing.any():
                median[missing] = np.median(plain, axis=0)[missing]
            image[top:bottom] = median
        self.image = image

    def _build_panorama(self) -> None:
        grid = PanoramaGrid(self.width, self.height, self.config.work_width, self._first_pose)
        self.grid = grid
        patches = []
        lo, hi = None, None
        for work, keep, pose in self.samples:
            col, row, _ = grid.lookup(pose)
            window = grid.window(col, row)
            if window is None:
                continue
            r0, r1, c0, c1 = window
            u, v, covered = grid.to_frame(pose, r0, r1, c0, c1)
            patch = cv2.remap(work, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            held = cv2.remap(keep.astype(np.uint8), u, v, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) > 0
            patches.append((patch, covered & held, r0, c0))
            lo = c0 if lo is None else min(lo, c0)
            hi = c1 if hi is None else max(hi, c1)
        mean = np.zeros((grid.rows, grid.cols, 3), np.float32)
        seen = np.zeros((grid.rows, grid.cols), bool)
        if patches:
            assert lo is not None and hi is not None
            span = hi - lo
            columns = np.arange(lo, hi) % grid.cols
            for top in range(0, grid.rows, 48):
                bottom = min(grid.rows, top + 48)
                stack = np.full((len(patches), bottom - top, span, 3), np.nan, np.float32)
                for index, (patch, ok, r0, c0) in enumerate(patches):
                    a, b = max(top, r0), min(bottom, r0 + patch.shape[0])
                    if a >= b:
                        continue
                    part = patch[a - r0 : b - r0].astype(np.float32)
                    part[~ok[a - r0 : b - r0]] = np.nan
                    stack[index, a - top : b - top, c0 - lo : c0 - lo + patch.shape[1]] = part
                median = masked_median(stack)
                good = np.isfinite(median).all(axis=2)
                block = mean[top:bottom][:, columns]
                block[good] = median[good]
                mean[top:bottom, columns] = block
                flags = seen[top:bottom][:, columns]
                flags |= good
                seen[top:bottom, columns] = flags
        self.panorama = mean
        self.panorama_seen = seen

    def background_for(self, pose) -> tuple[np.ndarray, np.ndarray]:
        """Фон и маска «фон известен» в сетке рабочего кадра."""
        if not self.rotating:
            assert self.image is not None
            return self.image, np.ones(self.image.shape[:2], bool)
        assert self.grid is not None and self.panorama is not None and self.panorama_seen is not None
        col, row, inside = self.grid.lookup(pose)
        mean = cv2.remap(self.panorama, col, row, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        seen = cv2.remap(
            self.panorama_seen.astype(np.uint8), col, row, cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP
        ) > 0
        return mean, inside & seen

    def apply(self, image: np.ndarray, pose) -> np.ndarray:
        work = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA).astype(np.float32)
        mean, valid = self.background_for(pose)
        return classify(work, mean, self.config.init_sigma**2, valid, self.config)


def _same_rotation(first, second) -> bool:
    return bool(
        np.allclose(np.asarray(first.rvec).ravel(), np.asarray(second.rvec).ravel(), atol=1e-9)
        and np.allclose(np.asarray(first.tvec).ravel(), np.asarray(second.tvec).ravel(), atol=1e-9)
    )
