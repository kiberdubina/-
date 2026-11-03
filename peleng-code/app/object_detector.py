"""Цель по описанию: класс COCO + цвет (YOLO) или текстовый запрос (YOLO-World).

Интерфейс тот же, что у MotionDetector: detect(image) -> list[Detection].
Дальше всё как раньше: 2D-трек, триангуляция, 3D-трек.

Режимы профиля:
  object — модель YOLO ищет объекты класса, внутри рамки проверяется цвет;
  text   — YOLO-World ищет по фразе («person in black hoodie», «красная машина»).

ultralytics импортируется лениво. Нет пакета или весов — DetectorError,
build_detector отдаёт MotionDetector и текст ошибки для строки статуса.
Веса сами не скачиваются: файл кладут в models/ или корень проекта.
"""

from __future__ import annotations

import difflib
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from app.motion import Detection, MotionDetector
from app.profile import TargetProfile

ROOT = Path(__file__).resolve().parent.parent
WEIGHT_DIRS = (ROOT / "models", ROOT)

MODES = ("object", "text")

# Строки рамки по высоте для проверки цвета: доля сверху, доля снизу.
REGIONS = {
    "full": (0.0, 1.0),
    "upper": (0.15, 0.55),  # торс без головы: кофта, куртка
    "middle": (0.30, 0.70),
    "lower": (0.55, 0.95),  # брюки, юбка
}

DEFAULTS = {
    "object": {"conf": 0.4, "model": "yolov8n.pt", "det_every": 3},
    "text": {"conf": 0.25, "model": "yolov8s-worldv2.pt", "det_every": 5},
}
DET_WIDTH = 640
ROI_SCALE = 2.5
ROI_MIN_PX = 96
COLOR_SHARE = 0.3

COCO_NAMES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
)


class DetectorError(RuntimeError):
    """Детектор по описанию не готов: нет пакета, весов или профиль неверный."""


@dataclass
class HsvRange:
    h_min: int
    h_max: int
    s_min: int = 0
    s_max: int = 255
    v_min: int = 0
    v_max: int = 255


# Цвета для текстового запроса. H в шкале OpenCV 0..180; красный через 0° — два диапазона.
COLOR_TABLE = {
    "black": [HsvRange(0, 180, 0, 255, 0, 60)],
    "white": [HsvRange(0, 180, 0, 40, 180, 255)],
    "red": [HsvRange(0, 10, 120, 255, 70, 255), HsvRange(170, 180, 120, 255, 70, 255)],
    "blue": [HsvRange(100, 130, 80, 255, 50, 255)],
    "green": [HsvRange(35, 85, 60, 255, 40, 255)],
    "yellow": [HsvRange(20, 35, 100, 255, 100, 255)],
}

# Русские слова по основе: длинная основа проверяется раньше короткой.
_RU_WORDS = {
    "челов": "person",
    "машин": "car",
    "грузов": "truck",
    "автобус": "bus",
    "мяч": "ball",
    "черн": "black",
    "бел": "white",
    "красн": "red",
    "син": "blue",
    "зелен": "green",
    "желт": "yellow",
    "кофт": "hoodie",
    "футболк": "t-shirt",
    "куртк": "jacket",
    "в": "in",
    "во": "in",
    "с": "with",
    "со": "with",
    "на": "on",
}
_CLOTHES = {"hoodie", "t-shirt", "jacket"}
_CYRILLIC = re.compile(r"[а-яё]", re.IGNORECASE)


@dataclass
class DetectSpec:
    mode: str
    classes: list[str]
    query: list[str]
    conf: float
    region: str
    colors: list[HsvRange]
    color_share: float
    model: str
    det_width: int
    det_every: int
    roi_scale: float


@dataclass
class Box:
    """Ответ модели в координатах поданного ей кадра."""

    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    name: str


@dataclass
class Translation:
    text: str
    color: str | None
    region: str


# ---------------------------------------------------------------- профиль


def parse_spec(raw: dict) -> DetectSpec:
    """Поля режима object/text из профиля config.yaml. Ошибка — DetectorError."""
    mode = str(raw.get("mode", "")).lower()
    if mode not in MODES:
        raise DetectorError(f"режим «{mode}» не object и не text")
    base = DEFAULTS[mode]
    classes: list[str] = []
    query: list[str] = []
    region = str(raw.get("region") or "full").lower()
    colors = _colors(raw.get("color"))
    color_share = float(raw.get("color_share", COLOR_SHARE))
    if mode == "object":
        classes = _names(raw.get("class"))
        if not classes:
            raise DetectorError("в профиле object нет поля class")
        for name in classes:
            if name not in COCO_NAMES:
                similar = difflib.get_close_matches(name, COCO_NAMES, n=3, cutoff=0.5)
                hint = f" Похожие: {', '.join(similar)}." if similar else ""
                raise DetectorError(f"неизвестный класс «{name}».{hint}")
    else:
        phrases = _names(raw.get("query"))
        if not phrases:
            raise DetectorError("в профиле text нет поля query")
        for phrase in phrases:
            if _CYRILLIC.search(phrase):
                found = translate_query(phrase)
                query.append(found.text)
                # Цвет из запроса подстраховывается HSV, если профиль цвет не задал.
                if not colors and found.color is not None:
                    colors = list(COLOR_TABLE[found.color])
                    if raw.get("region") is None:
                        region = found.region
            else:
                query.append(phrase)
    if region not in REGIONS:
        raise DetectorError(f"region «{region}»: нужно {', '.join(REGIONS)}")
    return DetectSpec(
        mode=mode,
        classes=classes,
        query=query,
        conf=float(raw.get("conf", base["conf"])),
        region=region,
        colors=colors,
        color_share=color_share,
        model=str(raw.get("model", base["model"])),
        det_width=int(raw.get("det_width", DET_WIDTH)),
        det_every=max(1, int(raw.get("det_every", base["det_every"]))),
        roi_scale=float(raw.get("roi_scale", ROI_SCALE)),
    )


def _names(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    return [str(item).strip() for item in value if str(item).strip()]


def _colors(value) -> list[HsvRange]:
    if not value:
        return []
    items = [value] if isinstance(value, dict) else list(value)
    ranges = []
    for item in items:
        ranges.append(
            HsvRange(
                h_min=int(item["h_min"]),
                h_max=int(item["h_max"]),
                s_min=int(item.get("s_min", 0)),
                s_max=int(item.get("s_max", 255)),
                v_min=int(item.get("v_min", 0)),
                v_max=int(item.get("v_max", 255)),
            )
        )
    return ranges


def translate_query(text: str) -> Translation:
    """Русский запрос → английская фраза, цвет и часть рамки для проверки цвета."""
    stems = sorted(_RU_WORDS, key=len, reverse=True)
    words = []
    for word in re.findall(r"[\w-]+", text.lower().replace("ё", "е")):
        english = word
        for stem in stems:
            # Короткие предлоги — только целым словом, иначе «с» съест «синий».
            if len(stem) <= 2:
                if word == stem:
                    english = _RU_WORDS[stem]
                    break
            elif word.startswith(stem):
                english = _RU_WORDS[stem]
                break
        words.append(english)
    color = next((word for word in words if word in COLOR_TABLE), None)
    region = "upper" if any(word in _CLOTHES for word in words) or "person" in words else "full"
    return Translation(" ".join(words), color, region)


# ---------------------------------------------------------------- цвет


def region_rows(y: int, h: int, region: str) -> tuple[int, int]:
    """Строки [y0, y1) рамки высотой h, где проверяется цвет."""
    top, bottom = REGIONS[region]
    y0 = y + int(round(h * top))
    y1 = y + int(round(h * bottom))
    return y0, max(y0 + 1, y1)


def color_share(image: np.ndarray, bbox: tuple[int, int, int, int], region: str, colors: list[HsvRange]) -> float:
    """Доля пикселей области рамки, попавших хотя бы в один диапазон HSV."""
    x, y, w, h = bbox
    height, width = image.shape[:2]
    y0, y1 = region_rows(y, h, region)
    x0, x1 = max(0, x), min(width, x + w)
    y0, y1 = max(0, y0), min(height, y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    hsv = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for item in colors:
        low = np.array([item.h_min, item.s_min, item.v_min], np.uint8)
        high = np.array([item.h_max, item.s_max, item.v_max], np.uint8)
        mask |= cv2.inRange(hsv, low, high)
    return float(np.count_nonzero(mask)) / float(mask.size)


# ---------------------------------------------------------------- модель

# Одна модель на процесс, общая для всех камер.
_MODELS: dict[tuple[str, str], object] = {}


def find_weights(name: str) -> Path:
    path = Path(name)
    if path.is_absolute():
        if path.is_file():
            return path
    else:
        for folder in WEIGHT_DIRS:
            candidate = folder / name
            if candidate.is_file():
                return candidate
    raise DetectorError(f"нет весов {name}: положите файл в models/")


def load_model(spec: DetectSpec):
    try:
        import ultralytics  # noqa: F401
    except ImportError as error:
        raise DetectorError("нет ultralytics: pip install ultralytics") from error
    path = find_weights(spec.model)
    key = (spec.mode, str(path))
    model = _MODELS.get(key)
    if model is None:
        model = _YoloModel(path, world=spec.mode == "text")
        _MODELS[key] = model
    if spec.mode == "text":
        model.set_query(spec.query)
    return model


class _YoloModel:
    """Обёртка ultralytics: predict отдаёт Box в координатах поданного кадра."""

    def __init__(self, path: Path, world: bool):
        import ultralytics

        self.world = world
        self.model = ultralytics.YOLOWorld(str(path)) if world else ultralytics.YOLO(str(path))
        self.device = _device()
        self.query: list[str] | None = None

    def set_query(self, query: list[str]) -> None:
        # set_classes пересчитывает текстовые признаки — только при смене запроса.
        if query != self.query:
            self.model.set_classes(list(query))
            self.query = list(query)

    def predict(self, image: np.ndarray, conf: float, classes: list[str] | None) -> list[Box]:
        ids = None
        names = self.model.names
        if classes and not self.world:
            ids = [index for index, name in names.items() if name in classes]
        results = self.model.predict(image, conf=conf, classes=ids, device=self.device, verbose=False)
        boxes: list[Box] = []
        for result in results:
            data = result.boxes
            if data is None:
                continue
            xyxy = data.xyxy.cpu().numpy()
            scores = data.conf.cpu().numpy()
            labels = data.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), score, label in zip(xyxy, scores, labels):
                boxes.append(Box(float(x1), float(y1), float(x2), float(y2), float(score), names[int(label)]))
        return boxes


def _device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ---------------------------------------------------------------- детектор


@dataclass
class Label:
    detection: Detection
    name: str
    conf: float


@dataclass
class _State:
    tick: int = 0
    roi_failed: bool = False
    hint: tuple[float, float, float, float] | None = None
    labels: list[Label] = field(default_factory=list)


class ObjectDetector:
    """Объекты класса или запроса на кадре. Модель — раз в det_every кадров."""

    def __init__(self, profile: TargetProfile, spec: DetectSpec, model=None):
        self.profile = profile
        self.spec = spec
        self.model = model if model is not None else load_model(spec)
        self.device = getattr(self.model, "device", "cpu")
        self.ms = 0.0
        self.calls = 0
        self._state = _State()

    @property
    def labels(self) -> list[Label]:
        """Подписи к рамкам последнего запуска: класс или запрос и уверенность."""
        return list(self._state.labels)

    def set_hint(self, bbox: tuple[float, float, float, float] | None) -> None:
        """Прогноз трека (cx, cy, w, h) в пикселях: следующий запуск ищет сначала вокруг него."""
        self._state.hint = bbox

    def detect(self, image: np.ndarray) -> list[Detection]:
        state = self._state
        run = state.tick % self.spec.det_every == 0
        state.tick += 1
        if not run:
            # Между запусками рамку ведёт 2D-Калман: трек переживает пропуски.
            return []
        roi = None if state.roi_failed else self._roi(image.shape[1], image.shape[0])
        if roi is not None:
            x0, y0, x1, y1 = roi
            found = self._run(image, image[y0:y1, x0:x1], x0, y0)
            # Не нашли в окне — следующий запуск по всему кадру.
            state.roi_failed = not found
            return found
        state.roi_failed = False
        return self._run(image, image, 0, 0)

    def _roi(self, width: int, height: int) -> tuple[int, int, int, int] | None:
        hint = self._state.hint
        if hint is None:
            return None
        cx, cy, w, h = hint
        half_w = max(w * self.spec.roi_scale, ROI_MIN_PX) / 2.0
        half_h = max(h * self.spec.roi_scale, ROI_MIN_PX) / 2.0
        x0, x1 = max(0, int(cx - half_w)), min(width, int(cx + half_w))
        y0, y1 = max(0, int(cy - half_h)), min(height, int(cy + half_h))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        # Окно почти во весь кадр ничего не экономит.
        if (x1 - x0) * (y1 - y0) >= 0.9 * width * height:
            return None
        return x0, y0, x1, y1

    def _run(self, image: np.ndarray, part: np.ndarray, x0: int, y0: int) -> list[Detection]:
        scale = min(1.0, self.spec.det_width / float(part.shape[1]))
        small = part if scale >= 1.0 else cv2.resize(part, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        classes = self.spec.classes if self.spec.mode == "object" else None
        started = time.perf_counter()
        boxes = self.model.predict(small, self.spec.conf, classes)
        self.ms = (time.perf_counter() - started) * 1000.0
        self.calls += 1
        labels: list[Label] = []
        for box in boxes:
            label = self._accept(image, box, scale, x0, y0)
            if label is not None:
                labels.append(label)
        # Без трека берётся самый уверенный, при равенстве — крупнейший.
        labels.sort(key=lambda item: (-item.conf, -item.detection.area))
        self._state.labels = labels
        return [item.detection for item in labels]

    def _accept(self, image: np.ndarray, box: Box, scale: float, x0: int, y0: int) -> Label | None:
        if box.conf < self.spec.conf:
            return None
        if self.spec.mode == "object" and box.name not in self.spec.classes:
            return None
        left = x0 + box.x1 / scale
        top = y0 + box.y1 / scale
        w = (box.x2 - box.x1) / scale
        h = (box.y2 - box.y1) / scale
        bbox = (int(round(left)), int(round(top)), max(1, int(round(w))), max(1, int(round(h))))
        area = float(bbox[2] * bbox[3])
        # Площадь рамки, не контура: s_min/s_max профиля ставятся под рамку.
        if area < self.profile.s_min or area > self.profile.s_max:
            return None
        if self.spec.colors:
            share = color_share(image, bbox, self.spec.region, self.spec.colors)
            if share < self.spec.color_share:
                return None
        detection = Detection(cx=left + w / 2.0, cy=top + h / 2.0, area=area, bbox=bbox)
        return Label(detection, box.name, box.conf)


def build_detector(profile: TargetProfile, raw: dict | None, model=None):
    """Детектор под профиль и текст ошибки. object/text не готов — MotionDetector."""
    mode = str((raw or {}).get("mode", profile.mode or "auto")).lower()
    if raw is None or mode not in MODES:
        return MotionDetector(profile), None
    try:
        return ObjectDetector(profile, parse_spec(raw), model=model), None
    except DetectorError as error:
        return MotionDetector(profile), str(error)
