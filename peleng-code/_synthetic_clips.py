"""Ролики с известной правдой — замена исходных роликов проверок check_trail.py и _eval_hold_trail.py.

Исходные ролики (video_2026-11-01_22-06-07.mp4, video_2026-11-02_15-54-01.mp4, IMG_8241.MOV,
IMG_0985.MOV) снимались у участника команды и в репозиторий не попали: *.mp4 и *.mov
в .gitignore. Здесь — сцены тех же задач: кадр 1280×720, 30 к/с, цель того цвета и размера,
которых ждут профили config.yaml (green_ball, sky_uav, white_drone).

Каждый ролик — mp4 и правда: центр цели на каждом кадре, px. Генерация детерминированная.
Это проверка того, что конвейер работает, а не точность на реальной съёмке: на исходных
роликах приёмку проходил только мяч (см. _eval_hold_trail_report.json).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

WIDTH = 1280
HEIGHT = 720
FPS = 30.0


@dataclass
class Clip:
    path: Path
    truth: np.ndarray  # (кадров, 2): центр цели, px
    kind: str


def _writer(path: Path) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"кодек mp4v не открылся: {path}")
    return writer


def _noisy(image: np.ndarray, rng: np.random.Generator, sigma: float) -> np.ndarray:
    noise = rng.standard_normal(image.shape, dtype=np.float32) * np.float32(sigma)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _texture(rng: np.random.Generator, sigma: float, blur: int) -> np.ndarray:
    field = rng.normal(0.0, sigma, (HEIGHT, WIDTH, 1)).astype(np.float32)
    return cv2.GaussianBlur(field, (blur, blur), 0).reshape(HEIGHT, WIDTH, 1)


def _vertical(top: tuple, bottom: tuple, height: int) -> np.ndarray:
    share = np.linspace(0.0, 1.0, height, dtype=np.float32).reshape(height, 1, 1)
    return (1.0 - share) * np.array(top, np.float32) + share * np.array(bottom, np.float32)


# ---------------------------------------------------------------- мяч


def _room(rng: np.random.Generator) -> np.ndarray:
    """Комната без зелёного: стена, пол, полка."""
    image = np.zeros((HEIGHT, WIDTH, 3), np.float32)
    image[:600] = _vertical((170, 178, 186), (140, 148, 158), 600)
    image[600:] = _vertical((70, 88, 108), (55, 70, 90), HEIGHT - 600)
    image += _texture(rng, 14.0, 9)
    cv2.rectangle(image, (60, 300), (420, 316), (90, 100, 115), -1)
    cv2.rectangle(image, (980, 120), (1180, 330), (120, 110, 100), -1)
    return np.clip(image, 0, 255).astype(np.uint8)


def green_ball(folder: Path, frames: int = 190, seed: int = 0) -> Clip:
    """Мяч водят рукой: крупный (≈28 000 px), быстрый (до ≈1400 px/с, до ≈45 px за кадр).

    Путь руки не повторяется: сумма синусов с несоизмеримыми частотами. Строго периодичный
    мах (маятник) MOG2 за пару периодов выучивает второй модой фона — так фон гасит качание
    листвы, — и по движению мяч пропадает на всей дуге.
    """
    rng = np.random.default_rng(seed)
    room = _room(rng)
    phase = rng.uniform(0.0, 2.0 * np.pi, 4)
    radius = 95
    path = Path(folder) / "green_ball.mp4"
    truth = []
    writer = _writer(path)
    try:
        for index in range(frames):
            t = index / FPS
            center = np.array([
                640.0 + 330.0 * np.sin(2.0 * np.pi * 0.37 * t + phase[0]) + 120.0 * np.sin(2.0 * np.pi * 0.83 * t + phase[1]),
                430.0 + 120.0 * np.sin(2.0 * np.pi * 0.53 * t + phase[2]) + 50.0 * np.sin(2.0 * np.pi * 1.21 * t + phase[3]),
            ])
            image = room.copy()
            # Посторонний предмет: тёмная коробка медленно едет по полке, она мельче мяча.
            box_x = int(150 + 60 * np.sin(t * 0.7))
            cv2.rectangle(image, (box_x, 250), (box_x + 50, 300), (60, 55, 50), -1)
            cx, cy = int(round(center[0])), int(round(center[1]))
            cv2.circle(image, (cx, cy), radius, (40, 150, 45), -1, cv2.LINE_AA)
            cv2.circle(image, (cx - 25, cy - 25), radius // 2, (70, 200, 80), -1, cv2.LINE_AA)
            gain = 1.0 + 0.01 * np.sin(t * 9.0)  # мерцание света
            image = np.clip(image.astype(np.float32) * gain, 0, 255).astype(np.uint8)
            writer.write(_noisy(image, rng, 2.0))
            truth.append(center)
    finally:
        writer.release()
    return Clip(path, np.array(truth), "green")


# ---------------------------------------------------------------- небо


def _sky(rng: np.random.Generator) -> np.ndarray:
    """Голубое небо (верхние 62 %), крыши и земля ниже."""
    horizon = int(HEIGHT * 0.62)
    image = np.zeros((HEIGHT, WIDTH, 3), np.float32)
    image[:horizon] = _vertical((215, 170, 120), (232, 212, 190), horizon)
    image[horizon:] = _vertical((70, 95, 85), (40, 60, 55), HEIGHT - horizon)
    image[horizon:] += _texture(rng, 18.0, 7)[horizon:]
    x = 0
    while x < WIDTH:
        width = int(rng.integers(70, 160))
        top = horizon - int(rng.integers(5, 40))
        cv2.rectangle(image, (x, top), (x + width, horizon + 5), (75, 80, 95), -1)
        x += width + int(rng.integers(10, 40))
    return np.clip(image, 0, 255).astype(np.uint8)


def _clouds(rng: np.random.Generator) -> np.ndarray:
    """Светлые облака: добавка к небу, шире кадра — её можно сдвигать."""
    field = np.zeros((HEIGHT, WIDTH * 2, 1), np.float32)
    for _ in range(14):
        center = (int(rng.integers(0, WIDTH * 2)), int(rng.integers(20, int(HEIGHT * 0.5))))
        axes = (int(rng.integers(60, 180)), int(rng.integers(20, 50)))
        cv2.ellipse(field, center, axes, 0, 0, 360, 22.0, -1)
    return cv2.GaussianBlur(field, (61, 61), 0).reshape(HEIGHT, WIDTH * 2, 1)


def _watermark(image: np.ndarray) -> np.ndarray:
    """Полупрозрачная надпись в центре кадра — как на исходном ролике (детектор её вырезает)."""
    layer = image.copy()
    cv2.putText(layer, "PREVIEW", (330, 330), cv2.FONT_HERSHEY_SIMPLEX, 4.0, (255, 255, 255), 10, cv2.LINE_AA)
    return cv2.addWeighted(layer, 0.3, image, 0.7, 0.0)


def sky_uav(folder: Path, frames: int = 300, seed: int = 1) -> Clip:
    """Дрон на голубом небе в верхней полосе кадра, вне окна надписи.

    Детектор sky_dark (app/motion.py) берёт пиксели, которые одновременно «тёмные»
    (яркость ≤ 85) и «небо» — голубые (H 90–135, S ≥ 25, V ≥ 120). Тёмный предмет
    на голубом небе сквозь дымку именно тёмно-синий; такого цвета и силуэт дрона.
    """
    rng = np.random.default_rng(seed)
    base = _sky(rng)
    clouds = _clouds(rng)
    path = Path(folder) / "sky_uav.mp4"
    truth = []
    writer = _writer(path)
    try:
        for index in range(frames):
            t = index / FPS
            x = 120.0 + (WIDTH - 240.0) * (0.5 - 0.5 * np.cos(2.0 * np.pi * t / 10.0))
            y = 120.0 + 45.0 * np.sin(2.0 * np.pi * t / 3.3)
            scale = 1.0 + 0.15 * np.sin(2.0 * np.pi * t / 6.0)
            drift = int(t * 8.0) % WIDTH
            image = base.astype(np.float32)
            horizon = int(HEIGHT * 0.62)
            image[:horizon] += clouds[:horizon, drift : drift + WIDTH]
            image = np.clip(image, 0, 255).astype(np.uint8)
            cx, cy = int(round(x)), int(round(y))
            body = (max(3, int(9 * scale)), max(2, int(4 * scale)))
            cv2.ellipse(image, (cx, cy), body, 0, 0, 360, (130, 45, 22), -1, cv2.LINE_AA)
            arm = int(11 * scale)
            for dx, dy in ((-arm, -arm // 2), (arm, -arm // 2), (-arm, arm // 2), (arm, arm // 2)):
                cv2.line(image, (cx, cy), (cx + dx, cy + dy), (135, 50, 25), 2, cv2.LINE_AA)
            image = _watermark(image)
            writer.write(_noisy(image, rng, 1.5))
            truth.append((x, y))
    finally:
        writer.release()
    return Clip(path, np.array(truth), "sky_dark")


# ---------------------------------------------------------------- дрон в помещении


def _hall(rng: np.random.Generator) -> np.ndarray:
    """Помещение: серые стены, пол, лампы у потолка, светлое окно у края, белая розетка."""
    image = np.zeros((HEIGHT, WIDTH, 3), np.float32)
    image[:540] = _vertical((128, 138, 148), (112, 120, 130), 540)
    image[540:] = _vertical((80, 95, 110), (60, 72, 85), HEIGHT - 540)
    image += _texture(rng, 8.0, 9)
    for x in (200, 560, 920):
        cv2.rectangle(image, (x, 10), (x + 160, 40), (250, 250, 250), -1)  # лампы: верхние 10 % кадра
    cv2.rectangle(image, (0, 120), (40, 480), (240, 240, 235), -1)  # окно у левого края
    # Розетка мельче дрона и в стороне от его пути. На пути она перехватывает трек white_blob:
    # дрон касается её винтами, трек переходит на неподвижное пятно и с него уже не уходит.
    cv2.rectangle(image, (120, 470), (150, 510), (238, 238, 238), -1)
    return np.clip(image, 0, 255).astype(np.uint8)


def white_drone(folder: Path, frames: int = 300, seed: int = 2) -> Clip:
    """Светлый квадрокоптер (≈3000 px) летает восьмёркой; на полу его тень."""
    rng = np.random.default_rng(seed)
    hall = _hall(rng)
    path = Path(folder) / "white_drone.mp4"
    truth = []
    writer = _writer(path)
    try:
        for index in range(frames):
            t = index / FPS
            x = WIDTH / 2.0 + 380.0 * np.sin(2.0 * np.pi * t / 7.0)
            y = 300.0 + 140.0 * np.sin(4.0 * np.pi * t / 7.0)
            image = hall.copy()
            cx, cy = int(round(x)), int(round(y))
            cv2.ellipse(image, (cx, 640), (60, 12), 0, 0, 360, (45, 55, 65), -1, cv2.LINE_AA)
            for dx, dy in ((-38, -16), (38, -16), (-38, 16), (38, 16)):
                cv2.line(image, (cx, cy), (cx + dx, cy + dy), (225, 225, 228), 4, cv2.LINE_AA)
                cv2.circle(image, (cx + dx, cy + dy), 12, (215, 215, 220), -1, cv2.LINE_AA)
            cv2.ellipse(image, (cx, cy), (34, 13), 0, 0, 360, (242, 242, 242), -1, cv2.LINE_AA)
            writer.write(_noisy(image, rng, 2.0))
            truth.append((x, y))
    finally:
        writer.release()
    return Clip(path, np.array(truth), "white_drone")


# Вид правды в _eval_hold_trail.py → сцена.
KINDS = {"green": green_ball, "sky_dark": sky_uav, "white_drone": white_drone}
