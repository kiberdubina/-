from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Frame:
    image: np.ndarray
    time_s: float
    camera_id: str


class SyntheticSource:
    """Движущийся круг на неподвижном фоне. Замена вебкамеры."""

    def __init__(self, width: int = 640, height: int = 480, fps: float = 25):
        self.width = width
        self.height = height
        self.fps = fps
        self.index = 0
        rng = np.random.default_rng(0)
        self.background = rng.integers(35, 55, (height, width, 3), dtype=np.uint8)

    def read(self) -> Frame:
        t = self.index / self.fps
        self.index += 1
        image = self.background.copy()
        cx = int(self.width * (0.5 + 0.32 * np.sin(t * 1.3)))
        cy = int(self.height * (0.5 + 0.28 * np.sin(t * 0.85 + 0.6)))
        cv2.circle(image, (cx, cy), 72, (40, 180, 40), -1)
        return Frame(image, t, "demo")


class VideoSource:
    def __init__(self, path: str, camera_id: str):
        if isinstance(path, int):
            self.capture = cv2.VideoCapture(path, cv2.CAP_DSHOW)
        elif "://" in str(path):
            self.capture = cv2.VideoCapture()
            self.capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)
            self.capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000)
            self.capture.open(str(path))
        else:
            self.capture = cv2.VideoCapture(path)
        self.kind = camera_id
        self.camera_id = Path(path).name if camera_id == "file" else camera_id
        self.index = 0
        self.rewound = False
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 1 else 25

    def is_open(self) -> bool:
        return self.capture.isOpened()

    def read(self) -> Frame | None:
        self.rewound = False
        ok, image = self.capture.read()
        if not ok or image is None:
            if self.kind == "file":
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.index = 0
                self.rewound = True
                ok, image = self.capture.read()
            if not ok or image is None:
                return None
        t = self.index / self.fps
        self.index += 1
        return Frame(image, t, self.camera_id)

    def close(self) -> None:
        self.capture.release()


def _open_capture(target: str | int) -> VideoSource | None:
    source = VideoSource(target, "phone-1")
    if not source.is_open():
        source.close()
        return None
    frame = source.read()
    if frame is None:
        source.close()
        return None
    source.index = 0
    return source


class BlankSource:
    """Пустой кадр, пока телефон не подключился. Это не тестовый круг."""

    def __init__(self) -> None:
        self.kind = "blank"
        self.camera_id = "wait"
        self.rewound = False

    def read(self):
        return None

    def is_open(self) -> bool:
        return True

    def close(self) -> None:
        return None


def find_phone_source() -> VideoSource | None:
    """Камера, которую Windows уже отдала как обычное USB-устройство."""
    for index in range(3):
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            continue
        ok, image = capture.read()
        capture.release()
        if ok and image is not None:
            return _open_capture(index)
    return None
