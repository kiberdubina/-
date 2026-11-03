from dataclasses import dataclass

import cv2
import numpy as np

from app.profile import TargetProfile


@dataclass
class Detection:
    cx: float
    cy: float
    area: float
    bbox: tuple[int, int, int, int]


class MotionDetector:
    """Пятна цели. С цветом в профиле — зелёные, иначе движение на неподвижном фоне."""

    def __init__(self, profile: TargetProfile):
        self.profile = profile
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=120, varThreshold=24, detectShadows=False
        )
        self.kernel = np.ones((5, 5), np.uint8)
        self.small_kernel = np.ones((3, 3), np.uint8)
        self.seen = 0
        self.prev_gray: np.ndarray | None = None
        self.prev_xy: tuple[float, float] | None = None
        self.ego_miss = 0
        self.still_gray: np.ndarray | None = None
        self._still_prev: np.ndarray | None = None
        # Дырявое пятно руки сшивается, шум в пару пикселей — нет.
        self.diff_close = np.ones((15, 15), np.uint8)

    def detect(self, image: np.ndarray) -> list[Detection]:
        mode = (self.profile.mode or "auto").lower()
        if mode == "sky_dark":
            return self._spots(self._sky_dark_mask(image))
        if mode == "white_blob":
            return self._spots(self._white_blob_mask(image))
        if self.profile.moving_camera or mode == "ego":
            return self._ego(image)
        if self.profile.color is not None and self.profile.require_motion:
            color = self._color_mask(image)
            motion = self._motion_mask(image)
            if motion is None:
                return []
            return self._spots(cv2.bitwise_and(color, motion))
        if self.profile.color is not None or mode == "color":
            if self.profile.color is None:
                return []
            return self._spots(self._color_mask(image))
        self._push_still(image)
        motion = self._motion_mask(image)
        if motion is None:
            return []
        spots = self._spots(motion)
        if spots:
            return spots
        # Крупная область слабого контраста не проходит MOG2: дисперсия фона
        # уже широкая, а OPEN 5×5 стирает оставшуюся сетку. Разница соседних
        # кадров, без общего сдвига яркости, оставляет само пятно.
        diff = self._still_diff_mask()
        if diff is None:
            return []
        return self._spots(diff)

    def _sky_dark_mask(self, image: np.ndarray) -> np.ndarray:
        """Тёмный БПЛА на ярком небе; низ кадра (земля/здания) отсекается."""
        height, width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        sky = cv2.inRange(hsv, (0, 0, 145), (180, 80, 255))
        sky_blue = cv2.inRange(hsv, (90, 25, 120), (135, 255, 255))
        sky = cv2.bitwise_or(sky, sky_blue)
        # только верхние 62% — убрать крыши/пикселизацию
        sky[int(height * 0.62) :, :] = 0
        # вырезать центр кадра с водяными знаками (полупрозрачный текст)
        cy0, cy1 = int(height * 0.28), int(height * 0.55)
        cx0, cx1 = int(width * 0.15), int(width * 0.85)
        sky[cy0:cy1, cx0:cx1] = 0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dark = cv2.inRange(gray, 0, 85)
        mask = cv2.bitwise_and(dark, sky)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.small_kernel)

    def _white_blob_mask(self, image: np.ndarray) -> np.ndarray:
        """Светлый квадрокоптер в помещении (не лампы у верха, не пол)."""
        height, width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 0, 175), (180, 55, 255))
        mask[: int(height * 0.10), :] = 0
        mask[int(height * 0.75) :, :] = 0
        # боковые стены/окно часто очень светлые — слегка прижать края
        mask[:, : int(width * 0.04)] = 0
        mask[:, int(width * 0.96) :] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.small_kernel)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    def _push_still(self, image: np.ndarray) -> None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        self._still_prev = self.still_gray
        self.still_gray = gray

    def _still_diff_mask(self) -> np.ndarray | None:
        previous = self._still_prev
        current = self.still_gray
        if previous is None or current is None or previous.shape != current.shape:
            return None
        sample_prev = previous[::4, ::4].astype(np.int16)
        sample_cur = current[::4, ::4].astype(np.int16)
        delta = int(np.median(sample_cur - sample_prev))
        aligned = np.clip(previous.astype(np.int16) + delta, 0, 255).astype(np.uint8)
        diff = cv2.absdiff(current, aligned)
        _, mask = cv2.threshold(diff, 16, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.diff_close)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.small_kernel)

    def _motion_mask(self, image: np.ndarray) -> np.ndarray | None:
        mask = self.subtractor.apply(image)
        self.seen += 1
        if self.seen < 4:
            return None
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)

    def _color_mask(self, image: np.ndarray) -> np.ndarray:
        gate = self.profile.color
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            (gate.h_min, gate.s_min, gate.v_min),
            (gate.h_max, gate.s_max, gate.v_max),
        )
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)

    def _spots(self, mask: np.ndarray) -> list[Detection]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        spots: list[Detection] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.profile.s_min or area > self.profile.s_max:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            spots.append(
                Detection(
                    cx=float(x) + w / 2.0,
                    cy=float(y) + h / 2.0,
                    area=area,
                    bbox=(int(x), int(y), int(w), int(h)),
                )
            )
        return spots

    def _ego(self, image: np.ndarray) -> list[Detection]:
        """Пятно, которое двигается иначе, чем поворот камеры."""
        height, width = image.shape[:2]
        work_w = 360
        work_h = max(1, int(round(height * (work_w / float(width)))))
        small = cv2.resize(image, (work_w, work_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        sx = width / float(work_w)
        sy = height / float(work_h)
        self._sx = sx
        self._sy = sy
        previous = self.prev_gray
        self.prev_gray = gray
        if previous is None or previous.shape != gray.shape:
            return []
        points = cv2.goodFeaturesToTrack(previous, 160, 0.02, 12)
        if points is None:
            return []
        moved, status, _ = cv2.calcOpticalFlowPyrLK(previous, gray, points, None)
        status = status.reshape(-1).astype(bool)
        if int(status.sum()) < 12:
            return []
        affine, _ = cv2.estimateAffinePartial2D(
            points[status], moved[status], method=cv2.RANSAC, ransacReprojThreshold=3
        )
        if affine is None:
            return []
        warped = cv2.warpAffine(previous, affine, (work_w, work_h))
        diff = cv2.absdiff(gray, warped)
        diff[:4, :] = 0
        diff[-4:, :] = 0
        diff[:, :4] = 0
        diff[:, -4:] = 0
        predicted = self._warp_point(affine, self.prev_xy)
        _, mask = cv2.threshold(diff, 22, 255, cv2.THRESH_BINARY)
        if predicted is not None:
            x0 = int(max(0, predicted[0] - 48))
            y0 = int(max(0, predicted[1] - 48))
            x1 = int(min(work_w, predicted[0] + 48))
            y1 = int(min(work_h, predicted[1] + 48))
            if x1 > x0 and y1 > y0:
                region = diff[y0:y1, x0:x1]
                if region.size:
                    _, soft = cv2.threshold(region, 12, 255, cv2.THRESH_BINARY)
                    if soft.shape == region.shape:
                        mask[y0:y1, x0:x1] = cv2.bitwise_or(mask[y0:y1, x0:x1], soft)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.small_kernel)
        candidates = self._ego_candidates(mask, diff, sx, sy, work_w, work_h)
        chosen = self._choose_ego(candidates, predicted)
        if chosen is None and self.ego_miss > 8:
            self.prev_xy = None
            chosen = self._choose_ego(candidates, None)
        if chosen is None:
            self.ego_miss += 1
            self.prev_xy = predicted
            return []
        self.ego_miss = 0
        self.prev_xy = (chosen.cx / sx, chosen.cy / sy)
        return [chosen]

    def _warp_point(
        self, affine: np.ndarray, point: tuple[float, float] | None
    ) -> tuple[float, float] | None:
        if point is None:
            return None
        x, y = point
        return (
            float(affine[0, 0] * x + affine[0, 1] * y + affine[0, 2]),
            float(affine[1, 0] * x + affine[1, 1] * y + affine[1, 2]),
        )

    def _ego_candidates(
        self,
        mask: np.ndarray,
        diff: np.ndarray,
        sx: float,
        sy: float,
        work_w: int,
        work_h: int,
    ) -> list[tuple[float, Detection]]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found: list[tuple[float, Detection]] = []
        area_scale = sx * sy
        for contour in contours:
            area = float(cv2.contourArea(contour)) * area_scale
            if area < self.profile.s_min or area > self.profile.s_max:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w < 3 or h < 3 or max(w / h, h / w) > 4.2:
                continue
            if x < 3 or y < 3 or x + w > work_w - 3 or y + h > work_h - 3:
                continue
            patch = diff[y : y + h, x : x + w]
            strength = float(patch.mean()) if patch.size else 0.0
            detection = Detection(
                cx=(x + w / 2.0) * sx,
                cy=(y + h / 2.0) * sy,
                area=area,
                bbox=(int(x * sx), int(y * sy), max(1, int(w * sx)), max(1, int(h * sy))),
            )
            found.append((strength, detection))
        return found

    def _choose_ego(
        self,
        candidates: list[tuple[float, Detection]],
        predicted: tuple[float, float] | None,
    ) -> Detection | None:
        if not candidates:
            return None
        if predicted is None:
            return max(candidates, key=lambda item: item[0])[1]
        best: tuple[float, Detection] | None = None
        for strength, detection in candidates:
            distance = float(
                np.hypot(
                    detection.cx / self._sx - predicted[0],
                    detection.cy / self._sy - predicted[1],
                )
            )
            # Координаты предсказания в уменьшенном кадре. Дальше ~70 px там — уже другой предмет.
            if distance > 70:
                continue
            score = distance - strength * 0.15
            if best is None or score < best[0]:
                best = (score, detection)
        return None if best is None else best[1]
