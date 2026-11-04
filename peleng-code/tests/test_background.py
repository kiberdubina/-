import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.background import (
    BackgroundConfig,
    RecordingBackground,
    RotatingBackground,
    StaticBackground,
    foreground_blobs,
    masked_median,
)
from app.geom import CameraPose, default_intrinsics

W, H = 320, 180
CONFIG = BackgroundConfig(work_width=W)


def textured(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(90, 170, (H // 4, W // 4, 3), dtype=np.uint8)
    return cv2.resize(base, (W, H), interpolation=cv2.INTER_LINEAR)


def yaw_pose(yaw: float) -> CameraPose:
    """Камера в начале координат, смотрит горизонтально по азимуту yaw (Z мира вверх)."""
    forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    right = np.array([np.sin(yaw), -np.cos(yaw), 0.0])
    down = np.array([0.0, 0.0, -1.0])
    rotation = np.stack([right, down, forward])
    rvec, _ = cv2.Rodrigues(rotation)
    matrix, dist = default_intrinsics(W, H)
    return CameraPose("rot", matrix, dist, rvec, np.zeros((3, 1)), W, H)


def render_world(pose: CameraPose) -> np.ndarray:
    """Кадр сцены «на бесконечности»: цвет зависит только от направления луча."""
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    rays = np.stack([u, v, np.ones_like(u)], axis=-1) @ np.linalg.inv(pose.K).T
    rotation, _ = cv2.Rodrigues(pose.rvec)
    world = rays @ rotation
    world /= np.linalg.norm(world, axis=-1, keepdims=True)
    azimuth = np.arctan2(world[..., 1], world[..., 0])
    elevation = np.arcsin(world[..., 2])
    image = np.stack(
        [
            120 + 60 * np.sin(9 * azimuth),
            120 + 60 * np.cos(7 * azimuth + 5 * elevation),
            120 + 50 * np.sin(11 * elevation + 3 * azimuth),
        ],
        axis=-1,
    )
    return np.clip(image, 0, 255).astype(np.uint8)


class StaticBackgroundTest(unittest.TestCase):
    def test_moving_square_found_and_shadow_ignored(self) -> None:
        scene = textured()
        model = StaticBackground(W, H, CONFIG)
        for _ in range(10):
            model.apply(scene)
        frame = scene.copy()
        cv2.rectangle(frame, (40, 60), (90, 110), (20, 30, 220), -1)
        # Тень: та же текстура, яркость ×0.7, цветность сохранена.
        frame[120:170, 200:300] = (frame[120:170, 200:300].astype(np.float32) * 0.7).astype(np.uint8)
        mask = model.apply(frame)
        self.assertGreater(mask[60:110, 40:90].mean(), 0.9)
        self.assertLess(mask[125:165, 205:295].mean(), 0.05)
        blobs = foreground_blobs(mask, 1.0)
        biggest = max(blobs, key=lambda blob: blob.area)
        self.assertAlmostEqual(biggest.x, 40, delta=2)
        self.assertAlmostEqual(biggest.w, 51, delta=3)

    def test_exposure_change_is_not_foreground(self) -> None:
        scene = textured(1)
        model = StaticBackground(W, H, CONFIG)
        for _ in range(10):
            model.apply(scene)
        brighter = np.clip(scene.astype(np.float32) * 1.25, 0, 255).astype(np.uint8)
        self.assertLess(model.apply(brighter).mean(), 0.01)

    def test_hold_keeps_still_target_out_of_background(self) -> None:
        # Цель, что едет на камеру, почти не меняет пиксели: без hold она «призрак» и уходит в фон.
        scene = textured(2)
        frame = scene.copy()
        cv2.rectangle(frame, (100, 60), (160, 120), (20, 30, 220), -1)
        hold = np.zeros((H, W), bool)
        hold[50:130, 90:170] = True
        held = StaticBackground(W, H, CONFIG)
        plain = StaticBackground(W, H, CONFIG)
        for _ in range(10):
            held.apply(scene)
            plain.apply(scene)
        for _ in range(CONFIG.ghost_frames + 10):
            held_mask = held.apply(frame, hold=hold)
            plain_mask = plain.apply(frame)
        self.assertGreater(held_mask[60:120, 100:160].mean(), 0.9)
        self.assertLess(plain_mask[60:120, 100:160].mean(), 0.1)


class RotatingBackgroundTest(unittest.TestCase):
    def test_pure_rotation_gives_no_foreground(self) -> None:
        model = RotatingBackground(W, H, CONFIG)
        yaws = np.linspace(0.0, 0.8, 40)
        masks = [model.apply(render_world(yaw_pose(yaw)), yaw_pose(yaw)) for yaw in yaws]
        self.assertLess(float(np.mean([mask.mean() for mask in masks[5:]])), 0.01)

    def test_object_found_while_rotating(self) -> None:
        model = RotatingBackground(W, H, CONFIG)
        for yaw in np.linspace(0.0, 0.5, 25):
            model.apply(render_world(yaw_pose(yaw)), yaw_pose(yaw))
        pose = yaw_pose(0.45)
        frame = render_world(pose)
        cv2.circle(frame, (160, 90), 20, (255, 255, 255), -1)
        mask = model.apply(frame, pose)
        self.assertGreater(mask[80:100, 150:170].mean(), 0.9)
        self.assertLess(mask[:, :100].mean(), 0.02)


class MaskedMedianTest(unittest.TestCase):
    def test_matches_nanmedian(self) -> None:
        rng = np.random.default_rng(5)
        for count in (1, 2, 7, 8):
            stack = rng.normal(size=(count, 6, 5, 3)).astype(np.float32)
            stack[rng.random(stack.shape) < 0.3] = np.nan
            stack[:, 0, 0, 0] = np.nan
            with np.errstate(all="ignore"):
                import warnings

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    expected = np.nanmedian(stack, axis=0)
            np.testing.assert_allclose(masked_median(stack), expected, atol=1e-6, equal_nan=True)


class RecordingBackgroundTest(unittest.TestCase):
    def test_median_removes_passing_object(self) -> None:
        scene = textured(2)
        recording = RecordingBackground(W, H, CONFIG)
        pose = yaw_pose(0.0)
        for step in range(15):
            frame = scene.copy()
            cv2.rectangle(frame, (10 + 20 * step, 60), (40 + 20 * step, 100), (0, 0, 0), -1)
            recording.add(frame, pose)
        recording.build()
        self.assertFalse(recording.rotating)
        frame = scene.copy()
        cv2.rectangle(frame, (150, 60), (180, 100), (0, 0, 0), -1)
        mask = recording.apply(frame, pose)
        self.assertGreater(mask[62:98, 152:178].mean(), 0.9)
        self.assertLess(mask[:, :140].mean(), 0.01)

    def test_rotating_recording_uses_panorama(self) -> None:
        recording = RecordingBackground(W, H, CONFIG)
        for yaw in np.linspace(0.0, 0.6, 13):
            recording.add(render_world(yaw_pose(yaw)), yaw_pose(yaw))
        recording.build()
        self.assertTrue(recording.rotating)
        pose = yaw_pose(0.3)
        frame = render_world(pose)
        cv2.rectangle(frame, (100, 50), (140, 90), (255, 255, 255), -1)
        mask = recording.apply(frame, pose)
        self.assertGreater(mask[55:85, 105:135].mean(), 0.9)
        self.assertLess(mask[:, 200:].mean(), 0.02)


if __name__ == "__main__":
    unittest.main()
