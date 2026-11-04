"""Устройство модели YOLO: CUDA, затем Apple MPS, затем CPU; PELENG_DEVICE задаёт явно."""

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.object_detector import DEVICE_ENV, _device, pick_device


class _Tensor:
    def __init__(self, value: float):
        self.value = value

    def add(self, other: float) -> "_Tensor":
        return _Tensor(self.value + other)

    def cpu(self) -> list[float]:
        return [self.value]


def _torch(cuda: bool, mps: bool, mps_works: bool = True):
    def ones(size, device="cpu"):
        if device == "mps" and not mps_works:
            raise RuntimeError("MPS backend out of memory")
        return _Tensor(1.0)

    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
        ones=ones,
    )


class PickDeviceTest(unittest.TestCase):
    def test_cuda_first(self):
        self.assertEqual(pick_device(_torch(cuda=True, mps=True)), "cuda")

    def test_mps_without_cuda(self):
        self.assertEqual(pick_device(_torch(cuda=False, mps=True)), "mps")

    def test_cpu_when_nothing_else(self):
        self.assertEqual(pick_device(_torch(cuda=False, mps=False)), "cpu")

    def test_broken_mps_falls_back_to_cpu(self):
        self.assertEqual(pick_device(_torch(cuda=False, mps=True, mps_works=False)), "cpu")

    def test_old_torch_without_mps_backend(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False), backends=SimpleNamespace())
        self.assertEqual(pick_device(torch), "cpu")

    def test_forced_cpu_wins_over_gpu(self):
        self.assertEqual(pick_device(_torch(cuda=True, mps=True), " CPU "), "cpu")

    def test_forced_device_missing_gives_cpu(self):
        self.assertEqual(pick_device(_torch(cuda=False, mps=True), "cuda"), "cpu")
        self.assertEqual(pick_device(_torch(cuda=True, mps=False), "mps"), "cpu")
        self.assertEqual(pick_device(_torch(cuda=True, mps=True), "gpu"), "cpu")

    def test_forced_mps(self):
        self.assertEqual(pick_device(_torch(cuda=True, mps=True), "mps"), "mps")

    def test_environment_variable_overrides(self):
        with mock.patch.dict(os.environ, {DEVICE_ENV: "cpu"}):
            self.assertEqual(_device(), "cpu")


class RealTorchTest(unittest.TestCase):
    def test_real_device_is_known(self):
        try:
            import torch
        except ImportError:
            self.skipTest("нет torch")
        with mock.patch.dict(os.environ, {DEVICE_ENV: ""}):
            device = _device()
        self.assertIn(device, ("cuda", "mps", "cpu"))
        if not torch.cuda.is_available() and torch.backends.mps.is_available():
            self.assertEqual(device, "mps")


if __name__ == "__main__":
    unittest.main()
