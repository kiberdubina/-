import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.metrics3d import (
    HIT_THRESHOLD_M,
    compare_tracks,
    load_truth_meters,
    write_metrics_report,
)


class Metrics3DTest(unittest.TestCase):
    def test_noisy_estimate_has_finite_rmse(self):
        rng = np.random.default_rng(0)
        truth = np.array(
            [
                [10.0, 0.0, 2.0],
                [12.0, 1.0, 2.5],
                [14.0, -0.5, 3.0],
                [16.0, 0.2, 2.8],
            ],
            np.float64,
        )
        est = truth + rng.normal(0.0, 0.3, truth.shape)
        stats = compare_tracks(est, truth)
        self.assertTrue(np.isfinite(stats["rmse_m"]))
        self.assertGreater(stats["rmse_m"], 0.0)
        self.assertLess(stats["rmse_m"], HIT_THRESHOLD_M)
        self.assertAlmostEqual(stats["hit_ratio"], 1.0)

    def test_empty_estimate_has_zero_hit_ratio(self):
        truth = np.ones((20, 3), np.float64)
        est = np.zeros((0, 3), np.float64)
        stats = compare_tracks(est, truth)
        self.assertEqual(stats["hit_ratio"], 0.0)
        self.assertFalse(np.isfinite(stats["rmse_m"]))

    def test_write_report_is_utf8(self):
        stats = compare_tracks(
            np.array([[0.0, 0.0, 0.0]], np.float64),
            np.array([[0.1, 0.0, 0.0]], np.float64),
        )
        with tempfile.TemporaryDirectory() as folder:
            out = write_metrics_report(stats, Path(folder) / "metrics.txt")
            text = out.read_text(encoding="utf-8")
        self.assertIn("RMSE, м", text)
        self.assertIn("Доля попаданий", text)
        self.assertIn("Метрики 3D-трассы", text)

    def test_load_truth_from_project_csv(self):
        track = load_truth_meters(ROOT)
        self.assertEqual(track.shape, (1002, 3))
        self.assertTrue(np.isfinite(track).all())


if __name__ == "__main__":
    unittest.main()
