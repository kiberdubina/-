"""Таблица точности по числу камер: сводка ошибки, разбивка по кадрам, наборы камер."""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.control_run import write_camera_table
from app.metrics3d import error_summary
from run_control import camera_sets


class ErrorSummaryTest(unittest.TestCase):
    def test_values(self):
        summary = error_summary(np.array([1.0, 2.0, 2.0, 3.0, np.nan]))
        self.assertEqual(summary["frames"], 4)
        self.assertAlmostEqual(summary["mae_m"], 2.0)
        self.assertAlmostEqual(summary["median_m"], 2.0)
        self.assertAlmostEqual(summary["rmse_m"], np.sqrt(4.5))
        self.assertAlmostEqual(summary["max_m"], 3.0)

    def test_empty(self):
        summary = error_summary(np.array([]))
        self.assertEqual(summary["frames"], 0)
        self.assertTrue(np.isnan(summary["rmse_m"]))


class CameraTableTest(unittest.TestCase):
    def test_rows_by_camera_count(self):
        truth = np.zeros((4, 3))
        anchors = np.array([[1.0, 0, 0], [2.0, 0, 0], [0, 3.0, 0], [0, 0, 4.0]])
        observed = {0: {"cam-1": 1}, 1: {"cam-1": 1, "cam-2": 1}, 2: {"cam-1": 1, "cam-2": 1}, 3: {}}
        with tempfile.TemporaryDirectory() as folder:
            path = write_camera_table(Path(folder) / "t.csv", [0, 1, 2, 3], anchors, truth, observed)
            rows = list(csv.DictReader(path.open(encoding="utf-8")))
        by_count = {int(row["cameras"]): row for row in rows}
        self.assertEqual(sorted(by_count), [0, 1, 2])
        self.assertEqual(int(by_count[2]["frames"]), 2)
        self.assertAlmostEqual(float(by_count[2]["mae_m"]), 2.5)
        self.assertAlmostEqual(float(by_count[2]["share"]), 0.5)
        self.assertAlmostEqual(float(by_count[0]["max_m"]), 4.0)


class CameraSetsTest(unittest.TestCase):
    def test_from_all_to_one_with_limit(self):
        sets = camera_sets(["a", "b", "c", "d"], 2)
        self.assertEqual(sets[0], ("a", "b", "c", "d"))
        sizes = [len(item) for item in sets]
        self.assertEqual(sizes, sorted(sizes, reverse=True))
        for size in (1, 2, 3):
            self.assertLessEqual(sizes.count(size), 2)
            self.assertGreaterEqual(sizes.count(size), 1)


if __name__ == "__main__":
    unittest.main()
