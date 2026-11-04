import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.excel_export import control_workbook, summary_sheet, table_from_csv, write_workbook
from app.export_result import write_xyz_csv

try:
    import openpyxl
except ImportError:  # pragma: no cover - пакет из requirements.txt
    openpyxl = None


class ExcelExportTest(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.folder)

    def _trajectory_csv(self) -> Path:
        path = self.folder / "trajectory.csv"
        path.write_text(
            "frame,t_s,x_m,y_m,z_m,cameras,error_m\n"
            "0,0.0000,-213.7808,129.3102,3.6205,cam-1 cam-5,1.8942\n"
            "1,0.0333,-213.6130,129.1023,3.6219,,\n",
            encoding="utf-8",
        )
        return path

    def test_csv_numbers_become_numbers_and_headers_russian(self):
        sheet = table_from_csv("Траектория", self._trajectory_csv())
        self.assertEqual(sheet.header[:3], ["Кадр", "Время, с", "X, м"])
        self.assertEqual(sheet.rows[0][0], 0)
        self.assertIsInstance(sheet.rows[0][0], int)
        self.assertAlmostEqual(sheet.rows[0][2], -213.7808)
        self.assertEqual(sheet.rows[0][5], "cam-1 cam-5")
        # Пустые ячейки CSV — пустые, не нули.
        self.assertIsNone(sheet.rows[1][5])
        self.assertIsNone(sheet.rows[1][6])

    def test_summary_has_metric_rows(self):
        sheet = summary_sheet({"frames": 1001, "coverage": 1.0, "rmse_m": 2.93, "median_m": 1.55})
        labels = [row[0] for row in sheet.rows]
        self.assertIn("RMSE, м", labels)
        self.assertIn("Кадров с оценкой", labels)
        self.assertEqual(dict(sheet.rows)["Кадров с оценкой"], 1001)

    @unittest.skipIf(openpyxl is None, "нет openpyxl")
    def test_control_workbook_sheets_and_values(self):
        files = {"csv": self._trajectory_csv()}
        path = control_workbook(self.folder / "trajectory.xlsx", files, {"frames": 2, "rmse_m": 1.5})
        book = openpyxl.load_workbook(path)
        self.assertEqual(book.sheetnames, ["Итог", "Траектория"])
        page = book["Траектория"]
        self.assertEqual(page["A1"].value, "Кадр")
        self.assertAlmostEqual(page["C2"].value, -213.7808)
        self.assertEqual(page.freeze_panes, "A2")

    @unittest.skipIf(openpyxl is None, "нет openpyxl")
    def test_xyz_csv_round_trip(self):
        csv_path = write_xyz_csv([0.0, 0.5], [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])],
                                 self.folder / "xyz.csv")
        path = write_workbook(self.folder / "xyz.xlsx", [table_from_csv("Траектория", csv_path)])
        page = openpyxl.load_workbook(path)["Траектория"]
        self.assertEqual([cell.value for cell in page[1]], ["Время, с", "X, м", "Y, м", "Z, м"])
        self.assertEqual([cell.value for cell in page[3]], [0.5, 4.0, 5.0, 6.0])

    def test_without_openpyxl_returns_none(self):
        with mock.patch.dict(sys.modules, {"openpyxl": None}):
            path = write_workbook(self.folder / "x.xlsx", [table_from_csv("Траектория", self._trajectory_csv())])
        self.assertIsNone(path)
        self.assertFalse((self.folder / "x.xlsx").exists())


if __name__ == "__main__":
    unittest.main()
