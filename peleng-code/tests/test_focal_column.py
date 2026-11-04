"""Фокус камеры из таблицы позы: в шапке на всю запись (Way1) или колонкой по кадрам (Way2)."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.cine_table import camera_table, pose_in_table

WAY1_HEADER = "Frame;Loc_X;Loc_Y;Loc_Z;Rot_Roll;Rot_Pitch;Rot_Yaw;;;Focal length;35"
WAY2_HEADER = "Frame;Loc_X;Loc_Y;Loc_Z;Rot_Roll;Rot_Pitch;Rot_Yaw;Focal_Length_mm"
SENSOR_W_MM = 23.76


def _table(path: Path, header: str, focals: list[str | None]) -> Path:
    rows = [header]
    for frame, focal in enumerate(focals):
        cells = [str(frame), "100.0", "200.0", "500.0", "0.0", "-5.0", "30.0"]
        if focal is not None:
            cells.append(focal)
        rows.append(";".join(cells))
    path.write_text("\n".join(rows), encoding="utf-8")
    return path


def _fx(focal_mm: float, width: int = 1920) -> float:
    return focal_mm / SENSOR_W_MM * width


class FocalColumnTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)

    def tearDown(self):
        self.folder.cleanup()

    def test_way2_focal_column_is_used(self):
        path = _table(self.root / "cam5.csv", WAY2_HEADER, ["25.0"] * 4)
        clip = camera_table(path)
        self.assertAlmostEqual(clip.focal_mm, 25.0)
        pose = pose_in_table(path, 2, 1920, 1080)
        self.assertAlmostEqual(pose.K[0, 0], _fx(25.0))

    def test_focal_changes_by_frame_and_gaps_are_filled(self):
        path = _table(self.root / "zoom.csv", WAY2_HEADER, ["35.0", "", "#ЗНАЧ!", "70.0"])
        self.assertAlmostEqual(pose_in_table(path, 0, 1920, 1080).K[0, 0], _fx(35.0))
        self.assertAlmostEqual(pose_in_table(path, 3, 1920, 1080).K[0, 0], _fx(70.0))
        middle = pose_in_table(path, 1, 1920, 1080).K[0, 0]
        self.assertTrue(_fx(35.0) < middle < _fx(70.0))
        # Кадр за концом таблицы — крайний, как у позы.
        self.assertAlmostEqual(pose_in_table(path, 99, 1920, 1080).K[0, 0], _fx(70.0))

    def test_way1_header_focal_is_kept(self):
        path = _table(self.root / "way1.csv", WAY1_HEADER, [None] * 3)
        clip = camera_table(path)
        self.assertIsNone(clip.focals)
        self.assertAlmostEqual(clip.focal_mm, 35.0)
        self.assertAlmostEqual(pose_in_table(path, 1, 1920, 1080).K[0, 0], _fx(35.0))

    def test_column_without_values_falls_back_to_default(self):
        path = _table(self.root / "empty.csv", WAY2_HEADER, [None] * 3)
        clip = camera_table(path)
        self.assertIsNone(clip.focals)
        self.assertAlmostEqual(clip.focal_mm, 35.0)

    def test_real_way2_tables(self):
        folder = ROOT / "scene2"
        if not (folder / "object_transform_CineCameraActor5.csv").is_file():
            self.skipTest("нет записи scene2")
        expected = {1: 35.0, 2: 60.0, 4: 50.0, 5: 25.0}
        for number, focal in expected.items():
            pose = pose_in_table(folder / f"object_transform_CineCameraActor{number}.csv", 500, 1920, 1080)
            self.assertTrue(np.isclose(pose.K[0, 0], _fx(focal)), f"cam-{number}")


if __name__ == "__main__":
    unittest.main()
