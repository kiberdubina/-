import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.cine_table import camera_table, pose_in_table, table_kind, track_table
from app.control_run import scene_clips
from app.scene_folder import FilePose, safe_name, scan_folder, scene_poses, scene_truth

HEADER = "Frame;Loc_X;Loc_Y;Loc_Z;Rot_Roll;Rot_Pitch;Rot_Yaw;;;Focal length;35"


def _video(path: Path, frames: int = 5) -> None:
    sink = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    for index in range(frames):
        sink.write(np.full((48, 64, 3), 20 * index, np.uint8))
    sink.release()


def _camera_csv(path: Path, start: int, count: int, x_cm: float = 0.0, yaw_step: float = 0.0, sep: str = ";") -> None:
    rows = [HEADER.replace(";", sep)]
    for frame in range(start, start + count):
        cells = [str(frame), f"{x_cm:.1f}", "100.0", "500.0", "0.0", "0.0", f"{-70.0 + yaw_step * (frame - start):.2f}"]
        rows.append(sep.join(cells))
    path.write_bytes("\r\n".join(rows).encode("cp1251"))


def _track_csv(path: Path, start: int, count: int) -> None:
    rows = ["Frame;Loc_X;Loc_Y;Loc_Z"]
    rows += [f"{frame};{100.0 * frame:.1f};0.0;300.0" for frame in range(start, start + count)]
    path.write_bytes("\r\n".join(rows).encode("cp1251"))


class SceneFolderTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)

    def test_match_by_name_number_and_order_with_offset(self):
        _video(self.root / "Way2.FinalImage.CineCameraActor2.mp4")
        _video(self.root / "camera_04.mp4")
        _video(self.root / "povorot.mp4")
        _video(self.root / "control_cam-1.mp4")  # свой результат прошлого прогона — не камера
        _camera_csv(self.root / "object_transform_CineCameraActor2.csv", 300, 6)
        _camera_csv(self.root / "object_transform_CineCameraActor4.csv", 300, 6, x_cm=1000.0)
        _camera_csv(self.root / "pose_C.csv", 300, 6, yaw_step=5.0)
        _track_csv(self.root / "SM_AA_track_cut.csv", 300, 6)
        scene = scan_folder(self.root)
        pairs = {camera.video.name: camera.table.name for camera in scene.cameras}
        self.assertEqual(pairs, {
            "Way2.FinalImage.CineCameraActor2.mp4": "object_transform_CineCameraActor2.csv",
            "camera_04.mp4": "object_transform_CineCameraActor4.csv",
            "povorot.mp4": "pose_C.csv",
        })
        self.assertTrue(scene.has_poses)
        self.assertEqual(scene.frame_offset, 300)
        self.assertEqual(scene.target.name, "SM_AA_track_cut.csv")
        self.assertTrue(any("по порядку" in note for note in scene.notes))
        turning = {camera.video.name for camera in scene.cameras if camera.turning}
        self.assertEqual(turning, {"povorot.mp4"})
        truth = scene_truth(scene)
        # Кадр 0 роликов = кадр 300 таблицы: X = 300 м.
        self.assertAlmostEqual(truth[0][0], 300.0)
        self.assertEqual(len(scene_clips(scene)), 3)

    def test_file_pose_uses_offset(self):
        _video(self.root / "cam1.mp4")
        _camera_csv(self.root / "cam1.csv", 10, 5, yaw_step=10.0)
        scene = scan_folder(self.root)
        pose = scene_poses(scene)[scene.cameras[0].camera_id]
        first = pose(0)
        direct = pose_in_table(self.root / "cam1.csv", 10, 64, 48)
        self.assertTrue(np.allclose(first.rvec, direct.rvec))
        self.assertFalse(np.allclose(pose(2).rvec, first.rvec))
        self.assertIsInstance(pose, FilePose)

    def test_no_tables_is_two_d(self):
        _video(self.root / "любой ролик.mp4")
        scene = scan_folder(self.root)
        self.assertEqual(len(scene.cameras), 1)
        self.assertFalse(scene.has_poses)
        self.assertTrue(any("только 2D" in note for note in scene.notes))

    def test_videos_in_subfolders_and_bad_video(self):
        (self.root / "видео").mkdir()
        (self.root / "execl").mkdir()
        _video(self.root / "видео" / "A_1.mp4")
        _video(self.root / "видео" / "B_2.mp4")
        (self.root / "видео" / "битый.mp4").write_bytes(b"not a video")
        _camera_csv(self.root / "execl" / "object_transform_Cam1.csv", 0, 5)
        _camera_csv(self.root / "execl" / "object_transform_Cam2.csv", 0, 5)
        scene = scan_folder(self.root)
        self.assertEqual(len(scene.cameras), 2)
        self.assertTrue(scene.has_poses)
        self.assertTrue(any("не открылся" in note for note in scene.notes))

    def test_synthehicle_layout(self):
        import json

        calib = self.root / "calibration" / "overlapping" / "Town04" / "camera_info"
        calib.mkdir(parents=True)
        for number, x in ((1, 0.0), (2, 40.0)):
            extrinsic = np.eye(4)
            extrinsic[:3, 3] = [-x, 0.0, -6.0]  # мир→камера: центр (x, 0, 6), смотрит вдоль X Unreal
            (calib / f"camera_{number}.txt").write_text(json.dumps({
                "intrinsic_matrix": [[960.0, 0, 960.0], [0, 960.0, 540.0], [0, 0, 1]],
                "extrinsic_matrix": extrinsic.tolist(), "x": x, "y": 0.0, "z": 6.0,
            }))
        scene = self.root / "train" / "Town04-O-day"
        for camera in ("C01", "C02"):
            (scene / camera / "out_rgb").mkdir(parents=True)
            _video(scene / camera / "out_rgb" / "video.mp4")
            (scene / camera / "gt").mkdir()
            (scene / camera / "gt" / "gt.txt").write_text("1,7,100,200,50,40,1,-1,-1,-1\n")
        found = scan_folder(scene)
        pairs = {camera.video.parent.parent.name: camera.table.name for camera in found.cameras}
        self.assertEqual(pairs, {"C01": "camera_1.txt", "C02": "camera_2.txt"})
        self.assertTrue(found.has_poses)
        self.assertEqual([camera.camera_id for camera in found.cameras], ["cam-1", "cam-2"])
        self.assertFalse(any("gt.txt" in note for note in found.notes))
        pose = scene_poses(found)["cam-2"](0)
        rotation, _ = cv2.Rodrigues(pose.rvec)
        self.assertTrue(np.allclose((-rotation.T @ pose.tvec).ravel(), [40.0, 0.0, 6.0]))

    def test_empty_folder(self):
        scene = scan_folder(self.root)
        self.assertEqual(scene.cameras, [])
        self.assertTrue(scene.notes)

    def test_safe_name(self):
        self.assertEqual(safe_name('Финал: 3 камеры?'), "Финал__3_камеры")
        self.assertEqual(safe_name("..."), "scene")


class RobustTableTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)

    def test_bad_rows_are_filled_not_fatal(self):
        path = self.root / "cam.csv"
        _camera_csv(path, 0, 10, x_cm=1000.0)
        lines = path.read_bytes().decode("cp1251").split("\r\n")
        lines[3] = "2;12.янв;100.0;500.0;0.0;0.0;-70.0"   # Excel: число стало датой
        lines[5] = "4;;;"                                  # пустая строка
        path.write_bytes("\r\n".join(lines).encode("cp1251"))
        clip = camera_table(path)
        self.assertEqual(clip.bad_rows, 2)
        self.assertEqual(clip.centers.shape[0], 10)
        self.assertTrue(np.allclose(clip.centers[2], clip.centers[1]))
        self.assertTrue(np.isfinite(clip.rvecs).all())

    def test_comma_separated_table(self):
        path = self.root / "cam.csv"
        _camera_csv(path, 0, 4, x_cm=250.0, sep=",")
        self.assertEqual(table_kind(path), "camera")
        clip = camera_table(path)
        self.assertAlmostEqual(clip.centers[0][0], 2.5)
        self.assertAlmostEqual(clip.focal_mm, 35.0)

    def test_track_missing_frame_and_bad_cell(self):
        path = self.root / "track.csv"
        rows = ["Frame;Loc_X;Loc_Y;Loc_Z", "0;0;0;300", "1;#ЗНАЧ!;0;300", "3;300;0;300"]
        path.write_bytes("\r\n".join(rows).encode("cp1251"))
        self.assertEqual(table_kind(path), "track")
        track = track_table(path)
        self.assertEqual(track.shape, (4, 3))
        self.assertTrue(np.isfinite(track).all())
        self.assertAlmostEqual(track[2][0], 2.0)

    def test_excel_table(self):
        try:
            from openpyxl import Workbook
        except ImportError:
            self.skipTest("нет openpyxl")
        book = Workbook()
        sheet = book.active
        sheet.append(["Frame", "Loc_X", "Loc_Y", "Loc_Z", "Rot_Roll", "Rot_Pitch", "Rot_Yaw", None, None, "Focal length", 35])
        for frame in range(3):
            sheet.append([frame, 100.0 * frame, 0.0, 500.0, 0.0, 0.0, -70.0])
        path = self.root / "cam.xlsx"
        book.save(path)
        self.assertEqual(table_kind(path), "camera")
        clip = camera_table(path)
        self.assertAlmostEqual(clip.centers[2][0], 2.0)


class SessionPoseTablesTest(unittest.TestCase):
    def test_refresh_pose_from_table(self):
        from types import SimpleNamespace

        from app.profile import load_patterns
        from app.session import Session
        from app.source import Frame, SyntheticSource

        session = Session(load_patterns(ROOT / "config.yaml")["motion"], SyntheticSource())
        calls = []

        def table(frame):
            calls.append(frame)
            return f"поза {frame}"

        session.set_pose_tables({"cam-7": table}, {"cam-7"})
        frame = Frame(np.zeros((4, 4, 3), np.uint8), 0.0, "cam-7", 12)
        session._refresh_pose(SimpleNamespace(source=None), frame)
        self.assertEqual(session.poses["cam-7"], "поза 12")
        self.assertEqual(calls, [12])
        session.set_sources([SyntheticSource()])
        self.assertEqual(session.pose_tables, {})
        self.assertEqual(session.moving_pose_ids, {"cam-5"})


if __name__ == "__main__":
    unittest.main()
