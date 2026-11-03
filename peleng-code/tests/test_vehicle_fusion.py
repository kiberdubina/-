import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.background import Blob
from app.metrics3d import compare_by_frame, frame_errors
from app.vehicle import anchor_point, ctrv_step, predicted_sides
from app.vehicle_fusion import CameraView, FusionConfig, VehicleFusion, gather_box
from app.way1_files import find_file
from app.way1_sources import video_paths

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_vehicle import MODEL, RIG  # noqa: E402


def blobs_for(state: np.ndarray, pose, rng: np.random.Generator) -> list[Blob]:
    """Силуэт ящика (иногда разрезанный «деревом») и лист рядом — как даёт детектор."""
    sides = predicted_sides(state, MODEL, pose)
    if sides is None:
        return []
    x0, y0, x1, y1 = sides
    blobs = []
    if rng.random() < 0.3:
        cut = x0 + (x1 - x0) * rng.uniform(0.3, 0.6)
        blobs.append(Blob(x0, y0, cut - 2 - x0, y1 - y0, (cut - 2 - x0) * (y1 - y0) * 0.7))
        blobs.append(Blob(cut + 2, y0, x1 - cut - 2, y1 - y0, (x1 - cut - 2) * (y1 - y0) * 0.7))
    else:
        blobs.append(Blob(x0, y0, x1 - x0, y1 - y0, (x1 - x0) * (y1 - y0) * 0.7))
    blobs.append(Blob(x1 + 60, y0 - 40, 12, 12, 100.0))
    return blobs


class GatherBoxTest(unittest.TestCase):
    def test_split_silhouette_is_merged_and_far_leaf_is_not(self) -> None:
        predicted = np.array([100.0, 100.0, 300.0, 200.0])
        blobs = [
            Blob(100, 100, 90, 100, 7000),
            Blob(210, 100, 90, 100, 7000),
            Blob(330, 60, 20, 20, 300),
        ]
        rect = gather_box(predicted, blobs, FusionConfig())
        self.assertEqual(rect, (100, 100, 300, 200))

    def test_blob_mostly_outside_window_is_rejected(self) -> None:
        # Склейка цели с артефактом фона: большое пятно, лишь краем задевающее прогноз.
        predicted = np.array([100.0, 100.0, 300.0, 200.0])
        blobs = [Blob(120, 110, 160, 80, 9000), Blob(50, -400, 600, 700, 200000)]
        self.assertEqual(gather_box(predicted, blobs, FusionConfig()), (120, 110, 280, 190))

    def test_nothing_near_prediction(self) -> None:
        predicted = np.array([100.0, 100.0, 300.0, 200.0])
        self.assertIsNone(gather_box(predicted, [Blob(900, 600, 50, 50, 2500)], FusionConfig()))


class FusionTest(unittest.TestCase):
    def test_starts_from_blobs_and_tracks_moving_box(self) -> None:
        rng = np.random.default_rng(3)
        dt = 1.0 / 30.0
        state = np.array([-8.0, 2.0, 0.0, -0.2, 0.0, 7.0, 0.1])
        truth = []
        fusion = VehicleFusion(MODEL, FusionConfig(init_min_area_px=50.0, range_m=(1.0, 500.0)))
        for frame in range(120):
            truth.append(anchor_point(state, MODEL))
            views = {pose.camera_id: CameraView(pose, blobs_for(state, pose, rng)) for pose in RIG}
            fusion.step(frame * dt, views, key=frame)
            state = ctrv_step(state, dt, yaw_tau_s=0.0)
        smoothed = fusion.smoothed_states()
        self.assertGreaterEqual(len(smoothed), 115)
        frames = sorted(smoothed)
        est = np.stack([anchor_point(smoothed[f], MODEL) for f in frames])
        errors = frame_errors(frames, est, np.stack(truth))
        self.assertLess(float(np.median(errors[30:])), 0.2)


class MetricsByFrameTest(unittest.TestCase):
    def test_estimate_is_compared_with_its_own_frame(self) -> None:
        truth = np.stack([np.array([float(i), 0.0, 0.0]) for i in range(100)])
        frames = [40, 41, 42]
        est = truth[frames] + np.array([0.0, 0.3, 0.4])
        stats = compare_by_frame(frames, est, truth)
        self.assertAlmostEqual(stats["rmse_m"], 0.5)
        self.assertEqual(stats["frames"], 3)
        self.assertAlmostEqual(stats["coverage"], 0.03)

    def test_frames_outside_table_are_ignored(self) -> None:
        truth = np.zeros((10, 3))
        stats = compare_by_frame([5, 50], np.zeros((2, 3)), truth)
        self.assertEqual(stats["frames"], 1)
        self.assertTrue(np.isnan(frame_errors([50], np.zeros((1, 3)), truth)[0]))


class Way1FilesTest(unittest.TestCase):
    def test_files_found_in_subfolders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "видео").mkdir()
            (root / "execl").mkdir()
            for number in range(1, 6):
                (root / "видео" / f"Way1.FinalImage.CineCameraActor{number}.mp4").write_bytes(b"")
            (root / "execl" / "object_transform_SM_AA_track.csv").write_text("Frame;Loc_X;Loc_Y;Loc_Z\n")
            self.assertEqual(find_file(root, "object_transform_SM_AA_track.csv").parent.name, "execl")
            paths = video_paths(root)
            self.assertEqual(sorted(paths), [f"cam-{n}" for n in range(1, 6)])
            self.assertEqual(paths["cam-3"].parent.name, "видео")

    def test_missing_video_names_the_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(KeyError) as caught:
                video_paths(Path(tmp))
            self.assertIn("CineCameraActor1", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
