import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.export_result import (
    export_session_sample,
    write_control_video,
    write_xyz_charts,
    write_xyz_csv,
)
from app.profile import load_profile
from app.session import Session
from app.source import SyntheticSource


class ExportResultTest(unittest.TestCase):
    def test_synthetic_writes_files(self) -> None:
        frames = [
            np.full((48, 64, 3), (20 * i, 40, 60), np.uint8) for i in range(5)
        ]
        times = [0.0, 0.04, 0.08, 0.12, 0.16]
        xyz = [np.array([i * 0.1, 0.0, 1.0 + i * 0.05]) for i in range(5)]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = write_control_video(frames, root / "control_overlay.mp4", fps=25.0)
            self.assertTrue(video.is_file())
            self.assertGreater(video.stat().st_size, 0)

            csv_path = write_xyz_csv(times, xyz, root / "xyz.csv")
            self.assertTrue(csv_path.is_file())
            text = csv_path.read_text(encoding="utf-8")
            self.assertIn("t,x,y,z", text.replace(" ", ""))
            self.assertEqual(text.strip().count("\n"), 5)

            charts = write_xyz_charts(times, xyz, root / "xyz")
            if charts:
                for chart in charts:
                    self.assertTrue(chart.is_file())
            else:
                fallback = root / "xyz.csv"
                self.assertTrue(fallback.is_file())

    def test_export_session_sample_demo(self) -> None:
        profile = load_profile(ROOT / "config.yaml")
        session = Session(profile, SyntheticSource())
        with tempfile.TemporaryDirectory() as tmp:
            info = export_session_sample(session, Path(tmp), max_frames=30)
            self.assertGreater(info.get("frames", 0), 0)
            video = info.get("video")
            self.assertIsNotNone(video)
            assert video is not None
            self.assertTrue(Path(video).is_file())


if __name__ == "__main__":
    unittest.main()
