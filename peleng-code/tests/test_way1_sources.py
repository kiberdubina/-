import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.way1_sources import open_sources, video_info, video_paths

_IDS = [f"cam-{number}" for number in range(1, 6)]


class Way1SourcesTest(unittest.TestCase):
    def test_five_video_paths_exist(self):
        paths = video_paths(ROOT)
        self.assertEqual(list(paths), _IDS)
        for path in paths.values():
            self.assertTrue(path.is_file())

    def test_video_info_same_size_and_long_enough(self):
        info = video_info(ROOT)
        self.assertEqual(list(info), _IDS)
        widths = {row["width"] for row in info.values()}
        heights = {row["height"] for row in info.values()}
        self.assertEqual(len(widths), 1)
        self.assertEqual(len(heights), 1)
        self.assertGreater(widths.pop(), 0)
        self.assertGreater(heights.pop(), 0)
        for row in info.values():
            self.assertIsInstance(row["width"], int)
            self.assertIsInstance(row["height"], int)
            self.assertIsInstance(row["fps"], float)
            self.assertIsInstance(row["frames"], int)
            self.assertGreaterEqual(row["frames"], 1000)
            self.assertGreater(row["fps"], 1)

    def test_open_sources_keeps_camera_ids(self):
        sources = open_sources(ROOT)
        try:
            self.assertEqual(len(sources), 5)
            for camera_id, source in zip(_IDS, sources):
                self.assertEqual(source.camera_id, camera_id)
                self.assertTrue(source.is_open())
        finally:
            for source in sources:
                source.close()


if __name__ == "__main__":
    unittest.main()
