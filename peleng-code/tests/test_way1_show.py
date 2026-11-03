import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class Way1ShowTest(unittest.TestCase):
    def test_file_menu_has_five_camera_path(self):
        text = (ROOT / "app" / "main.py").read_text(encoding="utf-8")
        self.assertIn("Путь 5 камер", text)
        self.assertIn("def use_way1", text)

    def test_load_way1_when_neighbor_modules_exist(self):
        try:
            import app.cine_table  # noqa: F401
            import app.way1_sources  # noqa: F401
        except ImportError as error:
            self.skipTest(f"соседние модули ещё не импортируются: {error}")

        from app.way1_show import load_way1

        sources: list = []
        try:
            sources, poses = load_way1(ROOT)
            self.assertEqual(len(sources), 5)
            self.assertEqual(len(poses), 5)
        finally:
            for source in sources:
                close = getattr(source, "close", None)
                if close is not None:
                    close()

    def test_track_grows_only_up_to_the_current_frame(self):
        from app.way1_show import points_until

        track = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            np.float64,
        )
        first, shown, restart = points_until(track, -1, 0)
        self.assertEqual(len(first), 1)
        self.assertEqual(shown, 0)
        self.assertFalse(restart)
        self.assertTrue(np.allclose(first[0], [0.0, 0.0, 0.0]))
        more, shown, restart = points_until(track, shown, 2)
        self.assertEqual(len(more), 2)
        self.assertEqual(shown, 2)
        self.assertFalse(restart)
        self.assertTrue(np.allclose(more[-1], [2.0, 0.0, 0.0]))
        ahead, shown, restart = points_until(track, shown, 2)
        self.assertEqual(ahead, [])
        self.assertEqual(shown, 2)
        self.assertFalse(restart)
        again, shown, restart = points_until(track, shown, 0)
        self.assertTrue(restart)
        self.assertEqual(shown, 0)
        self.assertEqual(len(again), 1)


if __name__ == "__main__":
    unittest.main()
