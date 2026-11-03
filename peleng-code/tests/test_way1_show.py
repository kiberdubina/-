import sys
import unittest
from pathlib import Path

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

    def test_points_until_not_exported(self):
        import app.way1_show as way1_show

        self.assertFalse(hasattr(way1_show, "points_until"))


if __name__ == "__main__":
    unittest.main()
