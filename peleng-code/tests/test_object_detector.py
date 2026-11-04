import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.motion import MotionDetector
from app.object_detector import (
    COLOR_TABLE,
    Box,
    DetectorError,
    HsvRange,
    ObjectDetector,
    build_detector,
    color_share,
    parse_spec,
    region_rows,
    translate_query,
)
from app.profile import TargetProfile


def _profile(s_min: float = 0.0, s_max: float = 1e9) -> TargetProfile:
    return TargetProfile(
        name="test",
        s_min=s_min,
        s_max=s_max,
        u_min=0.0,
        u_max=1e9,
        r_min=0.0,
        r_max=1e9,
        v_min=0.0,
        v_max=1e9,
        d_m=None,
        predict_s=0.5,
        color=None,
    )


class FakeModel:
    """Заглушка модели: рамки в долях поданного кадра, запоминает размеры входа."""

    device = "cpu"

    def __init__(self, boxes=None):
        # (доля x1, доля y1, доля x2, доля y2, уверенность, имя)
        self.boxes = boxes if boxes is not None else [(0.25, 0.25, 0.5, 0.5, 0.9, "person")]
        self.shapes = []

    def predict(self, image, conf, classes):
        height, width = image.shape[:2]
        self.shapes.append((width, height))
        return [
            Box(x1 * width, y1 * height, x2 * width, y2 * height, score, name)
            for x1, y1, x2, y2, score, name in self.boxes
        ]


def _bgr(h: int, s: int, v: int) -> tuple[int, int, int]:
    pixel = np.array([[[h, s, v]]], np.uint8)
    b, g, r = cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


class ColorTest(unittest.TestCase):
    def test_share_counts_only_the_region_rows(self):
        image = np.full((100, 100, 3), 200, np.uint8)
        image[0:50] = 10  # верх чёрный, низ светлый
        black = COLOR_TABLE["black"]
        self.assertGreater(color_share(image, (0, 0, 100, 100), "upper", black), 0.85)
        self.assertEqual(color_share(image, (0, 0, 100, 100), "lower", black), 0.0)
        self.assertAlmostEqual(color_share(image, (0, 0, 100, 100), "full", black), 0.5, places=2)

    def test_red_uses_both_hue_ranges(self):
        image = np.zeros((10, 20, 3), np.uint8)
        image[:, :10] = _bgr(5, 200, 200)
        image[:, 10:] = _bgr(175, 200, 200)
        red = COLOR_TABLE["red"]
        self.assertAlmostEqual(color_share(image, (0, 0, 20, 10), "full", red), 1.0, places=2)
        self.assertAlmostEqual(color_share(image, (0, 0, 20, 10), "full", red[:1]), 0.5, places=2)

    def test_box_outside_image_gives_zero(self):
        image = np.zeros((10, 10, 3), np.uint8)
        self.assertEqual(color_share(image, (50, 50, 10, 10), "full", COLOR_TABLE["black"]), 0.0)

    def test_region_rows(self):
        self.assertEqual(region_rows(0, 100, "full"), (0, 100))
        self.assertEqual(region_rows(0, 100, "upper"), (15, 55))
        self.assertEqual(region_rows(0, 100, "middle"), (30, 70))
        self.assertEqual(region_rows(10, 100, "lower"), (65, 105))

    def test_color_share_threshold_drops_box(self):
        image = np.full((200, 200, 3), 220, np.uint8)  # светлая одежда
        raw = {"mode": "object", "class": "person", "region": "upper", "color_share": 0.35,
               "color": {"h_min": 0, "h_max": 180, "v_max": 60}}
        detector = ObjectDetector(_profile(), parse_spec(raw), model=FakeModel())
        self.assertEqual(detector.detect(image), [])
        image[:] = 20  # тёмная
        detector = ObjectDetector(_profile(), parse_spec(raw), model=FakeModel())
        self.assertEqual(len(detector.detect(image)), 1)


class SpecTest(unittest.TestCase):
    def test_object_defaults(self):
        spec = parse_spec({"mode": "object", "class": "car"})
        self.assertEqual(spec.classes, ["car"])
        self.assertEqual(spec.conf, 0.4)
        self.assertEqual(spec.region, "full")
        self.assertEqual(spec.colors, [])
        self.assertEqual(spec.model, "yolov8n.pt")
        self.assertEqual(spec.det_width, 640)
        self.assertEqual(spec.det_every, 3)
        self.assertEqual(spec.roi_scale, 2.5)

    def test_text_defaults(self):
        spec = parse_spec({"mode": "text", "query": ["truck", "cargo truck"]})
        self.assertEqual(spec.query, ["truck", "cargo truck"])
        self.assertEqual(spec.conf, 0.25)
        self.assertEqual(spec.det_every, 5)
        self.assertEqual(spec.model, "yolov8s-worldv2.pt")

    def test_color_list_of_ranges(self):
        spec = parse_spec({"mode": "object", "class": "car", "color": [
            {"h_min": 0, "h_max": 10, "s_min": 120, "v_min": 70},
            {"h_min": 170, "h_max": 180, "s_min": 120, "v_min": 70},
        ]})
        self.assertEqual(spec.colors[1], HsvRange(170, 180, 120, 255, 70, 255))

    def test_required_fields(self):
        with self.assertRaises(DetectorError):
            parse_spec({"mode": "object"})
        with self.assertRaises(DetectorError):
            parse_spec({"mode": "text"})
        with self.assertRaises(DetectorError):
            parse_spec({"mode": "object", "class": "car", "region": "head"})

    def test_unknown_class_names_similar(self):
        with self.assertRaises(DetectorError) as caught:
            parse_spec({"mode": "object", "class": "persn"})
        self.assertIn("person", str(caught.exception))


class QueryTest(unittest.TestCase):
    def test_hoodie(self):
        found = translate_query("человек в чёрной кофте")
        self.assertEqual(found.text, "person in black hoodie")
        self.assertEqual(found.color, "black")
        self.assertEqual(found.region, "upper")

    def test_red_car(self):
        found = translate_query("Красная машина")
        self.assertEqual(found.text, "red car")
        self.assertEqual(found.color, "red")
        self.assertEqual(found.region, "full")

    def test_unknown_word_kept(self):
        self.assertEqual(translate_query("синий дрон").text, "blue дрон")

    def test_russian_query_turns_on_color(self):
        spec = parse_spec({"mode": "text", "query": "красная машина"})
        self.assertEqual(spec.query, ["red car"])
        self.assertEqual(spec.colors, COLOR_TABLE["red"])
        self.assertEqual(spec.region, "full")

    def test_profile_color_wins_over_query(self):
        spec = parse_spec({"mode": "text", "query": "красная машина",
                           "color": {"h_min": 100, "h_max": 130}})
        self.assertEqual(spec.colors, [HsvRange(100, 130)])


class CoordinatesTest(unittest.TestCase):
    def test_small_frame_maps_back_to_full(self):
        model = FakeModel()
        detector = ObjectDetector(_profile(), parse_spec({"mode": "object", "class": "person"}), model=model)
        found = detector.detect(np.zeros((720, 1280, 3), np.uint8))
        self.assertEqual(model.shapes, [(640, 360)])
        self.assertEqual(found[0].bbox, (320, 180, 320, 180))
        self.assertAlmostEqual(found[0].cx, 480.0)
        self.assertAlmostEqual(found[0].cy, 270.0)
        self.assertEqual(detector.labels[0].name, "person")

    def test_roi_maps_back_with_offset(self):
        model = FakeModel([(0.0, 0.0, 0.5, 0.5, 0.9, "person")])
        detector = ObjectDetector(_profile(), parse_spec({"mode": "object", "class": "person"}), model=model)
        detector.set_hint((600.0, 400.0, 100.0, 80.0))
        found = detector.detect(np.zeros((720, 1280, 3), np.uint8))
        # Окно 250×200 вокруг (600, 400), на полном разрешении.
        self.assertEqual(model.shapes, [(250, 200)])
        self.assertEqual(found[0].bbox, (475, 300, 125, 100))

    def test_roi_miss_falls_back_to_full_frame(self):
        model = FakeModel([])
        spec = parse_spec({"mode": "object", "class": "person", "det_every": 1})
        detector = ObjectDetector(_profile(), spec, model=model)
        detector.set_hint((600.0, 400.0, 100.0, 80.0))
        image = np.zeros((720, 1280, 3), np.uint8)
        detector.detect(image)
        detector.detect(image)
        self.assertEqual(model.shapes, [(250, 200), (640, 360)])

    def test_area_gate_and_order(self):
        model = FakeModel([
            (0.0, 0.0, 0.1, 0.1, 0.5, "person"),
            (0.5, 0.5, 0.9, 0.9, 0.5, "person"),
            (0.2, 0.2, 0.25, 0.25, 0.95, "person"),
            (0.2, 0.2, 0.6, 0.6, 0.99, "car"),
        ])
        spec = parse_spec({"mode": "object", "class": "person"})
        detector = ObjectDetector(_profile(s_min=10.0), spec, model=model)
        found = detector.detect(np.zeros((100, 100, 3), np.uint8))
        # Машина не тот класс; при равной уверенности крупнейший раньше.
        self.assertEqual([item.bbox for item in found], [(20, 20, 5, 5), (50, 50, 40, 40), (0, 0, 10, 10)])


class MergeTest(unittest.TestCase):
    def test_cab_and_trailer_become_one_box(self):
        # Кабина и прицеп рядом, третья машина далеко — она остаётся отдельной.
        model = FakeModel([
            (0.10, 0.40, 0.25, 0.60, 0.8, "truck"),
            (0.26, 0.38, 0.60, 0.62, 0.6, "truck"),
            (0.85, 0.10, 0.95, 0.20, 0.7, "truck"),
        ])
        spec = parse_spec({"mode": "object", "class": "truck", "merge": True})
        detector = ObjectDetector(_profile(), spec, model=model)
        found = detector.detect(np.zeros((100, 100, 3), np.uint8))
        self.assertEqual([item.bbox for item in found], [(10, 38, 50, 24), (85, 10, 10, 10)])
        self.assertAlmostEqual(detector.labels[0].conf, 0.8)

    def test_without_merge_boxes_stay_apart(self):
        model = FakeModel([
            (0.10, 0.40, 0.25, 0.60, 0.8, "truck"),
            (0.26, 0.38, 0.60, 0.62, 0.6, "truck"),
        ])
        spec = parse_spec({"mode": "object", "class": "truck"})
        detector = ObjectDetector(_profile(), spec, model=model)
        self.assertEqual(len(detector.detect(np.zeros((100, 100, 3), np.uint8))), 2)


class ScheduleTest(unittest.TestCase):
    def test_model_runs_every_n_frames(self):
        model = FakeModel()
        detector = ObjectDetector(_profile(), parse_spec({"mode": "object", "class": "person"}), model=model)
        image = np.zeros((100, 100, 3), np.uint8)
        results = [detector.detect(image) for _ in range(7)]
        self.assertEqual(detector.calls, 3)
        self.assertEqual([bool(item) for item in results], [True, False, False, True, False, False, True])


class FallbackTest(unittest.TestCase):
    def test_without_ultralytics_falls_back_to_motion(self):
        with mock.patch.dict(sys.modules, {"ultralytics": None}):
            detector, error = build_detector(_profile(), {"mode": "object", "class": "person"})
        self.assertIsInstance(detector, MotionDetector)
        self.assertIn("нет ultralytics", error)

    def test_without_weights_falls_back_to_motion(self):
        fake = types.ModuleType("ultralytics")
        raw = {"mode": "object", "class": "person", "model": "нет_такого_файла.pt"}
        with mock.patch.dict(sys.modules, {"ultralytics": fake}):
            detector, error = build_detector(_profile(), raw)
        self.assertIsInstance(detector, MotionDetector)
        self.assertIn("нет весов нет_такого_файла.pt", error)

    def test_bad_profile_falls_back_to_motion(self):
        detector, error = build_detector(_profile(), {"mode": "object", "class": "persn"}, model=FakeModel())
        self.assertIsInstance(detector, MotionDetector)
        self.assertIn("person", error)

    def test_old_profiles_keep_motion_detector(self):
        detector, error = build_detector(_profile(), None)
        self.assertIsInstance(detector, MotionDetector)
        self.assertIsNone(error)
        detector, error = build_detector(_profile(), {"mode": "sky_dark"})
        self.assertIsInstance(detector, MotionDetector)
        self.assertIsNone(error)

    def test_model_given_builds_object_detector(self):
        detector, error = build_detector(_profile(), {"mode": "object", "class": "car"}, model=FakeModel())
        self.assertIsInstance(detector, ObjectDetector)
        self.assertIsNone(error)


class ProfileTest(unittest.TestCase):
    def setUp(self):
        import shutil
        import tempfile

        self.folder = Path(tempfile.mkdtemp())
        self.config = self.folder / "config.yaml"
        shutil.copy(ROOT / "config.yaml", self.config)
        self.addCleanup(shutil.rmtree, self.folder)

    def test_described_profiles_load_with_default_gates(self):
        from app.profile import load_patterns

        patterns = load_patterns(self.config)
        red = patterns["red_car"]
        self.assertEqual(red.mode, "object")
        self.assertIsNone(red.color)
        self.assertEqual(red.detect["class"], "car")
        self.assertEqual(red.s_min, 0.0)
        self.assertEqual(len(parse_spec(red.detect).colors), 2)
        self.assertIsNone(patterns["motion"].detect)

    def test_save_active_keeps_comments_and_profiles(self):
        from app.profile import load_document, save_active

        before = self.config.read_text(encoding="utf-8")
        save_active(self.config, "red_car", 1.0)
        after = self.config.read_text(encoding="utf-8")
        self.assertIn("# Цель по описанию", after)
        self.assertEqual(before.count("\n"), after.count("\n"))
        data = load_document(self.config)
        self.assertEqual(data["active"], "red_car")
        self.assertEqual(data["predict_s"], 1.0)
        self.assertEqual(data["patterns"]["red_car"]["class"], "car")

    def test_save_custom_keeps_mode_and_detector_fields(self):
        from app.profile import load_document, load_patterns, save_custom

        patterns = load_patterns(self.config)
        save_custom(self.config, patterns["sky_uav"], 0.5)
        save_custom(self.config, patterns["red_car"], 0.5)
        saved = load_document(self.config)["patterns"]
        self.assertEqual(saved["sky_uav"]["mode"], "sky_dark")
        self.assertEqual(saved["red_car"]["mode"], "object")
        self.assertEqual(len(saved["red_car"]["color"]), 2)

    def test_saving_keeps_crlf(self):
        from app.profile import save_active, save_pattern

        self.config.write_bytes(self.config.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
        lines = self.config.read_bytes().count(b"\r\n")
        save_active(self.config, "red_car", 0.5)
        data = self.config.read_bytes()
        self.assertEqual(data.count(b"\r\n"), lines)
        self.assertEqual(data.count(b"\n"), lines)
        save_pattern(self.config, "text_custom", {"mode": "text", "query": "car"}, 0.5)
        data = self.config.read_bytes()
        self.assertEqual(data.count(b"\n"), data.count(b"\r\n"))

    def test_save_pattern_makes_text_profile(self):
        from app.profile import load_patterns, save_pattern

        save_pattern(self.config, "text_custom", {"mode": "text", "query": "красная машина"}, 0.5)
        profile = load_patterns(self.config)["text_custom"]
        self.assertEqual(parse_spec(profile.detect).query, ["red car"])


class _Frames:
    def __init__(self, count: int = 4):
        from app.source import Frame

        image = np.zeros((360, 640, 3), np.uint8)
        self.frames = [Frame(image, index / 30.0, "cam-1", index) for index in range(count)]

    def read(self):
        return self.frames.pop(0) if self.frames else None


class PipelineTest(unittest.TestCase):
    def _profile(self, name: str):
        from app.profile import load_patterns

        return load_patterns(ROOT / "config.yaml")[name]

    def test_pipeline_uses_described_detector_and_labels(self):
        from app.pipeline import Pipeline

        model = FakeModel([(0.4, 0.3, 0.6, 0.7, 0.9, "car")])
        profile = self._profile("red_car")
        profile.detect = dict(profile.detect, color=None, det_every=1)
        from app.track2d import CONFIRM_HITS

        with mock.patch("app.object_detector.load_model", return_value=model):
            pipeline = Pipeline(profile, _Frames(CONFIRM_HITS + 1))
            steps = [pipeline.step() for _ in range(CONFIRM_HITS + 1)]
        self.assertIsInstance(pipeline.detector, ObjectDetector)
        self.assertIsNone(pipeline.detector_error)
        self.assertEqual(steps[0].labels[0].name, "car")
        self.assertIsNotNone(steps[0].detection)
        # Пока трек не подтверждён — весь кадр; потом поиск в окне вокруг трека.
        self.assertEqual(model.shapes[0], (640, 360))
        self.assertTrue(steps[CONFIRM_HITS - 1].track.active)
        self.assertLess(model.shapes[CONFIRM_HITS][0], 640)
        self.assertIsNotNone(steps[CONFIRM_HITS].detection)

    def test_pipeline_falls_back_and_reports_error(self):
        from app.pipeline import Pipeline

        with mock.patch.dict(sys.modules, {"ultralytics": None}):
            pipeline = Pipeline(self._profile("red_car"), _Frames())
        self.assertIsInstance(pipeline.detector, MotionDetector)
        self.assertIn("нет ultralytics", pipeline.detector_error)

    def test_profile_switch_rebuilds_detector(self):
        from app.pipeline import Pipeline

        pipeline = Pipeline(self._profile("motion"), _Frames())
        motion = pipeline.detector
        pipeline.set_profile(self._profile("black_shirt"))
        self.assertIs(pipeline.detector, motion)
        with mock.patch("app.object_detector.load_model", return_value=FakeModel()):
            pipeline.set_profile(self._profile("black_hoodie_person"))
        self.assertIsInstance(pipeline.detector, ObjectDetector)
        pipeline.set_profile(self._profile("motion"))
        self.assertIsInstance(pipeline.detector, MotionDetector)


if __name__ == "__main__":
    unittest.main()
