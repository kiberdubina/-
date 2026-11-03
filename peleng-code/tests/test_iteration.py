import sys
import threading
import time
import tkinter as tk
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.calib import BoardDetector, pose_from_view
from app.geom import triangulate
from app.motion import Detection
from app.pipeline import Pipeline
from app.profile import load_board, load_patterns, load_profile
from app.rig import Rig
from app.session import Session
from app.phone_usb import (
    _KEY,
    _PROFILES,
    _SESSION,
    FILE_TRANSFER_REQUIRED,
    KEEP_LAST_S,
    STALL_S,
    VIDEO_SOURCE,
    PhonePreviewSource,
    capture_pts_us,
    device_listed,
    median_latency_ms,
    slower_profile,
    suspend_seconds,
    usb_link_mode,
)
from app.source import SyntheticSource, VideoSource
from app.track2d import Track2D
from app.view3d import SceneCanvas, fit_distance, project_points, scene_radius

VIDEO = ROOT / "video_2026-11-01_22-06-07.mp4"


class IterationTest(unittest.TestCase):
    def test_demo_circle_has_box_area_and_speed(self):
        profile = load_profile(ROOT / "config.yaml")
        pipeline = Pipeline(profile, SyntheticSource())
        areas = []
        speeds = []
        for _ in range(40):
            step = pipeline.step()
            self.assertIsNotNone(step)
            if step.detection is not None:
                self.assertGreater(step.detection.area, profile.s_min)
                areas.append(step.detection.area)
            if step.track.active:
                speeds.append(step.track.speed_px_s)
        self.assertGreaterEqual(len(areas), 30)
        self.assertGreater(max(speeds), profile.u_min)
        self.assertIn(pipeline.tracker.update(None, 0).state, {"поиск", "сопровождение", "вне порога скорости", "захват"})

    def test_outlier_does_not_enter_the_trail(self):
        profile = load_profile(ROOT / "config.yaml")
        tracker = Track2D(profile)
        for index in range(15):
            tracker.update(Detection(100 + index * 8, 200, 1000, (90, 190, 20, 20)), index * 0.04)
        points = []
        before = tracker.update(Detection(100 + 15 * 8, 200, 1000, (90, 190, 20, 20)), 15 * 0.04)
        points.append((before.cx, before.cy))
        jumped = tracker.update(Detection(before.cx, before.cy + 400, 1000, (90, 500, 20, 20)), 16 * 0.04)
        points.append((jumped.cx, jumped.cy))
        after = tracker.update(Detection(before.cx + 8, before.cy, 1000, (90, 190, 20, 20)), 17 * 0.04)
        points.append((after.cx, after.cy))
        self.assertLess(abs(jumped.cy - before.cy), 140)
        jumps = [
            np.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1])
            for i in range(1, len(points))
        ]
        self.assertLess(max(jumps), 140)

    def test_ball_video_follows_green_without_a_spike(self):
        profile = load_profile(ROOT / "config.yaml")
        pipeline = Pipeline(profile, VideoSource(str(VIDEO), "file"))
        errors = []
        jumps = []
        previous = None
        matched = 0
        while True:
            step = pipeline.step()
            if step is None or step.frame.time_s < 0.01 and matched > 10:
                break
            green = _green_center(step.frame.image)
            if step.track.active and previous is not None:
                jumps.append(
                    np.hypot(step.track.cx - previous[0], step.track.cy - previous[1])
                )
            if step.track.active:
                previous = (step.track.cx, step.track.cy)
            else:
                previous = None
            if green is not None and step.detection is not None:
                matched += 1
                errors.append(np.hypot(step.detection.cx - green[0], step.detection.cy - green[1]))
            if step.frame.time_s > 7.2:
                break
        self.assertGreaterEqual(matched, 50)
        self.assertLess(float(np.median(errors)), 40)
        self.assertLess(max(jumps), 140)

    def test_two_cameras_recover_meters(self):
        profile = load_patterns(ROOT / "config.yaml")["green_ball"]
        rig = Rig()
        session = Session(profile, SyntheticSource())
        session.set_rig(rig)
        errors = []
        direct = []
        last = None
        for index in range(40):
            tick = session.step()
            self.assertIsNotNone(tick)
            last = tick
            self.assertEqual(len(tick.steps), 2)
            self.assertTrue(all(step.track.active for step in tick.steps))
            ids = {step.frame.camera_id for step in tick.steps}
            self.assertEqual(ids, {"A", "B"})
            self.assertEqual(set(session.poses), {"A", "B"})
            if tick.world.xyz is not None:
                errors.append(float(np.linalg.norm(tick.world.xyz - rig.truth[-1])))
                self.assertIsNotNone(tick.world.pred)
                self.assertEqual(len(tick.rays), 2)
            if index % 10 == 0:
                truth = rig.point_at(tick.steps[0].frame.time_s)
                poses = list(rig.poses.values())
                pixels = []
                for pose in poses:
                    projected, _ = cv2.projectPoints(
                        truth.reshape(1, 1, 3), pose.rvec, pose.tvec, pose.K, pose.dist
                    )
                    pixels.append((float(projected[0, 0, 0]), float(projected[0, 0, 1])))
                recovered = triangulate(poses, pixels)
                self.assertIsNotNone(recovered)
                direct.append(float(np.linalg.norm(recovered - truth)))
        self.assertGreaterEqual(len(errors), 20)
        self.assertLess(float(np.median(errors)), 0.02)
        self.assertLess(float(np.max(direct)), 0.01)
        root = tk.Tk()
        try:
            canvas = SceneCanvas(root)
            canvas.set_scene(
                last.trail,
                last.world.xyz,
                last.world.pred,
                list(session.poses.values()),
                "демо",
                last.rays,
            )
            self.assertGreaterEqual(len(canvas.find_withtag("camera")), 8)
            self.assertGreaterEqual(len(canvas.find_withtag("ray")), 2)
            self.assertTrue(canvas.find_withtag("trail"))
            self.assertTrue(canvas.find_withtag("forecast"))
            self.assertGreater(_item_span(canvas, "camera"), 40.0)
            self.assertGreater(_item_span(canvas, "ray"), 30.0)
            labels = {
                canvas.itemcget(item, "text")
                for item in canvas.find_withtag("camera")
                if canvas.type(item) == "text"
            }
            self.assertEqual(labels, {"A", "B"})
        finally:
            root.destroy()

    def test_charuco_pose_is_metric(self):
        spec = load_board(ROOT / "config.yaml")
        detector = BoardDetector(spec)
        board = detector.image()
        width, height = 960, 540
        focal = 900.0
        matrix = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], np.float64)
        dist = np.zeros(5)
        square_w = spec.squares_x * spec.square_m
        square_h = spec.squares_y * spec.square_m
        rvec, _ = cv2.Rodrigues(np.eye(3))
        tvec = np.array([[-square_w / 2], [-square_h / 2], [0.55]], np.float64)
        image = _render_board(board, rvec, tvec, matrix, dist, width, height, square_w, square_h)
        view = detector.detect(image)
        self.assertIsNotNone(view)
        pose = pose_from_view("A", view, matrix, dist)
        self.assertIsNotNone(pose)
        self.assertLess(float(np.linalg.norm(pose.tvec.ravel() - tvec.ravel())), 0.02)

    def test_moving_camera_keeps_one_small_target(self):
        profile = load_patterns(ROOT / "config.yaml")["aircraft"]
        pipeline = Pipeline(profile, VideoSource(str(ROOT / "IMG_0985.MOV"), "file"))
        hits = 0
        jumps = []
        previous = None
        frames = 0
        while frames < 931:
            step = pipeline.step()
            if step is None or (frames > 20 and step.frame.time_s < 0.02):
                break
            frames += 1
            if step.detection is None:
                previous = None
                continue
            hits += 1
            if previous is not None:
                jumps.append(
                    np.hypot(step.detection.cx - previous[0], step.detection.cy - previous[1])
                )
            previous = (step.detection.cx, step.detection.cy)
        self.assertGreater(hits, 350)
        self.assertLess(float(np.median(jumps)), 45)
        self.assertLess(float(np.percentile(jumps, 95)), 120)

    def test_scene_zoom_changes_scale(self):
        points = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])
        target = np.zeros(3)
        near = project_points(points, 0.4, 0.5, 1.0, target, 520, 520)
        far = project_points(points, 0.4, 0.5, 2.0, target, 520, 520)
        self.assertGreater(abs(far[0, 0] - far[1, 0]), abs(near[0, 0] - near[1, 0]) * 1.5)

    def test_orbit_moves_a_point_and_keeps_the_trail_in_frame(self):
        times = np.linspace(0, 6.0, 50)
        trail = np.array(
            [[0.12 * np.sin(t * 1.3), 0.08 * np.cos(t * 1.1), 0.05] for t in times]
        )
        current = trail[-1]
        pred = current + (trail[-1] - trail[-4]) * 2.0
        target = trail.mean(axis=0)
        cloud = np.vstack([trail, current, pred, np.zeros(3)])
        radius = scene_radius(cloud, target)
        distance = fit_distance(radius, 520, 520)
        home = project_points(
            np.vstack([trail, current, pred]), 0.7, 0.4, 1.0, target, 520, 520, distance
        )
        turned = project_points(
            np.vstack([trail, current, pred]), 1.6, 0.4, 1.0, target, 520, 520, distance
        )
        self.assertGreater(float(np.linalg.norm(home[0, :2] - turned[0, :2])), 20.0)
        for projected in (home, turned):
            self.assertTrue(np.all(projected[:, 2] > 0.05))
            self.assertGreater(float(np.min(projected[:, 0])), 8.0)
            self.assertLess(float(np.max(projected[:, 0])), 512.0)
            self.assertGreater(float(np.min(projected[:, 1])), 8.0)
            self.assertLess(float(np.max(projected[:, 1])), 512.0)
        span = np.max(home[:-1, :2], axis=0) - np.min(home[:-1, :2], axis=0)
        self.assertGreater(float(span.sum()), 80.0)
        self.assertGreater(float(np.linalg.norm(home[-1, :2] - home[-2, :2])), 8.0)

    def test_canvas_draws_orbit_zoom_trail_and_forecast(self):
        root = tk.Tk()
        try:
            canvas = SceneCanvas(root)
            trail = [
                np.array([0.0, 0.0, 0.05]),
                np.array([0.08, 0.02, 0.05]),
                np.array([0.16, 0.0, 0.07]),
            ]
            current = trail[-1]
            pred = current + np.array([0.07, 0.01, 0.02])
            canvas.set_scene(trail, current, pred, [], "проверка")
            self.assertTrue(canvas.find_withtag("axis"))
            self.assertTrue(canvas.find_withtag("trail"))
            self.assertTrue(canvas.find_withtag("now"))
            self.assertEqual(len(canvas.find_withtag("forecast")), 2)
            coords = canvas.coords(canvas.find_withtag("trail")[0])
            xs, ys = coords[0::2], coords[1::2]
            self.assertGreater((max(xs) - min(xs)) + (max(ys) - min(ys)), 60.0)
            yaw, pitch, zoom = canvas.yaw, canvas.pitch, canvas.zoom
            canvas._press(_Event(x=40, y=40))
            canvas._drag_move(_Event(x=110, y=15))
            self.assertNotAlmostEqual(canvas.yaw, yaw)
            self.assertNotAlmostEqual(canvas.pitch, pitch)
            self.assertLess(canvas.pitch, 1.1)
            canvas._wheel(_Event(delta=120))
            self.assertGreater(canvas.zoom, zoom)
            canvas._wheel(_Event(num=5))
            self.assertLess(canvas.zoom, zoom * 1.1)
        finally:
            root.destroy()

    def test_phone_mjpeg_url_opens(self):
        holder = {}

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                for index in range(12):
                    image = np.zeros((240, 320, 3), np.uint8)
                    cv2.circle(image, (40 + index * 15, 120), 18, (40, 180, 40), -1)
                    ok, encoded = cv2.imencode(".jpg", image)
                    if not ok:
                        break
                    payload = encoded.tobytes()
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(payload)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()

            def log_message(self, _format, *_args):
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_address[1]}/video"
        source = VideoSource(url, "phone-1")
        try:
            self.assertTrue(source.is_open(), url)
            frame = source.read()
            holder["frame"] = frame
            self.assertIsNotNone(frame)
            self.assertEqual(frame.camera_id, "phone-1")
            self.assertEqual(frame.image.shape[0], 240)
        finally:
            source.close()
            server.shutdown()


class LatencyBudgetTest(unittest.TestCase):
    def test_capture_pts_ignores_flags(self):
        self.assertIsNone(capture_pts_us(_SESSION | 50))
        self.assertEqual(capture_pts_us(_KEY | 12345), 12345)
        self.assertEqual(capture_pts_us(80), 80)

    def test_suspend_is_uptime_minus_frame_clock(self):
        text = "84951.46 111748.91 --------- beginning of main  16281.901 28409 28409 I pelengclk: mark\n"
        self.assertAlmostEqual(suspend_seconds(text), 84951.46 - 16281.901, places=3)
        self.assertIsNone(suspend_seconds("нет метки"))

    def test_median_and_smaller_profile(self):
        self.assertEqual(median_latency_ms([10, 40, 90]), 40)
        self.assertEqual(_PROFILES[0][0], "640x360")
        self.assertGreater(int(_PROFILES[0][1]), int(_PROFILES[1][1]))
        self.assertIsNone(slower_profile(0, 50))
        self.assertEqual(slower_profile(0, 51), 1)
        self.assertIsNone(slower_profile(1, 80))

    def test_stale_hold_expires_without_showing_the_phone_screen(self):
        self.assertEqual(VIDEO_SOURCE, "camera")
        self.assertAlmostEqual(STALL_S, 1.0)
        self.assertLessEqual(KEEP_LAST_S, 1.5)
        self.assertFalse(device_listed("List of devices attached\n", "abc"))
        self.assertFalse(device_listed("abc\tunauthorized\n", "abc"))
        self.assertTrue(device_listed("List of devices attached\nabc\tdevice\n", "abc"))

        source = PhonePreviewSource.__new__(PhonePreviewSource)
        source._closed = False
        source._lock = threading.Lock()
        source._frame = np.zeros((8, 8, 3), np.uint8)
        source._seq = 1
        source._served = 0
        source._keep_last = False
        source._keep_since = None
        source._fresh_at = time.monotonic()
        source._capture_mono = None
        source.stale = False
        source.rewound = False
        source.index = 0
        source.camera_id = "phone-1"
        source._started = time.monotonic()
        source._presented_capture = None

        self.assertIsNotNone(source.read())
        self.assertFalse(source.stale)
        self.assertIsNone(source.read())
        self.assertFalse(source.stale)

        source._keep_last = True
        source._keep_since = time.monotonic()
        self.assertIsNotNone(source.read())
        self.assertFalse(source.stale)

        source._keep_since = time.monotonic() - (KEEP_LAST_S + 0.1)
        source._fresh_at = source._keep_since
        self.assertIsNone(source.read())
        self.assertTrue(source.stale)

        source._seq = 2
        source._fresh_at = time.monotonic()
        source._keep_last = False
        self.assertIsNotNone(source.read())
        self.assertFalse(source.stale)

    def test_usb_file_transfer_is_not_charging_only(self):
        self.assertEqual(usb_link_mode("mtp,adb"), "files")
        self.assertEqual(usb_link_mode("mtp"), "files")
        self.assertEqual(usb_link_mode("none"), "none")
        self.assertEqual(usb_link_mode("adb"), "none")
        self.assertEqual(usb_link_mode("charging"), "none")
        self.assertEqual(usb_link_mode("adb\nnone"), "none")
        self.assertEqual(usb_link_mode("none\nmtp"), "files")
        self.assertEqual(usb_link_mode("ptp,adb"), "other")
        self.assertEqual(usb_link_mode(""), "unknown")
        self.assertEqual(usb_link_mode("error: closed"), "unknown")
        self.assertEqual(usb_link_mode("adb.exe: no devices/emulators found"), "unknown")
        self.assertEqual(usb_link_mode("adb.exe: device offline\nmtp,adb"), "files")
        self.assertIn("Передача файлов", FILE_TRANSFER_REQUIRED)
        self.assertIn("Без передачи данных", FILE_TRANSFER_REQUIRED)


class _Event:
    def __init__(self, x: int = 0, y: int = 0, delta: int = 0, num: int | None = None):
        self.x = x
        self.y = y
        self.delta = delta
        self.num = num


def _item_span(canvas: tk.Canvas, tag: str) -> float:
    coords: list[float] = []
    for item in canvas.find_withtag(tag):
        coords.extend(canvas.coords(item))
    if len(coords) < 2:
        return 0.0
    xs = coords[0::2]
    ys = coords[1::2]
    return (max(xs) - min(xs)) + (max(ys) - min(ys))


def _green_center(image: np.ndarray):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (25, 80, 40), (100, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 400:
        return None
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]


def _render_board(board, rvec, tvec, matrix, dist, width, height, square_w, square_h):
    image = np.full((height, width, 3), 48, np.uint8)
    if board.ndim == 2:
        board = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)
    board_h, board_w = board.shape[:2]
    src = np.float32([[0, 0], [board_w - 1, 0], [board_w - 1, board_h - 1], [0, board_h - 1]])
    obj = np.float32([[0, 0, 0], [square_w, 0, 0], [square_w, square_h, 0], [0, square_h, 0]])
    projected, _ = cv2.projectPoints(obj, rvec, tvec, matrix, dist)
    transform = cv2.getPerspectiveTransform(src, projected.reshape(-1, 2).astype(np.float32))
    warped = cv2.warpPerspective(board, transform, (width, height))
    mask = cv2.warpPerspective(np.full(board.shape[:2], 255, np.uint8), transform, (width, height))
    image[mask > 10] = warped[mask > 10]
    return image


if __name__ == "__main__":
    unittest.main()
