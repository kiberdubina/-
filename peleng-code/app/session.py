from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from app.background import BackgroundConfig, Blob, RotatingBackground, StaticBackground, foreground_blobs
from app.calib import BoardDetector, intrinsics_from_views, pose_from_view
from app.geom import CameraPose, _reprojection_error, range_from_cameras, sight_rays, triangulate, velocity_in_camera
from app.motion import Detection
from app.pipeline import Pipeline, Step
from app.profile import TargetProfile
from app.rig import Rig, RigCamera
from app.track2d import TrackView
from app.track3d import FORECAST_POINTS, Track3D, WorldView
from app.vehicle import V, VehicleModel, anchor_point, box_corners, ctrv_step, project_points, state_axes
from app.vehicle_fusion import CameraView, FusionConfig, VehicleFusion

# Два кадра дальше этого зазора — разные положения цели. Такую пару не триангулируем.
PAIR_SYNC_S = 0.08
MAX_TRACKS3D = 4
# Живой режим машины: фон считается на кадре этой ширины (быстрее, чем в контрольном прогоне).
# Ширина кадра живого фона. Way1: 480 — RMSE 8,1 м, 640 — 6,3 м при 1,6× времени, 960 не лучше.
LIVE_WORK_WIDTH = 640
# Запас вокруг проекции прогнозного ящика, где живой фон не учится, px полного кадра.
HOLD_PAD_PX = 30
# След живого режима машины (то, что идёт в отчёт погрешности по кадрам).
TRAIL_WARMUP = 8          # первые точки после старта трека не пишутся: ящик ещё сходится
TRAIL_STEP_M = 2.0        # дальше за кадр от прошлой точки — выброс (фура Way1: до ~1 м/кадр)
# Точку в след пишем, когда цель видят хотя бы две камеры: по одной дальность не измерена.
# Трек и прогноз при этом держатся. Way1: RMSE 6,3 → 3,3 м при покрытии 90 → 73 %.
TRAIL_MIN_CAMERAS = 2
# Неопределённость позиции: σ фильтра (VehicleView.sigma_m) занижена — на Way1 ошибка в среднем
# в ~12 раз больше σ. Калиброванная σ = SIGMA_SCALE·σ; дальше TRAIL_SIGMA_M точка не пишется:
# геометрия доступных камер не задаёт положение (одна камера: калиброванная σ ≈ 5 м).
SIGMA_SCALE = 12.0
TRAIL_SIGMA_M = 4.0
TRAIL_REJECT_LIMIT = 5    # столько выбросов подряд — это не выброс, а исправление трека
TRAIL_COAST = 5           # столько кадров без измерения пишется прогноз фильтра


# Выбранная машина на другой камере: трек берётся при невязке не больше MATCH_PICK_PX,
# отпускается после MATCH_LOSE_FRAMES кадров подряд с невязкой больше MATCH_KEEP_PX.
MATCH_PICK_PX = 50.0
MATCH_KEEP_PX = 90.0
MATCH_LOSE_FRAMES = 8
# Есть свежая 3D-точка выбранной (не старше MATCH_REF_S): та же машина на камере — трек у её
# проекции, не дальше max(MATCH_NEAR_PX, 0,6 · размер рамки). Нет точки — по лучу, но только если
# лучший кандидат в MATCH_UNIQUE раз лучше второго: на перекрёстке вдоль луча стоит очередь машин.
MATCH_REF_S = 1.0
# Камера «едет», если её центр ушёл от первой позы дальше этого, м: фон по ней не строится.
TRANSLATE_M = 0.5
MATCH_NEAR_PX = 30.0
MATCH_UNIQUE = 2.0
# Двусмысленность по лучу копится по кадрам (машины едут по-разному, чужая уходит с луча);
# не разрешилась за MATCH_SETTLE кадров — берётся лучший по средней невязке.
MATCH_SETTLE = 8


def vehicle_forecast(state: np.ndarray, model: VehicleModel, motion, horizon_s: float) -> np.ndarray:
    """Прогноз траектории машины: FORECAST_POINTS точек опорной точки до горизонта по её модели
    движения (дуга с поворотом и уклоном, как у фильтра). Последняя — прогноз на горизонт."""
    return np.stack([
        anchor_point(ctrv_step(state, horizon_s * k / FORECAST_POINTS, motion.yaw_tau_s, motion.pitch_tau_s), model)
        for k in range(1, FORECAST_POINTS + 1)
    ])


def project_point(pose: CameraPose, xyz: np.ndarray) -> tuple[float, float] | None:
    """Пиксель точки мира на камере; за камерой — None."""
    pixels, depth = project_points(pose, np.asarray(xyz, np.float64).reshape(1, 3))
    if float(depth[0]) <= 0.5:
        return None
    return float(pixels[0, 0]), float(pixels[0, 1])


@dataclass
class Tick:
    steps: list[Step]
    world: WorldView
    trail: list[np.ndarray] = field(default_factory=list)
    rays: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    # Режим машины: 8 углов 3D-ящика в мире (порядок app.vehicle.BOX_EDGES).
    box: np.ndarray | None = None
    # Калиброванная неопределённость позиции цели, м (режим машины).
    sigma_m: float | None = None


class Session:
    def __init__(self, profile: TargetProfile, source):
        self.profile = profile
        self.pipelines: list[Pipeline] = []
        self.poses: dict[str, CameraPose] = {}
        self.samples: dict[str, list] = {}
        self.pose_root: Path | None = None
        self.moving_pose_ids: set[str] = {"cam-5"}
        # Папка любой записи (app.scene_folder): поза каждой камеры по номеру кадра из её таблицы.
        self.pose_tables: dict = {}
        # Машина, выбранная оператором: камера, где её выбрали, и счётчики несогласия других камер.
        self.selection_camera: str | None = None
        self._mismatch: dict[str, int] = {}
        self._selection_ref: tuple[np.ndarray, float] | None = None
        self._evidence: dict[str, dict] = {}
        # Детектор грузовика по камерам для режима машины (vehicle.detector в профиле); False — не собрался.
        self._truck_detectors: dict[str, object] = {}
        # Камеры, которые едут (центр ушёл от первой позы): у них пятна только от детектора.
        self._first_centers: dict[str, np.ndarray] = {}
        self._translating: set[str] = set()
        self.tracks3d: list[Track3D] = [Track3D(profile)]
        self.track3d = self.tracks3d[0]
        self.trail3d: deque[np.ndarray] = deque(maxlen=4000)
        # Номер кадра записи и точка следа — для сверки с таблицей по кадрам.
        self.trail_frames: deque[tuple[int, np.ndarray]] = deque(maxlen=4000)
        self.target_id = 1
        self.vehicle: VehicleFusion | None = None
        self.backgrounds: dict[str, StaticBackground | RotatingBackground] = {}
        # Последнее состояние 3D-ящика и его время: прогноз маски, где фон не учится.
        self._held: tuple[np.ndarray, float] | None = None
        self._trail_reset()
        self._setup_vehicle()
        self.set_sources([source])

    def _setup_vehicle(self) -> None:
        spec = getattr(self.profile, "vehicle", None)
        if not spec:
            self.vehicle = None
            return
        config = FusionConfig(range_m=(self.profile.r_min, self.profile.r_max), hold_motion_single=True)
        self.vehicle = VehicleFusion(VehicleModel.from_config(spec), config)

    def set_profile(self, profile: TargetProfile) -> None:
        self.profile = profile
        self.tracks3d = [Track3D(profile)]
        self.track3d = self.tracks3d[0]
        self.trail3d.clear()
        self.trail_frames.clear()
        self.backgrounds.clear()
        self._truck_detectors = {}
        self._first_centers = {}
        self._translating = set()
        self._held = None
        self._setup_vehicle()
        for pipeline in self.pipelines:
            pipeline.set_profile(profile)

    def set_sources(self, sources, poses: dict[str, CameraPose] | None = None) -> None:
        self._close()
        self.pipelines = [Pipeline(self.profile, source) for source in sources]
        self.poses = dict(poses or {})
        self.samples.clear()
        self.pose_root = None
        self.pose_tables = {}
        self.moving_pose_ids = {"cam-5"}
        self.selection_camera = None
        self._mismatch = {}
        self._selection_ref = None
        self._evidence = {}
        self.tracks3d = [Track3D(self.profile)]
        self.track3d = self.tracks3d[0]
        self.trail3d.clear()
        self.trail_frames.clear()
        self.backgrounds.clear()
        self._held = None
        self._trail_reset()
        if self.vehicle is not None:
            self.vehicle.reset()

    def set_pose_source(self, root: Path | None) -> None:
        """Корень Way1 с CSV поз. После set_sources вызовите из use_way1(ROOT)."""
        self.pose_root = None if root is None else Path(root)
        self._mark_moving()

    # ------------------------------------------------------------ выбранная машина

    def select_target(self, camera_id: str, x: float, y: float) -> bool:
        """Оператор выбрал машину в точке кадра камеры. Путь 3D начинается заново."""
        pipeline = self._pipeline(camera_id)
        if pipeline is None or not pipeline.select_at(x, y):
            return False
        for other in self.pipelines:
            if other is not pipeline:
                other.clear_selection()
        self.selection_camera = camera_id
        self._mismatch = {}
        self._selection_ref = None
        self._evidence = {}
        self.tracks3d = [Track3D(self.profile)]
        self.track3d = self.tracks3d[0]
        self.trail3d.clear()
        self.trail_frames.clear()
        self.target_id += 1
        return True

    def clear_target(self) -> None:
        for pipeline in self.pipelines:
            pipeline.clear_selection()
        self.selection_camera = None
        self._mismatch = {}
        self._selection_ref = None
        self._evidence = {}

    def selection_cameras(self) -> list[str]:
        """Камеры, которые сейчас ведут выбранную машину."""
        return [
            getattr(pipeline.source, "camera_id", "")
            for pipeline in self.pipelines
            if pipeline.tracker.has_target
        ]

    def _pipeline(self, camera_id: str) -> Pipeline | None:
        return next(
            (pipeline for pipeline in self.pipelines if getattr(pipeline.source, "camera_id", None) == camera_id),
            None,
        )

    def _match_selection(self, steps: list[Step]) -> None:
        """Та же машина на других камерах: у проекции её 3D-точки, а пока точки нет — по лучу."""
        if self.selection_camera is None:
            return
        by_id = {step.frame.camera_id: step for step in steps}
        anchor = by_id.get(self.selection_camera)
        pose_a = self.poses.get(self.selection_camera)
        if anchor is None or pose_a is None or not anchor.selected:
            return
        pixel_a = (anchor.track.cx, anchor.track.cy)
        reference = None
        if self._selection_ref is not None and anchor.frame.time_s - self._selection_ref[1] <= MATCH_REF_S:
            reference = self._selection_ref[0]
        for pipeline in self.pipelines:
            camera_id = getattr(pipeline.source, "camera_id", None)
            pose_b = self.poses.get(camera_id) if camera_id is not None else None
            if camera_id == self.selection_camera or camera_id not in by_id or pose_b is None:
                continue
            tracker = pipeline.tracker
            if tracker.has_target:
                track = by_id[camera_id].track
                bad = self._match_score(pose_a, pixel_a, pose_b, track, reference, loose=True) is None
                self._mismatch[camera_id] = self._mismatch.get(camera_id, 0) + 1 if bad else 0
                if self._mismatch[camera_id] < MATCH_LOSE_FRAMES:
                    continue
                # Долго не сходится с выбранной — это другая машина: ищем заново.
                tracker.clear_selection()
                self._mismatch[camera_id] = 0
            items = tracker.track_items
            scored = []
            for track, view in items:
                if not view.active:
                    continue
                score = self._match_score(pose_a, pixel_a, pose_b, view, reference)
                if score is not None:
                    scored.append((score, id(track), track))
            if not scored:
                continue
            scored.sort(key=lambda item: item[0])
            if reference is not None or len(scored) == 1:
                tracker.select_track(scored[0][2])
                self._evidence.pop(camera_id, None)
                continue
            # Несколько машин на луче выбранной: копим невязку каждого трека по кадрам.
            history = self._evidence.setdefault(camera_id, {})
            alive = {key for _, key, _ in scored}
            for key in list(history):
                if key not in alive:
                    del history[key]
            for score, key, track in scored:
                errors = history.setdefault(key, [track, []])[1]
                errors.append(score)
                del errors[:-MATCH_SETTLE]
            ranked = sorted(history.values(), key=lambda item: float(np.mean(item[1])))
            best_track, best_errors = ranked[0]
            best_mean = float(np.mean(best_errors))
            second_mean = float(np.mean(ranked[1][1])) if len(ranked) > 1 else float("inf")
            settled = len(best_errors) >= MATCH_SETTLE
            if (len(best_errors) >= 3 and second_mean >= MATCH_UNIQUE * best_mean + 5.0) or settled:
                tracker.select_track(best_track)
                self._evidence.pop(camera_id, None)

    def _match_score(self, pose_a, pixel_a, pose_b, view, reference, loose: bool = False) -> float | None:
        """Насколько трек камеры B похож на выбранную машину, px; не похож — None."""
        widen = 1.6 if loose else 1.0
        if reference is not None:
            pixel = project_point(pose_b, reference)
            if pixel is None:
                return None
            size = max(view.bbox[2], view.bbox[3]) if view.bbox is not None else 0.0
            distance = float(np.hypot(view.cx - pixel[0], view.cy - pixel[1]))
            return distance if distance <= widen * max(MATCH_NEAR_PX, 0.6 * size) else None
        error = self._pair_error(pose_a, pixel_a, pose_b, (view.cx, view.cy))
        limit = MATCH_KEEP_PX if loose else MATCH_PICK_PX
        return error if error is not None and error <= limit else None

    def _pair_error(self, pose_a: CameraPose, pixel_a, pose_b: CameraPose, pixel_b) -> float | None:
        """Невязка репроекции точки по двум камерам, px. Точка вне дальности профиля — None."""
        xyz = triangulate([pose_a, pose_b], [pixel_a, pixel_b], max_reproj_px=1e9)
        if xyz is None:
            return None
        distance = range_from_cameras(xyz, [pose_a, pose_b])
        if not (self.profile.r_min <= distance <= self.profile.r_max):
            return None
        return _reprojection_error([pose_a, pose_b], [pixel_a, pixel_b], xyz)

    def set_pose_tables(self, tables: dict, moving: set[str]) -> None:
        """Папка любой записи: поза камеры по кадру — tables[camera_id](frame). После set_sources.

        moving — камеры, которые поворачиваются или едут: у них фон по направлению луча.
        """
        self.pose_tables = dict(tables)
        self.moving_pose_ids = set(moving)
        self.backgrounds.clear()
        self._mark_moving()

    def _mark_moving(self) -> None:
        """Камеры с известным поворотом по кадрам: их детектор движения — с компенсацией поворота."""
        for pipeline in self.pipelines:
            camera_id = getattr(pipeline.source, "camera_id", None)
            known = self.pose_root is not None or camera_id in self.pose_tables
            pipeline.set_moving(known and camera_id in self.moving_pose_ids)

    def set_rig(self, rig: Rig) -> None:
        sources = [
            RigCamera(rig, camera_id, leader=(index == 0))
            for index, camera_id in enumerate(rig.camera_ids)
        ]
        self.set_sources(sources, rig.poses)

    def add_source(self, source) -> None:
        self.pipelines.append(Pipeline(self.profile, source))
        self._mark_moving()

    def drop_pipeline(self, pipeline: Pipeline) -> None:
        """Закрывает один источник. Остальные камеры и след 3D не сбрасывает."""
        if pipeline not in self.pipelines:
            return
        self.pipelines.remove(pipeline)
        close = getattr(pipeline.source, "close", None)
        if close is not None:
            close()

    def step(self) -> Tick | None:
        if self.vehicle is not None and len(self.poses) >= 2:
            return self._step_vehicle()
        steps: list[Step] = []
        for pipeline in self.pipelines:
            step = pipeline.step()
            if step is not None:
                self._refresh_moving_pose(pipeline, step)
                steps.append(step)
        if not steps:
            return None
        self._match_selection(steps)
        world, rays = self._fuse(steps)
        if world.fresh and world.xyz is not None:
            if self.selection_camera is not None:
                self._selection_ref = (world.xyz.copy(), max(step.frame.time_s for step in steps))
            self.trail3d.append(world.xyz.copy())
            index = steps[0].frame.index
            if index is not None:
                self.trail_frames.append((int(index), world.xyz.copy()))
        return Tick(steps, world, list(self.trail3d), rays)

    # ------------------------------------------------------------ режим машины

    def _background(self, camera_id: str, image: np.ndarray) -> StaticBackground | RotatingBackground:
        model = self.backgrounds.get(camera_id)
        if model is None:
            height, width = image.shape[:2]
            config = BackgroundConfig(work_width=LIVE_WORK_WIDTH)
            known = self.pose_root is not None or camera_id in self.pose_tables
            rotating = known and camera_id in self.moving_pose_ids
            model = (RotatingBackground if rotating else StaticBackground)(width, height, config)
            self.backgrounds[camera_id] = model
        return model

    def _step_vehicle(self) -> Tick | None:
        """Кадр каждой камеры → фон → пятна → подгонка 3D-ящика по рамкам всех камер."""
        assert self.vehicle is not None
        frames = []
        views: dict[str, CameraView] = {}
        for pipeline in self.pipelines:
            frame = pipeline.source.read()
            if frame is None:
                continue
            if getattr(pipeline.source, "rewound", False):
                self.backgrounds.pop(frame.camera_id, None)
                self.vehicle.reset()
                self._held = None
                self._trail_reset()
            self._refresh_pose(pipeline, frame)
            frames.append(frame)
            pose = self.poses.get(frame.camera_id)
            if pose is None:
                continue
            detected = self._truck_blobs(frame)
            if self._detector_only(frame.camera_id, pose) and self._truck_detectors.get(frame.camera_id):
                # Едущая камера (или профиль «только детектор»): фон с параллаксом только мешает.
                blobs = detected
            else:
                background = self._background(frame.camera_id, frame.image)
                mask = background.apply(frame.image, pose, self._hold_mask(pose, frame.time_s))
                blobs = foreground_blobs(mask, background.scale) + detected
            views[frame.camera_id] = CameraView(pose, blobs)
        if not frames:
            return None
        time_s = max(frame.time_s for frame in frames)
        key = frames[0].index
        view = self.vehicle.step(time_s, views, key=key)
        steps = [self._vehicle_step(frame, view.observed.get(frame.camera_id)) for frame in frames]
        if not view.active or view.state is None:
            self._held = None
            self._trail_reset()
            return Tick(steps, WorldView(False, None, 0.0, None), list(self.trail3d))
        state = view.state
        self._held = (np.array(state, np.float64, copy=True), time_s)
        model = self.vehicle.model
        anchor = anchor_point(state, model)
        motion = self.vehicle.motion
        path = vehicle_forecast(state, model, motion, float(self.profile.predict_s))
        ahead = path[-1]
        velocity = float(state[V]) * state_axes(state)[0]
        relative = {camera_id: velocity_in_camera(velocity, pose) for camera_id, pose in self.poses.items()}
        world = WorldView(True, anchor, view.speed_m_s, ahead, relative, view.fresh, path)
        sigma = None if view.sigma_m is None else SIGMA_SCALE * view.sigma_m
        if self._trail_accept(key, anchor, view.fresh, len(view.observed), sigma):
            self.trail3d.append(anchor.copy())
            if key is not None:
                self.trail_frames.append((int(key), anchor.copy()))
        return Tick(steps, world, list(self.trail3d), [], view.corners, sigma)

    def _trail_reset(self) -> None:
        self._trail_run = 0
        self._trail_coast = 0
        self._trail_rejects = 0
        self._trail_last: tuple[int, np.ndarray] | None = None

    def _trail_accept(
        self, key, anchor: np.ndarray, fresh: bool, cameras: int, sigma_m: float | None = None
    ) -> bool:
        """Писать ли точку в след: прогрев, короткий пропуск — прогноз, скачок — выброс."""
        self._trail_run += 1
        self._trail_coast = 0 if fresh else self._trail_coast + 1
        if self._trail_run <= TRAIL_WARMUP or self._trail_coast > TRAIL_COAST:
            return False
        if fresh and cameras < TRAIL_MIN_CAMERAS:
            return False
        if sigma_m is not None and sigma_m > TRAIL_SIGMA_M:
            return False
        if key is not None and self._trail_last is not None:
            last_key, last_point = self._trail_last
            frames = max(1, int(key) - last_key)
            if float(np.linalg.norm(anchor - last_point)) > TRAIL_STEP_M * frames:
                self._trail_rejects += 1
                if self._trail_rejects < TRAIL_REJECT_LIMIT:
                    return False
        self._trail_rejects = 0
        if key is not None:
            self._trail_last = (int(key), anchor.copy())
        return True

    def _hold_mask(self, pose: CameraPose, time_s: float) -> np.ndarray | None:
        """Проекция прогнозного 3D-ящика на кадр камеры с запасом: там цель, фон не учится."""
        if self._held is None or self.vehicle is None:
            return None
        state, last_s = self._held
        motion = self.vehicle.motion
        ahead = ctrv_step(state, max(0.0, time_s - last_s), motion.yaw_tau_s, motion.pitch_tau_s)
        pixels, depth = project_points(pose, box_corners(ahead, self.vehicle.model))
        if float(depth.min()) <= 0.5:
            return None
        hull = cv2.convexHull(pixels.astype(np.float32)).astype(np.int32)
        mask = np.zeros((pose.height, pose.width), np.uint8)
        cv2.fillConvexPoly(mask, hull, 1)
        size = 2 * HOLD_PAD_PX + 1
        return cv2.dilate(mask, np.ones((size, size), np.uint8)) > 0

    def _detector_only(self, camera_id: str, pose: CameraPose) -> bool:
        """Профиль «только детектор» или камера уже сдвинулась больше TRANSLATE_M от первой позы."""
        if (self.profile.vehicle or {}).get("detector_mode") == "detector":
            return True
        rotation, _ = cv2.Rodrigues(pose.rvec)
        center = (-rotation.T @ pose.tvec).ravel()
        first = self._first_centers.setdefault(camera_id, center)
        if float(np.linalg.norm(center - first)) > TRANSLATE_M:
            self._translating.add(camera_id)
        return camera_id in self._translating

    def _truck_blobs(self, frame) -> list[Blob]:
        """Рамки детектора грузовика (vehicle.detector в профиле) как пятна для подгонки ящика."""
        spec = self.profile.vehicle or {}
        if not spec.get("detector") or spec.get("detector_mode", "both") == "background":
            return []
        detector = self._truck_detectors.get(frame.camera_id)
        if detector is None:
            from app.object_detector import DetectorError, ObjectDetector, parse_spec

            # Без порогов площади профиля: близкий грузовик крупнее s_max профиля way1.
            wide = TargetProfile("truck_detector", 0.0, 1e12, 0.0, 1e9, 0.0, 1e9, 0.0, 1e9, None, 0.5, None)
            try:
                detector = ObjectDetector(wide, parse_spec({
                    "mode": "object", "class": "truck", "model": str(spec["detector"]), "conf": 0.25,
                    "det_every": 1, "det_width": 960, "roi_scale": 0,
                }))
            except DetectorError:
                detector = False
            self._truck_detectors[frame.camera_id] = detector
        if detector is False:
            return []
        return [
            Blob(float(x), float(y), float(w), float(h), float(w * h))
            for x, y, w, h in (item.bbox for item in detector.detect(frame.image))
        ]

    def _vehicle_step(self, frame, bbox) -> Step:
        """Шаг камеры для окна: рамка — силуэт, вошедший в измерение 3D-ящика."""
        horizon = float(self.profile.predict_s)
        if bbox is None:
            idle = TrackView(False, 0.0, 0.0, 0.0, 0.0, 0.0, "поиск", False, False,
                             0.0, 0.0, None, None, None, (), horizon)
            return Step(frame, None, idle)
        x, y, w, h = (int(round(value)) for value in bbox)
        cx, cy = x + 0.5 * w, y + 0.5 * h
        detection = Detection(cx, cy, float(w * h), (x, y, w, h))
        track = TrackView(True, cx, cy, 0.0, 0.0, 0.0, "сопровождение", True, True,
                          cx, cy, (x, y, w, h), None, float(w * h), (), horizon)
        return Step(frame, detection, track)

    def snap_board(self, detector: BoardDetector, frames: list) -> list[str]:
        found = []
        for frame in frames:
            view = detector.detect(frame.image)
            if view is None:
                continue
            self.samples.setdefault(frame.camera_id, []).append(view)
            found.append(frame.camera_id)
        return found

    def solve_poses(self) -> list[str]:
        solved = []
        for camera_id, views in self.samples.items():
            intrinsics = intrinsics_from_views(views)
            matrix, dist = intrinsics if intrinsics else (None, None)
            pose = pose_from_view(camera_id, views[-1], matrix, dist)
            if pose is None:
                continue
            self.poses[camera_id] = pose
            solved.append(camera_id)
        return solved

    def _refresh_moving_pose(self, pipeline: Pipeline, step: Step) -> None:
        self._refresh_pose(pipeline, step.frame)

    def _refresh_pose(self, pipeline: Pipeline, frame) -> None:
        """Поза камеры, которая поворачивается, — по номеру кадра из таблицы записи."""
        camera_id = frame.camera_id
        table = self.pose_tables.get(camera_id)
        if table is not None:
            number = int(frame.index) if frame.index is not None else max(0, int(getattr(pipeline.source, "index", 1)) - 1)
            self.poses[camera_id] = table(number)
            return
        if self.pose_root is None or camera_id not in self.moving_pose_ids:
            return
        from app.cine_table import pose_at

        if frame.index is not None:
            number = int(frame.index)
        else:
            number = max(0, int(getattr(pipeline.source, "index", 1)) - 1)
        height, width = frame.image.shape[:2]
        self.poses[camera_id] = pose_at(self.pose_root, camera_id, number, int(width), int(height))

    def _fuse(self, steps: list[Step]) -> tuple[WorldView, list[tuple[np.ndarray, np.ndarray]]]:
        poses = []
        pixels = []
        times = []
        ids = []
        for step in steps:
            if not step.track.measured or step.detection is None:
                continue
            # Выбрана машина: в 3D только камеры, которые ведут именно её.
            if self.selection_camera is not None and not step.selected:
                continue
            pose = self.poses.get(step.frame.camera_id)
            if pose is None:
                continue
            poses.append(pose)
            pixels.append((step.detection.cx, step.detection.cy))
            times.append(step.frame.time_s)
            ids.append(step.frame.camera_id)
        if self.selection_camera in ids and len(ids) > 2:
            # Камера с ошибочно выбранной чужой машиной не должна сорвать всю точку.
            a = ids.index(self.selection_camera)
            keep = [
                i for i in range(len(ids))
                if i == a or (self._pair_error(poses[a], pixels[a], poses[i], pixels[i]) or 1e9) <= MATCH_KEEP_PX
            ]
            poses = [poses[i] for i in keep]
            pixels = [pixels[i] for i in keep]
            times = [times[i] for i in keep]
        span = max(times) - min(times) if times else 0.0
        # Дальняя трасса: центры 2D шумнее, допуск репроекции шире.
        reproj = 60.0 if self.profile.v_max > 20.0 else 40.0
        xyz = (
            triangulate(poses, pixels, max_reproj_px=reproj)
            if len(poses) >= 2 and span <= PAIR_SYNC_S
            else None
        )
        if xyz is not None:
            distance = range_from_cameras(xyz, poses)
            if not (self.profile.r_min <= distance <= self.profile.r_max):
                xyz = None
        rays = [] if xyz is None else sight_rays(poses, xyz)
        time_s = max(step.frame.time_s for step in steps)
        return self._update_tracks3d(xyz, time_s, self.poses), rays

    def _jump_cap(self, dt: float) -> float:
        p = self.profile
        return max(0.15, 0.35 * p.v_max * dt, p.v_max * dt)

    @staticmethod
    def _predicted_xyz(track: Track3D, time_s: float) -> np.ndarray:
        state = track.kf.statePost.ravel()
        dt = max(1e-3, time_s - track.last_t)
        return state[:3] + state[3:6] * dt

    def _xyz_in_range(self, xyz: np.ndarray, poses: dict[str, CameraPose]) -> bool:
        distance = range_from_cameras(xyz, poses)
        return self.profile.r_min <= distance <= self.profile.r_max

    def _pick_track_for_xyz(
        self, xyz: np.ndarray, time_s: float, poses: dict[str, CameraPose]
    ) -> int | None:
        if not self._xyz_in_range(xyz, poses):
            return None
        best_i: int | None = None
        best_jump = float("inf")
        for i, track in enumerate(self.tracks3d):
            if not track.ready:
                continue
            pred = self._predicted_xyz(track, time_s)
            jump = float(np.linalg.norm(xyz - pred))
            dt = max(1e-3, time_s - track.last_t)
            if jump <= self._jump_cap(dt) and jump < best_jump:
                best_jump = jump
                best_i = i
        if best_i is not None:
            return best_i
        for i, track in enumerate(self.tracks3d):
            if not track.ready:
                return i
        if len(self.tracks3d) < MAX_TRACKS3D:
            self.tracks3d.append(Track3D(self.profile))
            self.target_id += 1
            return len(self.tracks3d) - 1
        return None

    def _update_tracks3d(
        self,
        xyz: np.ndarray | None,
        time_s: float,
        poses: dict[str, CameraPose],
    ) -> WorldView:
        assigned: int | None = None
        if xyz is not None:
            assigned = self._pick_track_for_xyz(xyz, time_s, poses)
        main: WorldView | None = None
        for i, track in enumerate(self.tracks3d):
            measurement = xyz if i == assigned else None
            world = track.update(measurement, time_s, poses)
            if i == 0:
                main = world
        self.track3d = self.tracks3d[0]
        assert main is not None
        return main

    def _close(self) -> None:
        for pipeline in self.pipelines:
            close = getattr(pipeline.source, "close", None)
            if close is not None:
                close()
