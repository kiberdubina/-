from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from app.background import BackgroundConfig, RotatingBackground, StaticBackground, foreground_blobs
from app.calib import BoardDetector, intrinsics_from_views, pose_from_view
from app.geom import CameraPose, range_from_cameras, sight_rays, triangulate, velocity_in_camera
from app.motion import Detection
from app.pipeline import Pipeline, Step
from app.profile import TargetProfile
from app.rig import Rig, RigCamera
from app.track2d import TrackView
from app.track3d import Track3D, WorldView
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
TRAIL_REJECT_LIMIT = 5    # столько выбросов подряд — это не выброс, а исправление трека
TRAIL_COAST = 5           # столько кадров без измерения пишется прогноз фильтра


@dataclass
class Tick:
    steps: list[Step]
    world: WorldView
    trail: list[np.ndarray] = field(default_factory=list)
    rays: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    # Режим машины: 8 углов 3D-ящика в мире (порядок app.vehicle.BOX_EDGES).
    box: np.ndarray | None = None


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
        config = FusionConfig(range_m=(self.profile.r_min, self.profile.r_max))
        self.vehicle = VehicleFusion(VehicleModel.from_config(spec), config)

    def set_profile(self, profile: TargetProfile) -> None:
        self.profile = profile
        self.tracks3d = [Track3D(profile)]
        self.track3d = self.tracks3d[0]
        self.trail3d.clear()
        self.trail_frames.clear()
        self.backgrounds.clear()
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

    def set_pose_tables(self, tables: dict, moving: set[str]) -> None:
        """Папка любой записи: поза камеры по кадру — tables[camera_id](frame). После set_sources.

        moving — камеры, которые поворачиваются или едут: у них фон по направлению луча.
        """
        self.pose_tables = dict(tables)
        self.moving_pose_ids = set(moving)
        self.backgrounds.clear()

    def set_rig(self, rig: Rig) -> None:
        sources = [
            RigCamera(rig, camera_id, leader=(index == 0))
            for index, camera_id in enumerate(rig.camera_ids)
        ]
        self.set_sources(sources, rig.poses)

    def add_source(self, source) -> None:
        self.pipelines.append(Pipeline(self.profile, source))

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
        world, rays = self._fuse(steps)
        if world.fresh and world.xyz is not None:
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
            background = self._background(frame.camera_id, frame.image)
            mask = background.apply(frame.image, pose, self._hold_mask(pose, frame.time_s))
            views[frame.camera_id] = CameraView(pose, foreground_blobs(mask, background.scale))
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
        ahead = anchor_point(
            ctrv_step(state, self.profile.predict_s, motion.yaw_tau_s, motion.pitch_tau_s), model
        )
        velocity = float(state[V]) * state_axes(state)[0]
        relative = {camera_id: velocity_in_camera(velocity, pose) for camera_id, pose in self.poses.items()}
        world = WorldView(True, anchor, view.speed_m_s, ahead, relative, view.fresh)
        if self._trail_accept(key, anchor, view.fresh, len(view.observed)):
            self.trail3d.append(anchor.copy())
            if key is not None:
                self.trail_frames.append((int(key), anchor.copy()))
        return Tick(steps, world, list(self.trail3d), [], view.corners)

    def _trail_reset(self) -> None:
        self._trail_run = 0
        self._trail_coast = 0
        self._trail_rejects = 0
        self._trail_last: tuple[int, np.ndarray] | None = None

    def _trail_accept(self, key, anchor: np.ndarray, fresh: bool, cameras: int) -> bool:
        """Писать ли точку в след: прогрев, короткий пропуск — прогноз, скачок — выброс."""
        self._trail_run += 1
        self._trail_coast = 0 if fresh else self._trail_coast + 1
        if self._trail_run <= TRAIL_WARMUP or self._trail_coast > TRAIL_COAST:
            return False
        if fresh and cameras < TRAIL_MIN_CAMERAS:
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
        for step in steps:
            if not step.track.measured or step.detection is None:
                continue
            pose = self.poses.get(step.frame.camera_id)
            if pose is None:
                continue
            poses.append(pose)
            pixels.append((step.detection.cx, step.detection.cy))
            times.append(step.frame.time_s)
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
