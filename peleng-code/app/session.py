from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.calib import BoardDetector, intrinsics_from_views, pose_from_view
from app.geom import CameraPose, range_from_cameras, sight_rays, triangulate
from app.pipeline import Pipeline, Step
from app.profile import TargetProfile
from app.rig import Rig, RigCamera
from app.track3d import Track3D, WorldView

# Два кадра дальше этого зазора — разные положения цели. Такую пару не триангулируем.
PAIR_SYNC_S = 0.08
MAX_TRACKS3D = 4


@dataclass
class Tick:
    steps: list[Step]
    world: WorldView
    trail: list[np.ndarray] = field(default_factory=list)
    rays: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)


class Session:
    def __init__(self, profile: TargetProfile, source):
        self.profile = profile
        self.pipelines: list[Pipeline] = []
        self.poses: dict[str, CameraPose] = {}
        self.samples: dict[str, list] = {}
        self.pose_root: Path | None = None
        self.moving_pose_ids: set[str] = {"cam-5"}
        self.tracks3d: list[Track3D] = [Track3D(profile)]
        self.track3d = self.tracks3d[0]
        self.trail3d: deque[np.ndarray] = deque(maxlen=4000)
        self.target_id = 1
        self.set_sources([source])

    def set_profile(self, profile: TargetProfile) -> None:
        self.profile = profile
        self.tracks3d = [Track3D(profile)]
        self.track3d = self.tracks3d[0]
        self.trail3d.clear()
        for pipeline in self.pipelines:
            pipeline.profile = profile
            pipeline.detector.profile = profile
            pipeline.tracker.profile = profile
            pipeline.tracker.reset()

    def set_sources(self, sources, poses: dict[str, CameraPose] | None = None) -> None:
        self._close()
        self.pipelines = [Pipeline(self.profile, source) for source in sources]
        self.poses = dict(poses or {})
        self.samples.clear()
        self.pose_root = None
        self.tracks3d = [Track3D(self.profile)]
        self.track3d = self.tracks3d[0]
        self.trail3d.clear()

    def set_pose_source(self, root: Path | None) -> None:
        """Корень Way1 с CSV поз. После set_sources вызовите из use_way1(ROOT)."""
        self.pose_root = None if root is None else Path(root)

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
        return Tick(steps, world, list(self.trail3d), rays)

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
        camera_id = step.frame.camera_id
        if self.pose_root is None or camera_id not in self.moving_pose_ids:
            return
        from app.cine_table import pose_at

        source = pipeline.source
        frame = max(0, int(getattr(source, "index", 1)) - 1)
        height, width = step.frame.image.shape[:2]
        self.poses[camera_id] = pose_at(
            self.pose_root, camera_id, frame, int(width), int(height)
        )

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
