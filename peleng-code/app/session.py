from collections import deque
from dataclasses import dataclass, field

import numpy as np

from app.calib import BoardDetector, intrinsics_from_views, pose_from_view
from app.geom import CameraPose, sight_rays, triangulate
from app.pipeline import Pipeline, Step
from app.profile import TargetProfile
from app.rig import Rig, RigCamera
from app.track3d import Track3D, WorldView

# Два кадра дальше этого зазора — разные положения цели. Такую пару не триангулируем.
PAIR_SYNC_S = 0.08


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
        self.track3d = Track3D(profile)
        self.trail3d: deque[np.ndarray] = deque(maxlen=600)
        self.target_id = 1
        self.set_sources([source])

    def set_profile(self, profile: TargetProfile) -> None:
        self.profile = profile
        self.track3d = Track3D(profile)
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
        self.track3d.reset()
        self.trail3d.clear()

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
        xyz = triangulate(poses, pixels) if len(poses) >= 2 and span <= PAIR_SYNC_S else None
        if xyz is not None:
            distance = float(np.linalg.norm(xyz))
            if not (self.profile.r_min <= distance <= self.profile.r_max):
                xyz = None
        rays = [] if xyz is None else sight_rays(poses, xyz)
        time_s = max(step.frame.time_s for step in steps)
        return self.track3d.update(xyz, time_s, self.poses), rays

    def _close(self) -> None:
        for pipeline in self.pipelines:
            close = getattr(pipeline.source, "close", None)
            if close is not None:
                close()
