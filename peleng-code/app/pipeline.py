from dataclasses import dataclass

from app.motion import Detection, MotionDetector
from app.profile import TargetProfile
from app.source import Frame
from app.track2d import MultiTrack2D, TrackView


@dataclass
class Step:
    frame: Frame
    detection: Detection | None
    track: TrackView
    tracks: list[TrackView] | None = None


class Pipeline:
    def __init__(self, profile: TargetProfile, source):
        self.profile = profile
        self.source = source
        self.detector = MotionDetector(profile)
        self.tracker = MultiTrack2D(profile)

    def set_source(self, source) -> None:
        old = self.source
        self.source = source
        if hasattr(old, "close"):
            old.close()
        self.detector = MotionDetector(self.profile)
        self.tracker.reset()

    def step(self) -> Step | None:
        frame = self.source.read()
        if frame is None:
            return None
        if getattr(self.source, "rewound", False):
            self.detector = MotionDetector(self.profile)
            self.tracker.reset()
        spots = self.detector.detect(frame.image)
        track = self.tracker.update(spots, frame.time_s)
        all_tracks = list(self.tracker.tracks)
        extra = all_tracks if len(all_tracks) > 1 else None
        return Step(frame, self.tracker.accepted, track, extra)
