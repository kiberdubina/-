from dataclasses import dataclass

from app.motion import Detection
from app.object_detector import Label, build_detector
from app.profile import DESCRIBED_MODES, TargetProfile
from app.source import Frame
from app.track2d import MultiTrack2D, TrackView


@dataclass
class Step:
    frame: Frame
    detection: Detection | None
    track: TrackView
    tracks: list[TrackView] | None = None
    # Цель по описанию: рамки модели с классом или запросом и уверенностью.
    labels: list[Label] | None = None
    # Главный трек — машина, выбранная оператором (Shift+клик).
    selected: bool = False


class Pipeline:
    def __init__(self, profile: TargetProfile, source):
        self.profile = profile
        self.source = source
        # Детектор по описанию не собрался (нет пакета, весов) — текст для строки статуса.
        self.detector_error: str | None = None
        # Камера поворачивается (Session знает её позы по кадрам): детектор движения — ego.
        self.moving = False
        self.detector = self._make_detector()
        self.tracker = MultiTrack2D(profile)
        self._last: TrackView | None = None

    def _make_detector(self):
        detector, self.detector_error = build_detector(self.profile, self.profile.detect)
        if hasattr(detector, "force_ego"):
            detector.force_ego = self.moving
        return detector

    def set_moving(self, moving: bool) -> None:
        self.moving = bool(moving)
        if hasattr(self.detector, "force_ego"):
            self.detector.force_ego = self.moving

    def set_profile(self, profile: TargetProfile) -> None:
        described = self.profile.mode in DESCRIBED_MODES or profile.mode in DESCRIBED_MODES
        self.profile = profile
        self.tracker.profile = profile
        self.tracker.reset()
        self._last = None
        if described:
            self.detector = self._make_detector()
        else:
            # Фон MOG2 не сбрасывается при смене порогов.
            self.detector.profile = profile

    def set_source(self, source) -> None:
        old = self.source
        self.source = source
        if hasattr(old, "close"):
            old.close()
        self.detector = self._make_detector()
        self.tracker.reset()
        self._last = None

    def step(self) -> Step | None:
        frame = self.source.read()
        if frame is None:
            return None
        if getattr(self.source, "rewound", False):
            self.detector = self._make_detector()
            self.tracker.reset()
            self._last = None
        set_hint = getattr(self.detector, "set_hint", None)
        if set_hint is not None:
            last = self._last
            if last is not None and last.active and last.bbox is not None:
                set_hint((last.cx, last.cy, float(last.bbox[2]), float(last.bbox[3])))
            else:
                set_hint(None)
        spots = self.detector.detect(frame.image)
        track = self.tracker.update(spots, frame.time_s)
        self._last = track
        all_tracks = list(self.tracker.tracks)
        extra = all_tracks if len(all_tracks) > 1 else None
        labels = getattr(self.detector, "labels", None)
        return Step(frame, self.tracker.accepted, track, extra, labels, self.tracker.has_target)

    def select_at(self, x: float, y: float) -> bool:
        """Оператор выбрал машину в точке кадра этой камеры."""
        found = self.tracker.select_at(x, y)
        if found:
            self._last = None
        return found

    def clear_selection(self) -> None:
        self.tracker.clear_selection()
