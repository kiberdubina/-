"""След мяча без выброса и прогноз на 0,5 с. Запуск: python check_trail.py [--video РОЛИК]

Сначала ворота Track2D на искусственных измерениях. Затем ролик с мячом проходит Pipeline
дважды — по цвету и по движению: рамка держится (не меньше 80 измерений), центр не прыгает
дальше 140 px за кадр, прогноз идёт по скорости, след не поднимается выше пути мяча.

Исходный ролик video_2026-11-01_22-06-07.mp4 снимался у участника и в репозиторий не попал
(*.mp4 в .gitignore). Нет его и не задан --video — ролик с известной правдой из _synthetic_clips.py.
Профиль — green_ball с горизонтом 0,5 с, а не активный профиль config.yaml.
"""

import argparse
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from app.motion import Detection
from app.pipeline import Pipeline
from app.profile import load_patterns
from app.source import VideoSource
from app.track2d import Track2D

ROOT = Path(__file__).resolve().parent
VIDEO = ROOT / "video_2026-11-01_22-06-07.mp4"
MAX_JUMP_PX = 140.0
HORIZON_S = 0.5
# Ложный центр на исходном ролике был около y=160–270, настоящий верх маха мяча — ниже.
VIDEO_TOP_Y = 280.0
# На ролике с известной правдой след не выше верха пути мяча с этим запасом.
TOP_MARGIN_PX = 40.0


def ball_profile():
    """Профиль мяча с горизонтом прогноза проверки (в config.yaml горизонт меняется кнопками)."""
    return replace(load_patterns(ROOT / "config.yaml")["green_ball"], predict_s=HORIZON_S)


def _blob(x: float, y: float, area: float = 30000.0) -> Detection:
    return Detection(x, y, area, (int(x) - 20, int(y) - 20, 40, 40))


def check_gate() -> None:
    profile = ball_profile()
    tracker = Track2D(profile)
    tracker.update([_blob(100, 200)], 0.0)
    view = None
    for index in range(1, 8):
        view = tracker.update([_blob(100 + index * 10, 200)], index * 0.04)
    assert view is not None and view.measured and tracker.accepted is not None

    near = _blob(180, 200, area=22000)
    far = _blob(530, 80, area=80000)
    view = tracker.update([far, near], 8 * 0.04)
    assert tracker.accepted is near
    assert view.measured

    before = view.cx
    view = tracker.update([_blob(before + 400, 200, area=50000)], 9 * 0.04)
    assert tracker.accepted is None
    assert not view.measured
    assert view.active
    assert abs(view.cx - before) < tracker.gate_px
    jumps = [
        float(np.hypot(view.trail[i][0] - view.trail[i - 1][0], view.trail[i][1] - view.trail[i - 1][1]))
        for i in range(1, len(view.trail))
    ]
    assert jumps and max(jumps) < MAX_JUMP_PX

    tracker.horizon_s = 0.25
    view = tracker.update([], 10 * 0.04)
    assert abs(view.horizon_s - 0.25) < 1e-9
    assert abs((view.pred_cx - view.cx) - view.vx * 0.25) < 1e-3
    assert abs((view.pred_cy - view.cy) - view.vy * 0.25) < 1e-3


def _run_video(profile, label: str, video: Path, top_y: float | None) -> None:
    """top_y — выше этой строки след подниматься не должен; None — правды нет, не проверяется."""
    source = VideoSource(str(video), "file")
    pipeline = Pipeline(profile, source)
    previous: tuple[int, float, float] | None = None
    max_jump = 0.0
    worst: tuple[int, int, float] | None = None
    measured = 0
    ahead = 0
    min_y = 1e9
    index = 0
    while True:
        step = pipeline.step()
        if step is None:
            break
        if source.rewound and index > 0:
            break
        track = step.track
        if not track.active:
            previous = None
        elif track.measured:
            measured += 1
            min_y = min(min_y, track.cy)
            if previous is not None and index == previous[0] + 1:
                jump = float(np.hypot(track.cx - previous[1], track.cy - previous[2]))
                if jump > max_jump:
                    max_jump = jump
                    worst = (previous[0], index, jump)
            previous = (index, track.cx, track.cy)
            if track.speed_px_s > 400:
                forward = track.vx * (track.pred_cx - track.cx) + track.vy * (track.pred_cy - track.cy)
                reach = float(np.hypot(track.pred_cx - track.cx, track.pred_cy - track.cy))
                if forward <= 0 or abs(reach - track.speed_px_s * track.horizon_s) > 1.5:
                    raise SystemExit(f"{label}: прогноз не по скорости на кадре {index}")
                ahead += 1
        index += 1
    where = f" кадры {worst[0]}–{worst[1]}" if worst else ""
    print(
        f"{label}: кадров {index}, измерений {measured}, "
        f"скачок {max_jump:.1f} px{where}, прогноз впереди {ahead}, верх следа y={min_y:.0f}"
    )
    if measured < 80:
        raise SystemExit(f"{label}: мало измерений, рамка не держится на мяче")
    if max_jump > MAX_JUMP_PX:
        raise SystemExit(f"{label}: скачок {max_jump:.1f} px выше {MAX_JUMP_PX:.0f}")
    if ahead < 10:
        raise SystemExit(f"{label}: прогноз почти не проверен")
    # Ложный центр уводит след вверх, к руке, и сразу возвращается.
    if top_y is not None and min_y < top_y:
        raise SystemExit(f"{label}: след поднялся до y={min_y:.0f}, выше y={top_y:.0f}")


def check_video(video: Path, top_y: float | None) -> None:
    profile = ball_profile()
    _run_video(profile, "цвет", video, top_y)
    motion = replace(profile, color=None, s_min=20000, s_max=90000)
    _run_video(motion, "движение", video, top_y)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="След мяча без выброса и прогноз на 0,5 с")
    parser.add_argument("--video", type=Path, help="свой ролик с зелёным мячом (верх следа не проверяется)")
    args = parser.parse_args(argv)
    check_gate()
    if args.video is not None:
        if not args.video.is_file():
            raise SystemExit(f"нет ролика {args.video}")
        check_video(args.video, None)
    elif VIDEO.is_file():
        check_video(VIDEO, VIDEO_TOP_Y)
    else:
        from _synthetic_clips import green_ball

        print(f"нет ролика {VIDEO.name} — синтетический мяч с известной правдой (_synthetic_clips.py)")
        with tempfile.TemporaryDirectory() as folder:
            clip = green_ball(Path(folder))
            check_video(clip.path, float(clip.truth[:, 1].min()) - TOP_MARGIN_PX)
    print(f"след без выброса, прогноз {HORIZON_S:g} с")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as error:
        sys.exit(f"ворота не держат измерение: {error}")
