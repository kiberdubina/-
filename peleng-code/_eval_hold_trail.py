"""Оценка удержания рамки и совпадения траектории.

PASS: hold>=90% кадров с видимой целью; trail>=80% (ошибка центра < 6% диагонали).
Правда на кадре — цветовая маска цели (GT_FN), от детектора конвейера она не зависит.

Ролики CASES снимались у участника и в репозиторий не попали (*.mp4 и *.mov в .gitignore).
Нет ролика — вместо него сцена того же вида из _synthetic_clips.py (кроме --no-synthetic):
это проверка того, что конвейер работает, а не точность на реальной съёмке.
Отчёт по исходным роликам — _eval_hold_trail_report.json; синтетика и --video пишут его
во временную папку и исходный отчёт не трогают. Ни одного ролика — код 2, отчёт не пишется.

Свой ролик: python _eval_hold_trail.py --video РОЛИК --profiles green_ball,motion --gt green
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from app.pipeline import Pipeline
from app.profile import load_patterns
from app.source import VideoSource

ROOT = Path(__file__).resolve().parent
HOLD_PASS = 0.90
TRAIL_PASS = 0.80
ERR_DIAG_FRAC = 0.06
WARMUP_S = 0.5

# исходники (не output_*)
CASES = [
    {
        "video": "video_2026-11-01_22-06-07.mp4",
        "profiles": ["green_ball", "motion"],
        "gt": "green",
    },
    {
        "video": "video_2026-11-02_15-54-01.mp4",
        "profiles": ["sky_uav", "aircraft", "motion"],
        "gt": "sky_dark",
    },
    {
        "video": "IMG_8241.MOV",
        "profiles": ["sky_uav", "aircraft", "motion"],
        "gt": "sky_dark",
    },
    {
        "video": "IMG_0985.MOV",
        "profiles": ["white_drone", "aircraft", "motion"],
        "gt": "white_drone",
    },
]


def green_center(image: np.ndarray):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (25, 80, 40), (100, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return _largest(mask, min_area=400)


def sky_dark_center(image: np.ndarray):
    """Тёмное пятно на ярком небе (дрон/БПЛА)."""
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sky = cv2.inRange(hsv, (0, 0, 145), (180, 80, 255))
    sky2 = cv2.inRange(hsv, (90, 25, 120), (135, 255, 255))
    sky = cv2.bitwise_or(sky, sky2)
    sky[int(h * 0.62) :, :] = 0
    sky[int(h * 0.28) : int(h * 0.55), int(w * 0.15) : int(w * 0.85)] = 0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dark = cv2.inRange(gray, 0, 85)
    mask = cv2.bitwise_and(dark, sky)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return _largest(mask, min_area=8, max_area=1200)


def white_drone_center(image: np.ndarray):
    """Светлый квадрокоптер в помещении."""
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 175), (180, 55, 255))
    mask[: int(h * 0.10), :] = 0
    mask[int(h * 0.75) :, :] = 0
    mask[:, : int(w * 0.04)] = 0
    mask[:, int(w * 0.96) :] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return _largest(mask, min_area=120, max_area=5000)


def _largest(mask: np.ndarray, min_area: float, max_area: float | None = None):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_a = 0.0
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        if area > best_a:
            best_a = area
            best = contour
    if best is None:
        return None
    moments = cv2.moments(best)
    if moments["m00"] == 0:
        return None
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"], best_a


GT_FN = {
    "green": green_center,
    "sky_dark": sky_dark_center,
    "white_drone": white_drone_center,
}


def point_in_bbox(cx: float, cy: float, bbox) -> bool:
    x, y, w, h = bbox
    return x <= cx <= x + w and y <= cy <= y + h


def evaluate(video: str, profile_name: str, gt_name: str) -> dict:
    path = Path(video) if Path(video).is_absolute() else ROOT / video
    patterns = load_patterns(ROOT / "config.yaml")
    profile = patterns[profile_name]
    source = VideoSource(str(path), "file")
    pipeline = Pipeline(profile, source)
    gt_fn = GT_FN[gt_name]

    total = 0
    after = 0
    gt_visible = 0
    hold = 0
    trail_ok = 0
    trail_n = 0
    errors: list[float] = []
    corr = None
    track_xy: list[tuple[float, float]] = []
    gt_xy: list[tuple[float, float]] = []
    max_err = None

    while True:
        step = pipeline.step()
        if step is None:
            break
        if source.rewound and total > 0:
            break
        total += 1
        image = step.frame.image
        h, w = image.shape[:2]
        if max_err is None:
            max_err = float(np.hypot(w, h)) * ERR_DIAG_FRAC
        if step.frame.time_s < WARMUP_S:
            continue
        after += 1
        gt = gt_fn(image)
        if gt is None:
            continue
        gx, gy, _ = gt
        gt_visible += 1

        if step.track.active:
            err = float(np.hypot(step.track.cx - gx, step.track.cy - gy))
            bbox = step.track.bbox
            on_target = err <= max_err or (bbox is not None and point_in_bbox(gx, gy, bbox))
            # удержание рамки: любой active (измерение или прогноз), если квадрат на цели
            if on_target:
                hold += 1
            if step.track.measured:
                errors.append(err)
                trail_n += 1
                if on_target:
                    trail_ok += 1
                track_xy.append((step.track.cx, step.track.cy))
                gt_xy.append((gx, gy))

    hold_r = hold / gt_visible if gt_visible else 0.0
    trail_r = trail_ok / trail_n if trail_n else 0.0
    if len(track_xy) >= 15:
        tx = np.array([p[0] for p in track_xy])
        ty = np.array([p[1] for p in track_xy])
        gx = np.array([p[0] for p in gt_xy])
        gy = np.array([p[1] for p in gt_xy])
        cx = float(np.corrcoef(tx, gx)[0, 1]) if np.std(tx) > 1 and np.std(gx) > 1 else 0.0
        cy = float(np.corrcoef(ty, gy)[0, 1]) if np.std(ty) > 1 and np.std(gy) > 1 else 0.0
        if np.isnan(cx):
            cx = 0.0
        if np.isnan(cy):
            cy = 0.0
        corr = (cx + cy) / 2.0

    # trail reality: комбинация overlap и корреляции, если corr есть
    trail_score = trail_r
    if corr is not None:
        trail_score = 0.7 * trail_r + 0.3 * max(0.0, corr)

    passed = gt_visible >= 8 and hold_r >= HOLD_PASS and trail_score >= TRAIL_PASS
    return {
        "video": Path(video).name,
        "profile": profile_name,
        "gt": gt_name,
        "frames": total,
        "after_warmup": after,
        "gt_visible": gt_visible,
        "hold": round(hold_r, 4),
        "trail_overlap": round(trail_r, 4),
        "trail_score": round(trail_score, 4),
        "corr": None if corr is None else round(corr, 4),
        "median_err_px": None if not errors else round(float(np.median(errors)), 1),
        "err_limit_px": None if max_err is None else round(max_err, 1),
        "PASS": passed,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Удержание рамки и совпадение траектории")
    parser.add_argument("--video", type=Path, help="свой ролик вместо списка CASES")
    parser.add_argument("--profiles", default="motion", help="профили через запятую (с --video)")
    parser.add_argument("--gt", choices=sorted(GT_FN), default="green", help="правда на кадре (с --video)")
    parser.add_argument("--no-synthetic", action="store_true", help="нет ролика — пропустить, без синтетики")
    parser.add_argument("--report", type=Path, help="куда писать отчёт JSON")
    args = parser.parse_args(argv)
    if args.video is not None:
        cases = [{"video": str(args.video.resolve()), "profiles": args.profiles.split(","), "gt": args.gt}]
    else:
        cases = CASES

    rows = []
    best_by_video: dict[str, dict] = {}
    synthetic = False
    with tempfile.TemporaryDirectory() as folder:
        made: set[str] = set()
        for case in cases:
            video = case["video"]
            path = Path(video) if Path(video).is_absolute() else ROOT / video
            label, source = Path(video).name, "real"
            if not path.is_file():
                if args.video is not None or args.no_synthetic:
                    rows.append({"video": label, "error": "missing"})
                    continue
                kind = case["gt"]
                label = f"synthetic:{kind}"
                rows.append({"video": Path(video).name, "error": "missing", "replaced_by": label})
                # Одна сцена заменяет все пропавшие ролики своего вида.
                if kind in made:
                    continue
                from _synthetic_clips import KINDS

                path = KINDS[kind](Path(folder)).path
                made.add(kind)
                source = f"synthetic, вместо {Path(video).name}"
                synthetic = True
            for pname in case["profiles"]:
                row = evaluate(str(path), pname, case["gt"])
                row["video"] = label
                row["source"] = source
                rows.append(row)
                prev = best_by_video.get(label)
                if prev is None or (row["hold"], row["trail_score"]) > (prev["hold"], prev["trail_score"]):
                    best_by_video[label] = row

    if not best_by_video:
        print("Ни одного ролика не оценено: " + ", ".join(row["video"] for row in rows if "error" in row))
        raise SystemExit(2)
    report = {
        "criteria": {"hold": HOLD_PASS, "trail": TRAIL_PASS, "err_diag_frac": ERR_DIAG_FRAC},
        "all_runs": rows,
        "best_per_video": list(best_by_video.values()),
    }
    if args.report is not None:
        path = args.report
    elif args.video is None and not synthetic:
        path = ROOT / "_eval_hold_trail_report.json"
    else:
        path = Path(tempfile.gettempdir()) / "peleng_eval_hold_trail.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 76)
    print("BEST per video (acceptance)")
    if synthetic:
        print("synthetic — проверка работоспособности, не точность на реальной съёмке")
    print("=" * 76)
    fails = 0
    for video, row in ((v, best_by_video[v]) for v in best_by_video):
        flag = "PASS" if row["PASS"] else "FAIL"
        if not row["PASS"]:
            fails += 1
        print(
            f"{flag}  {video}  [{row['profile']}/{row['gt']}]\n"
            f"      hold={row['hold']*100:.1f}%  trail={row['trail_score']*100:.1f}%  "
            f"overlap={row['trail_overlap']*100:.1f}%  corr={row['corr']}  "
            f"med_err={row['median_err_px']}px  gt_vis={row['gt_visible']}"
        )
    print("=" * 76)
    print(f"report: {path}")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
