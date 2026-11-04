"""Контрольный прогон Way1 из консоли: python run_control.py [папка_отчёта] [--passes N]

Ролики и таблицы ищутся в корне проекта и в подпапках (видео/, execl/ …).
Результат: reports/way1_control/ — trajectory.csv, metrics.txt, графики, видео.

Любая другая запись: python run_control.py --folder ПАПКА_С_РОЛИКАМИ
Только часть камер: --cameras cam-1,cam-3. Таблица точности при 1–5 камерах (серия прогонов
без видео и графиков): --camera-table N — до N наборов камер на каждое их число.
Ролики и таблицы поз с любыми именами (app/scene_folder.py), отчёт — reports/<папка>_control/.
"""

import argparse
import csv
import itertools
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from app.cine_table import track_meters  # noqa: E402
from app.control_run import ControlConfig, run_control, way1_clips  # noqa: E402
from app.profile import load_patterns  # noqa: E402
from app.vehicle import VehicleModel  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Контрольный прогон Way1")
    parser.add_argument("out", nargs="?", default=str(ROOT / "reports" / "way1_control"))
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--folder", help="папка с роликами и таблицами поз любой записи")
    parser.add_argument("--detector", help="веса детектора грузовика (.pt); по умолчанию — из профиля way1")
    parser.add_argument("--detector-mode", choices=("background", "detector", "both"),
                        help="источник пятен: фон, детектор или оба (по умолчанию — из профиля или фон)")
    parser.add_argument("--detector-conf", type=float, default=0.25)
    parser.add_argument("--cameras", help="только эти камеры, через запятую: cam-1,cam-3")
    parser.add_argument("--camera-table", type=int, metavar="N",
                        help="серия прогонов: до N наборов камер на каждое их число, итог — camera_table.csv")
    args = parser.parse_args()

    profile = load_patterns(ROOT / "config.yaml")["way1"]
    if not profile.vehicle:
        print("В профиле way1 нет блока vehicle (габарит цели).")
        return 1
    model = VehicleModel.from_config(profile.vehicle)
    if args.folder:
        from app.control_run import scene_clips
        from app.scene_folder import describe, scan_folder, scene_truth

        scene = scan_folder(Path(args.folder))
        print(describe(scene))
        clips = scene_clips(scene)
        if len(clips) < 2:
            print("Для 3D нужны минимум 2 ролика с таблицами поз.")
            return 1
        truth = scene_truth(scene)
        if args.out == parser.get_default("out"):
            args.out = str(ROOT / "reports" / f"{_safe(scene.root.name)}_control")
    else:
        clips = way1_clips(ROOT)
        truth = track_meters(ROOT)
    if args.cameras:
        wanted = [name.strip() for name in args.cameras.split(",") if name.strip()]
        missing = sorted(set(wanted) - {clip.camera_id for clip in clips})
        if missing:
            print(f"Нет камер: {', '.join(missing)}. Есть: {', '.join(clip.camera_id for clip in clips)}")
            return 1
        clips = [clip for clip in clips if clip.camera_id in wanted]
    config = ControlConfig(passes=args.passes, workers=args.workers)
    config.fusion.range_m = (profile.r_min, profile.r_max)
    from app.control_run import apply_detector

    config.detector_conf = args.detector_conf
    print(apply_detector(config, profile.vehicle, args.detector, args.detector_mode))
    started = time.perf_counter()
    last = [""]

    def progress(stage: str, share: float) -> None:
        if stage != last[0]:
            print(f"[{time.perf_counter() - started:6.1f} с] {stage}", flush=True)
            last[0] = stage

    if args.camera_table:
        if truth is None:
            print("Для таблицы нужна таблица пути цели.")
            return 1
        config.media = False
        return camera_table(clips, model, Path(args.out), truth, config, args.camera_table)

    result = run_control(clips, model, Path(args.out), truth, config, progress)
    print()
    if result.stats:
        stats = result.stats
        print(f"Кадров с оценкой: {stats['frames']} ({100 * stats['coverage']:.1f}% записи)")
        print(f"RMSE {stats['rmse_m']:.3f} м   медиана {stats['median_m']:.3f} м   "
              f"p95 {stats['p95_m']:.3f} м   максимум {stats['max_m']:.3f} м")
        print(f"по осям RMSE: X {stats['rmse_x']:.3f}  Y {stats['rmse_y']:.3f}  Z {stats['rmse_z']:.3f} м")
    for name, path in sorted(result.files.items()):
        print(f"{name}: {path}")
    print(f"Время: {time.perf_counter() - started:.0f} с")
    return 0


def camera_sets(names: list[str], limit: int) -> list[tuple[str, ...]]:
    """Наборы камер от всех к одной, на каждое число — не больше limit, равномерно по сочетаниям."""
    sets = []
    for size in range(len(names), 0, -1):
        combos = list(itertools.combinations(names, size))
        step = max(1, len(combos) // max(limit, 1))
        sets += combos[::step][:limit]
    return sets


def camera_table(clips, model, out: Path, truth, config, limit: int) -> int:
    """Прогоны с разными наборами камер и сводная таблица погрешности по их числу."""
    from app.control_run import CAMERA_TABLE_HEADER

    rows = []
    for names in camera_sets([clip.camera_id for clip in clips], limit):
        chosen = [clip for clip in clips if clip.camera_id in names]
        started = time.perf_counter()
        result = run_control(chosen, model, out / "_".join(names), truth, config, None)
        stats = result.stats or {}
        frames = int(stats.get("frames", 0))
        row = [len(names), " ".join(names), frames, f"{stats.get('coverage', 0.0):.4f}"]
        row += [f"{stats.get(key, float('nan')):.4f}" for key in ("rmse_m", "mean_m", "median_m", "p95_m", "max_m")]
        rows.append(row)
        print(f"{' '.join(names):32s} кадров {frames:5d}  покрытие {row[3]}  RMSE {row[4]}  MAE {row[5]}  "
              f"медиана {row[6]}  P95 {row[7]}  макс {row[8]}  ({time.perf_counter() - started:.0f} с)", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "camera_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["count", "set", "frames", "coverage", *CAMERA_TABLE_HEADER[3:]])
        writer.writerows(rows)
    print(f"Таблица: {out / 'camera_table.csv'}")
    return 0


def _safe(name: str) -> str:
    """Имя папки отчёта без символов, которые не любит Windows."""
    cleaned = "".join("_" if char in '<>:"/\\|?* ' else char for char in name).strip("._")
    return cleaned or "scene"


if __name__ == "__main__":
    raise SystemExit(main())
