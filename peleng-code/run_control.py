"""Контрольный прогон Way1 из консоли: python run_control.py [папка_отчёта] [--passes N]

Ролики и таблицы ищутся в корне проекта и в подпапках (видео/, execl/ …).
Результат: reports/way1_control/ — trajectory.csv, metrics.txt, графики, видео.

Любая другая запись: python run_control.py --folder ПАПКА_С_РОЛИКАМИ
Ролики и таблицы поз с любыми именами (app/scene_folder.py), отчёт — reports/<папка>_control/.
"""

import argparse
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


def _safe(name: str) -> str:
    """Имя папки отчёта без символов, которые не любит Windows."""
    cleaned = "".join("_" if char in '<>:"/\\|?* ' else char for char in name).strip("._")
    return cleaned or "scene"


if __name__ == "__main__":
    raise SystemExit(main())
