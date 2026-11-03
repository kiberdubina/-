"""Контрольный прогон Way1 из консоли: python run_control.py [папка_отчёта] [--passes N]

Ролики и таблицы ищутся в корне проекта и в подпапках (видео/, execl/ …).
Результат: reports/way1_control/ — trajectory.csv, metrics.txt, графики, видео.
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
    args = parser.parse_args()

    profile = load_patterns(ROOT / "config.yaml")["way1"]
    if not profile.vehicle:
        print("В профиле way1 нет блока vehicle (габарит цели).")
        return 1
    model = VehicleModel.from_config(profile.vehicle)
    clips = way1_clips(ROOT)
    truth = track_meters(ROOT)
    config = ControlConfig(passes=args.passes, workers=args.workers)
    config.fusion.range_m = (profile.r_min, profile.r_max)
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


if __name__ == "__main__":
    raise SystemExit(main())
