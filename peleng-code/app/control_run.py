"""Контрольный прогон записи: максимальная точность 3D-трека цели по всем камерам.

Запись обрабатывается целиком (это постобработка, как у траекторных измерений):
  1. Фон каждой камеры — медиана по выборке кадров записи
     (поворотная камера — медиана в ячейках сферической панорамы).
  2. Пятна переднего плана на каждом кадре.
  3. Слияние камер: CTRV + итерационный EKF по рамкам 3D-ящика, затем сглаживание RTS.
  4. Второй проход: медиана фона без пикселей цели по треку первого прохода,
     заново пятна и трек. Камера, сопровождающая цель, иначе «впитывает» её в фон.

Выход: trajectory.csv, графики XYZ(t) (с эталоном, если он есть), отчёт погрешности
по номерам кадров, видео каждой камеры с проекцией 3D-ящика, мозаика всех камер.
"""

from __future__ import annotations

import csv
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pickle import PicklingError
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from app.background import BackgroundConfig, Blob, RecordingBackground, foreground_blobs
from app.export_result import VideoSink
from app.geom import CameraPose
from app.metrics3d import compare_by_frame, frame_errors, write_metrics_report
from app.vehicle import (
    BOX_EDGES,
    PSI,
    THETA,
    V,
    VehicleModel,
    anchor_point,
    box_center,
    box_corners,
    project_points,
    wrap_angle,
)
from app.vehicle_fusion import CameraView, FusionConfig, VehicleFusion

Progress = Callable[[str, float], None]


@dataclass
class CameraClip:
    """Запись одной камеры и её поза на каждом кадре."""

    camera_id: str
    path: Path
    width: int
    height: int
    frames: int
    fps: float
    pose_at: Callable[[int], CameraPose]


@dataclass
class ControlConfig:
    sample_every: int = 8          # шаг выборки кадров для медианы фона
    passes: int = 2                # 1 — без уточнения фона по треку
    refine: int = 10               # предел итераций сглаживания (сопоставление по прошлой траектории)
    refine_first: int = 3          # в первом проходе траектория нужна только для масок фона
    refine_tol_m: float = 0.25     # итерации до сходимости: 95% точек сдвинулись меньше этого
    refine_relax: float = 0.5      # доля новой траектории в следующей направляющей
    exclude_pad_px: int = 30       # запас вокруг проекции ящика при исключении из фона, px
    video_width: int = 960         # ширина кадров в видео результата
    workers: int = 5               # потоков на камеры
    # Кадр дальше этого от ближайшего наблюдения цели (хоть одной камерой) — дыра, а не
    # точка: без наблюдений трек летит по инерции и уходит на десятки метров. None — все кадры.
    coast_frames: int | None = 15
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    # Детектор грузовика (YOLO, дообученный на Way1). Источник пятен для подгонки 3D-ящика:
    # «background» — фон записи (как раньше), «detector» — рамки детектора (не зависят от
    # движения камеры), «both» — пятна фона и рамки детектора вместе. Без весов — фон.
    detector_weights: str | None = None
    detector_mode: str = "background"
    detector_conf: float = 0.25
    detector_width: int = 1280
    fusion: FusionConfig = field(default_factory=FusionConfig)


@dataclass
class ControlResult:
    frames: list[int]
    states: dict[int, np.ndarray]
    filtered: dict[int, np.ndarray]
    observed: dict[int, dict[str, tuple]]
    stats: dict | None
    files: dict[str, Path]


# ---------------------------------------------------------------- источники


@dataclass(frozen=True)
class TablePose:
    """Поза камеры Way1 на кадре из таблицы object_transform. Переносится в другой процесс."""

    root: str
    camera_id: str
    width: int
    height: int

    def __call__(self, frame: int) -> CameraPose:
        from app.cine_table import pose_at

        return pose_at(Path(self.root), self.camera_id, frame, self.width, self.height)


def way1_clips(root: Path) -> list[CameraClip]:
    from app.way1_sources import video_paths

    clips = []
    for camera_id, path in video_paths(Path(root)).items():
        capture = cv2.VideoCapture(str(path))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
        capture.release()
        poses = TablePose(str(Path(root).resolve()), camera_id, width, height)
        clips.append(CameraClip(camera_id, path, width, height, count, fps, poses))
    return clips


def scene_clips(scene) -> list[CameraClip]:
    """Камеры любой папки (app.scene_folder.SceneFolder) с таблицами поз — для контрольного прогона."""
    from app.scene_folder import FilePose

    clips = []
    for camera in scene.cameras:
        if camera.table is None:
            continue
        poses = FilePose(str(camera.table), scene.frame_offset, camera.width, camera.height, camera.camera_id)
        clips.append(
            CameraClip(camera.camera_id, camera.video, camera.width, camera.height, camera.frames, camera.fps, poses)
        )
    return clips


def _frames(clip: CameraClip):
    capture = cv2.VideoCapture(str(clip.path))
    index = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok or image is None:
                return
            yield index, image
            index += 1
    finally:
        capture.release()


# ---------------------------------------------------------------- фон и пятна


def _box_mask(state: np.ndarray, model: VehicleModel, pose: CameraPose, pad_px: int) -> np.ndarray | None:
    pixels, depth = project_points(pose, box_corners(state, model))
    if float(depth.min()) <= 0.5:
        return None
    hull = cv2.convexHull(pixels.astype(np.float32)).astype(np.int32)
    mask = np.zeros((pose.height, pose.width), np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)
    if pad_px > 0:
        mask = cv2.dilate(mask, np.ones((2 * pad_px + 1, 2 * pad_px + 1), np.uint8))
    return mask > 0


def camera_blobs(
    clip: CameraClip,
    config: ControlConfig,
    model: VehicleModel,
    exclude: dict[int, np.ndarray] | None = None,
) -> dict[int, list[Blob]]:
    """Фон-медиана записи и пятна переднего плана на каждом кадре одной камеры."""
    mode = config.detector_mode if config.detector_weights else "background"
    if mode == "both" and clip_translates(clip):
        # Камера едет: медиана фона по записи размазана параллаксом — пятна только от детектора.
        mode = "detector"
    detected = detector_blobs(clip, config) if mode in ("detector", "both") else {}
    if mode == "detector":
        return detected
    background = RecordingBackground(clip.width, clip.height, config.background)
    for index, image in _frames(clip):
        if index % config.sample_every:
            continue
        pose = clip.pose_at(index)
        mask = None
        if exclude is not None and index in exclude:
            mask = _box_mask(exclude[index], model, pose, config.exclude_pad_px)
        background.add(image, pose, mask)
    background.build()
    blobs: dict[int, list[Blob]] = {}
    for index, image in _frames(clip):
        foreground = background.apply(image, clip.pose_at(index))
        blobs[index] = foreground_blobs(foreground, background.scale)
    for index, boxes in detected.items():
        blobs.setdefault(index, []).extend(boxes)
    return blobs


def clip_translates(clip: CameraClip, limit_m: float = 0.5, samples: int = 25) -> bool:
    """Камера записи сдвигается (а не только поворачивается) больше limit_m за запись."""
    centers = []
    for frame in np.linspace(0, max(0, clip.frames - 1), samples).astype(int):
        pose = clip.pose_at(int(frame))
        rotation, _ = cv2.Rodrigues(pose.rvec)
        centers.append((-rotation.T @ np.asarray(pose.tvec, np.float64).reshape(3, 1)).ravel())
    centers = np.asarray(centers)
    return bool(np.linalg.norm(centers - centers[0], axis=1).max() > limit_m)


def apply_detector(config: ControlConfig, vehicle: dict | None, weights: str | None = None,
                   mode: str | None = None) -> str:
    """Детектор грузовика для прогона: веса из аргумента или из блока vehicle профиля
    (ключи detector, detector_mode). Нет файла весов — только фон. Возвращает строку для отчёта."""
    vehicle = vehicle or {}
    weights = weights or vehicle.get("detector")
    if not weights:
        return "детектор грузовика не задан — пятна по фону"
    path = Path(weights)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path
    if not path.is_file():
        return f"нет весов детектора {path.name} — пятна по фону"
    config.detector_weights = str(path)
    config.detector_mode = mode or vehicle.get("detector_mode", "both")
    return f"детектор грузовика {path.name}, режим {config.detector_mode}"


def detector_blobs(clip: CameraClip, config: ControlConfig) -> dict[int, list[Blob]]:
    """Рамки детектора грузовика на каждом кадре камеры как пятна для подгонки ящика.

    Детекции одного ролика с одними весами считаются один раз: второй проход и повторный
    прогон берут их из кэша во временной папке.
    """
    import hashlib
    import tempfile

    from app.object_detector import ObjectDetector, parse_spec
    from app.profile import TargetProfile

    weights = Path(config.detector_weights or "")
    if not weights.is_absolute():
        weights = Path(__file__).resolve().parent.parent / weights
    video = Path(clip.path)
    key = "|".join(
        str(item)
        for item in (video.resolve(), video.stat().st_mtime_ns, weights.resolve(), weights.stat().st_mtime_ns,
                     config.detector_conf, config.detector_width)
    )
    cache = Path(tempfile.gettempdir()) / f"peleng_truck_{hashlib.sha1(key.encode()).hexdigest()[:16]}.json"
    if cache.is_file():
        stored = json.loads(cache.read_text(encoding="utf-8"))
        return {int(frame): [Blob(x, y, w, h, w * h) for x, y, w, h in boxes] for frame, boxes in stored.items()}
    spec = parse_spec({
        "mode": "object", "class": "truck", "model": str(weights), "conf": config.detector_conf,
        "det_every": 1, "det_width": config.detector_width, "roi_scale": 0,
    })
    profile = TargetProfile("truck_detector", 0.0, 1e12, 0.0, 1e9, 0.0, 1e9, 0.0, 1e9, None, 0.5, None)
    detector = ObjectDetector(profile, spec)
    found: dict[int, list[list[float]]] = {}
    for index, image in _frames(clip):
        found[index] = [[float(value) for value in item.bbox] for item in detector.detect(image)]
    cache.write_text(json.dumps(found), encoding="utf-8")
    return {frame: [Blob(x, y, w, h, w * h) for x, y, w, h in boxes] for frame, boxes in found.items()}


def all_blobs(
    clips: list[CameraClip],
    config: ControlConfig,
    model: VehicleModel,
    exclude: dict[int, np.ndarray] | None = None,
) -> dict[str, dict[int, list[Blob]]]:
    """Камеры обрабатываются параллельно: по процессу на камеру (numpy держит GIL),
    при отказе процессов — потоками, при workers=1 — по очереди."""
    workers = max(1, min(config.workers, len(clips)))
    if workers == 1:
        return {clip.camera_id: camera_blobs(clip, config, model, exclude) for clip in clips}
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                clip.camera_id: pool.submit(camera_blobs, clip, config, model, exclude) for clip in clips
            }
            return {camera_id: future.result() for camera_id, future in futures.items()}
    except (BrokenProcessPool, PicklingError, AttributeError, TypeError):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                clip.camera_id: pool.submit(camera_blobs, clip, config, model, exclude) for clip in clips
            }
            return {camera_id: future.result() for camera_id, future in futures.items()}


# ---------------------------------------------------------------- трек


def fuse(
    clips: list[CameraClip],
    blobs: dict[str, dict[int, list[Blob]]],
    model: VehicleModel,
    config: ControlConfig,
    progress: Progress | None = None,
    guide: dict[int, np.ndarray] | None = None,
) -> tuple[VehicleFusion, dict[int, dict]]:
    fusion = VehicleFusion(model, config.fusion)
    fusion.set_guide(guide)
    count = min(clip.frames for clip in clips)
    fps = clips[0].fps
    observed: dict[int, dict] = {}
    for frame in range(count):
        views = {
            clip.camera_id: CameraView(clip.pose_at(frame), blobs[clip.camera_id].get(frame, []))
            for clip in clips
        }
        view = fusion.step(frame / fps, views, key=frame)
        if view.active:
            observed[frame] = dict(view.observed)
        if progress is not None and frame % 25 == 0:
            progress("трек", frame / max(1, count))
    return fusion, observed


def blend_states(before: dict, after: dict, share: float) -> dict:
    """(1 − share)·before + share·after по кадрам; курс усредняется по окружности.
    Кадры, которые есть только в одной траектории, берутся из неё."""
    result = {}
    for key in set(before) | set(after):
        if key not in before:
            result[key] = after[key].copy()
            continue
        if key not in after:
            result[key] = before[key].copy()
            continue
        mixed = (1.0 - share) * before[key] + share * after[key]
        turn = wrap_angle(float(after[key][PSI]) - float(before[key][PSI]))
        mixed[PSI] = wrap_angle(float(before[key][PSI]) + share * turn)
        result[key] = mixed
    return result


def trajectory_change(model: VehicleModel, before: dict, after: dict) -> float:
    """95-й перцентиль сдвига опорной точки между двумя траекториями по общим кадрам, м.
    Кадры, которые добавились или пропали, сходимость не определяют: число итераций
    ограничено отдельно."""
    common = sorted(set(before) & set(after))
    if not common:
        return float("inf")
    shifts = [np.linalg.norm(anchor_point(after[f], model) - anchor_point(before[f], model)) for f in common]
    return float(np.percentile(shifts, 95))


def refine(
    clips: list[CameraClip],
    blobs: dict[str, dict[int, list[Blob]]],
    model: VehicleModel,
    config: ControlConfig,
    fusion: VehicleFusion,
    observed: dict[int, dict],
    progress: Progress | None = None,
    label: str = "",
    rounds: int | None = None,
) -> tuple[VehicleFusion, dict[int, dict]]:
    """Итерационное сглаживание до сходимости.

    Каждая итерация сопоставляет пятна не с прогнозом прямого фильтра, а со сглаженной
    траекторией прошлой итерации (она уже учла всю запись), затем заново фильтрует и
    сглаживает. Останов — когда траектория перестала меняться.
    """
    previous = fusion.smoothed_states()
    guide = previous
    for round_number in range(config.refine if rounds is None else rounds):
        fusion, observed = fuse(clips, blobs, model, config, None, guide)
        current = fusion.smoothed_states()
        change = trajectory_change(model, previous, current)
        # Нижняя релаксация: следующая направляющая — середина двух последних траекторий.
        # Дискретные решения «пятно взято / не взято» иначе раскачивают итерации в цикл.
        guide = blend_states(previous, current, config.refine_relax)
        previous = current
        if progress is not None:
            progress(f"{label}: сглаживание {round_number + 1}, сдвиг p95 {change:.3f} м", 0.0)
        if change < config.refine_tol_m:
            break
    return fusion, observed


# ---------------------------------------------------------------- прогон


def run_control(
    clips: list[CameraClip],
    model: VehicleModel,
    out_dir: Path,
    truth: np.ndarray | None = None,
    config: ControlConfig | None = None,
    progress: Progress | None = None,
) -> ControlResult:
    config = config or ControlConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def say(stage: str, share: float) -> None:
        if progress is not None:
            progress(stage, share)

    exclude = None
    fusion = None
    observed: dict[int, dict] = {}
    for number in range(max(1, config.passes)):
        say(f"проход {number + 1}: фон и пятна", 0.0)
        blobs = all_blobs(clips, config, model, exclude)
        say(f"проход {number + 1}: трек", 0.0)
        fusion, observed = fuse(clips, blobs, model, config, say)
        last = number == max(1, config.passes) - 1
        rounds = config.refine if last else min(config.refine, config.refine_first)
        fusion, observed = refine(
            clips, blobs, model, config, fusion, observed, say, f"проход {number + 1}", rounds
        )
        exclude = fusion.smoothed_states()
    assert fusion is not None
    states = fusion.smoothed_states()
    filtered = fusion.filtered_states()
    frames = supported_frames(sorted(states), observed, config.coast_frames)
    count = min(clip.frames for clip in clips)

    files: dict[str, Path] = {}
    files["csv"] = write_trajectory(
        out_dir / "trajectory.csv", frames, states, filtered, observed, model, clips[0].fps, truth
    )
    files["boxes2d"] = write_boxes2d(out_dir / "boxes2d.csv", clips, states, observed, model)
    stats = None
    if truth is not None and frames:
        anchors = np.stack([anchor_point(states[f], model) for f in frames])
        stats = compare_by_frame(frames, anchors, truth, total_frames=count)
        files["report"] = write_metrics_report(stats, out_dir / "metrics.txt")
        files["segments"] = write_segment_table(out_dir / "metrics_by_segment.csv", frames, anchors, truth)
        (out_dir / "metrics.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    from app.excel_export import control_workbook

    workbook = control_workbook(out_dir / "trajectory.xlsx", files, stats)
    if workbook is not None:
        files["xlsx"] = workbook
    say("графики", 0.0)
    files.update(write_charts(out_dir, frames, states, model, clips[0].fps, truth))
    say("видео", 0.0)
    files.update(write_videos(out_dir, clips, states, observed, model, config, say))
    say("готово", 1.0)
    return ControlResult(frames, states, filtered, observed, stats, files)


# ---------------------------------------------------------------- вывод


def supported_frames(frames: list[int], observed: dict[int, dict], coast: int | None) -> list[int]:
    """Кадры не дальше coast от ближайшего кадра, где цель видела хоть одна камера."""
    if coast is None:
        return list(frames)
    seen = np.array(sorted(frame for frame in frames if observed.get(frame)), np.int64)
    if not len(seen):
        return []
    keep = []
    for frame in frames:
        index = int(np.searchsorted(seen, frame))
        near = min(abs(int(seen[min(index, len(seen) - 1)]) - frame), abs(int(seen[max(index - 1, 0)]) - frame))
        if near <= coast:
            keep.append(frame)
    return keep


def write_trajectory(
    path: Path,
    frames: list[int],
    states: dict[int, np.ndarray],
    filtered: dict[int, np.ndarray],
    observed: dict[int, dict],
    model: VehicleModel,
    fps: float,
    truth: np.ndarray | None,
) -> Path:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = [
            "frame", "t_s", "x_m", "y_m", "z_m",
            "box_cx_m", "box_cy_m", "box_cz_m", "yaw_deg", "pitch_deg", "speed_m_s",
            "filtered_x_m", "filtered_y_m", "filtered_z_m", "cameras",
        ]
        if truth is not None:
            header += ["truth_x_m", "truth_y_m", "truth_z_m", "error_m"]
        writer.writerow(header)
        for frame in frames:
            state = states[frame]
            anchor = anchor_point(state, model)
            center = box_center(state, model)
            raw = anchor_point(filtered[frame], model) if frame in filtered else anchor
            row = [
                frame, f"{frame / fps:.4f}",
                *(f"{value:.4f}" for value in anchor),
                *(f"{value:.4f}" for value in center),
                f"{np.degrees(state[PSI]):.3f}", f"{np.degrees(state[THETA]):.3f}", f"{abs(state[V]):.4f}",
                *(f"{value:.4f}" for value in raw),
                " ".join(sorted(observed.get(frame, {}))),
            ]
            if truth is not None:
                if 0 <= frame < len(truth):
                    row += [*(f"{value:.4f}" for value in truth[frame]), f"{np.linalg.norm(anchor - truth[frame]):.4f}"]
                else:
                    row += ["", "", "", ""]
            writer.writerow(row)
    return Path(path)


def write_boxes2d(
    path: Path,
    clips: list[CameraClip],
    states: dict[int, np.ndarray],
    observed: dict[int, dict],
    model: VehicleModel,
) -> Path:
    """Рамки цели по кадрам и камерам: силуэт, вошедший в измерение, и проекция 3D-ящика.

    Проекция — рамка оболочки 8 углов ящика после сглаживания, обрезанная кадром.
    """
    from app.vehicle import predicted_sides

    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "frame", "camera",
            "silhouette_x", "silhouette_y", "silhouette_w", "silhouette_h",
            "box3d_x", "box3d_y", "box3d_w", "box3d_h",
        ])
        for frame in sorted(states):
            for clip in clips:
                sides = predicted_sides(states[frame], model, clip.pose_at(frame))
                seen = observed.get(frame, {}).get(clip.camera_id)
                if sides is None and seen is None:
                    continue
                row = [frame, clip.camera_id]
                row += ["", "", "", ""] if seen is None else [f"{value:.1f}" for value in seen]
                row += ["", "", "", ""] if sides is None else [
                    f"{sides[0]:.1f}", f"{sides[1]:.1f}", f"{sides[2] - sides[0]:.1f}", f"{sides[3] - sides[1]:.1f}"
                ]
                writer.writerow(row)
    return Path(path)


def write_segment_table(
    path: Path, frames: list[int], anchors: np.ndarray, truth: np.ndarray, size: int = 100
) -> Path:
    errors = frame_errors(frames, anchors, truth)
    frames_arr = np.asarray(frames)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["from_frame", "to_frame", "frames", "median_m", "rmse_m", "max_m"])
        for start in range(0, int(frames_arr.max()) + 1, size):
            pick = (frames_arr >= start) & (frames_arr < start + size) & np.isfinite(errors)
            if not pick.any():
                continue
            part = errors[pick]
            writer.writerow([
                start, start + size - 1, int(pick.sum()),
                f"{np.median(part):.4f}", f"{np.sqrt(np.mean(part**2)):.4f}", f"{part.max():.4f}",
            ])
    return Path(path)


def write_charts(
    out_dir: Path,
    frames: list[int],
    states: dict[int, np.ndarray],
    model: VehicleModel,
    fps: float,
    truth: np.ndarray | None,
) -> dict[str, Path]:
    if not frames:
        return {}
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = np.asarray(frames) / fps
    anchors = np.stack([anchor_point(states[f], model) for f in frames])
    files: dict[str, Path] = {}
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), dpi=110, sharex=True)
    for axis, (ax, name) in enumerate(zip(axes, ("X", "Y", "Z"))):
        if truth is not None:
            span = np.arange(len(truth))
            ax.plot(span / fps, truth[:, axis], color="#888888", linewidth=2.5, label="эталон")
        ax.plot(times, anchors[:, axis], color="#1f77b4", linewidth=1.3, label="оценка")
        ax.set_ylabel(f"{name}, м")
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("t, с")
    fig.tight_layout()
    files["chart_xyz"] = out_dir / "xyz_t.png"
    fig.savefig(files["chart_xyz"])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 7), dpi=110)
    if truth is not None:
        ax.plot(truth[:, 0], truth[:, 1], color="#888888", linewidth=3, label="эталон")
    ax.plot(anchors[:, 0], anchors[:, 1], color="#1f77b4", linewidth=1.2, label="оценка")
    ax.set_aspect("equal")
    ax.set_xlabel("X, м")
    ax.set_ylabel("Y, м")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    files["chart_plan"] = out_dir / "plan_xy.png"
    fig.savefig(files["chart_plan"])
    plt.close(fig)

    if truth is not None:
        errors = frame_errors(frames, anchors, truth)
        fig, ax = plt.subplots(figsize=(10, 3.2), dpi=110)
        ax.plot(times, errors, color="#d62728", linewidth=1.0)
        ax.set_xlabel("t, с")
        ax.set_ylabel("ошибка, м")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        files["chart_error"] = out_dir / "error_t.png"
        fig.savefig(files["chart_error"])
        plt.close(fig)
    return files


def draw_box(
    image: np.ndarray,
    state: np.ndarray,
    model: VehicleModel,
    pose: CameraPose,
    scale: float,
    color: tuple[int, int, int],
) -> None:
    """Каркас 3D-ящика и опорная точка в кадре (кадр уже уменьшен в scale раз)."""
    corners = box_corners(state, model)
    pixels, depth = project_points(pose, np.vstack([corners, anchor_point(state, model)]))
    if float(depth.min()) <= 0.5:
        return
    points = pixels * scale
    if not np.isfinite(points).all() or np.abs(points).max() > 1e5:
        return
    for a, b in BOX_EDGES:
        start = tuple(int(round(v)) for v in points[a])
        end = tuple(int(round(v)) for v in points[b])
        cv2.line(image, start, end, color, 2, cv2.LINE_AA)
    anchor = tuple(int(round(v)) for v in points[8])
    cv2.circle(image, anchor, 5, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(image, anchor, 5, (0, 0, 0), 1, cv2.LINE_AA)


def _label(image: np.ndarray, text: str, origin: tuple[int, int], size: float = 0.8) -> None:
    # Шрифты Hershey в OpenCV — только латиница: подписи на кадрах латиницей.
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), 2, cv2.LINE_AA)


def write_videos(
    out_dir: Path,
    clips: list[CameraClip],
    states: dict[int, np.ndarray],
    observed: dict[int, dict],
    model: VehicleModel,
    config: ControlConfig,
    progress: Progress | None = None,
) -> dict[str, Path]:
    files: dict[str, Path] = {}
    count = min(clip.frames for clip in clips)
    tile_w = 640
    readers = {clip.camera_id: _frames(clip) for clip in clips}
    sinks = {clip.camera_id: VideoSink(out_dir / f"control_{clip.camera_id}.mp4", clip.fps) for clip in clips}
    mosaic = VideoSink(out_dir / "control_all.mp4", clips[0].fps)
    for frame in range(count):
        tiles = []
        for clip in clips:
            item = next(readers[clip.camera_id], None)
            if item is None:
                continue
            _, image = item
            scale = config.video_width / float(clip.width)
            size = (config.video_width, int(round(clip.height * scale)))
            small = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            seen = observed.get(frame, {})
            box = seen.get(clip.camera_id)
            if box is not None:
                x, y, w, h = (value * scale for value in box)
                cv2.rectangle(small, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 255), 1)
            if frame in states:
                color = (60, 220, 60) if clip.camera_id in seen else (0, 200, 255)
                draw_box(small, states[frame], model, clip.pose_at(frame), scale, color)
            _label(small, f"{clip.camera_id}  frame {frame}", (12, 28))
            sinks[clip.camera_id].write(small)
            tiles.append(cv2.resize(small, (tile_w, int(round(small.shape[0] * tile_w / small.shape[1])))))
        if tiles:
            mosaic.write(_mosaic(tiles, 3, frame, states.get(frame), model))
        if progress is not None and frame % 25 == 0:
            progress("видео", frame / max(1, count))
    for camera_id, sink in sinks.items():
        try:
            files[f"video_{camera_id}"] = sink.close()
        except ValueError:
            pass
    try:
        files["video_all"] = mosaic.close()
    except ValueError:
        pass
    return files


def _mosaic(
    tiles: list[np.ndarray], columns: int, frame: int, state: np.ndarray | None, model: VehicleModel
) -> np.ndarray:
    tile_h, tile_w = tiles[0].shape[:2]
    rows = (len(tiles) + 1 + columns - 1) // columns
    canvas = np.zeros((rows * tile_h, columns * tile_w, 3), np.uint8)
    for index, tile in enumerate(tiles):
        r, c = divmod(index, columns)
        canvas[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w] = cv2.resize(tile, (tile_w, tile_h))
    r, c = divmod(len(tiles), columns)
    x0, y0 = c * tile_w + 20, r * tile_h + 44
    lines = [f"frame {frame}"]
    if state is not None:
        anchor = anchor_point(state, model)
        lines += [
            f"X {anchor[0]:9.2f} m",
            f"Y {anchor[1]:9.2f} m",
            f"Z {anchor[2]:9.2f} m",
            f"heading {np.degrees(state[PSI]):7.1f} deg",
            f"pitch {np.degrees(state[THETA]):7.1f} deg",
            f"speed {abs(state[V]):6.2f} m/s",
        ]
    else:
        lines.append("no target")
    for index, text in enumerate(lines):
        _label(canvas, text, (x0, y0 + 38 * index), 0.9)
    return canvas
