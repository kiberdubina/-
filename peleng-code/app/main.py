import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog

import cv2
import numpy as np
from PIL import Image, ImageTk

from app.calib import BoardDetector
from app.profile import ColorGate, TargetProfile, load_board, load_patterns, load_profile, save_custom
from app.rig import Rig
from app.session import Session
from app import phone_usb
from app.source import BlankSource, SyntheticSource, VideoSource, find_phone_source
from app.export_result import export_session_sample
from app.view3d import SceneCanvas

ROOT = Path(__file__).resolve().parent.parent


def advance_zoom(
    state: dict[str, float],
    active: bool,
    cx: float,
    cy: float,
    box_w: float,
    width: float,
) -> dict[str, float]:
    """Центр и масштаб едут за целью. После пропажи держатся, потом масштаб спадает."""
    if active:
        state["miss"] = 0.0
        target_zoom = width / (3.0 * box_w) if box_w > 1.0 else 1.0
        target_zoom = float(min(4.0, max(1.0, target_zoom)))
        state["cx"] += 0.55 * (cx - state["cx"])
        state["cy"] += 0.55 * (cy - state["cy"])
        state["zoom"] += 0.55 * (target_zoom - state["zoom"])
    else:
        state["miss"] += 1.0
        if state["miss"] > 8.0:
            state["zoom"] += 0.12 * (1.0 - state["zoom"])
    state["zoom"] = float(min(4.0, max(1.0, state["zoom"])))
    return state


# Обычный размер кадра, когда слева хватает места и поле 3D не отъело его.
TILE_W = 400
TILE_H = 225
TILE_BG = (28, 28, 28)
# Ниже этого кадр уже не сжимается: поле 3D упирается в камеры.
TILE_MIN_W = 160
TILE_MIN_H = 90


def grid_shape(count: int) -> tuple[int, int]:
    """Сколько столбцов и строк занять слева. Две камеры стоят рядом."""
    if count <= 1:
        return 1, 1
    if count == 2:
        return 2, 1
    if count <= 4:
        return 2, 2
    cols = 3 if count <= 9 else int(count**0.5 + 0.999)
    rows = (count + cols - 1) // cols
    return cols, rows


def camera_layout(count: int, area_w: float, area_h: float) -> tuple[int, int, int]:
    """Размер окна камеры и зазор под прямоугольник слева от поля 3D.

    Поле больше — прямоугольник уже, кадры сжимаются. Поле меньше — кадры
    крупнее, но не вплотную: часть места остаётся свободной, зазор шире.
    """
    cols, rows = grid_shape(max(int(count), 1))
    area_w = max(float(area_w), 1.0)
    area_h = max(float(area_h), 1.0)
    pref_w = cols * (TILE_W + 16)
    pref_h = rows * (TILE_H + 16)
    roomy = area_w >= pref_w and area_h >= pref_h
    if roomy:
        slack_w = area_w * 0.14
        slack_h = area_h * 0.16
        gap = 16
    else:
        slack_w = area_w * 0.04
        slack_h = area_h * 0.04
        gap = 6
    use_w = max(1.0, area_w - slack_w)
    use_h = max(1.0, area_h - slack_h)
    scale = min(
        (use_w / cols - 2 * gap) / TILE_W,
        (use_h / rows - 2 * gap) / TILE_H,
    )
    scale = float(min(2.0, max(TILE_MIN_W / TILE_W, scale)))
    return int(TILE_W * scale), int(TILE_H * scale), gap


def fit_camera_tile(
    image: np.ndarray,
    tile_w: int = TILE_W,
    tile_h: int = TILE_H,
) -> np.ndarray:
    """Вписать кадр в прямоугольник постоянного размера. Края не растягивают окно."""
    height, width = image.shape[:2]
    scale = min(tile_w / max(width, 1), tile_h / max(height, 1))
    out_w = max(1, min(tile_w, int(width * scale)))
    out_h = max(1, min(tile_h, int(height * scale)))
    resized = image
    if out_w != width or out_h != height:
        resized = cv2.resize(
            image,
            (out_w, out_h),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
        )
    canvas = np.full((tile_h, tile_w, 3), TILE_BG, np.uint8)
    x = (tile_w - out_w) // 2
    y = (tile_h - out_h) // 2
    canvas[y : y + out_h, x : x + out_w] = resized
    return canvas


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Трекинг 3D")
        self.configure(bg="#1c1c1c")
        self.profile = load_profile(ROOT / "config.yaml")
        self.patterns = load_patterns(ROOT / "config.yaml")
        self.board = load_board(ROOT / "config.yaml")
        self.detector_board = BoardDetector(self.board)
        self.session = Session(self.profile, BlankSource())
        self.focus: str | None = None
        self.labels: dict[str, tk.Label] = {}
        self.photos: list[ImageTk.PhotoImage] = []
        self.trails: dict[str, list] = {}
        self.zoom_state: dict[str, dict[str, float]] = {}
        self.auto_phone = True
        self._phone_wait_text = "Подключаю камеру телефона…"
        self._scan_running = False
        self._last_tiles: dict[str, np.ndarray] = {}
        self._dismissed_phones: set[str] = set()
        self._scene_drag: tuple[int, int, int, int] | None = None

        body = tk.Frame(self, bg="#1c1c1c")
        body.pack(fill="both", expand=True)
        self.body = body
        self.cameras = tk.Frame(body, bg="#1c1c1c")
        self.scene = SceneCanvas(body)
        # Сначала поле 3D справа, остаток ширины — камерам. Рост поля забирает этот остаток.
        self.scene.pack(side="right", anchor="n", padx=(0, 8), pady=8)
        self.cameras.pack(side="left", fill="both", expand=True, padx=(8, 4), pady=8)
        self._scene_grip = tk.Frame(
            self.scene, width=16, height=16, bg="#5c6574", cursor="size_ne_sw"
        )
        self._scene_grip.place(relx=0.0, rely=1.0, anchor="sw", width=16, height=16)
        self._scene_grip.bind("<ButtonPress-1>", self._scene_grip_press)
        self._scene_grip.bind("<B1-Motion>", self._scene_grip_drag)
        self.status = tk.Label(
            self,
            text="",
            fg="#e8e8e8",
            bg="#1c1c1c",
            font=("Segoe UI", 11),
            justify=tk.LEFT,
            anchor="w",
        )
        self.status.pack(fill="x", padx=12, pady=(0, 12))
        self._menu()
        self.geometry("1440x820")
        self.minsize(960, 560)
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.after(50, self._open_phones_first)
        self.after(16, self.tick)

    def _menu(self) -> None:
        menu = tk.Menu(self)
        file_menu = tk.Menu(menu, tearoff=0)
        file_menu.add_command(label="Телефон", command=self.open_phone)
        file_menu.add_command(label="Демо", command=self.use_demo)
        file_menu.add_command(label="Демо 3D", command=self.use_rig)
        file_menu.add_command(label="Путь 5 камер", command=self.use_way1)
        file_menu.add_command(label="Открыть видео…", command=self.open_video)
        file_menu.add_command(label="Добавить видео…", command=self.add_video)
        file_menu.add_command(label="URL потока…", command=self.open_url)
        file_menu.add_command(label="Сохранить контроль…", command=self.save_control)
        file_menu.add_command(label="Отчёт погрешности…", command=self.save_metrics)
        file_menu.add_command(label="Контрольный прогон Way1…", command=self.run_way1_control)
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self.destroy)
        menu.add_cascade(label="Файл", menu=file_menu)

        target_menu = tk.Menu(menu, tearoff=0)
        target_menu.add_command(label="Всё движение", command=lambda: self.use_pattern("motion"))
        target_menu.add_command(label="Чёрная футболка", command=lambda: self.use_pattern("black_shirt"))
        target_menu.add_command(label="Зелёный мяч", command=lambda: self.use_pattern("green_ball"))
        target_menu.add_command(label="БПЛА", command=lambda: self.use_pattern("aircraft"))
        target_menu.add_command(label="Своя цель…", command=self.edit_custom)
        target_menu.add_separator()
        for horizon in (0.25, 0.5, 1.0):
            target_menu.add_command(
                label=f"Прогноз {horizon:.2f} с".replace(".", ","),
                command=lambda value=horizon: self.set_horizon(value),
            )
        menu.add_cascade(label="Цель", menu=target_menu)

        board_menu = tk.Menu(menu, tearoff=0)
        board_menu.add_command(label="Сохранить лист ChArUco…", command=self.save_board)
        board_menu.add_command(label="Снять кадр листа", command=self.snap_board)
        board_menu.add_command(label="Позы по листу", command=self.solve_board)
        menu.add_cascade(label="Калибровка", menu=board_menu)
        self.config(menu=menu)

    def save_control(self) -> None:
        folder = filedialog.askdirectory(
            title="Папка для контрольного результата",
            initialdir=str(ROOT / "reports"),
        )
        if not folder:
            return
        out_dir = Path(folder)
        try:
            info = export_session_sample(self.session, out_dir)
        except Exception as error:
            messagebox.showerror("Сохранить контроль", str(error))
            return
        parts = [f"Кадров: {info.get('frames', 0)}"]
        video = info.get("video")
        if video is not None:
            parts.append(f"Видео: {Path(video).name}")
        if info.get("csv") is not None:
            parts.append("xyz.csv")
        charts = info.get("charts") or []
        if charts:
            parts.append(f"графики: {len(charts)}")
        messagebox.showinfo("Сохранить контроль", "\n".join(parts))

    def save_metrics(self) -> None:
        """Сравнить след сессии с таблицей цели по номерам кадров (эталон локально, не в git)."""
        from app.metrics3d import compare_by_frame, load_truth_meters, write_metrics_report

        marked = list(self.session.trail_frames)
        if not marked:
            messagebox.showwarning(
                "Погрешность", "Нет точек 3D с номером кадра. Сначала прогон записи с XYZ."
            )
            return
        try:
            truth = load_truth_meters(ROOT)
        except Exception as error:
            messagebox.showerror("Погрешность", f"Эталон не прочитан: {error}")
            return
        frames = [frame for frame, _ in marked]
        est = np.stack([np.asarray(point, np.float64).reshape(3) for _, point in marked])
        stats = compare_by_frame(frames, est, truth)
        out = ROOT / "reports" / "metrics.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        write_metrics_report(stats, out)
        messagebox.showinfo(
            "Погрешность",
            f"Кадров: {stats['frames']} ({100 * stats['coverage']:.1f}% записи)\n"
            f"RMSE {stats.get('rmse_m', float('nan')):.3f} м\n"
            f"медиана {stats.get('median_m', float('nan')):.3f} м\n"
            f"файл {out.name}",
        )

    def run_way1_control(self) -> None:
        """Контрольный прогон записи Way1 целиком в фоне: трек, 3D-ящик, графики, видео."""
        if getattr(self, "_control_thread", None) is not None and self._control_thread.is_alive():
            messagebox.showinfo("Контрольный прогон", "Прогон уже идёт.")
            return
        from app.cine_table import track_meters
        from app.control_run import ControlConfig, run_control, way1_clips
        from app.vehicle import VehicleModel

        pattern = self.patterns.get("way1")
        if pattern is None or not pattern.vehicle:
            messagebox.showerror("Контрольный прогон", "В config.yaml у профиля way1 нет блока vehicle.")
            return
        try:
            clips = way1_clips(ROOT)
            truth = track_meters(ROOT)
        except Exception as error:
            messagebox.showerror("Контрольный прогон", f"Нет данных Way1: {error}")
            return
        out_dir = ROOT / "reports" / "way1_control"
        model = VehicleModel.from_config(pattern.vehicle)
        config = ControlConfig()
        config.fusion.range_m = (pattern.r_min, pattern.r_max)
        self._control_note = "Контрольный прогон: старт…"
        self._control_result = None

        def progress(stage: str, share: float) -> None:
            self._control_note = f"Контрольный прогон: {stage} {100 * share:.0f}%"

        def work() -> None:
            try:
                self._control_result = run_control(clips, model, out_dir, truth, config, progress)
            except Exception as error:  # отчёт в окно, поток не роняет программу
                self._control_result = error

        self._control_thread = threading.Thread(target=work, daemon=True)
        self._control_thread.start()
        self.after(500, self._watch_control)

    def _watch_control(self) -> None:
        thread = getattr(self, "_control_thread", None)
        if thread is not None and thread.is_alive():
            self._phone_wait_text = self._control_note
            self.after(500, self._watch_control)
            return
        self._phone_wait_text = ""
        result = self._control_result
        if isinstance(result, Exception):
            messagebox.showerror("Контрольный прогон", str(result))
            return
        lines = [f"Папка: {ROOT / 'reports' / 'way1_control'}"]
        if result is not None and result.stats:
            stats = result.stats
            lines += [
                f"Кадров с оценкой: {stats['frames']} ({100 * stats['coverage']:.1f}%)",
                f"RMSE {stats['rmse_m']:.3f} м, медиана {stats['median_m']:.3f} м",
                f"95% кадров точнее {stats['p95_m']:.3f} м, максимум {stats['max_m']:.3f} м",
            ]
        messagebox.showinfo("Контрольный прогон", "\n".join(lines))

    def open_phone(self) -> None:
        self._dismissed_phones.clear()
        self.status.configure(text="Подключаю камеру телефона…")
        self.update_idletasks()
        self.auto_phone = True
        self._attach_usb_phones(menu=True)

    def use_demo(self) -> None:
        self.auto_phone = False
        self.focus = None
        self.session.set_sources([SyntheticSource()])

    def use_rig(self) -> None:
        self.auto_phone = False
        self.focus = None
        pattern = self.patterns.get("green_ball")
        if pattern is not None:
            pattern.predict_s = self.profile.predict_s
            self.profile = pattern
            self.session.set_profile(pattern)
        self.session.set_rig(Rig())

    def use_way1(self) -> None:
        from app.way1_show import load_way1

        self.auto_phone = False
        self.focus = None
        pattern = self.patterns.get("way1")
        if pattern is not None:
            pattern.predict_s = self.profile.predict_s
            self.profile = pattern
            self.session.set_profile(pattern)
        try:
            sources, poses = load_way1(ROOT)
        except Exception as error:
            messagebox.showerror("Путь 5 камер", str(error))
            return
        self.session.set_sources(sources, poses)
        self.session.set_pose_source(ROOT)
        self.trails.clear()
        self.zoom_state.clear()
        self.status.configure(
            text=f"{len(sources)} камер. След 3D — только триангуляция по кадрам."
        )

    def open_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Видео",
            filetypes=[("Видео", "*.mp4 *.avi *.mkv *.mov"), ("Все файлы", "*.*")],
        )
        if not path:
            return
        self.auto_phone = False
        source = VideoSource(path, "file")
        if not source.is_open():
            messagebox.showerror("Трекинг 3D", "Файл не открылся.")
            return
        self.focus = None
        self.session.set_sources([source])

    def add_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Ещё одна камера",
            filetypes=[("Видео", "*.mp4 *.avi *.mkv *.mov"), ("Все файлы", "*.*")],
        )
        if not path:
            return
        source = VideoSource(path, f"cam-{len(self.session.pipelines) + 1}")
        if not source.is_open():
            messagebox.showerror("Трекинг 3D", "Файл не открылся.")
            return
        self.session.add_source(source)

    def open_url(self) -> None:
        url = simpledialog.askstring(
            "Поток с телефона",
            "Адрес картинки с телефона, своя сеть, не сеть зала.\nНапример http://192.168.42.129:8080/video",
            parent=self,
        )
        if not url:
            return
        replace = len(self.session.pipelines) == 1 and isinstance(
            self.session.pipelines[0].source, SyntheticSource
        )
        camera_id = "phone-1" if replace else f"phone-{len(self.session.pipelines) + 1}"
        source = VideoSource(url.strip(), camera_id)
        if not source.is_open():
            messagebox.showerror("Трекинг 3D", "Поток не открылся.")
            return
        self.focus = None
        if replace:
            self.session.set_sources([source])
        else:
            self.session.add_source(source)

    def use_pattern(self, name: str) -> None:
        if name not in self.patterns:
            return
        profile = self.patterns[name]
        profile.predict_s = self.profile.predict_s
        self.profile = profile
        self.session.set_profile(profile)
        save_custom(ROOT / "config.yaml", profile, profile.predict_s)

    def set_horizon(self, value: float) -> None:
        self.profile.predict_s = value
        self.session.profile.predict_s = value
        self.session.track3d.profile.predict_s = value
        for pipeline in self.session.pipelines:
            pipeline.profile.predict_s = value
            pipeline.tracker.profile.predict_s = value

    def edit_custom(self) -> None:
        current = self.patterns.get("custom", self.profile)
        dialog = tk.Toplevel(self)
        dialog.title("Своя цель")
        dialog.configure(bg="#1c1c1c")
        fields = {
            "s_min": tk.StringVar(value=str(int(current.s_min))),
            "s_max": tk.StringVar(value=str(int(current.s_max))),
            "u_min": tk.StringVar(value=str(int(current.u_min))),
            "u_max": tk.StringVar(value=str(int(current.u_max))),
        }
        labels = {
            "s_min": "Площадь от, px",
            "s_max": "Площадь до, px",
            "u_min": "Скорость от, px/с",
            "u_max": "Скорость до, px/с",
        }
        for row, key in enumerate(fields):
            tk.Label(dialog, text=labels[key], fg="#e8e8e8", bg="#1c1c1c").grid(
                row=row, column=0, sticky="w", padx=8, pady=4
            )
            tk.Entry(dialog, textvariable=fields[key], width=12).grid(
                row=row, column=1, padx=8, pady=4
            )
        use_color = tk.BooleanVar(value=current.color is not None)
        tk.Checkbutton(
            dialog,
            text="Зелёный цвет",
            variable=use_color,
            fg="#e8e8e8",
            bg="#1c1c1c",
            selectcolor="#333333",
        ).grid(row=4, column=0, columnspan=2, sticky="w", padx=8)

        def apply() -> None:
            try:
                numbers = {key: float(var.get().replace(",", ".")) for key, var in fields.items()}
            except ValueError:
                messagebox.showerror("Своя цель", "Нужны числа.", parent=dialog)
                return
            color = None
            if use_color.get():
                color = current.color or ColorGate(25, 100, 80, 40)
            profile = TargetProfile(
                "custom",
                numbers["s_min"],
                numbers["s_max"],
                numbers["u_min"],
                numbers["u_max"],
                current.r_min,
                current.r_max,
                current.v_min,
                current.v_max,
                current.d_m,
                self.profile.predict_s,
                color,
            )
            self.patterns["custom"] = profile
            self.profile = profile
            self.session.set_profile(profile)
            save_custom(ROOT / "config.yaml", profile, profile.predict_s)
            dialog.destroy()

        tk.Button(dialog, text="Применить", command=apply).grid(
            row=5, column=0, columnspan=2, pady=8
        )

    def save_board(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Лист ChArUco",
            defaultextension=".png",
            filetypes=[("PNG", "*.png")],
        )
        if not path:
            return
        image = self.detector_board.image()
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            messagebox.showerror("Калибровка", "Лист не сохранился.")
            return
        Path(path).write_bytes(encoded.tobytes())
        messagebox.showinfo(
            "Калибровка",
            f"Клетка {self.board.square_m * 1000:.0f} мм. После печати проверьте линейкой.",
        )

    def snap_board(self) -> None:
        if not self.session.pipelines:
            return
        frames = []
        for pipeline in self.session.pipelines:
            step = pipeline.step()
            if step is not None:
                frames.append(step.frame)
        found = self.session.snap_board(self.detector_board, frames)
        if not found:
            messagebox.showwarning("Калибровка", "Лист ни на одной камере не найден.")
            return
        messagebox.showinfo("Калибровка", "Кадр листа: " + ", ".join(found))

    def solve_board(self) -> None:
        solved = self.session.solve_poses()
        if not solved:
            messagebox.showwarning("Калибровка", "Сначала снимите кадр, где виден лист.")
            return
        messagebox.showinfo("Калибровка", "Позы записаны: " + ", ".join(solved))

    def toggle_focus(self, camera_id: str) -> None:
        self.focus = None if self.focus == camera_id else camera_id

    def _camera_popup(self, event: tk.Event, camera_id: str) -> None:
        """Правый клик по окну камеры: добавить другое видео или убрать это."""
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="Добавить видео…", command=self.add_video)
        menu.add_command(
            label="Удалить камеру",
            command=lambda camera_id=camera_id: self.remove_camera(camera_id),
        )
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def remove_camera(self, camera_id: str) -> None:
        pipeline = next(
            (
                item
                for item in self.session.pipelines
                if getattr(item.source, "camera_id", None) == camera_id
            ),
            None,
        )
        if pipeline is None:
            return
        serial = getattr(pipeline.source, "serial", None)
        if serial:
            self._dismissed_phones.add(str(serial))
        self.session.drop_pipeline(pipeline)
        self.session.poses.pop(camera_id, None)
        self.session.samples.pop(camera_id, None)
        self.zoom_state.pop(camera_id, None)
        self.trails.pop(camera_id, None)
        self._last_tiles.pop(camera_id, None)
        label = self.labels.pop(camera_id, None)
        if label is not None:
            label.destroy()
        if self.focus == camera_id:
            self.focus = None
        if self.session.pipelines:
            return
        self.focus = None
        self.session.set_sources([BlankSource()])
        self._forget_camera_images()
        self.scene.set_scene([], None, None, [], "Вторая камера даст метры")
        self.status.configure(text="Камер нет. Правый клик по окну камеры — добавить видео.")

    def _scene_grip_press(self, event: tk.Event) -> None:
        width = self.scene.winfo_width()
        height = self.scene.winfo_height()
        if width < 2 or height < 2:
            width = int(self.scene.cget("width"))
            height = int(self.scene.cget("height"))
        self._scene_drag = (event.x_root, event.y_root, width, height)

    def _scene_grip_drag(self, event: tk.Event) -> None:
        if self._scene_drag is None:
            return
        x0, y0, width, height = self._scene_drag
        new_w = width - (event.x_root - x0)
        new_h = height + (event.y_root - y0)
        self.update_idletasks()
        body_w = int(self.body.winfo_width())
        if body_w > 200:
            new_w = min(new_w, max(240, body_w - TILE_MIN_W - 24))
        self.scene.set_view_size(new_w, new_h)
        self.update_idletasks()
        need_h = int(self.scene.winfo_reqheight()) + int(self.status.winfo_reqheight()) + 36
        if int(self.winfo_height()) + 8 < need_h:
            self.geometry(f"{int(self.winfo_width())}x{need_h}")
            self.update_idletasks()
        self._relayout_cameras()

    def _relayout_cameras(self) -> None:
        """Пересчитать кадры по уже показанным картинкам, не дожидаясь следующего кадра."""
        if not self._last_tiles:
            return
        order = [camera_id for camera_id in self.labels if camera_id in self._last_tiles]
        if not order:
            order = list(self._last_tiles)
        tiles = [(camera_id, self._last_tiles[camera_id]) for camera_id in order]
        self.photos = []
        self._fit_tiles(tiles)

    def _open_phones_first(self) -> None:
        self.status.configure(text=self._phone_wait_text)
        self._schedule_phone_scan()

    def tick(self) -> None:
        alive = self.session.step()
        stuck = self._release_stuck_phone()
        if alive is not None and not stuck:
            self._draw(alive)
        if self._phone_wait_text:
            self.status.configure(text=self._phone_wait_text)
        self.after(self._tick_delay(), self.tick)

    def _phone_pipelines(self) -> list:
        return [
            pipeline
            for pipeline in self.session.pipelines
            if getattr(pipeline.source, "kind", None) == "phone"
        ]

    def _release_stuck_phone(self) -> bool:
        """Убирает телефон, чей источник уже закрыт. Короткий простой кадра его не снимает."""
        phones = self._phone_pipelines()
        if not phones:
            pipelines = self.session.pipelines
            source = pipelines[0].source if pipelines else None
            keep_wait = self.auto_phone and (
                getattr(source, "kind", None) == "blank" or isinstance(source, SyntheticSource)
            )
            if not keep_wait:
                self._phone_wait_text = ""
            return False
        stuck = [pipeline for pipeline in phones if not pipeline.source.is_open()]
        if not stuck:
            self._phone_wait_text = ""
            return False
        message = ""
        for pipeline in stuck:
            source = pipeline.source
            message = (
                getattr(source, "error", "")
                or message
                or "Нет нового кадра с телефона. Подключаю камеру снова."
            )
            camera_id = getattr(source, "camera_id", None)
            label = self.labels.pop(camera_id, None) if camera_id else None
            if label is not None:
                label.destroy()
            if self.focus == camera_id:
                self.focus = None
            self.session.drop_pipeline(pipeline)
        if self._phone_pipelines() or self.session.pipelines:
            self._phone_wait_text = ""
            return False
        self.auto_phone = True
        self._phone_wait_text = message
        self._forget_camera_images()
        self.focus = None
        self.trails.clear()
        self.session.set_sources([BlankSource()])
        return True

    def _forget_camera_images(self) -> None:
        self.photos = []
        for camera_id in list(self.labels):
            self.labels[camera_id].destroy()
            del self.labels[camera_id]

    def _tick_delay(self) -> int:
        for pipeline in self.session.pipelines:
            if getattr(pipeline.source, "kind", None) == "phone":
                return 30
        return 33

    def _schedule_phone_scan(self) -> None:
        """Ищет новый кабель в фоне. Открытие камеры окно не ждёт."""
        if self._scan_running or not self.auto_phone:
            return
        self._scan_running = True
        already = [pipeline.source for pipeline in self._phone_pipelines()]
        before = {
            source.serial: source.camera_id
            for source in already
            if getattr(source, "serial", None)
        }
        dismissed = set(self._dismissed_phones)
        threading.Thread(
            target=self._scan_phones, args=(already, before, dismissed), daemon=True
        ).start()

    def _scan_phones(self, already: list, before: dict, dismissed: set | None = None) -> None:
        opened: list = []
        error = ""
        dismissed = set(dismissed or ())
        try:
            serials = phone_usb._ready_serials()
            attached = {getattr(source, "serial", None) for source in already}
            pending = [
                serial
                for serial in serials
                if serial not in attached
                and serial not in phone_usb.skipped
                and serial not in dismissed
            ]
            if pending or (not already and serials and not dismissed.issuperset(serials)):
                opened = phone_usb.open_phones(already=already, skip=dismissed)
                error = phone_usb.last_error or ""
        except Exception as exc:
            error = str(exc)
        self._scan_running = False
        try:
            self.after(0, lambda: self._apply_opened(opened, before, error))
        except RuntimeError:
            for source in opened:
                close = getattr(source, "close", None)
                if close is not None:
                    close()

    def _apply_opened(self, opened: list, before: dict, error: str) -> None:
        phones = self._phone_pipelines()
        self._keep_renamed_phone(before, [pipeline.source for pipeline in phones])
        if opened:
            self._phone_wait_text = ""
            if phones:
                for source in opened:
                    self.session.add_source(source)
            else:
                self.focus = None
                self.trails.clear()
                self.session.set_sources(opened)
        elif not phones and error:
            self._phone_wait_text = error
            self.status.configure(text=error)
        if self.auto_phone:
            delay = 3000 if phones or opened else 2000
            self.after(delay, self._schedule_phone_scan)

    def _attach_usb_phones(self, menu: bool) -> None:
        """Меню только просит фоновый поиск. Уже открытый телефон не перезапускает."""
        del menu
        self._schedule_phone_scan()

    def _keep_renamed_phone(self, before: dict, sources: list) -> None:
        """Имя phone-1 / phone-2 сменилось вместе с порядком серийников. Порт тот же."""
        moved = {}
        for source in sources:
            serial = getattr(source, "serial", None)
            old = before.get(serial)
            new = getattr(source, "camera_id", None)
            if not old or not new or old == new:
                continue
            if self.focus == old:
                self.focus = new
            state = self.zoom_state.pop(old, None)
            if state is not None:
                moved[new] = state
        self.zoom_state.update(moved)

    def _camera_area(self, count: int) -> tuple[int, int, int]:
        """Прямоугольник, который остался камерам после поля 3D."""
        self.cameras.update_idletasks()
        pane_w = int(self.cameras.winfo_width())
        pane_h = int(self.cameras.winfo_height())
        if pane_w < 80 or pane_h < 80:
            return TILE_W, TILE_H, 8
        return camera_layout(count, pane_w, pane_h)

    def _fit_tiles(self, tiles: list[tuple[str, np.ndarray]]) -> None:
        """Кадры делят место слева от поля 3D. Рост поля их сжимает, уменьшение — увеличивает."""
        count = len(tiles)
        if count == 0:
            return
        cols, _rows = grid_shape(count)
        tile_w, tile_h, gap = self._camera_area(count)
        for index in range(8):
            self.cameras.grid_columnconfigure(index, weight=0)
            self.cameras.grid_rowconfigure(index, weight=0)
        shown: set[str] = set()
        for index, (camera_id, image) in enumerate(tiles):
            image = fit_camera_tile(image, tile_w, tile_h)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.photos.append(photo)
            label = self.labels.get(camera_id)
            if label is None:
                label = tk.Label(
                    self.cameras,
                    bg="#1c1c1c",
                    cursor="hand2",
                    bd=0,
                    highlightthickness=1,
                    highlightbackground="#333333",
                )
                label.bind(
                    "<Button-1>",
                    lambda _event, camera_id=camera_id: self.toggle_focus(camera_id),
                )
                label.bind(
                    "<Button-3>",
                    lambda event, camera_id=camera_id: self._camera_popup(event, camera_id),
                )
                self.labels[camera_id] = label
            label.configure(image=photo)
            label._photo = photo
            label.grid(
                row=index // cols,
                column=index % cols,
                padx=gap,
                pady=gap,
                sticky="nw",
            )
            shown.add(camera_id)
        for camera_id in list(self.labels):
            if camera_id not in shown:
                self.labels[camera_id].grid_forget()
                self.labels[camera_id].destroy()
                del self.labels[camera_id]
                self._last_tiles.pop(camera_id, None)

    def _draw(self, tick) -> None:
        visible = [
            step
            for step in tick.steps
            if self.focus is None or step.frame.camera_id == self.focus
        ]
        seen = {step.frame.camera_id for step in visible}
        held = {
            pipeline.source.camera_id
            for pipeline in self.session.pipelines
            if getattr(pipeline.source, "kind", None) == "phone"
            and pipeline.source.is_open()
            and getattr(pipeline.source, "camera_id", None)
            and (self.focus is None or pipeline.source.camera_id == self.focus)
        }
        tiles: list[tuple[str, np.ndarray]] = []
        for step in visible:
            image = self._annotate(step, getattr(tick, "box", None))
            image = self._zoom_frame(step.frame.camera_id, image, step.track)
            self._last_tiles[step.frame.camera_id] = image
            tiles.append((step.frame.camera_id, image))
        for camera_id in held - seen:
            image = self._last_tiles.get(camera_id)
            if image is not None:
                tiles.append((camera_id, image))
        self.photos = []
        self._fit_tiles(tiles)
        shown = {step.frame.camera_id for step in tick.steps}
        for pipeline in self.session.pipelines:
            if getattr(pipeline.source, "camera_id", None) not in shown:
                continue
            note = getattr(pipeline.source, "note_presented", None)
            if note is not None:
                note()
        self._draw_scene(tick)
        self._draw_status(tick)

    def _annotate(self, step, box: np.ndarray | None = None) -> np.ndarray:
        image = step.frame.image.copy()
        pose = self.session.poses.get(step.frame.camera_id)
        if box is not None and pose is not None:
            self._draw_box(image, box, pose, step.track.measured)
        track = step.track
        if track.bbox is not None:
            x, y, w, h = track.bbox
            color = (80, 220, 80) if track.in_gate else (0, 200, 255)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                image,
                str(self.session.target_id),
                (x, max(16, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
        if track.pred_bbox is not None:
            x, y, w, h = track.pred_bbox
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 220, 255), 1)
        trail = track.trail
        for index in range(1, len(trail)):
            x0, y0 = trail[index - 1]
            x1, y1 = trail[index]
            if np.hypot(x1 - x0, y1 - y0) > 140:
                continue
            cv2.line(image, (int(x0), int(y0)), (int(x1), int(y1)), (255, 180, 40), 2)
        if track.active:
            origin = (int(track.cx), int(track.cy))
            ahead = (int(track.pred_cx), int(track.pred_cy))
            cv2.line(image, origin, ahead, (255, 0, 255), 2)
            cv2.circle(image, origin, 4, (255, 255, 255), -1)
            cv2.circle(image, ahead, 4, (255, 0, 255), -1)
        return image

    @staticmethod
    def _draw_box(image: np.ndarray, box: np.ndarray, pose, measured: bool) -> None:
        """Каркас 3D-ящика цели в кадре камеры: зелёный — камера дала измерение."""
        from app.vehicle import BOX_EDGES, project_points

        pixels, depth = project_points(pose, np.asarray(box, np.float64))
        if float(depth.min()) <= 0.5 or not np.isfinite(pixels).all() or np.abs(pixels).max() > 1e5:
            return
        color = (60, 220, 60) if measured else (0, 200, 255)
        thickness = max(2, int(round(image.shape[1] / 640)))
        for a, b in BOX_EDGES:
            start = tuple(int(round(v)) for v in pixels[a])
            end = tuple(int(round(v)) for v in pixels[b])
            cv2.line(image, start, end, color, thickness, cv2.LINE_AA)

    def _zoom_frame(self, camera_id: str, image: np.ndarray, track) -> np.ndarray:
        """Crop the annotated frame only. Detection still sees the full image."""
        height, width = image.shape[:2]
        state = self.zoom_state.get(camera_id)
        if state is None:
            state = {"cx": width / 2.0, "cy": height / 2.0, "zoom": 1.0, "miss": 0.0}
            self.zoom_state[camera_id] = state

        if track.active:
            box_w = float(track.bbox[2]) if track.bbox is not None else 0.0
            advance_zoom(state, True, float(track.cx), float(track.cy), box_w, float(width))
        else:
            advance_zoom(state, False, state["cx"], state["cy"], 0.0, float(width))

        zoom = state["zoom"]
        if zoom <= 1.001:
            return image

        crop_w = width / zoom
        crop_h = height / zoom
        left = min(max(state["cx"] - crop_w / 2.0, 0.0), max(0.0, width - crop_w))
        top = min(max(state["cy"] - crop_h / 2.0, 0.0), max(0.0, height - crop_h))
        x0 = int(left)
        y0 = int(top)
        x1 = min(width, max(x0 + 1, int(round(left + crop_w))))
        y1 = min(height, max(y0 + 1, int(round(top + crop_h))))
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            return image
        return cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)

    def _draw_scene(self, tick) -> None:
        world = tick.world
        current = None if world.xyz is None else np.asarray(world.xyz, np.float64).reshape(3)
        if current is None and tick.trail:
            current = np.asarray(tick.trail[-1], np.float64).reshape(3)
        hint = "Вторая камера даст метры"
        if current is not None and world.active and world.xyz is not None:
            hint = (
                f"цель {self.session.target_id}   "
                f"X {world.xyz[0]:.2f}  Y {world.xyz[1]:.2f}  Z {world.xyz[2]:.2f} м   "
                f"{world.speed_m_s:.2f} м/с   прогноз {self.profile.predict_s:.2f} с"
            )
        elif current is not None:
            hint = (
                f"цель {self.session.target_id}   "
                f"X {current[0]:.2f}  Y {current[1]:.2f}  Z {current[2]:.2f} м   "
                f"пройдено {len(tick.trail)}"
            )
        elif self.session.poses:
            hint = "Камеры стоят. Цель ждут две из них одновременно."
        self.scene.set_scene(
            tick.trail,
            current,
            None if not world.active else world.pred,
            list(self.session.poses.values()),
            hint,
            tick.rays,
            getattr(tick, "box", None),
        )

    def _draw_status(self, tick) -> None:
        parts = []
        for step in tick.steps:
            detection = step.detection
            track = step.track
            area = f"{detection.area:.0f} px" if detection else "—"
            speed = f"{track.speed_px_s:.0f} px/с" if track.active else "—"
            parts.append(
                f"цель {self.session.target_id}  {step.frame.camera_id}  {track.state}  "
                f"площадь {area}  скорость {speed}"
            )
        world = tick.world
        if world.active and world.xyz is not None:
            parts.append(
                f"XYZ {world.xyz[0]:.2f} {world.xyz[1]:.2f} {world.xyz[2]:.2f} м   "
                f"дальность {np.linalg.norm(world.xyz):.2f} м   "
                f"{world.speed_m_s:.2f} м/с"
            )
            focus = self.focus or (tick.steps[0].frame.camera_id if tick.steps else None)
            relative = world.relative.get(focus) if focus else None
            if relative is not None:
                parts.append(
                    f"относительно {focus}: "
                    f"{relative[0]:.2f} {relative[1]:.2f} {relative[2]:.2f} м/с"
                )
        parts.append(
            f"паттерн {self.profile.name}   "
            f"площадь {self.profile.s_min:.0f}–{self.profile.s_max:.0f} px   "
            f"скорость {self.profile.u_min:.0f}–{self.profile.u_max:.0f} px/с   "
            f"прогноз {self.profile.predict_s:.2f} с"
        )
        for pipeline in self.session.pipelines:
            latency = getattr(pipeline.source, "latency_ms", None)
            if isinstance(latency, (int, float)) and phone_usb.believable_latency_ms(float(latency)):
                parts.append(f"задержка {int(round(latency))} мс")
        for note in phone_usb.skipped.values():
            if note and note not in parts:
                parts.append(note)
        self.status.configure(text="\n".join(parts))

    def destroy(self) -> None:
        for pipeline in self.session.pipelines:
            close = getattr(pipeline.source, "close", None)
            if close is not None:
                close()
        super().destroy()


def main() -> None:
    App().mainloop()


if __name__ == "__main__":
    main()
