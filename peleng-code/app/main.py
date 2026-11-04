import dataclasses
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog

import cv2
import numpy as np
from PIL import Image, ImageTk

from app.calib import BoardDetector
from app.profile import (
    ColorGate,
    TargetProfile,
    load_board,
    load_patterns,
    load_profile,
    save_active,
    save_custom,
    save_pattern,
)
from app.rig import Rig
from app.session import Session
from app import phone_usb
from app.source import BlankSource, SyntheticSource, VideoSource, find_phone_source
from app.export_result import export_session_sample
from app.view3d import SceneCanvas

ROOT = Path(__file__).resolve().parent.parent

# Цвета окна: фон, панели, текст, приглушённый текст, акцент.
BG = "#1c1c1c"
PANEL = "#262626"
FG = "#e8e8e8"
MUTED = "#9a9a9a"
ACCENT = "#4fa3ff"
WARN = "#ffb84d"
INFO = "#8cc4ff"
GOOD = "#5cd65c"
UI_FONT = "Segoe UI" if sys.platform == "win32" else "Helvetica"

# Цели в меню и на панели. None — разделитель.
TARGETS: list[tuple[str, str] | None] = [
    ("motion", "Всё движение"),
    ("black_shirt", "Чёрная футболка"),
    ("green_ball", "Зелёный мяч"),
    ("aircraft", "БПЛА"),
    None,
    ("black_hoodie_person", "Человек в чёрной кофте"),
    ("red_car", "Красная машина"),
    ("truck_text", "Грузовик (текст)"),
    ("truck_detector", "Грузовик (детектор)"),
    None,
    ("traffic", "Транспорт"),
]
TARGET_NAMES = {key: label for key, label in (item for item in TARGETS if item)}
TARGET_NAMES.update({"custom": "Своя цель", "text_custom": "Своя цель текстом", "way1": "Машина (Way1)"})
HORIZONS = (0.25, 0.5, 1.0)

HELP_TEXT = """Как пользоваться

1. Источник — откуда кадры: телефон по USB, папка с роликами, видео или демо.
2. Цель — что искать. «Транспорт»: Shift+клик по машине в окне камеры.
3. Метры появляются, когда цель видят две камеры с известными позами
   (папка с таблицами поз, «Демо 3D» или лист ChArUco в «Калибровке»).

Мышь в окне камеры
  клик — показать только эту камеру (ещё клик — все)
  Shift+клик — следить за машиной
  правый клик — выбрать машину, добавить или убрать камеру

Клавиши
  Ctrl+O — открыть видео        Ctrl+Shift+O — папка с роликами
  Ctrl+S — сохранить контроль   Esc — все камеры / снять выбор
  Пробел — пауза                1 / 2 / 3 — прогноз 0,25 / 0,5 / 1 с
  F2 — подробности              F1 — эта справка

Рамка камеры: зелёная — цель ведётся, серая — цели нет, синяя — показана одна.
Поле 3D: угол слева внизу тянет размер."""


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


def tile_to_image(
    px: float,
    py: float,
    frame_w: int,
    frame_h: int,
    tile_w: int,
    tile_h: int,
    crop: tuple[int, int, int, int] | None = None,
) -> tuple[float, float] | None:
    """Точка окна камеры → пиксель полного кадра. Учитывает вписывание (fit_camera_tile)
    и приближение (crop — видимая часть кадра). Мимо картинки — None."""
    scale = min(tile_w / max(frame_w, 1), tile_h / max(frame_h, 1))
    out_w = max(1, min(tile_w, int(frame_w * scale)))
    out_h = max(1, min(tile_h, int(frame_h * scale)))
    left = (tile_w - out_w) // 2
    top = (tile_h - out_h) // 2
    u = (px - left) / out_w
    v = (py - top) / out_h
    if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
        return None
    x0, y0, x1, y1 = crop if crop is not None else (0, 0, frame_w, frame_h)
    return x0 + u * (x1 - x0), y0 + v * (y1 - y0)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Трекинг 3D")
        self.configure(bg=BG)
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
        # Ширина и высота показанного кадра и окна камеры: клик → пиксель кадра.
        self._tile_geom: dict[str, tuple[int, int, int, int]] = {}
        self.auto_phone = True
        self._phone_wait_text = "Подключаю камеру телефона…"
        self._scan_running = False
        self._last_tiles: dict[str, np.ndarray] = {}
        self._dismissed_phones: set[str] = set()
        self._scene_drag: tuple[int, int, int, int] | None = None
        # Короткое сообщение под сводкой: текст и момент, когда оно гаснет (мс).
        self._notice = ""
        self._notice_until = 0
        self.show_details = tk.BooleanVar(value=False)
        self._focus_shown: str | None = None
        self._target_shown = ""
        self.paused = False
        # На паузе: последний кадр и просьба перерисовать его (сменили камеру).
        self._last_tick = None
        self._redraw = False
        # Ведёт ли камера цель: от этого цвет рамки её окна.
        self._tile_active: dict[str, bool] = {}
        self.horizon_var = tk.DoubleVar(value=self.profile.predict_s)

        self._toolbar()
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)
        self.body = body
        self.cameras = tk.Frame(body, bg=BG)
        self.scene = SceneCanvas(body)
        # Сначала поле 3D справа, остаток ширины — камерам. Рост поля забирает этот остаток.
        self.scene.pack(side="right", anchor="n", padx=(0, 8), pady=8)
        self.cameras.pack(side="left", fill="both", expand=True, padx=(8, 4), pady=8)
        self._start_screen()
        # size_ne_sw есть только в Tk под Windows; на macOS и Linux — свой угловой курсор.
        grip_cursor = "size_ne_sw" if sys.platform == "win32" else "bottom_left_corner"
        self._scene_grip = tk.Frame(
            self.scene, width=16, height=16, bg="#5c6574", cursor=grip_cursor
        )
        self._scene_grip.place(relx=0.0, rely=1.0, anchor="sw", width=16, height=16)
        self._scene_grip.bind("<ButtonPress-1>", self._scene_grip_press)
        self._scene_grip.bind("<B1-Motion>", self._scene_grip_drag)
        self._info_panel()
        self._menu()
        self._keys()
        self.geometry("1440x820")
        self.minsize(960, 560)
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.after(50, self._open_phones_first)
        self.after(16, self.tick)

    # --- меню и панели -------------------------------------------------

    def _source_menu(self, parent: tk.Misc) -> tk.Menu:
        menu = tk.Menu(parent, tearoff=0)
        menu.add_command(label="Телефон по USB", command=self.open_phone)
        menu.add_command(label="Папка с роликами…", command=self.open_scene_folder, accelerator="Ctrl+Shift+O")
        menu.add_command(label="Видео…", command=self.open_video, accelerator="Ctrl+O")
        menu.add_command(label="Добавить ещё видео…", command=self.add_video)
        menu.add_command(label="Поток по адресу…", command=self.open_url)
        menu.add_separator()
        menu.add_command(label="Демо (одна камера)", command=self.use_demo)
        menu.add_command(label="Демо 3D (две камеры)", command=self.use_rig)
        menu.add_command(label="Путь 5 камер (Way1)", command=self.use_way1)
        return menu

    def _target_menu(self, parent: tk.Misc) -> tk.Menu:
        menu = tk.Menu(parent, tearoff=0)
        # Цели по описанию (вторая группа) — модель YOLO / YOLO-World (requirements-detect.txt).
        for item in TARGETS:
            if item is None:
                menu.add_separator()
                continue
            key, label = item
            menu.add_command(label=label, command=lambda key=key: self.use_pattern(key))
        menu.add_command(label="Снять выбор машины", command=self.clear_vehicle, accelerator="Esc")
        menu.add_separator()
        menu.add_command(label="Своя цель по размеру…", command=self.edit_custom)
        menu.add_command(label="Своя цель текстом…", command=self.edit_text_query)
        return menu

    def _save_menu(self, parent: tk.Misc) -> tk.Menu:
        menu = tk.Menu(parent, tearoff=0)
        menu.add_command(label="Контрольный результат…", command=self.save_control, accelerator="Ctrl+S")
        menu.add_command(label="Путь выбранной машины…", command=self.save_vehicle_path)
        menu.add_command(label="Отчёт погрешности", command=self.save_metrics)
        menu.add_separator()
        menu.add_command(label="Контрольный прогон Way1…", command=self.run_way1_control)
        menu.add_command(label="Контрольный прогон папки…", command=self.run_folder_control)
        return menu

    def _menu(self) -> None:
        menu = tk.Menu(self)
        menu.add_cascade(label="Источник", menu=self._source_menu(menu))
        menu.add_cascade(label="Цель", menu=self._target_menu(menu))

        board_menu = tk.Menu(menu, tearoff=0)
        board_menu.add_command(label="1. Сохранить лист ChArUco…", command=self.save_board)
        board_menu.add_command(label="2. Снять кадр листа", command=self.snap_board)
        board_menu.add_command(label="3. Позы по листу", command=self.solve_board)
        menu.add_cascade(label="Калибровка", menu=board_menu)

        menu.add_cascade(label="Сохранить", menu=self._save_menu(menu))

        view_menu = tk.Menu(menu, tearoff=0)
        view_menu.add_command(label="Все камеры", command=self.show_all_cameras, accelerator="Esc")
        view_menu.add_checkbutton(
            label="Подробности", variable=self.show_details, command=self._toggle_details, accelerator="F2"
        )
        view_menu.add_separator()
        view_menu.add_command(label="Справка", command=self.show_help, accelerator="F1")
        view_menu.add_separator()
        view_menu.add_command(label="Выход", command=self.destroy)
        menu.add_cascade(label="Вид", menu=view_menu)
        self.config(menu=menu)

    def _menu_button(self, parent: tk.Misc, text: str, build) -> tk.Menubutton:
        button = tk.Menubutton(
            parent,
            text=text,
            bg=PANEL,
            fg=FG,
            activebackground="#333333",
            activeforeground=FG,
            relief="flat",
            padx=10,
            pady=4,
            font=(UI_FONT, 11),
        )
        button.configure(menu=build(button))
        return button

    def _toolbar(self) -> None:
        """Верхняя полоса: откуда кадры, что ищем, прогноз, сохранить, справка."""
        bar = tk.Frame(self, bg=PANEL)
        bar.pack(fill="x")
        self._menu_button(bar, "Источник ▾", self._source_menu).pack(side="left", padx=(8, 2), pady=4)
        self.target_button = self._menu_button(bar, "", self._target_menu)
        self.target_button.pack(side="left", padx=2, pady=4)
        self._update_target_button()

        tk.Label(bar, text="Прогноз", bg=PANEL, fg=MUTED, font=(UI_FONT, 11)).pack(side="left", padx=(16, 4))
        for value in HORIZONS:
            tk.Radiobutton(
                bar,
                text=f"{value:g} с".replace(".", ","),
                value=value,
                variable=self.horizon_var,
                command=lambda value=value: self.set_horizon(value),
                indicatoron=False,
                bg=PANEL,
                fg=FG,
                selectcolor="#3a4a5e",
                activebackground="#333333",
                activeforeground=FG,
                relief="flat",
                offrelief="flat",
                padx=8,
                pady=2,
                font=(UI_FONT, 10),
            ).pack(side="left", padx=1)

        self.pause_button = tk.Button(
            bar,
            text="⏸ Пауза",
            command=self.toggle_pause,
            bg=PANEL,
            fg=FG,
            activebackground="#333333",
            relief="flat",
            font=(UI_FONT, 11),
        )
        self.pause_button.pack(side="left", padx=(16, 2), pady=4)

        tk.Button(
            bar,
            text="?",
            command=self.show_help,
            bg=PANEL,
            fg=FG,
            activebackground="#333333",
            relief="flat",
            width=2,
            font=(UI_FONT, 11, "bold"),
        ).pack(side="right", padx=(2, 8), pady=4)
        self._menu_button(bar, "Сохранить ▾", self._save_menu).pack(side="right", padx=2, pady=4)
        # Видна, только когда показана одна камера.
        self.all_button = tk.Button(
            bar,
            text="← Все камеры",
            command=self.show_all_cameras,
            bg=PANEL,
            fg=ACCENT,
            activebackground="#333333",
            relief="flat",
            font=(UI_FONT, 11),
        )

    def _info_panel(self) -> None:
        """Нижняя панель: крупно — где цель, ниже — камеры и подсказка."""
        panel = tk.Frame(self, bg=BG)
        panel.pack(fill="x", padx=12, pady=(0, 10))
        self.info = panel
        self.summary = tk.Label(panel, text="Нет камер — выберите «Источник» на панели сверху", fg=MUTED, bg=BG, font=(UI_FONT, 15, "bold"), anchor="w")
        self.summary.pack(fill="x")
        self.cams_line = tk.Label(panel, text="", fg=MUTED, bg=BG, font=(UI_FONT, 11), anchor="w")
        self.cams_line.pack(fill="x")
        self.status = tk.Label(panel, text="", fg=WARN, bg=BG, font=(UI_FONT, 11), anchor="w", justify=tk.LEFT)
        self.status.pack(fill="x")
        self.details = tk.Label(
            panel, text="", fg=MUTED, bg=BG, font=(UI_FONT, 10), anchor="w", justify=tk.LEFT
        )

    def _keys(self) -> None:
        self.bind("<F1>", lambda _e: self.show_help())
        self.bind("<F2>", lambda _e: (self.show_details.set(not self.show_details.get()), self._toggle_details()))
        self.bind("<Escape>", lambda _e: self._escape())
        self.bind("<Control-o>", lambda _e: self.open_video())
        self.bind("<Control-O>", lambda _e: self.open_scene_folder())
        self.bind("<Control-s>", lambda _e: self.save_control())
        self.bind("<space>", lambda _e: self.toggle_pause())
        for key, value in zip("123", HORIZONS):
            self.bind(key, lambda _e, value=value: self.set_horizon(value))

    def _escape(self) -> None:
        if self.focus is not None:
            self.show_all_cameras()
        elif self.session.selection_camera is not None:
            self.clear_vehicle()

    def _toggle_details(self) -> None:
        if self.show_details.get():
            self.details.pack(fill="x", pady=(4, 0))
        else:
            self.details.pack_forget()

    def show_help(self) -> None:
        messagebox.showinfo("Справка", HELP_TEXT, parent=self)

    def show_all_cameras(self) -> None:
        self.focus = None
        self._update_focus_button()
        self._redraw = True

    def _update_focus_button(self) -> None:
        if self.focus is None:
            self.all_button.pack_forget()
        else:
            self.all_button.configure(text=f"← Все камеры (сейчас {self.focus})")
            self.all_button.pack(side="right", padx=8, pady=4)

    def _update_target_button(self) -> None:
        name = TARGET_NAMES.get(self.profile.name, self.profile.name)
        self.target_button.configure(text=f"Цель: {name} ▾")

    def _start_screen(self) -> None:
        """Пока камер нет — крупные кнопки «с чего начать» вместо чёрного окна."""
        screen = tk.Frame(self.cameras, bg=BG)
        self.start = screen
        tk.Label(screen, text="С чего начать", fg=FG, bg=BG, font=(UI_FONT, 20, "bold")).pack(pady=(0, 4))
        tk.Label(
            screen,
            text="Телефон по USB подключится сам. Или выберите запись:",
            fg=MUTED,
            bg=BG,
            font=(UI_FONT, 12),
        ).pack(pady=(0, 16))
        choices = [
            ("Папка с роликами", "все видео и таблицы поз — сразу 3D", self.open_scene_folder),
            ("Видео", "один файл, только 2D", self.open_video),
            ("Демо 3D", "две виртуальные камеры, метры сразу", self.use_rig),
            ("Телефон по USB", "искать снова", self.open_phone),
        ]
        for title, hint, command in choices:
            row = tk.Frame(screen, bg=PANEL, cursor="hand2")
            row.pack(fill="x", pady=4)
            head = tk.Label(row, text=title, fg=FG, bg=PANEL, font=(UI_FONT, 14, "bold"), anchor="w")
            head.pack(fill="x", padx=16, pady=(10, 0))
            note = tk.Label(row, text=hint, fg=MUTED, bg=PANEL, font=(UI_FONT, 11), anchor="w")
            note.pack(fill="x", padx=16, pady=(0, 10))
            for widget in (row, head, note):
                widget.bind("<Button-1>", lambda _e, command=command: command())
                widget.bind("<Enter>", lambda _e, row=row: self._hover(row, "#323232"))
                widget.bind("<Leave>", lambda _e, row=row: self._hover(row, PANEL))
        tk.Label(screen, text="F1 — справка", fg=MUTED, bg=BG, font=(UI_FONT, 11)).pack(pady=(16, 0))
        self._start_shown = False

    @staticmethod
    def _hover(row: tk.Frame, color: str) -> None:
        row.configure(bg=color)
        for child in row.winfo_children():
            child.configure(bg=color)

    def _update_start_screen(self) -> None:
        empty = all(getattr(p.source, "kind", None) == "blank" for p in self.session.pipelines)
        if empty == self._start_shown:
            return
        self._start_shown = empty
        if empty:
            self.start.place(relx=0.5, rely=0.45, anchor="center", width=420)
            self.start.lift()
        else:
            self.start.place_forget()

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        self.pause_button.configure(text="▶ Дальше" if self.paused else "⏸ Пауза")
        self.notify("Пауза — пробел, чтобы продолжить." if self.paused else "Продолжаю.", 3)

    def notify(self, text: str, seconds: float = 8.0, warn: bool = False) -> None:
        """Сообщение под сводкой. Держится seconds, кадры его не затирают."""
        self._notice = text
        self._notice_until = int(self.tk.call("clock", "milliseconds")) + int(seconds * 1000)
        self.status.configure(text=text, fg=WARN if warn else INFO)

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
        if info.get("xlsx") is not None:
            parts.append("Excel: xyz.xlsx")
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
        scene = getattr(self, "scene_folder", None)
        try:
            if self.session.pose_tables and scene is not None:
                # Открыта папка любой записи: эталон — её таблица пути цели.
                from app.scene_folder import scene_truth

                truth = scene_truth(scene)
                if truth is None:
                    raise ValueError("в папке нет таблицы пути цели")
            else:
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
        from app.control_run import ControlConfig, way1_clips
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
        from app.control_run import apply_detector

        self._control_detector = apply_detector(config, pattern.vehicle)
        self._launch_control(clips, model, config, truth, out_dir)

    def run_folder_control(self) -> None:
        """Контрольный прогон любой записи: ролики и таблицы поз с любыми именами, в фоне."""
        if getattr(self, "_control_thread", None) is not None and self._control_thread.is_alive():
            messagebox.showinfo("Контрольный прогон", "Прогон уже идёт.")
            return
        from app.control_run import ControlConfig, scene_clips
        from app.scene_folder import describe, safe_name, scan_folder, scene_truth
        from app.vehicle import VehicleModel

        pattern = self.patterns.get("way1")
        if pattern is None or not pattern.vehicle:
            messagebox.showerror("Контрольный прогон", "В config.yaml у профиля way1 нет блока vehicle.")
            return
        opened = getattr(self, "scene_folder", None)
        folder = filedialog.askdirectory(
            title="Папка с роликами и таблицами поз",
            initialdir=str(opened.root) if opened is not None else str(ROOT),
        )
        if not folder:
            return
        try:
            scene = scan_folder(Path(folder))
            clips = scene_clips(scene)
        except Exception as error:
            messagebox.showerror("Контрольный прогон папки", f"Папка не разобралась: {error}")
            return
        if len(clips) < 2:
            messagebox.showerror(
                "Контрольный прогон папки",
                "Для 3D нужны минимум 2 ролика с таблицами поз.\n\n" + describe(scene),
            )
            return
        config = ControlConfig()
        config.fusion.range_m = (pattern.r_min, pattern.r_max)
        from app.control_run import apply_detector

        self._control_detector = apply_detector(config, pattern.vehicle)
        out_dir = ROOT / "reports" / f"{safe_name(scene.root.name)}_control"
        self.notify(describe(scene))
        self._launch_control(clips, VehicleModel.from_config(pattern.vehicle), config, scene_truth(scene), out_dir)

    def _launch_control(self, clips, model, config, truth, out_dir: Path) -> None:
        """Контрольный прогон в фоновом потоке; итог — окном из _watch_control."""
        from app.control_run import run_control

        self._control_out = out_dir
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
        lines = [f"Папка: {getattr(self, '_control_out', ROOT / 'reports' / 'way1_control')}"]
        if getattr(self, "_control_detector", None):
            lines.append(self._control_detector)
        if result is not None and result.stats:
            stats = result.stats
            lines += [
                f"Кадров с оценкой: {stats['frames']} ({100 * stats['coverage']:.1f}%)",
                f"RMSE {stats['rmse_m']:.3f} м, медиана {stats['median_m']:.3f} м",
                f"95% кадров точнее {stats['p95_m']:.3f} м, максимум {stats['max_m']:.3f} м",
            ]
        if result is not None and result.files.get("xlsx") is not None:
            lines.append(f"Excel: {Path(result.files['xlsx']).name}")
        messagebox.showinfo("Контрольный прогон", "\n".join(lines))

    def open_phone(self) -> None:
        self._dismissed_phones.clear()
        self.notify("Подключаю камеру телефона…")
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
        self.notify(f"{len(sources)} камер. След 3D — только триангуляция по кадрам.")

    def open_scene_folder(self) -> None:
        """Ролики любой записи: все видео папки, таблицы поз по именам. Позы есть — 3D, нет — 2D."""
        from app.scene_folder import describe, scan_folder, scene_poses, video_motion_px

        folder = filedialog.askdirectory(title="Папка с роликами (и таблицами поз, если есть)")
        if not folder:
            return
        try:
            scene = scan_folder(Path(folder))
        except Exception as error:
            messagebox.showerror("Папка с роликами", f"Папка не разобралась: {error}")
            return
        if not scene.cameras:
            messagebox.showerror("Папка с роликами", describe(scene))
            return
        sources = []
        for camera in scene.cameras:
            source = VideoSource(str(camera.video), camera.camera_id)
            if source.is_open():
                sources.append(source)
            else:
                source.close()
        if not sources:
            messagebox.showerror("Папка с роликами", "Ни один ролик не открылся.")
            return
        self.auto_phone = False
        self.focus = None
        tables = scene_poses(scene)
        if len(tables) >= 2:
            # Позы двух и более камер: цель — машина, 3D-ящик по рамкам, как Way1.
            pattern = self.patterns.get("way1")
            if pattern is not None:
                pattern.predict_s = self.profile.predict_s
                self.profile = pattern
                self.session.set_profile(pattern)
            poses = {camera_id: table(0) for camera_id, table in tables.items()}
            self.session.set_sources(sources, poses)
            moving = {camera.camera_id for camera in scene.cameras if camera.moving or camera.turning}
            self.session.set_pose_tables(tables, moving)
        else:
            # Без поз — 2D. Картинка гуляет (камера в руке или едет) — детектор с компенсацией движения.
            shaky = any(video_motion_px(camera.video) > 1.5 for camera in scene.cameras)
            if shaky and not self.profile.moving_camera:
                profile = dataclasses.replace(self.profile, moving_camera=True)
                self.profile = profile
                self.session.set_profile(profile)
            self.session.set_sources(sources)
        # Не self.scene: это холст поля 3D (SceneCanvas).
        self.scene_folder = scene
        self.trails.clear()
        self.zoom_state.clear()
        self.notify(describe(scene))
        if any(word in note for note in scene.notes for word in ("проверьте", "не прочиталась", "не открылся")):
            messagebox.showwarning("Папка с роликами", describe(scene))

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
            self.notify(f"Цели «{TARGET_NAMES.get(name, name)}» нет в config.yaml.", warn=True)
            return
        profile = self.patterns[name]
        profile.predict_s = self.profile.predict_s
        self.profile = profile
        self.session.set_profile(profile)
        # Только выбор: профиль в config.yaml не переписывается и не теряет поля.
        save_active(ROOT / "config.yaml", name, profile.predict_s)
        self._update_target_button()
        if name == "traffic":
            self.notify("Транспорт: Shift+клик по машине в окне камеры — следить за ней.")

    def edit_text_query(self) -> None:
        current = self.patterns.get("text_custom")
        start = ""
        if current is not None and current.detect is not None:
            query = current.detect.get("query") or ""
            start = query if isinstance(query, str) else ", ".join(query)
        text = simpledialog.askstring(
            "Своя цель текстом",
            "Что искать, например «красная машина» или «person in black hoodie».\n"
            "Несколько вариантов — через запятую.",
            initialvalue=start,
            parent=self,
        )
        if not text or not text.strip():
            return
        phrases = [part.strip() for part in text.split(",") if part.strip()]
        fields = {"mode": "text", "query": phrases[0] if len(phrases) == 1 else phrases, "conf": 0.25}
        save_pattern(ROOT / "config.yaml", "text_custom", fields, self.profile.predict_s)
        self.patterns = load_patterns(ROOT / "config.yaml")
        self.use_pattern("text_custom")

    def set_horizon(self, value: float) -> None:
        self.horizon_var.set(value)
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
        dialog.configure(bg=BG)
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
            tk.Label(dialog, text=labels[key], fg=FG, bg=BG).grid(
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
            fg=FG,
            bg=BG,
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
        self._update_focus_button()
        self._redraw = True

    def _camera_popup(self, event: tk.Event, camera_id: str) -> None:
        """Правый клик по окну камеры: добавить другое видео или убрать это."""
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(
            label="Следить за этой машиной",
            command=lambda x=event.x, y=event.y: self.select_vehicle(camera_id, x, y),
        )
        menu.add_command(label="Снять выбор машины", command=self.clear_vehicle)
        menu.add_separator()
        menu.add_command(label="Добавить видео…", command=self.add_video)
        menu.add_command(
            label="Удалить камеру",
            command=lambda camera_id=camera_id: self.remove_camera(camera_id),
        )
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def select_vehicle(self, camera_id: str, px: float, py: float) -> None:
        """Shift+клик по машине в окне камеры: следить за ней на всех камерах."""
        if self.session.vehicle is not None and len(self.session.poses) >= 2:
            self.notify("Режим 3D-ящика ведёт одну цель сам. Для выбора машины — «Цель → Транспорт».", warn=True)
            return
        geom = self._tile_geom.get(camera_id)
        if geom is None:
            return
        crop = self.zoom_state.get(camera_id, {}).get("crop")
        point = tile_to_image(px, py, *geom, crop=crop)
        if point is None or not self.session.select_target(camera_id, *point):
            self.notify("Здесь машины нет — Shift+клик прямо по рамке машины.", warn=True)
            return
        self.notify(f"Выбрана машина на {camera_id}: на других камерах она найдётся по геометрии.")

    def clear_vehicle(self) -> None:
        self.session.clear_target()
        self.notify("Выбор машины снят.")

    def save_vehicle_path(self) -> None:
        """Путь выбранной машины (или главной цели) по кадрам: CSV и Excel."""
        from app.excel_export import table_from_csv, write_workbook

        marked = list(self.session.trail_frames)
        if not marked:
            messagebox.showwarning("Путь машины", "Пути 3D ещё нет: нужна машина, которую видят две камеры с позами.")
            return
        path = filedialog.asksaveasfilename(
            title="Путь выбранной машины",
            initialdir=str(ROOT / "reports"),
            initialfile=f"path_target_{self.session.target_id}.csv",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
        )
        if not path:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = ["frame,x_m,y_m,z_m,step_m"]
        previous = None
        for frame, point in marked:
            point = np.asarray(point, np.float64).reshape(3)
            step = "" if previous is None else f"{np.linalg.norm(point - previous):.4f}"
            rows.append(f"{frame},{point[0]:.4f},{point[1]:.4f},{point[2]:.4f},{step}")
            previous = point
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        book = write_workbook(path.with_suffix(".xlsx"), [table_from_csv("Путь машины", path)])
        names = [path.name] + ([book.name] if book is not None else [])
        messagebox.showinfo("Путь машины", f"Точек: {len(marked)}\n" + "\n".join(names))

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
        self.notify("Камер нет. Правый клик по окну камеры — добавить видео.")

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
        need_h = int(self.scene.winfo_reqheight()) + int(self.info.winfo_reqheight()) + 36
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
        self._show_wait()
        self._schedule_phone_scan()

    def tick(self) -> None:
        # На паузе кадры не берутся: видео стоит, картинка и след замирают.
        alive = None if self.paused else self.session.step()
        stuck = self._release_stuck_phone()
        self._update_start_screen()
        if alive is not None and not stuck:
            self._last_tick = alive
            self._draw(alive)
        elif self.paused and self._redraw and self._last_tick is not None:
            self._draw(self._last_tick)
        self._redraw = False
        if self._phone_wait_text:
            self._show_wait()
        elif self._notice and int(self.tk.call("clock", "milliseconds")) > self._notice_until:
            self._notice = ""
            self.status.configure(text="")
        self.after(self._tick_delay(), self.tick)

    def _show_wait(self) -> None:
        """Ожидание телефона или ход контрольного прогона — пока не кончится."""
        self.status.configure(text=self._phone_wait_text or self._notice, fg=WARN)

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
            self.notify(error, warn=True)
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
            self._tile_geom[camera_id] = (image.shape[1], image.shape[0], tile_w, tile_h)
            image = fit_camera_tile(image, tile_w, tile_h)
            # Имя камеры в углу окна, поверх уже вписанного кадра — всегда одного размера.
            cv2.rectangle(image, (0, 0), (12 + 9 * len(camera_id), 22), (20, 20, 20), -1)
            cv2.putText(image, camera_id, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.photos.append(photo)
            label = self.labels.get(camera_id)
            if label is None:
                label = tk.Label(
                    self.cameras,
                    bg=BG,
                    cursor="hand2",
                    bd=0,
                    highlightthickness=2,
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
                # Shift+клик по машине — следить за ней (обычный клик остаётся «только эта камера»).
                label.bind(
                    "<Shift-Button-1>",
                    lambda event, camera_id=camera_id: self.select_vehicle(camera_id, event.x, event.y),
                )
                self.labels[camera_id] = label
            if self.focus == camera_id:
                border = ACCENT
            elif self._tile_active.get(camera_id):
                border = GOOD
            else:
                border = "#333333"
            label.configure(image=photo, highlightbackground=border)
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
            self._tile_active[step.frame.camera_id] = bool(step.track.active)
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
        for label in step.labels or []:
            x, y, w, h = label.detection.bbox
            cv2.rectangle(image, (x, y), (x + w, y + h), (200, 200, 200), 1)
            cv2.putText(
                image,
                f"{label.name} {label.conf:.2f}",
                (x, y + h + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )
        track = step.track
        # Остальные машины — тонкие рамки: по ним Shift+клик выбирает цель.
        for other in (step.tracks or [])[1:]:
            if other.bbox is not None:
                x, y, w, h = other.bbox
                cv2.rectangle(image, (x, y), (x + w, y + h), (150, 150, 150), 1)
        if track.bbox is not None:
            x, y, w, h = track.bbox
            color = (80, 220, 80) if track.in_gate else (0, 200, 255)
            if step.selected:
                color = (0, 230, 255)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 4 if step.selected else 2)
            cv2.putText(
                image,
                # cv2.putText не рисует кириллицу.
                f"{self.session.target_id} TARGET" if step.selected else str(self.session.target_id),
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
        # Видимая часть кадра — по ней клик в окне камеры переводится в пиксели кадра.
        state["crop"] = (0, 0, width, height)
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
        state["crop"] = (x0, y0, x1, y1)
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
            None if not world.active else getattr(world, "path", None),
        )

    def _draw_status(self, tick) -> None:
        self._draw_summary(tick)
        if self._focus_shown != self.focus:
            self._focus_shown = self.focus
            self._update_focus_button()
        if self._target_shown != self.profile.name:
            self._target_shown = self.profile.name
            self._update_target_button()
        if not self.show_details.get():
            return
        parts = []
        if self.session.selection_camera is not None:
            following = self.session.selection_cameras()
            parts.append(
                f"выбрана машина {self.session.target_id}: ведут {', '.join(following) or 'нет — потеряна, жду её возвращения'}"
            )
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
                + (f"   ±{tick.sigma_m:.1f} м" if getattr(tick, "sigma_m", None) is not None else "")
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
        detector_note = self._detector_note()
        if detector_note:
            parts.append(detector_note)
        for pipeline in self.session.pipelines:
            latency = getattr(pipeline.source, "latency_ms", None)
            if isinstance(latency, (int, float)) and phone_usb.believable_latency_ms(float(latency)):
                parts.append(f"задержка {int(round(latency))} мс")
        for note in phone_usb.skipped.values():
            if note and note not in parts:
                parts.append(note)
        self.details.configure(text="\n".join(parts))

    def _draw_summary(self, tick) -> None:
        """Одна крупная строка — где цель; вторая — что видит каждая камера."""
        world = tick.world
        steps = [step for step in tick.steps if step.frame.camera_id != "wait"]
        target = self.session.target_id
        if world.active and world.xyz is not None:
            x, y, z = world.xyz[:3]
            text = (
                f"● Цель {target}    X {x:.2f}   Y {y:.2f}   Z {z:.2f} м    "
                f"{world.speed_m_s:.2f} м/с    дальность {np.linalg.norm(world.xyz):.2f} м"
            )
            sigma = getattr(tick, "sigma_m", None)
            if sigma is not None:
                text += f"    ±{sigma:.1f} м"
            color = FG
        elif not steps:
            text = "Нет камер — выберите «Источник» на панели сверху"
            color = MUTED
        elif len(self.session.poses) >= 2:
            text = "Цель ищется — для метров её должны видеть две камеры сразу"
            color = MUTED
        else:
            text = "Только 2D — для метров нужны две камеры с позами (см. «Калибровка» или F1)"
            color = MUTED
        self.summary.configure(text=text, fg=color)

        cams = []
        for step in steps:
            track = step.track
            mark = "●" if track.active else "○"
            pose = "" if step.frame.camera_id in self.session.poses else " без позы"
            cams.append(f"{mark} {step.frame.camera_id}: {track.state}{pose}")
        if self.session.selection_camera is not None:
            following = self.session.selection_cameras()
            cams.append(f"выбрана машина — ведут {', '.join(following) or 'никто, жду её'}")
        self.cams_line.configure(text="     ".join(cams))

    def _detector_note(self) -> str | None:
        """Режим детектора по описанию, устройство и мс; или почему он не включился."""
        for pipeline in self.session.pipelines:
            if pipeline.detector_error:
                return f"детектор {self.profile.mode} не включён: {pipeline.detector_error} — работает движение"
        timings = [
            pipeline.detector.ms
            for pipeline in self.session.pipelines
            if getattr(pipeline.detector, "spec", None) is not None
        ]
        if not timings:
            return None
        detector = next(p.detector for p in self.session.pipelines if getattr(p.detector, "spec", None) is not None)
        spec = detector.spec
        what = ", ".join(spec.classes) if spec.mode == "object" else ", ".join(spec.query)
        return (
            f"детектор {spec.mode}: {what}   {detector.device}   "
            f"{max(timings):.0f} мс на запуск, раз в {spec.det_every} кадра"
        )

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
