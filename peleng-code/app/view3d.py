import math
import tkinter as tk

import numpy as np

from app.geom import CameraPose, camera_center, frustum_edges

# Близкая плоскость отсечения, в тех же единицах, что и сцена (метры).
NEAR = 0.05
# Силуэт ограничивающей сферы занимает эту долю половины кадра при zoom = 1.
SPHERE_FILL = 0.78
TRAIL_COLOR = "#f0b45a"


def scene_radius(points: np.ndarray, target: np.ndarray) -> float:
    """Радиус облака вокруг точки, вокруг которой крутится сцена."""
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    finite = pts[np.isfinite(pts).all(axis=1)]
    if len(finite) == 0:
        return 0.25
    center = np.asarray(target, np.float64).reshape(3)
    radius = float(np.max(np.linalg.norm(finite - center, axis=1)))
    return max(radius, 0.05)


def fit_distance(radius: float, width: int, height: int) -> float:
    """Дистанция глаза, при которой вся сфера радиуса radius видна целиком."""
    radius = max(float(radius), 1e-3)
    half = 0.5 * float(max(min(int(width), int(height)), 1))
    tangent = SPHERE_FILL * half / 420.0
    sine = tangent / math.hypot(1.0, tangent)
    return radius / max(sine, 1e-3)


def project_points(
    points: np.ndarray,
    yaw: float,
    pitch: float,
    zoom: float,
    target: np.ndarray,
    width: int,
    height: int,
    distance: float | None = None,
) -> np.ndarray:
    """Экранные координаты и глубина. Масштаб траектории задаёт zoom.

    distance — удаление глаза от target. По умолчанию 0,9 м, как у короткой
    сцены; холст подставляет fit_distance, чтобы в кадр входила вся траектория.
    """
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    eye_distance = 0.9 if distance is None else max(float(distance), 1e-3)
    direction = np.array(
        [
            np.cos(pitch) * np.cos(yaw),
            np.cos(pitch) * np.sin(yaw),
            np.sin(pitch),
        ]
    )
    eye = np.asarray(target, np.float64).reshape(3) + eye_distance * direction
    forward = np.asarray(target, np.float64).reshape(3) - eye
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    relative = pts - eye
    depth = relative @ forward
    focal = 420.0 * float(zoom)
    screen_x = width / 2.0 + focal * (relative @ right) / np.maximum(depth, 1e-3)
    screen_y = height / 2.0 - focal * (relative @ up) / np.maximum(depth, 1e-3)
    return np.stack([screen_x, screen_y, depth], axis=1)


class SceneCanvas(tk.Canvas):
    """Правая часть окна: орбита, масштаб, след, прогноз, конусы камер и лучи."""

    def __init__(self, parent: tk.Misc):
        super().__init__(
            parent,
            width=520,
            height=520,
            bg="#12141a",
            highlightthickness=0,
            takefocus=True,
        )
        self.yaw = 0.7
        self.pitch = 0.4
        self.zoom = 1.0
        self.trail: list[np.ndarray] = []
        self.current: np.ndarray | None = None
        self.pred: np.ndarray | None = None
        self.poses: list[CameraPose] = []
        self.rays: list[tuple[np.ndarray, np.ndarray]] = []
        self.hint = "Вторая камера даст метры"
        self._drag: tuple[int, int] | None = None
        self._seen_size: tuple[int, int] | None = None
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._drag_move)
        self.bind("<MouseWheel>", self._wheel)
        self.bind("<Button-4>", self._wheel)
        self.bind("<Button-5>", self._wheel)
        self.bind("<Enter>", lambda _event: self.focus_set())
        self.bind("<Configure>", self._on_configure)
        self.redraw()

    def set_view_size(self, width: int, height: int) -> tuple[int, int]:
        """Поле 3D само не тянется за окном. Размер меняется только по просьбе."""
        width = int(min(1600, max(240, width)))
        height = int(min(1200, max(240, height)))
        self.configure(width=width, height=height)
        return width, height

    def set_scene(
        self,
        trail: list[np.ndarray],
        current: np.ndarray | None,
        pred: np.ndarray | None,
        poses: list[CameraPose],
        hint: str,
        rays: list[tuple[np.ndarray, np.ndarray]] | None = None,
        box: np.ndarray | None = None,
        path: np.ndarray | None = None,
        trail_color: str | None = None,
    ) -> None:
        # Цвет пройденного пути: по умолчанию оранжевый, для примерного пути одной камеры — жёлтый.
        self.trail_color = trail_color or TRAIL_COLOR
        self.box = None if box is None else np.asarray(box, np.float64).reshape(8, 3).copy()
        # Прогноз траектории до горизонта (WorldView.path); нет — прямой отрезок до pred.
        self.path = None if path is None else np.asarray(path, np.float64).reshape(-1, 3).copy()
        self.trail = [np.asarray(point, np.float64).reshape(3).copy() for point in trail]
        self.current = None if current is None else np.asarray(current, np.float64).reshape(3).copy()
        self.pred = None if pred is None else np.asarray(pred, np.float64).reshape(3).copy()
        self.poses = list(poses)
        self.rays = [
            (
                np.asarray(start, np.float64).reshape(3).copy(),
                np.asarray(end, np.float64).reshape(3).copy(),
            )
            for start, end in (rays or [])
        ]
        self.hint = hint
        self.redraw()

    def _press(self, event: tk.Event) -> None:
        self._drag = (event.x, event.y)

    def _drag_move(self, event: tk.Event) -> None:
        if self._drag is None:
            return
        dx = event.x - self._drag[0]
        dy = event.y - self._drag[1]
        self._drag = (event.x, event.y)
        self.yaw -= dx * 0.01
        self.pitch = float(np.clip(self.pitch + dy * 0.01, -1.1, 1.1))
        self.redraw()

    def _wheel(self, event: tk.Event) -> str:
        delta = int(getattr(event, "delta", 0) or 0)
        number = getattr(event, "num", None)
        if number == 4:
            delta = 120
        elif number == 5:
            delta = -120
        if delta != 0:
            factor = 1.1 if delta > 0 else 1 / 1.1
            self.zoom = float(np.clip(self.zoom * factor, 0.3, 8.0))
            self.redraw()
        return "break"

    def _on_configure(self, event: tk.Event) -> None:
        if event.widget is not self or event.width <= 1 or event.height <= 1:
            return
        size = (int(event.width), int(event.height))
        if size == self._seen_size:
            return
        self._seen_size = size
        self.redraw()

    def _target(self) -> np.ndarray:
        if self.trail:
            return np.mean(np.stack(self.trail, axis=0), axis=0)
        if self.current is not None:
            return np.asarray(self.current, np.float64).reshape(3)
        return np.zeros(3)

    def _size(self) -> tuple[int, int]:
        width = int(self.winfo_width())
        height = int(self.winfo_height())
        if width <= 1 or height <= 1:
            return 520, 520
        return width, height

    def _layout(self) -> tuple[np.ndarray, float, float, list]:
        target = np.asarray(self._target(), np.float64).reshape(3)
        frusta: list[tuple[CameraPose, list]] = []
        samples = [np.zeros(3, np.float64)]
        for point in self.trail:
            samples.append(np.asarray(point, np.float64).reshape(3))
        if self.current is not None:
            samples.append(np.asarray(self.current, np.float64).reshape(3))
        if self.pred is not None:
            samples.append(np.asarray(self.pred, np.float64).reshape(3))
        for pose in self.poses:
            samples.append(camera_center(pose))
        for start, end in self.rays:
            samples.append(np.asarray(start, np.float64).reshape(3))
            samples.append(np.asarray(end, np.float64).reshape(3))
        box = getattr(self, "box", None)
        if box is not None:
            samples.extend(box)
        bare = (
            not self.trail
            and self.current is None
            and self.pred is None
            and not self.poses
            and not self.rays
        )
        if bare:
            return target, 0.28, 0.2, frusta
        span = scene_radius(np.stack(samples, axis=0), target)
        # На столе конус как раньше, 15 см. На трассе в сотни метров — доля поля, иначе камера точка.
        depth = 0.15 if span < 2.0 else span * 0.055
        for pose in self.poses:
            edges = frustum_edges(pose, depth)
            frusta.append((pose, edges))
            for start, end in edges:
                samples.append(np.asarray(start, np.float64).reshape(3))
                samples.append(np.asarray(end, np.float64).reshape(3))
        radius = scene_radius(np.stack(samples, axis=0), target)
        room = max(radius - float(np.linalg.norm(target)), 0.0)
        axis_len = min(0.2, room * 0.55)
        return target, radius, axis_len, frusta

    def _project(
        self,
        points: np.ndarray,
        target: np.ndarray,
        distance: float,
        width: int,
        height: int,
    ) -> np.ndarray:
        return project_points(
            points, self.yaw, self.pitch, self.zoom, target, width, height, distance
        )

    def _polyline(
        self,
        projected: np.ndarray,
        color: str,
        width: int,
        tag: str,
        dash: tuple[int, int] | None = None,
    ) -> None:
        run: list[tuple[float, float]] = []

        def flush() -> None:
            if len(run) < 2:
                run.clear()
                return
            flat = [coord for xy in run for coord in xy]
            options = {"fill": color, "width": width, "tags": tag}
            if dash is not None:
                options["dash"] = dash
            self.create_line(*flat, **options)
            run.clear()

        for row in projected:
            if not np.isfinite(row).all() or float(row[2]) <= NEAR:
                flush()
                continue
            run.append((float(row[0]), float(row[1])))
        flush()

    def _xy(self, projected_row: np.ndarray) -> tuple[float, float] | None:
        if not np.isfinite(projected_row).all() or float(projected_row[2]) <= NEAR:
            return None
        return float(projected_row[0]), float(projected_row[1])

    def redraw(self) -> None:
        self.delete("all")
        width, height = self._size()
        target, radius, axis_len, frusta = self._layout()
        distance = fit_distance(radius, width, height)
        origin = np.zeros(3)
        if axis_len >= 0.02:
            tips = np.eye(3) * axis_len
            axis_pts = np.vstack([origin, tips])
            projected = self._project(axis_pts, target, distance, width, height)
            colors = ("#c45b5b", "#5ba36a", "#6a8fd4")
            names = ("X", "Y", "Z")
            start = self._xy(projected[0])
            for index, (color, name) in enumerate(zip(colors, names)):
                end = self._xy(projected[index + 1])
                if start is None or end is None:
                    continue
                self.create_line(*start, *end, fill=color, width=2, tags="axis")
                span_x = end[0] - start[0]
                span_y = end[1] - start[1]
                norm = math.hypot(span_x, span_y) or 1.0
                self.create_text(
                    end[0] + 11 * span_x / norm,
                    end[1] + 11 * span_y / norm,
                    text=name,
                    fill=color,
                    font=("Segoe UI", 9),
                    tags="axis",
                )
        for pose, edges in frusta:
            edge_pts = np.stack([np.stack(edge, axis=0) for edge in edges], axis=0)
            flat = edge_pts.reshape(-1, 3)
            projected = self._project(flat, target, distance, width, height)
            for index in range(0, len(projected), 2):
                pa = self._xy(projected[index])
                pb = self._xy(projected[index + 1])
                if pa is None or pb is None:
                    continue
                self.create_line(*pa, *pb, fill="#8eb4ff", width=1, tags="camera")
            center = self._xy(projected[0])
            if center is not None:
                self.create_text(
                    center[0],
                    center[1] - 10,
                    text=pose.camera_id,
                    fill="#d5e2ff",
                    font=("Segoe UI", 9),
                    tags="camera",
                )
        for start, end in self.rays:
            segment = np.stack([start, end], axis=0)
            self._polyline(
                self._project(segment, target, distance, width, height),
                "#9eb0d8",
                1,
                "ray",
                (3, 3),
            )
        box = getattr(self, "box", None)
        if box is not None:
            from app.vehicle import BOX_EDGES

            projected = self._project(box, target, distance, width, height)
            for a, b in BOX_EDGES:
                pa = self._xy(projected[a])
                pb = self._xy(projected[b])
                if pa is not None and pb is not None:
                    self.create_line(*pa, *pb, fill="#5fd35f", width=2, tags="box")
        if len(self.trail) >= 2:
            trail = np.stack(self.trail, axis=0)
            self._polyline(
                self._project(trail, target, distance, width, height),
                getattr(self, "trail_color", TRAIL_COLOR),
                2,
                "trail",
            )
        if self.current is not None and self.pred is not None:
            path = getattr(self, "path", None)
            if path is not None and len(path):
                segment = np.vstack([self.current.reshape(1, 3), path])
            else:
                segment = np.stack([self.current, self.pred], axis=0)
            self._polyline(
                self._project(segment, target, distance, width, height),
                "#3ec6e0",
                2,
                "forecast",
                (6, 4),
            )
            ahead = self._xy(self._project(self.pred.reshape(1, 3), target, distance, width, height)[0])
            if ahead is not None:
                x, y = ahead
                self.create_rectangle(
                    x - 7, y - 7, x + 7, y + 7, outline="#3ec6e0", width=2, tags="forecast"
                )
        if self.current is not None:
            place = self._xy(
                self._project(self.current.reshape(1, 3), target, distance, width, height)[0]
            )
            if place is not None:
                x, y = place
                self.create_oval(
                    x - 11,
                    y - 11,
                    x + 11,
                    y + 11,
                    outline="#f0b45a",
                    width=2,
                    tags="now",
                )
                self.create_oval(
                    x - 6,
                    y - 6,
                    x + 6,
                    y + 6,
                    fill="#ffffff",
                    outline="#12141a",
                    width=1,
                    tags="now",
                )
        self.create_text(
            12,
            16,
            anchor="w",
            text=self.hint,
            fill="#c8c8c8",
            font=("Segoe UI", 10),
            tags="hud",
        )
        self.create_text(
            width - 12,
            height - 14,
            anchor="e",
            text="мышь — поворот, колёсико — масштаб, угол — размер поля",
            fill="#8d93a3",
            font=("Segoe UI", 9),
            tags="hud",
        )
