import re
from dataclasses import dataclass
from pathlib import Path

import yaml

# Цель по описанию (app.object_detector): класс COCO + цвет или текстовый запрос.
DESCRIBED_MODES = ("object", "text")

# Пороги профиля по описанию, если в config.yaml их нет: отбор делает модель.
_DESCRIBED_GATES = {
    "s_min": 0.0,
    "s_max": 1e9,
    "u_min": 0.0,
    "u_max": 1e5,
    "r_min": 0.02,
    "r_max": 1000.0,
    "v_min": 0.0,
    "v_max": 200.0,
}


@dataclass
class ColorGate:
    h_min: int
    h_max: int
    s_min: int
    v_min: int
    s_max: int = 255
    v_max: int = 255


@dataclass
class BoardSpec:
    squares_x: int
    squares_y: int
    square_m: float
    marker_m: float


@dataclass
class TargetProfile:
    name: str
    s_min: float
    s_max: float
    u_min: float
    u_max: float
    r_min: float
    r_max: float
    v_min: float
    v_max: float
    d_m: float | None
    predict_s: float
    color: ColorGate | None
    moving_camera: bool = False
    require_motion: bool = False
    # motion | ego(via moving_camera) | color | sky_dark | white_blob
    mode: str = "auto"
    # Наземная машина: габарит 3D-ящика и опорная точка (см. app.vehicle.VehicleModel).
    # Есть — 3D считается подгонкой ящика к рамкам камер, нет — триангуляцией точки.
    vehicle: dict | None = None
    # mode object | text: поля профиля для app.object_detector.parse_spec.
    detect: dict | None = None

    @property
    def horizon_s(self) -> float:
        return self.predict_s


def _color(data) -> ColorGate | None:
    raw = data.get("color")
    if not raw:
        return None
    return ColorGate(
        h_min=int(raw["h_min"]),
        h_max=int(raw["h_max"]),
        s_min=int(raw["s_min"]),
        v_min=int(raw["v_min"]),
        s_max=int(raw.get("s_max", 255)),
        v_max=int(raw.get("v_max", 255)),
    )


def _profile(name: str, target: dict, predict_s: float) -> TargetProfile:
    mode = str(target.get("mode", "auto"))
    described = mode.lower() in DESCRIBED_MODES
    gates = {**_DESCRIBED_GATES, **target} if described else target
    return TargetProfile(
        name=name,
        s_min=float(gates["s_min"]),
        s_max=float(gates["s_max"]),
        u_min=float(gates["u_min"]),
        u_max=float(gates["u_max"]),
        r_min=float(gates["r_min"]),
        r_max=float(gates["r_max"]),
        v_min=float(gates["v_min"]),
        v_max=float(gates["v_max"]),
        d_m=None if target.get("d_m") is None else float(target["d_m"]),
        predict_s=predict_s,
        # У профиля по описанию цвет — список диапазонов внутри рамки, не ColorGate.
        color=None if described else _color(target),
        moving_camera=bool(target.get("moving_camera", False)),
        require_motion=bool(target.get("require_motion", False)),
        mode=mode,
        vehicle=dict(target["vehicle"]) if target.get("vehicle") else None,
        detect=dict(target) if described else None,
    )


def load_document(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_profile(path: Path) -> TargetProfile:
    data = load_document(path)
    predict_s = float(data.get("predict_s", 0.5))
    patterns = data.get("patterns")
    if patterns:
        name = data.get("active", next(iter(patterns)))
        return _profile(name, patterns[name], predict_s)
    return _profile("target", data["target"], predict_s)


def load_patterns(path: Path) -> dict[str, TargetProfile]:
    data = load_document(path)
    predict_s = float(data.get("predict_s", 0.5))
    patterns = data.get("patterns")
    if not patterns:
        return {"target": _profile("target", data["target"], predict_s)}
    return {name: _profile(name, raw, predict_s) for name, raw in patterns.items()}


def load_board(path: Path) -> BoardSpec:
    data = load_document(path).get("board", {})
    return BoardSpec(
        squares_x=int(data.get("squares_x", 5)),
        squares_y=int(data.get("squares_y", 7)),
        square_m=float(data.get("square_m", 0.04)),
        marker_m=float(data.get("marker_m", 0.03)),
    )


def save_active(path: Path, name: str, predict_s: float) -> None:
    """Только активный профиль и горизонт. Строки правятся на месте: комментарии остаются."""
    text = path.read_text(encoding="utf-8")
    lines = {"active": name, "predict_s": predict_s}
    for key, value in lines.items():
        pattern = re.compile(rf"^{key}:.*$", re.MULTILINE)
        if pattern.search(text) is None:
            data = load_document(path)
            data["active"] = name
            data["predict_s"] = predict_s
            _dump(path, data)
            return
        text = pattern.sub(f"{key}: {value}", text, count=1)
    path.write_text(text, encoding="utf-8")


def save_pattern(path: Path, name: str, fields: dict, predict_s: float) -> None:
    """Профиль name целиком из fields и сделать его активным."""
    data = load_document(path)
    data["predict_s"] = predict_s
    data["active"] = name
    data.setdefault("patterns", {})[name] = dict(fields)
    _dump(path, data)


def save_custom(path: Path, profile: TargetProfile, predict_s: float) -> None:
    data = load_document(path)
    data["predict_s"] = predict_s
    data["active"] = profile.name
    if "patterns" not in data:
        data["patterns"] = {}
    # Поля, которых профиль не задаёт (mode, vehicle, class…), остаются как были.
    entry = dict(data["patterns"].get(profile.name) or {})
    entry.update({
        "s_min": profile.s_min,
        "s_max": profile.s_max,
        "u_min": profile.u_min,
        "u_max": profile.u_max,
        "r_min": profile.r_min,
        "r_max": profile.r_max,
        "v_min": profile.v_min,
        "v_max": profile.v_max,
        "d_m": profile.d_m,
        "color": None
        if profile.color is None
        else {
            "h_min": profile.color.h_min,
            "h_max": profile.color.h_max,
            "s_min": profile.color.s_min,
            "s_max": profile.color.s_max,
            "v_min": profile.color.v_min,
            "v_max": profile.color.v_max,
        },
        "moving_camera": profile.moving_camera,
        "require_motion": profile.require_motion,
    })
    if profile.detect is not None:
        entry["color"] = profile.detect.get("color")
    data["patterns"][profile.name] = entry
    _dump(path, data)


def _dump(path: Path, data: dict) -> None:
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
