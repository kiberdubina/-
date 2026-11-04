"""Собирает models/truck_world_top.pt — детектор грузовика, который видит его и сверху (дрон Way2 cam-2).

YOLO-World x (models/yolov8x-worldv2.pt) с вшитыми запросами; все классы названы truck, поэтому
программа берёт файл как обычный детектор класса truck (профиль way2 в config.yaml).
Нужны ultralytics и CLIP (weights/clip/ViT-B-32.pt); запуск из корня проекта: python _make_truck_world_top.py
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROMPTS = [
    "truck",
    "cargo truck",
    "lorry",
    "box truck",
    "white box truck seen from above",
    "lorry top view",
    "truck roof from above",
]


def main() -> None:
    import ultralytics

    model = ultralytics.YOLOWorld(str(ROOT / "models" / "yolov8x-worldv2.pt"))
    model.set_classes(PROMPTS)
    model.model.names = {index: "truck" for index in range(len(PROMPTS))}
    out = ROOT / "models" / "truck_world_top.pt"
    model.save(str(out))
    print(f"{out}: классы {ultralytics.YOLO(str(out)).names}")


if __name__ == "__main__":
    main()
