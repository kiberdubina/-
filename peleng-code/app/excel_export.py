"""Таблицы результата в Excel (.xlsx) рядом с CSV.

CSV остаётся основным форматом (frame, x, y, z для сдачи). Книга .xlsx собирается
из тех же CSV: числа — числами, заголовки по-русски, шапка закреплена.
Русский Excel сам показывает десятичную запятую.

openpyxl импортируется лениво: без него книга не пишется, CSV остаются.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from app.metrics3d import HIT_THRESHOLD_M, _METRIC_LABELS

# Подписи колонок CSV. Чего нет в словаре — остаётся как в CSV.
COLUMN_LABELS = {
    "frame": "Кадр",
    "t": "Время, с",
    "t_s": "Время, с",
    "x": "X, м",
    "y": "Y, м",
    "z": "Z, м",
    "x_m": "X, м",
    "y_m": "Y, м",
    "z_m": "Z, м",
    "box_cx_m": "Центр ящика X, м",
    "box_cy_m": "Центр ящика Y, м",
    "box_cz_m": "Центр ящика Z, м",
    "yaw_deg": "Курс, °",
    "pitch_deg": "Тангаж, °",
    "speed_m_s": "Скорость, м/с",
    "filtered_x_m": "Без сглаживания X, м",
    "filtered_y_m": "Без сглаживания Y, м",
    "filtered_z_m": "Без сглаживания Z, м",
    "cameras": "Камеры",
    "truth_x_m": "Эталон X, м",
    "truth_y_m": "Эталон Y, м",
    "truth_z_m": "Эталон Z, м",
    "error_m": "Ошибка, м",
    "from_frame": "С кадра",
    "to_frame": "По кадр",
    "frames": "Кадров",
    "median_m": "Медиана, м",
    "rmse_m": "RMSE, м",
    "max_m": "Максимум, м",
    "camera": "Камера",
    "silhouette_x": "Силуэт x, px",
    "silhouette_y": "Силуэт y, px",
    "silhouette_w": "Силуэт ширина, px",
    "silhouette_h": "Силуэт высота, px",
    "box3d_x": "Ящик 3D x, px",
    "box3d_y": "Ящик 3D y, px",
    "box3d_w": "Ящик 3D ширина, px",
    "box3d_h": "Ящик 3D высота, px",
}

# Целые колонки: без дробной части в формате ячейки.
_INTEGER_COLUMNS = {"frame", "from_frame", "to_frame", "frames"}


@dataclass
class Sheet:
    title: str
    header: list[str]
    rows: list[list]
    # Формат дробных чисел по колонкам; None — общий.
    formats: list[str | None] | None = None


def table_from_csv(title: str, path: Path) -> Sheet:
    """Лист из CSV: числа в числа, пустые ячейки пустые, подписи по-русски."""
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        names = next(reader, [])
        rows = [[_cell(value) for value in row] for row in reader]
    header = [COLUMN_LABELS.get(name, name) for name in names]
    formats = ["0" if name in _INTEGER_COLUMNS else "0.0000" for name in names]
    return Sheet(title, header, rows, formats)


def summary_sheet(stats: dict) -> Sheet:
    """Итог погрешности: те же строки, что metrics.txt."""
    rows: list[list] = [["Порог попадания, м", HIT_THRESHOLD_M]]
    for key, label in _METRIC_LABELS.items():
        if key in stats:
            value = stats[key]
            rows.append([label, int(value) if key == "frames" else float(value)])
    return Sheet("Итог", ["Показатель", "Значение"], rows, [None, "0.0000"])


def write_workbook(path: Path, sheets: list[Sheet]) -> Path | None:
    """Книга .xlsx из листов. Нет openpyxl — None, ничего не пишется."""
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font
        from openpyxl.utils import get_column_letter
    except ImportError:
        return None
    book = Workbook()
    book.remove(book.active)
    for sheet in sheets:
        page = book.create_sheet(_title(sheet.title))
        page.append(sheet.header)
        for cell in page[1]:
            cell.font = Font(bold=True)
        for row in sheet.rows:
            page.append(row)
        page.freeze_panes = "A2"
        for index, name in enumerate(sheet.header, start=1):
            number_format = None
            if sheet.formats and index - 1 < len(sheet.formats):
                number_format = sheet.formats[index - 1]
            if number_format:
                for (cell,) in page.iter_rows(min_row=2, min_col=index, max_col=index):
                    if isinstance(cell.value, float):
                        cell.number_format = number_format
            width = max(10, min(28, len(str(name)) + 2))
            page.column_dimensions[get_column_letter(index)].width = width
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    book.save(path)
    return path


def control_workbook(path: Path, files: dict, stats: dict | None) -> Path | None:
    """Книга контрольного прогона: итог, траектория, ошибка по отрезкам, рамки 2D."""
    sheets: list[Sheet] = []
    if stats:
        sheets.append(summary_sheet(stats))
    for key, title in (("csv", "Траектория"), ("segments", "Ошибка по отрезкам"), ("boxes2d", "Рамки 2D")):
        source = files.get(key)
        if source is not None and Path(source).is_file():
            sheets.append(table_from_csv(title, Path(source)))
    return write_workbook(path, sheets) if sheets else None


def _cell(text: str):
    if text == "":
        return None
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer() and "." not in text and "e" not in text.lower():
        return int(number)
    return number


def _title(text: str) -> str:
    # Excel: лист не длиннее 31 знака и без []:*?/\
    for bad in "[]:*?/\\":
        text = text.replace(bad, " ")
    return text[:31]
