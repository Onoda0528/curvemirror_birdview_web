#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_makesense_points_to_quad_csv.py

makeSense.ai の Point CSV を、4点学習用 labels.csv へ変換する。

想定入力（makeSense Point CSV）:
  label, x, y, filename, image_width, image_height
※ makeSense 標準はヘッダなし。ヘッダがあっても読み飛ばす。

出力:
  filename,lt_x,lt_y,rt_x,rt_y,rb_x,rb_y,lb_x,lb_y
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


CORNER_ORDER = ("lt", "rt", "rb", "lb")

DEFAULT_LABEL_ALIASES: dict[str, tuple[str, ...]] = {
    "lt": ("lt", "lefttop", "topleft", "left_upper", "upper_left", "lu"),
    "rt": ("rt", "righttop", "topright", "right_upper", "upper_right", "ru"),
    "rb": ("rb", "rightbottom", "bottomright", "right_lower", "lower_right", "rd"),
    "lb": ("lb", "leftbottom", "bottomleft", "left_lower", "lower_left", "ld"),
}


@dataclass
class RowPoint:
    x: float
    y: float
    image_width: float | None
    image_height: float | None
    row_number: int


def normalize_label_name(label: str) -> str:
    # 表記ゆれ（空白・ハイフン・アンダースコア・記号）を吸収する。
    return "".join(ch for ch in label.lower() if ch.isalnum())


def parse_alias_argument(value: str | None, default_aliases: Iterable[str]) -> set[str]:
    raw_items: list[str] = []
    if value is not None and value.strip():
        raw_items = [item.strip() for item in value.split(",") if item.strip()]
    else:
        raw_items = list(default_aliases)
    normalized = {normalize_label_name(item) for item in raw_items}
    return {item for item in normalized if item}


def looks_like_header(row: list[str]) -> bool:
    if len(row) < 4:
        return False
    c0 = row[0].strip().lower()
    c1 = row[1].strip().lower()
    c2 = row[2].strip().lower()
    c3 = row[3].strip().lower()
    return (
        c0 in {"label", "class", "name"}
        and c1 in {"x", "point_x", "cx"}
        and c2 in {"y", "point_y", "cy"}
        and c3 in {"filename", "file", "image", "image_name"}
    )


def parse_float(value: str, column_name: str, row_number: int) -> float:
    try:
        return float(value)
    except Exception as exc:
        raise RuntimeError(
            f"{row_number}行目の {column_name} が数値ではありません: {value}"
        ) from exc


def parse_optional_float(value: str, column_name: str, row_number: int) -> float | None:
    if value.strip() == "":
        return None
    return parse_float(value, column_name, row_number)


def main() -> None:
    parser = argparse.ArgumentParser(description="makeSense Point CSV を4点学習CSVへ変換")
    parser.add_argument("--input", required=True, help="makeSense Point CSV")
    parser.add_argument("--output", required=True, help="変換後 labels.csv")
    parser.add_argument(
        "--delimiter",
        default=",",
        help="入力CSV区切り文字（既定: ,）",
    )
    parser.add_argument(
        "--lt-labels",
        default=None,
        help="LTとして扱うラベル名（カンマ区切り）",
    )
    parser.add_argument(
        "--rt-labels",
        default=None,
        help="RTとして扱うラベル名（カンマ区切り）",
    )
    parser.add_argument(
        "--rb-labels",
        default=None,
        help="RBとして扱うラベル名（カンマ区切り）",
    )
    parser.add_argument(
        "--lb-labels",
        default=None,
        help="LBとして扱うラベル名（カンマ区切り）",
    )
    parser.add_argument(
        "--normalize-output",
        action="store_true",
        help="出力座標を 0..1 正規化にする（既定はピクセル座標）",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="欠損・未知ラベルを許容せずエラーにする",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"入力CSVが存在しません: {input_path}")

    alias_map = {
        "lt": parse_alias_argument(args.lt_labels, DEFAULT_LABEL_ALIASES["lt"]),
        "rt": parse_alias_argument(args.rt_labels, DEFAULT_LABEL_ALIASES["rt"]),
        "rb": parse_alias_argument(args.rb_labels, DEFAULT_LABEL_ALIASES["rb"]),
        "lb": parse_alias_argument(args.lb_labels, DEFAULT_LABEL_ALIASES["lb"]),
    }

    label_to_corner: dict[str, str] = {}
    for corner, aliases in alias_map.items():
        for alias in aliases:
            # 複数cornerに同じエイリアスが入っていたら後勝ちにせずエラーにする。
            if alias in label_to_corner and label_to_corner[alias] != corner:
                raise RuntimeError(
                    f"ラベルエイリアス '{alias}' が複数cornerへ割り当てられています。"
                )
            label_to_corner[alias] = corner

    image_points: dict[str, dict[str, RowPoint]] = {}
    skipped_unknown_labels = 0
    duplicate_overwrites = 0
    read_rows = 0

    with open(input_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter=args.delimiter)
        for row_number, row in enumerate(reader, start=1):
            if not row or all(cell.strip() == "" for cell in row):
                continue
            read_rows += 1

            if looks_like_header(row):
                continue

            if len(row) < 4:
                if args.strict:
                    raise RuntimeError(
                        f"{row_number}行目の列数が不足しています（最低4列必要）: {row}"
                    )
                continue

            label_raw = row[0].strip()
            x_raw = row[1].strip()
            y_raw = row[2].strip()
            filename = row[3].strip()
            image_width_raw = row[4].strip() if len(row) > 4 else ""
            image_height_raw = row[5].strip() if len(row) > 5 else ""

            if filename == "":
                if args.strict:
                    raise RuntimeError(f"{row_number}行目の filename が空です。")
                continue

            corner = label_to_corner.get(normalize_label_name(label_raw))
            if corner is None:
                skipped_unknown_labels += 1
                if args.strict:
                    raise RuntimeError(
                        f"{row_number}行目のラベル '{label_raw}' は4点ラベルに割り当てできません。"
                    )
                continue

            x = parse_float(x_raw, "x", row_number)
            y = parse_float(y_raw, "y", row_number)
            image_width = parse_optional_float(image_width_raw, "image_width", row_number)
            image_height = parse_optional_float(image_height_raw, "image_height", row_number)

            per_image = image_points.setdefault(filename, {})
            if corner in per_image:
                duplicate_overwrites += 1
            per_image[corner] = RowPoint(
                x=x,
                y=y,
                image_width=image_width,
                image_height=image_height,
                row_number=row_number,
            )

    rows_out: list[list[str]] = []
    dropped_incomplete = 0
    for filename in sorted(image_points.keys()):
        per_image = image_points[filename]
        missing = [corner for corner in CORNER_ORDER if corner not in per_image]
        if missing:
            dropped_incomplete += 1
            if args.strict:
                raise RuntimeError(
                    f"画像 '{filename}' に欠損ラベルがあります: {', '.join(missing)}"
                )
            continue

        # 4点のうちどれかにサイズ情報がある場合は採用する。
        width = next(
            (per_image[c].image_width for c in CORNER_ORDER if per_image[c].image_width is not None),
            None,
        )
        height = next(
            (per_image[c].image_height for c in CORNER_ORDER if per_image[c].image_height is not None),
            None,
        )

        coords: list[float] = []
        for corner in CORNER_ORDER:
            point = per_image[corner]
            if args.normalize_output:
                if width is None or height is None or width <= 0 or height <= 0:
                    raise RuntimeError(
                        f"画像 '{filename}' の正規化出力に必要なサイズ情報がありません。"
                    )
                coords.extend([point.x / width, point.y / height])
            else:
                coords.extend([point.x, point.y])

        rows_out.append(
            [
                filename,
                f"{coords[0]:.6f}",
                f"{coords[1]:.6f}",
                f"{coords[2]:.6f}",
                f"{coords[3]:.6f}",
                f"{coords[4]:.6f}",
                f"{coords[5]:.6f}",
                f"{coords[6]:.6f}",
                f"{coords[7]:.6f}",
            ]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["filename", "lt_x", "lt_y", "rt_x", "rt_y", "rb_x", "rb_y", "lb_x", "lb_y"]
        )
        writer.writerows(rows_out)

    print(f"[done] input: {input_path}")
    print(f"[done] output: {output_path}")
    print(f"[info] read_rows: {read_rows}")
    print(f"[info] exported_images: {len(rows_out)}")
    if skipped_unknown_labels > 0:
        print(f"[warn] skipped_unknown_labels: {skipped_unknown_labels}")
    if duplicate_overwrites > 0:
        print(f"[warn] duplicate_overwrites(last_wins): {duplicate_overwrites}")
    if dropped_incomplete > 0:
        print(f"[warn] dropped_incomplete_images: {dropped_incomplete}")


if __name__ == "__main__":
    main()
