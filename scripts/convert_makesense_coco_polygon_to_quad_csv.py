#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_makesense_coco_polygon_to_quad_csv.py

makeSense.ai の COCO Polygon JSON から、4点学習用 labels.csv を生成する。

入力:
  - COCO JSON（images / annotations / categories を含む）
  - 各 annotation の segmentation は 4点ポリゴン（座標8値）を想定

出力:
  filename,lt_x,lt_y,rt_x,rt_y,rb_x,rb_y,lb_x,lb_y
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _order_quad_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """
    入力4点を [lt, rt, rb, lb] の順に並べ替える。
    まず sum/diff ベースで決定し、重複した場合は y 2分割でフォールバックする。
    """
    if len(points) != 4:
        raise ValueError(f"4点が必要です。受け取った点数: {len(points)}")

    sums = [x + y for x, y in points]
    diffs = [y - x for x, y in points]

    idx_lt = min(range(4), key=lambda i: sums[i])
    idx_rb = max(range(4), key=lambda i: sums[i])
    idx_rt = min(range(4), key=lambda i: diffs[i])
    idx_lb = max(range(4), key=lambda i: diffs[i])

    idx_set = {idx_lt, idx_rt, idx_rb, idx_lb}
    if len(idx_set) == 4:
        return [points[idx_lt], points[idx_rt], points[idx_rb], points[idx_lb]]

    # まれに sum/diff で重複が出るケースに対応する。
    pts_sorted = sorted(points, key=lambda p: (p[1], p[0]))
    top2 = sorted(pts_sorted[:2], key=lambda p: p[0])
    bottom2 = sorted(pts_sorted[2:], key=lambda p: p[0])
    lt, rt = top2[0], top2[1]
    lb, rb = bottom2[0], bottom2[1]
    return [lt, rt, rb, lb]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="makeSense COCO Polygon JSON を4点学習CSVへ変換"
    )
    parser.add_argument("--input", required=True, help="入力COCO JSON")
    parser.add_argument("--output", required=True, help="出力 labels.csv")
    parser.add_argument(
        "--category",
        default=None,
        help="対象カテゴリ名（未指定時は全カテゴリ）",
    )
    parser.add_argument(
        "--normalize-output",
        action="store_true",
        help="座標を 0..1 に正規化して出力",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="想定外データをスキップせずエラーにする",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"入力JSONが存在しません: {input_path}")

    data = json.loads(input_path.read_text(encoding="utf-8"))

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    image_by_id: dict[int, dict[str, object]] = {}
    for image in images:
        image_id = image.get("id")
        if isinstance(image_id, int):
            image_by_id[image_id] = image

    category_name_by_id: dict[int, str] = {}
    for category in categories:
        category_id = category.get("id")
        category_name = category.get("name")
        if isinstance(category_id, int) and isinstance(category_name, str):
            category_name_by_id[category_id] = category_name

    rows_out: list[list[str]] = []
    skipped = 0

    for ann in annotations:
        image_id = ann.get("image_id")
        category_id = ann.get("category_id")
        segmentation = ann.get("segmentation")

        if not isinstance(image_id, int) or image_id not in image_by_id:
            if args.strict:
                raise RuntimeError(f"image_id が不正なannotationがあります: {ann}")
            skipped += 1
            continue

        if args.category is not None:
            ann_category = category_name_by_id.get(category_id, "")
            if ann_category != args.category:
                continue

        if (
            not isinstance(segmentation, list)
            or not segmentation
            or not isinstance(segmentation[0], list)
        ):
            if args.strict:
                raise RuntimeError(
                    f"segmentation 形式が想定外です (annotation id={ann.get('id')})"
                )
            skipped += 1
            continue

        polygon = segmentation[0]
        if len(polygon) != 8:
            if args.strict:
                raise RuntimeError(
                    f"4点ポリゴンではありません (annotation id={ann.get('id')}, coords={len(polygon)})"
                )
            skipped += 1
            continue

        points = [(float(polygon[i]), float(polygon[i + 1])) for i in range(0, 8, 2)]
        lt, rt, rb, lb = _order_quad_points(points)

        image_info = image_by_id[image_id]
        filename = str(image_info.get("file_name", "")).strip()
        width = float(image_info.get("width", 0) or 0)
        height = float(image_info.get("height", 0) or 0)

        if not filename:
            if args.strict:
                raise RuntimeError(
                    f"file_name が空です (image_id={image_id}, annotation id={ann.get('id')})"
                )
            skipped += 1
            continue

        if args.normalize_output:
            if width <= 0 or height <= 0:
                raise RuntimeError(
                    f"正規化出力に必要な width/height が不正です: {filename}"
                )
            coords = [
                lt[0] / width,
                lt[1] / height,
                rt[0] / width,
                rt[1] / height,
                rb[0] / width,
                rb[1] / height,
                lb[0] / width,
                lb[1] / height,
            ]
        else:
            coords = [lt[0], lt[1], rt[0], rt[1], rb[0], rb[1], lb[0], lb[1]]

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

    rows_out.sort(key=lambda row: row[0])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["filename", "lt_x", "lt_y", "rt_x", "rt_y", "rb_x", "rb_y", "lb_x", "lb_y"]
        )
        writer.writerows(rows_out)

    print(f"[done] input: {input_path}")
    print(f"[done] output: {output_path}")
    print(f"[info] exported_images: {len(rows_out)}")
    if skipped > 0:
        print(f"[warn] skipped_annotations: {skipped}")


if __name__ == "__main__":
    main()
