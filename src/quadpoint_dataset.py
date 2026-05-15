from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math
from typing import Optional

import cv2
import numpy as np


REQUIRED_COLUMNS = (
    "filename",
    "lt_x",
    "lt_y",
    "rt_x",
    "rt_y",
    "rb_x",
    "rb_y",
    "lb_x",
    "lb_y",
)

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class QuadpointDatasetInfo:
    images_dir: Path
    labels_csv: Path
    sample_count: int
    masks_dir: Path | None = None
    output_root: Path | None = None


def _iter_image_files(directory: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
    return files


def _build_image_indices(images_dir: Path) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    by_relative: dict[str, Path] = {}
    by_name: dict[str, list[Path]] = {}

    for path in sorted(images_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(images_dir).as_posix()
        by_relative[rel] = path
        by_name.setdefault(path.name, []).append(path)

    return by_relative, by_name


def _mask_stem(stem: str) -> str:
    return stem[:-5] if stem.endswith("_mask") else stem


def _relative_stem_key(path: Path, root: Path, is_mask: bool) -> str:
    rel = path.relative_to(root)
    stem = _mask_stem(rel.stem) if is_mask else rel.stem
    if str(rel.parent) == ".":
        return stem
    return f"{rel.parent.as_posix()}/{stem}"


def _stem_key(path: Path, is_mask: bool) -> str:
    return _mask_stem(path.stem) if is_mask else path.stem


def _build_mask_indices(masks_dir: Path) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    files = _iter_image_files(masks_dir)
    by_rel_stem: dict[str, list[Path]] = {}
    by_stem: dict[str, list[Path]] = {}
    for path in files:
        key_rel = _relative_stem_key(path, masks_dir, is_mask=True)
        by_rel_stem.setdefault(key_rel, []).append(path)

        key_stem = _stem_key(path, is_mask=True)
        by_stem.setdefault(key_stem, []).append(path)
    return by_rel_stem, by_stem


def resolve_image_path(
    filename: str,
    images_dir: Path,
    by_relative: dict[str, Path],
    by_name: dict[str, list[Path]],
) -> Path:
    """
    CSV 上の filename から実ファイルパスを解決する。
    """
    normalized = filename.replace("\\", "/").strip()
    if not normalized:
        raise RuntimeError("filename が空です。")

    direct = (images_dir / normalized).resolve()
    if direct.exists() and direct.is_file():
        return direct

    from_rel = by_relative.get(normalized)
    if from_rel is not None:
        return from_rel

    basename = Path(normalized).name
    matches = by_name.get(basename, [])
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(
            f"同名画像が複数あり一意に解決できません: {basename}. "
            "CSV ではサブディレクトリを含む相対パスで指定してください。"
        )

    raise RuntimeError(f"画像が見つかりません: {filename}")


def resolve_mask_path_for_image(
    image_path: Path,
    images_dir: Path,
    masks_dir: Path,
    mask_by_rel_stem: dict[str, list[Path]],
    mask_by_stem: dict[str, list[Path]],
) -> Path:
    """
    画像に対応するマスクパスを解決する。
    優先順:
    1) 同一相対パス（同名）
    2) 同一相対パス + `_mask` サフィックス
    3) 相対パスの stem 一致（拡張子違い許容）
    4) basename stem 一致（一意な場合のみ）
    """
    rel = image_path.relative_to(images_dir)

    direct = (masks_dir / rel).resolve()
    if direct.exists() and direct.is_file():
        return direct

    rel_masked = rel.with_name(f"{rel.stem}_mask{rel.suffix}")
    direct_masked = (masks_dir / rel_masked).resolve()
    if direct_masked.exists() and direct_masked.is_file():
        return direct_masked

    rel_key = _relative_stem_key(image_path, images_dir, is_mask=False)
    rel_matches = mask_by_rel_stem.get(rel_key, [])
    if len(rel_matches) == 1:
        return rel_matches[0]
    if len(rel_matches) > 1:
        raise RuntimeError(
            "マスクが複数候補あり一意に解決できません: "
            f"{rel_key}. マスク名を整理してください。"
        )

    stem_key = _stem_key(image_path, is_mask=False)
    stem_matches = mask_by_stem.get(stem_key, [])
    if len(stem_matches) == 1:
        return stem_matches[0]
    if len(stem_matches) > 1:
        raise RuntimeError(
            "同名stemのマスクが複数あり一意に解決できません: "
            f"{stem_key}. CSVのfilenameにサブディレクトリを含めてください。"
        )

    raise RuntimeError(f"対応マスクが見つかりません: {rel.as_posix()}")


def _parse_float(value: str, column: str, row_number: int) -> float:
    try:
        parsed = float(value)
    except Exception as exc:
        raise RuntimeError(
            f"{row_number}行目の {column} が数値ではありません: {value}"
        ) from exc

    if not math.isfinite(parsed):
        raise RuntimeError(f"{row_number}行目の {column} が有限値ではありません。")
    return parsed


def validate_quadpoint_dataset(
    images_dir: Path,
    labels_csv: Path,
    masks_dir: Path | None = None,
) -> QuadpointDatasetInfo:
    """
    4点学習データセットを検証する。
    CSV は以下列を持つ必要がある:
    filename, lt_x, lt_y, rt_x, rt_y, rb_x, rb_y, lb_x, lb_y
    """
    if not images_dir.exists():
        raise FileNotFoundError(f"画像ディレクトリが存在しません: {images_dir}")
    if not labels_csv.exists():
        raise FileNotFoundError(f"4点ラベルCSVが存在しません: {labels_csv}")
    if masks_dir is not None and not masks_dir.exists():
        raise FileNotFoundError(f"マスクディレクトリが存在しません: {masks_dir}")

    by_relative, by_name = _build_image_indices(images_dir)
    if not by_relative:
        raise RuntimeError(f"画像ディレクトリに画像ファイルがありません: {images_dir}")
    mask_by_rel_stem: dict[str, list[Path]] = {}
    mask_by_stem: dict[str, list[Path]] = {}
    if masks_dir is not None:
        mask_by_rel_stem, mask_by_stem = _build_mask_indices(masks_dir)
        if not mask_by_rel_stem:
            raise RuntimeError(f"マスクディレクトリにマスク画像がありません: {masks_dir}")

    with open(labels_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError("CSVヘッダを読み取れません。")

        missing = [col for col in REQUIRED_COLUMNS if col not in reader.fieldnames]
        if missing:
            raise RuntimeError(
                "4点ラベルCSVに必要列が不足しています: "
                + ", ".join(missing)
            )

        count = 0
        for row_index, row in enumerate(reader, start=2):
            filename = (row.get("filename") or "").strip()
            image_path = resolve_image_path(filename, images_dir, by_relative, by_name)
            if masks_dir is not None:
                _ = resolve_mask_path_for_image(
                    image_path=image_path,
                    images_dir=images_dir,
                    masks_dir=masks_dir,
                    mask_by_rel_stem=mask_by_rel_stem,
                    mask_by_stem=mask_by_stem,
                )

            for column in REQUIRED_COLUMNS[1:]:
                _parse_float(row.get(column, ""), column, row_index)

            if not image_path.exists():
                raise RuntimeError(f"{row_index}行目の画像が存在しません: {filename}")
            count += 1

    if count <= 0:
        raise RuntimeError("4点ラベルCSVに有効なデータ行がありません。")

    return QuadpointDatasetInfo(
        images_dir=images_dir,
        labels_csv=labels_csv,
        sample_count=count,
        masks_dir=masks_dir,
    )


def load_quadpoint_records(
    images_dir: Path,
    labels_csv: Path,
    masks_dir: Optional[Path] = None,
) -> list[dict[str, object]]:
    """
    学習用に 4点レコードを読み込む。
    戻り値:
    - image_path: Path
    - points_norm: (4,2) の np.float32（0..1）
    """
    dataset_info = validate_quadpoint_dataset(images_dir, labels_csv, masks_dir=masks_dir)
    by_relative, by_name = _build_image_indices(dataset_info.images_dir)
    mask_by_rel_stem: dict[str, list[Path]] = {}
    mask_by_stem: dict[str, list[Path]] = {}
    if dataset_info.masks_dir is not None:
        mask_by_rel_stem, mask_by_stem = _build_mask_indices(dataset_info.masks_dir)

    records: list[dict[str, object]] = []
    with open(dataset_info.labels_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames is not None

        for row_index, row in enumerate(reader, start=2):
            filename = (row.get("filename") or "").strip()
            image_path = resolve_image_path(filename, dataset_info.images_dir, by_relative, by_name)

            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(f"{row_index}行目の画像を読み込めません: {image_path}")
            h, w = image.shape[:2]

            raw = np.array(
                [
                    [
                        _parse_float(row.get("lt_x", ""), "lt_x", row_index),
                        _parse_float(row.get("lt_y", ""), "lt_y", row_index),
                    ],
                    [
                        _parse_float(row.get("rt_x", ""), "rt_x", row_index),
                        _parse_float(row.get("rt_y", ""), "rt_y", row_index),
                    ],
                    [
                        _parse_float(row.get("rb_x", ""), "rb_x", row_index),
                        _parse_float(row.get("rb_y", ""), "rb_y", row_index),
                    ],
                    [
                        _parse_float(row.get("lb_x", ""), "lb_x", row_index),
                        _parse_float(row.get("lb_y", ""), "lb_y", row_index),
                    ],
                ],
                dtype=np.float32,
            )

            if float(np.max(raw)) <= 1.5 and float(np.min(raw)) >= -0.5:
                points_norm = raw.copy()
            else:
                points_norm = raw.copy()
                points_norm[:, 0] /= max(float(w), 1.0)
                points_norm[:, 1] /= max(float(h), 1.0)

            points_norm[:, 0] = np.clip(points_norm[:, 0], 0.0, 1.0)
            points_norm[:, 1] = np.clip(points_norm[:, 1], 0.0, 1.0)

            mask_path: Path | None = None
            if dataset_info.masks_dir is not None:
                mask_path = resolve_mask_path_for_image(
                    image_path=image_path,
                    images_dir=dataset_info.images_dir,
                    masks_dir=dataset_info.masks_dir,
                    mask_by_rel_stem=mask_by_rel_stem,
                    mask_by_stem=mask_by_stem,
                )

            records.append(
                {
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "points_norm": points_norm.astype(np.float32),
                }
            )

    return records
