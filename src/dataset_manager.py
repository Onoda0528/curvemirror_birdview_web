from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import zipfile

import cv2
import numpy as np

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class PreparedDatasetInfo:
    images_dir: Path
    masks_dir: Path
    sample_count: int
    output_root: Path


def _iter_image_files(directory: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
    return files


def _mask_stem(stem: str) -> str:
    return stem[:-5] if stem.endswith("_mask") else stem


def _relative_key(path: Path, root: Path, is_mask: bool) -> str:
    """
    ルートからの相対パス（拡張子除く）をキー化する。
    masks 側は末尾の `_mask` を除去して画像名と合わせる。
    """
    rel = path.relative_to(root)
    stem = _mask_stem(rel.stem) if is_mask else rel.stem
    if str(rel.parent) == ".":
        return stem
    return f"{rel.parent.as_posix()}/{stem}"


def _stem_key(path: Path, is_mask: bool) -> str:
    return _mask_stem(path.stem) if is_mask else path.stem


def _index_by_key(files: list[Path], key_fn) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in files:
        key = key_fn(path)
        index.setdefault(key, []).append(path)
    return index


def _match_unique_pairs(
    image_files: list[Path],
    mask_files: list[Path],
    image_key_fn,
    mask_key_fn,
) -> tuple[list[tuple[Path, Path, str]], set[Path], set[Path]]:
    """
    同じキーに対して「画像1枚 + マスク1枚」の場合のみペア化する。
    衝突（同一キー複数件）は曖昧なのでこの段階では採用しない。
    """
    image_index = _index_by_key(image_files, image_key_fn)
    mask_index = _index_by_key(mask_files, mask_key_fn)

    pairs: list[tuple[Path, Path, str]] = []
    used_images: set[Path] = set()
    used_masks: set[Path] = set()

    for key in sorted(set(image_index.keys()) & set(mask_index.keys())):
        images = image_index[key]
        masks = mask_index[key]
        if len(images) == 1 and len(masks) == 1:
            image_path = images[0]
            mask_path = masks[0]
            pairs.append((image_path, mask_path, key))
            used_images.add(image_path)
            used_masks.add(mask_path)

    return pairs, used_images, used_masks


def _collect_pairs(images_dir: Path, masks_dir: Path) -> list[tuple[Path, Path, str]]:
    image_files = _iter_image_files(images_dir)
    mask_files = _iter_image_files(masks_dir)

    # 1) 相対パス一致を優先（サブディレクトリ構成に強い）
    pairs, used_images, used_masks = _match_unique_pairs(
        image_files=image_files,
        mask_files=mask_files,
        image_key_fn=lambda p: _relative_key(p, images_dir, is_mask=False),
        mask_key_fn=lambda p: _relative_key(p, masks_dir, is_mask=True),
    )

    # 2) 未対応分はファイル名（stem）一致で補完
    remaining_images = [p for p in image_files if p not in used_images]
    remaining_masks = [p for p in mask_files if p not in used_masks]
    fallback_pairs, _, _ = _match_unique_pairs(
        image_files=remaining_images,
        mask_files=remaining_masks,
        image_key_fn=lambda p: _stem_key(p, is_mask=False),
        mask_key_fn=lambda p: _stem_key(p, is_mask=True),
    )
    pairs.extend(fallback_pairs)
    pairs.sort(key=lambda item: item[2])
    return pairs


def _save_png_image(src: Path, dst: Path) -> None:
    image = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"画像を読み込めませんでした: {src}")
    ok = cv2.imwrite(str(dst), image)
    if not ok:
        raise RuntimeError(f"画像を書き出せませんでした: {dst}")


def _save_png_mask(src: Path, dst: Path) -> None:
    mask = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"マスクを読み込めませんでした: {src}")

    if np.max(mask) <= 1:
        mask = (mask * 255).astype(np.uint8)

    ok = cv2.imwrite(str(dst), mask)
    if not ok:
        raise RuntimeError(f"マスクを書き出せませんでした: {dst}")


def prepare_dataset_from_dirs(
    images_dir: Path,
    masks_dir: Path,
    output_root: Path,
) -> PreparedDatasetInfo:
    """
    任意形式の画像・マスクを学習スクリプト向けの PNG ペアに正規化する。
    """

    if not images_dir.exists():
        raise FileNotFoundError(f"学習画像ディレクトリが存在しません: {images_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"学習マスクディレクトリが存在しません: {masks_dir}")

    pairs = _collect_pairs(images_dir, masks_dir)
    if not pairs:
        raise RuntimeError(
            "画像とマスクの対応ペアが見つかりません。"
            "同名ファイル、またはマスク側が *_mask の命名になっているか確認してください。"
        )

    if output_root.exists():
        shutil.rmtree(output_root)
    normalized_images = output_root / "images"
    normalized_masks = output_root / "masks"
    normalized_images.mkdir(parents=True, exist_ok=True)
    normalized_masks.mkdir(parents=True, exist_ok=True)

    used_names: set[str] = set()
    for index, (img_path, mask_path, stem) in enumerate(pairs, start=1):
        safe_stem = "".join(ch for ch in stem if ch.isalnum() or ch in {"-", "_"})
        if not safe_stem:
            safe_stem = f"sample_{index:06d}"
        filename = f"{safe_stem}.png"
        if filename in used_names:
            filename = f"{safe_stem}_{index:06d}.png"
        used_names.add(filename)

        _save_png_image(img_path, normalized_images / filename)
        _save_png_mask(mask_path, normalized_masks / filename)

    return PreparedDatasetInfo(
        images_dir=normalized_images,
        masks_dir=normalized_masks,
        sample_count=len(pairs),
        output_root=output_root,
    )


def extract_zip_safe(zip_path: Path, dest_dir: Path) -> None:
    """Zip Slip を防いで zip を展開する。"""

    dest_dir.mkdir(parents=True, exist_ok=True)
    base = dest_dir.resolve()

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute():
                continue
            target = (dest_dir / member_path).resolve()
            if not str(target).startswith(str(base)):
                continue
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member, "r") as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
