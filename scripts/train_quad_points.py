#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_quad_points.py
- 画像から 4点(LT, RT, RB, LB) を回帰する学習スクリプト
- 任意で道路マスクを追加入力（RGB+Mask）できる
- 4点ラベルCSVを使い、学習済み重みを .pth で保存する
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.models as models

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.quadpoint_dataset import load_quadpoint_records, validate_quadpoint_dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class QuadPointDataset(Dataset):
    """
    4点回帰用データセット。
    入力: RGB画像 または RGB+Mask(4ch)
    出力: 正規化座標 [lt_x, lt_y, rt_x, rt_y, rb_x, rb_y, lb_x, lb_y]
    """

    def __init__(
        self,
        records: list[dict[str, object]],
        input_size: int,
        use_mask_input: bool = False,
    ) -> None:
        self.records = records
        self.input_size = int(input_size)
        self.use_mask_input = bool(use_mask_input)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        rec = self.records[index]
        image_path = Path(rec["image_path"])  # type: ignore[arg-type]
        points_norm = np.asarray(rec["points_norm"], dtype=np.float32)  # (4,2)

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise RuntimeError(f"画像を読み込めませんでした: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(
            image_rgb,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_LINEAR,
        )

        image_float = image_resized.astype(np.float32) / 255.0
        image_float = (image_float - self.mean) / self.std
        if self.use_mask_input:
            mask_path_value = rec.get("mask_path")
            if mask_path_value is None:
                raise RuntimeError("mask_path が未設定です。masks_dir とレコード整合を確認してください。")

            mask_path = Path(mask_path_value)  # type: ignore[arg-type]
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"マスクを読み込めませんでした: {mask_path}")
            mask_resized = cv2.resize(
                mask,
                (self.input_size, self.input_size),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.float32)
            if float(np.max(mask_resized)) > 1.0:
                mask_resized /= 255.0
            mask_resized = np.clip(mask_resized, 0.0, 1.0)

            image_float = np.concatenate([image_float, mask_resized[..., None]], axis=2)

        image_tensor = torch.from_numpy(image_float.transpose(2, 0, 1))  # CHW

        target = torch.from_numpy(points_norm.reshape(-1))  # (8,)
        return image_tensor, target


class QuadPointRegressor(nn.Module):
    """
    ResNet18 を使った 4点回帰モデル。
    入力チャネル数は 3(RGB) / 4(RGB+Mask) に対応する。
    出力は sigmoid で 0..1 に制約する。
    """

    def __init__(self, input_channels: int = 3) -> None:
        super().__init__()
        backbone = models.resnet18(weights=None)
        if input_channels != 3:
            if input_channels != 4:
                raise RuntimeError(f"未対応の入力チャネル数です: {input_channels}")
            old_conv = backbone.conv1
            new_conv = nn.Conv2d(
                input_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
            with torch.no_grad():
                new_conv.weight.zero_()
                new_conv.weight[:, :3] = old_conv.weight
                # 追加のマスクチャネルはRGB平均重みで初期化する。
                new_conv.weight[:, 3:4] = old_conv.weight.mean(dim=1, keepdim=True)
            backbone.conv1 = new_conv
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, 8)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.backbone(x))


def point_order_penalty(pred: torch.Tensor) -> torch.Tensor:
    """
    幾何的一貫性の弱い予測を抑える補助損失。
    """
    pts = pred.view(-1, 4, 2)  # LT, RT, RB, LB
    lt, rt, rb, lb = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]

    # 左右関係: 左点x <= 右点x
    lr_penalty = torch.relu(lt[:, 0] - rt[:, 0]) + torch.relu(lb[:, 0] - rb[:, 0])
    # 上下関係: 上点y <= 下点y
    tb_penalty = torch.relu(lt[:, 1] - lb[:, 1]) + torch.relu(rt[:, 1] - rb[:, 1])
    return (lr_penalty + tb_penalty).mean()


def coord_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_pts = pred.view(-1, 4, 2)
    tgt_pts = target.view(-1, 4, 2)
    err = torch.norm(pred_pts - tgt_pts, dim=2).mean()
    return float(err.item())


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    images_dir = Path(args.images_dir).expanduser().resolve()
    masks_dir = (
        Path(args.masks_dir).expanduser().resolve()
        if args.masks_dir is not None and args.masks_dir.strip()
        else None
    )
    use_mask_input = masks_dir is not None
    labels_csv = Path(args.labels_csv).expanduser().resolve()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    info = validate_quadpoint_dataset(images_dir, labels_csv, masks_dir=masks_dir)
    records = load_quadpoint_records(images_dir, labels_csv, masks_dir=masks_dir)
    if len(records) != info.sample_count:
        raise RuntimeError("データセット検証件数と読込件数が一致しません。")

    dataset = QuadPointDataset(
        records=records,
        input_size=args.input_size,
        use_mask_input=use_mask_input,
    )
    n_total = len(dataset)
    if n_total < 2:
        raise RuntimeError("学習には最低2サンプル以上が必要です。")

    n_val = int(n_total * args.val_ratio)
    n_val = max(1, min(n_total - 1, n_val))
    n_train = n_total - n_val
    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QuadPointRegressor(input_channels=(4 if use_mask_input else 3)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()

    print("device:", device)
    print("samples:", n_total, "(train:", n_train, "val:", n_val, ")")
    print("input_size:", args.input_size)
    print("use_mask_input:", use_mask_input)

    best_val_loss = float("inf")
    best_val_err = float("inf")
    best_path = checkpoint_dir / args.ckpt_name
    last_path = checkpoint_dir / "quadpoint_last.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_err_sum = 0.0
        train_count = 0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            preds = model(images)
            loss_reg = loss_fn(preds, targets)
            loss_geo = point_order_penalty(preds)
            loss = loss_reg + 0.2 * loss_geo
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            train_loss_sum += float(loss.item()) * batch_size
            train_err_sum += coord_error(preds.detach(), targets.detach()) * batch_size
            train_count += batch_size

        train_loss = train_loss_sum / max(train_count, 1)
        train_err = train_err_sum / max(train_count, 1)

        model.eval()
        val_loss_sum = 0.0
        val_err_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                preds = model(images)
                loss_reg = loss_fn(preds, targets)
                loss_geo = point_order_penalty(preds)
                loss = loss_reg + 0.2 * loss_geo

                batch_size = images.size(0)
                val_loss_sum += float(loss.item()) * batch_size
                val_err_sum += coord_error(preds, targets) * batch_size
                val_count += batch_size

        val_loss = val_loss_sum / max(val_count, 1)
        val_err = val_err_sum / max(val_count, 1)

        print(
            f"[Epoch {epoch:03d}/{args.epochs}] "
            f"train_loss={train_loss:.6f} train_err={train_err:.6f} "
            f"val_loss={val_loss:.6f} val_err={val_err:.6f}"
        )

        ckpt = {
            "model_state_dict": model.state_dict(),
            "input_size": args.input_size,
            "input_channels": 4 if use_mask_input else 3,
            "use_mask_input": use_mask_input,
            "sample_count": n_total,
            "epoch": epoch,
            "val_loss": val_loss,
            "val_err": val_err,
            "masks_dir": str(masks_dir) if masks_dir is not None else None,
            "columns": ["lt_x", "lt_y", "rt_x", "rt_y", "rb_x", "rb_y", "lb_x", "lb_y"],
        }
        torch.save(ckpt, str(last_path))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_err = val_err
            torch.save(ckpt, str(best_path))
            print(f"  -> best updated: {best_path}")

    print("training finished")
    print("best checkpoint:", best_path)
    print("best val_loss:", best_val_loss)
    print("best val_err(normalized):", best_val_err)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="4点回帰モデル学習")
    parser.add_argument("--images_dir", type=str, required=True, help="画像ディレクトリ")
    parser.add_argument(
        "--masks_dir",
        type=str,
        default=None,
        help="任意: 学習マスクディレクトリ（指定時はRGB+Mask入力で学習）",
    )
    parser.add_argument("--labels_csv", type=str, required=True, help="4点ラベルCSV")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="models",
        help="チェックポイント保存先",
    )
    parser.add_argument("--ckpt_name", type=str, default="quadpoint_best.pth", help="保存ファイル名")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--input_size", type=int, default=384)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
