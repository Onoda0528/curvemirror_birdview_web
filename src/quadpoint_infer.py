from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _extract_state_dict(checkpoint: Any) -> dict[str, Any]:
    if hasattr(checkpoint, "state_dict"):
        state_dict = checkpoint.state_dict()
        if isinstance(state_dict, dict):
            return state_dict

    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        return checkpoint

    raise RuntimeError("4点推論モデルの形式を解釈できません。")


def _normalize_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    prefixes = ("module.", "model.", "net.", "backbone.")
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        normalized[new_key] = value
    return normalized


def _order_points(pts: np.ndarray) -> np.ndarray:
    """点列を [LT, RT, RB, LB] に並べ替える。"""
    points = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)

    idx_lt = int(np.argmin(sums))
    idx_rb = int(np.argmax(sums))
    idx_rt = int(np.argmin(diffs))
    idx_lb = int(np.argmax(diffs))
    indices = {idx_lt, idx_rt, idx_rb, idx_lb}
    if len(indices) == 4:
        return np.float32([points[idx_lt], points[idx_rt], points[idx_rb], points[idx_lb]])

    # sum/diff で重複が出るケースは、上側2点・下側2点で分けて補正する。
    pts_sorted = sorted(points.tolist(), key=lambda p: (p[1], p[0]))
    top2 = sorted(pts_sorted[:2], key=lambda p: p[0])
    bottom2 = sorted(pts_sorted[2:], key=lambda p: p[0])
    lt = np.asarray(top2[0], dtype=np.float32)
    rt = np.asarray(top2[1], dtype=np.float32)
    lb = np.asarray(bottom2[0], dtype=np.float32)
    rb = np.asarray(bottom2[1], dtype=np.float32)
    return np.float32([lt, rt, rb, lb])


def _is_valid_quad(points: np.ndarray, h: int, w: int) -> bool:
    if points.shape != (4, 2):
        return False
    if points[0, 0] >= points[1, 0] or points[3, 0] >= points[2, 0]:
        return False
    area = cv2.contourArea(points.astype(np.float32))
    if area < max(300.0, 0.003 * h * w):
        return False
    return True


class QuadPointPredictor:
    """
    quadpoint_best.pth 形式を読み込んで、画像から4点を推論する。
    学習スクリプト scripts/train_quad_points.py の構成に合わせる。
    """

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        torch, models = self._import_modules()
        self._torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(str(model_path), map_location=self.device)
        state_dict = _normalize_state_dict(_extract_state_dict(checkpoint))
        self.input_size = int(checkpoint.get("input_size", 384)) if isinstance(checkpoint, dict) else 384

        detected_channels = 3
        if isinstance(checkpoint, dict) and "input_channels" in checkpoint:
            detected_channels = int(checkpoint["input_channels"])
        elif "conv1.weight" in state_dict and hasattr(state_dict["conv1.weight"], "shape"):
            detected_channels = int(state_dict["conv1.weight"].shape[1])

        self.input_channels = detected_channels
        self.use_mask_input = self.input_channels >= 4
        if self.input_channels not in (3, 4):
            raise RuntimeError(
                f"未対応の4点モデル入力チャネル数です: {self.input_channels} "
                "(対応: 3 or 4)"
            )

        backbone = models.resnet18(weights=None)
        if self.input_channels == 4:
            old_conv = backbone.conv1
            new_conv = torch.nn.Conv2d(
                4,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
            with torch.no_grad():
                new_conv.weight.zero_()
                new_conv.weight[:, :3] = old_conv.weight
                new_conv.weight[:, 3:4] = old_conv.weight.mean(dim=1, keepdim=True)
            backbone.conv1 = new_conv
        in_features = backbone.fc.in_features
        backbone.fc = torch.nn.Linear(in_features, 8)
        self.model = backbone

        model_keys = set(self.model.state_dict().keys())
        loaded_keys = set(state_dict.keys()) & model_keys
        if not loaded_keys:
            raise RuntimeError(
                f"{model_path.name} は読み込めましたが、4点推論モデルの重みキーが一致しません。"
            )

        # 保存形式差異に対応するため strict=False で読み込む。
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _import_modules():
        try:
            import torch  # type: ignore
            import torchvision.models as models  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "4点推論に必要な PyTorch / torchvision の読み込みに失敗しました。"
            ) from exc
        return torch, models

    def predict(self, image_bgr: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray | None:
        torch = self._torch
        h, w = image_bgr.shape[:2]
        if h < 2 or w < 2:
            return None

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(
            image_rgb,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_LINEAR,
        )

        image_float = image_resized.astype(np.float32) / 255.0
        image_float = (image_float - self.mean) / self.std
        if self.input_channels == 4:
            if mask is None:
                raise RuntimeError(
                    "この4点モデルはマスク入力を必要としますが、推論時マスクが渡されていません。"
                )
            mask_gray = (
                cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                if mask.ndim == 3
                else mask.astype(np.uint8)
            )
            mask_resized = cv2.resize(
                mask_gray,
                (self.input_size, self.input_size),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.float32)
            if float(np.max(mask_resized)) > 1.0:
                mask_resized /= 255.0
            mask_resized = np.clip(mask_resized, 0.0, 1.0)
            image_float = np.concatenate([image_float, mask_resized[..., None]], axis=2)

        x = torch.from_numpy(image_float.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.model(x)
            if float(pred.min()) < 0.0 or float(pred.max()) > 1.0:
                # 出力がロジットの場合のみ sigmoid を適用する。
                pred = torch.sigmoid(pred)
            pred_np = pred.squeeze(0).detach().cpu().numpy().astype(np.float32)

        if pred_np.size != 8:
            raise RuntimeError(f"4点推論出力形状が想定外です: {pred_np.shape}")

        # 学習時の出力順 (LT, RT, RB, LB) をまず優先し、破綻時のみ並べ替え補正する。
        points_raw = pred_np.reshape(4, 2)
        points_raw[:, 0] *= float(w - 1)
        points_raw[:, 1] *= float(h - 1)
        points_raw[:, 0] = np.clip(points_raw[:, 0], 0, w - 1)
        points_raw[:, 1] = np.clip(points_raw[:, 1], 0, h - 1)

        if _is_valid_quad(points_raw, h, w):
            return points_raw.astype(np.float32)

        points_ordered = _order_points(points_raw)
        points_ordered[:, 0] = np.clip(points_ordered[:, 0], 0, w - 1)
        points_ordered[:, 1] = np.clip(points_ordered[:, 1], 0, h - 1)
        if _is_valid_quad(points_ordered, h, w):
            return points_ordered.astype(np.float32)

        return None


_PREDICTOR_CACHE: dict[str, Any] = {"path": None, "mtime_ns": None, "predictor": None}


def _load_predictor(model_path: Path) -> QuadPointPredictor:
    path_resolved = model_path.expanduser().resolve()
    if not path_resolved.exists():
        raise FileNotFoundError(f"4点推論モデルが存在しません: {path_resolved}")

    mtime_ns = path_resolved.stat().st_mtime_ns
    cached_path = _PREDICTOR_CACHE["path"]
    cached_mtime = _PREDICTOR_CACHE["mtime_ns"]
    cached_predictor = _PREDICTOR_CACHE["predictor"]

    if (
        isinstance(cached_path, str)
        and cached_path == str(path_resolved)
        and cached_mtime == mtime_ns
        and isinstance(cached_predictor, QuadPointPredictor)
    ):
        return cached_predictor

    predictor = QuadPointPredictor(path_resolved)
    _PREDICTOR_CACHE["path"] = str(path_resolved)
    _PREDICTOR_CACHE["mtime_ns"] = mtime_ns
    _PREDICTOR_CACHE["predictor"] = predictor
    return predictor


def estimate_quad_by_learned_model(
    image_bgr: np.ndarray,
    model_path: Path,
    mask: np.ndarray | None = None,
) -> np.ndarray | None:
    """学習済み4点モデルで [LT, RT, RB, LB] を推定する。"""
    predictor = _load_predictor(model_path)
    return predictor.predict(image_bgr, mask=mask)
