from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class SegmentationBackendInfo:
    """現在有効な道路セグメンテーション実装の情報。"""

    name: str
    using_model: bool
    detail: str


class RoadMaskPredictor:
    """道路マスク推定器の共通インターフェース。"""

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class PlaceholderRoadPredictor(RoadMaskPredictor):
    """
    仮マスク生成器。
    学習済みモデル未配置時のフォールバックとして使用する。
    """

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        pts = np.array(
            [
                [int(w * 0.42), int(h * 0.20)],
                [int(w * 0.58), int(h * 0.20)],
                [int(w * 0.82), int(h * 0.95)],
                [int(w * 0.18), int(h * 0.95)],
            ],
            dtype=np.int32,
        )

        cv2.fillPoly(mask, [pts], 255)
        return mask


class DeepLabV3PlusPredictor(RoadMaskPredictor):
    """
    DeepLabV3+ 推論器。
    後から学習済みモデル差し替えをしやすいよう、推論処理を独立クラス化している。
    """

    def __init__(self, model_path: Path, threshold: float = 0.5) -> None:
        self.model_path = model_path
        self.threshold = threshold

        torch, smp = self._import_torch_modules()
        self._torch = torch

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = smp.DeepLabV3Plus(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )

        checkpoint = torch.load(str(model_path), map_location=self.device)
        state_dict = self._extract_state_dict(checkpoint)
        normalized_state_dict = self._normalize_state_dict(state_dict)

        model_keys = set(self.model.state_dict().keys())
        loaded_keys = set(normalized_state_dict.keys()) & model_keys
        if not loaded_keys:
            raise RuntimeError(
                "best_model.pth は読み込めましたが、DeepLabV3+ の重みに一致するキーが見つかりません。"
            )

        # 学習時の保存形式差異に対応するため strict=False でロードする。
        self.model.load_state_dict(normalized_state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _import_torch_modules():
        try:
            import torch  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "PyTorch の読み込みに失敗しました。`pip install -r requirements.txt` を確認してください。"
            ) from exc

        try:
            import segmentation_models_pytorch as smp  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "segmentation-models-pytorch の読み込みに失敗しました。"
            ) from exc

        return torch, smp

    @staticmethod
    def _extract_state_dict(checkpoint: Any) -> dict[str, Any]:
        if hasattr(checkpoint, "state_dict"):
            state_dict = checkpoint.state_dict()
            if isinstance(state_dict, dict):
                return state_dict

        if isinstance(checkpoint, dict):
            candidate_keys = ["state_dict", "model_state_dict", "model", "net"]
            for key in candidate_keys:
                value = checkpoint.get(key)
                if isinstance(value, dict):
                    return value
            return checkpoint

        raise RuntimeError(
            "best_model.pth の形式を解釈できません。state_dict 形式で保存されているか確認してください。"
        )

    @staticmethod
    def _normalize_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        prefixes = ("module.", "model.", "net.", "segmentation_model.")
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

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        torch = self._torch

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_float = image_rgb.astype(np.float32) / 255.0
        input_tensor = torch.from_numpy(image_float.transpose(2, 0, 1)).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            if isinstance(output, (list, tuple)):
                output = output[0]
            prob = torch.sigmoid(output)

        prob_np = prob.squeeze().detach().cpu().numpy()
        if prob_np.ndim != 2:
            raise RuntimeError(
                "DeepLabV3+ の出力形状が想定外です。出力チャネル設定（classes=1）を確認してください。"
            )

        if prob_np.shape != image_bgr.shape[:2]:
            prob_np = cv2.resize(
                prob_np,
                (image_bgr.shape[1], image_bgr.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        binary_mask = (prob_np >= self.threshold).astype(np.uint8) * 255
        return binary_mask


class RoadSegmentationPipeline:
    """モデル有無を判定して適切な推論器を選択するパイプライン。"""

    def __init__(self, model_path: Optional[Path] = None) -> None:
        if model_path is None:
            model_path = Path(__file__).resolve().parents[1] / "models" / "best_model.pth"
        self.model_path = model_path

        if self.model_path.exists():
            predictor = DeepLabV3PlusPredictor(self.model_path)
            backend_info = SegmentationBackendInfo(
                name="DeepLabV3+（学習済みモデル）",
                using_model=True,
                detail=f"モデル: {self.model_path.name}",
            )
        else:
            predictor = PlaceholderRoadPredictor()
            backend_info = SegmentationBackendInfo(
                name="仮マスク（フォールバック）",
                using_model=False,
                detail="models/best_model.pth が見つからないため仮マスクを使用",
            )

        self.predictor = predictor
        self.backend_info = backend_info

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        return self.predictor.predict(image_bgr)


_default_pipeline: Optional[RoadSegmentationPipeline] = None
_default_signature: Optional[tuple[str, bool, Optional[int]]] = None


def get_default_pipeline(model_path: Optional[Path] = None) -> RoadSegmentationPipeline:
    """
    デフォルトパイプラインを遅延初期化して返す。
    初期化時に失敗した場合は例外を上位へ伝播し、Web側で表示する。
    """

    global _default_pipeline, _default_signature

    if model_path is None:
        model_path = Path(__file__).resolve().parents[1] / "models" / "best_model.pth"

    exists = model_path.exists()
    mtime_ns = model_path.stat().st_mtime_ns if exists else None
    signature = (str(model_path.resolve()), exists, mtime_ns)

    if _default_pipeline is None or _default_signature != signature:
        _default_pipeline = RoadSegmentationPipeline(model_path=model_path)
        _default_signature = signature

    return _default_pipeline


def reset_default_pipeline() -> None:
    """再読み込み用にパイプラインを初期化し直す。"""

    global _default_pipeline, _default_signature
    _default_pipeline = None
    _default_signature = None


def predict_road_mask(image_bgr: np.ndarray) -> np.ndarray:
    """後方互換用の簡易関数。"""

    return get_default_pipeline().predict(image_bgr)


def get_segmentation_backend_info() -> SegmentationBackendInfo:
    """現在利用中のセグメンテーション方式情報を返す。"""

    return get_default_pipeline().backend_info
