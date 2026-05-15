from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
SUPPORTED_MODEL_EXTENSIONS = (".pth", ".xml")
PREFERRED_MODEL_NAMES = ("best_model.pth", "best_model.xml")
EXCLUDED_PTH_PREFIXES = ("quadpoint_",)
MODEL_CHOICE_AUTO = "auto"
MODEL_CHOICE_PLACEHOLDER = "placeholder"


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

    def __init__(
        self,
        model_path: Path,
        threshold: float = 0.5,
        input_size: int = 512,
    ) -> None:
        self.model_path = model_path
        self.threshold = threshold
        self.input_size = int(input_size)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

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
                f"{model_path.name} は読み込めましたが、DeepLabV3+ の重みに一致するキーが見つかりません。"
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
            "モデル形式を解釈できません。state_dict 形式で保存されているか確認してください。"
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

        h, w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(
            image_rgb,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_LINEAR,
        )

        image_float = image_resized.astype(np.float32) / 255.0
        image_float = (image_float - self.mean) / self.std
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

        if prob_np.shape != (self.input_size, self.input_size):
            prob_np = cv2.resize(
                prob_np,
                (self.input_size, self.input_size),
                interpolation=cv2.INTER_LINEAR,
            )

        binary_mask_resized = (prob_np > self.threshold).astype(np.uint8) * 255
        binary_mask = cv2.resize(
            binary_mask_resized,
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )
        return binary_mask


class OpenVinoIRPredictor(RoadMaskPredictor):
    """
    OpenVINO IR（.xml + .bin）推論器。
    """

    def __init__(self, model_path: Path, threshold: float = 0.5) -> None:
        self.model_path = model_path
        self.bin_path = model_path.with_suffix(".bin")
        self.threshold = threshold

        if not self.bin_path.exists():
            raise RuntimeError(
                f"{model_path.name} を使うには {self.bin_path.name} が必要です。"
            )

        self.net = cv2.dnn.readNet(str(self.model_path), str(self.bin_path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    @staticmethod
    def _to_probability_map(raw_output: np.ndarray) -> np.ndarray:
        """
        モデル出力を 2次元の道路確率マップへ変換する。
        """

        prob = raw_output.astype(np.float32)

        if prob.ndim == 4:
            if prob.shape[1] == 1:
                prob = prob[0, 0]
            else:
                channel_index = 1 if prob.shape[1] > 1 else 0
                prob = prob[0, channel_index]
        elif prob.ndim == 3:
            if prob.shape[0] in (1, 2):
                prob = prob[1] if prob.shape[0] == 2 else prob[0]
            else:
                prob = prob[0]
        elif prob.ndim != 2:
            raise RuntimeError(
                f"OpenVINO 出力形状が想定外です: shape={tuple(raw_output.shape)}"
            )

        # ロジット出力の可能性を考慮して sigmoid へ変換する。
        if float(np.max(prob)) > 1.0 or float(np.min(prob)) < 0.0:
            prob = 1.0 / (1.0 + np.exp(-np.clip(prob, -40.0, 40.0)))

        return prob

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            image_bgr,
            scalefactor=1.0 / 255.0,
            size=(w, h),
            mean=(0.0, 0.0, 0.0),
            swapRB=True,
            crop=False,
        )

        self.net.setInput(blob)
        raw_output = self.net.forward()
        prob = self._to_probability_map(raw_output)

        if prob.shape != (h, w):
            prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)

        binary_mask = (prob >= self.threshold).astype(np.uint8) * 255
        return binary_mask


class RoadSegmentationPipeline:
    """モデル有無を判定して適切な推論器を選択するパイプライン。"""

    def __init__(self, model_path: Optional[Path] = None) -> None:
        self.model_path = model_path

        if self.model_path is not None and self.model_path.exists():
            suffix = self.model_path.suffix.lower()
            if suffix == ".pth":
                predictor = DeepLabV3PlusPredictor(self.model_path)
                backend_info = SegmentationBackendInfo(
                    name="DeepLabV3+（学習済みモデル）",
                    using_model=True,
                    detail=(
                        f"モデル: {self.model_path.name} / 前処理: 512x512 + ImageNet正規化 "
                        "(infer_road.py準拠)"
                    ),
                )
            elif suffix == ".xml":
                predictor = OpenVinoIRPredictor(self.model_path)
                backend_info = SegmentationBackendInfo(
                    name="OpenVINO IR（XML/BIN）",
                    using_model=True,
                    detail=f"モデル: {self.model_path.name} (+ {self.model_path.with_suffix('.bin').name})",
                )
            else:
                raise RuntimeError(
                    f"未対応のモデル拡張子です: {self.model_path.suffix}"
                )
        else:
            predictor = PlaceholderRoadPredictor()
            backend_info = SegmentationBackendInfo(
                name="仮マスク（フォールバック）",
                using_model=False,
                detail="models/ に .pth または .xml(+.bin) が見つからないため仮マスクを使用",
            )

        self.predictor = predictor
        self.backend_info = backend_info

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        return self.predictor.predict(image_bgr)


_default_pipeline: Optional[RoadSegmentationPipeline] = None
_default_signature: Optional[tuple[str, Optional[str], Optional[int]]] = None
_AUTO_MODEL_SENTINEL = object()


def _scan_model_candidates(models_dir: Path = MODELS_DIR) -> tuple[list[Path], list[Path]]:
    """利用可能な pth / xml(+bin) モデルを抽出する。"""

    if not models_dir.exists():
        return [], []

    pth_candidates: list[Path] = []
    xml_candidates: list[Path] = []
    for path in sorted(models_dir.iterdir()):
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_MODEL_EXTENSIONS:
            continue
        if suffix == ".pth":
            # 4点推定モデルなどセグメンテーション以外の重みを除外する。
            if path.name.lower().startswith(EXCLUDED_PTH_PREFIXES):
                continue
            pth_candidates.append(path)
        elif suffix == ".xml" and path.with_suffix(".bin").exists():
            xml_candidates.append(path)

    return pth_candidates, xml_candidates


def list_available_model_paths(models_dir: Path = MODELS_DIR) -> list[Path]:
    """UI表示用に利用可能モデルの一覧を返す。"""

    pth_candidates, xml_candidates = _scan_model_candidates(models_dir)
    return sorted(
        [*pth_candidates, *xml_candidates],
        key=lambda p: (p.suffix.lower(), p.name.lower()),
    )


def find_model_path(models_dir: Path = MODELS_DIR) -> Optional[Path]:
    """
    models ディレクトリから利用可能なモデルを探す。

    優先順位:
    1) best_model.pth
    2) best_model.xml(+.bin)
    3) 名前任意の .pth（更新時刻が新しいもの）
    4) 名前任意の .xml(+.bin)（更新時刻が新しいもの）
    """

    pth_candidates, xml_candidates = _scan_model_candidates(models_dir)

    best_pth = models_dir / PREFERRED_MODEL_NAMES[0]
    if best_pth in pth_candidates:
        return best_pth

    best_xml = models_dir / PREFERRED_MODEL_NAMES[1]
    if best_xml in xml_candidates:
        return best_xml

    if pth_candidates:
        return max(pth_candidates, key=lambda p: p.stat().st_mtime_ns)

    if xml_candidates:
        return max(xml_candidates, key=lambda p: p.stat().st_mtime_ns)

    return None


def resolve_model_choice(
    model_choice: Optional[str],
    models_dir: Path = MODELS_DIR,
) -> Optional[Path]:
    """
    UIで選択されたモデル指定を解釈して返す。

    - auto: 自動選択
    - placeholder: 仮マスク固定
    - それ以外: ファイル名一致で明示モデル選択
    """

    choice = (model_choice or MODEL_CHOICE_AUTO).strip()
    if choice == MODEL_CHOICE_AUTO:
        return find_model_path(models_dir)
    if choice == MODEL_CHOICE_PLACEHOLDER:
        return None

    available = {path.name: path for path in list_available_model_paths(models_dir)}
    safe_name = Path(choice).name
    selected = available.get(safe_name)
    if selected is None:
        raise ValueError(
            "選択したモデルが見つかりません。models/ 内の .pth または .xml(+.bin) を確認してください。"
        )
    return selected


def get_default_pipeline(
    model_path: Optional[Path] | object = _AUTO_MODEL_SENTINEL,
) -> RoadSegmentationPipeline:
    """
    デフォルトパイプラインを遅延初期化して返す。
    初期化時に失敗した場合は例外を上位へ伝播し、Web側で表示する。
    """

    global _default_pipeline, _default_signature

    if model_path is _AUTO_MODEL_SENTINEL:
        selection_mode = "auto"
        resolved_model_path = find_model_path()
    else:
        selection_mode = "manual"
        if model_path is not None and not isinstance(model_path, Path):
            raise TypeError("model_path には Path または None を指定してください。")
        resolved_model_path = model_path

    if resolved_model_path is None:
        signature = (selection_mode, None, None)
    else:
        signature = (
            selection_mode,
            str(resolved_model_path.resolve()),
            resolved_model_path.stat().st_mtime_ns,
        )

    if _default_pipeline is None or _default_signature != signature:
        _default_pipeline = RoadSegmentationPipeline(model_path=resolved_model_path)
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
