from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from flask import Flask, render_template, request
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from src.birdview import (
    draw_quad,
    estimate_quad_by_contour,
    estimate_quad_width_filter_ransac,
    make_birdview,
    preprocess_mask,
)
from src.segmentation import get_default_pipeline

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
OUTPUT_DIR = BASE_DIR / "static" / "outputs"
MODEL_PATH = BASE_DIR / "models" / "best_model.pth"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}

ESTIMATION_METHODS = {
    "contour": ("最大輪郭四角形近似", estimate_quad_by_contour),
    "ransac": ("幅フィルタ + RANSAC", estimate_quad_width_filter_ransac),
}

MODE_TO_METHODS = {
    "compare": ["contour", "ransac"],
    "contour": ["contour"],
    "ransac": ["ransac"],
}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def build_unique_stem(filename: str) -> str:
    """保存ファイル名衝突を避けるための一意な接尾辞を作る。"""
    raw_stem = Path(filename).stem
    safe_stem = "".join(ch for ch in raw_stem if ch.isalnum() or ch in {"-", "_"})
    safe_stem = safe_stem if safe_stem else "upload"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{safe_stem}_{timestamp}_{uuid4().hex[:8]}"


def model_notice() -> str:
    if MODEL_PATH.exists():
        return "models/best_model.pth を検出しました。DeepLabV3+ 推論を使用します。"
    return "models/best_model.pth がないため、仮マスクで処理します。"


def render_index(error_message: str | None = None):
    return render_template(
        "index.html",
        error_message=error_message,
        model_notice=model_notice(),
    )


def validate_upload(file: FileStorage | None) -> str | None:
    if file is None:
        return "画像ファイルが受け取れませんでした。ファイルを選択して再実行してください。"

    if not file.filename:
        return "画像ファイルが選択されていません。"

    if not allowed_file(file.filename):
        return "対応拡張子は jpg / jpeg / png / bmp / webp です。"

    return None


def run_estimation(
    method_key: str,
    image: np.ndarray,
    clean_mask: np.ndarray,
    stem: str,
) -> dict[str, str | bool | None]:
    method_name, estimator = ESTIMATION_METHODS[method_key]
    result: dict[str, str | bool | None] = {
        "key": method_key,
        "name": method_name,
        "success": False,
        "error_message": None,
        "quad_image": None,
        "birdview_image": None,
    }

    try:
        src_pts = estimator(clean_mask)
        if src_pts is None:
            result["error_message"] = (
                "4点を推定できませんでした。道路マスクが小さい、途切れている、"
                "または境界が不明瞭な可能性があります。"
            )
            return result

        quad = draw_quad(image, src_pts)
        quad_rel = f"outputs/{stem}_{method_key}_quad.png"
        cv2.imwrite(str(BASE_DIR / "static" / quad_rel), quad)

        birdview = make_birdview(image, src_pts, out_w=512, out_h=768)
        birdview_rel = f"outputs/{stem}_{method_key}_birdview.png"
        cv2.imwrite(str(BASE_DIR / "static" / birdview_rel), birdview)

        result["success"] = True
        result["quad_image"] = quad_rel
        result["birdview_image"] = birdview_rel
    except Exception as exc:
        result["error_message"] = f"手法処理中にエラーが発生しました: {exc}"

    return result


@app.route("/", methods=["GET"])
def index():
    return render_index()


@app.route("/process", methods=["POST"])
def process():
    file = request.files.get("image")
    validation_error = validate_upload(file)
    if validation_error is not None:
        return render_index(error_message=validation_error)

    assert file is not None
    mode = request.form.get("method_mode", "compare")
    selected_methods = MODE_TO_METHODS.get(mode, MODE_TO_METHODS["compare"])

    original_name = secure_filename(file.filename)
    ext = Path(original_name).suffix.lower()
    stem = build_unique_stem(original_name)
    input_filename = f"{stem}{ext}"
    input_path = UPLOAD_DIR / input_filename

    try:
        file.save(input_path)

        image = cv2.imread(str(input_path))
        if image is None:
            return render_index(error_message="画像を読み込めませんでした。別の画像で再試行してください。")

        pipeline = get_default_pipeline(model_path=MODEL_PATH)
        road_mask = pipeline.predict(image)
        clean_mask = preprocess_mask(road_mask)

        mask_rel = f"outputs/{stem}_mask.png"
        cv2.imwrite(str(BASE_DIR / "static" / mask_rel), clean_mask)

        method_results = [
            run_estimation(method_key, image, clean_mask, stem)
            for method_key in selected_methods
        ]

        return render_template(
            "result.html",
            input_image=f"uploads/{input_filename}",
            mask_image=mask_rel,
            method_results=method_results,
            segmentation_backend=pipeline.backend_info.name,
            segmentation_detail=pipeline.backend_info.detail,
            mode=mode,
        )
    except Exception as exc:
        return render_index(
            error_message=(
                "処理中にエラーが発生しました。入力画像やモデル設定を確認してください。"
                f" 詳細: {exc}"
            )
        )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
