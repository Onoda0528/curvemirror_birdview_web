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
    estimate_quad_width_filter_ransac,
    make_birdview,
    preprocess_mask,
)
from src.segmentation import (
    MODEL_CHOICE_AUTO,
    MODEL_CHOICE_PLACEHOLDER,
    find_model_path,
    get_default_pipeline,
    list_available_model_paths,
    resolve_model_choice,
)
from src.external_pipeline import (
    DEEPLAB_ROOT,
    get_training_status,
    run_distortion_correction,
    scripts_ready,
    start_training,
    stop_training,
)
from src.dataset_manager import extract_zip_safe, prepare_dataset_from_dirs
from src.quadpoint_dataset import QuadpointDatasetInfo, validate_quadpoint_dataset
from src.quadpoint_pipeline import (
    get_quadpoint_training_status,
    quadpoint_training_script_ready,
    start_quadpoint_training,
    stop_quadpoint_training,
)
from src.quadpoint_infer import estimate_quad_by_learned_model

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
OUTPUT_DIR = BASE_DIR / "static" / "outputs"
MODELS_DIR = BASE_DIR / "models"
DISTORT_OUTPUT_DIR = OUTPUT_DIR / "distortion"
USER_DATASETS_DIR = BASE_DIR / "data" / "user_datasets"
QUAD_USER_DATASETS_DIR = USER_DATASETS_DIR / "quadpoint"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
USER_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
QUAD_USER_DATASETS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}
QUADPOINT_MODEL_PATH = MODELS_DIR / "quadpoint_best.pth"


def estimate_quad_by_learned_wrapper(image: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """学習済み4点回帰モデルで4点を推定する。"""
    if not QUADPOINT_MODEL_PATH.exists():
        return None
    return estimate_quad_by_learned_model(image, QUADPOINT_MODEL_PATH, mask=mask)

ESTIMATION_METHODS = {
    "ransac": {
        "name": "幅フィルタ + RANSAC",
        "estimator": lambda _image, mask: estimate_quad_width_filter_ransac(mask),
        "fail_hint": (
            "4点を推定できませんでした。道路マスクが小さい、途切れている、"
            "または境界が不明瞭な可能性があります。"
        ),
    },
    "learned": {
        "name": "学習4点回帰モデル（quadpoint_best.pth）",
        "estimator": estimate_quad_by_learned_wrapper,
        "fail_hint": (
            "学習4点モデルで推定できませんでした。"
            "models/quadpoint_best.pth の配置と学習データ分布を確認してください。"
        ),
    },
}

MODE_TO_METHODS = {
    "ransac": ["ransac"],
    "learned": ["learned"],
    "compare_rl": ["ransac", "learned"],
}
DEFAULT_METHOD_MODE = "ransac"


def method_mode_title(mode: str) -> str:
    """選択された4点推定モードの表示名を返す。"""
    if mode == "learned":
        return "4点推定結果（学習4点回帰モデル）"
    if mode == "compare_rl":
        return "4点推定結果（RANSAC / 学習4点モデル 比較）"
    return "4点推定結果（幅フィルタ + RANSAC）"


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
    model_path = find_model_path(MODELS_DIR)
    if model_path is not None:
        if model_path.suffix.lower() == ".xml":
            return (
                f"models/{model_path.name}（+ {model_path.with_suffix('.bin').name}）を検出しました。"
                "OpenVINO 推論を使用します。"
            )
        return f"models/{model_path.name} を検出しました。DeepLabV3+ 推論を使用します。"
    return "models/ に .pth または .xml(+.bin) がないため、仮マスクで処理します。"


def format_model_option_label(model_path: Path) -> str:
    if model_path.suffix.lower() == ".xml":
        return f"{model_path.name} + {model_path.with_suffix('.bin').name}（OpenVINO）"
    return f"{model_path.name}（PyTorch .pth）"


def model_options() -> list[dict[str, str]]:
    options = [
        {"value": MODEL_CHOICE_AUTO, "label": "自動選択（推奨）"},
        {"value": MODEL_CHOICE_PLACEHOLDER, "label": "仮マスクを強制使用"},
    ]
    for model_path in list_available_model_paths(MODELS_DIR):
        options.append(
            {
                "value": model_path.name,
                "label": format_model_option_label(model_path),
            }
        )
    return options


def describe_selected_model(model_choice: str, model_path: Path | None) -> str:
    if model_choice == MODEL_CHOICE_PLACEHOLDER:
        return "仮マスクを強制使用"
    if model_path is None:
        return "自動選択（利用可能なモデルなし）"
    if model_choice == MODEL_CHOICE_AUTO:
        return f"自動選択: {format_model_option_label(model_path)}"
    return f"手動選択: {format_model_option_label(model_path)}"


def render_index(
    error_message: str | None = None,
    selected_model: str = MODEL_CHOICE_AUTO,
    selected_method_mode: str = DEFAULT_METHOD_MODE,
    info_message: str | None = None,
    train_dataset_mode: str = "path",
    train_images_dir: str = "data/images",
    train_masks_dir: str = "data/masks",
    train_checkpoint_dir: str = "train/checkpoints",
    train_epochs: int = 50,
    train_batch_size: int = 4,
    train_lr: float = 1e-4,
    train_num_workers: int = 4,
    quad_train_dataset_mode: str = "path",
    quad_train_input_mode: str = "rgb",
    quad_train_images_dir: str = "data/quadpoint/images",
    quad_train_masks_dir: str = "data/quadpoint/masks",
    quad_train_labels_csv: str = "data/quadpoint/labels.csv",
    quad_train_checkpoint_dir: str = "models",
    quad_train_epochs: int = 80,
    quad_train_batch_size: int = 8,
    quad_train_lr: float = 1e-4,
    quad_train_num_workers: int = 4,
    quad_train_input_size: int = 384,
    distortion_target_width: int = 500,
):
    train_script_ready, distortion_script_ready = scripts_ready()
    training_status = get_training_status()
    quad_training_status = get_quadpoint_training_status()

    return render_template(
        "index.html",
        error_message=error_message,
        info_message=info_message,
        model_notice=model_notice(),
        model_options=model_options(),
        selected_model=selected_model,
        selected_method_mode=selected_method_mode,
        train_script_ready=train_script_ready,
        distortion_script_ready=distortion_script_ready,
        training_status=training_status,
        train_dataset_mode=train_dataset_mode,
        train_images_dir=train_images_dir,
        train_masks_dir=train_masks_dir,
        train_checkpoint_dir=train_checkpoint_dir,
        train_epochs=train_epochs,
        train_batch_size=train_batch_size,
        train_lr=train_lr,
        train_num_workers=train_num_workers,
        quad_train_script_ready=quadpoint_training_script_ready(),
        quad_training_status=quad_training_status,
        quad_train_dataset_mode=quad_train_dataset_mode,
        quad_train_input_mode=quad_train_input_mode,
        quad_train_images_dir=quad_train_images_dir,
        quad_train_masks_dir=quad_train_masks_dir,
        quad_train_labels_csv=quad_train_labels_csv,
        quad_train_checkpoint_dir=quad_train_checkpoint_dir,
        quad_train_epochs=quad_train_epochs,
        quad_train_batch_size=quad_train_batch_size,
        quad_train_lr=quad_train_lr,
        quad_train_num_workers=quad_train_num_workers,
        quad_train_input_size=quad_train_input_size,
        distortion_target_width=distortion_target_width,
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
    method_conf = ESTIMATION_METHODS[method_key]
    method_name = str(method_conf["name"])
    estimator = method_conf["estimator"]
    fail_hint = str(method_conf["fail_hint"])
    result: dict[str, str | bool | None] = {
        "key": method_key,
        "name": method_name,
        "success": False,
        "error_message": None,
        "quad_image": None,
        "birdview_image": None,
    }

    try:
        src_pts = estimator(image, clean_mask)
        if src_pts is None:
            result["error_message"] = fail_hint
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


def make_mask_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """道路マスク領域を赤で重畳した可視化画像を作る。"""
    overlay = image.copy()
    red = np.zeros_like(image)
    red[:, :, 2] = 255
    mask_bool = mask > 0
    overlay[mask_bool] = cv2.addWeighted(
        image[mask_bool],
        1.0 - alpha,
        red[mask_bool],
        alpha,
        0,
    )
    return overlay


def parse_int(value: str | None, default: int, min_value: int, max_value: int) -> int:
    try:
        parsed = int(value) if value is not None else default
    except ValueError:
        return default
    return max(min_value, min(max_value, parsed))


def parse_float(value: str | None, default: float, min_value: float, max_value: float) -> float:
    try:
        parsed = float(value) if value is not None else default
    except ValueError:
        return default
    return max(min_value, min(max_value, parsed))


def to_static_relative(path: Path) -> str:
    return str(path.relative_to(BASE_DIR / "static"))


def resolve_training_data_dir(path_text: str) -> Path:
    """
    学習データディレクトリを解決する。
    - 絶対パス: そのまま
    - 相対パス: DeepLabV3Plus-Pytorch 基準
    """
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return (DEEPLAB_ROOT / path).resolve()


def normalize_train_dataset_mode(mode: str | None) -> str:
    return mode if mode in {"path", "zip"} else "path"


def is_zip_filename(filename: str | None) -> bool:
    return bool(filename) and filename.lower().endswith(".zip")


def is_csv_filename(filename: str | None) -> bool:
    return bool(filename) and filename.lower().endswith(".csv")


def resolve_local_data_dir(path_text: str) -> Path:
    """
    アプリローカル基準のパス解決。
    - 絶対パス: そのまま
    - 相対パス: このリポジトリ基準
    """
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return (BASE_DIR / path).resolve()


def prepare_uploaded_training_dataset(
    images_zip: FileStorage,
    masks_zip: FileStorage,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{uuid4().hex[:8]}"
    run_dir = USER_DATASETS_DIR / run_id

    zip_dir = run_dir / "zip"
    zip_dir.mkdir(parents=True, exist_ok=True)
    images_zip_path = zip_dir / "images.zip"
    masks_zip_path = zip_dir / "masks.zip"
    images_zip.save(images_zip_path)
    masks_zip.save(masks_zip_path)

    raw_images_dir = run_dir / "raw" / "images"
    raw_masks_dir = run_dir / "raw" / "masks"
    extract_zip_safe(images_zip_path, raw_images_dir)
    extract_zip_safe(masks_zip_path, raw_masks_dir)

    prepared_dir = run_dir / "prepared"
    info = prepare_dataset_from_dirs(raw_images_dir, raw_masks_dir, prepared_dir)
    return info


def prepare_uploaded_quad_training_dataset(
    images_zip: FileStorage,
    labels_csv_file: FileStorage,
    masks_zip: FileStorage | None = None,
) -> QuadpointDatasetInfo:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{uuid4().hex[:8]}"
    run_dir = QUAD_USER_DATASETS_DIR / run_id

    zip_dir = run_dir / "zip"
    zip_dir.mkdir(parents=True, exist_ok=True)
    images_zip_path = zip_dir / "images.zip"
    labels_csv_path = zip_dir / "labels.csv"
    images_zip.save(images_zip_path)
    labels_csv_file.save(labels_csv_path)
    masks_zip_path = zip_dir / "masks.zip"
    if masks_zip is not None:
        masks_zip.save(masks_zip_path)

    raw_images_dir = run_dir / "raw" / "images"
    raw_masks_dir = run_dir / "raw" / "masks"
    extract_zip_safe(images_zip_path, raw_images_dir)
    if masks_zip is not None:
        extract_zip_safe(masks_zip_path, raw_masks_dir)

    validated = validate_quadpoint_dataset(
        raw_images_dir,
        labels_csv_path,
        masks_dir=raw_masks_dir if masks_zip is not None else None,
    )
    return QuadpointDatasetInfo(
        images_dir=validated.images_dir,
        masks_dir=validated.masks_dir,
        labels_csv=validated.labels_csv,
        sample_count=validated.sample_count,
        output_root=run_dir,
    )


@app.route("/", methods=["GET"])
def index():
    return render_index()


@app.route("/process", methods=["POST"])
def process():
    model_choice = request.form.get("model_choice", MODEL_CHOICE_AUTO)
    mode = request.form.get("method_mode", DEFAULT_METHOD_MODE)
    selected_methods = MODE_TO_METHODS.get(mode, MODE_TO_METHODS[DEFAULT_METHOD_MODE])
    output_mode = request.form.get("output_mode", "full")
    file = request.files.get("image")
    validation_error = validate_upload(file)
    if validation_error is not None:
        return render_index(
            error_message=validation_error,
            selected_model=model_choice,
            selected_method_mode=mode,
        )

    assert file is not None
    try:
        selected_model_path = resolve_model_choice(model_choice, MODELS_DIR)
    except ValueError as exc:
        return render_index(
            error_message=str(exc),
            selected_model=model_choice,
            selected_method_mode=mode,
        )

    original_name = secure_filename(file.filename)
    ext = Path(original_name).suffix.lower()
    stem = build_unique_stem(original_name)
    input_filename = f"{stem}{ext}"
    input_path = UPLOAD_DIR / input_filename

    try:
        file.save(input_path)

        image = cv2.imread(str(input_path))
        if image is None:
            return render_index(
                error_message="画像を読み込めませんでした。別の画像で再試行してください。",
                selected_model=model_choice,
                selected_method_mode=mode,
            )

        pipeline = get_default_pipeline(model_path=selected_model_path)
        raw_mask = pipeline.predict(image)
        clean_mask = preprocess_mask(raw_mask)

        raw_mask_rel = f"outputs/{stem}_mask_raw.png"
        clean_mask_rel = f"outputs/{stem}_mask_clean.png"
        cv2.imwrite(str(BASE_DIR / "static" / raw_mask_rel), raw_mask)
        cv2.imwrite(str(BASE_DIR / "static" / clean_mask_rel), clean_mask)

        raw_overlay = make_mask_overlay(image, raw_mask)
        clean_overlay = make_mask_overlay(image, clean_mask)
        raw_overlay_rel = f"outputs/{stem}_mask_raw_overlay.png"
        clean_overlay_rel = f"outputs/{stem}_mask_clean_overlay.png"
        cv2.imwrite(str(BASE_DIR / "static" / raw_overlay_rel), raw_overlay)
        cv2.imwrite(str(BASE_DIR / "static" / clean_overlay_rel), clean_overlay)

        if output_mode == "segmentation":
            return render_template(
                "segmentation_result.html",
                input_image=f"uploads/{input_filename}",
                raw_mask_image=raw_mask_rel,
                clean_mask_image=clean_mask_rel,
                raw_overlay_image=raw_overlay_rel,
                clean_overlay_image=clean_overlay_rel,
                segmentation_backend=pipeline.backend_info.name,
                segmentation_detail=pipeline.backend_info.detail,
                selected_model_label=describe_selected_model(model_choice, selected_model_path),
            )

        method_results = [
            run_estimation(method_key, image, clean_mask, stem)
            for method_key in selected_methods
        ]

        return render_template(
            "result.html",
            input_image=f"uploads/{input_filename}",
            raw_mask_image=raw_mask_rel,
            clean_mask_image=clean_mask_rel,
            method_results=method_results,
            method_title=method_mode_title(mode),
            segmentation_backend=pipeline.backend_info.name,
            segmentation_detail=pipeline.backend_info.detail,
            selected_model_label=describe_selected_model(model_choice, selected_model_path),
            mode=mode,
        )
    except Exception as exc:
        return render_index(
            error_message=(
                "処理中にエラーが発生しました。入力画像やモデル設定を確認してください。"
                f" 詳細: {exc}"
            ),
            selected_model=model_choice,
            selected_method_mode=mode,
        )


@app.route("/training/start", methods=["POST"])
def training_start():
    train_dataset_mode = normalize_train_dataset_mode(request.form.get("train_dataset_mode"))
    images_dir = (request.form.get("train_images_dir") or "data/images").strip()
    masks_dir = (request.form.get("train_masks_dir") or "data/masks").strip()
    checkpoint_dir = (request.form.get("train_checkpoint_dir") or "train/checkpoints").strip()
    epochs = parse_int(request.form.get("train_epochs"), default=50, min_value=1, max_value=10000)
    batch_size = parse_int(request.form.get("train_batch_size"), default=4, min_value=1, max_value=256)
    num_workers = parse_int(request.form.get("train_num_workers"), default=4, min_value=0, max_value=32)
    lr = parse_float(request.form.get("train_lr"), default=1e-4, min_value=1e-7, max_value=1.0)

    dataset_info_message = ""
    try:
        if train_dataset_mode == "zip":
            images_zip = request.files.get("train_images_zip")
            masks_zip = request.files.get("train_masks_zip")

            if images_zip is None or masks_zip is None:
                return render_index(
                    error_message="zip学習モードでは images.zip と masks.zip の両方が必要です。",
                    train_dataset_mode=train_dataset_mode,
                    train_images_dir=images_dir,
                    train_masks_dir=masks_dir,
                    train_checkpoint_dir=checkpoint_dir,
                    train_epochs=epochs,
                    train_batch_size=batch_size,
                    train_lr=lr,
                    train_num_workers=num_workers,
                )

            if not is_zip_filename(images_zip.filename) or not is_zip_filename(masks_zip.filename):
                return render_index(
                    error_message="アップロードする学習データは zip 形式にしてください。",
                    train_dataset_mode=train_dataset_mode,
                    train_images_dir=images_dir,
                    train_masks_dir=masks_dir,
                    train_checkpoint_dir=checkpoint_dir,
                    train_epochs=epochs,
                    train_batch_size=batch_size,
                    train_lr=lr,
                    train_num_workers=num_workers,
                )

            prepared_info = prepare_uploaded_training_dataset(images_zip, masks_zip)
            images_dir = str(prepared_info.images_dir)
            masks_dir = str(prepared_info.masks_dir)
            dataset_info_message = (
                f"アップロードデータセットを準備しました: {prepared_info.sample_count} ペア "
                f"({prepared_info.output_root})"
            )
        else:
            abs_images_dir = resolve_training_data_dir(images_dir)
            abs_masks_dir = resolve_training_data_dir(masks_dir)
            if not abs_images_dir.exists():
                return render_index(
                    error_message=f"学習画像ディレクトリが存在しません: {abs_images_dir}",
                    train_dataset_mode=train_dataset_mode,
                    train_images_dir=images_dir,
                    train_masks_dir=masks_dir,
                    train_checkpoint_dir=checkpoint_dir,
                    train_epochs=epochs,
                    train_batch_size=batch_size,
                    train_lr=lr,
                    train_num_workers=num_workers,
                )
            if not abs_masks_dir.exists():
                return render_index(
                    error_message=f"学習マスクディレクトリが存在しません: {abs_masks_dir}",
                    train_dataset_mode=train_dataset_mode,
                    train_images_dir=images_dir,
                    train_masks_dir=masks_dir,
                    train_checkpoint_dir=checkpoint_dir,
                    train_epochs=epochs,
                    train_batch_size=batch_size,
                    train_lr=lr,
                    train_num_workers=num_workers,
                )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = f"{timestamp}_{uuid4().hex[:8]}"
            prepared_root = USER_DATASETS_DIR / run_id / "prepared"
            prepared_info = prepare_dataset_from_dirs(
                abs_images_dir,
                abs_masks_dir,
                prepared_root,
            )
            images_dir = str(prepared_info.images_dir)
            masks_dir = str(prepared_info.masks_dir)
            dataset_info_message = (
                f"ディレクトリデータセットを正規化しました: {prepared_info.sample_count} ペア "
                f"({prepared_info.output_root})"
            )
    except Exception as exc:
        return render_index(
            error_message=f"学習データ準備に失敗しました: {exc}",
            train_dataset_mode=train_dataset_mode,
            train_images_dir=images_dir,
            train_masks_dir=masks_dir,
            train_checkpoint_dir=checkpoint_dir,
            train_epochs=epochs,
            train_batch_size=batch_size,
            train_lr=lr,
            train_num_workers=num_workers,
        )

    success, message = start_training(
        images_dir=images_dir,
        masks_dir=masks_dir,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        checkpoint_dir=checkpoint_dir,
        num_workers=num_workers,
    )

    info_message = message if success else None
    if success and dataset_info_message:
        info_message = f"{dataset_info_message}\n{message}"

    return render_index(
        error_message=None if success else message,
        info_message=info_message,
        train_dataset_mode=train_dataset_mode,
        train_images_dir=images_dir,
        train_masks_dir=masks_dir,
        train_checkpoint_dir=checkpoint_dir,
        train_epochs=epochs,
        train_batch_size=batch_size,
        train_lr=lr,
        train_num_workers=num_workers,
    )


@app.route("/training/stop", methods=["POST"])
def training_stop():
    success, message = stop_training()
    return render_index(
        error_message=None if success else message,
        info_message=message if success else None,
    )


@app.route("/quad-training/start", methods=["POST"])
def quad_training_start():
    quad_train_dataset_mode = normalize_train_dataset_mode(
        request.form.get("quad_train_dataset_mode")
    )
    quad_train_input_mode = (request.form.get("quad_train_input_mode") or "rgb").strip()
    if quad_train_input_mode not in {"rgb", "rgb_mask"}:
        quad_train_input_mode = "rgb"
    use_quad_mask_input = quad_train_input_mode == "rgb_mask"
    quad_train_images_dir = (
        request.form.get("quad_train_images_dir") or "data/quadpoint/images"
    ).strip()
    quad_train_masks_dir = (
        request.form.get("quad_train_masks_dir") or "data/quadpoint/masks"
    ).strip()
    quad_train_labels_csv = (
        request.form.get("quad_train_labels_csv") or "data/quadpoint/labels.csv"
    ).strip()
    quad_train_checkpoint_dir = (
        request.form.get("quad_train_checkpoint_dir") or "models"
    ).strip()
    quad_train_epochs = parse_int(
        request.form.get("quad_train_epochs"),
        default=80,
        min_value=1,
        max_value=10000,
    )
    quad_train_batch_size = parse_int(
        request.form.get("quad_train_batch_size"),
        default=8,
        min_value=1,
        max_value=256,
    )
    quad_train_num_workers = parse_int(
        request.form.get("quad_train_num_workers"),
        default=4,
        min_value=0,
        max_value=32,
    )
    quad_train_input_size = parse_int(
        request.form.get("quad_train_input_size"),
        default=384,
        min_value=96,
        max_value=2048,
    )
    quad_train_lr = parse_float(
        request.form.get("quad_train_lr"),
        default=1e-4,
        min_value=1e-7,
        max_value=1.0,
    )

    dataset_info_message = ""
    try:
        if quad_train_dataset_mode == "zip":
            images_zip = request.files.get("quad_train_images_zip")
            labels_csv_file = request.files.get("quad_train_labels_csv_file")

            if images_zip is None or labels_csv_file is None:
                return render_index(
                    error_message="4点zip学習モードでは images.zip と labels.csv の両方が必要です。",
                    quad_train_dataset_mode=quad_train_dataset_mode,
                    quad_train_input_mode=quad_train_input_mode,
                    quad_train_images_dir=quad_train_images_dir,
                    quad_train_masks_dir=quad_train_masks_dir,
                    quad_train_labels_csv=quad_train_labels_csv,
                    quad_train_checkpoint_dir=quad_train_checkpoint_dir,
                    quad_train_epochs=quad_train_epochs,
                    quad_train_batch_size=quad_train_batch_size,
                    quad_train_lr=quad_train_lr,
                    quad_train_num_workers=quad_train_num_workers,
                    quad_train_input_size=quad_train_input_size,
                )

            if not is_zip_filename(images_zip.filename):
                return render_index(
                    error_message="4点学習の画像データは zip 形式でアップロードしてください。",
                    quad_train_dataset_mode=quad_train_dataset_mode,
                    quad_train_input_mode=quad_train_input_mode,
                    quad_train_images_dir=quad_train_images_dir,
                    quad_train_masks_dir=quad_train_masks_dir,
                    quad_train_labels_csv=quad_train_labels_csv,
                    quad_train_checkpoint_dir=quad_train_checkpoint_dir,
                    quad_train_epochs=quad_train_epochs,
                    quad_train_batch_size=quad_train_batch_size,
                    quad_train_lr=quad_train_lr,
                    quad_train_num_workers=quad_train_num_workers,
                    quad_train_input_size=quad_train_input_size,
                )

            if not is_csv_filename(labels_csv_file.filename):
                return render_index(
                    error_message="4点ラベルは csv 形式でアップロードしてください。",
                    quad_train_dataset_mode=quad_train_dataset_mode,
                    quad_train_input_mode=quad_train_input_mode,
                    quad_train_images_dir=quad_train_images_dir,
                    quad_train_masks_dir=quad_train_masks_dir,
                    quad_train_labels_csv=quad_train_labels_csv,
                    quad_train_checkpoint_dir=quad_train_checkpoint_dir,
                    quad_train_epochs=quad_train_epochs,
                    quad_train_batch_size=quad_train_batch_size,
                    quad_train_lr=quad_train_lr,
                    quad_train_num_workers=quad_train_num_workers,
                    quad_train_input_size=quad_train_input_size,
                )

            masks_zip = request.files.get("quad_train_masks_zip")
            if use_quad_mask_input:
                if masks_zip is None:
                    return render_index(
                        error_message="RGB+Mask学習では masks.zip もアップロードしてください。",
                        quad_train_dataset_mode=quad_train_dataset_mode,
                        quad_train_input_mode=quad_train_input_mode,
                        quad_train_images_dir=quad_train_images_dir,
                        quad_train_masks_dir=quad_train_masks_dir,
                        quad_train_labels_csv=quad_train_labels_csv,
                        quad_train_checkpoint_dir=quad_train_checkpoint_dir,
                        quad_train_epochs=quad_train_epochs,
                        quad_train_batch_size=quad_train_batch_size,
                        quad_train_lr=quad_train_lr,
                        quad_train_num_workers=quad_train_num_workers,
                        quad_train_input_size=quad_train_input_size,
                    )
                if not is_zip_filename(masks_zip.filename):
                    return render_index(
                        error_message="4点学習マスクは zip 形式でアップロードしてください。",
                        quad_train_dataset_mode=quad_train_dataset_mode,
                        quad_train_input_mode=quad_train_input_mode,
                        quad_train_images_dir=quad_train_images_dir,
                        quad_train_masks_dir=quad_train_masks_dir,
                        quad_train_labels_csv=quad_train_labels_csv,
                        quad_train_checkpoint_dir=quad_train_checkpoint_dir,
                        quad_train_epochs=quad_train_epochs,
                        quad_train_batch_size=quad_train_batch_size,
                        quad_train_lr=quad_train_lr,
                        quad_train_num_workers=quad_train_num_workers,
                        quad_train_input_size=quad_train_input_size,
                    )

            prepared = prepare_uploaded_quad_training_dataset(
                images_zip,
                labels_csv_file,
                masks_zip=masks_zip if use_quad_mask_input else None,
            )
            quad_train_images_dir = str(prepared.images_dir)
            quad_train_masks_dir = str(prepared.masks_dir) if prepared.masks_dir is not None else ""
            quad_train_labels_csv = str(prepared.labels_csv)
            dataset_mode_text = "RGB+Mask" if use_quad_mask_input else "RGB"
            dataset_info_message = (
                f"4点zipデータセットを準備しました ({dataset_mode_text}): "
                f"{prepared.sample_count} 件 ({prepared.output_root})"
            )
        else:
            abs_images_dir = resolve_local_data_dir(quad_train_images_dir)
            abs_labels_csv = resolve_local_data_dir(quad_train_labels_csv)
            abs_masks_dir = (
                resolve_local_data_dir(quad_train_masks_dir)
                if use_quad_mask_input
                else None
            )
            validated = validate_quadpoint_dataset(
                abs_images_dir,
                abs_labels_csv,
                masks_dir=abs_masks_dir,
            )
            quad_train_images_dir = str(validated.images_dir)
            quad_train_masks_dir = str(validated.masks_dir) if validated.masks_dir is not None else ""
            quad_train_labels_csv = str(validated.labels_csv)
            dataset_mode_text = "RGB+Mask" if use_quad_mask_input else "RGB"
            dataset_info_message = (
                f"4点データセットを検証しました ({dataset_mode_text}): "
                f"{validated.sample_count} 件"
            )
    except Exception as exc:
        return render_index(
            error_message=f"4点学習データ準備に失敗しました: {exc}",
            quad_train_dataset_mode=quad_train_dataset_mode,
            quad_train_input_mode=quad_train_input_mode,
            quad_train_images_dir=quad_train_images_dir,
            quad_train_masks_dir=quad_train_masks_dir,
            quad_train_labels_csv=quad_train_labels_csv,
            quad_train_checkpoint_dir=quad_train_checkpoint_dir,
            quad_train_epochs=quad_train_epochs,
            quad_train_batch_size=quad_train_batch_size,
            quad_train_lr=quad_train_lr,
            quad_train_num_workers=quad_train_num_workers,
            quad_train_input_size=quad_train_input_size,
        )

    success, message = start_quadpoint_training(
        images_dir=quad_train_images_dir,
        labels_csv=quad_train_labels_csv,
        masks_dir=quad_train_masks_dir if use_quad_mask_input else None,
        batch_size=quad_train_batch_size,
        epochs=quad_train_epochs,
        lr=quad_train_lr,
        checkpoint_dir=quad_train_checkpoint_dir,
        num_workers=quad_train_num_workers,
        input_size=quad_train_input_size,
    )

    info_message = message if success else None
    if success and dataset_info_message:
        info_message = f"{dataset_info_message}\n{message}"

    return render_index(
        error_message=None if success else message,
        info_message=info_message,
        quad_train_dataset_mode=quad_train_dataset_mode,
        quad_train_input_mode=quad_train_input_mode,
        quad_train_images_dir=quad_train_images_dir,
        quad_train_masks_dir=quad_train_masks_dir,
        quad_train_labels_csv=quad_train_labels_csv,
        quad_train_checkpoint_dir=quad_train_checkpoint_dir,
        quad_train_epochs=quad_train_epochs,
        quad_train_batch_size=quad_train_batch_size,
        quad_train_lr=quad_train_lr,
        quad_train_num_workers=quad_train_num_workers,
        quad_train_input_size=quad_train_input_size,
    )


@app.route("/quad-training/stop", methods=["POST"])
def quad_training_stop():
    success, message = stop_quadpoint_training()
    return render_index(
        error_message=None if success else message,
        info_message=message if success else None,
    )


@app.route("/distortion/process", methods=["POST"])
def distortion_process():
    file = request.files.get("distortion_image")
    target_width = parse_int(
        request.form.get("distortion_target_width"),
        default=500,
        min_value=100,
        max_value=3000,
    )

    validation_error = validate_upload(file)
    if validation_error is not None:
        return render_index(
            error_message=f"歪み補正用画像: {validation_error}",
            distortion_target_width=target_width,
        )

    assert file is not None
    original_name = secure_filename(file.filename)
    ext = Path(original_name).suffix.lower()
    stem = build_unique_stem(original_name)
    input_filename = f"{stem}{ext}"
    input_path = UPLOAD_DIR / input_filename
    file.save(input_path)

    success, message, corrected_path, mirror_path, raw_log = run_distortion_correction(
        input_image=input_path,
        output_dir=DISTORT_OUTPUT_DIR,
        target_width=target_width,
    )
    log_excerpt = "\n".join(raw_log.splitlines()[-80:])

    if not success:
        error_detail = message
        if log_excerpt:
            error_detail += f"\n\n--- script log ---\n{log_excerpt}"
        return render_index(
            error_message=error_detail,
            distortion_target_width=target_width,
        )

    return render_template(
        "distortion_result.html",
        input_image=f"uploads/{input_filename}",
        corrected_image=to_static_relative(corrected_path) if corrected_path else None,
        mirror_image=to_static_relative(mirror_path) if mirror_path else None,
        distortion_log=log_excerpt,
        target_width=target_width,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
