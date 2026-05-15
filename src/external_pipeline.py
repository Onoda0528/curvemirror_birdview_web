from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import subprocess
from typing import Optional


# サーバーごとに配置場所が異なるため、環境変数で上書き可能にする。
# 例: export DEEPLAB_ROOT=/mnt/work/DeepLabV3Plus-Pytorch
DEEPLAB_ROOT = Path(
    os.environ.get("DEEPLAB_ROOT", "/home/onoda/DeepLabV3Plus-Pytorch")
).expanduser()
TRAIN_SCRIPT = DEEPLAB_ROOT / "train" / "scripts" / "train_road_smp.py"
DISTORTION_SCRIPT = DEEPLAB_ROOT / "hosei" / "hosei.py"

_training_process: Optional[subprocess.Popen] = None
_training_log_path: Optional[Path] = None
_training_started_at: Optional[datetime] = None
_training_command: Optional[list[str]] = None


@dataclass(frozen=True)
class TrainingStatus:
    state: str
    message: str
    is_running: bool
    pid: Optional[int]
    returncode: Optional[int]
    log_path: Optional[Path]
    started_at: Optional[str]
    command: Optional[str]
    log_tail: str


def _tail_text(path: Optional[Path], max_lines: int = 60) -> str:
    if path is None or not path.exists():
        return ""

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return ""

    return "".join(lines[-max_lines:])


def scripts_ready() -> tuple[bool, bool]:
    """(学習スクリプト有無, 歪み補正スクリプト有無) を返す。"""
    return TRAIN_SCRIPT.exists(), DISTORTION_SCRIPT.exists()


def _resolve_path(path_text: str) -> Path:
    """
    学習データパスの解決:
    - 絶対パスならそのまま
    - 相対パスなら DeepLabV3Plus-Pytorch 基準
    """
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return (DEEPLAB_ROOT / path).resolve()


def start_training(
    images_dir: str,
    masks_dir: str,
    batch_size: int,
    epochs: int,
    lr: float,
    checkpoint_dir: str,
    num_workers: int = 4,
) -> tuple[bool, str]:
    global _training_process, _training_log_path, _training_started_at, _training_command

    if _training_process is not None and _training_process.poll() is None:
        return False, "学習ジョブはすでに実行中です。"

    if not TRAIN_SCRIPT.exists():
        return False, f"学習スクリプトが見つかりません: {TRAIN_SCRIPT}"

    images_path = _resolve_path(images_dir)
    masks_path = _resolve_path(masks_dir)
    ckpt_path = _resolve_path(checkpoint_dir)

    if not images_path.exists():
        return False, f"images_dir が存在しません: {images_path}"
    if not masks_path.exists():
        return False, f"masks_dir が存在しません: {masks_path}"

    ckpt_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path.cwd() / "static" / "outputs" / "train_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"train_{timestamp}.log"

    command = [
        "python3",
        str(TRAIN_SCRIPT),
        "--images_dir",
        str(images_path),
        "--masks_dir",
        str(masks_path),
        "--batch_size",
        str(batch_size),
        "--epochs",
        str(epochs),
        "--lr",
        str(lr),
        "--num_workers",
        str(num_workers),
        "--checkpoint_dir",
        str(ckpt_path),
    ]

    try:
        with open(log_path, "w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                cwd=str(DEEPLAB_ROOT),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
    except Exception as exc:
        return False, f"学習ジョブの起動に失敗しました: {exc}"

    _training_process = process
    _training_log_path = log_path
    _training_started_at = datetime.now()
    _training_command = command
    return True, f"学習ジョブを開始しました (PID: {process.pid})"


def stop_training() -> tuple[bool, str]:
    global _training_process

    if _training_process is None or _training_process.poll() is not None:
        return False, "停止できる学習ジョブはありません。"

    _training_process.terminate()
    return True, "学習ジョブに停止シグナルを送信しました。"


def get_training_status() -> TrainingStatus:
    process = _training_process
    log_tail = _tail_text(_training_log_path)

    if process is None:
        return TrainingStatus(
            state="idle",
            message="学習ジョブは実行されていません。",
            is_running=False,
            pid=None,
            returncode=None,
            log_path=_training_log_path,
            started_at=None,
            command=None,
            log_tail=log_tail,
        )

    returncode = process.poll()
    if returncode is None:
        state = "running"
        message = "学習ジョブ実行中です。"
        is_running = True
    elif returncode == 0:
        state = "succeeded"
        message = "学習ジョブは正常終了しました。"
        is_running = False
    else:
        state = "failed"
        message = f"学習ジョブはエラー終了しました (code={returncode})。"
        is_running = False

    started_at = _training_started_at.isoformat(timespec="seconds") if _training_started_at else None
    command = " ".join(_training_command) if _training_command else None

    return TrainingStatus(
        state=state,
        message=message,
        is_running=is_running,
        pid=process.pid,
        returncode=returncode,
        log_path=_training_log_path,
        started_at=started_at,
        command=command,
        log_tail=log_tail,
    )


def run_distortion_correction(
    input_image: Path,
    output_dir: Path,
    target_width: int = 500,
) -> tuple[bool, str, Optional[Path], Optional[Path], str]:
    """
    歪み補正スクリプトを実行し、(成功可否, メッセージ, 補正画像, ミラー画像, ログ) を返す。
    """
    if not DISTORTION_SCRIPT.exists():
        return False, f"歪み補正スクリプトが見つかりません: {DISTORTION_SCRIPT}", None, None, ""

    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "python3",
        str(DISTORTION_SCRIPT),
        str(input_image),
        "--no-display",
        "--target-width",
        str(target_width),
        "--output-dir",
        str(output_dir),
    ]

    try:
        completed = subprocess.run(
            command,
            cwd=str(DEEPLAB_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        return False, f"歪み補正スクリプト実行に失敗しました: {exc}", None, None, ""

    stem = input_image.stem
    corrected = output_dir / f"{stem}_correctedimg_view.jpg"
    mirror = output_dir / f"{stem}_mirror_img_view.jpg"
    stdout_stderr = (completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else "")

    if completed.returncode != 0:
        return (
            False,
            f"歪み補正スクリプトがエラー終了しました (code={completed.returncode})。",
            corrected if corrected.exists() else None,
            mirror if mirror.exists() else None,
            stdout_stderr,
        )

    if not corrected.exists():
        return (
            False,
            "歪み補正結果画像が生成されませんでした。",
            None,
            mirror if mirror.exists() else None,
            stdout_stderr,
        )

    return True, "歪み補正が完了しました。", corrected, mirror if mirror.exists() else None, stdout_stderr
