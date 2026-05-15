from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import subprocess
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUAD_TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_quad_points.py"

_quad_training_process: Optional[subprocess.Popen] = None
_quad_training_log_path: Optional[Path] = None
_quad_training_started_at: Optional[datetime] = None
_quad_training_command: Optional[list[str]] = None


@dataclass(frozen=True)
class QuadTrainingStatus:
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


def quadpoint_training_script_ready() -> bool:
    return QUAD_TRAIN_SCRIPT.exists()


def _resolve_local_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def start_quadpoint_training(
    images_dir: str,
    labels_csv: str,
    masks_dir: str | None,
    batch_size: int,
    epochs: int,
    lr: float,
    checkpoint_dir: str,
    num_workers: int = 4,
    input_size: int = 384,
) -> tuple[bool, str]:
    global _quad_training_process, _quad_training_log_path, _quad_training_started_at, _quad_training_command

    if _quad_training_process is not None and _quad_training_process.poll() is None:
        return False, "4点学習ジョブはすでに実行中です。"

    if not QUAD_TRAIN_SCRIPT.exists():
        return False, f"4点学習スクリプトが見つかりません: {QUAD_TRAIN_SCRIPT}"

    images_path = _resolve_local_path(images_dir)
    labels_path = _resolve_local_path(labels_csv)
    masks_path = _resolve_local_path(masks_dir) if masks_dir else None
    ckpt_path = _resolve_local_path(checkpoint_dir)

    if not images_path.exists():
        return False, f"images_dir が存在しません: {images_path}"
    if not labels_path.exists():
        return False, f"labels_csv が存在しません: {labels_path}"
    if masks_path is not None and not masks_path.exists():
        return False, f"masks_dir が存在しません: {masks_path}"

    ckpt_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = PROJECT_ROOT / "static" / "outputs" / "train_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"quad_train_{timestamp}.log"

    command = [
        "python3",
        str(QUAD_TRAIN_SCRIPT),
        "--images_dir",
        str(images_path),
        "--labels_csv",
        str(labels_path),
        "--batch_size",
        str(batch_size),
        "--epochs",
        str(epochs),
        "--lr",
        str(lr),
        "--num_workers",
        str(num_workers),
        "--input_size",
        str(input_size),
        "--checkpoint_dir",
        str(ckpt_path),
    ]
    if masks_path is not None:
        command.extend(["--masks_dir", str(masks_path)])

    try:
        with open(log_path, "w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                cwd=str(PROJECT_ROOT),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
    except Exception as exc:
        return False, f"4点学習ジョブの起動に失敗しました: {exc}"

    _quad_training_process = process
    _quad_training_log_path = log_path
    _quad_training_started_at = datetime.now()
    _quad_training_command = command
    return True, f"4点学習ジョブを開始しました (PID: {process.pid})"


def stop_quadpoint_training() -> tuple[bool, str]:
    global _quad_training_process

    if _quad_training_process is None or _quad_training_process.poll() is not None:
        return False, "停止できる4点学習ジョブはありません。"

    _quad_training_process.terminate()
    return True, "4点学習ジョブに停止シグナルを送信しました。"


def get_quadpoint_training_status() -> QuadTrainingStatus:
    process = _quad_training_process
    log_tail = _tail_text(_quad_training_log_path)

    if process is None:
        return QuadTrainingStatus(
            state="idle",
            message="4点学習ジョブは実行されていません。",
            is_running=False,
            pid=None,
            returncode=None,
            log_path=_quad_training_log_path,
            started_at=None,
            command=None,
            log_tail=log_tail,
        )

    returncode = process.poll()
    if returncode is None:
        state = "running"
        message = "4点学習ジョブ実行中です。"
        is_running = True
    elif returncode == 0:
        state = "succeeded"
        message = "4点学習ジョブは正常終了しました。"
        is_running = False
    else:
        state = "failed"
        message = f"4点学習ジョブはエラー終了しました (code={returncode})。"
        is_running = False

    started_at = (
        _quad_training_started_at.isoformat(timespec="seconds")
        if _quad_training_started_at
        else None
    )
    command = " ".join(_quad_training_command) if _quad_training_command else None

    return QuadTrainingStatus(
        state=state,
        message=message,
        is_running=is_running,
        pid=process.pid,
        returncode=returncode,
        log_path=_quad_training_log_path,
        started_at=started_at,
        command=command,
        log_tail=log_tail,
    )
