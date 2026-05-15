# カーブミラー俯瞰画像生成 Webアプリ

カーブミラー画像に対して以下を実行する Flask アプリです。

- 道路領域抽出
  - `models/*.pth` があれば DeepLabV3+ 推論
  - `models/*.xml` と対応する `models/*.bin` があれば OpenVINO 推論
  - 上記がなければ仮マスク
- 4点推定
  - 提案手法: 幅フィルタ + RANSAC
- 射影変換による俯瞰画像生成
- セグメンテーション学習ジョブ開始/停止（`DeepLabV3Plus-Pytorch/train/scripts/train_road_smp.py`）
- 4点推定学習ジョブ開始/停止（`scripts/train_quad_points.py`）
- カーブミラー歪み補正（`DeepLabV3Plus-Pytorch/hosei/hosei.py`）

## 研究目的と提案手法

- 研究目的:
カーブミラー画像における道路領域抽出に基づく俯瞰画像生成
- 提案手法:
道路幅フィルタリングと RANSAC 境界推定に基づく射影変換点自動推定

## セットアップ

1. 仮想環境を作成して有効化

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows（PowerShell）の場合:

```powershell
.venv\Scripts\Activate.ps1
```

2. 依存関係をインストール

```bash
pip install -r requirements.txt
```

3. （任意）学習済みモデルを配置  
以下のどちらかを `models/` に置くと自動で推論に使用します（名前は任意）。

- `*.pth`（DeepLabV3+ PyTorch重み）
- `*.xml` + `*.bin`（OpenVINO IR）

4. アプリ起動

```bash
python3 app.py
```

5. ブラウザで開く

```text
http://127.0.0.1:5000
```

## 外部フォルダ依存

本アプリは以下の外部フォルダを参照します。

- `/home/onoda/DeepLabV3Plus-Pytorch`

参照スクリプト:

- 学習: `train/scripts/train_road_smp.py`
- 歪み補正: `hosei/hosei.py`

`DEEPLAB_ROOT` 環境変数で参照先を上書きできます。

```bash
export DEEPLAB_ROOT=/path/to/DeepLabV3Plus-Pytorch
```

## 福岡大学サーバーで効率よく学習する手順

1. サーバーにログインしてリポジトリを配置

```bash
git clone https://github.com/Onoda0528/curvemirror_birdview_web.git
git clone https://github.com/Onoda0528/DeepLabV3Plus-Pytorch.git
cd curvemirror_birdview_web
```

2. Python環境を準備

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r ~/DeepLabV3Plus-Pytorch/requirements.txt
```

3. 参照先を設定

```bash
export DEEPLAB_ROOT=~/DeepLabV3Plus-Pytorch
```

4. 学習実行方法を選ぶ

Slurm が使える場合（推奨）:

```bash
# 必要に応じて scripts/slurm_train_*.sbatch の #SBATCH --partition を環境に合わせて修正
sbatch scripts/slurm_train_segmentation.sbatch
```

セグメンテーション学習（データセットを上書き指定）:

```bash
sbatch --export=ALL,IMAGES_DIR=/path/to/images,MASKS_DIR=/path/to/masks,TRAIN_MODE=smp50 scripts/slurm_train_segmentation.sbatch
```

4点学習（RGBのみ）:

```bash
sbatch --export=ALL,IMAGES_DIR=/path/to/quad/images,LABELS_CSV=/path/to/quad/labels.csv scripts/slurm_train_quadpoint.sbatch
```

4点学習（RGB+Mask）:

```bash
sbatch --export=ALL,IMAGES_DIR=/path/to/quad/images,MASKS_DIR=/path/to/quad/masks,LABELS_CSV=/path/to/quad/labels.csv scripts/slurm_train_quadpoint.sbatch
```

Slurm ジョブ確認・停止:

```bash
squeue -u $USER
scancel <JOB_ID>
```

Slurm ログ確認:

```bash
tail -f cm_seg_train_<JOB_ID>.out
tail -f cm_quad_train_<JOB_ID>.out
```

Slurm が使えない場合（nohup 実行）:

```bash
chmod +x scripts/train_on_server.sh
TRAIN_MODE=smp50 \
IMAGES_DIR=~/DeepLabV3Plus-Pytorch/train/datasets/data_makassar/images \
MASKS_DIR=~/DeepLabV3Plus-Pytorch/train/datasets/data_makassar/masks \
BATCH_SIZE=16 \
EPOCHS=30 \
IMG_SIZE=320 \
NUM_WORKERS=8 \
./scripts/train_on_server.sh
```

5. nohup 実行時のログ監視

```bash
tail -f static/outputs/train_logs/train_smp50_*.log
```

6. 生成された重みをWebで利用  
`models/` に保存された `.pth` は、Web画面のモデル選択で指定できます。

7. 4点学習を nohup で開始（RGB / RGB+Mask 対応）

```bash
chmod +x scripts/train_quad_on_server.sh
```

RGBのみ:

```bash
IMAGES_DIR=/path/to/quad/images \
LABELS_CSV=/path/to/quad/labels.csv \
CHECKPOINT_DIR=$PWD/models \
CKPT_NAME=quadpoint_best.pth \
BATCH_SIZE=16 \
EPOCHS=80 \
INPUT_SIZE=384 \
NUM_WORKERS=8 \
./scripts/train_quad_on_server.sh
```

RGB+Mask:

```bash
IMAGES_DIR=/path/to/quad/images \
MASKS_DIR=/path/to/quad/masks \
LABELS_CSV=/path/to/quad/labels.csv \
CHECKPOINT_DIR=$PWD/models \
CKPT_NAME=quadpoint_best.pth \
BATCH_SIZE=16 \
EPOCHS=80 \
INPUT_SIZE=384 \
NUM_WORKERS=8 \
./scripts/train_quad_on_server.sh
```

4点学習ログ監視:

```bash
tail -f static/outputs/train_logs/quad_train_*.log
```

前面実行したい場合:

```bash
RUN_FOREGROUND=1 ./scripts/train_quad_on_server.sh
```

効率化の目安:
- `nvidia-smi` でGPU使用率を確認し、`BATCH_SIZE` をVRAM上限まで上げる
- CPUコアに合わせて `NUM_WORKERS` を増やす
- まず `IMG_SIZE=320` 前後で回し、精度不足なら `384` へ上げる

8. Webアプリ起動（学習済み重みの確認用）

```bash
source .venv/bin/activate
export DEEPLAB_ROOT=~/DeepLabV3Plus-Pytorch
python3 app.py
```

## 使い方

1. 画像をアップロード
2. 道路マスク推論モデルを選択
- 自動選択（推奨）
- 仮マスクを強制使用
- `models/` 内の任意の `.pth` または `.xml(+.bin)` を明示選択
3. 4点推定手法を選択（推論フォーム）
- 幅フィルタ + RANSAC（提案手法）
- 学習4点回帰モデル（`models/quadpoint_best.pth`）
- RANSAC と学習4点モデルの比較
4. 出力モードを選択
- 俯瞰変換まで表示
- セグメンテーション結果のみ表示
5. 実行すると以下を表示
- 入力画像
- 道路マスク（生出力）
- 道路マスク（後処理後）
- マスク重畳画像
- （俯瞰変換モード時）4点推定結果・俯瞰画像
6. 学習フォームから学習ジョブを開始/停止できる
7. 歪み補正フォームから補正画像を生成できる

## 自分のデータセットで学習する方法

学習フォームの「データセット指定方法」で以下を選べます。

- `ディレクトリ指定`
  - `学習画像ディレクトリ` と `学習マスクディレクトリ` を入力
  - 相対パスは `DEEPLAB_ROOT` 基準、絶対パスも指定可能
  - 実行時に `png` ペアへ自動正規化してから学習
- `zipアップロード`
  - `images.zip` と `masks.zip` をアップロード
  - 画像とマスクは同名（またはマスク側が `*_mask`）で対応付け
  - 内部で `png` に正規化して学習を開始

## 4点選択を学習する方法

4点推定学習フォームでは、次の2モードで回帰モデルを学習できます。

- `RGBのみ`
  - 入力: 画像
- `RGB + Mask`
  - 入力: 画像 + 道路マスク
  - 推論時は道路セグメンテーションで得たマスクを自動で4点モデルへ入力します。

- `ディレクトリ指定`
  - `学習画像ディレクトリ` と `4点ラベルCSV` を指定
  - `RGB + Mask` の場合は `学習マスクディレクトリ` も指定
  - 相対パスはこのリポジトリ基準、絶対パスも指定可能
- `zipアップロード`
  - `images.zip` と `labels.csv` をアップロード
  - `RGB + Mask` の場合は `masks.zip` もアップロード

CSV必須列:
- `filename, lt_x, lt_y, rt_x, rt_y, rb_x, rb_y, lb_x, lb_y`

座標値:
- ピクセル座標（例: `256.0`）または `0..1` 正規化座標のどちらでも可

備考:
- 同名画像が複数ある場合は、`filename` にサブディレクトリを含めて一意にしてください。
- マスクは同名ファイル、または `*_mask` 命名で画像と対応付けできます。
- 学習済み重みは `quadpoint_best.pth` として保存されます（保存先はフォームで指定可能）。

## 実装メモ

- 道路セグメンテーションは `src/segmentation.py` の `RoadSegmentationPipeline` で管理しています。
- 推論器をクラス分離しているため、DeepLabV3+ 学習済みモデルの実装差し替えが容易です。
- Web 推論の前処理は `infer/scripts/infer_road.py` と合わせて、`512x512` + ImageNet 正規化にしています。
- 例外は Web 画面に日本語で表示されるようにしています。
