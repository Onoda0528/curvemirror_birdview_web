# カーブミラー俯瞰画像生成 Webアプリ

カーブミラー画像に対して以下を実行する Flask アプリです。

- 道路領域抽出（`models/best_model.pth` があれば DeepLabV3+ 推論、なければ仮マスク）
- 4点推定（比較手法: 最大輪郭四角形近似 / 提案手法: 幅フィルタ + RANSAC）
- 射影変換による俯瞰画像生成

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
`models/best_model.pth` が存在すると DeepLabV3+ 推論を使用します。

4. アプリ起動

```bash
python3 app.py
```

5. ブラウザで開く

```text
http://127.0.0.1:5000
```

## 使い方

1. 画像をアップロード
2. 4点推定モードを選択
- 2手法比較
- 幅フィルタ + RANSAC のみ
- 最大輪郭四角形近似のみ
3. 実行すると以下を表示
- 入力画像
- 道路マスク
- 4点推定結果
- 俯瞰画像

## 実装メモ

- 道路セグメンテーションは `src/segmentation.py` の `RoadSegmentationPipeline` で管理しています。
- 推論器をクラス分離しているため、DeepLabV3+ 学習済みモデルの実装差し替えが容易です。
- 例外は Web 画面に日本語で表示されるようにしています。
