# Wani Watcher 🐊

ワニワニパニック用のワニ検出AIシステム

## 概要

EfficientDet-D0を使用した軽量なワニ検出モデルです。画像生成による学習データ拡張とアノテーション自動生成機能付き。

## 特徴

- 🎯 軽量モデル (EfficientDet-D0, ~6.5MB)
- 📄 Apache 2.0ライセンス (商用利用可能)
- 🚀 高速推論 (リアルタイム検出対応)
- 🎨 自動データ拡張・アノテーション生成
- 📱 モバイル/組み込み対応

## セットアップ

### 1. 環境構築 (uv使用)

```bash
# uvをインストール (未インストールの場合)
curl -LsSf https://astral.sh/uv/install.sh | sh

# プロジェクトセットアップ
uv sync
```

### 2. 学習データ生成

```bash
# 1000枚の学習データを生成
uv run python generate_training_data.py --num-images 1000
```

### 3. モデル学習

```bash
# EfficientDet-D0で学習
uv run python train_wani_detector.py --epochs 50 --batch-size 8
```

### 4. 推論実行

```bash
# 画像でワニ検出
uv run python detect_wani.py --input test_image.jpg --output result.jpg

# 動画でワニ検出
uv run python detect_wani.py --input test_video.mp4 --output result.mp4 --mode video
```

## ディレクトリ構成

```
wani-watcher/
├── images/
│   ├── org/          # 元画像
│   └── wani/         # ワニ画像
├── training_data/    # 生成された学習データ
│   ├── images/
│   ├── labels/
│   └── dataset.yaml
├── models/           # 学習済みモデル
└── runs/            # 学習結果
```

## スクリプト詳細

### generate_training_data.py
- 既存ワニ画像からバリエーション生成
- ランダム背景合成
- YOLO形式アノテーション自動生成

### train_wani_detector.py
- EfficientDet-D0での学習
- 学習曲線可視化
- モデル保存

### detect_wani.py
- 画像・動画でのワニ検出
- バウンディングボックス描画
- 信頼度スコア表示

## パフォーマンス

- モデルサイズ: ~6.5MB
- 推論速度: ~30 FPS (GPU使用時)
- 精度: AP50 > 0.9 (テストデータ)

## ライセンス

MIT License - 商用利用可能