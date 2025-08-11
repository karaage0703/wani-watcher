# Wani Watcher 🐊

ワニワニパニック用のワニ検出AIシステム

## 概要

EfficientDet (D0/D2)およびRT-DETRを使用した軽量なワニ検出モデルです。画像生成による学習データ拡張とアノテーション自動生成機能付き。Docker環境での開発に最適化されており、コードの変更が即座に反映されるため、効率的な開発が可能です。

## 特徴

- 🎯 複数のモデル対応（用途に応じて選択可能）
  - **EfficientDet-D0** (~6.5MB) - 最軽量、組み込み向け
  - **EfficientDet-D2** (~20MB) - 高精度、バランス型
  - **RT-DETR** (~32-65MB) - 最高速、NMS不要
- 📄 Apache 2.0ライセンス (商用利用可能)
- 🚀 高速推論 (リアルタイム検出対応)
- 🎨 自動データ拡張・アノテーション生成
- 📱 モバイル/組み込み対応

## クイックスタート 🚀

```bash
# 1. リポジトリをクローン
git clone https://github.com/karaage0703/wani-watcher.git
cd wani-watcher

# 2. Dockerイメージをビルド
docker compose build

# 3. 学習データを生成（1000枚）
docker compose --profile generate run --rm wani-generator

# 4. モデルを学習（テスト用：10エポック）
# EfficientDet-D0の場合（最軽量）
docker compose run --rm wani-trainer uv run python train_wani_detector.py --epochs 10

# EfficientDet-D2の場合（高精度）
docker compose --profile d2 run --rm efficientdet-d2-trainer

# RT-DETRの場合（最高速）
docker compose --profile rtdetr run --rm rtdetr-trainer

# 5. 学習結果を確認
ls -la models/
```

## セットアップ

### Docker を使用する場合（推奨）🐳

#### 前提条件

1. **NVIDIA GPU**が搭載されたマシン
2. **NVIDIA Container Toolkit**のインストール
3. **Docker**と**Docker Compose**のインストール

##### NVIDIA Container Toolkitのインストール

```bash
# Ubuntuの場合
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

#### 1. Dockerイメージのビルド

```bash
# 初回または依存関係変更時のみ必要
docker compose build
```

#### 2. 学習データ生成

```bash
# 1000枚の学習データを生成（デフォルト）
docker compose --profile generate run --rm wani-generator

# カスタムパラメータで生成（例：2000枚）
docker compose run --rm wani-generator uv run python generate_training_data.py --num-images 2000
```

#### 3. モデル学習

##### EfficientDet-D0で学習（最軽量）
```bash
# デフォルト設定（50エポック、バッチサイズ8）で学習
docker compose run --rm wani-trainer

# カスタム設定で学習
docker compose run --rm wani-trainer uv run python train_wani_detector.py --epochs 100 --batch-size 16 --lr 1e-3
```

##### EfficientDet-D2で学習（高精度）
```bash
# デフォルト設定で学習（バッチサイズは2に減らしてメモリ対応）
docker compose --profile d2 run --rm efficientdet-d2-trainer

# カスタム設定で学習
docker compose run --rm wani-trainer uv run python train_wani_d2.py --epochs 100 --batch-size 2
```

##### RT-DETRで学習（最高速）
```bash
# デフォルト設定で学習
docker compose --profile rtdetr run --rm rtdetr-trainer

# カスタム設定で学習
docker compose run --rm rtdetr-trainer uv run python train_rtdetr.py --epochs 100 --batch-size 16 --model-size l
```

#### 4. 推論実行

```bash
# test_images/ ディレクトリの画像で検出実行
docker compose --profile detect run --rm wani-detector

# 特定の画像で推論
docker compose run --rm wani-detector uv run python detect_wani.py --input test.jpg --output result.jpg
```

#### GPUの確認

```bash
# コンテナ内でGPUが認識されているか確認
docker compose run --rm wani-trainer nvidia-smi

# PyTorchでGPUが利用可能か確認
docker compose run --rm wani-trainer uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

### ローカル環境を使用する場合

#### 1. 環境構築 (uv使用)

```bash
# uvをインストール (未インストールの場合)
curl -LsSf https://astral.sh/uv/install.sh | sh

# プロジェクトセットアップ
uv sync
```

#### 2. 学習データ生成

```bash
# 1000枚の学習データを生成
uv run python generate_training_data.py --num-images 1000
```

#### 3. モデル学習

```bash
# EfficientDet-D0で学習（最軽量）
uv run python train_wani_detector.py --epochs 50 --batch-size 8

# EfficientDet-D2で学習（高精度）
uv run python train_wani_d2.py --epochs 50 --batch-size 2

# RT-DETRで学習（最高速）
uv run python train_rtdetr.py --epochs 50 --batch-size 16 --model-size l
```

#### 4. 推論実行

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
├── runs/            # 学習結果
├── test_images/     # テスト用画像
├── results/         # 検出結果
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
└── uv.lock
```

## Docker構成の特徴

- **コードのマウント**: ソースコードはコンテナにマウントされるため、コード変更時の再ビルドが不要
- **依存関係のキャッシュ**: `uv`のキャッシュをボリュームで永続化し、ビルド時間を短縮
- **GPU対応**: NVIDIA GPUを自動検出して使用
- **プロファイル機能**: `generate`と`detect`プロファイルで必要なサービスのみ起動

### ボリュームマウント

Docker環境では、コンテナ内の `/workspace` にホストのカレントディレクトリが全てマウントされます。
これにより、コードやデータの変更が即座に反映され、生成されたファイルもホスト側に保存されます。

### トラブルシューティング

#### GPUが認識されない場合

1. NVIDIA Driverの確認
```bash
nvidia-smi
```

2. Docker runtimeの確認
```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

#### メモリ不足エラー

バッチサイズを小さくして実行：
```bash
docker compose run --rm wani-trainer uv run python train_wani_detector.py --batch-size 4
```

#### 権限エラー

生成されたファイルの権限を修正：
```bash
sudo chown -R $USER:$USER models/ training_data/ runs/
```

### パフォーマンス設定

#### GPU使用率の最適化

```bash
# 特定のGPUを指定（例：GPU 0）
export CUDA_VISIBLE_DEVICES=0
docker compose run --rm wani-trainer

# 複数GPUを使用（例：GPU 0と1）
export CUDA_VISIBLE_DEVICES=0,1
docker compose run --rm wani-trainer
```

### 開発モード

インタラクティブモードでコンテナに入る：
```bash
docker compose run --rm -it wani-trainer /bin/bash
```

コンテナ内でJupyter Notebookを起動：
```bash
docker compose run --rm -p 8888:8888 wani-trainer uv run jupyter notebook --ip=0.0.0.0 --allow-root
```

### クリーンアップ

```bash
# コンテナとネットワークの削除
docker compose down

# イメージの削除
docker rmi wani-watcher:latest

# 未使用のボリュームを削除
docker volume prune
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

### train_wani_d2.py
- EfficientDet-D2での学習
- D0より高精度（768x768入力）
- バッチサイズを小さくしてメモリ対応

### train_rtdetr.py
- RT-DETRでの学習（Apache 2.0ライセンス）
- NMS不要で高速推論
- YOLOv11より高速
- モデルサイズ: l (large) / x (extra large)

### detect_wani.py
- 画像・動画でのワニ検出
- バウンディングボックス描画
- 信頼度スコア表示

## パフォーマンス

### モデル比較

| モデル | サイズ | 入力解像度 | FPS (GPU) | AP50 | 用途 |
|--------|-------|-----------|-----------|------|------|
| **EfficientDet-D0** | ~6.5MB | 512x512 | ~30 | >0.9 | 組み込み/モバイル |
| **EfficientDet-D2** | ~20MB | 768x768 | ~20 | >0.93 | 高精度要求 |
| **RT-DETR-L** | ~32MB | 640x640 | ~100+ | >0.92 | リアルタイム |
| **RT-DETR-X** | ~65MB | 640x640 | ~80 | >0.94 | 最高精度 |

### 選択指針
- **組み込み/モバイル**: EfficientDet-D0
- **バランス重視**: EfficientDet-D2
- **速度最優先**: RT-DETR-L
- **精度最優先**: RT-DETR-X

## ライセンス

MIT License - 商用利用可能

### 使用モデルのライセンス
- EfficientDet (D0/D2): Apache 2.0
- RT-DETR: Apache 2.0
- 全モデル商用利用可能、ソース公開義務なし