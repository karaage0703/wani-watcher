# Wani Watcher 🐊

ワニワニパニック用のワニ検出AIシステム

## 概要

EfficientDet-D0を使用した軽量なワニ検出モデルです。画像生成による学習データ拡張とアノテーション自動生成機能付き。Docker環境での開発に最適化されており、コードの変更が即座に反映されるため、効率的な開発が可能です。

## 特徴

- 🎯 軽量モデル (EfficientDet-D0, ~6.5MB)
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
docker compose run --rm wani-trainer uv run python train_wani_detector.py --epochs 10

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

```bash
# デフォルト設定（50エポック、バッチサイズ8）で学習
docker compose run --rm wani-trainer

# カスタム設定で学習
docker compose run --rm wani-trainer uv run python train_wani_detector.py --epochs 100 --batch-size 16 --lr 1e-3
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
# EfficientDet-D0で学習
uv run python train_wani_detector.py --epochs 50 --batch-size 8
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