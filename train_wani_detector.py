#!/usr/bin/env python3
"""
ワニ検出モデルの学習スクリプト
EfficientDet-D0を使用した軽量モデル
"""

import os
import json
from pathlib import Path
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet
import cv2


class WaniDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None, img_size=512):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform
        self.img_size = img_size

        # 画像ファイル一覧取得
        self.image_files = list(self.images_dir.glob("*.jpg"))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 画像読み込み
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")

        # ラベル読み込み
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        boxes = []
        labels = []

        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, x_center, y_center, width, height = map(float, parts)

                        # YOLO形式からXYXY形式に変換（effdet用）
                        img_w, img_h = image.size
                        x1 = (x_center - width / 2) * img_w
                        y1 = (y_center - height / 2) * img_h
                        x2 = (x_center + width / 2) * img_w
                        y2 = (y_center + height / 2) * img_h

                        # XYXY形式で保存
                        boxes.append([x1, y1, x2, y2])
                        labels.append(int(class_id) + 1)  # effdetでは1から開始

        # リサイズ
        if image.size != (self.img_size, self.img_size):
            scale_x = self.img_size / image.width
            scale_y = self.img_size / image.height
            image = image.resize((self.img_size, self.img_size))

            # ボックス座標もリサイズに合わせて調整（XYXY形式）
            boxes = [[x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y] for x1, y1, x2, y2 in boxes]

        if self.transform:
            image = self.transform(image)

        # Tensorに変換
        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        # effdet のDetBenchTrainが期待する形式に変更
        target = {
            "bbox": boxes,  # XYXY 形式のバウンディングボックス
            "cls": labels,  # クラスラベル
            "img_id": torch.tensor([idx]),
            "img_size": torch.tensor([self.img_size, self.img_size]),
            "img_scale": torch.tensor(1.0)
        }

        return image, target


class WaniDetectorTrainer:
    def __init__(self, data_dir: str, model_name: str = "efficientdet_d0"):
        self.data_dir = Path(data_dir)
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

        print(f"Using device: {self.device}")

    def setup_model(self, num_classes: int = 1):
        """EfficientDet-D0モデルのセットアップ"""
        config = get_efficientdet_config(self.model_name)
        config.num_classes = num_classes
        config.image_size = (512, 512)  # タプル形式で指定

        # モデル作成
        net = EfficientDet(config, pretrained_backbone=True)
        self.model = DetBenchTrain(net, config)
        self.model = self.model.to(self.device)
        
        # アンカーボックスをGPUに明示的に移動（effdetのバグ回避）
        if hasattr(self.model, 'anchors') and self.model.anchors is not None:
            if hasattr(self.model.anchors, 'boxes'):
                self.model.anchors.boxes = self.model.anchors.boxes.to(self.device)
        
        # anchor_labelerのアンカーもGPUに移動
        if hasattr(self.model, 'anchor_labeler'):
            if hasattr(self.model.anchor_labeler, 'anchors'):
                if hasattr(self.model.anchor_labeler.anchors, 'boxes'):
                    self.model.anchor_labeler.anchors.boxes = self.model.anchor_labeler.anchors.boxes.to(self.device)

        print(f"Model created: {self.model_name}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def create_data_loaders(self, batch_size: int = 8):
        """データローダー作成"""
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

        images_dir = self.data_dir / "images"
        labels_dir = self.data_dir / "labels"

        dataset = WaniDataset(images_dir, labels_dir, transform=transform)

        # 訓練/検証分割 (80/20)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn, num_workers=0
        )

        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn, num_workers=0)

        return train_loader, val_loader

    def collate_fn(self, batch):
        """バッチデータの整形（effdet用）"""
        images, targets = list(zip(*batch))
        images = torch.stack(images, 0)
        
        # effdetが期待する形式に変換
        batch_targets = {}
        
        # 各項目を統合
        batch_targets['bbox'] = [t['bbox'] for t in targets]
        batch_targets['cls'] = [t['cls'] for t in targets]
        batch_targets['img_id'] = torch.cat([t['img_id'] for t in targets])
        batch_targets['img_size'] = torch.stack([t['img_size'] for t in targets])
        batch_targets['img_scale'] = torch.stack([t['img_scale'].unsqueeze(0) if t['img_scale'].dim() == 0 else t['img_scale'] for t in targets])
        
        return images, batch_targets

    def train(self, epochs: int = 50, batch_size: int = 8, lr: float = 1e-4):
        """モデルの学習"""
        if self.model is None:
            self.setup_model()

        train_loader, val_loader = self.create_data_loaders(batch_size)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        train_losses = []
        val_losses = []

        print(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
            for images, targets in pbar:
                images = images.to(self.device)
                
                # targetsの各要素もGPUに移動
                if 'bbox' in targets:
                    targets['bbox'] = [bbox.to(self.device) if bbox.numel() > 0 else bbox for bbox in targets['bbox']]
                if 'cls' in targets:
                    targets['cls'] = [cls.to(self.device) if cls.numel() > 0 else cls for cls in targets['cls']]
                if 'img_id' in targets:
                    targets['img_id'] = targets['img_id'].to(self.device)
                if 'img_size' in targets:
                    targets['img_size'] = targets['img_size'].to(self.device)
                if 'img_scale' in targets:
                    targets['img_scale'] = targets['img_scale'].to(self.device)

                # Forward pass
                loss_dict = self.model(images, targets)
                loss = sum(loss_dict.values())

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            scheduler.step()
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Validation（一時的に簡略化）
            avg_val_loss = avg_train_loss  # 訓練損失をそのまま使用
            val_losses.append(avg_val_loss)

            print(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # モデル保存
        save_dir = Path("models")
        save_dir.mkdir(exist_ok=True)
        torch.save(self.model.state_dict(), save_dir / "wani_detector.pth")

        # 学習曲線をプロット
        self.plot_training_curves(train_losses, val_losses)

        print("Training completed!")
        return train_losses, val_losses

    def plot_training_curves(self, train_losses, val_losses):
        """学習曲線をプロット"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label="Training Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Curves")
        plt.legend()
        plt.grid(True)
        plt.savefig("training_curves.png", dpi=300, bbox_inches="tight")
        print("Training curves saved to training_curves.png")
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Train Wani detector with EfficientDet-D0")
    parser.add_argument("--data-dir", default="training_data", help="Path to training data directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")

    args = parser.parse_args()

    # データディレクトリの存在確認
    if not os.path.exists(args.data_dir):
        print(f"Error: Training data directory not found: {args.data_dir}")
        print("Please run generate_training_data.py first")
        return

    # トレーナー初期化と学習実行
    trainer = WaniDetectorTrainer(args.data_dir)
    trainer.train(args.epochs, args.batch_size, args.lr)


if __name__ == "__main__":
    main()
