#!/usr/bin/env python3
"""
EfficientDet-D2モデルでワニ検出器を学習するスクリプト
D2はD0/D1より大きく、より高精度な検出が期待できます
"""

import os
import sys
import argparse
from pathlib import Path

# train_wani_detector.pyの関数を再利用
from train_wani_detector import WaniDetectorTrainer

def main():
    parser = argparse.ArgumentParser(description="Train Wani Detector with EfficientDet-D2")
    parser.add_argument("--data-dir", default="training_data", help="Training data directory")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size (D2 is much larger)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output", default="models/wani_detector_d2.pth", help="Output model path")
    
    args = parser.parse_args()
    
    # EfficientDet-D2でトレーナー作成
    print("=== Training Wani Detector with EfficientDet-D2 ===")
    print("D2 model specifications:")
    print("- Input size: 768x768 (vs D1: 640x640, D0: 512x512)")
    print("- Significantly more parameters than D0/D1")
    print("- Higher accuracy but requires more memory")
    print("- Batch size reduced to 2 due to memory constraints")
    
    trainer = WaniDetectorTrainer(args.data_dir, model_name="efficientdet_d2")
    
    # モデルセットアップ（D2は768x768入力）
    # 注意: EfficientDet-D2のデフォルト入力サイズは768x768
    trainer.setup_model(num_classes=1)
    
    # 学習実行（trainメソッドはepochs, batch_size, lrを受け取る）
    trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
    
    # モデルを指定の場所に保存
    import torch
    from pathlib import Path
    save_path = Path(args.output)
    save_path.parent.mkdir(exist_ok=True)
    torch.save(trainer.model.state_dict(), save_path)
    
    print(f"Training completed! Model saved to {args.output}")
    print("\nModel comparison:")
    print("- D0: 512x512, fastest, lowest accuracy")
    print("- D1: 640x640, balanced")  
    print("- D2: 768x768, highest accuracy, slowest")

if __name__ == "__main__":
    main()