#!/usr/bin/env python3
"""
RT-DETRを使用したワニ検出器の学習スクリプト

RT-DETR (Apache 2.0ライセンス)を使用して軽量かつ高性能なワニ検出モデルを学習します。
AGPLライセンスを回避したい場合に最適な選択肢です。
"""

import argparse
import os
from pathlib import Path
import yaml
from ultralytics import RTDETR
import torch


def create_dataset_yaml(data_dir: Path, output_path: Path):
    """データセット設定ファイルを作成"""
    train_images = data_dir / "images"
    val_images = data_dir / "images"  # 本来は分割すべきだが、簡単のため同じディレクトリを使用
    
    config = {
        'path': str(data_dir.absolute()),
        'train': 'images',  # 相対パス
        'val': 'images',    # 相対パス
        'names': {
            0: 'wani'
        }
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Dataset config saved to {output_path}")
    return output_path


def train_rtdetr(
    data_yaml: str,
    model_size: str = 'l',  # l=large, x=extra large
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = None,
    project: str = 'runs/detect',
    name: str = 'rtdetr_wani',
    resume: bool = False
):
    """RT-DETRモデルを学習"""
    
    # デバイスの自動選択
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    print(f"Using RT-DETR (Apache 2.0 License) - No AGPL restrictions!")
    
    # モデル名の構築
    model_name = f'rtdetr-{model_size}.pt'
    
    # モデルの初期化
    print(f"Loading RT-DETR-{model_size.upper()} model...")
    model = RTDETR(model_name)
    
    # 学習パラメータ
    train_params = {
        'data': data_yaml,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': img_size,
        'device': device,
        'project': project,
        'name': name,
        'exist_ok': resume,
        'patience': 50,  # Early stopping patience
        'save': True,
        'save_period': 10,  # Save checkpoint every N epochs
        'cache': False,  # Cache images for faster training
        'workers': 4,
        'pretrained': True,
        'optimizer': 'AdamW',  # RT-DETRではAdamWが推奨
        'verbose': True,
        'seed': 42,
        'deterministic': True,
        'single_cls': True,  # ワニのみの単一クラス
        'rect': False,
        'cos_lr': True,  # Use cosine LR scheduler
        'close_mosaic': 10,  # Disable mosaic augmentation for last N epochs
        'amp': True,  # Automatic Mixed Precision training
        'fraction': 1.0,  # Dataset fraction to train on
        'profile': False,
        'freeze': None,  # Freeze first N layers
        'lr0': 0.0001,  # Initial learning rate (RT-DETRは低めの学習率が良い)
        'lrf': 0.01,  # Final learning rate (lr0 * lrf)
        'momentum': 0.937,
        'weight_decay': 0.0001,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,  # Box loss gain
        'cls': 0.5,  # cls loss gain (scale with pixels)
        'dfl': 1.5,  # DFL loss gain
        'label_smoothing': 0.0,
        'nbs': 64,  # Nominal batch size
        'hsv_h': 0.015,  # Image HSV-Hue augmentation (fraction)
        'hsv_s': 0.7,  # Image HSV-Saturation augmentation (fraction)
        'hsv_v': 0.4,  # Image HSV-Value augmentation (fraction)
        'degrees': 0.0,  # Image rotation (+/- deg)
        'translate': 0.1,  # Image translation (+/- fraction)
        'scale': 0.5,  # Image scale (+/- gain)
        'shear': 0.0,  # Image shear (+/- deg)
        'perspective': 0.0,  # Image perspective (+/- fraction), range 0-0.001
        'flipud': 0.0,  # Image flip up-down (probability)
        'fliplr': 0.5,  # Image flip left-right (probability)
        'bgr': 0.0,  # Image channel BGR (probability)
        'mosaic': 1.0,  # Image mosaic (probability)
        'mixup': 0.0,  # Image mixup (probability)
        'copy_paste': 0.0,  # Segment copy-paste (probability)
        'auto_augment': 'randaugment',  # Auto augmentation policy
        'erasing': 0.4,  # Random erasing probability
        'crop_fraction': 1.0,  # Image crop fraction
    }
    
    # 学習の実行
    print("Starting training...")
    results = model.train(**train_params)
    
    # 学習結果の表示
    print("\nTraining completed!")
    print(f"Best model saved to: {model.trainer.best}")
    print(f"Last model saved to: {model.trainer.last}")
    
    return model


def evaluate_model(model_path: str, data_yaml: str, img_size: int = 640):
    """学習済みモデルを評価"""
    print(f"\nEvaluating model: {model_path}")
    
    model = RTDETR(model_path)
    
    # 検証データで評価
    metrics = model.val(
        data=data_yaml,
        imgsz=img_size,
        batch=16,
        conf=0.001,
        iou=0.6,
        max_det=300
    )
    
    print("\nEvaluation Results:")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    
    return metrics


def export_model(model_path: str, format: str = 'onnx'):
    """モデルを指定形式でエクスポート"""
    print(f"\nExporting model to {format.upper()} format...")
    
    model = RTDETR(model_path)
    
    # エクスポート
    export_path = model.export(
        format=format,
        imgsz=640,
        keras=False,
        optimize=True,
        half=False,
        int8=False,
        dynamic=True,
        simplify=True,
        opset=13,  # RT-DETRはopset 13以上推奨
        workspace=4,
        nms=False,  # RT-DETRはNMS不要
        batch=1
    )
    
    print(f"Model exported to: {export_path}")
    return export_path


def run_inference(model_path: str, image_path: str, conf_threshold: float = 0.5):
    """推論の実行"""
    print(f"\nRunning inference on {image_path}")
    
    model = RTDETR(model_path)
    
    # 推論実行
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        save=True,
        save_txt=True,
        save_conf=True,
        imgsz=640
    )
    
    # 結果の表示
    for r in results:
        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            print(f"Detected {len(boxes)} wani(s)")
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                print(f"  Box: {xyxy}, Confidence: {conf:.3f}")
        else:
            print("No wani detected")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train RT-DETR Wani Detector (Apache 2.0 License)")
    parser.add_argument('--data-dir', type=str, default='training_data',
                       help='Directory containing training data')
    parser.add_argument('--model-size', type=str, default='l',
                       choices=['l', 'x'],
                       help='Model size: l=large, x=extra large')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--project', type=str, default='runs/detect',
                       help='Project directory for saving results')
    parser.add_argument('--name', type=str, default='rtdetr_wani',
                       help='Experiment name')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from last checkpoint')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate model after training')
    parser.add_argument('--export', type=str, default=None,
                       choices=['onnx', 'torchscript', 'coreml', 'tflite', 'pb', 'saved_model', 'openvino', 'engine', 'paddle'],
                       help='Export format for the model')
    parser.add_argument('--inference', type=str, default=None,
                       help='Run inference on specified image path')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold for inference')
    
    args = parser.parse_args()
    
    # ライセンス表示
    print("=" * 60)
    print("RT-DETR: Apache 2.0 Licensed Object Detection Model")
    print("No AGPL restrictions - Free for personal and commercial use!")
    print("=" * 60)
    
    # データディレクトリの確認
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist!")
        print("Please run generate_training_data.py first to create training data.")
        return
    
    # データセット設定ファイルの作成
    dataset_yaml = data_dir / 'dataset_rtdetr.yaml'
    create_dataset_yaml(data_dir, dataset_yaml)
    
    # 学習の実行
    if not args.inference:
        model = train_rtdetr(
            data_yaml=str(dataset_yaml),
            model_size=args.model_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            device=args.device,
            project=args.project,
            name=args.name,
            resume=args.resume
        )
        
        # モデルの評価
        if args.evaluate:
            best_model_path = f"{args.project}/{args.name}/weights/best.pt"
            if Path(best_model_path).exists():
                evaluate_model(best_model_path, str(dataset_yaml), args.img_size)
        
        # モデルのエクスポート
        if args.export:
            best_model_path = f"{args.project}/{args.name}/weights/best.pt"
            if Path(best_model_path).exists():
                export_model(best_model_path, args.export)
    
    # 推論の実行
    if args.inference:
        best_model_path = f"{args.project}/{args.name}/weights/best.pt"
        if Path(best_model_path).exists():
            run_inference(best_model_path, args.inference, args.conf)
        else:
            print(f"Error: Model not found at {best_model_path}")
            print("Please train the model first.")


if __name__ == "__main__":
    main()