#!/usr/bin/env python3
"""
EfficientDet-D2モデルでワニを検出する推論スクリプト
"""

import os
import argparse
from pathlib import Path
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from effdet import get_efficientdet_config, EfficientDet, DetBenchPredict, DetBenchTrain

class WaniDetectorD2:
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = None
        self.img_size = 768  # D2は768x768
        
        self.setup_model()
        self.setup_transform()
    
    def setup_model(self):
        """D2モデルのセットアップ"""
        config = get_efficientdet_config("efficientdet_d2")
        config.num_classes = 1
        config.image_size = (768, 768)
        
        net = EfficientDet(config, pretrained_backbone=False)
        
        if os.path.exists(self.model_path):
            # DetBenchTrainで重みを読み込み
            from effdet import DetBenchTrain
            train_model = DetBenchTrain(net, config)
            checkpoint = torch.load(self.model_path, map_location=self.device)
            train_model.load_state_dict(checkpoint)
            
            # 推論用のDetBenchPredictを作成
            self.model = DetBenchPredict(net)
            print(f"Loaded D2 model from: {self.model_path}")
        else:
            print(f"Warning: Model file not found: {self.model_path}")
            self.model = DetBenchPredict(net)
        
        self.model.to(self.device)
        self.model.eval()
    
    def setup_transform(self):
        """画像変換の設定（D2用）"""
        self.transform = transforms.Compose([
            transforms.Resize((768, 768)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def detect_image(self, image_path: str) -> list:
        """画像からワニを検出"""
        # 画像読み込み
        image = Image.open(image_path).convert("RGB")
        original_size = image.size
        
        # 前処理
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 推論
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # 結果の解析
        detections = []
        if output is not None and output.numel() > 0:
            output = output.cpu().numpy()
            
            for i in range(output.shape[1]):
                detection = output[0, i]
                
                if len(detection) >= 6:
                    # 座標が入れ替わっている可能性を考慮
                    y1, x1, y2, x2, confidence, class_id = detection
                    
                    if confidence > self.confidence_threshold:
                        # 768x768から元のサイズにスケール
                        scale_x = original_size[0] / 768
                        scale_y = original_size[1] / 768
                        x1 = int(x1 * scale_x)
                        y1 = int(y1 * scale_y)
                        x2 = int(x2 * scale_x)
                        y2 = int(y2 * scale_y)
                        
                        detections.append({
                            "bbox": [x1, y1, x2, y2],
                            "confidence": float(confidence),
                            "class": "wani"
                        })
        
        return detections
    
    def draw_detections(self, image_path: str, output_path: str = None):
        """検出結果を画像に描画"""
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        
        # 検出実行
        detections = self.detect_image(image_path)
        
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            confidence = detection["confidence"]
            
            # バウンディングボックス描画
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            
            # ラベル描画
            label = f"Wani: {confidence:.2f}"
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            draw.text((x1, y1 - 25), label, fill="red", font=font)
        
        if output_path:
            image.save(output_path)
            print(f"Result saved to: {output_path}")
        else:
            image.show()
        
        return detections

def main():
    parser = argparse.ArgumentParser(description="Detect Wani using EfficientDet-D2")
    parser.add_argument("--model", default="models/wani_detector_d2.pth", help="Path to D2 model")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", help="Output path for result")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    
    args = parser.parse_args()
    
    # 入力ファイルの存在確認
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return
    
    # 検出器初期化
    print("=== Using EfficientDet-D2 (768x768) ===")
    detector = WaniDetectorD2(args.model, args.confidence)
    
    # 検出実行
    detections = detector.draw_detections(args.input, args.output)
    print(f"Detected {len(detections)} wani(s)")
    
    for i, detection in enumerate(detections):
        bbox = detection["bbox"]
        conf = detection["confidence"]
        print(f"Wani {i + 1}: bbox={bbox}, confidence={conf:.3f}")

if __name__ == "__main__":
    main()