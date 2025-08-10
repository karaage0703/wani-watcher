#!/usr/bin/env python3
"""
ワニ検出推論スクリプト
学習済みEfficientDet-D0モデルでワニを検出
"""

import os
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from effdet import get_efficientdet_config, EfficientDet, DetBenchPredict
from effdet.data import resolve_input_config


class WaniDetector:
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = None

        self.setup_model()
        self.setup_transform()

    def setup_model(self):
        """モデルのセットアップ"""
        config = get_efficientdet_config("efficientdet_d0")
        config.num_classes = 1
        config.image_size = (512, 512)  # タプル形式で指定

        net = EfficientDet(config, pretrained_backbone=False)
        self.model = DetBenchPredict(net)

        # 学習済み重みを読み込み
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            print(f"Loaded model from: {self.model_path}")
        else:
            print(f"Warning: Model file not found: {self.model_path}")

        self.model.to(self.device)
        self.model.eval()

    def setup_transform(self):
        """画像変換の設定"""
        self.transform = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

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

        # 結果の解析（DetBenchPredictの出力形式に合わせて修正）
        detections = []
        if output is not None and len(output) > 0:
            # DetBenchPredictの出力は (boxes, scores, classes) の形式
            if isinstance(output, (list, tuple)) and len(output) >= 2:
                boxes, scores = output[0], output[1]
                
                # GPUテンソルからCPUに移動
                if hasattr(boxes, 'cpu'):
                    boxes = boxes.cpu().numpy()
                if hasattr(scores, 'cpu'):
                    scores = scores.cpu().numpy()
                
                # 信頼度でフィルタリング
                valid_indices = scores > self.confidence_threshold
                
                for i, valid in enumerate(valid_indices):
                    if valid and i < len(boxes):
                        box = boxes[i]
                        score = scores[i]
                        
                        # YXYX形式からXYXY形式に変換
                        if len(box) >= 4:
                            y1, x1, y2, x2 = box[:4]
                            
                            # 座標を元の画像サイズに変換
                            x1 = int(x1 * original_size[0] / 512)
                            y1 = int(y1 * original_size[1] / 512)
                            x2 = int(x2 * original_size[0] / 512)
                            y2 = int(y2 * original_size[1] / 512)
                            
                            detections.append({"bbox": [x1, y1, x2, y2], "confidence": float(score), "class": "wani"})

        return detections

    def detect_video(self, video_path: str, output_path: str = None):
        """動画からワニを検出"""
        cap = cv2.VideoCapture(video_path)

        if output_path:
            # 動画書き込み設定
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # フレームを一時ファイルとして保存
            temp_path = "temp_frame.jpg"
            cv2.imwrite(temp_path, frame)

            # 検出実行
            detections = self.detect_image(temp_path)

            # 結果を描画
            for detection in detections:
                x1, y1, x2, y2 = detection["bbox"]
                confidence = detection["confidence"]

                # バウンディングボックス描画
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # ラベル描画
                label = f"Wani: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 結果表示
            cv2.imshow("Wani Detection", frame)

            if output_path:
                out.write(frame)

            # 'q'キーで終了
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames")

        # クリーンアップ
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()

        # 一時ファイル削除
        if os.path.exists("temp_frame.jpg"):
            os.remove("temp_frame.jpg")

        print(f"Video processing completed. Total frames: {frame_count}")

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
    parser = argparse.ArgumentParser(description="Detect Wani using EfficientDet-D0")
    parser.add_argument("--model", default="models/wani_detector.pth", help="Path to trained model")
    parser.add_argument("--input", required=True, help="Input image or video path")
    parser.add_argument("--output", help="Output path for result")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--mode", choices=["image", "video"], default="image", help="Processing mode")

    args = parser.parse_args()

    # 入力ファイルの存在確認
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return

    # 検出器初期化
    detector = WaniDetector(args.model, args.confidence)

    if args.mode == "image":
        detections = detector.draw_detections(args.input, args.output)
        print(f"Detected {len(detections)} wani(s)")

        for i, detection in enumerate(detections):
            bbox = detection["bbox"]
            conf = detection["confidence"]
            print(f"Wani {i + 1}: bbox={bbox}, confidence={conf:.3f}")

    else:  # video
        detector.detect_video(args.input, args.output)


if __name__ == "__main__":
    main()
