#!/usr/bin/env python3
"""
ワニ画像検出用の学習データ生成スクリプト

機能:
1. 既存のワニ画像を元にバリエーション生成
2. ランダム背景画像生成
3. ワニ画像の合成とアノテーション自動生成
4. YOLO形式での出力
"""

import os
import random
import json
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import argparse


class WaniDataGenerator:
    def __init__(self, wani_dir: str, output_dir: str):
        self.wani_dir = Path(wani_dir)
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"

        # ディレクトリ作成
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        # ワニ画像読み込み
        self.wani_images = self.load_wani_images()

    def load_wani_images(self) -> List[np.ndarray]:
        """ワニ画像を読み込み"""
        wani_images = []
        for img_path in self.wani_dir.glob("*.jpg"):
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is not None:
                wani_images.append(img)
        return wani_images

    def generate_background(self, width: int = 640, height: int = 640) -> np.ndarray:
        """実ゲーム筐体風の背景画像を生成"""
        background = np.zeros((height, width, 3), dtype=np.uint8)

        # 80%の確率でゲーム筐体風背景、20%で従来のランダム背景
        if random.random() < 0.8:
            # 上部：黄色いゲーム筐体部分（20-50%のランダム割合）
            yellow_ratio = random.uniform(0.2, 0.5)
            yellow_height = int(height * yellow_ratio)
            yellow_color = [30, 200, 255]  # BGR: 黄色
            background[:yellow_height, :] = yellow_color

            # 下部：青いゲーム盤面
            blue_color = [200, 100, 30]  # BGR: 青色
            background[yellow_height:, :] = blue_color

            # 微妙なノイズ追加でリアル感向上
            noise = np.random.randint(-15, 15, background.shape, dtype=np.int16)
            background = np.clip(background.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        else:
            # 従来のランダム背景も残す（バリエーション確保）
            bg_type = random.choice(["gradient", "noise", "solid"])

            if bg_type == "gradient":
                # グラデーション背景
                color1 = np.random.randint(0, 255, 3)
                color2 = np.random.randint(0, 255, 3)
                for y in range(height):
                    ratio = y / height
                    color = color1 * (1 - ratio) + color2 * ratio
                    background[y, :] = color.astype(np.uint8)

            elif bg_type == "noise":
                # ノイズ背景
                background = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
                background = cv2.GaussianBlur(background, (15, 15), 0)

            else:  # solid
                # 単色背景
                color = np.random.randint(0, 255, 3)
                background = np.full((height, width, 3), color, dtype=np.uint8)

        return background

    def augment_wani_image(self, wani_img: np.ndarray) -> np.ndarray:
        """ワニ画像にデータ拡張を適用"""
        # PILに変換
        pil_img = Image.fromarray(cv2.cvtColor(wani_img, cv2.COLOR_BGR2RGB))

        # ランダム変換
        transformations = []

        # 回転
        if random.random() < 0.7:
            angle = random.uniform(-30, 30)
            pil_img = pil_img.rotate(angle, fillcolor=(0, 0, 0))

        # スケーリング
        if random.random() < 0.5:
            scale = random.uniform(0.8, 1.2)
            new_size = (int(pil_img.width * scale), int(pil_img.height * scale))
            pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)

        # 明度調整
        if random.random() < 0.5:
            enhancer = ImageEnhance.Brightness(pil_img)
            pil_img = enhancer.enhance(random.uniform(0.7, 1.3))

        # コントラスト調整
        if random.random() < 0.5:
            enhancer = ImageEnhance.Contrast(pil_img)
            pil_img = enhancer.enhance(random.uniform(0.8, 1.2))

        # 色相調整
        if random.random() < 0.3:
            enhancer = ImageEnhance.Color(pil_img)
            pil_img = enhancer.enhance(random.uniform(0.8, 1.2))

        # ブラー
        if random.random() < 0.2:
            pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

        # OpenCVに戻す
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def create_mask_from_green(self, img: np.ndarray, tolerance: int = 30) -> np.ndarray:
        """ワニ全体のマスクを作成（青い背景を除去）"""
        # HSVに変換
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 青い背景を除去するマスク
        # 青色の範囲を定義
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # 青い部分を除去したマスク（ワニ部分）
        wani_mask = cv2.bitwise_not(blue_mask)

        # グレースケールでも追加の判定
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 暗すぎる部分（影など）も除去
        _, bright_mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

        # 2つのマスクを組み合わせ
        combined_mask = cv2.bitwise_and(wani_mask, bright_mask)

        # ノイズ除去
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

        # 輪郭検出で最大の領域のみ保持
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            # フォールバック：青以外の全部分
            return wani_mask

        # 最大面積の輪郭を選択
        largest_contour = max(contours, key=cv2.contourArea)

        # 最終マスク作成
        final_mask = np.zeros_like(combined_mask)
        cv2.fillPoly(final_mask, [largest_contour], 255)

        # 少し膨張させてエッジを確実にカバー
        kernel = np.ones((3, 3), np.uint8)
        final_mask = cv2.dilate(final_mask, kernel, iterations=1)

        return final_mask

    def composite_wani_on_background(self, wani_img: np.ndarray, background: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        """ワニ画像を背景に合成し、バウンディングボックスを返す"""
        # ワニ画像を拡張
        wani_augmented = self.augment_wani_image(wani_img)

        # マスクを作成
        mask = self.create_mask_from_green(wani_augmented)

        # ワニのサイズを調整
        bg_h, bg_w = background.shape[:2]
        wani_h, wani_w = wani_augmented.shape[:2]

        # スケーリング (背景の10-25%のサイズ - より小さく)
        scale = random.uniform(0.1, 0.25)
        new_wani_w = int(wani_w * scale)
        new_wani_h = int(wani_h * scale)

        # リサイズ
        wani_resized = cv2.resize(wani_augmented, (new_wani_w, new_wani_h))
        mask_resized = cv2.resize(mask, (new_wani_w, new_wani_h))

        # ゲーム画面に合わせた配置（穴の位置付近優先）
        max_x = bg_w - new_wani_w
        max_y = bg_h - new_wani_h

        if max_x <= 0 or max_y <= 0:
            # ワニが背景より大きい場合は小さくする
            scale = min(bg_w / wani_w, bg_h / wani_h) * 0.8
            new_wani_w = int(wani_w * scale)
            new_wani_h = int(wani_h * scale)
            wani_resized = cv2.resize(wani_augmented, (new_wani_w, new_wani_h))
            mask_resized = cv2.resize(mask, (new_wani_w, new_wani_h))
            max_x = bg_w - new_wani_w
            max_y = bg_h - new_wani_h

        # 完全ランダム配置
        x = random.randint(0, max(0, max_x))
        y = random.randint(0, max(0, max_y))

        # 合成
        result = background.copy()
        mask_3ch = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR) / 255.0

        roi = result[y : y + new_wani_h, x : x + new_wani_w]
        result[y : y + new_wani_h, x : x + new_wani_w] = roi * (1 - mask_3ch) + wani_resized * mask_3ch

        # YOLO形式のアノテーション（正規化座標）
        center_x = (x + new_wani_w / 2) / bg_w
        center_y = (y + new_wani_h / 2) / bg_h
        width = new_wani_w / bg_w
        height = new_wani_h / bg_h

        return result, [center_x, center_y, width, height]

    def generate_dataset(self, num_images: int = 1000):
        """データセットを生成"""
        print(f"Generating {num_images} training images...")

        for i in range(num_images):
            # ランダムなワニ画像を選択
            wani_img = random.choice(self.wani_images)

            # 背景生成
            background = self.generate_background()

            # 合成
            result_img, bbox = self.composite_wani_on_background(wani_img, background)

            # 保存
            img_filename = f"wani_{i:06d}.jpg"
            label_filename = f"wani_{i:06d}.txt"

            cv2.imwrite(str(self.images_dir / img_filename), result_img)

            # YOLO形式のラベル保存
            with open(self.labels_dir / label_filename, "w") as f:
                # class_id x_center y_center width height
                f.write(f"0 {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{num_images} images")

        # クラス設定ファイル作成
        classes_file = self.output_dir / "classes.txt"
        with open(classes_file, "w") as f:
            f.write("wani\n")

        # YOLO設定ファイル作成
        config = {"train": str(self.images_dir), "val": str(self.images_dir), "nc": 1, "names": ["wani"]}

        with open(self.output_dir / "dataset.yaml", "w") as f:
            for key, value in config.items():
                f.write(f"{key}: {value}\n")

        print(f"Dataset generation completed!")
        print(f"Images: {self.images_dir}")
        print(f"Labels: {self.labels_dir}")
        print(f"Config: {self.output_dir / 'dataset.yaml'}")


def main():
    parser = argparse.ArgumentParser(description="Generate Wani detection training data")
    parser.add_argument("--wani-dir", default="images/wani", help="Directory containing wani images")
    parser.add_argument("--output-dir", default="training_data", help="Output directory")
    parser.add_argument("--num-images", type=int, default=1000, help="Number of images to generate")

    args = parser.parse_args()

    generator = WaniDataGenerator(args.wani_dir, args.output_dir)
    generator.generate_dataset(args.num_images)


if __name__ == "__main__":
    main()
