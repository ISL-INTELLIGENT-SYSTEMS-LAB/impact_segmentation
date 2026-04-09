#!/usr/bin/env python3
"""
Create SAM detection figures for slide use.

Outputs:
- 01_map50_all_objects.png
- 02_map5095_strong_not_perfect.png
- 03_map90_strict_zoom.png
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from transformers import Sam3Model, Sam3Processor
from torchvision.ops import box_iou


IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def load_pil_image(path):
    return Image.open(path).convert("RGB")


def xywh_to_xyxy(x, y, w, h):
    return [float(x), float(y), float(x + w), float(y + h)]


def read_labels_csv(csv_path):
    by_image = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_name = row["image_name"]
            by_image.setdefault(image_name, []).append({
                "label_name": row["label_name"],
                "bbox_xyxy": xywh_to_xyxy(
                    float(row["bbox_x"]),
                    float(row["bbox_y"]),
                    float(row["bbox_width"]),
                    float(row["bbox_height"]),
                )
            })
    return by_image


class SamPredictor:
    def __init__(self, model_name: str, device: str, prompt: str, threshold: float = 0.5):
        self.device = device
        self.prompt = prompt
        self.threshold = threshold
        self.model = Sam3Model.from_pretrained(model_name).to(device)
        self.processor = Sam3Processor.from_pretrained(model_name)

    @torch.no_grad()
    def predict(self, image_path: str):
        image = load_pil_image(image_path)
        inputs = self.processor(images=image, text=self.prompt, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.threshold,
            mask_threshold=0.5,
            target_sizes=inputs["original_sizes"].tolist(),
        )[0]

        boxes = results["boxes"].detach().cpu().numpy().tolist() if "boxes" in results else []
        scores = results["scores"].detach().cpu().numpy().tolist() if "scores" in results else []
        return boxes, scores


def greedy_match(pred_boxes, gt_items, iou_thresh=0.5):
    if len(pred_boxes) == 0 or len(gt_items) == 0:
        return []

    pred_t = torch.tensor(pred_boxes, dtype=torch.float32)
    gt_t = torch.tensor([g["bbox_xyxy"] for g in gt_items], dtype=torch.float32)
    ious = box_iou(pred_t, gt_t)

    pairs = []
    for p in range(ious.shape[0]):
        for g in range(ious.shape[1]):
            pairs.append((float(ious[p, g]), p, g))
    pairs.sort(reverse=True, key=lambda x: x[0])

    used_p = set()
    used_g = set()
    matches = []

    for iou, p, g in pairs:
        if p in used_p or g in used_g:
            continue
        used_p.add(p)
        used_g.add(g)
        matches.append((p, g, iou))

    return matches


def draw_overlay(image_path, gt_items, pred_boxes, out_path, title=None, zoom_box=None):
    img = load_pil_image(image_path)

    if zoom_box is not None:
        x1, y1, x2, y2 = [int(v) for v in zoom_box]
        pad = 25
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(img.width, x2 + pad)
        y2 = min(img.height, y2 + pad)
        img = img.crop((x1, y1, x2, y2))

        # shift boxes into crop coordinates
        shifted_gt = []
        for item in gt_items:
            bx1, by1, bx2, by2 = item["bbox_xyxy"]
            shifted_gt.append({
                "label_name": item["label_name"],
                "bbox_xyxy": [bx1 - x1, by1 - y1, bx2 - x1, by2 - y1]
            })
        gt_items = shifted_gt
        pred_boxes = [[bx1 - x1, by1 - y1, bx2 - x1, by2 - y1] for bx1, by1, bx2, by2 in pred_boxes]

    draw = ImageDraw.Draw(img)

    # GT in blue
    for item in gt_items:
        x1, y1, x2, y2 = item["bbox_xyxy"]
        draw.rectangle([x1, y1, x2, y2], outline="blue", width=4)

    # SAM in red
    for box in pred_boxes:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

    if title:
        draw.rectangle([0, 0, img.width, 36], fill="white")
        draw.text((10, 8), title, fill="black")

    img.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str)
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--labels_csv", required=True)
    parser.add_argument("--output", default="sam_detection_slide_images")
    parser.add_argument("--sam", default="facebook/sam3")
    parser.add_argument("--sam_threshold", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    labels_by_image = read_labels_csv(args.labels_csv)
    predictor = SamPredictor(args.sam, device, args.prompt, args.sam_threshold)

    # Chosen images for the slide
    chosen = {
        "01_map50_all_objects.png": "All_balls_IMG2.png",
        "02_map5095_strong_not_perfect.png": "Ball_10_6_9_IMG.png",
        "03_map90_strict_zoom.png": "Ball_10_IMG.png",
    }

    for out_name, image_name in chosen.items():
        image_path = os.path.join(args.images_dir, image_name)
        gt_items = labels_by_image[image_name]
        pred_boxes, pred_scores = predictor.predict(image_path)

        title = None
        zoom_box = None

        if out_name.startswith("01_"):
            title = "mAP@50: all objects detected"
        elif out_name.startswith("02_"):
            title = "mAP@50:95: strong but not perfectly precise"
        elif out_name.startswith("03_"):
            title = "mAP@90: strict alignment example"
            # zoom around first GT box
            zoom_box = gt_items[0]["bbox_xyxy"]

        draw_overlay(image_path, gt_items, pred_boxes, os.path.join(args.output, out_name), title=title, zoom_box=zoom_box)

        matches = greedy_match(pred_boxes, gt_items)
        print(f"{image_name}: preds={len(pred_boxes)} gt={len(gt_items)} matches={len(matches)}")
        for p, g, iou in matches:
            print(f"  pred {p} ↔ gt {g} IoU={iou:.4f}")

    print(f"\nSaved images to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()