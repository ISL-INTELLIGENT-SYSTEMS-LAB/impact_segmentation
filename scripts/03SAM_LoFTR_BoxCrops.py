#!/usr/bin/env python3
"""
SAM -> LoFTR multi-reference ReID pipeline using SAM BOX CROPS
instead of SAM masked crops.

Why this version exists:
- Matches the SAM -> DINO crop style more closely
- Tests whether LoFTR performs better when given box crops
  rather than masked crops
- Keeps the same evaluation structure:
    * every usable object in every image becomes a reference
    * compare against every other image
    * GT used only for evaluation

Outputs:
- metrics_summary.json
- metrics_per_reference.csv
- metrics_per_query.csv
- metrics_report.md
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.ops import box_iou
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.retrieval import RetrievalMAP
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import torchvision.transforms.functional as TF
import kornia as K
import kornia.feature as KF

from transformers import Sam3Model, Sam3Processor


NOW = datetime.now().strftime("%Y%m%d_%H%M%S")
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
DEFAULT_SAM_THRESHOLD = 0.5
IOU_MATCH_THRESHOLD = 0.5
MIN_CROP_SIZE = 32
LOFTR_RESIZE = 256


# =========================
# IMAGE HELPERS
# =========================

def load_pil_image(path):
    return Image.open(path).convert("RGB")


def list_images(folder):
    return sorted(
        [str(p) for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS]
    )


def xywh_to_xyxy(x, y, w, h):
    return [float(x), float(y), float(x + w), float(y + h)]


def clip_box_to_image(box, w, h):
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(1, min(int(x2), w))
    y2 = max(1, min(int(y2), h))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return [x1, y1, x2, y2]


def crop_from_box(image: Image.Image, box):
    w, h = image.size
    x1, y1, x2, y2 = clip_box_to_image(box, w, h)
    if x2 <= x1 or y2 <= y1:
        return None
    return image.crop((x1, y1, x2, y2))


def pad_to_square(pil_img: Image.Image, fill=(0, 0, 0)):
    w, h = pil_img.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), fill)
    x = (side - w) // 2
    y = (side - h) // 2
    canvas.paste(pil_img, (x, y))
    return canvas


def prepare_crop_for_loftr(pil_img: Image.Image, size=LOFTR_RESIZE):
    """
    LoFTR expects grayscale tensor input.
    Returns [1,1,H,W] float32 tensor in [0,1].
    """
    if pil_img is None:
        return None
    if pil_img.size[0] < MIN_CROP_SIZE or pil_img.size[1] < MIN_CROP_SIZE:
        return None

    pil_img = pad_to_square(pil_img)
    pil_img = pil_img.resize((size, size), Image.BILINEAR)

    tensor = TF.to_tensor(pil_img)
    tensor = K.color.rgb_to_grayscale(tensor.unsqueeze(0))
    return tensor


# =========================
# GT HELPERS (EVAL ONLY)
# =========================

def read_labels_csv(csv_path):
    by_image = {}

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_name = row["label_name"]
            x = float(row["bbox_x"])
            y = float(row["bbox_y"])
            w = float(row["bbox_width"])
            h = float(row["bbox_height"])
            image_name = row["image_name"]

            item = {
                "label_name": label_name,
                "bbox_xyxy": xywh_to_xyxy(x, y, w, h),
            }
            by_image.setdefault(image_name, []).append(item)

    return by_image


def match_predictions_to_gt(pred_boxes, gt_items, iou_thresh=IOU_MATCH_THRESHOLD):
    if len(pred_boxes) == 0 or len(gt_items) == 0:
        return [None] * len(pred_boxes)

    pred_t = torch.tensor(pred_boxes, dtype=torch.float32)
    gt_t = torch.tensor([item["bbox_xyxy"] for item in gt_items], dtype=torch.float32)
    ious = box_iou(pred_t, gt_t)

    matched = [None] * len(pred_boxes)
    used_gt = set()

    pairs = []
    for p in range(ious.shape[0]):
        for g in range(ious.shape[1]):
            pairs.append((float(ious[p, g]), p, g))
    pairs.sort(reverse=True, key=lambda x: x[0])

    for iou, p, g in pairs:
        if iou < iou_thresh:
            break
        if matched[p] is not None:
            continue
        if g in used_gt:
            continue

        matched[p] = {
            "label_name": gt_items[g]["label_name"],
            "gt_index": g,
            "iou": iou,
        }
        used_gt.add(g)

    return matched


# =========================
# VIS HELPERS
# =========================

def draw_pred_boxes_on_image(image_path, pred_boxes, matched_info, output_path, highlight_pred_idx=None):
    img = load_pil_image(image_path)
    draw = ImageDraw.Draw(img)

    for idx, box in enumerate(pred_boxes):
        x1, y1, x2, y2 = box
        label = matched_info[idx]["label_name"] if matched_info[idx] is not None else "unmatched"
        color = (0, 255, 0) if idx == highlight_pred_idx else (255, 255, 0)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        draw.text((x1, max(0, y1 - 20)), label, fill=color)

    img.save(output_path)


# =========================
# REPORT HELPERS
# =========================

def safe_float(value):
    if value is None:
        return None
    return float(value)


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown_report(path, summary, reference_rows):
    top_refs = sorted(
        [r for r in reference_rows if r["positive_queries"] > 0 and r["reference_map"] is not None],
        key=lambda r: r["reference_map"],
        reverse=True,
    )[:10]

    hard_refs = sorted(
        [r for r in reference_rows if r["positive_queries"] > 0 and r["reference_map"] is not None],
        key=lambda r: r["reference_map"],
    )[:10]

    lines = []
    lines.append("# SAM -> LoFTR Box-Crop Multi-Reference Metrics Report")
    lines.append("")
    lines.append(f"Generated: {summary['generated_at']}")
    lines.append("")
    lines.append("## Run Summary")
    lines.append("")
    lines.append(f"- LoFTR type: `{summary['loftr_type']}`")
    lines.append(f"- SAM model: `{summary['sam_model']}`")
    lines.append(f"- Device: `{summary['device']}`")
    lines.append(f"- References evaluated: {summary['references_evaluated']}")
    lines.append(f"- Positive query cases: {summary['total_positive_query_cases']}")
    lines.append(f"- Negative query cases: {summary['total_negative_query_cases']}")
    lines.append(f"- Overall Top-1 Accuracy: {summary['overall_top1_accuracy']:.4f}" if summary['overall_top1_accuracy'] is not None else "- Overall Top-1 Accuracy: n/a")
    lines.append(f"- Overall Retrieval mAP: {summary['overall_map']:.4f}" if summary['overall_map'] is not None else "- Overall Retrieval mAP: n/a")
    lines.append(f"- Detection mAP@50: {summary['det_map_50']:.4f}" if summary['det_map_50'] is not None else "- Detection mAP@50: n/a")
    lines.append(f"- Detection mAP@90: {summary['det_map_90']:.4f}" if summary['det_map_90'] is not None else "- Detection mAP@90: n/a")
    lines.append(f"- Detection mAP@50:95: {summary['det_map']:.4f}" if summary['det_map'] is not None else "- Detection mAP@50:95: n/a")
    lines.append("")
    lines.append("## Best References by Retrieval mAP")
    lines.append("")
    if top_refs:
        lines.append("| Reference Image | Label | Positive Queries | Top-1 | mAP |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in top_refs:
            lines.append(
                f"| {row['reference_image']} | {row['reference_label']} | {row['positive_queries']} | {row['reference_top1']:.4f} | {row['reference_map']:.4f} |"
            )
    else:
        lines.append("No valid references.")
    lines.append("")
    lines.append("## Hardest References by Retrieval mAP")
    lines.append("")
    if hard_refs:
        lines.append("| Reference Image | Label | Positive Queries | Top-1 | mAP |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in hard_refs:
            lines.append(
                f"| {row['reference_image']} | {row['reference_label']} | {row['positive_queries']} | {row['reference_top1']:.4f} | {row['reference_map']:.4f} |"
            )
    else:
        lines.append("No valid references.")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =========================
# SAM
# =========================

class SamPredictor:
    def __init__(self, model_name: str, device: str, prompt: str, threshold: float = DEFAULT_SAM_THRESHOLD):
        self.model_name = model_name
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

        masks = []
        if "masks" in results:
            raw_masks = results["masks"]
            if isinstance(raw_masks, torch.Tensor):
                raw_masks = raw_masks.detach().cpu().numpy()
            for m in raw_masks:
                masks.append(np.asarray(m).astype(bool))

        while len(masks) < len(boxes):
            masks.append(None)

        return boxes, scores, masks


# =========================
# LOFTR
# =========================

class LoFTRScorer:
    def __init__(self, device: str, pretrained: str = "outdoor"):
        self.device = device
        self.matcher = KF.LoFTR(pretrained=pretrained).to(device).eval()

    @torch.inference_mode()
    def score_pair(self, ref_crop: Image.Image, query_crop: Image.Image):
        ref_t = prepare_crop_for_loftr(ref_crop)
        query_t = prepare_crop_for_loftr(query_crop)

        if ref_t is None or query_t is None:
            return {
                "score": -1.0,
                "num_matches": 0,
                "mean_conf": 0.0,
                "median_conf": 0.0,
            }

        ref_t = ref_t.to(self.device)
        query_t = query_t.to(self.device)

        try:
            out = self.matcher({"image0": ref_t, "image1": query_t})
        except Exception:
            return {
                "score": -1.0,
                "num_matches": 0,
                "mean_conf": 0.0,
                "median_conf": 0.0,
            }

        if "confidence" not in out or out["confidence"].numel() == 0:
            return {
                "score": -1.0,
                "num_matches": 0,
                "mean_conf": 0.0,
                "median_conf": 0.0,
            }

        conf = out["confidence"].detach().cpu().numpy()
        num_matches = int(len(conf))
        mean_conf = float(np.mean(conf))
        median_conf = float(np.median(conf))
        score = mean_conf * math.log1p(num_matches)

        return {
            "score": float(score),
            "num_matches": num_matches,
            "mean_conf": mean_conf,
            "median_conf": median_conf,
        }


# =========================
# DETECTION METRICS
# =========================

def build_detection_metric(iou_thresholds):
    return MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        iou_thresholds=iou_thresholds,
        rec_thresholds=[i / 100 for i in range(101)],
        max_detection_thresholds=[1, 10, 100],
        class_metrics=False,
        extended_summary=False,
    )


def update_detection_metric(metric, pred_boxes, pred_scores, gt_items):
    if len(pred_boxes) > 0:
        pred_boxes_t = torch.tensor(pred_boxes, dtype=torch.float32)
        pred_scores_t = torch.tensor(pred_scores, dtype=torch.float32)
        pred_labels_t = torch.zeros((len(pred_boxes),), dtype=torch.int64)
    else:
        pred_boxes_t = torch.zeros((0, 4), dtype=torch.float32)
        pred_scores_t = torch.zeros((0,), dtype=torch.float32)
        pred_labels_t = torch.zeros((0,), dtype=torch.int64)

    if len(gt_items) > 0:
        gt_boxes_t = torch.tensor([item["bbox_xyxy"] for item in gt_items], dtype=torch.float32)
        gt_labels_t = torch.zeros((len(gt_items),), dtype=torch.int64)
    else:
        gt_boxes_t = torch.zeros((0, 4), dtype=torch.float32)
        gt_labels_t = torch.zeros((0,), dtype=torch.int64)

    preds = [{
        "boxes": pred_boxes_t,
        "scores": pred_scores_t,
        "labels": pred_labels_t,
    }]
    target = [{
        "boxes": gt_boxes_t,
        "labels": gt_labels_t,
    }]

    metric.update(preds, target)


# =========================
# METRIC HELPERS
# =========================

def top_k_accuracy(scores, gt_indices, k=1):
    if len(scores) == 0 or len(gt_indices) == 0:
        return 0.0
    k = min(k, len(scores))
    ranked = np.argsort(scores)[::-1][:k]
    return float(any(idx in ranked for idx in gt_indices))


def average_precision(scores, gt_indices):
    if len(scores) == 0 or len(gt_indices) == 0:
        return 0.0

    ranked = np.argsort(scores)[::-1]
    gt_set = set(gt_indices)

    hits = 0
    precisions = []
    for rank_pos, idx in enumerate(ranked, start=1):
        if idx in gt_set:
            hits += 1
            precisions.append(hits / rank_pos)

    return float(np.mean(precisions)) if precisions else 0.0


def pad_scores_for_multiclass(scores, max_candidates, device):
    padded = torch.full((1, max_candidates), -1e9, device=device, dtype=torch.float32)
    padded[0, :len(scores)] = torch.tensor(scores, device=device, dtype=torch.float32)
    return padded


# =========================
# MAIN
# =========================

def main():
    parser = argparse.ArgumentParser(description="SAM -> LoFTR multi-reference ReID pipeline using box crops")
    parser.add_argument("prompt", type=str, help="Text prompt for SAM, e.g. 'yellow ball'")
    parser.add_argument("--images_dir", required=True, help="Directory containing dataset images")
    parser.add_argument("--labels_csv", required=True, help="CSV file with GT boxes and labels (eval only)")
    parser.add_argument("--output", default="output_sam_loftr_boxcrops", help="Output directory")
    parser.add_argument("--sam", default="facebook/sam3", help="SAM model name")
    parser.add_argument("--sam_threshold", type=float, default=DEFAULT_SAM_THRESHOLD, help="SAM score threshold")
    parser.add_argument("--loftr_type", default="outdoor", choices=["outdoor", "indoor"], help="LoFTR pretrained type")
    parser.add_argument("--iou_match", type=float, default=0.5, help="IoU threshold for eval-time pred->GT assignment")
    parser.add_argument("--max_refs", type=int, default=None, help="Optional limit on number of reference predictions")
    parser.add_argument("--max_queries", type=int, default=None, help="Optional limit on number of query images per reference")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_paths = list_images(args.images_dir)
    if len(image_paths) == 0:
        raise ValueError("No images found.")

    sam_predictor = SamPredictor(
        model_name=args.sam,
        device=device,
        prompt=args.prompt,
        threshold=args.sam_threshold,
    )
    loftr_scorer = LoFTRScorer(device=device, pretrained=args.loftr_type)

    labels_by_image = read_labels_csv(args.labels_csv)

    det_metric = build_detection_metric([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
    det_metric_90 = build_detection_metric([0.90])

    pred_records_by_image = {}
    max_candidates = 0

    print("[SAM PASS + DETECTION EVAL]")
    for image_path in image_paths:
        image_name = Path(image_path).name
        gt_items = labels_by_image.get(image_name, [])

        pred_boxes, pred_scores, pred_masks = sam_predictor.predict(image_path)
        matched_info = match_predictions_to_gt(pred_boxes, gt_items, iou_thresh=args.iou_match)

        update_detection_metric(det_metric, pred_boxes, pred_scores, gt_items)
        update_detection_metric(det_metric_90, pred_boxes, pred_scores, gt_items)

        image = load_pil_image(image_path)
        pred_records = []

        matched_count = sum(x is not None for x in matched_info)
        print(f"{image_name}: preds={len(pred_boxes)} gt={len(gt_items)} matched={matched_count}")

        for pred_idx, pred_box in enumerate(pred_boxes):
            crop = crop_from_box(image, pred_box)

            pred_records.append({
                "pred_idx": pred_idx,
                "bbox_xyxy": pred_box,
                "score": float(pred_scores[pred_idx]) if pred_idx < len(pred_scores) else None,
                "mask": pred_masks[pred_idx] if pred_idx < len(pred_masks) else None,
                "crop": crop,
                "matched_info": matched_info[pred_idx],
            })

        max_candidates = max(max_candidates, len(pred_records))
        pred_records_by_image[image_name] = pred_records

    det = det_metric.compute()
    det90 = det_metric_90.compute()

    print("\n" + "=" * 80)
    print("DETECTION SUMMARY")
    print(f"mAP@50:    {float(det['map_50']):.4f}")
    print(f"mAP@90:    {float(det90['map']):.4f}")
    print(f"mAP@50:95: {float(det['map']):.4f}")
    print(f"mAR@1:     {float(det['mar_1']):.4f}")
    print(f"mAR@10:    {float(det['mar_10']):.4f}")
    print(f"mAR@100:   {float(det['mar_100']):.4f}")

    overall_top1_metric = MulticlassAccuracy(
        num_classes=max(1, max_candidates),
        top_k=1
    ).to(device)
    overall_map_metric = RetrievalMAP().to(device)

    all_references = []
    for image_name, pred_items in pred_records_by_image.items():
        for pred_item in pred_items:
            if pred_item["matched_info"] is None:
                continue
            if pred_item["crop"] is None:
                continue
            all_references.append({
                "ref_image": image_name,
                "ref_item": pred_item,
            })

    if args.max_refs is not None:
        all_references = all_references[:args.max_refs]

    total_present_cases = 0
    total_absent_cases = 0
    global_query_id = 0

    per_reference_rows = []
    per_query_rows = []

    print(f"\nTotal reference predictions to evaluate: {len(all_references)}")

    for ref_num, ref_data in enumerate(all_references, start=1):
        ref_image = ref_data["ref_image"]
        ref_item = ref_data["ref_item"]
        ref_label = ref_item["matched_info"]["label_name"]
        ref_crop = ref_item["crop"]

        ref_top1_metric = MulticlassAccuracy(
            num_classes=max(1, max_candidates),
            top_k=1
        ).to(device)
        ref_map_metric = RetrievalMAP().to(device)

        ref_vis_name = f"reference_{Path(ref_image).stem}_{ref_label}_{NOW}.png"
        ref_vis_path = os.path.join(args.output, ref_vis_name)
        draw_pred_boxes_on_image(
            os.path.join(args.images_dir, ref_image),
            [r["bbox_xyxy"] for r in pred_records_by_image[ref_image]],
            [r["matched_info"] for r in pred_records_by_image[ref_image]],
            ref_vis_path,
            highlight_pred_idx=ref_item["pred_idx"]
        )

        print("\n" + "=" * 80)
        print(f"[REFERENCE {ref_num}/{len(all_references)}]")
        print(f"Reference image: {ref_image}")
        print(f"Reference label: {ref_label}")
        print(f"Reference pred idx: {ref_item['pred_idx']}")

        query_names = sorted(labels_by_image.keys())
        query_names = [name for name in query_names if name != ref_image]

        if args.max_queries is not None:
            query_names = query_names[:args.max_queries]

        valid_positive_queries_for_ref = 0
        ref_positive_ap_values = []

        for query_image in query_names:
            query_pred_items = pred_records_by_image.get(query_image, [])
            if len(query_pred_items) == 0:
                continue

            query_scores = []
            query_labels = []

            for item in query_pred_items:
                if item["matched_info"] is None:
                    continue
                if item["crop"] is None:
                    continue

                result = loftr_scorer.score_pair(ref_crop, item["crop"])
                query_scores.append(result["score"])
                query_labels.append(item["matched_info"]["label_name"])

            if len(query_scores) == 0:
                continue

            scores = np.array(query_scores, dtype=np.float32)

            print(f"\nQuery image: {query_image}")
            for i, (lbl, score) in enumerate(zip(query_labels, scores)):
                print(f"  [{i}] {lbl}: {score:.4f}")

            gt_indices = [i for i, lbl in enumerate(query_labels) if lbl == ref_label]

            query_vis_name = f"query_{Path(ref_image).stem}_{ref_label}__vs__{Path(query_image).stem}_{NOW}.png"
            query_vis_path = os.path.join(args.output, query_vis_name)
            draw_pred_boxes_on_image(
                os.path.join(args.images_dir, query_image),
                [r["bbox_xyxy"] for r in pred_records_by_image[query_image]],
                [r["matched_info"] for r in pred_records_by_image[query_image]],
                query_vis_path,
                highlight_pred_idx=None
            )

            if gt_indices:
                total_present_cases += 1
                valid_positive_queries_for_ref += 1

                gt_index = gt_indices[0]

                scores_tensor = pad_scores_for_multiclass(scores, max(1, max_candidates), device)
                target_class = torch.tensor([gt_index], device=device, dtype=torch.long)

                ref_top1_metric.update(scores_tensor, target_class)
                overall_top1_metric.update(scores_tensor, target_class)

                retrieval_scores = torch.tensor(scores, device=device, dtype=torch.float32)
                target_binary = torch.zeros(len(scores), device=device, dtype=torch.long)
                for idx in gt_indices:
                    target_binary[idx] = 1

                query_indexes = torch.full(
                    (len(scores),),
                    fill_value=global_query_id,
                    device=device,
                    dtype=torch.long
                )

                ref_map_metric.update(retrieval_scores, target_binary, query_indexes)
                overall_map_metric.update(retrieval_scores, target_binary, query_indexes)
                global_query_id += 1

                pred_best = int(torch.argmax(torch.tensor(scores)).item())
                query_top1 = float(pred_best in gt_indices)
                query_ap = average_precision(scores, gt_indices)
                ref_positive_ap_values.append(query_ap)

                print(f"GT indices found: {gt_indices}")
                print(f"Top-1: {query_top1:.4f}")
                print(f"AP: {query_ap:.4f}")

                per_query_rows.append({
                    "reference_num": ref_num,
                    "reference_image": ref_image,
                    "reference_label": ref_label,
                    "reference_pred_idx": ref_item["pred_idx"],
                    "query_image": query_image,
                    "is_positive_case": 1,
                    "gt_indices": ";".join(map(str, gt_indices)),
                    "top1": query_top1,
                    "ap": query_ap,
                    "best_score": safe_float(np.max(scores)) if len(scores) else None,
                    "num_candidates": int(len(scores)),
                })
            else:
                total_absent_cases += 1
                print("GT label not present in this query image; skipped metrics.")

                per_query_rows.append({
                    "reference_num": ref_num,
                    "reference_image": ref_image,
                    "reference_label": ref_label,
                    "reference_pred_idx": ref_item["pred_idx"],
                    "query_image": query_image,
                    "is_positive_case": 0,
                    "gt_indices": "",
                    "top1": None,
                    "ap": None,
                    "best_score": safe_float(np.max(scores)) if len(scores) else None,
                    "num_candidates": int(len(scores)),
                })

        print("\n--- Reference Summary ---")
        if valid_positive_queries_for_ref > 0:
            ref_top1_value = ref_top1_metric.compute().item()
            ref_map_value = ref_map_metric.compute().item()
            ref_ap_debug_mean = float(np.mean(ref_positive_ap_values)) if ref_positive_ap_values else None

            print(f"Reference Mean Top-1 Accuracy: {ref_top1_value:.4f}")
            print(f"Reference Retrieval mAP: {ref_map_value:.4f}")

            per_reference_rows.append({
                "reference_num": ref_num,
                "reference_image": ref_image,
                "reference_label": ref_label,
                "reference_pred_idx": ref_item["pred_idx"],
                "positive_queries": valid_positive_queries_for_ref,
                "reference_top1": ref_top1_value,
                "reference_map": ref_map_value,
                "reference_ap_debug_mean": ref_ap_debug_mean,
            })
        else:
            print("No valid positive query images for this reference.")
            per_reference_rows.append({
                "reference_num": ref_num,
                "reference_image": ref_image,
                "reference_label": ref_label,
                "reference_pred_idx": ref_item["pred_idx"],
                "positive_queries": 0,
                "reference_top1": None,
                "reference_map": None,
                "reference_ap_debug_mean": None,
            })

    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print(f"Total positive query cases (object present): {total_present_cases}")
    print(f"Total negative query cases (object absent): {total_absent_cases}")

    overall_top1_value = None
    overall_map_value = None
    if total_present_cases > 0:
        overall_top1_value = overall_top1_metric.compute().item()
        overall_map_value = overall_map_metric.compute().item()
        print(f"Overall Top-1 Accuracy: {overall_top1_value:.4f}")
        print(f"Overall Retrieval mAP: {overall_map_value:.4f}")
    else:
        print("No valid overall metrics were computed.")

    summary_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "loftr_type": args.loftr_type,
        "sam_model": args.sam,
        "device": device,
        "references_evaluated": len(all_references),
        "max_candidates": int(max_candidates),
        "total_positive_query_cases": int(total_present_cases),
        "total_negative_query_cases": int(total_absent_cases),
        "overall_top1_accuracy": overall_top1_value,
        "overall_map": overall_map_value,
        "det_map_50": float(det["map_50"]),
        "det_map_90": float(det90["map"]),
        "det_map": float(det["map"]),
        "det_mar_1": float(det["mar_1"]),
        "det_mar_10": float(det["mar_10"]),
        "det_mar_100": float(det["mar_100"]),
        "output_directory": os.path.abspath(args.output),
    }

    summary_json_path = os.path.join(args.output, "metrics_summary.json")
    per_reference_csv_path = os.path.join(args.output, "metrics_per_reference.csv")
    per_query_csv_path = os.path.join(args.output, "metrics_per_query.csv")
    markdown_report_path = os.path.join(args.output, "metrics_report.md")

    write_json(summary_json_path, summary_payload)
    write_csv(
        per_reference_csv_path,
        per_reference_rows,
        fieldnames=[
            "reference_num",
            "reference_image",
            "reference_label",
            "reference_pred_idx",
            "positive_queries",
            "reference_top1",
            "reference_map",
            "reference_ap_debug_mean",
        ],
    )
    write_csv(
        per_query_csv_path,
        per_query_rows,
        fieldnames=[
            "reference_num",
            "reference_image",
            "reference_label",
            "reference_pred_idx",
            "query_image",
            "is_positive_case",
            "gt_indices",
            "top1",
            "ap",
            "best_score",
            "num_candidates",
        ],
    )
    write_markdown_report(markdown_report_path, summary_payload, per_reference_rows)

    print("\nSaved comparison-friendly outputs:")
    print(f"- {summary_json_path}")
    print(f"- {per_reference_csv_path}")
    print(f"- {per_query_csv_path}")
    print(f"- {markdown_report_path}")


if __name__ == "__main__":
    main()