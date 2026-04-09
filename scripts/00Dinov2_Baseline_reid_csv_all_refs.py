#!/usr/bin/env python3
"""
CSV-driven multi-reference ReID pipeline.

What this version does:
- Uses every labeled object in every image as a reference
- Compares that reference against all other images
- Computes per-reference and overall metrics
- Uses DINOv2 + CSV boxes only (no SAM)
- Uses TorchMetrics for:
    - Top-1 Accuracy
    - mAP (mean Average Precision)
- Writes comparison-friendly output files:
    - metrics_summary.json
    - metrics_per_reference.csv
    - metrics_per_query.csv
    - metrics_report.md

This helps answer:
"If I pick any ball in any image, can the model find the same physical ball in other scenes?"
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.retrieval import RetrievalMAP

from dinov2_extractor import DINOv2FeatureExtractor


DEMO_SIZE = 224
NOW = datetime.now().strftime("%Y%m%d_%H%M%S")


# =========================
# TIMER
# =========================

class Timer:
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        return time.time() - self.start_time


# =========================
# IMAGE + CSV HELPERS
# =========================


def load_pil_image(path):
    return Image.open(path).convert("RGB")



def load_image_as_tensor(path):
    image = load_pil_image(path)
    transform = T.Compose([
        T.Resize((DEMO_SIZE, DEMO_SIZE)),
        T.ToTensor(),
    ])
    return transform(image).unsqueeze(0)



def xywh_to_xyxy(x, y, w, h):
    return [x, y, x + w, y + h]



def scale_box_to_224(box, orig_w, orig_h):
    x1, y1, x2, y2 = box
    sx = DEMO_SIZE / orig_w
    sy = DEMO_SIZE / orig_h

    x1_224 = max(0, min(DEMO_SIZE - 1, int(round(x1 * sx))))
    y1_224 = max(0, min(DEMO_SIZE - 1, int(round(y1 * sy))))
    x2_224 = max(0, min(DEMO_SIZE, int(round(x2 * sx))))
    y2_224 = max(0, min(DEMO_SIZE, int(round(y2 * sy))))

    if x1_224 >= x2_224:
        x2_224 = x1_224 + 1
    if y1_224 >= y2_224:
        y2_224 = y1_224 + 1

    return (x1_224, y1_224, x2_224, y2_224)



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
            image_width = int(row["image_width"])
            image_height = int(row["image_height"])

            item = {
                "label_name": label_name,
                "bbox_xyxy": xywh_to_xyxy(x, y, w, h),
                "image_width": image_width,
                "image_height": image_height,
            }

            by_image.setdefault(image_name, []).append(item)

    return by_image



def draw_boxes_on_image(image_path, items, output_path, highlight_label=None):
    img = load_pil_image(image_path)
    draw = ImageDraw.Draw(img)

    for item in items:
        x1, y1, x2, y2 = item["bbox_xyxy"]
        label = item["label_name"]
        color = (0, 255, 0) if label == highlight_label else (255, 255, 0)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        draw.text((x1, max(0, y1 - 20)), label, fill=color)

    img.save(output_path)


# =========================
# FEATURE HELPERS
# =========================


def get_feature_for_item(extractor, image_path, item, device):
    pil = load_pil_image(image_path)
    orig_w, orig_h = pil.size
    img_tensor = load_image_as_tensor(image_path).to(device)

    scaled_box = scale_box_to_224(item["bbox_xyxy"], orig_w, orig_h)
    feat = extractor.get_spot_features(img_tensor, scaled_box)
    return feat



def cosine_scores_against_gallery(query_feat, gallery_feats, extractor):
    if not gallery_feats:
        return np.array([])

    gallery_stack = torch.cat(gallery_feats, dim=0)
    query_stack = query_feat.repeat(gallery_stack.shape[0], 1)

    sim = extractor.cosine_similarity(query_stack, gallery_stack)
    return sim.detach().cpu().numpy()



def pad_scores_for_multiclass(scores, max_candidates, device):
    """
    TorchMetrics MulticlassAccuracy expects a fixed class dimension.
    Since each query image can have a different number of candidates,
    we pad the score vector to max_candidates using a very low value.
    """
    padded = torch.full((1, max_candidates), -1e9, device=device, dtype=torch.float32)
    padded[0, :len(scores)] = torch.tensor(scores, device=device, dtype=torch.float32)
    return padded


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



def write_markdown_report(path, summary, reference_rows, query_rows):
    top_refs = sorted(
        [r for r in reference_rows if r["positive_queries"] > 0],
        key=lambda r: r["reference_map"],
        reverse=True,
    )[:10]

    hard_refs = sorted(
        [r for r in reference_rows if r["positive_queries"] > 0],
        key=lambda r: r["reference_map"],
    )[:10]

    lines = []
    lines.append("# ReID Baseline Metrics Report")
    lines.append("")
    lines.append(f"Generated: {summary['generated_at']}")
    lines.append("")
    lines.append("## Run Summary")
    lines.append("")
    lines.append(f"- DINOv2 variant: `{summary['dinov2_variant']}`")
    lines.append(f"- Device: `{summary['device']}`")
    lines.append(f"- References evaluated: {summary['references_evaluated']}")
    lines.append(f"- Positive query cases: {summary['total_positive_query_cases']}")
    lines.append(f"- Negative query cases: {summary['total_negative_query_cases']}")
    lines.append(f"- Overall Top-1 Accuracy: {summary['overall_top1_accuracy']:.4f}" if summary['overall_top1_accuracy'] is not None else "- Overall Top-1 Accuracy: n/a")
    lines.append(f"- Overall mAP: {summary['overall_map']:.4f}" if summary['overall_map'] is not None else "- Overall mAP: n/a")
    lines.append("")
    lines.append("## Best References by mAP")
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
    lines.append("## Hardest References by mAP")
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
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    lines.append("- `metrics_summary.json` — overall run summary")
    lines.append("- `metrics_per_reference.csv` — one row per reference object")
    lines.append("- `metrics_per_query.csv` — one row per evaluated query pair")
    lines.append("- `metrics_report.md` — this report")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Top-1 and mAP are computed with TorchMetrics.")
    lines.append("- The per-query AP column is a simple reciprocal-rank style debug value for quick inspection.")
    lines.append("- Negative query cases are logged but skipped for positive-match metrics.")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =========================
# MAIN
# =========================


def main():
    parser = argparse.ArgumentParser(description="CSV-based DINOv2 multi-reference ReID pipeline with TorchMetrics")
    parser.add_argument("--images_dir", required=True, help="Directory containing dataset images")
    parser.add_argument("--labels_csv", required=True, help="CSV file with ground-truth boxes and labels")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--dinov2", default="vits14", help="DINOv2 model variant")
    parser.add_argument("--max_refs", type=int, default=None, help="Optional limit on number of reference objects")
    parser.add_argument("--max_queries", type=int, default=None, help="Optional limit on number of query images per reference")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    labels_by_image = read_labels_csv(args.labels_csv)
    extractor = DINOv2FeatureExtractor(model_name=args.dinov2, device=device)

    max_candidates = max(len(items) for items in labels_by_image.values())
    print(f"Max candidates in any image: {max_candidates}")

    overall_top1_metric = MulticlassAccuracy(
        num_classes=max_candidates,
        top_k=1
    ).to(device)
    overall_map_metric = RetrievalMAP().to(device)

    all_references = []
    for image_name, items in labels_by_image.items():
        for item_idx, item in enumerate(items):
            all_references.append({
                "ref_image": image_name,
                "ref_item": item,
                "ref_index_in_image": item_idx,
            })

    if args.max_refs is not None:
        all_references = all_references[:args.max_refs]

    total_present_cases = 0
    total_absent_cases = 0
    global_query_id = 0

    per_reference_rows = []
    per_query_rows = []

    print(f"Total reference objects to evaluate: {len(all_references)}")

    for ref_num, ref_data in enumerate(all_references, start=1):
        ref_image = ref_data["ref_image"]
        ref_item = ref_data["ref_item"]
        ref_label = ref_item["label_name"]
        ref_image_path = os.path.join(args.images_dir, ref_image)

        ref_top1_metric = MulticlassAccuracy(
            num_classes=max_candidates,
            top_k=1
        ).to(device)
        ref_map_metric = RetrievalMAP().to(device)

        timer = Timer()
        timer.start()
        ref_feat = get_feature_for_item(extractor, ref_image_path, ref_item, device)
        ref_time = timer.stop()

        print("\n" + "=" * 80)
        print(f"[REFERENCE {ref_num}/{len(all_references)}]")
        print(f"Reference image: {ref_image}")
        print(f"Reference label: {ref_label}")
        print(f"Reference feature extraction time: {ref_time:.4f}s")

        ref_vis_name = f"reference_{Path(ref_image).stem}_{ref_label}_{NOW}.png"
        ref_vis_path = os.path.join(args.output, ref_vis_name)
        draw_boxes_on_image(
            ref_image_path,
            labels_by_image[ref_image],
            ref_vis_path,
            highlight_label=ref_label
        )

        image_names = sorted(labels_by_image.keys())
        query_names = [name for name in image_names if name != ref_image]

        if args.max_queries is not None:
            query_names = query_names[:args.max_queries]

        valid_positive_queries_for_ref = 0
        ref_positive_ap_values = []

        for query_image in query_names:
            query_items = labels_by_image[query_image]
            query_image_path = os.path.join(args.images_dir, query_image)

            timer.start()
            query_feats = [
                get_feature_for_item(extractor, query_image_path, item, device)
                for item in query_items
            ]
            feat_time = timer.stop()

            scores = cosine_scores_against_gallery(ref_feat, query_feats, extractor)

            print(f"\nQuery image: {query_image}")
            for i, (item, score) in enumerate(zip(query_items, scores)):
                print(f"  [{i}] {item['label_name']}: {score:.4f}")
            print(f"Feature extraction time: {feat_time:.4f}s")

            gt_indices = [i for i, item in enumerate(query_items) if item["label_name"] == ref_label]

            query_vis_name = f"query_{Path(ref_image).stem}_{ref_label}__vs__{Path(query_image).stem}_{NOW}.png"
            query_vis_path = os.path.join(args.output, query_vis_name)
            draw_boxes_on_image(
                query_image_path,
                query_items,
                query_vis_path,
                highlight_label=ref_label
            )

            if gt_indices:
                total_present_cases += 1
                valid_positive_queries_for_ref += 1
                gt_index = gt_indices[0]

                scores_tensor = pad_scores_for_multiclass(scores, max_candidates, device)
                target_class = torch.tensor([gt_index], device=device, dtype=torch.long)

                ref_top1_metric.update(scores_tensor, target_class)
                overall_top1_metric.update(scores_tensor, target_class)

                retrieval_scores = torch.tensor(scores, device=device, dtype=torch.float32)
                target_binary = torch.zeros(len(scores), device=device, dtype=torch.long)
                target_binary[gt_index] = 1
                query_indexes = torch.full(
                    (len(scores),),
                    fill_value=global_query_id,
                    device=device,
                    dtype=torch.long
                )

                ref_map_metric.update(retrieval_scores, target_binary, query_indexes)
                overall_map_metric.update(retrieval_scores, target_binary, query_indexes)
                global_query_id += 1

                query_top1 = float(torch.argmax(torch.tensor(scores)).item() == gt_index)
                query_ap = 1.0 / (np.where(np.argsort(scores)[::-1] == gt_index)[0][0] + 1)
                ref_positive_ap_values.append(query_ap)

                print(f"GT label found at index: {gt_index}")
                print(f"Top-1: {query_top1:.4f}")
                print(f"AP (single-query reciprocal rank style): {query_ap:.4f}")

                per_query_rows.append({
                    "reference_num": ref_num,
                    "reference_image": ref_image,
                    "reference_label": ref_label,
                    "query_image": query_image,
                    "is_positive_case": 1,
                    "gt_index": gt_index,
                    "top1": query_top1,
                    "ap_debug": query_ap,
                    "best_score": safe_float(np.max(scores)) if len(scores) else None,
                    "gt_score": safe_float(scores[gt_index]) if len(scores) else None,
                    "num_candidates": int(len(scores)),
                    "feature_extraction_time_sec": feat_time,
                })
            else:
                total_absent_cases += 1
                print("GT label not present in this query image; skipped metrics.")

                per_query_rows.append({
                    "reference_num": ref_num,
                    "reference_image": ref_image,
                    "reference_label": ref_label,
                    "query_image": query_image,
                    "is_positive_case": 0,
                    "gt_index": None,
                    "top1": None,
                    "ap_debug": None,
                    "best_score": safe_float(np.max(scores)) if len(scores) else None,
                    "gt_score": None,
                    "num_candidates": int(len(scores)),
                    "feature_extraction_time_sec": feat_time,
                })

        print("\n--- Reference Summary ---")
        if valid_positive_queries_for_ref > 0:
            ref_top1_value = ref_top1_metric.compute().item()
            ref_map_value = ref_map_metric.compute().item()
            ref_ap_debug_mean = float(np.mean(ref_positive_ap_values)) if ref_positive_ap_values else None

            print(f"Reference Mean Top-1 Accuracy: {ref_top1_value:.4f}")
            print(f"Reference mAP: {ref_map_value:.4f}")

            per_reference_rows.append({
                "reference_num": ref_num,
                "reference_image": ref_image,
                "reference_label": ref_label,
                "positive_queries": valid_positive_queries_for_ref,
                "reference_top1": ref_top1_value,
                "reference_map": ref_map_value,
                "reference_ap_debug_mean": ref_ap_debug_mean,
                "reference_feature_extraction_time_sec": ref_time,
            })
        else:
            print("No valid positive query images for this reference.")
            per_reference_rows.append({
                "reference_num": ref_num,
                "reference_image": ref_image,
                "reference_label": ref_label,
                "positive_queries": 0,
                "reference_top1": None,
                "reference_map": None,
                "reference_ap_debug_mean": None,
                "reference_feature_extraction_time_sec": ref_time,
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
        print(f"Overall mAP: {overall_map_value:.4f}")
    else:
        print("No valid overall metrics were computed.")

    summary_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dinov2_variant": args.dinov2,
        "device": device,
        "references_evaluated": len(all_references),
        "max_candidates": int(max_candidates),
        "total_positive_query_cases": int(total_present_cases),
        "total_negative_query_cases": int(total_absent_cases),
        "overall_top1_accuracy": overall_top1_value,
        "overall_map": overall_map_value,
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
            "positive_queries",
            "reference_top1",
            "reference_map",
            "reference_ap_debug_mean",
            "reference_feature_extraction_time_sec",
        ],
    )
    write_csv(
        per_query_csv_path,
        per_query_rows,
        fieldnames=[
            "reference_num",
            "reference_image",
            "reference_label",
            "query_image",
            "is_positive_case",
            "gt_index",
            "top1",
            "ap_debug",
            "best_score",
            "gt_score",
            "num_candidates",
            "feature_extraction_time_sec",
        ],
    )
    write_markdown_report(markdown_report_path, summary_payload, per_reference_rows, per_query_rows)

    print("\nSaved comparison-friendly outputs:")
    print(f"- {summary_json_path}")
    print(f"- {per_reference_csv_path}")
    print(f"- {per_query_csv_path}")
    print(f"- {markdown_report_path}")


if __name__ == "__main__":
    main()
