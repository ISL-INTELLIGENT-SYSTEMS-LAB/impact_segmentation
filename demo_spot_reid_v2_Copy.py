#!/usr/bin/env python3
"""
SAM3 + DINOv2 spot-guided reidentification pipeline (Copy / refactor).

- One reference image + prompt → all reference objects (SAM3 boxes).
- Query set: directory of images, limited to 2–3 for testing.
- For every (reference object × query object) pair: similarity score.
- main() is the driver; Pipeline class and helpers do one job each.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from transformers import Sam3Model, Sam3Processor

from dinov2_extractor import DINOv2FeatureExtractor
from datetime import datetime

# Target size for DINOv2
DEMO_SIZE = 224
now = datetime.now().strftime("%Y%m%d")


# ---------------------------------------------------------------------------
# Helper: one job each
# ---------------------------------------------------------------------------

def load_pil_image(path):
    """Load image from path as PIL RGB. Single responsibility."""
    return Image.open(path).convert("RGB")


def load_image_as_tensor(path, size=DEMO_SIZE):
    """Load image from path and return tensor [1, 3, size, size] in [0, 1]. Single responsibility."""
    pil = load_pil_image(path)
    transform = T.Compose([T.Resize((size, size)), T.ToTensor()])
    return transform(pil).unsqueeze(0)


def scale_box_to_224(box_xyxy, orig_w, orig_h):
    """Scale box from original image coords to 224x224. Single responsibility."""
    x1, y1, x2, y2 = box_xyxy
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


def get_spot_features_for_boxes(extractor, image_path, boxes_xyxy, device):
    """
    For one image and multiple boxes: load image once, scale boxes to 224, extract one DINOv2
    feature vector per box. Returns list of tensors [1, embed_dim].
    Single responsibility: image + boxes → list of feature vectors.
    """
    pil = load_pil_image(image_path)
    orig_w, orig_h = pil.size
    img_tensor = load_image_as_tensor(image_path).to(device)
    features_list = []
    for box in boxes_xyxy:
        spot_224 = scale_box_to_224(box, orig_w, orig_h)
        feat = extractor.get_spot_features(img_tensor, spot_224)  # [1, D]
        features_list.append(feat)
    return features_list


def compute_similarity_matrix(ref_features, query_features, extractor):
    """
    Compute pairwise cosine similarity: ref_features[i] vs query_features[j].
    Returns 2D array of shape (num_ref, num_query). Single responsibility.
    """
    if not ref_features or not query_features:
        return np.array([[]])
    ref_stack = torch.cat(ref_features, dim=0)   # [R, D]
    query_stack = torch.cat(query_features, dim=0)  # [Q, D]
    sim = extractor.cosine_similarity(
        ref_stack.unsqueeze(1),   # [R, 1, D]
        query_stack.unsqueeze(0),  # [1, Q, D]
    ).squeeze()
    if sim.dim() == 0:
        sim = sim.unsqueeze(0).unsqueeze(0)
    elif sim.dim() == 1:
        sim = sim.unsqueeze(0) if ref_stack.shape[0] == 1 else sim.unsqueeze(1)
    return sim.cpu().numpy()


def list_image_paths(directory, limit=None):
    """List paths to images in directory (png, jpg, jpeg). Optionally limit count. Single responsibility."""
    path = Path(directory)
    if not path.is_dir():
        return []
    exts = {".png", ".jpg", ".jpeg"}
    paths = sorted([str(p) for p in path.iterdir() if p.suffix.lower() in exts])
    if limit is not None:
        paths = paths[:limit]
    return paths


def load_reid_ground_truth(gt_path):
    """
    Load optional re-id ground truth JSON for mAP evaluation.

    Expected JSON format (list of objects):
    [
      {
        "query_image": "query_a.png",
        "query_box_idx": 0,
        "relevant_ref_indices": [1, 3]
      }
    ]

    query_image may be basename or full path. Internally we key by basename.
    """
    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gt = {}
    for item in data:
        qimg = item["query_image"]
        qidx = int(item["query_box_idx"])
        refs = set(int(i) for i in item["relevant_ref_indices"])
        key = (os.path.basename(qimg), qidx)
        gt[key] = refs
    return gt


def compute_map_with_torchmetrics(eval_records):
    """
    Compute retrieval mAP using TorchMetrics from collected pairwise records.

    eval_records item format:
    {
      "query_id": int,
      "score": float,
      "target": int (0 or 1)
    }
    """
    try:
        from torchmetrics.retrieval import RetrievalMAP
    except Exception as exc:
        raise RuntimeError(
            "TorchMetrics RetrievalMAP is required for --gt_json evaluation. "
            "Install with: pip install torchmetrics"
        ) from exc

    if not eval_records:
        return None

    preds = torch.tensor([r["score"] for r in eval_records], dtype=torch.float32)
    target = torch.tensor([r["target"] for r in eval_records], dtype=torch.long)
    indexes = torch.tensor([r["query_id"] for r in eval_records], dtype=torch.long)

    metric = RetrievalMAP(empty_target_action="skip")
    metric.update(preds=preds, target=target, indexes=indexes)
    return float(metric.compute().item())


def draw_boxes_on_image(image_path, boxes_xyxy, output_path, color=(0, 255, 0), width=3, label=None):
    """
    Load image, draw each box in boxes_xyxy (x1,y1,x2,y2), save to output_path.
    If label is set, draw it at the top of the image (e.g. "yellow ball sam3").
    Single responsibility: image + boxes -> saved visualization.
    """
    from PIL import ImageDraw, ImageFont
    img = load_pil_image(image_path)
    draw = ImageDraw.Draw(img)
    for box in boxes_xyxy:
        x1, y1, x2, y2 = [int(round(x)) for x in box]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    if label:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except OSError:
            font = ImageFont.load_default()
        # Draw a dark outline so text is readable on any background
        for dx, dy in [(-1,-1),(-1,1),(1,-1),(1,1),(0,-1),(0,1),(-1,0),(1,0)]:
            draw.text((11+dx, 11+dy), label, font=font, fill=(0, 0, 0))
        draw.text((11, 11), label, font=font, fill=(255, 255, 0))
    img.save(output_path)


# ---------------------------------------------------------------------------
# Pipeline: holds config and runs SAM3 (single responsibility per method)
# ---------------------------------------------------------------------------

class SpotReidPipeline:
    """
    Holds pipeline config and runs SAM3 on images. Does not hold DINOv2;
    main() creates one extractor and passes it in. Pipeline only does:
    - run_sam3(image_path) → all detections (boxes, scores)
    - get_query_image_paths() → list of paths (limited for testing)
    """

    def __init__(
        self,
        ref_image_path: str,
        text_prompt: str,
        query_directory: str,
        sam3_model_name: str,
        device: str,
        max_query_images: int = 3,
        output_dir: str | None = None,
    ):
        self.ref_image_path = ref_image_path
        self.text_prompt = text_prompt
        self.query_directory = query_directory
        self.sam3_model_name = sam3_model_name
        self.device = device
        self.max_query_images = max_query_images
        self.output_dir = output_dir or "."
        self._sam3_model = None
        self._sam3_processor = None

    def _get_sam3(self):
        """Lazy-load SAM3 model and processor once."""
        if self._sam3_model is None:
            self._sam3_model = Sam3Model.from_pretrained(self.sam3_model_name).to(self.device)
            self._sam3_processor = Sam3Processor.from_pretrained(self.sam3_model_name)
        return self._sam3_model, self._sam3_processor

    def run_sam3(self, image_path):
        """
        Run SAM3 on one image with the pipeline prompt. Returns (boxes_xyxy, scores).
        boxes_xyxy: list of [x1, y1, x2, y2] per detection.
        Single responsibility: image path → detections.
        """
        model, processor = self._get_sam3()
        image = load_pil_image(image_path)
        inputs = processor(images=image, text=self.text_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]
        boxes_np = results["boxes"].cpu().numpy()
        scores_np = results["scores"].cpu().numpy()
        if boxes_np.ndim == 1:
            boxes_np = boxes_np.reshape(1, -1)
        boxes = boxes_np.tolist()  # list of [x1,y1,x2,y2]
        scores = scores_np.tolist()
        if isinstance(scores, np.ndarray):
            scores = scores.tolist()
        return boxes, scores

    def get_query_image_paths(self):
        """Return list of query image paths, limited to max_query_images. Single responsibility."""
        return list_image_paths(self.query_directory, limit=self.max_query_images)


# ---------------------------------------------------------------------------
# main(): driver only — parses args, then calls helpers / pipeline in sequence
# ---------------------------------------------------------------------------

def parse_args():
    """Parse CLI. Single responsibility."""
    parser = argparse.ArgumentParser(
        description="SAM3 + DINOv2 spot-guided reidentification (reference + query images)"
    )
    parser.add_argument(
        "text_prompt",
        type=str,
        help="SAM3 text prompt, required (e.g. 'yellow ball') — pass as first argument, no --flag",
    )
    parser.add_argument(
        "--ref_image_path",
        "-r",
        type=str,
        default="/home/fog/20250529_124829_Camera_1_2_270.png",
        help="Reference image path",
    )
    parser.add_argument(
        "--image_directory",
        "-i",
        type=str,
        default="/home/fog",
        help="Directory of query images",
    )
    parser.add_argument(
        "--max_query_images",
        "-max",
        type=int,
        default=None,
        help="Max number of query images (for testing)",
    )
    parser.add_argument(
        "--dinov2_model",
        type=str,
        default="vits14",
        choices=list(DINOv2FeatureExtractor.MODEL_CONFIGS.keys()),
        help="DINOv2 model variant",
    )
    parser.add_argument(
        "--sam3_model",
        type=str,
        default="facebook/sam3",
        help="SAM3 model name",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="/home/fog/isl_work/impact/astr_research/copy_v2_results",
        help="Output directory",
    )
    parser.add_argument(
        "--gt_json",
        type=str,
        default=None,
        help=(
            "Optional JSON with query-to-reference relevance labels. "
            "If provided, computes retrieval mAP using TorchMetrics."
        ),
    )
    return parser.parse_args()


def main():
    # 0) Auth for SAM3 if needed
    if os.environ.get("HUGGINGFACE_HUB_TOKEN"):
        from huggingface_hub import login
        login(os.environ["HUGGINGFACE_HUB_TOKEN"])

    # 1) Parse and setup
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output, exist_ok=True)
    gt_map = load_reid_ground_truth(args.gt_json) if args.gt_json else None
    eval_records = []
    next_query_id = 0
    evaluated_gt_keys: set[tuple[str, int]] = set()
    sam_query_counts: dict[str, int] = {}

    if args.max_query_images is None:
        max_query_images = len(list_image_paths(args.image_directory))
    else:
        max_query_images = args.max_query_images

    # 2) Pipeline config (reference + query set)
    pipeline = SpotReidPipeline(
        ref_image_path=args.ref_image_path,
        text_prompt=args.text_prompt,
        query_directory=args.image_directory,
        sam3_model_name=args.sam3_model,
        device=device,
        max_query_images=max_query_images,
        output_dir=args.output,
    )

    # 3) DINOv2 extractor — create once
    extractor = DINOv2FeatureExtractor(model_name=args.dinov2_model, device=device)

    # 4) Reference: one image, all SAM3 boxes
    ref_boxes, ref_scores = pipeline.run_sam3(args.ref_image_path)
    if not ref_boxes:
        print("No reference detections; exiting.")
        return
    
    base_name = os.path.basename(args.ref_image_path)
    ref_vis_path = os.path.join(args.output, f"reference_{base_name}_segmented_{now}.png")
    draw_boxes_on_image(
        args.ref_image_path, ref_boxes, ref_vis_path,
        label=f"{args.text_prompt} reference image",
    )
    print(f"Reference (segmented): {ref_vis_path}")
    ref_features = get_spot_features_for_boxes(
        extractor, args.ref_image_path, ref_boxes, device
    )

    # 5) Query images (limited)
    query_paths = pipeline.get_query_image_paths()
    if not query_paths:
        print("No query images found; exiting.")
        return

    # 6) For each query image: SAM3 → query features → similarity vs all ref → save segmented image
    for qpath in query_paths:
        query_boxes, query_scores = pipeline.run_sam3(qpath)
        
        qname = Path(qpath).stem
        q_vis_path = os.path.join(args.output, f"query_{qname}_segmented_{now}.png")
        
        print(f"  Query (segmented): {q_vis_path}")
        query_features = get_spot_features_for_boxes(
            extractor, qpath, query_boxes, device
        )
        sim_matrix = compute_similarity_matrix(ref_features, query_features, extractor)
        print(f"  {Path(qpath).name}: ref_objects={len(ref_features)}, query_objects={len(query_features)}")
        print(f"    Similarity matrix (ref x query):\n{sim_matrix}")

        qname_base = os.path.basename(qpath)
        num_query_boxes = len(query_boxes)
        sam_query_counts[qname_base] = num_query_boxes

        # Optional: build retrieval records for mAP.
        # Each query detection is one retrieval query; refs are candidates.
        if gt_map is not None and sim_matrix.size > 0:
            num_ref = sim_matrix.shape[0]
            num_query = sim_matrix.shape[1]
            for qidx in range(num_query):
                relevant_refs = gt_map.get((qname_base, qidx))
                if relevant_refs is None:
                    continue
                for ridx in range(num_ref):
                    eval_records.append(
                        {
                            "query_id": next_query_id,
                            "score": float(sim_matrix[ridx, qidx]),
                            "target": int(ridx in relevant_refs),
                        }
                    )
                next_query_id += 1
                evaluated_gt_keys.add((qname_base, qidx))

        label = f"{args.text_prompt} query image"
        if not query_boxes:
            label = f"{label} (no detections)"
        else:
            label = f"{label} Similarity: {sim_matrix.max():.4f}"

        draw_boxes_on_image(qpath, query_boxes, q_vis_path, label=label) 

    if gt_map is not None:
        processed_basenames = {os.path.basename(p) for p in query_paths}
        skipped_gt: list[dict[str, str | int]] = []
        for (basename, qidx), _refs in gt_map.items():
            if (basename, qidx) in evaluated_gt_keys:
                continue
            if basename not in processed_basenames:
                reason = "no matching query file was run (check basename vs --image_directory / --max_query_images)"
            elif sam_query_counts.get(basename, 0) == 0:
                reason = "SAM returned 0 detections for this image (prompt/threshold)"
            elif qidx >= sam_query_counts.get(basename, 0):
                reason = (
                    f"query_box_idx {qidx} not available: SAM only returned "
                    f"{sam_query_counts[basename]} box(es) (indices 0..{sam_query_counts[basename] - 1})"
                )
            else:
                reason = "similarity matrix empty for this image (unexpected)"
            row = {
                "query_image": basename,
                "query_box_idx": qidx,
                "reason": reason,
            }
            skipped_gt.append(row)
            print(
                f"Ground truth not evaluated: query_image={basename!r}, query_box_idx={qidx} — {reason}"
            )

        map_score = compute_map_with_torchmetrics(eval_records)
        metrics_dir = os.path.join(args.output, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_path = os.path.join(metrics_dir, f"metrics_{stamp}.json")
        payload = {
            "run_timestamp_iso": datetime.now().isoformat(timespec="seconds"),
            "metric_name": "RetrievalMAP",
            "torchmetrics_retrieval_map": map_score,
            "num_labeled_queries_evaluated": next_query_id,
            "num_pairwise_records": len(eval_records),
            "gt_json": os.path.abspath(args.gt_json),
            "text_prompt": args.text_prompt,
            "ref_image_path": os.path.abspath(args.ref_image_path),
            "query_directory": os.path.abspath(args.image_directory),
            "dinov2_model": args.dinov2_model,
        }
        if map_score is None:
            print("mAP: no evaluation records were built from gt_json (check labels and detections).")
            payload["note"] = "No eval records; mAP not computed."
        else:
            print(f"mAP (TorchMetrics RetrievalMAP): {map_score:.6f}")
        if skipped_gt:
            payload["skipped_ground_truth"] = skipped_gt
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Metrics saved: {metrics_path}")

if __name__ == "__main__":
    main()
