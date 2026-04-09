#!/usr/bin/env python3
"""
SAM3 + DINOv2 Instance Re-Identification Pipeline (with Metrics)

Pipeline Overview:
  1. Use SAM3 (text-prompt-based detection) to find objects in images
  2. Extract DINOv2 features (embeddings) for each detected object
  3. Compare reference object embeddings vs. query object embeddings using cosine similarity
  4. Rank query objects by similarity; compute metrics (Top-1 Accuracy, mAP)
  5. Visualize results with bounding boxes

Key Concepts:
  - Embedding: A numerical vector representing visual features of an object region
  - Cosine Similarity: Measures how similar two embeddings are (1=identical, 0=orthogonal)
  - Instance Re-ID: Determining if two detections are the same physical object (not just same category)

Usage:
  python demo_spot_reid_v2.py "yellow ball" --ref ref_image.png --queries ./queries_dir/ --output ./results/
"""

# ============================================================================
# IMPORTS
# ============================================================================

# Standard library: argument parsing, file I/O, paths, timing
import argparse
import os
from pathlib import Path
from datetime import datetime
import time

# Scientific computing: numerical arrays, tensor operations
import numpy as np
import torch

# Computer vision: image transformations, model loading
import torchvision.transforms as T
from PIL import Image
from transformers import Sam3Model, Sam3Processor

# Local module: DINOv2 feature extraction wrapper
from dinov2_extractor import DINOv2FeatureExtractor

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================
# DEMO_SIZE: All images resized to 224x224 because DINOv2 expects this input size.
#            224x224 is standard for Vision Transformers; balances speed vs. detail.
DEMO_SIZE = 224

# Timestamp for output filenames (e.g., "20260322" for March 22, 2026)
# Used to organize results: query_image_20260322.png
now = datetime.now().strftime("%Y%m%d")


# ============================================================================
# EVALUATION METRICS
# ============================================================================
# These metrics evaluate how well the pipeline ranks objects by similarity.
# **Important limitation**: Currently uses placeholder gt_index (see main()).
# Real instance re-ID requires ground-truth instance IDs parsed from dataset.

def top_k_accuracy(sim_matrix, gt_index, k=1):
    """
    Top-K Accuracy: Did the ground-truth match appear in top-k ranked objects?
    
    Args:
        sim_matrix: 2D array of [reference_obj_count x query_obj_count] similarities
        gt_index: Index of ground-truth positive match (which query is the right one?)
        k: Only check if gt_index ranked in top-k (default: top-1)
    
    Returns:
        1 if ground-truth in top-k, 0 otherwise
    
    Why this matters:
        If the correct match doesn't rank high, our embedding isn't discriminative enough.
        Top-1 = most challenging (must rank first); Top-5 = more lenient.
    """
    scores = sim_matrix.flatten()
    top_k = np.argsort(scores)[::-1][:k]
    return int(gt_index in top_k)


def compute_map(sim_matrix, gt_index):
    """
    Mean Average Precision (mAP): Reciprocal of ground-truth rank.
    
    Args:
        sim_matrix: 2D similarity matrix [reference x query]
        gt_index: Index of ground-truth match
    
    Returns:
        1.0 / rank_position (e.g., rank 1 → 1.0, rank 2 → 0.5, rank 5 → 0.2)
    
    Why this matters:
        Rewards correct matches that rank high. A match ranked #1 is better than #5.
        mAP averages this across multiple queries to get overall ranking quality.
    """
    scores = sim_matrix.flatten()
    sorted_idx = np.argsort(scores)[::-1]
    rank = np.where(sorted_idx == gt_index)[0][0] + 1
    return 1.0 / rank


class Timer:
    """
    Simple performance profiler. Measures wall-clock duration of operations.
    Used to track SAM3 detection time and DINOv2 feature extraction time.
    """
    def __init__(self):
        self.start_time = None

    def start(self):
        """Record current time as start marker"""
        self.start_time = time.time()

    def stop(self):
        """Return elapsed time (in seconds) since start() was called"""
        return time.time() - self.start_time


# ============================================================================
# HELPER FUNCTIONS: Image Loading and Preprocessing
# ============================================================================

def load_pil_image(path):
    """
    Load image from disk as PIL Image in RGB format.
    Ensures consistent color channel ordering (no alpha, no grayscale).
    """
    return Image.open(path).convert("RGB")


def load_image_as_tensor(path):
    """
    Load image and convert to PyTorch tensor ready for DINOv2.
    
    Steps:
      1. Load as PIL image
      2. Resize to 224x224 (DINOv2 standard input size)
      3. Convert to [0, 1] float tensor (via ToTensor)
      4. Add batch dimension: [1, 3, 224, 224]
    
    Returns:
        torch.Tensor of shape [1, 3, 224, 224] (batch_size=1)
    """
    pil = load_pil_image(path)
    transform = T.Compose([
        T.Resize((DEMO_SIZE, DEMO_SIZE)),  # Resize to 224x224
        T.ToTensor()  # Convert to tensor and normalize to [0, 1]
    ])
    return transform(pil).unsqueeze(0)  # Add batch dimension


def scale_box_to_224(box, w, h):
    """
    Convert bounding box from original image coordinates to 224x224 space.
    
    Why this is needed:
      - SAM3 returns boxes in original image coordinates (e.g., 4032x3024)
      - DINOv2 extracts features on 224x224 images
      - We must map boxes to the resized image space to extract region features
    
    Args:
        box: [x1, y1, x2, y2] in original image coordinates
        w: original image width
        h: original image height
    
    Returns:
        [x1_scaled, y1_scaled, x2_scaled, y2_scaled] in 224x224 space
    """
    x1, y1, x2, y2 = box
    # Compute scaling factors: how many original pixels map to one 224x224 pixel?
    sx, sy = DEMO_SIZE / w, DEMO_SIZE / h

    # Apply scaling and clamp to valid range [0, 224]
    x1 = max(0, int(x1 * sx))
    y1 = max(0, int(y1 * sy))
    x2 = min(DEMO_SIZE, int(x2 * sx))
    y2 = min(DEMO_SIZE, int(y2 * sy))

    # Ensure box has non-zero area (fallback if scaling collapses the box)
    if x1 >= x2:
        x2 = x1 + 1
    if y1 >= y2:
        y2 = y1 + 1

    return (x1, y1, x2, y2)


def get_spot_features_for_boxes(extractor, image_path, boxes, device):
    """
    Extract DINOv2 embeddings for a list of bounding boxes in an image.
    
    Workflow:
      1. Load image and get dimensions
      2. Convert to tensor on GPU/CPU
      3. For each bounding box:
         a. Map from original coordinates to 224x224 space
         b. Extract DINOv2 features for that region (spot = box)
      4. Return list of feature vectors
    
    Args:
        extractor: DINOv2FeatureExtractor instance
        image_path: Path to image file
        boxes: List of bounding boxes in original image coordinates
        device: "cuda" or "cpu"
    
    Returns:
        List of features, each shape [1, embed_dim] (e.g., [1, 1536] for ViT-g14)
    """
    pil = load_pil_image(image_path)
    w, h = pil.size  # Original image dimensions
    img_tensor = load_image_as_tensor(image_path).to(device)

    features = []
    for box in boxes:
        # Convert box from original image space to 224x224 space
        scaled = scale_box_to_224(box, w, h)
        # Extract embedding for this region
        feat = extractor.get_spot_features(img_tensor, scaled)
        features.append(feat)

    return features


def compute_similarity_matrix(ref_features, query_features, extractor):
    """
    Compute pairwise cosine similarity between reference and query embeddings.
    
    Cosine Similarity: Measures how aligned two vectors are in embedding space.
      - 1.0 = identical direction (same visual content)
      - 0.0 = orthogonal (unrelated)
      - -1.0 = opposite direction (very dissimilar)
    
    Result Shape: [num_ref_boxes, num_query_boxes]
      Each cell (i,j) = similarity between reference box i and query box j
    
    Args:
        ref_features: List of reference embeddings, each [1, embed_dim]
        query_features: List of query embeddings, each [1, embed_dim]
        extractor: DINOv2FeatureExtractor (used for cosine_similarity method)
    
    Returns:
        numpy array of shape [num_ref, num_query] with similarity scores in [0, 1]
    """
    if not ref_features or not query_features:
        return np.array([[]])

    # Stack features: [num_ref, embed_dim] and [num_query, embed_dim]
    ref = torch.cat(ref_features, dim=0)
    query = torch.cat(query_features, dim=0)

    # Compute pairwise similarity: [num_ref, 1, embed_dim] vs [1, num_query, embed_dim]
    # Broadcasting produces [num_ref, num_query] similarity matrix
    sim = extractor.cosine_similarity(
        ref.unsqueeze(1),
        query.unsqueeze(0)
    ).squeeze()

    # Handle edge cases: ensure output is always 2D
    if sim.dim() == 0:
        sim = sim.unsqueeze(0).unsqueeze(0)
    elif sim.dim() == 1:
        sim = sim.unsqueeze(0)

    return sim.cpu().numpy()


def draw_boxes_on_image(path, boxes, out_path, label=None):
    """
    Visualize detected bounding boxes on image; save to disk for inspection.
    
    Useful for:
      - Debugging SAM3 detection quality
      - Verifying box coordinates
      - Visual inspection of what the pipeline found
    
    Args:
        path: Path to image file
        boxes: List of bounding boxes (x1, y1, x2, y2) in original image coordinates
        out_path: Path to save visualization
        label: Optional text label to draw at top-left corner
    """
    from PIL import ImageDraw

    img = load_pil_image(path)
    draw = ImageDraw.Draw(img)

    # Draw green rectangles around each detected box
    for b in boxes:
        draw.rectangle(b, outline=(0, 255, 0), width=3)

    # Draw optional text label (e.g., metric scores)
    if label:
        draw.text((10, 10), label, fill=(255, 255, 0))

    img.save(out_path)


def list_images(folder, limit):
    """
    List all image files in a folder (sorted alphabetically).
    
    Why sorted?
      - Makes behavior deterministic and reproducible across runs
      - Easier to debug (same images processed in same order)
    
    Args:
        folder: Path to directory containing images
        limit: Maximum images to return (e.g., 5 for testing; 0 or None = all)
    
    Returns:
        List of absolute image paths, up to limit items
    """
    paths = sorted([
        str(p) for p in Path(folder).iterdir()
        if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
    ])
    return paths[:limit]


# ============================================================================
# PIPELINE: Main class orchestrating SAM3 + DINOv2 workflow
# ============================================================================

class SpotReidPipeline:
    """
    Main pipeline class for instance re-identification.
    
    Responsibilities:
      - Load and manage SAM3 model (lazy loading on first use)
      - Run SAM3 detection on images using text prompts
      - Interface with DINOv2FeatureExtractor for embeddings
    """

    def __init__(self, ref, prompt, query_dir, sam_model, device, max_imgs):
        """
        Initialize pipeline configuration (models loaded lazily).
        
        Args:
            ref: Path to reference image
            prompt: Text prompt for SAM3 (e.g., "yellow ball")
            query_dir: Directory containing query images
            sam_model: SAM3 model identifier (e.g., "facebook/sam3")
            device: "cuda" or "cpu"
            max_imgs: Limit on number of query images to process
        """
        self.ref = ref
        self.prompt = prompt
        self.query_dir = query_dir
        self.sam_model_name = sam_model
        self.device = device
        self.max_imgs = max_imgs

        self.model = None  # Lazy load
        self.processor = None

    def _load_sam(self):
        """
        Lazy-load SAM3 model and processor on first call.
        Saves GPU memory by not loading if SAM3 isn't used.
        """
        if self.model is None:
            self.model = Sam3Model.from_pretrained(self.sam_model_name).to(self.device)
            self.processor = Sam3Processor.from_pretrained(self.sam_model_name)
        return self.model, self.processor

    def run_sam3(self, path):
        """
        Run SAM3 instance segmentation on an image with the stored prompt.
        
        Workflow:
          1. Load SAM3 model (if first time)
          2. Load image and process with text prompt
          3. Run model inference
          4. Post-process to extract bounding boxes and confidence scores
        
        Args:
            path: Path to image to segment
        
        Returns:
            boxes: List of [x1, y1, x2, y2] bounding boxes in original coordinates
            scores: List of confidence scores per detection
        
        Note:
            Uses self.prompt (e.g., "yellow ball") for text-guided detection
        """
        model, processor = self._load_sam()

        image = load_pil_image(path)
        # Prepare input: image + text prompt
        inputs = processor(images=image, text=self.prompt, return_tensors="pt").to(self.device)

        # Run inference (no gradients needed)
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process: convert model outputs to instance masks and boxes
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,  # Confidence threshold
            target_sizes=inputs["original_sizes"].tolist()
        )[0]

        boxes = results["boxes"].cpu().numpy().tolist()
        scores = results["scores"].cpu().numpy().tolist()

        return boxes, scores


# ============================================================================
# MAIN: Entry point and orchestration
# ============================================================================

def main():
    """
    Main orchestration function. Runs the full instance re-ID pipeline.
    
    High-level workflow:
      1. Parse command-line arguments
      2. Initialize SAM3 + DINOv2 extractors
      3. Process reference image: detect → extract embeddings
      4. For each query image: detect → extract embeddings → compare → score
      5. Aggregate metrics and print results
    
    IMPORTANT LIMITATION:
      Uses placeholder gt_index=0. Real instance re-ID needs actual ground-truth
      instance IDs from annotated data. See README for Phase 1 improvements.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str, help="Text prompt for SAM3 (e.g., 'yellow ball')")
    parser.add_argument("--ref", required=True, help="Path to reference image")
    parser.add_argument("--queries", required=True, help="Path to queries directory")
    parser.add_argument("--output", default="output", help="Output directory for results")
    parser.add_argument("--max", type=int, default=3, help="Max query images to process (default: 3)")
    parser.add_argument("--dinov2", default="vits14", help="DINOv2 model variant (default: vits14)")
    parser.add_argument("--sam", default="facebook/sam3", help="SAM3 model name (default: facebook/sam3)")

    args = parser.parse_args()

    # Setup device: use GPU if available, fallback to CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output, exist_ok=True)

    # Initialize pipeline and feature extractor
    pipeline = SpotReidPipeline(
        args.ref, args.prompt, args.queries,
        args.sam, device, args.max
    )
    extractor = DINOv2FeatureExtractor(model_name=args.dinov2, device=device)

    # ====== STEP 1: Process Reference Image ======
    print(f"\n[REFERENCE] Processing: {args.ref}")
    ref_boxes, _ = pipeline.run_sam3(args.ref)

    if not ref_boxes:
        print("ERROR: No detections in reference image. Check your prompt.")
        return

    print(f"  Detected {len(ref_boxes)} objects in reference")
    
    # Extract embeddings for all reference objects
    ref_features = get_spot_features_for_boxes(extractor, args.ref, ref_boxes, device)
    print(f"  Extracted embeddings: {len(ref_features)} features")

    # ====== STEP 2: Process Query Images ======
    query_paths = list_images(args.queries, args.max)
    print(f"\n[QUERIES] Processing {len(query_paths)} images...")

    all_top1 = []
    all_map = []

    for q in query_paths:
        # Timing: measure SAM3 and feature extraction separately
        timer = Timer()

        # SAM3 detection
        timer.start()
        q_boxes, _ = pipeline.run_sam3(q)
        sam_time = timer.stop()

        # DINOv2 feature extraction
        timer.start()
        q_feats = get_spot_features_for_boxes(extractor, q, q_boxes, device)
        feat_time = timer.stop()

        # Compute similarity matrix: [ref_objects, query_objects]
        sim = compute_similarity_matrix(ref_features, q_feats, extractor)

        # Print per-query results
        print(f"\n{Path(q).name}:")
        print(f"  Detected: {len(q_boxes)} objects")
        print(f"  Similarity Matrix:\n{sim}")

        # Compute metrics (currently using placeholder gt_index=0)
        if sim.size > 0:
            gt_index = 0  # PLACEHOLDER: assumes first reference matches first query
                         # Replace with real labels from ground truth

            top1 = top_k_accuracy(sim, gt_index)
            map_score = compute_map(sim, gt_index)

            all_top1.append(top1)
            all_map.append(map_score)

            print(f"  Top-1: {top1} | mAP: {map_score:.4f}")

        print(f"  Timing → SAM: {sam_time:.3f}s | Feature: {feat_time:.3f}s")

        # Save visualization
        out_path = os.path.join(args.output, f"{Path(q).stem}_{now}.png")
        draw_boxes_on_image(q, q_boxes, out_path)
        print(f"  Saved visualization → {out_path}")

    # ====== STEP 3: Final Metrics Aggregation ======
    if all_top1:
        print("\n" + "="*50)
        print("===== FINAL METRICS (AGGREGATED) =====")
        print("="*50)
        print(f"Top-1 Accuracy (avg): {np.mean(all_top1):.4f}")
        print(f"mAP (avg):            {np.mean(all_map):.4f}")
        print("\n⚠️  WARNING: These metrics use placeholder gt_index=0")
        print("   For real instance re-ID, parse ground-truth labels from annotations")
        print("   and replace hardcoded gt_index assumption (see README, Phase 1)")
        print("="*50)


if __name__ == "__main__":
    main()