#!/usr/bin/env python3
"""

Demo: Spot-guided reidentification with DINOv2.

Run after setup:
    python demo_spot_reid.py
    python demo_spot_reid.py --image1 path/to/img1.jpg --image2 path/to/img2.jpg
    python demo_spot_reid.py --spot 80,100           # point (x,y)
    python demo_spot_reid.py --spot 50,50,150,150   # box (x1,y1,x2,y2)
    python demo_spot_reid.py --config pair_config.example.json  # SAM3-style: images + box in original coords
"""
import matplotlib
import numpy as np
import pillow_avif
import torch
from huggingface_hub import login

#################################### For Image ####################################
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor  # type: ignore
from sam3.model_builder import build_sam3_image_model  # type: ignore

#matplotlib.use("Agg")  # Use non-interactive backend
import json
import os

import requests
from transformers import Sam3Model, Sam3Processor

from __future__ import annotations

import argparse
from pathlib import Path


import torchvision.transforms as T


from dinov2_extractor import DINOv2FeatureExtractor

# Target size for DINOv2 (images are resized to this)
DEMO_SIZE = 224

hubtoken = os.environ["HUGGINGFACE_HUB_TOKEN"]
print(hubtoken)
login(hubtoken)


def _visualize_match(
    img1: torch.Tensor,
    img2: torch.Tensor,
    spot: tuple[int, int] | tuple[int, int, int, int],
    best_patch: tuple[int, int],
    patch_size: int,
    score: float,
    save_path: str | None = None,
) -> None:
    """Draw images side-by-side with spot (img1) and matched patch (img2) highlighted."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Convert tensors [1, 3, H, W] to [H, W, 3] for display
    def to_numpy(t: torch.Tensor):
        return t.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1)

    arr1 = to_numpy(img1)
    arr2 = to_numpy(img2)

    ax1.imshow(arr1)
    ax1.set_title("Image 1: Query spot")
    # Draw spot: point (circle) or box (rectangle)
    if len(spot) == 2:
        circle = plt.Circle((spot[0], spot[1]), radius=8, fill=False, color="lime", linewidth=2)
        ax1.add_patch(circle)
        ax1.plot(spot[0], spot[1], "g+", markersize=12)
    else:
        x1, y1, x2, y2 = spot
        rect = mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="lime", facecolor="none")
        ax1.add_patch(rect)
    ax1.axis("off")

    ax2.imshow(arr2)
    ax2.set_title(f"Image 2: Best match (score={score:.3f})")
    # Draw rectangle around matched patch (patch py, px → pixels)
    py, px = best_patch
    rect = mpatches.Rectangle(
        (px * patch_size, py * patch_size),
        patch_size,
        patch_size,
        linewidth=2,
        edgecolor="lime",
        facecolor="none",
    )
    ax2.add_patch(rect)
    ax2.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n[7] Visualization saved to: {save_path}")
    plt.show()


def run_demo(
    image1_path: str | None = "/home/fog/20250529_124829_Camera_1_2_270.png",
    image2_path: str | None = "/home/fog/20250529_124829_Camera_1_2_315.png",
    model_name: str = "vits14",
    device: str | None = None,
    output_path: str | None = None,
    spot: tuple[int, int] | tuple[int, int, int, int] | None = None,
    img1: torch.Tensor | None = None,
    img2: torch.Tensor | None = None,
) -> None:
    """
    Run spot-guided reidentification demo.

    1. Load two images (or use provided tensors / dummies).
    2. Extract patch features from both.
    3. Pick a spot in image1, get its feature.
    4. Find best-matching patch in image2.
    5. Report similarity score and location.
    """
    print("=" * 60)
    print("DINOv2 Spot-Guided Reidentification Demo")
    print("=" * 60)

    # Step 1: Initialize extractor
    print("\n[1] Loading DINOv2 model...")
    extractor = DINOv2FeatureExtractor(
        model_name=model_name,
        device=device,
    )
    print(f"    {extractor}")

    # Step 2: Load or create images
    print("\n[2] Loading images...")
    if img1 is not None and img2 is not None:
        # Pre-loaded (e.g. from SAM3 config)
        img1 = img1.to(extractor.device)
        img2 = img2.to(extractor.device)
        print(f"    Image 1: {image1_path}")
        print(f"    Image 2: {image2_path}")
        print(f"    (from config; box scaled to {DEMO_SIZE}x{DEMO_SIZE})")
    elif image1_path and Path(image1_path).exists():
        transform = T.Compose([
            T.Resize((DEMO_SIZE, DEMO_SIZE)),
            T.ToTensor(),
        ])
        img1 = transform(Image.open(image1_path).convert("RGB")).unsqueeze(0)
        img2 = transform(Image.open(image2_path or image1_path).convert("RGB")).unsqueeze(0)
        img1 = img1.to(extractor.device)
        img2 = img2.to(extractor.device)
        print(f"    Image 1: {image1_path}")
        print(f"    Image 2: {image2_path or image1_path}")
    else:
        img1 = create_dummy_images(DEMO_SIZE, DEMO_SIZE, 1)
        img2 = create_dummy_images(DEMO_SIZE, DEMO_SIZE, 1)
        img1 = img1.to(extractor.device)
        img2 = img2.to(extractor.device)
        print("    Using dummy images (provide --image1 --image2 or --config for real images)")
        if image1_path:
            print(f"    (Image not found: {image1_path})")

    # Step 3: Extract patch features
    print("\n[3] Extracting patch features...")
    patch1, h1, w1 = extractor.get_patch_features(img1)
    patch2, h2, w2 = extractor.get_patch_features(img2)
    print(f"    Image 1: {patch1.shape} (grid {h1}x{w1})")
    print(f"    Image 2: {patch2.shape} (grid {h2}x{w2})")

    # Step 4: Pick a spot in image1 (point or box)
    spot = spot or (112, 112)  # default: center of 224x224
    print(f"\n[4] Spot in image 1: {spot}")
    spot_feat = extractor.get_spot_features(img1, spot)
    print(f"    Spot feature shape: {spot_feat.shape}")

    # Step 5: Find best-matching patch in image2
    print("\n[5] Finding best-matching patch in image 2...")
    best_py, best_px, similarities = extractor.find_best_matching_patch(
        spot_feat.squeeze(0),
        patch2.squeeze(0),
        h2,
        w2,
    )
    best_score = similarities.max().item()
    print(f"    Best match: patch ({best_py}, {best_px})")
    print(f"    Similarity score: {best_score:.4f}")

    # Step 6: Image-level similarity (optional)
    print("\n[6] Image-level (CLS token) similarity...")
    cls1 = extractor.get_image_features(img1)
    cls2 = extractor.get_image_features(img2)
    cls_sim = extractor.cosine_similarity(cls1, cls2).item()
    print(f"    CLS similarity: {cls_sim:.4f}")

    # Step 7: Visualize spot and matched patch
    _visualize_match(
        img1=img1,
        img2=img2,
        spot=spot,
        best_patch=(best_py, best_px),
        patch_size=extractor.patch_size,
        score=best_score,
        save_path=output_path,
    )

    print("\n" + "=" * 60)
    print("Demo complete. Use --image1 and --image2 for your own images.")
    if output_path:
        print(f"Visualization saved to: {output_path}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DINOv2 spot-guided reidentification demo"
    )
    # Defaults below are only used when NO config file is used (see logic after parse_args).
    parser.add_argument(
        "--ref_image_path",
        type=str,
        default="/home/fog/20250529_124829_Camera_1_2_270.png",
        help="Path to reference image",
    )
    parser.add_argument(
        "--image_directory",
        type=str,
        default="/home/fog",
        help="Path to image directory",
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
        help="SAM3 model variant",
    )
    
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="/home/fog/isl_work/impact/spot_reid_results",
        help="Path to save visualization",
    )
   
    args = parser.parse_args()
    output_path = args.output
    os.makedirs(output_path, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    
    
    
    run_demo(
        ref_image_path = args.ref_image_path,
        image_directory = args.image_directory
        dinov2_model = args.dinov2_model,
        sam3_model = args.sam3_model,
        device=device,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
