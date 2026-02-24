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

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

from dinov2_extractor import DINOv2FeatureExtractor

# Target size for DINOv2 (images are resized to this)
DEMO_SIZE = 224


def create_dummy_images(
    height: int = 224,
    width: int = 224,
    batch_size: int = 1,
    device: str = "cpu",
) -> torch.Tensor:
    """Create dummy RGB images in [0, 1] for testing."""
    return torch.rand(batch_size, 3, height, width, device=device)


def load_sam3_config(config_path: str) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int, int], str, str]:
    """
    Load two images and a box from a SAM3-style config JSON.

    Config JSON format:
        Use boxes_file_image1 so the script uses the SAME image the SAM3 box was
        computed for (image_path from the boxes file). Coordinates will match.
        {
            "image2": "path/to/image2.png",
            "boxes_file_image1": "path/to/sam3_boxes.json"
        }
        Or with explicit image2 boxes file:
        {
            "boxes_file_image1": "path/to/sam3_boxes_for_image1.json",
            "boxes_file_image2": "path/to/sam3_boxes_for_image2.json"
        }
        Or manual box (image1 must match the coordinate system):
        {
            "image1": "path/to/image1.png",
            "image2": "path/to/image2.png",
            "box_in_image1": [x1, y1, x2, y2]
        }

    Images are resized to DEMO_SIZE (224). The box is scaled to 224x224 coordinates.

    Returns:
        img1, img2: tensors [1, 3, 224, 224]
        box_224: (x1, y1, x2, y2) in 224x224 space
        image1_path, image2_path: for logging
    """
    with open(config_path) as f:
        config = json.load(f)

    image1_path = config.get("image1")
    image2_path = config.get("image2")

    if "boxes_file_image1" in config:
        with open(config["boxes_file_image1"]) as f:
            boxes_data = json.load(f)
        detections = boxes_data.get("detections", [])
        if not detections:
            raise ValueError(
                f"No detections in {config['boxes_file_image1']}. "
                "SAM3 boxes file must have at least one detection with box_xyxy."
            )
        box_orig = detections[0]["box_xyxy"]  # [x1, y1, x2, y2]
        # Box coordinates are for the image in the boxes file - use that image so coords match
        image1_path = boxes_data["image_path"]
    else:
        box_orig = config["box_in_image1"]
        if image1_path is None:
            raise KeyError('Config must have "image1" or "boxes_file_image1"')

    if "boxes_file_image2" in config:
        with open(config["boxes_file_image2"]) as f:
            boxes_data2 = json.load(f)
        image2_path = boxes_data2["image_path"]
    elif image2_path is None:
        raise KeyError('Config must have "image2" or "boxes_file_image2"')

    if not Path(image1_path).exists():
        raise FileNotFoundError(
            f"Image 1 from config not found: {image1_path!r}. "
            "Check paths in your config file (e.g. pair_config.example.json)."
        )
    if not Path(image2_path).exists():
        raise FileNotFoundError(
            f"Image 2 from config not found: {image2_path!r}. "
            "Check paths in your config file (e.g. pair_config.example.json)."
        )

    pil1 = Image.open(image1_path).convert("RGB")
    pil2 = Image.open(image2_path).convert("RGB")
    w1, h1 = pil1.size
    w2, h2 = pil2.size

    transform = T.Compose([T.Resize((DEMO_SIZE, DEMO_SIZE)), T.ToTensor()])
    img1 = transform(pil1).unsqueeze(0)
    img2 = transform(pil2).unsqueeze(0)

    # Scale box from original image1 size to 224x224
    x1, y1, x2, y2 = box_orig
    sx = DEMO_SIZE / w1
    sy = DEMO_SIZE / h1
    x1_224 = max(0, min(DEMO_SIZE - 1, int(round(x1 * sx))))
    y1_224 = max(0, min(DEMO_SIZE - 1, int(round(y1 * sy))))
    x2_224 = max(0, min(DEMO_SIZE, int(round(x2 * sx))))
    y2_224 = max(0, min(DEMO_SIZE, int(round(y2 * sy))))
    if x1_224 >= x2_224:
        x2_224 = x1_224 + 1
    if y1_224 >= y2_224:
        y2_224 = y1_224 + 1
    box_224 = (x1_224, y1_224, x2_224, y2_224)

    return img1, img2, box_224, image1_path, image2_path


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
        "--image1",
        type=str,
        default="/home/fog/20250529_124829_Camera_1_2_270.png",
        help="Path to first image (ignored if --config is used)",
    )
    parser.add_argument(
        "--image2",
        type=str,
        default="/home/fog/20250529_124829_Camera_1_2_315.png",
        help="Path to second image (ignored if --config is used)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="vits14",
        choices=list(DINOv2FeatureExtractor.MODEL_CONFIGS.keys()),
        help="DINOv2 model variant",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cuda, cpu, or auto",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="dinov2_match_result.png",
        help="Path to save visualization",
    )
    parser.add_argument(
        "--spot",
        type=str,
        default=None,
        help='Spot to search for: "x,y" for a point or "x1,y1,x2,y2" for a box (image 1 is 224x224)',
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Path to SAM3-style JSON (e.g. pair_config.example.json): image1, image2, box_in_image1",
    )
    args = parser.parse_args()

    spot = None
    img1_preload = None
    img2_preload = None
    image1_path = args.image1
    image2_path = args.image2

    # Config overrides everything: when a config is used, images and box come from the JSON
    # (script defaults for --image1/--image2 are ignored).
    # Look for config next to this script first, then in current working directory.
    script_dir = Path(__file__).resolve().parent
    config_path = args.config
    if config_path is None:
        for name in ("pair_config.json", "pair_config.example.json"):
            for base in (script_dir, Path.cwd()):
                candidate = base / name
                if candidate.exists():
                    config_path = str(candidate)
                    print(f"Using config: {config_path}")
                    break
            if config_path is not None:
                break

    if config_path:
        if not Path(config_path).exists():
            raise SystemExit(f"Config file not found: {config_path}")
        img1_preload, img2_preload, spot, image1_path, image2_path = load_sam3_config(config_path)
    elif args.spot:
        parts = [int(p.strip()) for p in args.spot.split(",")]
        if len(parts) == 2:
            spot = (parts[0], parts[1])
        elif len(parts) == 4:
            spot = (parts[0], parts[1], parts[2], parts[3])
        else:
            raise SystemExit(f"--spot must be 'x,y' or 'x1,y1,x2,y2', got: {args.spot}")

    run_demo(
        image1_path=image1_path,
        image2_path=image2_path,
        model_name=args.model,
        device=args.device,
        output_path=args.output,
        spot=spot,
        img1=img1_preload,
        img2=img2_preload,
    )


if __name__ == "__main__":
    main()
