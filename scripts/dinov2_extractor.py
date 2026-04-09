"""
DINOv2 Feature Extractor — OOP wrapper for spot-guided instance re-identification.

What it does:
  - Loads pre-trained DINOv2 Vision Transformer models via PyTorch Hub
  - Extracts visual embeddings (features) from image regions or whole images
  - Provides cosine similarity computation for comparing objects

Key concepts:
  - Embedding: A numerical vector (e.g., 1536 values for ViT-g14) representing visual features
  - Patch: A small grid region (e.g., 14x14 pixels for patch_size=14) that ViT processes independently
  - Region (spot): An arbitrary rectangle in the image; we aggregate patches overlapping this region
  - Cosine similarity: How aligned two embeddings are (1=identical, 0=orthogonal, -1=opposite)

Why DINOv2?
  - Self-supervised pre-training learns rich, generalizable visual features
  - No task-specific fine-tuning needed for many re-ID tasks (yet)
  - Supports multiple model sizes (ViT-s: fast; ViT-g: high-quality)

Models available:
  - vits14, vitb14, vitl14, vitg14 (patch_size=14, increasing capacity)
  - _reg variants: Register tokens for better patch aggregation
  - Each loaded from facebookresearch/dinov2 PyTorch Hub

Usage example:
  extractor = DINOv2FeatureExtractor(model_name="vits14", device="cuda")
  image_tensor = torch.randn(1, 3, 224, 224)  # [batch, channels, height, width]
  region_features = extractor.get_spot_features(image_tensor, spot=(0, 0, 100, 100))
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F


# ============================================================================
# CONSTANTS: ImageNet Normalization
# ============================================================================
# DINOv2 expects images normalized with ImageNet statistics.
# These are standard pre-training values from ImageNet dataset.
# Formula: x_normalized = (x - mean) / std (applied per channel)
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # [R, G, B] means
IMAGENET_STD = (0.229, 0.224, 0.225)   # [R, G, B] standard deviations


class DINOv2FeatureExtractor:
    """
    Object-oriented DINOv2 feature extractor for instance re-identification.

    Overview:
      - Loads pre-trained DINOv2 models from PyTorch Hub
      - Extracts patch-level embeddings from images
      - Aggregates patches within regions (spots) for object-level features
      - Provides cosine similarity for comparing objects

    Architecture:
      - Vision Transformer (ViT): Divides image into non-overlapping patches
      - Each patch processed independently; outputs embed_dim-dimensional vector
      - CLS token: Image-level embedding (global context)
      - Patch tokens: Patch-level embeddings (local detail)
    
    Model sizes:
      - vits14 (384 dim): Smallest, fastest (~100ms per image on GPU)
      - vitb14 (768 dim): Medium (~150ms)
      - vitl14 (1024 dim): Large (~250ms)
      - vitg14 (1536 dim): Largest, best quality (~400ms)

    Typical usage in re-ID pipeline:
      1. Load extractor: DINOv2FeatureExtractor("vitl14", device="cuda")
      2. Forward image through model: image_tensor [1, 3, 224, 224]
      3. Extract spot features: get_spot_features(image_tensor, spot=(x1,y1,x2,y2))
      4. Compare with similarity: cosine_similarity(feat1, feat2)
    """

    # ========================================================================
    # MODEL CONFIGURATIONS
    # ========================================================================
    # Defines available DINOv2 models, their hub names, patch sizes, and dimensions
    MODEL_CONFIGS = {
        "vits14": ("dinov2_vits14", 14, 384),              # Small, 384-dim embeddings
        "vitb14": ("dinov2_vitb14", 14, 768),             # Base, 768-dim embeddings
        "vitl14": ("dinov2_vitl14", 14, 1024),            # Large, 1024-dim embeddings
        "vitg14": ("dinov2_vitg14", 14, 1536),            # Giant, 1536-dim embeddings
        "vits14_reg": ("dinov2_vits14_reg", 14, 384),     # Small + register tokens
        "vitb14_reg": ("dinov2_vitb14_reg", 14, 768),     # Base + register tokens
        "vitl14_reg": ("dinov2_vitl14_reg", 14, 1024),    # Large + register tokens
        "vitg14_reg": ("dinov2_vitg14_reg", 14, 1536),    # Giant + register tokens
    }
    # Register tokens improve spatial coherence when aggregating patches.
    # Not needed for basic region aggregation, but can help in some cases.

    def __init__(
        self,
        model_name: str = "vits14",
        device: Optional[str] = None,
        hub_repo: str = "facebookresearch/dinov2",
    ):
        """
        Initialize the DINOv2 feature extractor.

        Args:
            model_name: Model choice (default "vits14")
              Options: vits14, vitb14, vitl14, vitg14, or *_reg variants
              Tradeoff: larger=better features, slower inference
            
            device: Compute device; None auto-detects CUDA availability
              Use "cuda" for GPU (10x-50x faster) or "cpu" for CPU
            
            hub_repo: PyTorch Hub repository path
              Default: facebookresearch/dinov2 (official Facebook Research repo)
              Can be changed for custom/local models
        
        Raises:
            ValueError: If model_name not in MODEL_CONFIGS
        """
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Choices: {list(self.MODEL_CONFIGS.keys())}"
            )
        self.model_name = model_name
        self.hub_repo = hub_repo
        hub_name, self.patch_size, self.embed_dim = self.MODEL_CONFIGS[model_name]

        # Auto-detect GPU or use CPU
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pre-trained model from PyTorch Hub
        # First run downloads ~1.5GB for vit-g14; subsequent runs load from cache
        self.model = torch.hub.load(
            hub_repo,
            hub_name,
            pretrained=True,
            trust_repo=True,
        )
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode (no batch norm updates, no dropout)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply ImageNet normalization to image tensor.
        
        Why normalize?
          - DINOv2 was pre-trained on normalized ImageNet images
          - Normalization centers and scales image values for stable network input
          - Without normalization, features can be poor or unstable
        
        Formula (per-channel):
          x_normalized = (x - mean) / std
        
        Args:
            x: Batch of images [B, 3, H, W], float in [0, 1], RGB order
        
        Returns:
            Normalized tensor, same shape, optimized for DINOv2 model input
        """
        mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(
            1, 3, 1, 1  # Reshape for broadcasting: mean/std applied per channel
        )
        std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(
            1, 3, 1, 1
        )
        return (x - mean) / std

    def _get_patch_features(self, images: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Extract patch-level features from images using DINOv2.
        
        How ViT processes images:
          1. Input image [B, 3, 224, 224]
          2. Divide into non-overlapping 14×14 pixel patches
          3. Each patch embedded to 1536-D vector (for ViT-g)
          4. Output: [B, num_patches, 1536] batch of patch embeddings
        
        Patch grid:
          - 224 / 14 = 16 patches per side
          - 16 × 16 = 256 patches total per image
          - num_patches = 256 for any 224×224 image
        
        Args:
            images: [B, 3, H, W], float in [0, 1], RGB
        
        Returns:
            patch_tokens: [B, num_patches, embed_dim] (e.g., [1, 256, 1536])
            h_patches: Patch grid height (e.g., 16 for 224×224)
            w_patches: Patch grid width (e.g., 16 for 224×224)
        
        Why return patch grid dimensions?
          - Needed to map pixel coordinates (x, y) to patch indices
          - Enables region-of-interest aggregation
        """
        x = self._normalize(images)  # Normalize to match pre-training
        x = x.to(self.device)

        with torch.no_grad():  # No gradients; we're extracting features, not training
            out = self.model.forward_features(x)

        # Out is a dict. We extract patch tokens (skip CLS and register tokens).
        # Keys typically: "x_norm_clstoken", "x_norm_regtokens", "x_norm_patchtokens"
        patch_tokens = out["x_norm_patchtokens"]  # [B, N, D] where N = num_patches
        _, _, h, w = x.shape  # Original image height/width
        h_p = h // self.patch_size  # Patch grid height (e.g., 224 / 14 = 16)
        w_p = w // self.patch_size  # Patch grid width (e.g., 224 / 14 = 16)
        return patch_tokens, h_p, w_p

    def get_patch_features(
        self,
        images: torch.Tensor,
    ) -> Tuple[torch.Tensor, int, int]:
        """
        Public wrapper for extracting patch-level features.
        
        Typically used internally; advanced users may call directly for custom processing.
        
        Args:
            images: [B, 3, H, W], float in [0, 1], RGB
        
        Returns:
            patch_features: [B, num_patches, embed_dim]
            h_patches: Patch grid height
            w_patches: Patch grid width
        """
        return self._get_patch_features(images)

    def get_image_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract image-level (CLS token) features for whole-image embeddings.
        
        CLS token:
          - A special learnable token prepended to the image patches
          - Updated via self-attention to aggregate global image information
          - Represents the "summary" of the entire image
        
        When to use:
          - Image classification or global-level comparisons
          - Not typically used for instance re-ID (use get_spot_features instead)
          - Useful as a coarse-level check before fine-grained matching
        
        Args:
            images: [B, 3, H, W], float in [0, 1]
        
        Returns:
            features: [B, embed_dim] (e.g., [1, 1536] per image)
        """
        x = self._normalize(images)
        x = x.to(self.device)

        with torch.no_grad():
            out = self.model.forward_features(x)

        return out["x_norm_clstoken"]  # [B, embed_dim]

    def get_spot_features(
        self,
        images: torch.Tensor,
        spot: Union[Tuple[int, int], Tuple[int, int, int, int]],
        aggregate: str = "mean",
    ) -> torch.Tensor:
        """
        Extract features for a region-of-interest (spot) in each image.
        
        How it works:
          1. Extract all patch embeddings from image
          2. Identify which patches overlap the spot region
          3. Aggregate (mean or max) those patch embeddings
          4. Return single embedding representing the region
        
        Spot types:
          - Point: (x, y) → Single patch at that location
          - Box: (x1, y1, x2, y2) → Multiple patches in bounding box
        
        Aggregation strategies:
          - "mean": Average patch embeddings (stable, smooth)
          - "max": Maximum per feature dimension (picks strongest signal)
        
        Why region embedding?
          - Extracts features for a specific object within the image
          - Enables object-level comparison (core of re-ID)
          - Handles arbitrary box coordinates (e.g., from SAM3 detection)
        
        Args:
            images: [B, 3, H, W], float in [0, 1]
            spot: Either (x, y) for single patch, or (x1, y1, x2, y2) for box
                  Coordinates in 224×224 space (matching image size)
            aggregate: "mean" or "max" pooling over patches in region
        
        Returns:
            spot_features: [B, embed_dim] (e.g., [1, 1536])
        
        Example:
            # Extract embedding for yellow ball in bounding box [50, 50, 174, 174]
            feat = extractor.get_spot_features(image_tensor, spot=(50, 50, 174, 174), aggregate="mean")
        """
        patch_feats, h_p, w_p = self._get_patch_features(images)

        if len(spot) == 2:
            # Single point: map pixel (x,y) to patch index
            x, y = spot
            # Which patch contains this pixel?
            px = min(x // self.patch_size, w_p - 1)  # Patch column index
            py = min(y // self.patch_size, h_p - 1)  # Patch row index
            # Convert 2D patch coordinates to linear index
            idx = py * w_p + px
            return patch_feats[:, idx, :]  # [B, D]

        # Bounding box: identify all patches overlapping the box
        x1, y1, x2, y2 = spot
        # Map box coordinates to patch grid
        px1 = max(0, x1 // self.patch_size)
        py1 = max(0, y1 // self.patch_size)
        px2 = min(w_p, (x2 + self.patch_size - 1) // self.patch_size)
        py2 = min(h_p, (y2 + self.patch_size - 1) // self.patch_size)

        # Collect all patch embeddings within the box
        feats_list = []
        for py in range(py1, py2):
            for px in range(px1, px2):
                idx = py * w_p + px
                feats_list.append(patch_feats[:, idx, :])

        if not feats_list:
            # Fallback: if box is too small to contain any patch, use center point
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            return self.get_spot_features(images, (cx, cy), aggregate)

        # Stack patches from the region
        region = torch.stack(feats_list, dim=1)  # [B, num_patches_in_region, D]
        
        # Aggregate: pool over regions
        if aggregate == "mean":
            return region.mean(dim=1)  # [B, D]
        if aggregate == "max":
            return region.max(dim=1).values  # [B, D]
        raise ValueError(f"aggregate must be 'mean' or 'max', got {aggregate}")

    @staticmethod
    def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between feature vectors.
        
        Cosine similarity measures angular alignment of vectors in embedding space:
          - 1.0 = same direction (identical content)
          - 0.0 = orthogonal (unrelated)
          - -1.0 = opposite direction (very dissimilar)
        
        Formula:
          cos_sim(a, b) = (a · b) / (||a|| × ||b||)
        
        Normalization:
          - Normalizes each vector to unit length
          - Makes score independent of magnitude (only direction matters)
          - Important: two images that look identical but have different brightness
            will still have high cosine similarity (unlike L2 distance)
        
        Shape broadcasting:
          - a: [..., D] or [D]
          - b: [..., D] or [D]
          - output: [...] (all leading dims broadcast to same shape)
        
        Args:
            a: Feature tensor(s), last dimension is embedding
            b: Feature tensor(s), last dimension is embedding
        
        Returns:
            similarity: Values in [- 1, 1] (typically [0, 1] for image embeddings)
        
        Example:
            ref_feat = torch.randn(1, 1536)  # [B, D]
            query_feat = torch.randn(1, 1536)  # [B, D]
            sim = DINOv2FeatureExtractor.cosine_similarity(ref_feat, query_feat)
            # sim shape: [1] (single similarity score)
        """
        a_norm = F.normalize(a, p=2, dim=-1)  # Normalize along embedding dimension
        b_norm = F.normalize(b, p=2, dim=-1)
        return (a_norm * b_norm).sum(dim=-1)  # Dot product of normalized vectors

    def find_best_matching_patch(
        self,
        query_features: torch.Tensor,
        target_patch_features: torch.Tensor,
        h_patches: int,
        w_patches: int,
    ) -> Tuple[int, int, torch.Tensor]:
        """
        Find the best-matching patch in target given query features.
        
        Useful for:
          - Visualizing which part of an image matches a query object
          - Debugging feature extraction quality
          - Advanced matching strategies (e.g., spatial constraints)
        
        Args:
            query_features: [D] or [1, D], query object embedding
            target_patch_features: [num_patches, D], all patches from target image
            h_patches: Patch grid height (e.g., 16)
            w_patches: Patch grid width (e.g., 16)
        
        Returns:
            best_py: Row index of best-matching patch (0 to h_patches-1)
            best_px: Column index of best-matching patch (0 to w_patches-1)
            similarities: [num_patches] similarity scores (highest ≈ best match)
        """
        if query_features.dim() == 1:
            query_features = query_features.unsqueeze(0)  # Batch the query
        
        # Compute similarity with all patches
        sim = self.cosine_similarity(
            query_features,
            target_patch_features.unsqueeze(0),
        ).squeeze(0)  # [num_patches]
        
        # Find patch with highest similarity
        best_idx = sim.argmax().item()
        best_py = best_idx // w_patches  # Patch row
        best_px = best_idx % w_patches   # Patch column
        return best_py, best_px, sim

    def __repr__(self) -> str:
        """String representation: useful for debugging which model is loaded"""
        return f"DINOv2FeatureExtractor(model={self.model_name}, device={self.device})"