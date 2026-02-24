"""
DINOv2 Feature Extractor — OOP wrapper for spot-guided reidentification.

Provides:
- Patch-level feature extraction
- Spot (region) feature aggregation
- Cosine similarity for matching across images

Uses PyTorch Hub; requires: torch, torchvision
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F


# ImageNet normalization (DINOv2 expects this)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DINOv2FeatureExtractor:
    """
    Object-oriented DINOv2 feature extractor for spot-guided reidentification.

    Loads pretrained DINOv2 via PyTorch Hub. Extracts patch-level and region-level
    features for comparing objects across different views/images.
    """

    # Model variants: (hub_name, patch_size, embed_dim)
    MODEL_CONFIGS = {
        "vits14": ("dinov2_vits14", 14, 384),
        "vitb14": ("dinov2_vitb14", 14, 768),
        "vitl14": ("dinov2_vitl14", 14, 1024),
        "vitg14": ("dinov2_vitg14", 14, 1536),
        "vits14_reg": ("dinov2_vits14_reg", 14, 384),
        "vitb14_reg": ("dinov2_vitb14_reg", 14, 768),
        "vitl14_reg": ("dinov2_vitl14_reg", 14, 1024),
        "vitg14_reg": ("dinov2_vitg14_reg", 14, 1536),
    }

    def __init__(
        self,
        model_name: str = "vits14",
        device: Optional[str] = None,
        hub_repo: str = "facebookresearch/dinov2",
    ):
        """
        Initialize the DINOv2 feature extractor.

        Args:
            model_name: One of "vits14", "vitb14", "vitl14", "vitg14", or *_reg variants.
            device: "cuda", "cpu", or None (auto-detect).
            hub_repo: PyTorch Hub repo (default: facebookresearch/dinov2).
        """
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Choices: {list(self.MODEL_CONFIGS.keys())}"
            )
        self.model_name = model_name
        self.hub_repo = hub_repo
        hub_name, self.patch_size, self.embed_dim = self.MODEL_CONFIGS[model_name]

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load(
            hub_repo,
            hub_name,
            pretrained=True,
            trust_repo=True,
        )
        self.model.to(self.device)
        self.model.eval()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ImageNet normalization to a batch of images [B, 3, H, W]."""
        mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(
            1, 3, 1, 1
        )
        std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(
            1, 3, 1, 1
        )
        return (x - mean) / std

    def _get_patch_features(self, images: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Extract patch-level features from images.

        Args:
            images: [B, 3, H, W], float in [0, 1], RGB.

        Returns:
            patch_features: [B, num_patches, embed_dim]
            h_patches: number of patches in height
            w_patches: number of patches in width
        """
        x = self._normalize(images)
        x = x.to(self.device)

        with torch.no_grad():
            out = self.model.forward_features(x)

        # out is dict: x_norm_clstoken, x_norm_regtokens, x_norm_patchtokens
        patch_tokens = out["x_norm_patchtokens"]  # [B, N, D]
        _, _, h, w = x.shape
        h_p = h // self.patch_size
        w_p = w // self.patch_size
        return patch_tokens, h_p, w_p

    def get_patch_features(
        self,
        images: torch.Tensor,
    ) -> Tuple[torch.Tensor, int, int]:
        """
        Extract patch-level features.

        Args:
            images: [B, 3, H, W], float in [0, 1], RGB.

        Returns:
            patch_features: [B, num_patches, embed_dim]
            h_patches: patch grid height
            w_patches: patch grid width
        """
        return self._get_patch_features(images)

    def get_image_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract image-level (CLS token) features.

        Args:
            images: [B, 3, H, W], float in [0, 1].

        Returns:
            features: [B, embed_dim]
        """
        x = self._normalize(images)
        x = x.to(self.device)

        with torch.no_grad():
            out = self.model.forward_features(x)

        return out["x_norm_clstoken"]

    def get_spot_features(
        self,
        images: torch.Tensor,
        spot: Union[Tuple[int, int], Tuple[int, int, int, int]],
        aggregate: str = "mean",
    ) -> torch.Tensor:
        """
        Extract features for a spot (point or box) in each image.

        Args:
            images: [B, 3, H, W], float in [0, 1].
            spot: Either (x, y) for a single patch, or (x1, y1, x2, y2) for a box
                  in pixel coordinates.
            aggregate: "mean" or "max" to aggregate patches in the region.

        Returns:
            spot_features: [B, embed_dim]
        """
        patch_feats, h_p, w_p = self._get_patch_features(images)

        if len(spot) == 2:
            x, y = spot
            # Map pixel to patch index
            px = min(x // self.patch_size, w_p - 1)
            py = min(y // self.patch_size, h_p - 1)
            idx = py * w_p + px
            return patch_feats[:, idx, :]  # [B, D]

        x1, y1, x2, y2 = spot
        px1 = max(0, x1 // self.patch_size)
        py1 = max(0, y1 // self.patch_size)
        px2 = min(w_p, (x2 + self.patch_size - 1) // self.patch_size)
        py2 = min(h_p, (y2 + self.patch_size - 1) // self.patch_size)

        feats_list = []
        for py in range(py1, py2):
            for px in range(px1, px2):
                idx = py * w_p + px
                feats_list.append(patch_feats[:, idx, :])

        if not feats_list:
            # Fallback to center patch
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            return self.get_spot_features(images, (cx, cy), aggregate)

        region = torch.stack(feats_list, dim=1)  # [B, num_patches_in_region, D]
        if aggregate == "mean":
            return region.mean(dim=1)
        if aggregate == "max":
            return region.max(dim=1).values
        raise ValueError(f"aggregate must be 'mean' or 'max', got {aggregate}")

    @staticmethod
    def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between feature vectors.

        Args:
            a: [..., D]
            b: [..., D] or [D] for single vector.

        Returns:
            similarity: [...] in [-1, 1]
        """
        a_norm = F.normalize(a, p=2, dim=-1)
        b_norm = F.normalize(b, p=2, dim=-1)
        return (a_norm * b_norm).sum(dim=-1)

    def find_best_matching_patch(
        self,
        query_features: torch.Tensor,
        target_patch_features: torch.Tensor,
        h_patches: int,
        w_patches: int,
    ) -> Tuple[int, int, torch.Tensor]:
        """
        Find the patch in target that best matches the query features.

        Args:
            query_features: [D] or [1, D]
            target_patch_features: [num_patches, D]
            h_patches: patch grid height
            w_patches: patch grid width

        Returns:
            best_py: row index of best patch
            best_px: col index of best patch
            similarities: [num_patches] similarity scores
        """
        if query_features.dim() == 1:
            query_features = query_features.unsqueeze(0)
        sim = self.cosine_similarity(
            query_features,
            target_patch_features.unsqueeze(0),
        ).squeeze(0)
        best_idx = sim.argmax().item()
        best_py = best_idx // w_patches
        best_px = best_idx % w_patches
        return best_py, best_px, sim

    def __repr__(self) -> str:
        return f"DINOv2FeatureExtractor(model={self.model_name}, device={self.device})"
