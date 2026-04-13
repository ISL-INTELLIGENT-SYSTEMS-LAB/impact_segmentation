"""
Part-based Re-ID embeddings on top of DINOv2 patch features (PCB-style horizontal strips).

Composes :class:`DINOv2FeatureExtractor`: patch grid → split height into ``num_parts``
bands → mean-pool each band → concatenate → L2 normalize (same structure as Catherine's
CLIP PartBasedEmbedding, different backbone).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from dinov2_extractor import DINOv2FeatureExtractor


class DINOv2PartBasedEmbedder:
    """
    Horizontal part pooling over the DINOv2 patch grid, then concatenation + L2 norm.

    Expects square inputs whose side length is divisible by the model patch size (e.g. 224
    and patch 14 → 16×16 patches).
    """

    def __init__(
        self,
        extractor: DINOv2FeatureExtractor,
        num_parts: int = 4,
    ):
        if num_parts < 1:
            raise ValueError("num_parts must be >= 1")
        self.extractor = extractor
        self.num_parts = num_parts

    @property
    def embed_dim(self) -> int:
        return self.extractor.embed_dim * self.num_parts

    def embed_image_tensor(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, 3, H, W], float in [0, 1], RGB.

        Returns:
            [B, num_parts * embed_dim], L2-normalized along dim=-1.
        """
        patch_tokens, h_p, w_p = self.extractor.get_patch_features(images)
        # patch_tokens: [B, h_p * w_p, D]
        b, n, d = patch_tokens.shape
        if n != h_p * w_p:
            raise RuntimeError(
                f"Patch count {n} != h_p*w_p ({h_p}*{w_p})"
            )
        fmap = patch_tokens.view(b, h_p, w_p, d).permute(0, 3, 1, 2)  # [B, D, H, W]

        h = h_p
        part_h = h // self.num_parts
        if part_h == 0:
            raise ValueError(
                f"Patch grid height {h_p} is too small for num_parts={self.num_parts}"
            )

        parts: list[torch.Tensor] = []
        for p in range(self.num_parts):
            h0 = p * part_h
            h1 = h_p if p == self.num_parts - 1 else (p + 1) * part_h
            pooled = fmap[:, :, h0:h1, :].mean(dim=(2, 3))  # [B, D]
            parts.append(pooled)

        emb = torch.cat(parts, dim=1)  # [B, num_parts * D]
        emb = F.normalize(emb, p=2, dim=1)
        return emb

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return self.embed_image_tensor(images)
