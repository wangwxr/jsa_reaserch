"""Pure, training-free L3 affinity refinement operations."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def spatial_normalize(attention: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize each BxCxHxW map to unit mass over its spatial dimensions."""
    if attention.ndim != 4:
        raise ValueError(f"Expected BxCxHxW attention, got {tuple(attention.shape)}")
    denominator = attention.sum(dim=(-2, -1), keepdim=True)
    return attention / denominator.clamp_min(eps)


def affinity_from_seed(
    seed: torch.Tensor,
    feature: torch.Tensor,
    tau_aff: float,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a seed-weighted feature prototype and its cosine-affinity map.

    Args:
        seed: Spatial probability map with shape [B, 1, H, W].
        feature: Projected L3 feature with shape [B, C, H, W].
        tau_aff: Softmax temperature used only at evaluation.

    Returns:
        affinity: [B, 1, H, W], spatially normalized by softmax.
        prototype: [B, C], L2-normalized along channels.
    """
    if tau_aff <= 0:
        raise ValueError(f"tau_aff must be positive, got {tau_aff}")
    if seed.ndim != 4 or feature.ndim != 4:
        raise ValueError("seed and feature must both be four-dimensional")
    if seed.shape[0] != feature.shape[0] or seed.shape[-2:] != feature.shape[-2:]:
        raise ValueError(
            f"Seed/feature mismatch: seed={tuple(seed.shape)}, "
            f"feature={tuple(feature.shape)}"
        )
    if seed.shape[1] != 1:
        raise ValueError(f"Expected one seed channel, got {seed.shape[1]}")

    weights = spatial_normalize(seed, eps=eps).flatten(start_dim=2).transpose(1, 2)
    tokens = feature.flatten(start_dim=2).transpose(1, 2)
    tokens = F.normalize(tokens, dim=-1)

    prototype = (weights * tokens).sum(dim=1)
    prototype = F.normalize(prototype, dim=-1)
    similarity = torch.einsum("bnc,bc->bn", tokens, prototype)
    affinity = F.softmax(similarity / tau_aff, dim=-1)
    height, width = feature.shape[-2:]
    return affinity.reshape(feature.shape[0], 1, height, width), prototype


def build_refinement_maps(
    aud_l4: torch.Tensor,
    l3_native: torch.Tensor,
    tau_aff: float,
    alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute native and pooled-L3 zero-shot refinements from formal AUD_L4."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if aud_l4.shape[-2:] != (7, 7):
        raise ValueError(f"Formal AUD seed must be 7x7, got {aud_l4.shape[-2:]}")

    aud_l4 = spatial_normalize(aud_l4)
    a4_up = F.interpolate(
        aud_l4, size=l3_native.shape[-2:], mode="bilinear", align_corners=False
    )
    a4_up = spatial_normalize(a4_up)

    native_affinity, _ = affinity_from_seed(a4_up, l3_native, tau_aff)
    native_refined = spatial_normalize(
        alpha * a4_up + (1.0 - alpha) * native_affinity
    )

    l3_pooled = F.adaptive_avg_pool2d(l3_native, (7, 7))
    pooled_affinity, _ = affinity_from_seed(aud_l4, l3_pooled, tau_aff)
    pooled_refined = spatial_normalize(
        alpha * aud_l4 + (1.0 - alpha) * pooled_affinity
    )
    return a4_up, native_affinity, native_refined, pooled_affinity, pooled_refined
