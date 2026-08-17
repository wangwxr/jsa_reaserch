"""Experiment E: remove only fine AUD-to-IMG pixel-level imitation."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


EXPERIMENT_DIR = Path(__file__).resolve().parent
EXPERIMENT_D_DIR = EXPERIMENT_DIR.parent
sys.path.insert(0, str(EXPERIMENT_D_DIR))

from model import JointL3L4TopDownModel  # noqa: E402


class DecoupledFineSpatialModel(JointL3L4TopDownModel):
    """Experiment D architecture with independent coarse anchors only."""

    def refinement_losses(
        self,
        output: dict[str, torch.Tensor],
        lambda_match: float = 1.0,
        lambda_coarse: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        del lambda_match
        # Kept as an explicit logged scalar for D/E table compatibility. It is
        # not connected to either fine map and contributes no gradient.
        loss_fine_match = output["AUD_FINE"].new_zeros(())
        loss_coarse_aud = self.mse(
            self.sum_pool_2x2(output["AUD_FINE"]),
            output["AUD_L4"].detach(),
        )
        loss_coarse_img = self.mse(
            self.sum_pool_2x2(output["IMG_FINE"]),
            output["IMG_L4"].detach(),
        )
        return {
            "loss_fine_match": loss_fine_match,
            "loss_coarse_aud": loss_coarse_aud,
            "loss_coarse_img": loss_coarse_img,
            "refine_loss": lambda_coarse
            * (loss_coarse_aud + loss_coarse_img),
        }
