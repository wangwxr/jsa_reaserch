"""Experiment D: jointly fine-tune L3+L4 JSA and top-down L3 refinement."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopDownL3RefinementHead(nn.Module):
    """The unchanged zero-initialized adapter from frozen top-down stage-2."""

    def __init__(self, channels: int = 512, hidden_channels: int = 256):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=3, padding=1),
        )
        last_conv = self.adapter[-1]
        nn.init.zeros_(last_conv.weight)
        nn.init.zeros_(last_conv.bias)

    def forward(
        self, f3_native: torch.Tensor, f4_projected: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if f3_native.shape[1:] != (512, 14, 14):
            raise ValueError(f"Expected F3 [B,512,14,14], got {f3_native.shape}")
        if f4_projected.shape[1:] != (512, 7, 7):
            raise ValueError(f"Expected F4 [B,512,7,7], got {f4_projected.shape}")
        f4_up = F.interpolate(
            f4_projected,
            size=f3_native.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        delta_f3 = self.adapter(f3_native)
        return f4_up + delta_f3, f4_up, delta_f3


class _ProjectedFeatureHooks:
    """Capture the checkpoint's proj3/proj4 outputs from the formal forward."""

    def __init__(self, base_model: nn.Module):
        self.outputs: dict[str, torch.Tensor] = {}
        self.handles = [
            base_model.imgnet.proj3.register_forward_hook(self._capture("f3")),
            base_model.imgnet.proj4.register_forward_hook(self._capture("f4")),
        ]

    def _capture(self, name: str):
        def hook(_module: nn.Module, _inputs: Any, output: torch.Tensor) -> None:
            self.outputs[name] = output

        return hook

    def pop(self) -> tuple[torch.Tensor, torch.Tensor]:
        if set(self.outputs) != {"f3", "f4"}:
            raise RuntimeError(
                f"Expected proj3/proj4 hook outputs, got {sorted(self.outputs)}"
            )
        return self.outputs.pop("f3"), self.outputs.pop("f4")

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


class JointL3L4TopDownModel(nn.Module):
    """Original L3+L4 objectives plus the unchanged top-down refinement loss."""

    def __init__(
        self,
        base_model: nn.Module,
        original_trainability: dict[str, bool],
    ):
        super().__init__()
        self.base_model = base_model
        self.refinement_head = TopDownL3RefinementHead()
        self.feature_hooks = _ProjectedFeatureHooks(base_model)
        self.mse = nn.MSELoss()

        current_names = {name for name, _ in self.base_model.named_parameters()}
        if current_names != set(original_trainability):
            raise RuntimeError("Original trainability template does not match base model")
        for name, parameter in self.base_model.named_parameters():
            # Restore each parameter's pristine L3+L4 state; do not blanket-unfreeze.
            parameter.requires_grad = original_trainability[name]
        self.original_trainability = dict(original_trainability)

    @staticmethod
    def _to_tokens(feature: torch.Tensor) -> torch.Tensor:
        return feature.flatten(start_dim=2).transpose(1, 2)

    @staticmethod
    def _target_map(attention: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return attention[:, 0].reshape(attention.shape[0], 1, height, width)

    @staticmethod
    def sum_pool_2x2(probability: torch.Tensor) -> torch.Tensor:
        if probability.shape[-2:] != (14, 14):
            raise ValueError(
                f"sum_pool_2x2 requires 14x14 input, got {probability.shape[-2:]}"
            )
        pooled = F.avg_pool2d(probability, kernel_size=2, stride=2) * 4.0
        return pooled / pooled.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)

    def _base_losses(
        self,
        image_levels: tuple[torch.Tensor, ...],
        audio_tokens: torch.Tensor,
        encoded: dict[str, Any],
        training_attentions: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        fused_img_slots = self.base_model.slot_attn.slot_fusion(
            encoded["visual_slots"]
        )
        aud_slots = encoded["audio_slots"]
        img_recon = self.base_model.img_decoder(fused_img_slots)
        aud_recon = self.base_model.aud_decoder(aud_slots).flatten(start_dim=2)

        recon_loss = self.base_model.MSELoss(
            img_recon, image_levels[-1].detach()
        )
        recon_loss = recon_loss + self.base_model.MSELoss(
            aud_recon, audio_tokens.detach()
        )
        att_loss = self.base_model.MSELoss(
            training_attentions["audq_imgk_attn"][:, 0, :],
            training_attentions["imgq_imgk_attn"][:, 0, :].detach(),
        )
        att_loss = att_loss + self.base_model.MSELoss(
            training_attentions["imgq_audk_attn"][:, 0, :],
            training_attentions["audq_audk_attn"][:, 0, :].detach(),
        )
        normalized_img_slots = F.normalize(fused_img_slots, dim=2)
        normalized_aud_slots = F.normalize(aud_slots, dim=2)
        info_loss, div_loss = self.base_model.calculate(
            normalized_img_slots, normalized_aud_slots
        )
        return {
            "info_loss": info_loss,
            "recon_loss": recon_loss,
            "div_loss": div_loss,
            "att_loss": att_loss,
        }

    def forward(self, image: torch.Tensor, audio: torch.Tensor) -> dict[str, torch.Tensor]:
        # One shared forward preserves the original masking/sample semantics.
        image_levels = self.base_model.imgnet(image)
        f3_native, f4_projected = self.feature_hooks.pop()
        audio_tokens = self.base_model._audio_tokens(self.base_model.audnet(audio))
        encoded = self.base_model.slot_attn._encode(image_levels, audio_tokens)

        losses = {}
        if self.training:
            training_attentions = self.base_model.slot_attn._l4_attentions(
                encoded, scale_multiplier=1.0
            )
            losses = self._base_losses(
                image_levels, audio_tokens, encoded, training_attentions
            )

        # These are the exact formal L4 localization maps, used as detached
        # targets only inside refinement_losses().
        coarse = self.base_model.slot_attn._l4_attentions(
            encoded, scale_multiplier=self.base_model.infer_sharpening
        )
        aud_l4 = self._target_map(coarse["audq_imgk_attn"], 7, 7)
        img_l4 = self._target_map(coarse["imgq_imgk_attn"], 7, 7)

        f34, f4_up, delta_f3 = self.refinement_head(f3_native, f4_projected)
        l4_branch = self.base_model.slot_attn.visual_branches[-1]
        f34_tokens = self._to_tokens(f34)
        fine_keys = l4_branch.img_to_k(l4_branch.img_norm_input(f34_tokens))
        attention = self.base_model.slot_attn._attention
        scale_multiplier = self.base_model.infer_sharpening
        aud_fine_all = attention(
            encoded["audio_query"], fine_keys, scale_multiplier
        )
        img_fine_all = attention(
            encoded["visual_queries"][-1], fine_keys, scale_multiplier
        )
        aud_fine = self._target_map(aud_fine_all, 14, 14)
        img_fine = self._target_map(img_fine_all, 14, 14)

        f4_tokens = self._to_tokens(f4_projected)
        output = {
            **losses,
            "AUD_L4": aud_l4,
            "IMG_L4": img_l4,
            "AUD_FINE": aud_fine,
            "IMG_FINE": img_fine,
            "IQR_FINE": 0.6 * aud_fine + 0.4 * img_fine,
            "F34": f34,
            "F4_UP": f4_up,
            "DELTA_F3": delta_f3,
            "F3_NATIVE": f3_native,
            "F4_PROJECTED": f4_projected,
            "source_token_max_abs": (f4_tokens - image_levels[-1]).abs().max(),
        }
        output.update(self.refinement_losses(output))
        return output

    def refinement_losses(
        self,
        output: dict[str, torch.Tensor],
        lambda_match: float = 1.0,
        lambda_coarse: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        loss_fine_match = self.mse(
            output["AUD_FINE"], output["IMG_FINE"].detach()
        )
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
            "refine_loss": lambda_match * loss_fine_match
            + lambda_coarse * (loss_coarse_aud + loss_coarse_img),
        }

    def close(self) -> None:
        self.feature_hooks.close()
