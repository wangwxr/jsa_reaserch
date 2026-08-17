"""Frozen L3+L4 base with a trainable L4-anchored native-L3 adapter."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopDownL3RefinementHead(nn.Module):
    """Predict a native-L3 residual, initialized to exactly zero."""

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
    """Observe the exact proj3/proj4 outputs used by the existing backbone."""

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
        f3_native = self.outputs.pop("f3")
        f4_projected = self.outputs.pop("f4")
        return f3_native, f4_projected

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


class FrozenL3L4TopDownModel(nn.Module):
    """Keep the complete semantic base frozen and train only the refinement head."""

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        self.refinement_head = TopDownL3RefinementHead()
        self.feature_hooks = _ProjectedFeatureHooks(base_model)
        self.mse = nn.MSELoss()

        for parameter in self.base_model.parameters():
            parameter.requires_grad = False
        self.base_model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # super().train() also touches children, so restore the hard invariant.
        self.base_model.eval()
        self.refinement_head.train(mode)
        return self

    @staticmethod
    def _to_tokens(feature: torch.Tensor) -> torch.Tensor:
        return feature.flatten(start_dim=2).transpose(1, 2)

    @staticmethod
    def _target_map(attention: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch_size = attention.shape[0]
        return attention[:, 0].reshape(batch_size, 1, height, width)

    @staticmethod
    def sum_pool_2x2(probability: torch.Tensor) -> torch.Tensor:
        if probability.shape[-2:] != (14, 14):
            raise ValueError(
                f"sum_pool_2x2 requires 14x14 input, got {probability.shape[-2:]}"
            )
        pooled = F.avg_pool2d(probability, kernel_size=2, stride=2) * 4.0
        return pooled / pooled.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)

    def _extract_frozen_base(
        self, image: torch.Tensor, audio: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        self.base_model.eval()
        with torch.no_grad():
            image_levels = self.base_model.imgnet(image)
            f3_native, f4_projected = self.feature_hooks.pop()
            audio_tokens = self.base_model._audio_tokens(
                self.base_model.audnet(audio)
            )
            encoded = self.base_model.slot_attn._encode(image_levels, audio_tokens)
            coarse = self.base_model.slot_attn._l4_attentions(
                encoded,
                scale_multiplier=self.base_model.infer_sharpening,
            )

            f4_tokens = self._to_tokens(f4_projected)
            source_token_max_abs = (f4_tokens - image_levels[-1]).abs().max()

        return {
            "f3_native": f3_native.detach(),
            "f4_projected": f4_projected.detach(),
            "audio_query": encoded["audio_query"].detach(),
            "image_query_l4": encoded["visual_queries"][-1].detach(),
            "aud_l4": self._target_map(
                coarse["audq_imgk_attn"], 7, 7
            ).detach(),
            "img_l4": self._target_map(
                coarse["imgq_imgk_attn"], 7, 7
            ).detach(),
            "source_token_max_abs": source_token_max_abs.detach(),
        }

    def forward(self, image: torch.Tensor, audio: torch.Tensor) -> dict[str, torch.Tensor]:
        frozen = self._extract_frozen_base(image, audio)
        f34, f4_up, delta_f3 = self.refinement_head(
            frozen["f3_native"], frozen["f4_projected"]
        )

        # Reuse the exact trained L4 key coordinate space. Parameters are frozen,
        # while autograd still propagates through these operations to f34/head.
        l4_branch = self.base_model.slot_attn.visual_branches[-1]
        f34_tokens = self._to_tokens(f34)
        fine_keys = l4_branch.img_to_k(l4_branch.img_norm_input(f34_tokens))
        attention = self.base_model.slot_attn._attention
        scale_multiplier = self.base_model.infer_sharpening
        aud_fine_all = attention(
            frozen["audio_query"], fine_keys, scale_multiplier
        )
        img_fine_all = attention(
            frozen["image_query_l4"], fine_keys, scale_multiplier
        )
        aud_fine = self._target_map(aud_fine_all, 14, 14)
        img_fine = self._target_map(img_fine_all, 14, 14)

        return {
            "AUD_L4": frozen["aud_l4"],
            "IMG_L4": frozen["img_l4"],
            "AUD_FINE": aud_fine,
            "IMG_FINE": img_fine,
            "IQR_FINE": 0.6 * aud_fine + 0.4 * img_fine,
            "F34": f34,
            "F4_UP": f4_up,
            "DELTA_F3": delta_f3,
            "F3_NATIVE": frozen["f3_native"],
            "F4_PROJECTED": frozen["f4_projected"],
            "source_token_max_abs": frozen["source_token_max_abs"],
        }

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
        loss_refine = lambda_match * loss_fine_match + lambda_coarse * (
            loss_coarse_aud + loss_coarse_img
        )
        return {
            "loss_fine_match": loss_fine_match,
            "loss_coarse_aud": loss_coarse_aud,
            "loss_coarse_img": loss_coarse_img,
            "loss_refine": loss_refine,
        }

    def close(self) -> None:
        self.feature_hooks.close()
