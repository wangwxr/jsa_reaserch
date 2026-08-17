"""Frozen L3+L4 teacher with a trainable equivariant native-L3 student."""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopDownL3Adapter(nn.Module):
    """L3 residual adapter, with the last convolution initialized to zero."""

    def __init__(self, channels: int = 512, hidden_channels: int = 256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.layers(feature)


class FineSpatialStudent(nn.Module):
    """A private proj3 copy and the top-down residual adapter."""

    def __init__(self, teacher_proj3: nn.Conv2d):
        super().__init__()
        self.proj3_spatial = copy.deepcopy(teacher_proj3)
        self.adapter = TopDownL3Adapter()
        for parameter in self.parameters():
            parameter.requires_grad = True

    def forward(
        self, layer3_native: torch.Tensor, f4_projected: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        f3_spatial = self.proj3_spatial(layer3_native)
        if f3_spatial.shape[1:] != (512, 14, 14):
            raise ValueError(f"Expected spatial F3 [B,512,14,14], got {f3_spatial.shape}")
        if f4_projected.shape[1:] != (512, 7, 7):
            raise ValueError(f"Expected projected F4 [B,512,7,7], got {f4_projected.shape}")
        f4_up = F.interpolate(
            f4_projected,
            size=f3_spatial.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        delta_f3 = self.adapter(f3_spatial)
        return f4_up + delta_f3, f3_spatial, f4_up, delta_f3


class _TeacherFeatureHooks:
    """Observe native layer3 and projected layer4 in the formal teacher forward."""

    def __init__(self, teacher: nn.Module):
        self.outputs: dict[str, torch.Tensor] = {}
        self.handles = [
            teacher.imgnet.proj3.register_forward_pre_hook(self._capture_layer3),
            teacher.imgnet.proj4.register_forward_hook(self._capture_f4),
        ]

    def _capture_layer3(self, _module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        self.outputs["layer3_native"] = inputs[0]

    def _capture_f4(
        self, _module: nn.Module, _inputs: Any, output: torch.Tensor
    ) -> None:
        self.outputs["f4_projected"] = output

    def pop(self) -> tuple[torch.Tensor, torch.Tensor]:
        expected = {"layer3_native", "f4_projected"}
        if set(self.outputs) != expected:
            raise RuntimeError(
                f"Expected teacher hook outputs {sorted(expected)}, got {sorted(self.outputs)}"
            )
        layer3_native = self.outputs.pop("layer3_native")
        f4_projected = self.outputs.pop("f4_projected")
        return layer3_native, f4_projected

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


class FrozenTeacherEquivariantRefinement(nn.Module):
    """Train only a fine spatial student against a fixed semantic teacher."""

    def __init__(self, teacher: nn.Module):
        super().__init__()
        self.teacher = teacher
        self.student = FineSpatialStudent(teacher.imgnet.proj3)
        self.feature_hooks = _TeacherFeatureHooks(teacher)

        for parameter in self.teacher.parameters():
            parameter.requires_grad = False
        self.teacher.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        self.student.train(mode)
        return self

    @staticmethod
    def _to_tokens(feature: torch.Tensor) -> torch.Tensor:
        return feature.flatten(start_dim=2).transpose(1, 2)

    @staticmethod
    def _target_map(attention: torch.Tensor, size: int) -> torch.Tensor:
        return attention[:, 0].reshape(attention.shape[0], 1, size, size)

    @staticmethod
    def apply_horizontal_flip(
        tensor: torch.Tensor, flip_mask: torch.Tensor
    ) -> torch.Tensor:
        if flip_mask.shape != (tensor.shape[0],):
            raise ValueError(
                f"flip_mask must have shape [{tensor.shape[0]}], got {flip_mask.shape}"
            )
        flipped = torch.flip(tensor, dims=[-1])
        shape = (tensor.shape[0],) + (1,) * (tensor.ndim - 1)
        return torch.where(flip_mask.reshape(shape), flipped, tensor)

    @staticmethod
    def sum_pool_2x2(probability: torch.Tensor) -> torch.Tensor:
        if probability.shape[-2:] != (14, 14):
            raise ValueError(f"Expected 14x14 probability map, got {probability.shape}")
        pooled = F.avg_pool2d(probability, kernel_size=2, stride=2) * 4.0
        return pooled / pooled.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)

    @staticmethod
    def kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Mean KL(P||Q) over a batch of spatial distributions."""
        p_flat = p.float().flatten(start_dim=1).clamp_min(1e-8)
        q_flat = q.float().flatten(start_dim=1).clamp_min(1e-8)
        p_flat = p_flat / p_flat.sum(dim=-1, keepdim=True)
        q_flat = q_flat / q_flat.sum(dim=-1, keepdim=True)
        return (p_flat * (p_flat.log() - q_flat.log())).sum(dim=-1).mean()

    @classmethod
    def symmetric_kl(cls, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return 0.5 * (cls.kl_divergence(p, q) + cls.kl_divergence(q, p))

    def _extract_audio_query(self, audio: torch.Tensor) -> torch.Tensor:
        self.teacher.eval()
        with torch.no_grad():
            audio_tokens = self.teacher._audio_tokens(self.teacher.audnet(audio))
            initial_slots = self.teacher.slot_attn.slots.expand(
                audio_tokens.shape[0], -1, -1
            )
            _, audio_query, _ = self.teacher.slot_attn.audio_branch(
                audio_tokens, initial_slots
            )
        return audio_query.detach()

    def _extract_visual_teacher(
        self, image: torch.Tensor, audio_query: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        self.teacher.eval()
        with torch.no_grad():
            image_levels = self.teacher.imgnet(image)
            layer3_native, f4_projected = self.feature_hooks.pop()
            initial_slots = self.teacher.slot_attn.slots.expand(
                image.shape[0], -1, -1
            )
            l4_branch = self.teacher.slot_attn.visual_branches[-1]
            _, _image_query, image_keys_l4 = l4_branch(
                image_levels[-1], initial_slots
            )
            coarse_all = self.teacher.slot_attn._attention(
                audio_query,
                image_keys_l4,
                self.teacher.infer_sharpening,
            )
            aud_l4 = self._target_map(coarse_all, 7)
            f4_token_error = (
                self._to_tokens(f4_projected) - image_levels[-1]
            ).abs().max()
        return {
            "layer3_native": layer3_native.detach(),
            "f4_projected": f4_projected.detach(),
            "AUD_L4": aud_l4.detach(),
            "f4_token_error": f4_token_error.detach(),
        }

    def _fine_from_teacher_features(
        self,
        teacher_view: dict[str, torch.Tensor],
        audio_query: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        f34, f3_spatial, f4_up, delta_f3 = self.student(
            teacher_view["layer3_native"], teacher_view["f4_projected"]
        )
        l4_branch = self.teacher.slot_attn.visual_branches[-1]
        fine_tokens = self._to_tokens(f34)
        # The L4 LayerNorm/key transform is frozen, but its operations remain in
        # the graph so gradients reach the private spatial projection and adapter.
        fine_keys = l4_branch.img_to_k(l4_branch.img_norm_input(fine_tokens))
        fine_all = self.teacher.slot_attn._attention(
            audio_query,
            fine_keys,
            self.teacher.infer_sharpening,
        )
        return {
            "AUD_FINE": self._target_map(fine_all, 14),
            "F34": f34,
            "F3_SPATIAL": f3_spatial,
            "F4_UP": f4_up,
            "DELTA_F3": delta_f3,
        }

    def forward(self, image: torch.Tensor, audio: torch.Tensor) -> dict[str, torch.Tensor]:
        """Single-view inference; no test-time view averaging."""
        audio_query = self._extract_audio_query(audio)
        teacher_view = self._extract_visual_teacher(image, audio_query)
        fine = self._fine_from_teacher_features(teacher_view, audio_query)
        return {
            "AUD_L4": teacher_view["AUD_L4"],
            "AUD_FINE": fine["AUD_FINE"],
            "F34": fine["F34"],
            "F3_SPATIAL": fine["F3_SPATIAL"],
            "F4_UP": fine["F4_UP"],
            "DELTA_F3": fine["DELTA_F3"],
            "f4_token_error": teacher_view["f4_token_error"],
        }

    def forward_two_views(
        self,
        image_a: torch.Tensor,
        audio: torch.Tensor,
        flip_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Generate B from A, then align all B maps back to A coordinates."""
        image_b = self.apply_horizontal_flip(image_a, flip_mask)
        audio_query = self._extract_audio_query(audio)

        teacher_a = self._extract_visual_teacher(image_a, audio_query)
        fine_a = self._fine_from_teacher_features(teacher_a, audio_query)
        teacher_b = self._extract_visual_teacher(image_b, audio_query)
        fine_b = self._fine_from_teacher_features(teacher_b, audio_query)

        return {
            "AUD_L4_A": teacher_a["AUD_L4"],
            "AUD_L4_B_ALIGNED": self.apply_horizontal_flip(
                teacher_b["AUD_L4"], flip_mask
            ),
            "AUD_FINE_A": fine_a["AUD_FINE"],
            "AUD_FINE_B_ALIGNED": self.apply_horizontal_flip(
                fine_b["AUD_FINE"], flip_mask
            ),
            "F34_A": fine_a["F34"],
            "F4_UP_A": fine_a["F4_UP"],
            "DELTA_F3_A": fine_a["DELTA_F3"],
            "f4_token_error": torch.maximum(
                teacher_a["f4_token_error"], teacher_b["f4_token_error"]
            ),
            "flip_fraction": flip_mask.float().mean(),
        }

    def spatial_losses(
        self,
        output: dict[str, torch.Tensor],
        lambda_equiv: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        aud_a = output["AUD_FINE_A"]
        aud_b_aligned = output["AUD_FINE_B_ALIGNED"]
        teacher_a = output["AUD_L4_A"].detach()
        teacher_b_aligned = output["AUD_L4_B_ALIGNED"].detach()

        loss_equiv = self.symmetric_kl(aud_a, aud_b_aligned)
        loss_coarse_a = self.kl_divergence(
            teacher_a, self.sum_pool_2x2(aud_a)
        )
        loss_coarse_b = self.kl_divergence(
            teacher_b_aligned, self.sum_pool_2x2(aud_b_aligned)
        )
        loss_coarse = 0.5 * (loss_coarse_a + loss_coarse_b)
        loss_spatial = loss_coarse + lambda_equiv * loss_equiv
        return {
            "loss_equiv": loss_equiv,
            "loss_coarse_a": loss_coarse_a,
            "loss_coarse_b": loss_coarse_b,
            "loss_coarse": loss_coarse,
            "loss_spatial": loss_spatial,
        }

    def close(self) -> None:
        self.feature_hooks.close()
