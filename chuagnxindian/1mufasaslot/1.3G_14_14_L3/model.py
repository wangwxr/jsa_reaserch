"""Experiment G mechanism with a frozen native-L3 semantic teacher."""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from geometry import apply_to_view_a, crop_scale, warp_view_b_to_a


class TopDownL3Adapter(nn.Module):
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
            raise ValueError(f"Expected F3 [B,512,14,14], got {f3_spatial.shape}")
        if f4_projected.shape[1:] != (512, 7, 7):
            raise ValueError(f"Expected F4 [B,512,7,7], got {f4_projected.shape}")
        f4_up = F.interpolate(
            f4_projected,
            size=f3_spatial.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        delta_f3 = self.adapter(f3_spatial)
        return f4_up + delta_f3, f3_spatial, f4_up, delta_f3


class _TeacherFeatureHooks:
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
                f"Expected teacher hooks {sorted(expected)}, got {sorted(self.outputs)}"
            )
        return self.outputs.pop("layer3_native"), self.outputs.pop("f4_projected")

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


class MultiGeometryEquivariantRefinement(nn.Module):
    """Frozen semantic teacher; only proj3_spatial and adapter are trainable."""

    def __init__(self, teacher: nn.Module, minimum_valid_ratio: float = 0.2):
        super().__init__()
        self.teacher = teacher
        self.student = FineSpatialStudent(teacher.imgnet.proj3)
        self.feature_hooks = _TeacherFeatureHooks(teacher)
        self.minimum_valid_ratio = minimum_valid_ratio

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
    def sum_pool_2x2(probability: torch.Tensor) -> torch.Tensor:
        if probability.shape[-2:] != (14, 14):
            raise ValueError(f"Expected 14x14 map, got {probability.shape}")
        pooled = F.avg_pool2d(probability, kernel_size=2, stride=2) * 4.0
        return pooled / pooled.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)

    @staticmethod
    def _normalize_spatial(
        probability: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        values = probability.float()
        if mask is not None:
            values = values * mask.float()
        return values / values.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)

    @staticmethod
    def _kl_per_sample(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        p_flat = p.flatten(start_dim=1).clamp_min(1e-8)
        q_flat = q.flatten(start_dim=1).clamp_min(1e-8)
        p_flat = p_flat / p_flat.sum(dim=-1, keepdim=True)
        q_flat = q_flat / q_flat.sum(dim=-1, keepdim=True)
        return (p_flat * (p_flat.log() - q_flat.log())).sum(dim=-1)

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
                audio_query, image_keys_l4, self.teacher.infer_sharpening
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
        fine_keys = l4_branch.img_to_k(l4_branch.img_norm_input(fine_tokens))
        fine_all = self.teacher.slot_attn._attention(
            audio_query, fine_keys, self.teacher.infer_sharpening
        )
        return {
            "AUD_FINE": self._target_map(fine_all, 14),
            "F34": f34,
            "F3_SPATIAL": f3_spatial,
            "F4_UP": f4_up,
            "DELTA_F3": delta_f3,
            "FINE_TOKENS": fine_tokens,
            "FINE_KEYS": fine_keys,
            "QA": audio_query,
        }

    def forward(self, image: torch.Tensor, audio: torch.Tensor) -> dict[str, torch.Tensor]:
        """Unchanged single-view Experiment F evaluation path."""
        audio_query = self._extract_audio_query(audio)
        teacher_view = self._extract_visual_teacher(image, audio_query)
        fine = self._fine_from_teacher_features(teacher_view, audio_query)
        return {
            "AUD_L4": teacher_view["AUD_L4"],
            "F4": teacher_view["f4_projected"],
            "AUD_FINE": fine["AUD_FINE"],
            "F34": fine["F34"],
            "F3_SPATIAL": fine["F3_SPATIAL"],
            "F4_UP": fine["F4_UP"],
            "DELTA_F3": fine["DELTA_F3"],
            "FINE_TOKENS": fine["FINE_TOKENS"],
            "FINE_KEYS": fine["FINE_KEYS"],
            "QA": fine["QA"],
            "f4_token_error": teacher_view["f4_token_error"],
        }

    def forward_two_views(
        self,
        image_a: torch.Tensor,
        audio: torch.Tensor,
        geometry: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        image_b = apply_to_view_a(image_a, geometry)
        audio_query = self._extract_audio_query(audio)

        teacher_a = self._extract_visual_teacher(image_a, audio_query)
        fine_a = self._fine_from_teacher_features(teacher_a, audio_query)
        teacher_b = self._extract_visual_teacher(image_b, audio_query)
        fine_b = self._fine_from_teacher_features(teacher_b, audio_query)

        fine_b_to_a, valid14 = warp_view_b_to_a(
            fine_b["AUD_FINE"], geometry, output_size=(14, 14)
        )
        coarse_b_to_a, valid7 = warp_view_b_to_a(
            teacher_b["AUD_L4"], geometry, output_size=(7, 7)
        )
        return {
            "VIEW_B": image_b,
            "AUD_L4_A": teacher_a["AUD_L4"],
            "AUD_L4_B": teacher_b["AUD_L4"],
            "F4_A": teacher_a["f4_projected"],
            "AUD_L4_B_TO_A": coarse_b_to_a,
            "AUD_FINE_A": fine_a["AUD_FINE"],
            "AUD_FINE_B": fine_b["AUD_FINE"],
            "AUD_FINE_B_TO_A": fine_b_to_a,
            "VALID_MASK_14": valid14,
            "VALID_MASK_7": valid7,
            "F34_A": fine_a["F34"],
            "F3_SPATIAL_A": fine_a["F3_SPATIAL"],
            "F4_UP_A": fine_a["F4_UP"],
            "DELTA_F3_A": fine_a["DELTA_F3"],
            "FINE_TOKENS_A": fine_a["FINE_TOKENS"],
            "FINE_KEYS_A": fine_a["FINE_KEYS"],
            "QA": fine_a["QA"],
            "f4_token_error": torch.maximum(
                teacher_a["f4_token_error"], teacher_b["f4_token_error"]
            ),
            "actual_flip_ratio": geometry["flipped"].float().mean(),
            "mean_crop_scale": crop_scale(geometry).mean(),
        }

    def spatial_losses(
        self,
        output: dict[str, torch.Tensor],
        lambda_equiv: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        valid14 = output["VALID_MASK_14"]
        valid_ratio = valid14.mean(dim=(-3, -2, -1))
        use_sample = valid_ratio >= self.minimum_valid_ratio

        p = self._normalize_spatial(output["AUD_FINE_A"], valid14)
        q = self._normalize_spatial(output["AUD_FINE_B_TO_A"], valid14)
        symmetric_kl = 0.5 * (
            self._kl_per_sample(p, q) + self._kl_per_sample(q, p)
        )
        if use_sample.any():
            loss_equiv = symmetric_kl[use_sample].mean()
        else:
            loss_equiv = (
                output["AUD_FINE_A"].sum() + output["AUD_FINE_B_TO_A"].sum()
            ) * 0.0

        teacher_a = output["AUD_L4_A"].detach()
        pooled_a = self.sum_pool_2x2(output["AUD_FINE_A"])
        loss_coarse_a = self._kl_per_sample(
            self._normalize_spatial(teacher_a),
            self._normalize_spatial(pooled_a),
        ).mean()

        valid7 = output["VALID_MASK_7"]
        teacher_b_to_a = self._normalize_spatial(
            output["AUD_L4_B_TO_A"].detach(), valid7
        )
        pooled_b_to_a = self.sum_pool_2x2(output["AUD_FINE_B_TO_A"])
        pooled_b_to_a = self._normalize_spatial(pooled_b_to_a, valid7)
        loss_coarse_b = self._kl_per_sample(
            teacher_b_to_a, pooled_b_to_a
        ).mean()

        loss_coarse = 0.5 * (loss_coarse_a + loss_coarse_b)
        loss_total = loss_coarse + lambda_equiv * loss_equiv
        return {
            "loss_equiv": loss_equiv,
            "loss_coarse_a": loss_coarse_a,
            "loss_coarse_b": loss_coarse_b,
            "loss_coarse": loss_coarse,
            "loss_total": loss_total,
            "mean_valid_ratio": valid_ratio.mean(),
            "skipped_small_overlap_samples": (~use_sample).sum(),
        }

    def close(self) -> None:
        self.feature_hooks.close()
