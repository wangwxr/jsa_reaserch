"""Frozen 1.3G teacher plus one trainable semantic-spatial Slot branch."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from spatial_slot import SpatialSlotAttention


class SemanticSpatialDecoupledModel(nn.Module):
    def __init__(self, refinement: nn.Module, geometry_module, iters: int = 5):
        super().__init__()
        self.refinement = refinement
        self.geometry = geometry_module
        self.spatial_slot = SpatialSlotAttention(
            slot_dim=512, num_slots=2, iters=iters
        )
        for parameter in self.refinement.parameters():
            parameter.requires_grad = False
        self.refinement.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.refinement.eval()
        self.spatial_slot.train(mode)
        return self

    @property
    def teacher(self):
        return self.refinement.teacher

    @staticmethod
    def _tokens(feature: torch.Tensor) -> torch.Tensor:
        return feature.flatten(start_dim=2).transpose(1, 2)

    @staticmethod
    def _target_map(attention: torch.Tensor) -> torch.Tensor:
        return attention[:, 0].reshape(-1, 1, 14, 14)

    def _audio_query(self, audio: torch.Tensor) -> torch.Tensor:
        teacher = self.teacher
        with torch.no_grad():
            audio_tokens = teacher._audio_tokens(teacher.audnet(audio))
            initial = teacher.slot_attn.slots.expand(audio.shape[0], -1, -1)
            _slots, query, _keys = teacher.slot_attn.audio_branch(
                audio_tokens, initial
            )
        return query.detach()

    def _frozen_view(
        self, image: torch.Tensor, audio_query: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        teacher = self.teacher
        with torch.no_grad():
            levels = teacher.imgnet(image)
            layer3_native, f4_projected = self.refinement.feature_hooks.pop()
            initial = teacher.slot_attn.slots.expand(image.shape[0], -1, -1)
            l4_branch = teacher.slot_attn.visual_branches[-1]
            semantic_slots, semantic_q4, _k4 = l4_branch(levels[-1], initial)
            f34, _f3, _f4_up, _delta = self.refinement.student(
                layer3_native, f4_projected
            )
            fine_tokens = self._tokens(f34)
            k34 = l4_branch.img_to_k(l4_branch.img_norm_input(fine_tokens))
            aud_attention = teacher.slot_attn._attention(
                audio_query, k34, teacher.infer_sharpening
            )
            old_logits = (
                torch.einsum("bsd,bnd->bsn", semantic_q4, k34)
                * l4_branch.scale
            )
            old_ownership = old_logits.softmax(dim=1)
            f4_error = (
                self._tokens(f4_projected) - levels[-1]
            ).abs().max()
        return {
            "F34": f34.detach(),
            "SEMANTIC_L4_SLOTS": semantic_slots.detach(),
            "AUD_FINE": self._target_map(aud_attention).detach(),
            "OLD_OWNERSHIP": old_ownership.detach(),
            "f4_token_error": f4_error.detach(),
        }

    def _spatial_view(self, frozen: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        tokens = self._tokens(frozen["F34"]).detach()
        semantic_slots = frozen["SEMANTIC_L4_SLOTS"].detach()
        spatial_slots, ownership = self.spatial_slot(tokens, semantic_slots)
        return {
            **frozen,
            "SPATIAL_SLOTS": spatial_slots,
            "OWNERSHIP": ownership,
            "SPATIAL_SLOT0": ownership[:, 0].reshape(-1, 1, 14, 14),
        }

    def forward(
        self, image: torch.Tensor, audio: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        audio_query = self._audio_query(audio)
        return self._spatial_view(self._frozen_view(image, audio_query))

    def forward_two_views(
        self,
        image_a: torch.Tensor,
        audio: torch.Tensor,
        geometry: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        image_b = self.geometry.apply_to_view_a(image_a, geometry)
        audio_query = self._audio_query(audio)
        view_a = self._spatial_view(self._frozen_view(image_a, audio_query))
        view_b = self._spatial_view(self._frozen_view(image_b, audio_query))
        ownership_b_map = view_b["OWNERSHIP"].reshape(-1, 2, 14, 14)
        ownership_b_to_a, valid = self.geometry.warp_view_b_to_a(
            ownership_b_map, geometry, output_size=(14, 14)
        )
        return {
            "VIEW_B": image_b,
            "AUD_FINE_A": view_a["AUD_FINE"],
            "F34_A": view_a["F34"],
            "F34_B": view_b["F34"],
            "SEMANTIC_INITIAL_A": view_a["SEMANTIC_L4_SLOTS"],
            "SEMANTIC_INITIAL_B": view_b["SEMANTIC_L4_SLOTS"],
            "OWNERSHIP_A": view_a["OWNERSHIP"],
            "OWNERSHIP_B": view_b["OWNERSHIP"],
            "OWNERSHIP_B_TO_A": ownership_b_to_a,
            "OLD_OWNERSHIP_A": view_a["OLD_OWNERSHIP"],
            "OLD_OWNERSHIP_B": view_b["OLD_OWNERSHIP"],
            "VALID_MASK": valid,
            "f4_token_error": torch.maximum(
                view_a["f4_token_error"], view_b["f4_token_error"]
            ),
        }

    @staticmethod
    def _visual_coherence(
        ownership: torch.Tensor, feature_map: torch.Tensor
    ) -> torch.Tensor:
        features = F.normalize(
            feature_map.detach().flatten(start_dim=2).transpose(1, 2).float(),
            dim=-1,
        )
        probability = ownership.float()
        prototypes = torch.einsum("bsn,bnd->bsd", probability, features)
        prototypes = prototypes / probability.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        prototypes = F.normalize(prototypes, dim=-1)
        reconstructed = torch.einsum("bsn,bsd->bnd", probability, prototypes)
        return (1.0 - F.cosine_similarity(features, reconstructed, dim=-1)).mean()

    @staticmethod
    def _equivariance(
        ownership_a: torch.Tensor,
        ownership_b_to_a: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        p = ownership_a.reshape(-1, 2, 14, 14).float().clamp_min(1e-8)
        q = ownership_b_to_a.float().clamp_min(1e-8)
        p = p / p.sum(dim=1, keepdim=True).clamp_min(1e-8)
        q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-8)
        kl_pq = (p * (p.log() - q.log())).sum(dim=1, keepdim=True)
        kl_qp = (q * (q.log() - p.log())).sum(dim=1, keepdim=True)
        symmetric = 0.5 * (kl_pq + kl_qp)
        return (symmetric * valid.float()).sum() / valid.sum().clamp_min(1.0)

    def losses(
        self,
        output: dict[str, torch.Tensor],
        lambda_seed: float = 1.0,
        lambda_equiv: float = 1.0,
        lambda_visual: float = 0.1,
        lambda_mass: float = 0.1,
    ) -> dict[str, torch.Tensor]:
        ownership_a = output["OWNERSHIP_A"]
        ownership_b = output["OWNERSHIP_B"]
        aud = output["AUD_FINE_A"].detach().flatten(start_dim=1)
        seed_count = max(1, math.ceil(0.10 * aud.shape[-1]))
        seed_index = aud.topk(seed_count, dim=-1).indices
        slot0 = ownership_a[:, 0]
        seed_probability = slot0.gather(dim=1, index=seed_index).float()
        loss_seed = -seed_probability.clamp_min(1e-8).log().mean()

        loss_equiv = self._equivariance(
            ownership_a, output["OWNERSHIP_B_TO_A"], output["VALID_MASK"]
        )
        loss_visual = 0.5 * (
            self._visual_coherence(ownership_a, output["F34_A"])
            + self._visual_coherence(ownership_b, output["F34_B"])
        )
        new_mass_a = ownership_a.float().mean(dim=-1)
        new_mass_b = ownership_b.float().mean(dim=-1)
        old_mass_a = output["OLD_OWNERSHIP_A"].detach().float().mean(dim=-1)
        old_mass_b = output["OLD_OWNERSHIP_B"].detach().float().mean(dim=-1)
        loss_mass = 0.5 * (
            F.l1_loss(new_mass_a, old_mass_a)
            + F.l1_loss(new_mass_b, old_mass_b)
        )

        weighted_seed = lambda_seed * loss_seed
        weighted_equiv = lambda_equiv * loss_equiv
        weighted_visual = lambda_visual * loss_visual
        weighted_mass = lambda_mass * loss_mass
        loss_total = weighted_seed + weighted_equiv + weighted_visual + weighted_mass

        probability = ownership_a.float().clamp_min(1e-8)
        entropy = -(probability * probability.log()).sum(dim=1).mean() / math.log(2.0)
        return {
            "loss_seed": loss_seed,
            "loss_equiv": loss_equiv,
            "loss_visual": loss_visual,
            "loss_mass": loss_mass,
            "weighted_seed": weighted_seed,
            "weighted_equiv": weighted_equiv,
            "weighted_visual": weighted_visual,
            "weighted_mass": weighted_mass,
            "loss_total": loss_total,
            "slot0_mass": new_mass_a[:, 0].mean(),
            "slot1_mass": new_mass_a[:, 1].mean(),
            "ownership_entropy": entropy,
            "valid_overlap_ratio": output["VALID_MASK"].float().mean(),
        }

    def close(self) -> None:
        self.refinement.close()

