"""Original 1.3G Stage-2 refinement with an additive, direct C3 CRAM loss."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
RECIPE = HERE.parents[1] / "1mufasaslot/1.3G-multigeom_equivariant_l3_refine"
_spec = importlib.util.spec_from_file_location("_original_stage2_model", RECIPE / "model.py")
if _spec is None or _spec.loader is None:
    raise RuntimeError("Cannot load original Stage-2 model")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


class Stage2WithCRAM(_base.MultiGeometryEquivariantRefinement):
    """Frozen C3 teacher and unchanged Stage-2 student, with exposed CRAM tensors."""

    def _extract_visual_teacher(
        self, image: torch.Tensor, audio_query: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        self.teacher.eval()
        with torch.no_grad():
            image_levels = self.teacher.imgnet(image)
            layer3_native, f4_projected = self.feature_hooks.pop()
            slots = self.teacher.slot_attn.slots.expand(image.shape[0], -1, -1)
            branch = self.teacher.slot_attn.visual_branches[-1]
            _slots, visual_query, image_keys_l4 = branch(image_levels[-1], slots)
            coarse_all = self.teacher.slot_attn._attention(
                audio_query, image_keys_l4, self.teacher.infer_sharpening
            )
            aud_l4 = self._target_map(coarse_all, 7)
            f4_token_error = (self._to_tokens(f4_projected) - image_levels[-1]).abs().max()
        return {
            "layer3_native": layer3_native.detach(),
            "f4_projected": f4_projected.detach(),
            "AUD_L4": aud_l4.detach(),
            "VISUAL_QUERY": visual_query.detach(),
            "f4_token_error": f4_token_error.detach(),
        }

    def _fine_from_teacher_features(
        self, teacher_view: dict[str, torch.Tensor], audio_query: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        f34, f3_spatial, f4_up, delta_f3 = self.student(
            teacher_view["layer3_native"], teacher_view["f4_projected"]
        )
        branch = self.teacher.slot_attn.visual_branches[-1]
        fine_tokens = self._to_tokens(f34)
        fine_keys = branch.img_to_k(branch.img_norm_input(fine_tokens))
        fine_all = self.teacher.slot_attn._attention(
            audio_query, fine_keys, self.teacher.infer_sharpening
        )
        return {
            "AUD_FINE": self._target_map(fine_all, 14),
            "FINE_KEYS": fine_keys,
            "F34": f34,
            "F3_SPATIAL": f3_spatial,
            "F4_UP": f4_up,
            "DELTA_F3": delta_f3,
        }

    def forward_two_views(
        self, image_a: torch.Tensor, audio: torch.Tensor, geometry: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        image_b = _base.apply_to_view_a(image_a, geometry)
        audio_query = self._extract_audio_query(audio)
        teacher_a = self._extract_visual_teacher(image_a, audio_query)
        fine_a = self._fine_from_teacher_features(teacher_a, audio_query)
        teacher_b = self._extract_visual_teacher(image_b, audio_query)
        fine_b = self._fine_from_teacher_features(teacher_b, audio_query)
        fine_b_to_a, valid14 = _base.warp_view_b_to_a(fine_b["AUD_FINE"], geometry, output_size=(14, 14))
        coarse_b_to_a, valid7 = _base.warp_view_b_to_a(teacher_b["AUD_L4"], geometry, output_size=(7, 7))
        return {
            "VIEW_B": image_b,
            "AUD_L4_A": teacher_a["AUD_L4"], "AUD_L4_B": teacher_b["AUD_L4"],
            "AUD_L4_B_TO_A": coarse_b_to_a,
            "AUD_FINE_A": fine_a["AUD_FINE"], "AUD_FINE_B": fine_b["AUD_FINE"],
            "AUD_FINE_B_TO_A": fine_b_to_a,
            "VALID_MASK_14": valid14, "VALID_MASK_7": valid7,
            "F34_A": fine_a["F34"], "F4_UP_A": fine_a["F4_UP"],
            "DELTA_F3_A": fine_a["DELTA_F3"],
            "VISUAL_QUERY_A": teacher_a["VISUAL_QUERY"],
            "AUDIO_QUERY_A": audio_query,
            "FINE_KEYS_A": fine_a["FINE_KEYS"],
            "f4_token_error": torch.maximum(teacher_a["f4_token_error"], teacher_b["f4_token_error"]),
            "actual_flip_ratio": geometry["flipped"].float().mean(),
            "mean_crop_scale": _base.crop_scale(geometry).mean(),
        }

    def stage2_cram_components(
        self, output: dict[str, torch.Tensor], wrong_audio: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """The Stage-1 C3 softplus relative loss on the existing 14x14 decision keys."""
        if wrong_audio.ndim != 5:
            raise ValueError(f"Expected [B,K,C,F,T] wrong audio, got {wrong_audio.shape}")
        # Stage-2 attention deviations are around 1e-4. Compute their squared
        # distances in fp32 even when the original Stage-2 forward uses AMP;
        # otherwise fp16 MSE can make the new term silently zero-gradient.
        with torch.amp.autocast(device_type="cuda", enabled=False):
            keys = output["FINE_KEYS_A"].float()
            target = self.teacher.slot_attn._attention(
                output["VISUAL_QUERY_A"].float(), keys, self.teacher.infer_sharpening
            )[:, 0].detach()
            matched = self.teacher.slot_attn._attention(
                output["AUDIO_QUERY_A"].float(), keys, self.teacher.infer_sharpening
            )[:, 0]
            d_pos = F.mse_loss(matched, target, reduction="none").mean(1)
            negatives = []
            for index in range(wrong_audio.shape[1]):
                query = self._extract_audio_query(wrong_audio[:, index]).float()
                attention = self.teacher.slot_attn._attention(
                    query, keys, self.teacher.infer_sharpening
                )[:, 0]
                negatives.append(F.mse_loss(attention, target, reduction="none").mean(1))
            d_neg = torch.stack(negatives, dim=1).mean(1)
            cram = F.softplus(d_pos - d_neg).mean()
        return {"d_pos": d_pos.mean(), "d_neg": d_neg.mean(),
                "g_match": (d_neg - d_pos).mean(), "cram_loss": cram,
                "visual_anchor": target}
