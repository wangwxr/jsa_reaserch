"""1.3G-v2: Experiment G plus frozen visual-slot coarse preservation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
G_ROOT = HERE.parent / "1.3G-multigeom_equivariant_l3_refine"
if str(G_ROOT) not in sys.path:
    sys.path.append(str(G_ROOT))


def _load_g_model():
    path = G_ROOT / "model.py"
    spec = importlib.util.spec_from_file_location("experiment_g_model_for_v2", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


g_model = _load_g_model()
BaseMultiGeometryRefinement = g_model.MultiGeometryEquivariantRefinement


class VisualSemanticPreservingRefinement(BaseMultiGeometryRefinement):
    """No new parameters; Q4->K34 adds one frozen-query training constraint."""

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
            _, image_query, image_keys_l4 = l4_branch(
                image_levels[-1], initial_slots
            )
            aud_l4_all = self.teacher.slot_attn._attention(
                audio_query, image_keys_l4, self.teacher.infer_sharpening
            )
            img_l4_all = self.teacher.slot_attn._attention(
                image_query, image_keys_l4, self.teacher.infer_sharpening
            )
            f4_token_error = (
                self._to_tokens(f4_projected) - image_levels[-1]
            ).abs().max()
        return {
            "layer3_native": layer3_native.detach(),
            "f4_projected": f4_projected.detach(),
            "QA": audio_query.detach(),
            "Q4": image_query.detach(),
            "K4": image_keys_l4.detach(),
            "AUD_L4": self._target_map(aud_l4_all, 7).detach(),
            "IMG_L4": self._target_map(img_l4_all, 7).detach(),
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
        aud_fine_all = self.teacher.slot_attn._attention(
            audio_query, fine_keys, self.teacher.infer_sharpening
        )
        img_fine_all = self.teacher.slot_attn._attention(
            teacher_view["Q4"], fine_keys, self.teacher.infer_sharpening
        )
        return {
            "AUD_FINE": self._target_map(aud_fine_all, 14),
            "IMG_FINE": self._target_map(img_fine_all, 14),
            "F34": f34,
            "F3_SPATIAL": f3_spatial,
            "F4_UP": f4_up,
            "DELTA_F3": delta_f3,
            "K34": fine_keys,
        }

    def forward(
        self, image: torch.Tensor, audio: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        audio_query = self._extract_audio_query(audio)
        teacher_view = self._extract_visual_teacher(image, audio_query)
        fine = self._fine_from_teacher_features(teacher_view, audio_query)
        return {
            "AUD_L4": teacher_view["AUD_L4"],
            "IMG_L4": teacher_view["IMG_L4"],
            "AUD_FINE": fine["AUD_FINE"],
            "IMG_FINE": fine["IMG_FINE"],
            "F34": fine["F34"],
            "K34": fine["K34"],
            "F4_UP": fine["F4_UP"],
            "DELTA_F3": fine["DELTA_F3"],
            "QA": teacher_view["QA"],
            "Q4": teacher_view["Q4"],
            "K4": teacher_view["K4"],
            "f4_token_error": teacher_view["f4_token_error"],
        }

    def forward_two_views(
        self,
        image_a: torch.Tensor,
        audio: torch.Tensor,
        geometry: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        image_b = g_model.apply_to_view_a(image_a, geometry)
        audio_query = self._extract_audio_query(audio)

        teacher_a = self._extract_visual_teacher(image_a, audio_query)
        fine_a = self._fine_from_teacher_features(teacher_a, audio_query)
        teacher_b = self._extract_visual_teacher(image_b, audio_query)
        fine_b = self._fine_from_teacher_features(teacher_b, audio_query)

        fine_b_to_a, valid14 = g_model.warp_view_b_to_a(
            fine_b["AUD_FINE"], geometry, output_size=(14, 14)
        )
        coarse_b_to_a, valid7 = g_model.warp_view_b_to_a(
            teacher_b["AUD_L4"], geometry, output_size=(7, 7)
        )
        return {
            "VIEW_B": image_b,
            "AUD_L4_A": teacher_a["AUD_L4"],
            "AUD_L4_B": teacher_b["AUD_L4"],
            "AUD_L4_B_TO_A": coarse_b_to_a,
            "AUD_FINE_A": fine_a["AUD_FINE"],
            "AUD_FINE_B": fine_b["AUD_FINE"],
            "AUD_FINE_B_TO_A": fine_b_to_a,
            "IMG_L4_A": teacher_a["IMG_L4"],
            "IMG_L4_B": teacher_b["IMG_L4"],
            "IMG_FINE_A": fine_a["IMG_FINE"],
            "IMG_FINE_B": fine_b["IMG_FINE"],
            "VALID_MASK_14": valid14,
            "VALID_MASK_7": valid7,
            "F34_A": fine_a["F34"],
            "F4_UP_A": fine_a["F4_UP"],
            "DELTA_F3_A": fine_a["DELTA_F3"],
            "K34_A": fine_a["K34"],
            "QA": teacher_a["QA"],
            "Q4_A": teacher_a["Q4"],
            "K4_A": teacher_a["K4"],
            "f4_token_error": torch.maximum(
                teacher_a["f4_token_error"], teacher_b["f4_token_error"]
            ),
            "actual_flip_ratio": geometry["flipped"].float().mean(),
            "mean_crop_scale": g_model.crop_scale(geometry).mean(),
        }

    def spatial_losses(
        self,
        output: dict[str, torch.Tensor],
        lambda_equiv: float = 1.0,
        lambda_img: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        # This call is intentionally unchanged and is regression-tested against G.
        losses = super().spatial_losses(output, lambda_equiv=lambda_equiv)

        pooled_img_a = self.sum_pool_2x2(output["IMG_FINE_A"])
        loss_img_a = self._kl_per_sample(
            self._normalize_spatial(output["IMG_L4_A"].detach()),
            self._normalize_spatial(pooled_img_a),
        ).mean()
        pooled_img_b = self.sum_pool_2x2(output["IMG_FINE_B"])
        loss_img_b = self._kl_per_sample(
            self._normalize_spatial(output["IMG_L4_B"].detach()),
            self._normalize_spatial(pooled_img_b),
        ).mean()
        loss_img_coarse = 0.5 * (loss_img_a + loss_img_b)
        loss_aud_total = losses["loss_total"]
        return {
            **losses,
            "loss_aud_coarse": losses["loss_coarse"],
            "loss_aud_equiv": losses["loss_equiv"],
            "loss_aud_total": loss_aud_total,
            "loss_img_coarse_a": loss_img_a,
            "loss_img_coarse_b": loss_img_b,
            "loss_img_coarse": loss_img_coarse,
            "loss_total": loss_aud_total + lambda_img * loss_img_coarse,
        }
