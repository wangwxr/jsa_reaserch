"""Original 8.6 objective plus additive Counterfactual Relative Attention Matching."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


_source_path = Path(__file__).resolve().parent.parent / "8.6_loss_to_decision_causal_ablation" / "objective.py"
_spec = importlib.util.spec_from_file_location("_experiment_86_objective", _source_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError("Cannot load Experiment 8.6 objective.py")
_source = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_source)

attention = _source.attention
forward_components = _source.forward_components
inference_tensors = _source.inference_tensors


def _wrong_a2v(model, audio: torch.Tensor, visual_keys: torch.Tensor) -> torch.Tensor:
    """One wrong-audio A2V pass without updating audnet BatchNorm statistics."""
    was_training = model.audnet.training
    model.audnet.eval()
    try:
        tokens = model._audio_tokens(model.audnet(audio))
    finally:
        model.audnet.train(was_training)
    slots = model.slot_attn.slots.expand(len(audio), -1, -1)
    masked_audio = model.slot_attn._masked(tokens, model.slot_attn.mask_token_aud)
    _slots, query, _keys = model.slot_attn.audio_branch(masked_audio, slots)
    _logits, probabilities = attention(query, visual_keys)
    return probabilities[:, 0]


def cram_components(model, output: dict[str, torch.Tensor], wrong_spec: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compare same-image A2V attention for matched and K wrong audios.

    The visual keys and detached V2V target come from the matched forward pass.
    Wrong audios only traverse the existing audio branch; no new module or target
    is introduced.
    """
    _batch_size, negatives = wrong_spec.shape[:2]
    visual_keys = output["visual_keys"]
    target = output["v2v_prob"][:, 0].detach()
    d_pos = F.mse_loss(output["a2v_prob"][:, 0], target, reduction="none").mean(1)
    distances = []
    for index in range(negatives):
        # Reentrant checkpoint keeps only one wrong-audio branch resident. Both
        # audio and visual keys are explicit inputs, so CRAM still differentiates
        # through the existing audio query and L4 visual-key paths.
        audio = wrong_spec[:, index].detach().requires_grad_(True)
        wrong_attention = checkpoint(
            lambda item, keys: _wrong_a2v(model, item, keys),
            audio, visual_keys, use_reentrant=True,
        )
        distances.append(F.mse_loss(wrong_attention, target, reduction="none").mean(1))
    d_neg = torch.stack(distances, dim=1).mean(1)
    loss = F.softplus(d_pos - d_neg).mean()
    return {
        "d_pos": d_pos.mean(),
        "d_neg": d_neg.mean(),
        "g_match": (d_neg - d_pos).mean(),
        "cram_loss": loss,
    }


def total_loss(output: dict[str, torch.Tensor], cram_loss: torch.Tensor, lambda_cram: float) -> torch.Tensor:
    return (
        output["info_loss"]
        + 0.1 * output["recon_loss"]
        + 0.1 * output["div_loss"]
        + 100.0 * output["match_loss"]
        + float(lambda_cram) * cram_loss
    )
