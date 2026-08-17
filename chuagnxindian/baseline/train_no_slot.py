#!/usr/bin/env python3
"""Training entry for the no-slot conventional AV baseline."""

import math
import os
import sys
from pathlib import Path

import torch


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXPERIMENT_DIR))

import train_slot  # noqa: E402
from evaluation_no_slot import evaluate_no_slot  # noqa: E402
from model_no_slot import NoSlotAVBaseline  # noqa: E402


def validate_no_slot(
    test_loader,
    testset,
    model,
    object_saliency_model,
    epoch,
    best,
    model_dir,
    args,
):
    results = evaluate_no_slot(
        test_loader, model, object_saliency_model, args
    )
    aud_ciou, aud_auc = results["aud"]
    obj_ciou, obj_auc = results["obj_prior"]
    ogl_ciou, ogl_auc = results["ogl"]

    if aud_ciou > best[0]:
        checkpoint = {
            "model": model.state_dict(),
            "epoch": epoch + 1,
            "selection_metric": "AUD_cIoU",
            "selection_score": aud_ciou,
        }
        torch.save(
            checkpoint, os.path.join(model_dir, f"{testset}_best.pth")
        )
        print(
            f"Best model saved to {model_dir} "
            f"(AUD cIoU={aud_ciou:.4f}, epoch={epoch + 1})"
        )

    best[0] = max(aud_ciou, best[0])
    best[1] = max(aud_auc, best[1])
    best[6] = max(ogl_ciou, best[6])
    best[7] = max(ogl_auc, best[7])

    print(
        f"AUD_{testset}/cIoU, auc, best_cIoU, best_auc",
        f"{aud_ciou:.4f}", f"{aud_auc:.4f}",
        f"{best[0]:.4f}", f"{best[1]:.4f}",
    )
    print(f"IMG_QUERY_{testset}/cIoU, auc N/A N/A")
    print(f"IQR_{testset}/cIoU, auc N/A N/A")
    print(
        f"OBJ_PRIOR_{testset}/cIoU, auc",
        f"{obj_ciou:.4f}", f"{obj_auc:.4f}",
    )
    print(
        f"OGL_{testset}/cIoU, auc, best_cIoU, best_auc",
        f"{ogl_ciou:.4f}", f"{ogl_auc:.4f}",
        f"{best[6]:.4f}", f"{best[7]:.4f}",
    )
    print(f"EXTRA_IQR_OGL_{testset}/cIoU, auc N/A N/A")

    nan = math.nan
    metrics = {
        f"AUD_{testset}/cIoU": aud_ciou,
        f"AUD_{testset}/auc": aud_auc,
        f"AUD_{testset}/best_cIoU": best[0],
        f"AUD_{testset}/best_auc": best[1],
        f"IMG_QUERY_{testset}/cIoU": nan,
        f"IMG_QUERY_{testset}/auc": nan,
        f"IMG_QUERY_{testset}/best_cIoU": nan,
        f"IMG_QUERY_{testset}/best_auc": nan,
        f"IQR_{testset}/cIoU": nan,
        f"IQR_{testset}/auc": nan,
        f"IQR_{testset}/best_cIoU": nan,
        f"IQR_{testset}/best_auc": nan,
        f"OBJ_PRIOR_{testset}/cIoU": obj_ciou,
        f"OBJ_PRIOR_{testset}/auc": obj_auc,
        f"OGL_{testset}/cIoU": ogl_ciou,
        f"OGL_{testset}/auc": ogl_auc,
        f"OGL_{testset}/best_cIoU": best[6],
        f"OGL_{testset}/best_auc": best[7],
        f"EXTRA_IQR_OGL_{testset}/cIoU": nan,
        f"EXTRA_IQR_OGL_{testset}/auc": nan,
        f"EXTRA_IQR_OGL_{testset}/best_cIoU": nan,
        f"EXTRA_IQR_OGL_{testset}/best_auc": nan,
        "epoch": epoch,
    }
    train_slot.wandb.log(
        {
            f"AUD_{testset}/cIoU": aud_ciou,
            f"AUD_{testset}/auc": aud_auc,
            f"OBJ_PRIOR_{testset}/cIoU": obj_ciou,
            f"OBJ_PRIOR_{testset}/auc": obj_auc,
            f"OGL_{testset}/cIoU": ogl_ciou,
            f"OGL_{testset}/auc": ogl_auc,
            "epoch": epoch,
        }
    )
    return best, metrics


def main():
    train_slot.model_baseline.AudioVisualMIL = NoSlotAVBaseline
    train_slot.validate = validate_no_slot
    args = train_slot.get_arguments()
    if args.model != "av_mil":
        raise ValueError("B0 no-slot training requires --model av_mil")
    args.architecture = "b0_baseline"
    args.slot_attention = False
    args.available_eval_metrics = ["AUD", "OBJ_PRIOR", "OGL"]
    args.checkpoint_selection = "AUD_cIoU"
    train_slot.main(args)


if __name__ == "__main__":
    main()
