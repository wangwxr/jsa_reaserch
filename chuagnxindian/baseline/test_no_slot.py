#!/usr/bin/env python3
"""Test entry for the no-slot baseline with N/A slot-only metrics."""

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXPERIMENT_DIR))

import test_model  # noqa: E402
from dataset import get_test_dataset  # noqa: E402
from dataset_avs import get_ms3_dataset, get_s4_dataset  # noqa: E402
from evaluation_no_slot import evaluate_no_slot  # noqa: E402
from model_no_slot import NoSlotAVBaseline  # noqa: E402


def main():
    args = test_model.get_arguments()
    if getattr(args, "architecture", None) != "b0_baseline":
        raise RuntimeError(
            "This entry only tests b0_baseline checkpoints; "
            f"configs.json has architecture="
            f"{getattr(args, 'architecture', None)!r}"
        )
    if getattr(args, "model", None) != "av_mil":
        raise ValueError("B0 no-slot testing requires model='av_mil'")

    test_model.setup_seed(12345)
    model_dir = os.path.join(args.model_dir, args.experiment_name)
    model = NoSlotAVBaseline(args).cuda(args.gpu)

    object_saliency_model = torchvision.models.resnet18(
        weights="ResNet18_Weights.IMAGENET1K_V1"
    )
    object_saliency_model.avgpool = nn.Identity()
    object_saliency_model.fc = nn.Sequential(
        nn.Unflatten(1, (512, 7, 7)),
        test_model.NormReducer(dim=1),
        test_model.Unsqueeze(1),
    )
    object_saliency_model = object_saliency_model.cuda(args.gpu)

    checkpoint_name = getattr(args, "checkpoint", None)
    if checkpoint_name is None:
        checkpoint_name = f"{args.testset}_best.pth"
    checkpoint_path = os.path.join(model_dir, checkpoint_name)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model.load_state_dict(
        {
            key.replace("module.", ""): value
            for key, value in checkpoint["model"].items()
        }
    )
    print(f"loaded from {checkpoint_path}")

    if args.testset == "ms3":
        dataset = get_ms3_dataset(args)
    elif args.testset == "s4":
        dataset = get_s4_dataset(args)
    else:
        dataset = get_test_dataset(args, args.testset)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    print("Loaded dataloader.")

    results = evaluate_no_slot(
        loader,
        model,
        object_saliency_model,
        args,
        output_dir=os.path.join(model_dir, "eval_no_slot"),
    )
    aud_ciou, aud_auc = results["aud"]
    obj_ciou, obj_auc = results["obj_prior"]
    ogl_ciou, ogl_auc = results["ogl"]

    print(
        f"AUD_{args.testset}/cIoU, auc",
        f"{aud_ciou:.4f}", f"{aud_auc:.4f}",
    )
    print(f"IMG_QUERY_{args.testset}/cIoU, auc N/A N/A")
    print(f"IQR_{args.testset}/cIoU, auc N/A N/A")
    print(
        f"OBJ_PRIOR_{args.testset}/cIoU, auc",
        f"{obj_ciou:.4f}", f"{obj_auc:.4f}",
    )
    print(
        f"OGL_{args.testset}/cIoU, auc",
        f"{ogl_ciou:.4f}", f"{ogl_auc:.4f}",
    )
    print(f"EXTRA_IQR_OGL_{args.testset}/cIoU, auc N/A N/A")


if __name__ == "__main__":
    main()
