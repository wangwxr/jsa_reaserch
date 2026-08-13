#!/usr/bin/env python3
"""Diagnose whether JSA localization collapses or swaps foreground slots."""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import model_slot
import utils
from dataset import get_test_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name")
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--model-dir", default="./checkpoints")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--infer-sharpening", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def setup_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def all_slot_maps(model, image, audio):
    img = model.imgnet(image)
    aud = model.audnet(audio)
    batch, channels, height, width = img.shape
    img = img.reshape(batch, channels, height * width).permute(0, 2, 1)
    aud = aud.permute(0, 2, 1)
    img_attn, cross_attn, img_q, aud_q, img_k, _ = model.slot_attn.get_cross_attn(
        img, aud
    )
    scale = 512 ** -0.5
    img_logits = torch.einsum("bid,bjd->bij", img_q, img_k) * scale
    aud_logits = torch.einsum("bid,bjd->bij", aud_q, img_k) * scale
    return (
        img_attn.reshape(batch, model.num_slots, height, width),
        cross_attn.reshape(batch, model.num_slots, height, width),
        img_logits,
        aud_logits,
    )


def normalized_map(array):
    return utils.normalize_img(array)


@torch.no_grad()
def evaluate(model, loader, config):
    methods = ("aud", "img_query", "iqr")
    evaluators = [
        {name: utils.Evaluator() for name in methods}
        for _ in range(model.num_slots)
    ]
    argmax_cells = [np.zeros(49, dtype=np.int64) for _ in range(model.num_slots)]
    img_logit_margins = []
    aud_logit_margins = []

    model.eval()
    for image, audio, bboxes, names, _ in tqdm(loader, leave=False):
        image = image.cuda(config.gpu, non_blocking=True).float()
        audio = audio.cuda(config.gpu, non_blocking=True).float()
        img_maps, aud_maps, img_logits, aud_logits = all_slot_maps(
            model, image, audio
        )
        img_logit_margins.append(
            (img_logits[:, 0] - img_logits[:, 1]).abs().cpu().numpy().ravel()
        )
        aud_logit_margins.append(
            (aud_logits[:, 0] - aud_logits[:, 1]).abs().cpu().numpy().ravel()
        )
        img_maps = F.interpolate(
            img_maps, size=(224, 224), mode="bicubic", align_corners=False
        ).cpu().numpy()
        aud_maps = F.interpolate(
            aud_maps, size=(224, 224), mode="bicubic", align_corners=False
        ).cpu().numpy()
        bboxes = bboxes.numpy()

        for sample_index, name in enumerate(names):
            for slot in range(model.num_slots):
                pred_aud = normalized_map(aud_maps[sample_index, slot])
                pred_img = normalized_map(img_maps[sample_index, slot])
                pred_iqr = normalized_map(
                    pred_aud * config.alpha + pred_img * (1.0 - config.alpha)
                )
                predictions = {
                    "aud": pred_aud,
                    "img_query": pred_img,
                    "iqr": pred_iqr,
                }
                for method, prediction in predictions.items():
                    evaluators[slot][method].cal_CIOU(
                        prediction, bboxes[sample_index], name, 0.6
                    )
                coarse = aud_maps[sample_index, slot].reshape(224, 224)
                y, x = np.unravel_index(np.argmax(coarse), coarse.shape)
                argmax_cells[slot][min(y // 32, 6) * 7 + min(x // 32, 6)] += 1

    results = []
    for slot, slot_evaluators in enumerate(evaluators):
        slot_result = {"slot": slot}
        for method, evaluator in slot_evaluators.items():
            areas = np.fromiter(evaluator.infer_ratio.values(), dtype=np.float64)
            slot_result[method] = {
                "ap50": evaluator.finalize_AP50(),
                "auc": evaluator.finalize_AUC(),
                "mean_ciou": evaluator.finalize_cIoU(),
                "mean_area": float(areas.mean()),
            }
        counts = argmax_cells[slot]
        probabilities = counts[counts > 0] / counts.sum()
        entropy = -(probabilities * np.log(probabilities)).sum() / np.log(49)
        slot_result["aud_argmax_entropy"] = float(entropy)
        slot_result["aud_argmax_mode_fraction"] = float(counts.max() / counts.sum())
        results.append(slot_result)

    oracle_ciou = np.maximum(
        np.asarray(evaluators[0]["iqr"].ciou),
        np.asarray(evaluators[1]["iqr"].ciou),
    )
    oracle = {
        "iqr_ap50": float((oracle_ciou >= 0.5).mean()),
        "iqr_mean_ciou": float(oracle_ciou.mean()),
    }
    margins = {}
    for name, chunks in (
        ("img_query", img_logit_margins),
        ("aud", aud_logit_margins),
    ):
        values = np.concatenate(chunks)
        margins[name] = {
            "mean_abs_slot_logit_margin": float(values.mean()),
            "p50_abs_slot_logit_margin": float(np.percentile(values, 50)),
            "p90_abs_slot_logit_margin": float(np.percentile(values, 90)),
        }
    return results, oracle, margins


def main():
    args = parse_args()
    experiment_dir = os.path.join(args.model_dir, args.experiment_name)
    with open(os.path.join(experiment_dir, "configs.json"), encoding="utf-8") as handle:
        config = SimpleNamespace(**json.load(handle))
    config.gpu = args.gpu
    config.alpha = args.alpha
    config.infer_sharpening = args.infer_sharpening
    config.batch_size = args.batch_size
    config.workers = args.workers
    setup_seed(getattr(config, "seed", 12345))

    dataset = get_test_dataset(config, config.testset)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.workers,
        pin_memory=True,
    )

    for checkpoint_name in args.checkpoints:
        model = model_slot.mymodel(config).cuda(config.gpu)
        checkpoint_path = os.path.join(experiment_dir, checkpoint_name)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = {key.replace("module.", ""): value for key, value in checkpoint["model"].items()}
        model.load_state_dict(state)
        slots, oracle, margins = evaluate(model, loader, config)
        print(
            json.dumps(
                {
                    "checkpoint": checkpoint_name,
                    "infer_sharpening": config.infer_sharpening,
                    "slots": slots,
                    "oracle": oracle,
                    "unsharpened_slot_logit_margins": margins,
                },
                indent=2,
            )
        )
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
