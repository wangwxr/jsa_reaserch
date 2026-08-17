"""No-slot evaluation using the unchanged JSA cIoU/AUC protocol."""

import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

import utils


def _stats(evaluator):
    return evaluator.finalize_AP50(), evaluator.finalize_AUC()


def _save_sorted_metrics(evaluator, output_path):
    metrics = []
    for filename, ciou in evaluator.file_ciou.items():
        metrics.append(
            (filename, ciou, evaluator.infer_ratio[filename])
        )
    metrics.sort(key=lambda item: item[1], reverse=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("Filename\tcIoU\tInfer Ratio\n")
        for filename, ciou, infer_ratio in metrics:
            handle.write(f"{filename}\t{ciou:.4f}\t{infer_ratio:.4f}\n")


@torch.no_grad()
def evaluate_no_slot(
    test_loader,
    audio_visual_model,
    object_saliency_model,
    args,
    output_dir=None,
):
    audio_visual_model.eval()
    object_saliency_model.eval()

    evaluator_aud = utils.Evaluator()
    evaluator_obj_prior = utils.Evaluator()
    evaluator_ogl = utils.Evaluator()

    for image, spec, bboxes, names, _ in tqdm(test_loader):
        if image.ndim == 3:
            image = image.unsqueeze(0)
            spec = spec.unsqueeze(0)
            bboxes = bboxes.unsqueeze(0)

        if image.ndim == 5:
            batch_size, num_views, channels, height, width = image.shape
            image = image.reshape(
                batch_size * num_views, channels, height, width
            )
            _, _, channels, frequency, time_steps = spec.shape
            spec = spec.reshape(
                batch_size * num_views,
                channels,
                frequency,
                time_steps,
            )
            _, _, channels, height, width = bboxes.shape
            bboxes = bboxes.reshape(
                batch_size * num_views, channels, height, width
            ).squeeze(1)
            names = [
                sample_name
                for sample_name in names
                for _ in range(num_views)
            ]

        if args.gpu is not None:
            image = image.cuda(args.gpu, non_blocking=True)
            spec = spec.cuda(args.gpu, non_blocking=True)

        heatmap_aud = audio_visual_model(image.float(), spec.float())
        heatmap_obj_prior = object_saliency_model(image)

        heatmap_aud = F.interpolate(
            heatmap_aud,
            size=(224, 224),
            mode="bicubic",
            align_corners=False,
        ).cpu().numpy()
        heatmap_obj_prior = F.interpolate(
            heatmap_obj_prior,
            size=(224, 224),
            mode="bicubic",
            align_corners=False,
        ).cpu().numpy()
        bboxes = bboxes.cpu().numpy()

        for index in range(spec.shape[0]):
            pred_aud = utils.normalize_img(heatmap_aud[index, 0])
            pred_obj_prior = utils.normalize_img(
                heatmap_obj_prior[index, 0]
            )
            pred_ogl = utils.normalize_img(
                pred_aud * args.alpha
                + pred_obj_prior * (1 - args.alpha)
            )
            gt_map = bboxes[index]
            threshold = 0.6

            evaluator_aud.cal_CIOU(
                pred_aud, gt_map, names[index], threshold
            )
            evaluator_obj_prior.cal_CIOU(
                pred_obj_prior, gt_map, names[index], threshold
            )
            evaluator_ogl.cal_CIOU(
                pred_ogl, gt_map, names[index], threshold
            )

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        _save_sorted_metrics(
            evaluator_aud, os.path.join(output_dir, "aud_metrics.txt")
        )
        _save_sorted_metrics(
            evaluator_obj_prior,
            os.path.join(output_dir, "obj_prior_metrics.txt"),
        )
        _save_sorted_metrics(
            evaluator_ogl, os.path.join(output_dir, "ogl_metrics.txt")
        )

    return {
        "aud": _stats(evaluator_aud),
        "obj_prior": _stats(evaluator_obj_prior),
        "ogl": _stats(evaluator_ogl),
    }
