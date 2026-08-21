"""Shared paths, settings, and result helpers for Experiment 3.1."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
BASELINE_ROOT = PROJECT_ROOT / "chuagnxindian" / "mufasa_ablation2_l3_l4_ablation"
V11_ROOT = PROJECT_ROOT / "chuagnxindian" / "1mufasaslot"
CHECKPOINT_ROOT = HERE / "checkpoints"
RESULTS_ROOT = HERE / "results"

SETTINGS = {
    "vggss_10k": {
        "dataset": "vggss",
        "split": "10k",
        "baseline_experiment": "mufasa_ablation2_l3_l4_ablation_vggss_10k",
        "experiment": "3.1_hierarchical_audio_representation_vggss_10k",
        "checkpoint": "vggss_best.pth",
    },
    "vggss_144k": {
        "dataset": "vggss",
        "split": "144k",
        "baseline_experiment": "mufasa_ablation2_l3_l4_ablation_vggss_144k",
        "experiment": "3.1_hierarchical_audio_representation_vggss_144k",
        "checkpoint": "vggss_best.pth",
    },
    "flickr_10k": {
        "dataset": "flickr",
        "split": "10k",
        "baseline_experiment": "mufasa_ablation2_l3_l4_ablation_flickr_10k_frame8_center5",
        "experiment": "3.1_hierarchical_audio_representation_flickr_10k_frame8_center5",
        "checkpoint": "flickr_best.pth",
    },
    "flickr_144k": {
        "dataset": "flickr",
        "split": "144k",
        "baseline_experiment": "mufasa_ablation2_l3_l4_ablation_flickr_144k_frame8_center5",
        "experiment": "3.1_hierarchical_audio_representation_flickr_144k_frame8_center5",
        "checkpoint": "flickr_best.pth",
    },
}

METRIC_NAMES = (
    "aud",
    "img_query",
    "iqr",
    "obj_prior",
    "ogl",
    "extra_iqr_ogl",
)


def registry(setting: str) -> dict[str, Any]:
    if setting not in SETTINGS:
        raise KeyError(f"Unknown setting {setting!r}; expected one of {sorted(SETTINGS)}")
    return SETTINGS[setting]


def baseline_dir(setting: str) -> Path:
    return PROJECT_ROOT / "checkpoints" / registry(setting)["baseline_experiment"]


def experiment_dir(setting: str) -> Path:
    return CHECKPOINT_ROOT / registry(setting)["experiment"]


def result_dir(setting: str) -> Path:
    return RESULTS_ROOT / setting


def load_baseline_config(setting: str, gpu: int | None = None) -> argparse.Namespace:
    entry = registry(setting)
    path = baseline_dir(setting) / "configs.json"
    values = json.loads(path.read_text(encoding="utf-8"))
    values.update(
        {
            "architecture": "3.1_hierarchical_audio_representation",
            "baseline_experiment": entry["baseline_experiment"],
            "checkpoint_selection": "IQR_cIoU",
            "experiment_name": entry["experiment"],
            "model_dir": str(CHECKPOINT_ROOT),
            "spatial_attention": "l4_only_a4_audio_query",
            "audio_semantic_representation": "a3_a4_residual_fusion",
            "audio_localization_representation": "a4_only",
            "audio_reconstruction_representation": "a4_only",
            "audio_attention_loss_representation": "a4_only",
            "gpu": gpu if gpu is not None else values.get("gpu", 0),
            "wandb": False,
            "resume": False,
        }
    )
    return argparse.Namespace(**values)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        newline="",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def baseline_best_row(setting: str) -> dict[str, Any]:
    rows = read_csv(baseline_dir(setting) / "epoch_metrics.csv")
    if not rows:
        raise RuntimeError(f"No baseline epoch metrics for {setting}")
    best_index = max(range(len(rows)), key=lambda index: float(rows[index]["iqr_ciou"]))
    row: dict[str, Any] = {}
    for key, value in rows[best_index].items():
        try:
            row[key] = float(value)
        except (TypeError, ValueError):
            row[key] = value
    row["epoch"] = int(float(row["epoch"]))
    return row


def selected_row(path: Path) -> dict[str, Any]:
    rows = read_csv(path)
    if not rows:
        raise RuntimeError(f"No epoch metrics at {path}")
    best_index = max(range(len(rows)), key=lambda index: float(rows[index]["iqr_ciou"]))
    output: dict[str, Any] = {}
    for key, value in rows[best_index].items():
        try:
            output[key] = float(value)
        except (TypeError, ValueError):
            output[key] = value
    output["epoch"] = int(float(output["epoch"]))
    return output


def metric_block(row: dict[str, Any]) -> dict[str, dict[str, float]]:
    return {
        name.upper(): {
            "cIoU": float(row[f"{name}_ciou"]),
            "AUC": float(row[f"{name}_auc"]),
        }
        for name in METRIC_NAMES
    }


def count_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def parameter_audit(baseline, hierarchical) -> dict[str, Any]:
    baseline_names = dict(baseline.named_parameters())
    hierarchical_names = dict(hierarchical.named_parameters())
    added = {
        name: parameter.numel()
        for name, parameter in hierarchical_names.items()
        if name not in baseline_names
    }
    removed = sorted(set(baseline_names) - set(hierarchical_names))
    changed = {
        name: {
            "baseline": baseline_names[name].numel(),
            "hierarchical": hierarchical_names[name].numel(),
        }
        for name in set(baseline_names) & set(hierarchical_names)
        if baseline_names[name].shape != hierarchical_names[name].shape
    }
    baseline_total = count_parameters(baseline)
    hierarchical_total = count_parameters(hierarchical)
    added_total = sum(added.values())
    return {
        "baseline_total_params": baseline_total,
        "hierarchical_total_params": hierarchical_total,
        "added_params": added_total,
        "added_param_percent": 100.0 * added_total / baseline_total,
        "added_parameter_names": added,
        "removed_parameter_names": removed,
        "changed_parameter_shapes": changed,
        "passed": (
            hierarchical_total - baseline_total == added_total
            and not removed
            and not changed
            and all(
                name.startswith(
                    (
                        "audnet.aud_proj3.",
                        "slot_attn.audio_branch_a3.",
                        "slot_attn.audio_hierarchical_fusion.",
                    )
                )
                for name in added
            )
        ),
    }
