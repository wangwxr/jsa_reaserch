#!/usr/bin/env python3
"""Summarize frozen 8.7 maps across 10k and 144k runs (no training)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = HERE.parents[2]
MECH = EXP / "mechanism_audit"
HARD = EXP / "8.7_hard_negative_cram"
EXP80 = ROOT / "chuagnxindian/8.0_audio_only_residual_audit"
EXP41 = ROOT / "chuagnxindian/4.1_selective_fusion_capacity_evidence_probe"

sys.path.insert(0, str(MECH / "scripts"))
import analyze as A  # noqa: E402


def normalize(values):
    lo = values.min(axis=(-2, -1), keepdims=True)
    hi = values.max(axis=(-2, -1), keepdims=True)
    return (values - lo) / np.maximum(hi - lo, 1e-12)


def map_from_npz(path, image_path=None):
    with np.load(path, allow_pickle=False) as z:
        result = {"ids": z["ids"].astype(str), "matched": z["H_matched"].copy(), "wrong": z["H_wrong"].copy(), "wrong_indices": z["wrong_indices"].copy()}
        if "IMG_QUERY" in z.files:
            result["image"] = z["IMG_QUERY"].copy()
    if image_path is not None:
        with np.load(image_path, allow_pickle=False) as z:
            result["image"] = z["image_native"].reshape(len(result["ids"]), 49).copy()
    return result


def maps(scale, dataset, method):
    raw = HERE / "raw_maps" / scale / dataset / method / "maps.npz"
    if raw.is_file():
        return map_from_npz(raw)
    if scale != "144k":
        raise FileNotFoundError(raw)
    if method == "baseline":
        path = MECH / f"results/{dataset}/full/original_1.3g_final_maps.npz"
        out = map_from_npz(path)
        with np.load(EXP41 / f"results/{dataset}_144k/raw_maps.npz", allow_pickle=False) as z:
            if not np.array_equal(out["ids"], z["sample_id"].astype(str)):
                raise RuntimeError("baseline image-map IDs mismatch")
            out["image"] = z["IMG_L4"].reshape(len(out["ids"]), 49).copy()
        return out
    if method == "mean":
        return map_from_npz(MECH / f"results/{dataset}/full/cram_c3_stage2_maps.npz", EXP / f"natural_localization/{dataset}/C3/seed12345/natural_maps.npz")
    if method == "hard_default":
        return map_from_npz(HARD / f"mechanism/{dataset}/hard_maps.npz")
    raise ValueError((scale, dataset, method))


def stored_formal_metrics(scale, dataset, method):
    if method == "baseline":
        suffix = f"{dataset}_{scale}_best" if dataset == "vggss" else f"flickr_{scale}_frame8_center5_best"
        path = ROOT / "checkpoints" / f"1.3G-multigeom_equivariant_l3_refine_{suffix}" / "best_full_six_metrics.json"
        return json.loads(path.read_text())["metrics"]
    return None


def transition(reference, candidate, gt):
    base = reference >= 0.6; current = candidate >= 0.6; fg = gt >= 0.5
    add = current & ~base; remove = base & ~current
    return {
        "RemoveFP": (remove & ~fg).sum((1, 2)), "RemoveTP": (remove & fg).sum((1, 2)),
        "AddTP": (add & fg).sum((1, 2)), "AddFP": (add & ~fg).sum((1, 2)),
    }


def compact_stats(values, *seed):
    return A.stats(np.asarray(values, dtype=float), *[str(x) for x in seed])


def paired_rows(scale, dataset, natural, six):
    methods = list(natural.model.drop_duplicates())
    comparisons = [("Mean-minus-Baseline", "mean", "baseline"), ("Hard-default-minus-Mean", "hard_default", "mean")]
    if "hard_adjusted" in methods:
        comparisons += [("Hard-adjusted-minus-Mean", "hard_adjusted", "mean"), ("Hard-adjusted-minus-Hard-default", "hard_adjusted", "hard_default")]
    rows = []
    for label, candidate, reference in comparisons:
        a = six[(six.model == candidate) & (six.readout == "AUD")].sort_values("sample_index")
        b = six[(six.model == reference) & (six.readout == "AUD")].sort_values("sample_index")
        if not np.array_equal(a.sample_id.to_numpy(), b.sample_id.to_numpy()):
            raise RuntimeError("paired IDs differ")
        delta = a.iou.to_numpy() - b.iou.to_numpy()
        item = {"scale": scale, "dataset": dataset, "comparison": label, "candidate": candidate, "reference": reference,
                "delta_cIoU": float((a.iou.to_numpy() >= .5).mean() - (b.iou.to_numpy() >= .5).mean()),
                "success_count_change": int((a.iou.to_numpy() >= .5).sum() - (b.iou.to_numpy() >= .5).sum()),
                "wins": int((delta > 1e-12).sum()), "ties": int((np.abs(delta) <= 1e-12).sum()), "losses": int((delta < -1e-12).sum())}
        item.update(compact_stats(delta, scale, dataset, label, "iou_delta"))
        rows.append(item)
    return pd.DataFrame(rows)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    all_six = []; all_natural = []; all_cf = []; all_pairs = []; all_transition = []
    for scale in ("10k", "144k"):
        for dataset in ("vggss", "flickr"):
            method_names = ["baseline", "mean", "hard_default"] + (["hard_adjusted"] if scale == "144k" else [])
            collection = {name: maps(scale, dataset, name) for name in method_names}
            ids = collection["baseline"]["ids"]
            wrong = collection["baseline"]["wrong_indices"]
            for name, value in collection.items():
                if not np.array_equal(ids, value["ids"]) or not np.array_equal(wrong, value["wrong_indices"]):
                    raise RuntimeError(f"{scale}/{dataset}/{name}: IDs or swaps mismatch")
            with np.load(EXP / f"natural_localization/{dataset}/C3/seed12345/natural_maps.npz", allow_pickle=False) as z:
                if not np.array_equal(ids, z["ids"].astype(str)):
                    raise RuntimeError("GT IDs mismatch")
                gt = z["gt_masks"].astype(np.float32)
            with np.load(EXP80 / f"results/{dataset}_144k_sample_indices.npz", allow_pickle=False) as z:
                fixed = {k: z[k].copy() for k in z.files}
            if not np.array_equal(ids, fixed["ids"].astype(str)):
                raise RuntimeError("region IDs mismatch")
            # The frozen audit utility names its transition reference
            # ``original_1.3g_final``.  Retain that internal label only for the
            # calculation, then restore the cross-scale table's ``baseline``.
            audit_names = ["original_1.3g_final", *[name for name in method_names if name != "baseline"]]
            audit_collection = {"original_1.3g_final": collection["baseline"]} | {name: collection[name] for name in method_names if name != "baseline"}
            A.MODELS = tuple(audit_names)
            natural, audit_normed = A.natural_and_response_metrics(dataset, audit_collection, gt, fixed, device)
            natural["model"] = natural["model"].replace({"original_1.3g_final": "baseline"})
            normed = {"baseline": audit_normed["original_1.3g_final"]} | {name: audit_normed[name] for name in method_names if name != "baseline"}
            natural["scale"] = scale
            cf = pd.concat([A.counterfactual_metrics(dataset, name, collection[name]) for name in method_names], ignore_index=True)
            cf["scale"] = scale
            for name in method_names:
                _, image = A.resize_and_normalize(collection[name]["image"], device)
                _, prior = A.resize_and_normalize(np.load(EXP / f"stage_specific_audit/degradation_path/{dataset}/object_prior.npz")["maps"], device)
                aud = normed[name]
                readouts = {"AUD": aud, "IMG_QUERY": image, "IQR": normalize(.6 * aud + .4 * image), "OBJ_PRIOR": prior,
                            "OGL": normalize(.6 * aud + .4 * prior), "EXTRA_IQR_OGL": normalize(.6 * aud + .2 * image + .2 * prior)}
                for readout, values in readouts.items():
                    iou = np.asarray([A.official_iou(values[i] >= .6, gt[i]) for i in range(len(ids))])
                    all_six.extend({"scale": scale, "dataset": dataset, "model": name, "readout": readout, "sample_id": ids[i], "sample_index": i, "iou": iou[i]} for i in range(len(ids)))
            for name in method_names:
                if name != "mean":
                    t = transition(normed["mean"], normed[name], gt)
                    all_transition.extend({"scale": scale, "dataset": dataset, "model": name, "sample_id": ids[i], "sample_index": i, **{key: int(value[i]) for key, value in t.items()}} for i in range(len(ids)))
            all_natural.append(natural); all_cf.append(cf)
    six = pd.DataFrame(all_six); natural = pd.concat(all_natural, ignore_index=True); cf = pd.concat(all_cf, ignore_index=True); transitions = pd.DataFrame(all_transition)
    # Exact stored values are retained as an explicit validation field; the unified map replay is the paired source.
    perf = []
    for (scale, dataset, method, readout), frame in six.groupby(["scale", "dataset", "model", "readout"], sort=False):
        iou = frame.sort_values("sample_index").iou.to_numpy()
        thresholds = np.arange(21) * .05
        item = {"scale": scale, "dataset": dataset, "model": method, "readout": readout,
                "cIoU": float((iou >= .5).mean()), "AUC": float(np.trapezoid([(iou >= t).mean() for t in thresholds], thresholds)), "n": len(iou)}
        formal = stored_formal_metrics(scale, dataset, method)
        if formal is not None:
            item["stored_cIoU"] = formal[readout]["cIoU"]; item["stored_AUC"] = formal[readout]["AUC"]
            item["replay_cIoU_error"] = item["cIoU"] - item["stored_cIoU"]; item["replay_AUC_error"] = item["AUC"] - item["stored_AUC"]
        perf.append(item)
    performance = pd.DataFrame(perf)
    aud = performance[performance.readout.eq("AUD")].set_index(["scale", "dataset", "model"])
    ogl = performance[performance.readout.eq("OGL")].set_index(["scale", "dataset", "model"])
    performance["Delta_OGL"] = [aud.loc[(r.scale, r.dataset, r.model), "cIoU"] - ogl.loc[(r.scale, r.dataset, r.model), "cIoU"] if r.readout == "AUD" else np.nan for r in performance.itertuples()]
    performance["Delta_OGL_AUC"] = [aud.loc[(r.scale, r.dataset, r.model), "AUC"] - ogl.loc[(r.scale, r.dataset, r.model), "AUC"] if r.readout == "AUD" else np.nan for r in performance.itertuples()]
    for scale in ("10k", "144k"):
        for dataset in ("vggss", "flickr"):
            all_pairs.append(paired_rows(scale, dataset, natural[(natural.scale == scale) & (natural.dataset == dataset)], six[(six.scale == scale) & (six.dataset == dataset)]))
    pairs = pd.concat(all_pairs, ignore_index=True)
    # Mechanism global summary plus paired Hard-vs-Mean CIs for all requested metrics.
    mechanism_cols = ["gt_response", "nearfp_response", "gt_nearfp_gap", "gt_vs_nearfp_auroc", "coverage", "precision", "pred_area_ratio", "RemoveFP", "RemoveTP", "AddTP", "AddFP"]
    mechanism_rows = []
    for (scale, dataset, model), frame in natural.groupby(["scale", "dataset", "model"], sort=False):
        row = {"scale": scale, "dataset": dataset, "model": model}
        for metric in mechanism_cols: row[metric] = float(frame[metric].mean())
        mechanism_rows.append(row)
    mechanism = pd.DataFrame(mechanism_rows).rename(columns={"pred_area_ratio": "activated_area"})
    paired_mech = []
    for (scale, dataset), frame in natural.groupby(["scale", "dataset"], sort=False):
        mean = frame[frame.model.eq("mean")].sort_values("sample_index")
        for hard in [x for x in ("hard_default", "hard_adjusted") if x in set(frame.model)]:
            candidate = frame[frame.model.eq(hard)].sort_values("sample_index")
            for metric in mechanism_cols[:7]:
                paired_mech.append({"scale": scale, "dataset": dataset, "comparison": f"{hard}-minus-mean", "metric": metric, **compact_stats(candidate[metric].to_numpy() - mean[metric].to_numpy(), scale, dataset, hard, metric)})
    pd.DataFrame(paired_mech).to_csv(HERE / "mechanism/paired_hard_vs_mean.csv", index=False)
    six.to_csv(HERE / "performance/six_readout_per_sample.csv", index=False)
    performance.to_csv(HERE / "performance/all_readouts.csv", index=False)
    pairs.to_csv(HERE / "performance/aud_pairwise.csv", index=False)
    natural.to_csv(HERE / "mechanism/object_per_sample_all.csv", index=False)
    cf.to_csv(HERE / "mechanism/audio_counterfactual_per_sample_all.csv", index=False)
    mechanism.to_csv(HERE / "mechanism/mechanism_all.csv", index=False)
    transitions.to_csv(HERE / "mechanism/hard_minus_mean_transition_per_sample.csv", index=False)
    print(performance.to_string(index=False))


if __name__ == "__main__":
    main()
