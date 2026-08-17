#!/usr/bin/env python3
"""L4-key evaluation entry reusing the current JSA evaluator unchanged."""

import os
import sys
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXPERIMENT_DIR))

import test_model  # noqa: E402
from model_mufasa_jsa import MUFASAJSA  # noqa: E402


def main():
    test_model.model_slot.mymodel = MUFASAJSA
    args = test_model.get_arguments()
    architecture = getattr(args, "architecture", None)
    if architecture != "mufasa_jsa_v1":
        raise RuntimeError(
            "This entry only tests MUFASA-JSA v1 checkpoints; "
            f"configs.json has architecture={architecture!r}"
        )
    if getattr(args, "model", "jsa") != "jsa":
        raise ValueError("MUFASA-JSA v1 testing requires model='jsa'")

    eval_attention_mode = os.environ.get(
        "MUFASA_EVAL_ATTENTION_MODE", "l4_only"
    )
    if eval_attention_mode not in {"l4_only", "fused_query_l4"}:
        raise ValueError(
            "MUFASA_EVAL_ATTENTION_MODE must be 'l4_only' or "
            f"'fused_query_l4', got {eval_attention_mode!r}"
        )
    args.eval_attention_mode = eval_attention_mode

    if eval_attention_mode == "fused_query_l4":
        original_print = print

        def print_fused_query_labels(*values, **kwargs):
            if values and isinstance(values[0], str):
                label = values[0]
                if label.startswith("IMG_QUERY_"):
                    label = label.replace(
                        "IMG_QUERY_", "IMG_FUSED_QUERY_", 1
                    )
                elif label.startswith("IQR_"):
                    label = label.replace("IQR_", "IQR_NEW_", 1)
                values = (label, *values[1:])
            original_print(*values, **kwargs)

        # Change display labels only; validate_img_aud and all metrics stay intact.
        test_model.print = print_fused_query_labels

    original_save_all_metrics = test_model.save_all_metrics

    def save_l4_metrics(*save_args):
        metric_args = save_args[:-1]
        experiment_output_dir = save_args[-1]
        l4_output_dir = os.path.join(
            experiment_output_dir, f"eval_{eval_attention_mode}"
        )
        return original_save_all_metrics(*metric_args, l4_output_dir)

    test_model.save_all_metrics = save_l4_metrics
    test_model.main(args)


if __name__ == "__main__":
    main()
