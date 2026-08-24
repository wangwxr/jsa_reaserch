#!/usr/bin/env python3
"""Pure schedule and factorized-loss regression tests required before 6.1."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from common import EXPERIMENT_ROOT, setup_paths, write_json

setup_paths()

from model import LossWeights, compose_total
from schedule import lambda_att_space


EXPECTED = {
    "early_high_late_low": {1: 100.0, 20: 100.0, 21: 95.5, 30: 55.0, 40: 10.0, 41: 10.0, 100: 10.0},
    "early_low_late_high": {1: 0.0, 20: 0.0, 21: 5.0, 30: 50.0, 40: 100.0, 41: 100.0, 100: 100.0},
}


def scalar_losses(requires_grad=True):
    names = ("info", "rec_img", "rec_aud", "div_img", "div_aud", "att_space", "att_time")
    values = (0.91, 1.17, 1.31, 0.21, 0.29, 0.0043, 0.0061)
    return {
        name: torch.tensor(value, dtype=torch.float32, requires_grad=requires_grad)
        for name, value in zip(names, values)
    }


def run_tests():
    schedule_checks = {}
    for name, expected in EXPECTED.items():
        actual = {epoch: lambda_att_space(name, epoch) for epoch in expected}
        assert actual == expected, (name, actual, expected)
        schedule_checks[name] = actual

    losses = scalar_losses()
    full = LossWeights()
    new_full = compose_total(losses, full)
    old_full = (
        losses["info"]
        + 0.1 * (losses["rec_img"] + losses["rec_aud"])
        + 0.1 * (losses["div_img"] + losses["div_aud"])
        + 100.0 * (losses["att_space"] + losses["att_time"])
    )
    full_error = float(torch.max(torch.abs(new_full - old_full)))
    assert full_error <= 1e-7, full_error

    zero_weights = LossWeights(att_space=0.0)
    zero_total = compose_total(losses, zero_weights)
    weighted_space = losses["att_space"] * zero_weights.att_space
    weighted_grad = torch.autograd.grad(weighted_space, losses["att_space"], retain_graph=True)[0]
    assert float(weighted_space) == 0.0
    assert float(weighted_grad) == 0.0
    total_space_grad = torch.autograd.grad(zero_total, losses["att_space"])[0]
    assert float(total_space_grad) == 0.0

    for schedule_name in EXPECTED:
        for epoch in range(1, 101):
            weights = LossWeights(att_space=lambda_att_space(schedule_name, epoch))
            assert weights.rec_img == weights.rec_aud == 0.1
            assert weights.div_img == weights.div_aud == 0.1
            assert weights.att_time == 100.0

    result = {
        "passed": True,
        "schedule_points": schedule_checks,
        "full_total_max_abs_error": full_error,
        "lambda_zero_weighted_value": float(weighted_space),
        "lambda_zero_weighted_gradient": float(weighted_grad),
        "lambda_zero_total_gradient": float(total_space_grad),
        "fixed_weights_verified_all_epochs": True,
        "scope": "The schedule function returns only one scalar; forward/data/model are untouched.",
    }
    write_json(EXPERIMENT_ROOT / "results" / "preflight" / "schedule_tests.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    run_tests()
