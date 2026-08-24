"""Epoch-wise spatial-attention schedules for Experiment 6.1.

The public API uses human-readable one-based epochs.  Keeping the conversion
at the training-loop boundary prevents the 20/21 transition from drifting.
"""

from __future__ import annotations


SCHEDULES = ("early_high_late_low", "early_low_late_high")


def lambda_att_space(schedule: str, epoch: int) -> float:
    """Return the spatial-attention coefficient for one-based ``epoch``."""
    if schedule not in SCHEDULES:
        raise ValueError(f"Unknown schedule {schedule!r}; expected one of {SCHEDULES}")
    if not 1 <= epoch <= 100:
        raise ValueError(f"epoch must be in [1, 100], got {epoch}")

    if schedule == "early_high_late_low":
        if epoch <= 20:
            return 100.0
        if epoch <= 40:
            return 100.0 - 4.5 * (epoch - 20)
        return 10.0

    if epoch <= 20:
        return 0.0
    if epoch <= 40:
        return 5.0 * (epoch - 20)
    return 100.0


def schedule_table(schedule: str) -> list[dict[str, float]]:
    return [
        {"epoch": epoch, "lambda_att_space": lambda_att_space(schedule, epoch)}
        for epoch in range(1, 101)
    ]
