"""Auditable crop/resize/flip geometry shared by image and heatmap warps."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def sample_random_resized_crop(
    batch_size: int,
    image_height: int,
    image_width: int,
    device: torch.device,
    scale: tuple[float, float] = (0.6, 1.0),
    ratio: tuple[float, float] = (0.9, 1.1),
    flip_probability: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Match torchvision RandomResizedCrop.get_params while retaining parameters."""
    area = image_height * image_width
    log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
    tops: list[int] = []
    lefts: list[int] = []
    heights: list[int] = []
    widths: list[int] = []

    for _ in range(batch_size):
        crop = None
        for _attempt in range(10):
            target_area = area * float(torch.empty(1).uniform_(*scale))
            aspect_ratio = math.exp(float(torch.empty(1).uniform_(*log_ratio)))
            crop_width = int(round(math.sqrt(target_area * aspect_ratio)))
            crop_height = int(round(math.sqrt(target_area / aspect_ratio)))
            if 0 < crop_width <= image_width and 0 < crop_height <= image_height:
                top = int(torch.randint(0, image_height - crop_height + 1, size=(1,)))
                left = int(torch.randint(0, image_width - crop_width + 1, size=(1,)))
                crop = (top, left, crop_height, crop_width)
                break

        if crop is None:
            input_ratio = image_width / image_height
            if input_ratio < ratio[0]:
                crop_width = image_width
                crop_height = int(round(crop_width / ratio[0]))
            elif input_ratio > ratio[1]:
                crop_height = image_height
                crop_width = int(round(crop_height * ratio[1]))
            else:
                crop_height = image_height
                crop_width = image_width
            top = (image_height - crop_height) // 2
            left = (image_width - crop_width) // 2
            crop = (top, left, crop_height, crop_width)

        top, left, crop_height, crop_width = crop
        tops.append(top)
        lefts.append(left)
        heights.append(crop_height)
        widths.append(crop_width)

    return {
        "crop_top": torch.tensor(tops, device=device, dtype=torch.float32),
        "crop_left": torch.tensor(lefts, device=device, dtype=torch.float32),
        "crop_height": torch.tensor(heights, device=device, dtype=torch.float32),
        "crop_width": torch.tensor(widths, device=device, dtype=torch.float32),
        "flipped": torch.rand(batch_size, device=device) < flip_probability,
        "original_height": torch.full(
            (batch_size,), image_height, device=device, dtype=torch.float32
        ),
        "original_width": torch.full(
            (batch_size,), image_width, device=device, dtype=torch.float32
        ),
    }


def explicit_geometry(
    top: list[int],
    left: list[int],
    height: list[int],
    width: list[int],
    flipped: list[bool],
    original_height: int,
    original_width: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Construct deterministic geometry for sanity checks."""
    batch_size = len(top)
    if not all(len(values) == batch_size for values in (left, height, width, flipped)):
        raise ValueError("Every geometry list must have the same length")
    return {
        "crop_top": torch.tensor(top, device=device, dtype=torch.float32),
        "crop_left": torch.tensor(left, device=device, dtype=torch.float32),
        "crop_height": torch.tensor(height, device=device, dtype=torch.float32),
        "crop_width": torch.tensor(width, device=device, dtype=torch.float32),
        "flipped": torch.tensor(flipped, device=device, dtype=torch.bool),
        "original_height": torch.full(
            (batch_size,), original_height, device=device, dtype=torch.float32
        ),
        "original_width": torch.full(
            (batch_size,), original_width, device=device, dtype=torch.float32
        ),
    }


def _axis_centers(length: int, device: torch.device) -> torch.Tensor:
    return (torch.arange(length, device=device, dtype=torch.float32) + 0.5) / length


def forward_grid(
    geometry: dict[str, torch.Tensor], output_height: int, output_width: int
) -> torch.Tensor:
    """Grid mapping output View B pixels/tokens into View A coordinates."""
    batch_size = geometry["crop_top"].shape[0]
    device = geometry["crop_top"].device
    x_fraction = _axis_centers(output_width, device)[None, :].expand(batch_size, -1)
    x_fraction = torch.where(
        geometry["flipped"][:, None], 1.0 - x_fraction, x_fraction
    )
    y_fraction = _axis_centers(output_height, device)[None, :].expand(batch_size, -1)

    x_boundary = geometry["crop_left"][:, None] + (
        x_fraction * geometry["crop_width"][:, None]
    )
    y_boundary = geometry["crop_top"][:, None] + (
        y_fraction * geometry["crop_height"][:, None]
    )
    x_normalized = 2.0 * x_boundary / geometry["original_width"][:, None] - 1.0
    y_normalized = 2.0 * y_boundary / geometry["original_height"][:, None] - 1.0

    return torch.stack(
        (
            x_normalized[:, None, :].expand(-1, output_height, -1),
            y_normalized[:, :, None].expand(-1, -1, output_width),
        ),
        dim=-1,
    )


def apply_to_view_a(
    view_a: torch.Tensor, geometry: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Create crop-resized/flipped View B from View A in one operation."""
    grid = forward_grid(geometry, view_a.shape[-2], view_a.shape[-1])
    return F.grid_sample(
        view_a,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )


def inverse_grid(
    geometry: dict[str, torch.Tensor], output_height: int, output_width: int
) -> torch.Tensor:
    """Grid mapping View A token centers into final View B coordinates."""
    batch_size = geometry["crop_top"].shape[0]
    device = geometry["crop_top"].device
    x_a_boundary = (
        _axis_centers(output_width, device)[None, :]
        * geometry["original_width"][:, None]
    )
    y_a_boundary = (
        _axis_centers(output_height, device)[None, :]
        * geometry["original_height"][:, None]
    )
    x_fraction = (
        x_a_boundary - geometry["crop_left"][:, None]
    ) / geometry["crop_width"][:, None]
    y_fraction = (
        y_a_boundary - geometry["crop_top"][:, None]
    ) / geometry["crop_height"][:, None]
    x_normalized = 2.0 * x_fraction - 1.0
    x_normalized = torch.where(
        geometry["flipped"][:, None], -x_normalized, x_normalized
    )
    y_normalized = 2.0 * y_fraction - 1.0
    return torch.stack(
        (
            x_normalized[:, None, :].expand(-1, output_height, -1),
            y_normalized[:, :, None].expand(-1, -1, output_width),
        ),
        dim=-1,
    )


def warp_view_b_to_a(
    view_b_map: torch.Tensor,
    geometry: dict[str, torch.Tensor],
    output_size: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse-warp B to A and return a conservative full-support valid mask."""
    if output_size is None:
        output_size = view_b_map.shape[-2:]
    output_height, output_width = output_size
    grid = inverse_grid(geometry, output_height, output_width)
    warped = F.grid_sample(
        view_b_map,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )

    # Exclude locations whose bilinear footprint touches padding. This is a
    # conservative subset of the true crop overlap and prevents border zeros
    # from entering KL normalization.
    x_limit = 1.0 - 1.0 / view_b_map.shape[-1] + 1e-6
    y_limit = 1.0 - 1.0 / view_b_map.shape[-2] + 1e-6
    valid = (
        (grid[..., 0].abs() <= x_limit)
        & (grid[..., 1].abs() <= y_limit)
    ).unsqueeze(1)
    return warped, valid.to(dtype=view_b_map.dtype)


def crop_scale(geometry: dict[str, torch.Tensor]) -> torch.Tensor:
    return (
        geometry["crop_height"]
        * geometry["crop_width"]
        / (geometry["original_height"] * geometry["original_width"])
    )


def geometry_records(
    geometry: dict[str, torch.Tensor], limit: int | None = None
) -> list[dict[str, int | float | bool]]:
    count = geometry["crop_top"].shape[0]
    if limit is not None:
        count = min(count, limit)
    scales = crop_scale(geometry).detach().cpu()
    records = []
    for index in range(count):
        records.append(
            {
                "sample_index": index,
                "crop_top": int(geometry["crop_top"][index]),
                "crop_left": int(geometry["crop_left"][index]),
                "crop_height": int(geometry["crop_height"][index]),
                "crop_width": int(geometry["crop_width"][index]),
                "flipped": bool(geometry["flipped"][index]),
                "original_height": int(geometry["original_height"][index]),
                "original_width": int(geometry["original_width"][index]),
                "crop_scale": float(scales[index]),
            }
        )
    return records
