"""Auditable crop/resize/flip geometry shared by image and heatmap warps."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _sample_valid_random_resized_crop(
    image_height: int,
    image_width: int,
    scale: tuple[float, float],
    ratio: tuple[float, float],
) -> tuple[int, int, int, int]:
    """Sample one geometrically valid torchvision-style RRC candidate."""
    area = image_height * image_width
    log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
    for _ in range(10):
        target_area = area * float(torch.empty(1).uniform_(*scale))
        aspect_ratio = math.exp(float(torch.empty(1).uniform_(*log_ratio)))
        crop_width = int(round(math.sqrt(target_area * aspect_ratio)))
        crop_height = int(round(math.sqrt(target_area / aspect_ratio)))
        if 0 < crop_width <= image_width and 0 < crop_height <= image_height:
            top = int(torch.randint(0, image_height - crop_height + 1, size=(1,)))
            left = int(torch.randint(0, image_width - crop_width + 1, size=(1,)))
            return top, left, crop_height, crop_width

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
    return top, left, crop_height, crop_width


@torch.no_grad()
def sample_semantic_preserving_crop(
    teacher_a: torch.Tensor,
    image_height: int,
    image_width: int,
    scale: tuple[float, float] = (0.6, 1.0),
    ratio: tuple[float, float] = (0.9, 1.1),
    flip_probability: float = 0.5,
    min_teacher_mass: float = 0.60,
    max_crop_attempts: int = 10,
) -> dict[str, torch.Tensor]:
    """Accept the first RRC retaining enough frozen-teacher sound mass."""
    if teacher_a.ndim != 4 or teacher_a.shape[1] != 1:
        raise ValueError(f"Expected teacher map [B,1,H,W], got {teacher_a.shape}")
    if not 0.0 <= min_teacher_mass <= 1.0:
        raise ValueError("min_teacher_mass must be in [0,1]")
    if max_crop_attempts < 1:
        raise ValueError("max_crop_attempts must be positive")

    device = teacher_a.device
    batch_size = teacher_a.shape[0]
    teacher = teacher_a.float().clamp_min(0)
    teacher = teacher / teacher.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    teacher_pixels = F.interpolate(
        teacher,
        size=(image_height, image_width),
        mode="bilinear",
        align_corners=False,
    )
    teacher_pixels = teacher_pixels / teacher_pixels.sum(
        dim=(-2, -1), keepdim=True
    ).clamp_min(1e-8)

    candidates = [
        [
            _sample_valid_random_resized_crop(
                image_height, image_width, scale=scale, ratio=ratio
            )
            for _ in range(max_crop_attempts)
        ]
        for _ in range(batch_size)
    ]
    tops = torch.tensor(
        [[crop[0] for crop in sample] for sample in candidates],
        device=device,
        dtype=torch.long,
    )
    lefts = torch.tensor(
        [[crop[1] for crop in sample] for sample in candidates],
        device=device,
        dtype=torch.long,
    )
    heights = torch.tensor(
        [[crop[2] for crop in sample] for sample in candidates],
        device=device,
        dtype=torch.long,
    )
    widths = torch.tensor(
        [[crop[3] for crop in sample] for sample in candidates],
        device=device,
        dtype=torch.long,
    )

    # Integral-image rectangle sums avoid a GPU synchronization per candidate.
    integral = F.pad(
        teacher_pixels[:, 0].cumsum(dim=-2).cumsum(dim=-1),
        (1, 0, 1, 0),
    )
    flat_integral = integral.flatten(start_dim=1)
    stride = image_width + 1
    bottoms = tops + heights
    rights = lefts + widths

    def gather(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return flat_integral.gather(1, y * stride + x)

    masses = (
        gather(bottoms, rights)
        - gather(tops, rights)
        - gather(bottoms, lefts)
        + gather(tops, lefts)
    )
    accepted = masses >= min_teacher_mass
    has_accepted = accepted.any(dim=1)
    first_accepted = accepted.to(torch.int64).argmax(dim=1)
    gather_index = first_accepted[:, None]

    def select(values: torch.Tensor) -> torch.Tensor:
        return values.gather(1, gather_index).squeeze(1)

    selected_top = torch.where(has_accepted, select(tops), torch.zeros_like(first_accepted))
    selected_left = torch.where(has_accepted, select(lefts), torch.zeros_like(first_accepted))
    selected_height = torch.where(
        has_accepted,
        select(heights),
        torch.full_like(first_accepted, image_height),
    )
    selected_width = torch.where(
        has_accepted,
        select(widths),
        torch.full_like(first_accepted, image_width),
    )
    selected_mass = torch.where(
        has_accepted, select(masses), torch.ones_like(first_accepted, dtype=torch.float32)
    )
    attempts = torch.where(
        has_accepted,
        first_accepted + 1,
        torch.full_like(first_accepted, max_crop_attempts),
    )

    return {
        "crop_top": selected_top.float(),
        "crop_left": selected_left.float(),
        "crop_height": selected_height.float(),
        "crop_width": selected_width.float(),
        "flipped": torch.rand(batch_size, device=device) < flip_probability,
        "original_height": torch.full(
            (batch_size,), image_height, device=device, dtype=torch.float32
        ),
        "original_width": torch.full(
            (batch_size,), image_width, device=device, dtype=torch.float32
        ),
        "teacher_mass": selected_mass.float(),
        "crop_attempts": attempts.float(),
        "fallback_identity": ~has_accepted,
        "full_image_teacher_mass": teacher_pixels.sum(dim=(-3, -2, -1)),
    }


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


def transform_view_a_map_to_b(
    view_a_map: torch.Tensor,
    geometry: dict[str, torch.Tensor],
    output_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Apply the exact crop/resize/flip to an A-coordinate probability map."""
    if output_size is None:
        output_size = view_a_map.shape[-2:]
    transformed = F.grid_sample(
        view_a_map,
        forward_grid(geometry, output_size[0], output_size[1]),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    transformed = transformed.clamp_min(0)
    return transformed / transformed.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)


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
                **(
                    {"teacher_mass": float(geometry["teacher_mass"][index])}
                    if "teacher_mass" in geometry
                    else {}
                ),
                **(
                    {"crop_attempts": int(geometry["crop_attempts"][index])}
                    if "crop_attempts" in geometry
                    else {}
                ),
                **(
                    {"fallback_identity": bool(geometry["fallback_identity"][index])}
                    if "fallback_identity" in geometry
                    else {}
                ),
            }
        )
    return records
