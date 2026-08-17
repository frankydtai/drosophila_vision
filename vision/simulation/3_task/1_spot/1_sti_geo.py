# -*- coding: utf-8 -*-
"""Spot sti geometry: footprint, centers, sub-spot shifts, hex radii."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from network import path  # noqa: F401 -- FAFBv783 on sys.path
import build_hex

def standardize_spot_radius(spot_radius) -> float:
    """Require non-negative 0.5 multiple; return the canonical float."""
    spot_radius = float(spot_radius)
    if spot_radius < 0:
        raise ValueError(f"spot_radius must be >= 0, got {spot_radius!r}")
    if round(spot_radius * 2.0) / 2.0 != spot_radius:
        raise ValueError(
            f"spot_radius must be a non-negative 0.5 multiple, got {spot_radius!r}",
        )
    return spot_radius


def _spot_center_angle(u: int, v: int) -> float:
    """Degree-space angle of (u, v), for a stable angular tie-break ordering."""
    x_deg, y_deg = build_hex.xy_deg_from_uv(u, v)
    return float(np.arctan2(float(y_deg), float(x_deg)))


def spot_centers(
    connectome_radius: int = build_hex.DEFAULT_RADIUS,
    *,
    spot_radius: float,
    fully_inside: bool,
) -> list:
    """Axial centers of densest packing of radius-``floor(spot_radius)`` hexes."""
    radius_halves = int(round(2.0 * standardize_spot_radius(spot_radius)))
    radius_floor = radius_halves // 2
    if radius_halves % 2 == 1:
        step1_u = step1_v = (radius_halves + 1) // 2
    else:
        step1_u, step1_v = radius_halves + 1, -radius_floor
    step2_u, step2_v = -step1_v, step1_u + step1_v  # 60° CCW about origin
    hexes = build_hex.radius_hexes((radius_halves + 1) // 2)
    span = int(2 * (connectome_radius // max(radius_floor, 1) + 2))
    centers: list = []
    for n_step1 in range(-span, span + 1):
        for n_step2 in range(-span, span + 1):
            center_u = n_step1 * step1_u + n_step2 * step2_u
            center_v = n_step1 * step1_v + n_step2 * step2_v
            if build_hex.hex_radius(center_u, center_v) > connectome_radius:
                continue
            if fully_inside and any(
                build_hex.hex_radius(center_u + du, center_v + dv) > connectome_radius
                for du, dv in hexes
            ):
                continue
            centers.append((center_u, center_v))
    centers.sort(
        key=lambda center: (
            build_hex.hex_radius(*center),
            _spot_center_angle(*center),
        ),
    )
    return centers


@dataclass
class Spot:
    """Spot centers and sub-spot shifts over a loaded connectome."""

    centers: list[tuple[int, int]]
    shifts: list[tuple[int, int]]
    spot_radius: float


@dataclass(frozen=True)
class SpotB:
    """One simultaneous spot sti: all ``sti_uv`` hexes step in one b."""

    shift: tuple[int, int]
    sti_uv: tuple[tuple[int, int], ...]


def spot_sti_bs(spot: Spot) -> list[SpotB]:
    """One b per shift; each b steps all spot centers (+ shift) together."""
    return [
        SpotB(
            shift=(int(du), int(dv)),
            sti_uv=tuple(
                (int(center_u + du), int(center_v + dv))
                for center_u, center_v in spot.centers
            ),
        )
        for du, dv in spot.shifts
    ]


def _connectome_radius(connectome, spot_radius: float) -> int:
    """Hex-disc radius of the connectome."""
    meta_radius = int(connectome.meta.get("radius", -1))
    if meta_radius >= 0:
        return meta_radius
    positioned = connectome.column_ids >= 0
    radii = [
        build_hex.hex_radius(int(hex_u), int(hex_v))
        for hex_u, hex_v in zip(connectome.us[positioned], connectome.vs[positioned])
    ]
    return max(radii) if radii else int(spot_radius)


def build_spot(
    connectome,
    *,
    spot_radius: float,
    multi_spot: bool,
    fully_inside: bool,
) -> Spot:
    """Build a :class:`Spot` for the connectome."""
    spot_radius = standardize_spot_radius(spot_radius)
    connectome_radius = _connectome_radius(connectome, spot_radius)
    shifts = build_hex.radius_hexes(1)
    if not multi_spot:
        centers = [(0, 0)]
    else:
        centers = [
            (int(center_u), int(center_v))
            for center_u, center_v in spot_centers(
                connectome_radius=connectome_radius,
                spot_radius=spot_radius,
                fully_inside=fully_inside,
            )
        ]
    return Spot(centers, shifts, spot_radius)


def resolve_spot(
    connectome,
    *,
    spot_radius: float | None = None,
    shift_radius: int | None = None,
    multi_spot: bool | None = None,
    fully_inside: bool | None = None,
    sti_opts: dict | None = None,
) -> Spot:
    """Build :class:`Spot` with configurable sub-spot shift radius."""
    if sti_opts is not None:
        spot_radius = float(sti_opts["spot_radius"])
        shift_radius = int(sti_opts["shift_radius"])
        multi_spot = bool(sti_opts["multi_spot"])
        fully_inside = bool(sti_opts["fully_inside"])
    if (
        spot_radius is None
        or shift_radius is None
        or multi_spot is None
        or fully_inside is None
    ):
        raise TypeError(
            "resolve_spot requires spot_radius, shift_radius, multi_spot, and "
            "fully_inside (or sti_opts containing them)"
        )
    spot = build_spot(
        connectome, spot_radius=spot_radius, multi_spot=multi_spot, fully_inside=fully_inside,
    )
    spot.shifts = [
        (int(du), int(dv))
        for du, dv in build_hex.radius_hexes(int(shift_radius))
    ]
    return spot
