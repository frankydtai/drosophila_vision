# -*- coding: utf-8 -*-
"""Spot paradigm sti geometry: footprint, centers, sub-spot shifts, Euclidean radii."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from network import path  # noqa: F401 -- FAFBv783 on sys.path
import build_hex

_SPOT_RADIUS_HALF_STEP_TOL = 1e-9


def euclid_hex_dist(du: int, dv: int) -> float:
    """Euclidean distance (in hex nodes) between two axial cells."""
    return math.sqrt(du * du + du * dv + dv * dv)


def members_by_euclid_radius(radii) -> dict[float, list[tuple[int, int]]]:
    """Map each Euclidean radius to sti-centered axial ``(du, dv)`` members."""
    radii_set = {round(float(radius), 6) for radius in radii}
    max_shell = int(math.ceil(max(radii_set)))
    by_radius: dict[float, list[tuple[int, int]]] = {radius: [] for radius in radii_set}
    for du, dv in build_hex.members_in_radius(max_shell):
        radius = round(euclid_hex_dist(du, dv), 6)
        if radius in radii_set:
            by_radius[radius].append((int(du), int(dv)))
    missing = [radius for radius in radii_set if not by_radius[radius]]
    if missing:
        raise ValueError(f"no hex members for spot cost radii {missing}")
    return by_radius


def spot_radius_half_steps(spot_radius) -> int:
    """``spot_radius = 0.5 * m`` for non-negative integer ``m``; return ``m``."""
    value = float(spot_radius)
    if value < 0:
        raise ValueError(f"spot_radius must be >= 0, got {spot_radius!r}")
    half_steps = value * 2.0
    m = round(half_steps)
    if abs(half_steps - m) > _SPOT_RADIUS_HALF_STEP_TOL:
        raise ValueError(
            f"spot_radius must be a non-negative 0.5 multiple, got {spot_radius!r}",
        )
    return int(m)


def spot_radius_dist(spot_radius) -> int:
    """Axial center spacing: ``2*spot_radius + 1`` (``spot_radius`` in 0.5 steps)."""
    return spot_radius_half_steps(spot_radius) + 1


def spot_radius_folds_r2_into_r1(spot_radius) -> bool:
    """True when ``spot_radius == 1`` (``spot_radius_half_steps == 2``).

    Fold semantics live in :func:`task.spot.gt._spot_readout_a_radius`: r=1 gt
    ``a_radius`` is ``rf(1)+rf(2)`` and r=2 ``a_radius`` is 0. Non-center drive
    scales use ``a_sti_radius`` masked by cost-radius scale (scale==0 → force
    0). Center r=0 remains baked at scale 1.
    """
    return spot_radius_half_steps(spot_radius) == 2


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
    m = spot_radius_half_steps(spot_radius)
    k = m // 2
    if m % 2 == 1:
        e = (m + 1) // 2
        a1, b1 = e, e
    else:
        a1, b1 = m + 1, -k
    a2, b2 = -b1, a1 + b1  # 60° CCW about origin
    members = build_hex.members_in_radius((m + 1) // 2)
    span = int(2 * (connectome_radius // max(k, 1) + 2))
    centers: list = []
    for lm in range(-span, span + 1):
        for ln in range(-span, span + 1):
            cu = lm * a1 + ln * a2
            cv = lm * b1 + ln * b2
            if build_hex.hex_radius(cu, cv) > connectome_radius:
                continue
            if fully_inside and any(
                build_hex.hex_radius(cu + du, cv + dv) > connectome_radius
                for du, dv in members
            ):
                continue
            centers.append((cu, cv))
    centers.sort(
        key=lambda c: (build_hex.hex_radius(*c), _spot_center_angle(*c)),
    )
    return centers


@dataclass
class Spot:
    """Spot centers and sub-spot shifts over a loaded connectome."""

    centers: list[tuple[int, int]]
    shifts: list[tuple[int, int]]
    spot_radius: float


@dataclass(frozen=True)
class SpotBatch:
    """One simultaneous spot sti: all ``sti_uv`` hexes step in one batch."""

    shift: tuple[int, int]
    sti_uv: tuple[tuple[int, int], ...]


def spot_sti_batches(spot: Spot) -> list[SpotBatch]:
    """One batch per shift; each batch steps all spot centers (+ shift) together."""
    return [
        SpotBatch(
            shift=(int(du), int(dv)),
            sti_uv=tuple(
                (int(cu + du), int(cv + dv)) for cu, cv in spot.centers
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
    spot_radius_half_steps(spot_radius)
    connectome_radius = _connectome_radius(connectome, spot_radius)
    shifts = build_hex.members_in_radius(1)
    if not multi_spot:
        centers = [(0, 0)]
    else:
        centers = [
            (int(cu), int(cv))
            for cu, cv in spot_centers(
                connectome_radius=connectome_radius,
                spot_radius=spot_radius,
                fully_inside=fully_inside,
            )
        ]
    return Spot(centers, shifts, spot_radius)


def spot_from_opts(
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
            "spot_from_opts requires spot_radius, shift_radius, multi_spot, and "
            "fully_inside (or sti_opts containing them)"
        )
    spot = build_spot(
        connectome, spot_radius=spot_radius, multi_spot=multi_spot, fully_inside=fully_inside,
    )
    spot.shifts = [
        (int(du), int(dv))
        for du, dv in build_hex.members_in_radius(int(shift_radius))
    ]
    return spot
