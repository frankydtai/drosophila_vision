# -*- coding: utf-8 -*-
"""Spot paradigm INPUT: connectome spot geometry + PR drive waveform.

Geometry (centers, sub-spot shifts, Euclidean rings) is split out of the old
``network.spot_target`` Section A. The PR drive waveform ``u[t]`` is defined
here once (``spot_input_waveform``) and consumed by both the network signal and
the ImpR target in :mod:`task.spot.data`, so pulse duration has a single
source.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from network import path  # noqa: F401 -- FAFBv783 on sys.path
import column_mapper

from neuron.params import DELTAT_MS, ms_to_steps

_SPOT_EXTENT_HALF_STEP_TOL = 1e-9

# Default post-onset response window (ms). ``train.py`` sets
# ``maxtime = t_on + ms_to_steps(response_ms) + 1`` (inclusive endpoint sample).
RESPONSE_MS = 1500.0

# Default spot footprint / center-tiling radius (0.5 multiples).
DEFAULT_SPOT_EXTENT: float = 1.0
# Panel list for multi-spot visualisation.
DEFAULT_SPOT_EXTENTS: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
# Keep only centers whose spot footprint lies inside connectome extent.
DEFAULT_FULLY_INSIDE: bool = True
# Tile simultaneous spot centers on network connectome (``False`` -> center (0,0) only).
DEFAULT_MULTI_SPOT: bool = True
# Sub-spot shift hex-disc radius (``members_in_extent``; 1 -> 7 shifts).
DEFAULT_SHIFT_EXTENT: int = 1


def spot_input_waveform(t_on, maxtime, pulse_ms=None, *, deltat_ms: float = DELTAT_MS) -> np.ndarray:
    """Normalized 0/1 photoreceptor drive ``u[t]`` over ``maxtime`` steps.

    ``pulse_ms`` omitted -> continue-on step (``u[t_on:] = 1``). With a value the
    stimulus is on only for ``[t_on, t_on + round(pulse_ms/deltat))`` and returns
    to baseline afterward; ``maxtime`` is unchanged.
    """
    t_on = int(t_on)
    maxtime = int(maxtime)
    u = np.zeros(maxtime)
    if pulse_ms is None:
        u[t_on:] = 1.0
    else:
        width = max(1, ms_to_steps(pulse_ms, deltat_ms=deltat_ms))
        u[t_on:min(maxtime, t_on + width)] = 1.0
    return u


def _rot60(u: int, v: int) -> Tuple[int, int]:
    """Rotate an axial (u, v) member 60 degrees counter-clockwise about origin."""
    return -v, u + v


def euclid_hex_dist(du: int, dv: int) -> float:
    """Euclidean distance (in column units) between two axial cells."""
    return math.sqrt(du * du + du * dv + dv * dv)


def members_by_euclid_radius(radii) -> Dict[float, List[Tuple[int, int]]]:
    """Map each Euclidean radius to stim-centered axial ``(du, dv)`` members."""
    radii_set = {round(float(radius), 6) for radius in radii}
    max_shell = int(math.ceil(max(radii_set)))
    by_radius: Dict[float, List[Tuple[int, int]]] = {radius: [] for radius in radii_set}
    for du, dv in column_mapper.members_in_extent(max_shell):
        radius = round(euclid_hex_dist(du, dv), 6)
        if radius in radii_set:
            by_radius[radius].append((int(du), int(dv)))
    missing = [radius for radius in radii_set if not by_radius[radius]]
    if missing:
        raise ValueError(f"no hex members for spot cost radii {missing}")
    return by_radius


def spot_extent_half_steps(spot_extent) -> int:
    """``spot_extent = 0.5 * m`` for non-negative integer ``m``; return ``m``."""
    value = float(spot_extent)
    if value < 0:
        raise ValueError(f"spot_extent must be >= 0, got {spot_extent!r}")
    half_steps = value * 2.0
    m = round(half_steps)
    if abs(half_steps - m) > _SPOT_EXTENT_HALF_STEP_TOL:
        raise ValueError(
            f"spot_extent must be a non-negative 0.5 multiple, got {spot_extent!r}",
        )
    return int(m)


def spot_dist(spot_extent) -> int:
    """Axial center spacing: ``2*spot_extent + 1`` (``spot_extent`` in 0.5 steps)."""
    return spot_extent_half_steps(spot_extent) + 1


def spot_extent_folds_r2_into_r1(spot_extent) -> bool:
    """True when ``spot_extent == 1`` (``spot_extent_half_steps == 2``).

    Fold semantics live in :mod:`task.spot.data`: r=1 target amplitude is
    ``RecF(1)+RecF(2)`` and r=2 amplitude is 0.
    """
    return spot_extent_half_steps(spot_extent) == 2


def _spot_center_angle(u: int, v: int) -> float:
    """Degree-space angle of (u, v), for a stable angular tie-break ordering."""
    x_deg, y_deg = column_mapper.uv_to_xy_deg(u, v)
    return float(np.arctan2(float(y_deg), float(x_deg)))


def spot_centers(
    extent: int = column_mapper.DEFAULT_EXTENT,
    spot_extent=DEFAULT_SPOT_EXTENT,
    fully_inside: bool = DEFAULT_FULLY_INSIDE,
) -> list:
    """Axial centers of densest packing of radius-``floor(spot_extent)`` hexes."""
    m = spot_extent_half_steps(spot_extent)
    k = m // 2
    if m % 2 == 1:
        e = (m + 1) // 2
        a1, b1 = e, e
    else:
        a1, b1 = m + 1, -k
    a2, b2 = _rot60(a1, b1)
    members = column_mapper.members_in_extent((m + 1) // 2)
    span = int(2 * (extent // max(k, 1) + 2))
    centers: list = []
    for lm in range(-span, span + 1):
        for ln in range(-span, span + 1):
            cu = lm * a1 + ln * a2
            cv = lm * b1 + ln * b2
            if column_mapper.hex_radius(cu, cv) > extent:
                continue
            if fully_inside and any(
                column_mapper.hex_radius(cu + du, cv + dv) > extent
                for du, dv in members
            ):
                continue
            centers.append((cu, cv))
    centers.sort(
        key=lambda c: (column_mapper.hex_radius(*c), _spot_center_angle(*c)),
    )
    return centers


@dataclass
class Spot:
    """Spot centers and sub-spot shifts over a loaded connectome."""

    centers: List[Tuple[int, int]]
    shifts: List[Tuple[int, int]]
    spot_extent: float


@dataclass(frozen=True)
class SpotBatch:
    """One simultaneous spot stimulus: all ``stim_uv`` columns step in one batch."""

    shift: Tuple[int, int]
    stim_uv: Tuple[Tuple[int, int], ...]


def spot_stimulus_batches(spot: Spot) -> List[SpotBatch]:
    """One batch per shift; each batch steps all spot centers (+ shift) together."""
    batches: List[SpotBatch] = []
    for du, dv in spot.shifts:
        stim_uv = tuple(
            (int(cu + du), int(cv + dv))
            for cu, cv in spot.centers
        )
        batches.append(SpotBatch(shift=(int(du), int(dv)), stim_uv=stim_uv))
    return batches


def _connectome_extent(C, spot_extent: float) -> int:
    """Hex-disc radius of connectome ``C``."""
    meta_extent = int(C.meta.get("extent", -1))
    if meta_extent >= 0:
        return meta_extent
    positioned = C.column_id >= 0
    radii = [
        column_mapper.hex_radius(int(u), int(v))
        for u, v in zip(C.u[positioned], C.v[positioned])
    ]
    return max(radii) if radii else int(spot_extent)


def build_spot(
    C,
    spot_extent: float = DEFAULT_SPOT_EXTENT,
    multi_spot: bool = DEFAULT_MULTI_SPOT,
    fully_inside: bool = DEFAULT_FULLY_INSIDE,
) -> Spot:
    """Build a :class:`Spot` for connectome ``C``."""
    spot_extent_half_steps(spot_extent)
    connectome_extent = _connectome_extent(C, spot_extent)
    shifts = column_mapper.members_in_extent(1)
    if not multi_spot:
        centers = [(0, 0)]
    else:
        centers = [
            (int(cu), int(cv))
            for cu, cv in spot_centers(
                extent=connectome_extent,
                spot_extent=spot_extent,
                fully_inside=fully_inside,
            )
        ]
    return Spot(centers, shifts, spot_extent)


def spot_from_opts(
    C,
    spot_extent: float = DEFAULT_SPOT_EXTENT,
    shift_extent: int = DEFAULT_SHIFT_EXTENT,
    multi_spot: bool = DEFAULT_MULTI_SPOT,
    fully_inside: bool = DEFAULT_FULLY_INSIDE,
    *,
    stimulus_opts: Optional[Dict] = None,
) -> Spot:
    """Build :class:`Spot` with configurable sub-spot shift radius."""
    if stimulus_opts is not None:
        spot_extent = float(stimulus_opts.get("spot_extent", spot_extent))
        shift_extent = int(stimulus_opts.get("shift_extent", shift_extent))
        multi_spot = bool(stimulus_opts.get("multi_spot", multi_spot))
        fully_inside = bool(stimulus_opts.get("fully_inside", fully_inside))
    spot = build_spot(
        C, spot_extent, multi_spot=multi_spot, fully_inside=fully_inside,
    )
    spot.shifts = [
        (int(du), int(dv))
        for du, dv in column_mapper.members_in_extent(int(shift_extent))
    ]
    return spot
