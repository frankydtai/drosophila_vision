# -*- coding: utf-8 -*-
"""Connectome spotting layout and spot training targets.

Section A — connectome spotting (``Spotting``, ``build_spotting``, …).
Section B — training target (``build_shifted_target``, ``spot_cost_*``, …).

Hex disc enumeration lives in :mod:`column_mapper`; spot centre tiling and
Euclidean cost-radius grouping live in this module.

For every (spot, shift) stimulus the connectome is driven at ONE column; each
fit-cell readout is compared to ``RecF(radius) * ImpR(t)`` where ``radius`` is the
**Euclidean** hex distance (in column units) from the stimulated column to the
readout cell's column. The extent-2 patch is NOT iso-distant: 6 corners sit at
radius=2, 6 edge midpoints at radius=sqrt(3). ``RecF`` is sampled from the
continuous analytic Gaussian profile (``Medulla_Library.read_RecF_ImpR`` ->
RecF_data, 45 samples centred on index 22; column distance maps to sample 22 +
5*radius), so the sqrt(3) edge target is evaluated at its true radius rather
than snapped to col +/-2 (which would mis-sign L1's centre-surround near its
~1.6 col zero crossing).

When ``--spot-cost-r-w`` is omitted, weights default to
:data:`DEFAULT_SPOT_COST_RADIUS_WEIGHT` (``0=1,1=1/6,2=1/6``; ``sqrt3`` excluded).
With CLI, only listed radii are used; omitted radii get weight ``0`` (excluded).
Example: ``0=1,1=1/6,2=1/12`` also skips ``sqrt3``.

``build_shifted_target`` returns a :class:`ShiftedTarget` (timing: :mod:`training_config`):

    signal          (B, T, N)      per-batch stimulus current
    data            (n_cost, T')   target trace per cost cell (MSE compares this)
    power           scalar       weighted target power for cost normalisation
    cost_weight     (n_cost,)    per-cost-cell weight at each Euclidean radius
    cost_radius     (n_cost,)    Euclidean radius {0,1,sqrt3,2,...} per cost cell
    readout_batch   (n_cost,)    which batch (stimulus) each cost cell belongs to
    readout_unit    (n_cost,)    which unit each cost cell is

``info`` includes ``n_cost`` (readout rows) and ``n_cost_columns`` (member
columns in ``cost_extent`` per stimulus batch: ``int`` when uniform, else
``{batch: count}``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import network_bootstrap  # noqa: F401

import column_mapper

from Medulla_Library import DATA_AMP, I_DARK, I_BASELINE, I_BRIGHT, cell_list as _CELL_LIST, read_RecF_ImpR
from training_config import IMPULSE_MAXTIME, SIM_DTYPE_DEFAULT, T_ON
from .construction import col2fit, unit_type_names
from .stimulus import column_in_cost_extent

# RF sample index of the receptive-field centre, and samples per column step
# (data[i,j] = RecF_data[i, 5j+2]; j=4 -> sample 22 -> radius=0).
_RF_CENTER_SAMPLE = 22
_RF_SAMPLES_PER_COL = 5
_RF_NSAMPLES = 45

# Euclidean radii recognised by ``--spot-cost-r-w``.
DEFAULT_SPOT_COST_RADII: Tuple[float, ...] = (0.0, 1.0, math.sqrt(3), 2.0)

# Default ``--spot-cost-r-w`` when the CLI flag is omitted.
DEFAULT_SPOT_COST_RADIUS_WEIGHT: Dict[float, float] = {
    0.0: 1.0,
    1.0: 1.0 / 6.0,
    2.0: 1.0 / 6.0,
}

SPOT_COST_RADIUS_KEY_ALIASES: Dict[str, float] = {
    "sqrt3": math.sqrt(3),
}

_SPOT_EXTENT_HALF_STEP_TOL = 1e-9


def _rot60(u: int, v: int) -> Tuple[int, int]:
    """Rotate an axial (u, v) member 60 degrees counter-clockwise about origin."""
    return -v, u + v


def euclid_hex_dist(du: int, dv: int) -> float:
    """Euclidean distance (in column units) between two axial cells."""
    return math.sqrt(du * du + du * dv + dv * dv)


def members_by_euclid_radius(
    radii,
) -> Dict[float, List[Tuple[int, int]]]:
    """Map each Euclidean radius to stim-centred axial ``(du, dv)`` members."""
    radii_set = {round(float(radius), 6) for radius in radii}
    max_shell = int(math.ceil(max(radii_set)))
    by_radius: Dict[float, List[Tuple[int, int]]] = {
        radius: [] for radius in radii_set
    }
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
    """Axial centre spacing: ``2*spot_extent + 1`` (``spot_extent`` in 0.5 steps)."""
    return spot_extent_half_steps(spot_extent) + 1


def _spot_center_angle(u: int, v: int) -> float:
    """Degree-space angle of (u, v), for a stable angular tie-break ordering."""
    x_deg, y_deg = column_mapper.uv_to_xy_deg(u, v)
    return float(np.arctan2(float(y_deg), float(x_deg)))


def spot_centers(
    extent: int = column_mapper.DEFAULT_EXTENT,
    spot_extent=2,
    fully_inside: bool = True,
) -> list:
    """Axial centres of radius-``floor(spot_extent)`` hexes covering an ``extent`` disc."""
    m = spot_extent_half_steps(spot_extent)
    k = m // 2
    a1, b1 = m + 1, -k
    a2, b2 = _rot60(a1, b1)
    members = column_mapper.members_in_extent(k)
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


# -- Section A: connectome spotting layout ------------------------------------


@dataclass
class Spotting:
    """Spot centres and sub-spot shifts over a loaded connectome."""

    centers: List[Tuple[int, int]]
    shifts: List[Tuple[int, int]]
    spot_extent: float


def normalize_spot_cost_radius_key(key) -> float:
    """Parse one ``--spot-cost-r-w`` radius key (``0``, ``sqrt3``, …)."""
    if isinstance(key, (int, float)):
        return round(float(key), 6)
    text = str(key).strip().lower()
    if text in SPOT_COST_RADIUS_KEY_ALIASES:
        return round(SPOT_COST_RADIUS_KEY_ALIASES[text], 6)
    return round(float(text), 6)


def parse_spot_cost_radius_weight_value(text: str) -> float:
    """Parse one weight token (supports fractions like ``1/6``)."""
    tok = str(text).strip()
    if "/" in tok:
        num, den = tok.split("/", 1)
        return float(num.strip()) / float(den.strip())
    return float(tok)


def spot_cost_radius_weight_resolved(
    spot_cost_radius_weight: Optional[Dict[float, float]] = None,
) -> Dict[float, float]:
    """Return explicit weights or :data:`DEFAULT_SPOT_COST_RADIUS_WEIGHT`."""
    if spot_cost_radius_weight is None:
        return dict(DEFAULT_SPOT_COST_RADIUS_WEIGHT)
    return spot_cost_radius_weight


def expand_spot_cost_r_w_dict(
    kv: Optional[dict] = None,
    *,
    stimulus_opts: Optional[dict] = None,
) -> Optional[Dict[float, float]]:
    """CLI ``r=w`` dict → ``{round(radius): weight}``; empty/None → unresolved (use default).

    With ``stimulus_opts``, read ``spot_cost_radius_weight`` from a stimulus sidecar dict.
    """
    if stimulus_opts is not None:
        kv = (stimulus_opts or {}).get("spot_cost_radius_weight")
    if not kv:
        return None
    return {
        normalize_spot_cost_radius_key(k): parse_spot_cost_radius_weight_value(v)
        for k, v in kv.items()
    }


def resolve_spot_cost_radii(
    spot_cost_radius_weight: Optional[Dict[float, float]] = None,
    *,
    stimulus_opts: Optional[dict] = None,
) -> Tuple[float, ...]:
    """Radii in spot cost with non-zero weight.

    Unset CLI uses :data:`DEFAULT_SPOT_COST_RADIUS_WEIGHT`; explicit CLI keeps only
    listed radii (omitted radii → weight 0, excluded).

    With ``stimulus_opts``, read weights from ``spot_cost_radius_weight`` first.
    """
    if stimulus_opts is not None:
        spot_cost_radius_weight = expand_spot_cost_r_w_dict(stimulus_opts=stimulus_opts)
    weights = spot_cost_radius_weight_resolved(spot_cost_radius_weight)
    return tuple(
        radius for radius in DEFAULT_SPOT_COST_RADII
        if float(weights.get(round(radius, 6), 0.0)) != 0.0
    )


def spot_cost_cell_weight(
    radius: float,
    spot_cost_radius_weight: Optional[Dict[float, float]],
) -> float:
    """Per-cell cost weight from resolved CLI/default weights (missing radius → 0)."""
    weights = spot_cost_radius_weight_resolved(spot_cost_radius_weight)
    return float(weights.get(round(radius, 6), 0.0))


def spot_stimulus_batches(spotting: Spotting) -> List[Tuple[int, int, Tuple[int, int]]]:
    """One batch per (spot centre, shift): ``(stim_u, stim_v, center)``."""
    batches = []
    for center in spotting.centers:
        for du, dv in spotting.shifts:
            batches.append((center[0] + du, center[1] + dv, center))
    return batches


def _connectome_extent(C, spot_extent: int) -> int:
    """Hex-disc radius of connectome ``C``."""
    meta_extent = int(C.meta.get("extent", -1))
    if meta_extent >= 0:
        return meta_extent
    positioned = C.column_id >= 0
    radii = [
        column_mapper.hex_radius(int(u), int(v))
        for u, v in zip(C.u[positioned], C.v[positioned])
    ]
    return max(radii) if radii else spot_extent


def build_spotting(
    C,
    spot_extent: int = 2,
    single_spot: bool = None,
    fully_inside: bool = True,
) -> Spotting:
    """Build a :class:`Spotting` for connectome ``C``."""
    spot_extent_half_steps(spot_extent)
    connectome_extent = _connectome_extent(C, spot_extent)
    if single_spot is None:
        single_spot = connectome_extent <= spot_extent
    shifts = column_mapper.members_in_extent(1)
    if single_spot:
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
    return Spotting(centers, shifts, spot_extent)


def spotting_from_opts(
    C,
    spot_extent: int = 2,
    shift_extent: int = 0,
    single_spot: Optional[bool] = None,
    *,
    stimulus_opts: Optional[Dict] = None,
) -> Spotting:
    """Build :class:`Spotting` with configurable sub-spot shift radius.

    With ``stimulus_opts``, read ``spot_extent`` / ``shift_extent`` from a stimulus sidecar.
    """
    if stimulus_opts is not None:
        spot_extent = float(stimulus_opts.get("spot_extent", spot_extent))
        shift_extent = int(stimulus_opts.get("shift_extent", shift_extent))
    spotting = build_spotting(C, spot_extent, single_spot)
    spotting.shifts = [
        (int(du), int(dv))
        for du, dv in column_mapper.members_in_extent(int(shift_extent))
    ]
    return spotting


# -- Section B: training target -----------------------------------------------


@dataclass
class ShiftedTarget:
    signal: torch.Tensor          # (B, T, N)
    data: torch.Tensor            # (n_cost, T')
    power: torch.Tensor           # scalar
    cost_weight: torch.Tensor     # (n_cost,)
    cost_radius: torch.Tensor     # (n_cost,)
    readout_batch: torch.Tensor   # (n_cost,) long
    readout_unit: torch.Tensor    # (n_cost,) long
    n_batch: int
    info: dict


def _recf_at(recf_row: np.ndarray, radius: float) -> float:
    """Sample the continuous RF profile at column distance ``radius`` (interpolated)."""
    idx = _RF_CENTER_SAMPLE + _RF_SAMPLES_PER_COL * radius
    idx = min(max(idx, 0.0), _RF_NSAMPLES - 1)
    return float(np.interp(idx, np.arange(_RF_NSAMPLES), recf_row))


def spot_cost_columns(batches, cost_radii, cost_extent):
    """Cost columns at stim-centred Euclidean radii: ``(batch, mu, mv, radius)`` each."""
    by_radius = members_by_euclid_radius(cost_radii)
    cols = []
    for b, (su, sv, _center) in enumerate(batches):
        for radius_key, members in by_radius.items():
            for du, dv in members:
                mu, mv = su + du, sv + dv
                if not column_in_cost_extent(mu, mv, cost_extent):
                    continue
                cols.append((b, int(mu), int(mv), float(radius_key)))
    return cols


def spot_n_cost_columns(cost_cols):
    """Member columns in ``cost_extent`` per stimulus batch."""
    if not cost_cols:
        return 0
    counts: Dict[int, int] = {}
    for b, _mu, _mv, _radius in cost_cols:
        counts[b] = counts.get(b, 0) + 1
    vals = set(counts.values())
    if len(vals) == 1:
        return next(iter(vals))
    return {b: counts[b] for b in sorted(counts)}


def spot_cost_unit_radius_layout(C, batches, cost_radii, cost_extent):
    """All units on :func:`spot_cost_columns`, with batch and stim-centred radius."""
    u = C.u.detach().cpu().numpy() if hasattr(C.u, "detach") else np.asarray(C.u)
    v = C.v.detach().cpu().numpy() if hasattr(C.v, "detach") else np.asarray(C.v)
    type_all = (
        C.node_type.detach().cpu().numpy()
        if hasattr(C.node_type, "detach") else np.asarray(C.node_type)
    )
    batch_idx, unit_idx, radius, type_idx = [], [], [], []
    for b, mu, mv, cell_radius in spot_cost_columns(batches, cost_radii, cost_extent):
        on_col = (u == mu) & (v == mv)
        for uid in np.where(on_col)[0]:
            batch_idx.append(b)
            unit_idx.append(int(uid))
            radius.append(cell_radius)
            type_idx.append(int(type_all[uid]))
    return (
        np.asarray(batch_idx, dtype=np.int64),
        np.asarray(unit_idx, dtype=np.int64),
        np.asarray(radius, dtype=np.float64),
        np.asarray(type_idx, dtype=np.int64),
    )


def build_shifted_target(
    C,
    spot_extent: int = 2,
    single_spot: Optional[bool] = None,
    shift_extent: int = 0,
    maxtime: int = IMPULSE_MAXTIME,
    t_on: int = T_ON,
    i_baseline: float = I_BASELINE,
    i_bright: float = I_BRIGHT,
    i_dark: float = I_DARK,
    polarity: str = "bright",
    data_amp: float = DATA_AMP,
    device: Optional[str] = None,
    cost_extent: Optional[int] = None,
    spot_cost_radius_weight: Optional[Dict[float, float]] = None,
    sim_dtype: torch.dtype = SIM_DTYPE_DEFAULT,
) -> ShiftedTarget:
    if polarity not in ("bright", "dark"):
        raise ValueError(f"polarity must be 'bright' or 'dark', got {polarity!r}")
    i_step = float(i_bright if polarity == "bright" else i_dark)
    device = device or C.device
    recf_data, impr_data = read_RecF_ImpR()  # (13,45), (13,IMPULSE_MAXTIME)
    fit_row = {str(ft): i for i, ft in enumerate(_CELL_LIST)}

    spotting = spotting_from_opts(
        C, spot_extent, shift_extent, single_spot,
    )
    names = unit_type_names(C)
    present_fit = [str(ft) for ft in _CELL_LIST if str(ft) in set(names.tolist())]

    batches = spot_stimulus_batches(spotting)
    n_batch = len(batches)

    signal = torch.zeros((n_batch, maxtime, C.n_units), dtype=sim_dtype, device=device)
    for b, (su, sv, _center) in enumerate(batches):
        units = C.input_units_at(su, sv)
        if len(units):
            idx = torch.as_tensor(units, dtype=torch.long, device=device)
            signal[b, :t_on, idx] = i_baseline
            signal[b, t_on:, idx] = i_step

    resp = slice(t_on, maxtime)  # post-T_ON cost window; see training_config

    cost_radii = resolve_spot_cost_radii(spot_cost_radius_weight)
    cost_cols = spot_cost_columns(batches, cost_radii, cost_extent)

    cost_batch, cost_unit, cost_radius_list, cost_target, cost_weight_list = [], [], [], [], []
    for b, mu, mv, radius in cost_cols:
        w = spot_cost_cell_weight(radius, spot_cost_radius_weight)
        if w == 0.0:
            continue
        for ft in present_fit:
            units = col2fit(C, mu, mv, ft, names)
            if len(units) == 0:
                continue
            row = fit_row[ft]
            amp = _recf_at(recf_data[row], radius)
            trace = amp * impr_data[row][resp] * data_amp
            if polarity == "dark":
                trace = -trace
            for uidx in units:
                cost_batch.append(b)
                cost_unit.append(int(uidx))
                cost_radius_list.append(radius)
                cost_target.append(trace)
                cost_weight_list.append(w)

    if not cost_batch:
        raise ValueError("no spot cost cells (check cost_extent and fit cell types)")

    data = torch.tensor(np.asarray(cost_target), dtype=sim_dtype, device=device)  # (n_cost,T')
    cost_weight = torch.tensor(np.asarray(cost_weight_list), dtype=sim_dtype, device=device)
    cost_radius = torch.tensor(np.asarray(cost_radius_list), dtype=sim_dtype, device=device)
    readout_batch = torch.tensor(np.asarray(cost_batch), dtype=torch.long, device=device)
    readout_unit = torch.tensor(np.asarray(cost_unit), dtype=torch.long, device=device)

    power = torch.sum(cost_weight[:, None] * data ** 2)
    if float(power) == 0.0:
        power = torch.tensor(1.0, dtype=sim_dtype, device=device)

    info = {
        "n_batch": n_batch,
        "n_cost": data.shape[0],
        "n_cost_columns": spot_n_cost_columns(cost_cols),
        "n_centers": len(spotting.centers),
        "n_shifts": len(spotting.shifts),
        "cost_extent": cost_extent,
        "spot_cost_radius_weight": spot_cost_radius_weight,
        "spot_cost_radii": list(cost_radii),
        "present_fit": present_fit,
        "i_baseline": float(i_baseline),
        "i_bright": float(i_bright),
        "i_dark": float(i_dark),
        "polarity": str(polarity),
        "t_on": int(t_on),
        "maxtime": int(maxtime),
        "mode": "network",
    }
    return ShiftedTarget(
        signal=signal,
        data=data,
        power=power,
        cost_weight=cost_weight,
        cost_radius=cost_radius,
        readout_batch=readout_batch,
        readout_unit=readout_unit,
        n_batch=n_batch,
        info=info,
    )
