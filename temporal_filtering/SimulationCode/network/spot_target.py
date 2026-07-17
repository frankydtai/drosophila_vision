# -*- coding: utf-8 -*-
"""Connectome spotting layout and spot training targets.

Section A — connectome spotting (``Spotting``, ``build_spotting``, …).
Section B — training target (``build_shifted_target``, ``spot_cost_*``, …).

Hex disc enumeration lives in :mod:`column_mapper`; spot centre tiling and
Euclidean cost-radius grouping live in this module.

For every shift batch all spot centres step together at ``t_on`` (one batch per
shift, ``B = n_shifts``). Each fit-cell readout is compared to a
**superposed** target
``(Σ_{(su,sv) in batch} _spot_target_amp(RecF, dist, spot_extent)) * ImpR(t)``
where ``dist`` is the Euclidean hex distance from each simultaneous stim column
``(su, sv)`` to the readout column ``(mu, mv)``. Cost rows are enumerated per
stim and radius ring (no dedup); weight uses ``radius_key`` from
``--spot-cost-r-w``, target uses the full batch sum above (cached per
``(batch, mu, mv, fit type)``). The extent-2 patch is NOT iso-distant: 6 corners sit
at radius=2, 6 edge midpoints at radius=sqrt(3). ``RecF`` is sampled from the
continuous analytic Gaussian profile (``Medulla_Library.read_RecF_ImpR`` ->
RecF_data, 45 samples centred on index 22; column distance maps to sample 22 +
5*radius), so the sqrt(3) edge target is evaluated at its true radius rather
than snapped to col +/-2 (which would mis-sign L1's centre-surround near its
~1.6 col zero crossing).

When ``spot_extent == 1`` (:func:`spot_extent_folds_r2_into_r1`):

- r=1 target amplitude is ``RecF(1)+RecF(2)`` (r=2 folded in).
- r=2 target amplitude is ``0`` (not a separate ring target).

Other ``spot_extent`` values use ``RecF(radius)`` at each cost radius.
``--spot-cost-r-w`` controls **cost weights only**; it never changes
:func:`_spot_target_amp`.

When ``--spot-cost-r-w`` is omitted, weights default via
:func:`default_spot_cost_radius_weight`:

- ``spot_extent == 1`` → ``0=1,1=1/6`` (no r=2; ``sqrt3`` excluded).
- otherwise → ``0=1,1=1/6,2=1/6`` (``sqrt3`` excluded).

CLI tokens match ``--cost-weight``:

- ``r=w`` → merge onto those defaults (unlisted radii keep default / stay 0).
- bare ``r`` → exclusive: all ``DEFAULT_SPOT_COST_RADII`` → 0, then listed ``r=1``.
- mix e.g. ``0,1=1/6`` → exclusive zero + ``0=1``, then override ``1=1/6``.

At ``spot_extent == 1``, listing ``2=…`` can still add r=2 rings to cost, but
:func:`_spot_target_amp` keeps target amplitude zero there (fold rule).

``build_shifted_target`` returns a :class:`ShiftedTarget` (timing: :mod:`training_config`):

    signal          (B, T, N)      per-batch stimulus current
    data            (n_cost, T')   target trace per cost cell (MSE compares this)
    power           scalar       weighted target power for cost normalisation
    cost_weight     (n_cost,)    per-cost-cell weight at each Euclidean radius
    cost_radius     (n_cost,)    Euclidean radius {0,1,sqrt3,2,...} per cost cell
    readout_batch   (n_cost,)    which batch (stimulus) each cost cell belongs to
    readout_unit    (n_cost,)    which unit each cost cell is
    readout_stim_u  (n_cost,)    stim column u for this cost row's ring anchor
    readout_stim_v  (n_cost,)    stim column v for this cost row's ring anchor

``info`` includes ``n_cost`` (readout rows) and ``n_cost_columns`` (member
columns in ``cost_extent`` per stimulus batch: ``int`` when uniform, else
``{batch: count}``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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

# Default ``--spot-cost-r-w`` when omitted and ``spot_extent != 1`` (e.g. 1.5, 2).
DEFAULT_SPOT_COST_RADIUS_WEIGHT: Dict[float, float] = {
    0.0: 1.0,
    1.0: 1.0 / 6.0,
    2.0: 1.0 / 6.0,
}

# Default ``--spot-cost-r-w`` when omitted and ``spot_extent == 1`` (no r=2).
DEFAULT_SPOT_COST_RADIUS_WEIGHT_EXTENT1: Dict[float, float] = {
    0.0: 1.0,
    1.0: 1.0 / 6.0,
}

SPOT_COST_RADIUS_KEY_ALIASES: Dict[str, float] = {
    "sqrt3": math.sqrt(3),
}

_SPOT_EXTENT_HALF_STEP_TOL = 1e-9

# Default spot footprint / centre-tiling radius (0.5 multiples).
DEFAULT_SPOT_EXTENT: float = 1.0

# Panel list for multi-spot visualisation.
DEFAULT_SPOT_EXTENTS: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)

# Keep only centres whose spot footprint lies inside connectome extent.
DEFAULT_FULLY_INSIDE: bool = True

# Tile simultaneous spot centres on network connectome (``False`` → centre ``(0, 0)`` only).
DEFAULT_MULTI_SPOT: bool = True


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
    spot_extent=DEFAULT_SPOT_EXTENT,
    fully_inside: bool = DEFAULT_FULLY_INSIDE,
) -> list:
    """Axial centres of densest packing of radius-``floor(spot_extent)`` hexes.

    Half-integer ``spot_extent`` (odd half-steps): generators ``(e, e)`` with
    ``e = spot_extent + 0.5`` (edge-sharing of the drawn axial hex).
    Integer ``spot_extent``: generators ``(2k + 1, -k)``.
    """
    m = spot_extent_half_steps(spot_extent)
    k = m // 2
    if m % 2 == 1:
        e = (m + 1) // 2
        a1, b1 = e, e
    else:
        a1, b1 = m + 1, -k
    a2, b2 = _rot60(a1, b1)
    # Footprint for fully_inside: (m+1)//2 (=k for integer extent; e for half-integer).
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


def spot_extent_folds_r2_into_r1(spot_extent) -> bool:
    """True when ``spot_extent == 1`` (``spot_extent_half_steps == 2``).

    Fold semantics (target via :func:`_spot_target_amp`; default cost radii via
    :func:`default_spot_cost_radius_weight`):

    - r=1 → ``RecF(1) + RecF(2)``
    - r=2 → amplitude ``0``; omitted from default ``--spot-cost-r-w`` (weight 0).
    """
    return spot_extent_half_steps(spot_extent) == 2


def default_spot_cost_radius_weight(
    spot_extent: float = DEFAULT_SPOT_EXTENT,
) -> Dict[float, float]:
    """Default ``--spot-cost-r-w`` when the CLI flag is omitted.

    ``spot_extent == 1``: ``0=1, 1=1/6`` only (r=2 folded — not listed).
    Otherwise: ``0=1, 1=1/6, 2=1/6``. ``sqrt3`` is never in the default map.
    """
    if spot_extent_folds_r2_into_r1(spot_extent):
        return dict(DEFAULT_SPOT_COST_RADIUS_WEIGHT_EXTENT1)
    return dict(DEFAULT_SPOT_COST_RADIUS_WEIGHT)


def spot_cost_radius_weight_resolved(
    spot_cost_radius_weight: Optional[Dict[float, float]] = None,
    spot_extent: float = DEFAULT_SPOT_EXTENT,
) -> Dict[float, float]:
    """Resolved per-radius cost weights for ``spot_extent``.

    ``None`` → :func:`default_spot_cost_radius_weight`. Explicit CLI dict is the
    already-merged map from :func:`parse_spot_cost_r_w` / sidecar; missing radii
    → weight 0 at lookup.
    """
    if spot_cost_radius_weight is None:
        return default_spot_cost_radius_weight(spot_extent)
    return spot_cost_radius_weight


def expand_spot_cost_r_w_dict(
    kv: Optional[dict] = None,
    *,
    stimulus_opts: Optional[dict] = None,
) -> Optional[Dict[float, float]]:
    """Normalize a radius→weight dict; empty/None → unresolved (use default).

    With ``stimulus_opts``, read ``spot_cost_radius_weight`` from a stimulus sidecar.
    """
    if stimulus_opts is not None:
        kv = (stimulus_opts or {}).get("spot_cost_radius_weight")
    if not kv:
        return None
    return {
        normalize_spot_cost_radius_key(k): parse_spot_cost_radius_weight_value(v)
        for k, v in kv.items()
    }


def parse_spot_cost_r_w_tokens(
    text: str,
    spot_extent: float = DEFAULT_SPOT_EXTENT,
) -> Optional[Dict[float, float]]:
    """Parse ``--spot-cost-r-w`` tokens (same exclusive / merge rules as cost-weight).

    Empty → ``None`` (caller uses :func:`default_spot_cost_radius_weight`).
    Bare radius keys (``0``, ``sqrt3``, …) zero all ``DEFAULT_SPOT_COST_RADII``
    then set those radii to ``1``. ``r=w`` merges onto extent defaults (or onto
    the exclusive map when mixed with bare keys).
    """
    tokens = [t.strip() for t in str(text or "").split(",") if t.strip()]
    if not tokens:
        return None
    bare: list[float] = []
    explicit: Dict[float, float] = {}
    for tok in tokens:
        if "=" in tok:
            key, val = tok.split("=", 1)
            explicit[normalize_spot_cost_radius_key(key)] = (
                parse_spot_cost_radius_weight_value(val)
            )
        else:
            bare.append(normalize_spot_cost_radius_key(tok))
    if bare:
        weights = {round(float(r), 6): 0.0 for r in DEFAULT_SPOT_COST_RADII}
        for r in bare:
            weights[r] = 1.0
    else:
        weights = default_spot_cost_radius_weight(spot_extent)
    weights.update(explicit)
    return weights


def resolve_spot_cost_radii(
    spot_cost_radius_weight: Optional[Dict[float, float]] = None,
    *,
    spot_extent: float = DEFAULT_SPOT_EXTENT,
    stimulus_opts: Optional[dict] = None,
) -> Tuple[float, ...]:
    """Euclidean radii with non-zero cost weight (subset of ``DEFAULT_SPOT_COST_RADII``).

    Weights from :func:`spot_cost_radius_weight_resolved` (defaults depend on
    ``spot_extent`` when CLI omitted).

    With ``stimulus_opts``, read ``spot_cost_radius_weight`` / ``spot_extent``
    from the stimulus sidecar first.
    """
    if stimulus_opts is not None:
        spot_cost_radius_weight = expand_spot_cost_r_w_dict(stimulus_opts=stimulus_opts)
        spot_extent = float(stimulus_opts.get("spot_extent", spot_extent))
    weights = spot_cost_radius_weight_resolved(spot_cost_radius_weight, spot_extent)
    return tuple(
        radius for radius in DEFAULT_SPOT_COST_RADII
        if float(weights.get(round(radius, 6), 0.0)) != 0.0
    )


def spot_cost_cell_weight(
    radius: float,
    spot_cost_radius_weight: Optional[Dict[float, float]],
    spot_extent: float = DEFAULT_SPOT_EXTENT,
) -> float:
    """Per-cell MSE weight for one cost-radius ring (0 if radius not listed / weight 0)."""
    weights = spot_cost_radius_weight_resolved(spot_cost_radius_weight, spot_extent)
    return float(weights.get(round(radius, 6), 0.0))


@dataclass(frozen=True)
class SpotBatch:
    """One simultaneous spot stimulus: all ``stim_uv`` columns step in one batch."""

    shift: Tuple[int, int]
    stim_uv: Tuple[Tuple[int, int], ...]


def spot_stimulus_batches(spotting: Spotting) -> List[SpotBatch]:
    """One batch per shift; each batch steps all spot centres (+ shift) together."""
    batches: List[SpotBatch] = []
    for du, dv in spotting.shifts:
        stim_uv = tuple(
            (int(cu + du), int(cv + dv))
            for cu, cv in spotting.centers
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


def build_spotting(
    C,
    spot_extent: float = DEFAULT_SPOT_EXTENT,
    multi_spot: bool = DEFAULT_MULTI_SPOT,
    fully_inside: bool = DEFAULT_FULLY_INSIDE,
) -> Spotting:
    """Build a :class:`Spotting` for connectome ``C``."""
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
    return Spotting(centers, shifts, spot_extent)


def spotting_from_opts(
    C,
    spot_extent: float = DEFAULT_SPOT_EXTENT,
    shift_extent: int = 0,
    multi_spot: bool = DEFAULT_MULTI_SPOT,
    fully_inside: bool = DEFAULT_FULLY_INSIDE,
    *,
    stimulus_opts: Optional[Dict] = None,
) -> Spotting:
    """Build :class:`Spotting` with configurable sub-spot shift radius.

    With ``stimulus_opts``, read ``spot_extent`` / ``shift_extent`` / ``multi_spot`` /
    ``fully_inside`` from a stimulus sidecar.
    """
    if stimulus_opts is not None:
        spot_extent = float(stimulus_opts.get("spot_extent", spot_extent))
        shift_extent = int(stimulus_opts.get("shift_extent", shift_extent))
        multi_spot = bool(stimulus_opts.get("multi_spot", multi_spot))
        fully_inside = bool(stimulus_opts.get("fully_inside", fully_inside))
    spotting = build_spotting(
        C, spot_extent, multi_spot=multi_spot, fully_inside=fully_inside,
    )
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
    readout_stim_u: torch.Tensor  # (n_cost,) long
    readout_stim_v: torch.Tensor  # (n_cost,) long
    n_batch: int
    info: dict


def _recf_at(recf_row: np.ndarray, radius: float) -> float:
    """Sample the continuous RF profile at column distance ``radius`` (interpolated)."""
    idx = _RF_CENTER_SAMPLE + _RF_SAMPLES_PER_COL * radius
    idx = min(max(idx, 0.0), _RF_NSAMPLES - 1)
    return float(np.interp(idx, np.arange(_RF_NSAMPLES), recf_row))


def _spot_target_amp(recf_row: np.ndarray, radius: float, spot_extent: float) -> float:
    """RecF amplitude coefficient for one cost cell (before ``* ImpR``).

    ``spot_extent == 1`` (:func:`spot_extent_folds_r2_into_r1`):

    - r=1 → ``RecF(1) + RecF(2)``
    - r=2 → ``0`` (folded; default cost also omits r=2)

    Other extents: ``RecF(radius)`` at the ring's Euclidean radius.
    """
    r = round(float(radius), 6)
    if spot_extent_folds_r2_into_r1(spot_extent):
        if r == 1.0:
            return _recf_at(recf_row, 1.0) + _recf_at(recf_row, 2.0)
        if r == 2.0:
            return 0.0
    return _recf_at(recf_row, r)


def _spot_superposed_amp(
    recf_row: np.ndarray,
    stim_uv: Sequence[Tuple[int, int]],
    mu: int,
    mv: int,
    spot_extent: float,
) -> float:
    """Sum of :func:`_spot_target_amp` over all simultaneous stim columns."""
    total = 0.0
    for su, sv in stim_uv:
        dist = round(euclid_hex_dist(int(mu) - int(su), int(mv) - int(sv)), 6)
        total += _spot_target_amp(recf_row, dist, spot_extent)
    return total


def _spot_superposed_trace(
    recf_row: np.ndarray,
    stim_uv: Sequence[Tuple[int, int]],
    mu: int,
    mv: int,
    spot_extent: float,
    impr_row: np.ndarray,
    resp: slice,
    data_amp: float,
    *,
    polarity: str,
) -> np.ndarray:
    amp = _spot_superposed_amp(recf_row, stim_uv, mu, mv, spot_extent)
    trace = amp * impr_row[resp] * data_amp
    if polarity == "dark":
        trace = -trace
    return trace


def spot_cost_columns(
    batches: Sequence[SpotBatch],
    cost_radii,
    cost_extent,
) -> List[Tuple[int, int, int, float, int, int]]:
    """Cost readouts: ``(batch, mu, mv, radius_key, su, sv)`` per stim ring (no dedup)."""
    by_radius = members_by_euclid_radius(cost_radii)
    cols: List[Tuple[int, int, int, float, int, int]] = []
    for b, batch in enumerate(batches):
        for su, sv in batch.stim_uv:
            for radius_key, members in by_radius.items():
                for du, dv in members:
                    mu, mv = su + du, sv + dv
                    if not column_in_cost_extent(mu, mv, cost_extent):
                        continue
                    cols.append((
                        b, int(mu), int(mv), float(radius_key), int(su), int(sv),
                    ))
    return cols


def spot_n_cost_columns(cost_cols):
    """Member columns in ``cost_extent`` per stimulus batch (includes multi-row dup)."""
    if not cost_cols:
        return 0
    counts: Dict[int, int] = {}
    for b, _mu, _mv, _radius, _su, _sv in cost_cols:
        counts[b] = counts.get(b, 0) + 1
    vals = set(counts.values())
    if len(vals) == 1:
        return next(iter(vals))
    return {b: counts[b] for b in sorted(counts)}


def spot_cost_unit_radius_layout(C, batches, cost_radii, cost_extent):
    """Units on :func:`spot_cost_columns` with batch, radius, and stim anchor."""
    u = C.u.detach().cpu().numpy() if hasattr(C.u, "detach") else np.asarray(C.u)
    v = C.v.detach().cpu().numpy() if hasattr(C.v, "detach") else np.asarray(C.v)
    type_all = (
        C.node_type.detach().cpu().numpy()
        if hasattr(C.node_type, "detach") else np.asarray(C.node_type)
    )
    batch_idx, unit_idx, radius, type_idx, stim_u, stim_v = [], [], [], [], [], []
    for b, mu, mv, cell_radius, su, sv in spot_cost_columns(
        batches, cost_radii, cost_extent,
    ):
        on_col = (u == mu) & (v == mv)
        for uid in np.where(on_col)[0]:
            batch_idx.append(b)
            unit_idx.append(int(uid))
            radius.append(cell_radius)
            type_idx.append(int(type_all[uid]))
            stim_u.append(int(su))
            stim_v.append(int(sv))
    return (
        np.asarray(batch_idx, dtype=np.int64),
        np.asarray(unit_idx, dtype=np.int64),
        np.asarray(radius, dtype=np.float64),
        np.asarray(type_idx, dtype=np.int64),
        np.asarray(stim_u, dtype=np.int64),
        np.asarray(stim_v, dtype=np.int64),
    )


def spot_readout_duv(C, batch_idx, unit_idx, *, stim_u, stim_v):
    """Stim-centred axial ``(du, dv)`` per readout row (per-row stim anchor)."""
    u_all = C.u.detach().cpu().numpy() if hasattr(C.u, "detach") else np.asarray(C.u)
    v_all = C.v.detach().cpu().numpy() if hasattr(C.v, "detach") else np.asarray(C.v)
    stim_u = np.asarray(stim_u, dtype=np.int64)
    stim_v = np.asarray(stim_v, dtype=np.int64)
    mu = u_all[unit_idx]
    mv = v_all[unit_idx]
    return mu - stim_u, mv - stim_v


def spot_center_bin_layout(C, batches, cost_radii, cost_extent):
    """Cost-extent unit layout plus stim-centred ``center_row`` mask.

    Returns
    -------
    batch_idx, unit_idx, radius, type_idx, stim_u, stim_v, du, dv, center_row
        ``center_row`` is True where ``(du, dv) == (0, 0)``.
    """
    batch_idx, unit_idx, radius, type_idx, stim_u, stim_v = spot_cost_unit_radius_layout(
        C, batches, cost_radii, cost_extent,
    )
    du, dv = spot_readout_duv(C, batch_idx, unit_idx, stim_u=stim_u, stim_v=stim_v)
    du = np.asarray(du, dtype=np.int64)
    dv = np.asarray(dv, dtype=np.int64)
    center_row = (du == 0) & (dv == 0)
    return (
        np.asarray(batch_idx, dtype=np.int64),
        np.asarray(unit_idx, dtype=np.int64),
        np.asarray(radius, dtype=np.float64),
        np.asarray(type_idx, dtype=np.int64),
        np.asarray(stim_u, dtype=np.int64),
        np.asarray(stim_v, dtype=np.int64),
        du,
        dv,
        center_row,
    )


def build_shifted_target(
    C,
    spot_extent: float = DEFAULT_SPOT_EXTENT,
    multi_spot: bool = DEFAULT_MULTI_SPOT,
    fully_inside: bool = DEFAULT_FULLY_INSIDE,
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
        C, spot_extent, shift_extent,
        multi_spot=multi_spot, fully_inside=fully_inside,
    )
    names = unit_type_names(C)
    present_fit = [str(ft) for ft in _CELL_LIST if str(ft) in set(names.tolist())]

    batches = spot_stimulus_batches(spotting)
    n_batch = len(batches)

    signal = torch.zeros((n_batch, maxtime, C.n_units), dtype=sim_dtype, device=device)
    for b, batch in enumerate(batches):
        for su, sv in batch.stim_uv:
            units = C.input_units_at(su, sv)
            if len(units):
                idx = torch.as_tensor(units, dtype=torch.long, device=device)
                signal[b, :t_on, idx] = i_baseline
                signal[b, t_on:, idx] = i_step

    resp = slice(t_on, maxtime)  # post-T_ON cost window; see training_config

    cost_radii = resolve_spot_cost_radii(spot_cost_radius_weight, spot_extent=spot_extent)
    cost_cols = spot_cost_columns(batches, cost_radii, cost_extent)

    cost_batch, cost_unit, cost_radius_list, cost_target, cost_weight_list = [], [], [], [], []
    cost_stim_u, cost_stim_v = [], []
    trace_cache: Dict[Tuple[int, int, int, int], np.ndarray] = {}
    for b, mu, mv, radius, su, sv in cost_cols:
        w = spot_cost_cell_weight(radius, spot_cost_radius_weight, spot_extent)
        if w == 0.0:
            continue
        stim_uv = batches[b].stim_uv
        for ft in present_fit:
            units = col2fit(C, mu, mv, ft, names)
            if len(units) == 0:
                continue
            row = fit_row[ft]
            cache_key = (b, mu, mv, row)
            if cache_key not in trace_cache:
                trace_cache[cache_key] = _spot_superposed_trace(
                    recf_data[row],
                    stim_uv,
                    mu,
                    mv,
                    spot_extent,
                    impr_data[row],
                    resp,
                    data_amp,
                    polarity=polarity,
                )
            trace = trace_cache[cache_key]
            for uidx in units:
                cost_batch.append(b)
                cost_unit.append(int(uidx))
                cost_radius_list.append(radius)
                cost_target.append(trace)
                cost_weight_list.append(w)
                cost_stim_u.append(int(su))
                cost_stim_v.append(int(sv))

    if not cost_batch:
        raise ValueError("no spot cost cells (check cost_extent and fit cell types)")

    data = torch.tensor(np.asarray(cost_target), dtype=sim_dtype, device=device)  # (n_cost,T')
    cost_weight = torch.tensor(np.asarray(cost_weight_list), dtype=sim_dtype, device=device)
    cost_radius = torch.tensor(np.asarray(cost_radius_list), dtype=sim_dtype, device=device)
    readout_batch = torch.tensor(np.asarray(cost_batch), dtype=torch.long, device=device)
    readout_unit = torch.tensor(np.asarray(cost_unit), dtype=torch.long, device=device)
    readout_stim_u = torch.tensor(np.asarray(cost_stim_u), dtype=torch.long, device=device)
    readout_stim_v = torch.tensor(np.asarray(cost_stim_v), dtype=torch.long, device=device)

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
        "spot_extent": float(spot_extent),
        "multi_spot": bool(multi_spot),
        "fully_inside": bool(fully_inside),
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
        readout_stim_u=readout_stim_u,
        readout_stim_v=readout_stim_v,
        n_batch=n_batch,
        info=info,
    )
