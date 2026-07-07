# -*- coding: utf-8 -*-
"""Hex spot training target for connectome multi-column training.

For every (spot, shift) stimulus the connectome is driven at ONE column; each
fit-cell readout is compared to ``RecF(r) * ImpR(t)`` where ``r`` is the
**Euclidean** hex distance (in column units) from the stimulated column to the
readout cell's column. The extent-2 ring is NOT iso-distant: 6 corners sit at
r=2, 6 edge midpoints at r=sqrt(3). ``RecF`` is sampled from the continuous
analytic Gaussian profile (``Medulla_Library.read_RecF_ImpR`` -> RecF_data, 45
samples centred on index 22; column distance r maps to sample 22 + 5r), so the
r=sqrt(3) edge target is evaluated at its true radius rather than snapped to
col +/-2 (which would mis-sign L1's centre-surround near its ~1.6 col zero
crossing).

Each spot ring is weighted by 1/(columns in that ring) so the 4 radii
(0,1,sqrt3,2) contribute equally and the low-SNR outer surround can't dominate.

``build_shifted_target`` returns everything the simulator needs:
    signal          (B, T, N)   per-batch stimulus current
    data            (n_cost, T') target trace per cost cell (T' = response window)
    power           scalar       weighted target power for cost normalisation
    cost_weight     (n_cost,)    1/ring-size weight per cost cell
    cost_radius     (n_cost,)    Euclidean ring radius {0,1,sqrt3,2,...} per cost cell
    readout_batch   (n_cost,)    which batch (stimulus) each cost cell belongs to
    readout_unit    (n_cost,)    which unit each cost cell is

``info`` includes ``n_cost`` (readout rows) and ``n_cost_columns`` (member
columns in ``cost_extent`` per stimulus batch, excluding batch/shift/cell count).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from Medulla_Library import DATA_AMP, I_DARK, I_BASELINE, I_BRIGHT, read_RecF_ImpR
from training_config import IMPULSE_MAXTIME, SIM_DTYPE_DEFAULT, T_ON
from .stimulus import column_in_cost_extent
from .spotting import (
    FIT_CELL_TYPES,
    col2fit,
    euclid_hex_dist,
    spot_stimulus_batches,
    spotting_from_opts,
    unit_type_names,
)

CENTER_COLUMN_UV = (0, 0)

# RF sample index of the receptive-field centre, and samples per column step
# (data[i,j] = RecF_data[i, 5j+2]; j=4 -> sample 22 -> r=0).
_RF_CENTER_SAMPLE = 22
_RF_SAMPLES_PER_COL = 5
_RF_NSAMPLES = 45


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


def _recf_at(recf_row: np.ndarray, r: float) -> float:
    """Sample the continuous RF profile at column distance ``r`` (interpolated)."""
    idx = _RF_CENTER_SAMPLE + _RF_SAMPLES_PER_COL * r
    idx = min(max(idx, 0.0), _RF_NSAMPLES - 1)
    return float(np.interp(idx, np.arange(_RF_NSAMPLES), recf_row))


def spot_cost_columns(batches, spotting, cost_extent):
    """Tile member columns in cost_extent: ``(batch, mu, mv, radius)`` each."""
    cols = []
    for b, (su, sv, center) in enumerate(batches):
        for du, dv in spotting.members:
            mu = center[0] + du
            mv = center[1] + dv
            if not column_in_cost_extent(mu, mv, cost_extent):
                continue
            r = euclid_hex_dist(mu - su, mv - sv)
            cols.append((b, int(mu), int(mv), float(r)))
    return cols


def spot_n_cost_columns(cost_cols):
    """Member columns in ``cost_extent`` per stimulus batch (uniform across batches)."""
    if not cost_cols:
        return 0
    counts = {}
    for b, _mu, _mv, _r in cost_cols:
        counts[b] = counts.get(b, 0) + 1
    vals = set(counts.values())
    if len(vals) != 1:
        raise ValueError(
            f"spot n_cost_columns varies by batch: "
            f"{ {b: counts[b] for b in sorted(counts)} }",
        )
    return next(iter(vals))


def spot_cost_unit_ring_layout(C, batches, spotting, cost_extent):
    """All units on :func:`spot_cost_columns`, with batch and stim-centred radius."""
    u = C.u.detach().cpu().numpy() if hasattr(C.u, "detach") else np.asarray(C.u)
    v = C.v.detach().cpu().numpy() if hasattr(C.v, "detach") else np.asarray(C.v)
    type_all = (
        C.node_type.detach().cpu().numpy()
        if hasattr(C.node_type, "detach") else np.asarray(C.node_type)
    )
    batch_idx, unit_idx, radius, type_idx = [], [], [], []
    for b, mu, mv, r in spot_cost_columns(batches, spotting, cost_extent):
        on_col = (u == mu) & (v == mv)
        for uid in np.where(on_col)[0]:
            batch_idx.append(b)
            unit_idx.append(int(uid))
            radius.append(r)
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
    share_edges: bool = False,
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
    sim_dtype: torch.dtype = SIM_DTYPE_DEFAULT,
) -> ShiftedTarget:
    if polarity not in ("bright", "dark"):
        raise ValueError(f"polarity must be 'bright' or 'dark', got {polarity!r}")
    i_step = float(i_bright if polarity == "bright" else i_dark)
    device = device or C.device
    recf_data, impr_data = read_RecF_ImpR()  # (13,45), (13,IMPULSE_MAXTIME)
    fit_row = {ft: i for i, ft in enumerate(FIT_CELL_TYPES)}

    spotting = spotting_from_opts(
        C, spot_extent, share_edges, shift_extent, single_spot,
    )
    names = unit_type_names(C)
    present_fit = [ft for ft in FIT_CELL_TYPES if ft in set(names.tolist())]

    batches = spot_stimulus_batches(spotting)
    n_batch = len(batches)

    signal = torch.zeros((n_batch, maxtime, C.n_units), dtype=sim_dtype, device=device)
    for b, (su, sv, _center) in enumerate(batches):
        units = C.input_units_at(su, sv)
        if len(units):
            idx = torch.as_tensor(units, dtype=torch.long, device=device)
            signal[b, :t_on, idx] = i_baseline
            signal[b, t_on:, idx] = i_step

    resp = slice(t_on, maxtime)  # response window (matches Borst data[t_on:maxtime])
    Tp = maxtime - t_on

    # Per (batch, radius) ring size counted in COLUMNS (not cells), so every
    # spot ring gets weight 1/columns -> the 4 radii contribute equally.
    cost_cols = spot_cost_columns(batches, spotting, cost_extent)
    col_count = {}
    for b, _mu, _mv, r in cost_cols:
        rr = round(r, 6)
        col_count[(b, rr)] = col_count.get((b, rr), 0) + 1

    r_batch, r_unit, r_radius, r_target, r_weight = [], [], [], [], []
    for b, mu, mv, r in cost_cols:
        w = 1.0 / col_count[(b, round(r, 6))]
        for ft in present_fit:
                units = col2fit(C, mu, mv, ft, names)
                if len(units) == 0:
                    continue
                row = fit_row[ft]
                amp = _recf_at(recf_data[row], r)
                trace = amp * impr_data[row][resp] * data_amp
                if polarity == "dark":
                    trace = -trace
                for uidx in units:
                    r_batch.append(b)
                    r_unit.append(int(uidx))
                    r_radius.append(r)
                    r_target.append(trace)
                    r_weight.append(w)

    if not r_batch:
        raise ValueError("no spot cost cells (check cost_extent and fit cell types)")

    data = torch.tensor(np.asarray(r_target), dtype=sim_dtype, device=device)  # (n_cost,T')
    cost_weight = torch.tensor(np.asarray(r_weight), dtype=sim_dtype, device=device)
    cost_radius = torch.tensor(np.asarray(r_radius), dtype=sim_dtype, device=device)
    readout_batch = torch.tensor(np.asarray(r_batch), dtype=torch.long, device=device)
    readout_unit = torch.tensor(np.asarray(r_unit), dtype=torch.long, device=device)

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
        "present_fit": present_fit,
        "share_edges": share_edges,
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
