# -*- coding: utf-8 -*-
"""Hex radial training target for connectome multi-column training.

For every (tile, shift) stimulus the connectome is driven at ONE column; each
fit-cell readout is compared to ``RecF(r) * ImpR(t)`` where ``r`` is the
**Euclidean** hex distance (in column units) from the stimulated column to the
readout cell's column. The extent-2 ring is NOT iso-distant: 6 corners sit at
r=2, 6 edge midpoints at r=sqrt(3). ``RecF`` is sampled from the continuous
analytic Gaussian profile (``Medulla_Library.read_RecF_ImpR`` -> RecF_data, 45
samples centred on index 22; column distance r maps to sample 22 + 5r), so the
r=sqrt(3) edge target is evaluated at its true radius rather than snapped to
col +/-2 (which would mis-sign L1's centre-surround near its ~1.6 col zero
crossing).

Each radial ring is weighted by 1/(columns in that ring) so the 4 radii
(0,1,sqrt3,2) contribute equally and the low-SNR outer surround can't dominate.

``build_shifted_target`` returns everything the simulator needs:
    signal          (B, T, N)   per-batch stimulus current
    data            (n_cost, T') target trace per cost cell (T' = response window)
    power           scalar       weighted target power for cost normalisation
    cost_weight     (n_cost,)    1/ring-size weight per cost cell
    cost_radius     (n_cost,)    Euclidean ring radius {0,1,sqrt3,2,...} per cost cell
    readout_batch   (n_cost,)    which batch (stimulus) each cost cell belongs to
    readout_unit    (n_cost,)    which unit each cost cell is
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from Medulla_Library import read_RecF_ImpR
from connectome_tiling import (
    FIT_CELL_TYPES,
    build_tiling,
    col2fit,
    euclid_hex_dist,
    unit_type_names,
)

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


def build_shifted_target(
    C,
    tile_extent: int = 2,
    share_edges: bool = False,
    single_tile: Optional[bool] = None,
    single_shift: bool = False,
    maxtime: int = 200,
    t_on: int = 50,
    signal_baseline: float = 20.0,
    signal_amp: float = 40.0,
    data_amp: float = 20.0,
    device: Optional[str] = None,
) -> ShiftedTarget:
    device = device or C.device
    recf_data, impr_data = read_RecF_ImpR()  # (13,45), (13,200)
    fit_row = {ft: i for i, ft in enumerate(FIT_CELL_TYPES)}

    tiling = build_tiling(C, tile_extent, share_edges, single_tile)
    if single_shift:
        tiling.shifts = [(0, 0)]
    names = unit_type_names(C)
    present_fit = [ft for ft in FIT_CELL_TYPES if ft in set(names.tolist())]

    # one batch per (tile centre, shift); stimulus = photoreceptors at centre+shift.
    batches = []  # (stim_u, stim_v, center)
    for center in tiling.centers:
        for du, dv in tiling.shifts:
            batches.append((center[0] + du, center[1] + dv, center))
    n_batch = len(batches)

    signal = torch.zeros((n_batch, maxtime, C.n_units), dtype=torch.float64, device=device)
    for b, (su, sv, _center) in enumerate(batches):
        units = C.input_units_at(su, sv)
        if len(units):
            idx = torch.as_tensor(units, dtype=torch.long, device=device)
            signal[b, :t_on, idx] = signal_baseline
            signal[b, t_on:, idx] = signal_amp

    resp = slice(t_on, maxtime)  # response window (matches Borst data[50:200])
    Tp = maxtime - t_on

    # Per (batch, radius) ring size counted in COLUMNS (not cells), so every
    # radial ring gets weight 1/columns -> the 4 radii contribute equally.
    col_count = {}
    for b, (su, sv, center) in enumerate(batches):
        for du, dv in tiling.members:
            mu, mv = center[0] + du, center[1] + dv
            rr = round(euclid_hex_dist(mu - su, mv - sv), 6)
            col_count[(b, rr)] = col_count.get((b, rr), 0) + 1

    r_batch, r_unit, r_radius, r_target, r_weight = [], [], [], [], []
    for b, (su, sv, center) in enumerate(batches):
        for du, dv in tiling.members:
            mu, mv = center[0] + du, center[1] + dv
            r = euclid_hex_dist(mu - su, mv - sv)
            w = 1.0 / col_count[(b, round(r, 6))]
            for ft in present_fit:
                units = col2fit(C, mu, mv, ft, names)
                if len(units) == 0:
                    continue
                row = fit_row[ft]
                amp = _recf_at(recf_data[row], r)
                trace = amp * impr_data[row][resp] * data_amp  # (T',)
                for uidx in units:
                    r_batch.append(b)
                    r_unit.append(int(uidx))
                    r_radius.append(r)
                    r_target.append(trace)
                    r_weight.append(w)

    data = torch.tensor(np.asarray(r_target), dtype=torch.float64, device=device)  # (n_cost,T')
    cost_weight = torch.tensor(np.asarray(r_weight), dtype=torch.float64, device=device)
    cost_radius = torch.tensor(np.asarray(r_radius), dtype=torch.float64, device=device)
    readout_batch = torch.tensor(np.asarray(r_batch), dtype=torch.long, device=device)
    readout_unit = torch.tensor(np.asarray(r_unit), dtype=torch.long, device=device)

    power = torch.sum(cost_weight[:, None] * data ** 2)
    if float(power) == 0.0:
        power = torch.tensor(1.0, dtype=torch.float64, device=device)

    info = {
        "n_batch": n_batch,
        "n_cost": data.shape[0],
        "n_centers": len(tiling.centers),
        "n_shifts": len(tiling.shifts),
        "present_fit": present_fit,
        "share_edges": share_edges,
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
