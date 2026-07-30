# -*- coding: utf-8 -*-
"""Spot paradigm DATA: RecF x ImpR target traces and cost-ring layout.

Merges the old ``Medulla_Library`` RecF/ImpR reader (with its internal
bandpass/lowpass ImpR shaping -- a target-only signal path, NOT the Ca filter)
and ``network.spot_target`` Section B (target assembly + Euclidean cost rings).

New features handled here:
- ``pulse_ms`` (#1): the PR drive comes from
  :func:`task.spot.input.spot_input_waveform`, shared by the network signal
  and the ImpR target.
- ``readout_kind='v'`` (#2): the ImpR-based Ca-proxy target is inverted with
  :func:`neuron.filter_ca.ca_to_v_delta` to a ``'v'`` target and power is
  recomputed on it.

Sparse cost time points (#4) and the ``TargetPack`` wrapping live in the
``training`` layer, which reads the :class:`ShiftedTarget` returned here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from network import path  # noqa: F401 -- FAFBv783 on sys.path
from connectome_io import parse_comma_list

from network.construction import I_BASELINE, I_BRIGHT, I_DARK
from network.connectivity import SIM_DTYPE_DEFAULT
from neuron.params import DATA_AMP
import neuron.params as params
from neuron.filter_ca import ca_to_v_delta
from network.construction import col2fit, unit_type_names
from network.layout import column_in_cost_extent
from task.spot.input import (
    DEFAULT_FULLY_INSIDE,
    DEFAULT_MULTI_SPOT,
    DEFAULT_SHIFT_EXTENT,
    DEFAULT_SPOT_EXTENT,
    SpotBatch,
    euclid_hex_dist,
    members_by_euclid_radius,
    spot_extent_folds_r2_into_r1,
    spot_input_waveform,
    spot_stimulus_batches,
    spot_from_opts,
)

# ImpR / RecF target row order (13 fit cells).
cell_list = np.array(
    ['L1', 'L2', 'L3', 'L4', 'L5', 'Mi1', 'Tm3', 'Mi4', 'Mi9', 'Tm1', 'Tm2', 'Tm4', 'Tm9']
)

# Spot paradigm polarities (distinct from the target NAMES in training.config).
SPOT_POLARITIES = frozenset({"bright", "dark"})
_SPOT_STEP_KEY = {"bright": "i_bright", "dark": "i_dark"}

# RF sample index of the receptive-field center, and samples per column step
# (data[i,j] = RecF_data[i, 5j+2]; j=4 -> sample 22 -> radius=0).
_RF_CENTER_SAMPLE = 22
_RF_SAMPLES_PER_COL = 5
_RF_NSAMPLES = 45

# Euclidean radii recognised by ``--spot-cost-r-w``.
DEFAULT_SPOT_COST_RADII: Tuple[float, ...] = (0.0, 1.0, math.sqrt(3), 2.0)

DEFAULT_SPOT_COST_RADIUS_WEIGHT: Dict[float, float] = {
    0.0: 1.0,
    1.0: 1.0 / 6.0,
    2.0: 1.0 / 6.0,
}
DEFAULT_SPOT_COST_RADIUS_WEIGHT_EXTENT1: Dict[float, float] = {
    0.0: 1.0,
    1.0: 1.0 / 6.0,
}
SPOT_COST_RADIUS_KEY_ALIASES: Dict[str, float] = {
    "sqrt3": math.sqrt(3),
}


# -- RecF / ImpR readers (old Medulla_Library) --------------------------------
# Target-only ImpR shaping helpers (NOT the Ca filter). Inlined from the old
# blindschleiche_py3 module so spot/data owns this path alone.


def _gauss1d(fwhm, rfsize):
    myrange = rfsize / 2
    sigma = fwhm / (2.0 * np.sqrt(2 * np.log(2)))
    x = np.arange(-myrange, (myrange + 1), 1) * 1.0
    z = np.exp(-x ** 2 / (2 * (sigma ** 2)))
    return z / np.sum(z)


def _lowpass(x, tau):
    x = x.transpose(np.roll(np.arange(x.ndim), 1))
    n = x.shape[0]
    result = np.zeros_like(x)
    if tau < 1:
        result = x
    if tau >= 1:
        result[0] = x[0]
        for i in range(0, n - 1):
            result[i + 1] = 1.0 / tau * (x[i] - result[i]) + result[i]
    return result.transpose(np.roll(np.arange(result.ndim), -1))


def _highpass(x, tau):
    return x - _lowpass(x, tau)


def _bandpass(signal, hp_tau, lp_tau):
    result = _lowpass(signal, lp_tau)
    if hp_tau != 0:
        result = _highpass(result, hp_tau)
    return result


def normalize_data(x):
    x = x - x[0]
    mymax = np.nanmax(x)
    mymin = np.nanmin(x)
    absmax = np.abs(mymax) if np.abs(mymax) > np.abs(mymin) else np.abs(mymin)
    result = x / absmax
    if mymax == mymin:
        result = x * 0.0
    return result


def read_RecF_ImpR(*, t_on=None, n_t=None, pulse_ms=None):
    """Return ``(RecF_data, ImpR_data)`` for the 13 fit cell types.

    Shapes: ``RecF_data`` ``(13, 45)``; ``ImpR_data`` ``(13, n_t)``. The
    drive is :func:`task.spot.input.spot_input_waveform` (step or pulse).
    """
    if t_on is None or n_t is None:
        raise ValueError("read_RecF_ImpR requires t_on and n_t")
    t_on = int(t_on)
    n_t = int(n_t)

    RF_center_width = np.array([6, 7, 6, 8, 7, 6, 12, 6, 6, 8, 8, 11, 7])
    RF_surrnd_width = np.array([41, 29, 15, 33, 31, 29, 7, 16, 24, 27, 31, 35, 24])
    RF_surrnd_weight = np.array(
        [0.012, 0.013, 0.19, 0.046, 0.035, 0.022, 0.000, 0.132, 0.063, 0.040, 0.035, 0.054, 0.046]
    ) * 5.0
    RF_sign = np.array([-1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, -1])

    RecF_data = np.zeros((13, 45))
    for i in range(13):
        center = _gauss1d(RF_center_width[i], 44)
        surrnd = _gauss1d(RF_surrnd_width[i], 44)
        RecF_data[i] = (center - RF_surrnd_weight[i] * surrnd) * RF_sign[i]
        RecF_data[i] = normalize_data(RecF_data[i])

    # hp and lp time constants * 10 ms
    IR_hp = np.array([39.1, 28.8, 00.0, 38.1, 12.7, 31.8, 26.0, 0.00, 0.00, 29.6, 15.3, 24.9, 0.00])
    IR_lp = np.array([03.8, 05.8, 05.4, 02.3, 04.2, 05.4, 02.7, 03.8, 07.7, 04.4, 01.4, 02.4, 10.7])

    signal = spot_input_waveform(t_on, n_t, pulse_ms)
    signal = _lowpass(signal, 5)
    signal = signal / np.max(signal)

    ImpR_data = np.zeros((13, n_t))
    for i in range(13):
        if IR_hp[i] == 0:
            ImpR_data[i] = _lowpass(signal, IR_lp[i])
        else:
            ImpR_data[i] = _bandpass(signal, IR_hp[i], IR_lp[i])
        if i < 2:  # L1 and L2
            ImpR_data[i] = ImpR_data[i] + 0.4 * signal
        ImpR_data[i] = normalize_data(ImpR_data[i])

    return RecF_data, ImpR_data


def read_RecF_data(*, t_on=None, n_t=None, pulse_ms=None):
    """Spatial x temporal spot cube ``(13, 9, n_t)``."""
    RecF_data, ImpR_data = read_RecF_ImpR(t_on=t_on, n_t=n_t, pulse_ms=pulse_ms)
    mt = ImpR_data.shape[1]
    data = np.zeros((13, 9, mt))
    for i in range(13):
        for j in range(9):
            data[i, j] = RecF_data[i, j * 5 + 2] * ImpR_data[i]
    return data


def read_RecF_data_dark(*, t_on=None, n_t=None, pulse_ms=None):
    """Dark spot spatial x temporal cube: negated bright ``read_RecF_data()``."""
    return -read_RecF_data(t_on=t_on, n_t=n_t, pulse_ms=pulse_ms)


# -- Cost-radius weights ------------------------------------------------------


def normalize_spot_cost_radius_key(key) -> float:
    if isinstance(key, (int, float)):
        return round(float(key), 6)
    text = str(key).strip().lower()
    if text in SPOT_COST_RADIUS_KEY_ALIASES:
        return round(SPOT_COST_RADIUS_KEY_ALIASES[text], 6)
    return round(float(text), 6)


def parse_spot_cost_radius_weight_value(text: str) -> float:
    tok = str(text).strip()
    if "/" in tok:
        num, den = tok.split("/", 1)
        return float(num.strip()) / float(den.strip())
    return float(tok)


def default_spot_cost_radius_weight(
    spot_extent: float = DEFAULT_SPOT_EXTENT,
) -> Dict[float, float]:
    if spot_extent_folds_r2_into_r1(spot_extent):
        return dict(DEFAULT_SPOT_COST_RADIUS_WEIGHT_EXTENT1)
    return dict(DEFAULT_SPOT_COST_RADIUS_WEIGHT)


def spot_cost_radius_weight_resolved(
    spot_cost_radius_weight: Optional[Dict[float, float]] = None,
    spot_extent: float = DEFAULT_SPOT_EXTENT,
) -> Dict[float, float]:
    if spot_cost_radius_weight is None:
        return default_spot_cost_radius_weight(spot_extent)
    return spot_cost_radius_weight


def expand_spot_cost_r_w_dict(
    kv: Optional[dict] = None,
    *,
    stimulus_opts: Optional[dict] = None,
) -> Optional[Dict[float, float]]:
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
    tokens = parse_comma_list(text)
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
    weights = spot_cost_radius_weight_resolved(spot_cost_radius_weight, spot_extent)
    return float(weights.get(round(radius, 6), 0.0))


# -- RecF sampling / superposed target ----------------------------------------


def _recf_at(recf_row: np.ndarray, radius: float) -> float:
    idx = _RF_CENTER_SAMPLE + _RF_SAMPLES_PER_COL * radius
    idx = min(max(idx, 0.0), _RF_NSAMPLES - 1)
    return float(np.interp(idx, np.arange(_RF_NSAMPLES), recf_row))


def _spot_target_amp(recf_row: np.ndarray, radius: float, spot_extent: float) -> float:
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
    """Cost readouts: ``(batch, mu, mv, radius_key, su, sv)`` per stim ring."""
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
    u_all = C.u.detach().cpu().numpy() if hasattr(C.u, "detach") else np.asarray(C.u)
    v_all = C.v.detach().cpu().numpy() if hasattr(C.v, "detach") else np.asarray(C.v)
    stim_u = np.asarray(stim_u, dtype=np.int64)
    stim_v = np.asarray(stim_v, dtype=np.int64)
    mu = u_all[unit_idx]
    mv = v_all[unit_idx]
    return mu - stim_u, mv - stim_v


def spot_center_bin_layout(C, batches, cost_radii, cost_extent):
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


def build_shifted_target(
    C,
    spot_extent: float = DEFAULT_SPOT_EXTENT,
    multi_spot: bool = DEFAULT_MULTI_SPOT,
    fully_inside: bool = DEFAULT_FULLY_INSIDE,
    shift_extent: int = DEFAULT_SHIFT_EXTENT,
    n_t: int = None,
    t_on: int = None,
    i_baseline: float = I_BASELINE,
    i_bright: float = I_BRIGHT,
    i_dark: float = I_DARK,
    polarity: str = "bright",
    data_amp: float = DATA_AMP,
    device: Optional[str] = None,
    cost_extent: Optional[int] = None,
    spot_cost_radius_weight: Optional[Dict[float, float]] = None,
    sim_dtype: torch.dtype = SIM_DTYPE_DEFAULT,
    pulse_ms: Optional[float] = None,
    readout_kind: str = "ca",
) -> ShiftedTarget:
    if t_on is None or n_t is None:
        raise ValueError("build_shifted_target requires t_on and n_t")
    if polarity not in ("bright", "dark"):
        raise ValueError(f"polarity must be 'bright' or 'dark', got {polarity!r}")
    if readout_kind not in ("ca", "v"):
        raise ValueError(f"readout_kind must be 'ca' or 'v', got {readout_kind!r}")
    i_step = float(i_bright if polarity == "bright" else i_dark)
    device = device or C.device
    recf_data, impr_data = read_RecF_ImpR(t_on=t_on, n_t=n_t, pulse_ms=pulse_ms)
    fit_row = {str(ft): i for i, ft in enumerate(cell_list)}

    spot = spot_from_opts(
        C, spot_extent, shift_extent,
        multi_spot=multi_spot, fully_inside=fully_inside,
    )
    names = unit_type_names(C)
    present_fit = [str(ft) for ft in cell_list if str(ft) in set(names.tolist())]

    batches = spot_stimulus_batches(spot)
    n_batch = len(batches)

    # Single PR waveform source (step or pulse) shared with the ImpR target.
    u = spot_input_waveform(t_on, n_t, pulse_ms)
    drive = torch.as_tensor(
        i_baseline + (i_step - i_baseline) * u, dtype=sim_dtype, device=device,
    )
    signal = torch.zeros((n_batch, n_t, C.n_units), dtype=sim_dtype, device=device)
    for b, batch in enumerate(batches):
        for su, sv in batch.stim_uv:
            units = C.input_units_at(su, sv)
            if len(units):
                idx = torch.as_tensor(units, dtype=torch.long, device=device)
                signal[b, :, idx] = drive[:, None]

    resp = slice(t_on, n_t)  # post-onset cost window

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

    # #2 V training: the ImpR target is a Ca-proxy; invert to ``'v'`` so it can
    # be compared to the model's v_delta readout. Post-onset slice -> t_on=0.
    if readout_kind == "v":
        data = ca_to_v_delta(data, t_on=0)

    power = torch.sum(cost_weight[:, None] * data ** 2)
    if float(power) == 0.0:
        power = torch.tensor(1.0, dtype=sim_dtype, device=device)

    info = {
        "n_batch": n_batch,
        "n_cost": data.shape[0],
        "n_cost_columns": spot_n_cost_columns(cost_cols),
        "n_centers": len(spot.centers),
        "n_shifts": len(spot.shifts),
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
        "pulse_ms": None if pulse_ms is None else float(pulse_ms),
        "readout_kind": str(readout_kind),
        "t_on": int(t_on),
        "n_t": int(n_t),
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


def make_spot_stimulus_opts(
    polarity: str,
    *,
    i_baseline=None,
    i_step=None,
    mode="network",
    shift_extent=None,
    spot_extent=None,
    **extra,
):
    """PR step/pulse stimulus opts for ``spot_{polarity}``."""
    if polarity not in SPOT_POLARITIES:
        raise ValueError(f"spot polarity must be 'bright' or 'dark', got {polarity!r}")
    if shift_extent is None:
        shift_extent = extra.get("shift_extent", DEFAULT_SHIFT_EXTENT)
    step_key = _SPOT_STEP_KEY[polarity]
    if i_step is None:
        i_step = extra.get(step_key)
    step_default = I_BRIGHT if polarity == "bright" else I_DARK
    if spot_extent is None:
        spot_extent = extra.get("spot_extent", DEFAULT_SPOT_EXTENT)
    _pre_ms = extra.get("pre_ms")
    _response_ms = extra.get("response_ms")
    if _pre_ms is None or _response_ms is None:
        raise ValueError(
            "make_spot_stimulus_opts requires pre_ms and response_ms "
            "(pass via CLI --pre-ms / --response-ms)"
        )
    opts = {
        "mode": mode,
        "i_baseline": float(I_BASELINE if i_baseline is None else i_baseline),
        step_key: float(step_default if i_step is None else i_step),
        "pre_ms": float(_pre_ms),
        "response_ms": float(_response_ms),
        "delta_ms": float(params.delta_ms),
        "shift_extent": int(shift_extent),
        "spot_extent": float(spot_extent),
        "multi_spot": bool(extra.get("multi_spot", DEFAULT_MULTI_SPOT)),
        "fully_inside": bool(extra.get("fully_inside", DEFAULT_FULLY_INSIDE)),
    }
    # #1 pulse duration, #4 sparse cost time points, #2 readout kind.
    if extra.get("pulse_ms") is not None:
        opts["pulse_ms"] = float(extra["pulse_ms"])
    if extra.get("cost_interval_ms") is not None:
        opts["cost_interval_ms"] = float(extra["cost_interval_ms"])
    if extra.get("filter"):
        opts["filter"] = str(extra["filter"])
    return opts
