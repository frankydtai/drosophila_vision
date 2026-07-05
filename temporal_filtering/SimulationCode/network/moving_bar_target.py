# -*- coding: utf-8 -*-
"""Gruntman moving-bar training target: fig1_ci traces + per-column ``t_center``.

``build_moving_bar_target`` returns batched moving-bar ``signal``, per-readout
``data`` on a fixed ``COST_WINDOW_STEPS`` grid aligned to ``t_center ± 0.45 s``,
and ``cost_t0`` for :mod:`FiveCol_MedSim_Pytorch` windowed cost.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

import Medulla_Library as ml
from Medulla_Library import I_BASELINE, I_BRIGHT, I_DARK, T_ON
from network.stimulus import build_moving_bar_signals, cost_photo_columns
from t4_t5_preference import (
    READOUT_SUBTYPES,
    fig1_key_for_stimulus,
    motion_preference,
    normalize_side,
)
from training_config import (
    COST_HALF_WINDOW_STEPS,
    COST_WINDOW_STEPS,
    FIG1_CI_NPZ,
)
from visual_stimulus.moving_bar_stimulus import (
    HexColumn,
    build_batched_column_current,
    column_bar_center_step,
    field_bounds,
    gruntman_moving_bar_specs,
    moving_bar_maxtime,
    moving_bar_sweep_end_step,
)
from column_mapper import DEFAULT_KERNEL_SIZE, hex_to_pixel, hex_vertices

_TRACE_CACHE: Dict[str, np.ndarray] = {}
BORST_READOUT_SUBTYPES = ("T4a", "T4b", "T5a", "T5b")
PD_IDX, ND_IDX = 0, 1


def _pd_nd_index(pd_nd: str) -> int:
    return PD_IDX if pd_nd == "PD" else ND_IDX


@dataclass
class MovingBarTarget:
    signal: torch.Tensor          # (B, T, N)
    data: torch.Tensor            # (n_cost, COST_WINDOW_STEPS)
    power: torch.Tensor           # scalar
    cost_weight: torch.Tensor     # (n_cost,)
    cost_t0: torch.Tensor         # (n_cost,) absolute simulation step
    readout_batch: torch.Tensor   # (n_cost,) long
    readout_unit: torch.Tensor    # (n_cost,) long
    cost_pd_nd: torch.Tensor      # (n_cost,) long; 0=PD, 1=ND
    n_batch: int
    maxtime: int
    info: dict


def _fig1_trace_ids(npz_path: Path) -> List[str]:
    with np.load(npz_path) as d:
        return sorted({k.replace("__time_ms", "") for k in d.files if k.endswith("__time_ms")})


def load_fig1_trace(
    trace_id: str,
    npz_path: Path = FIG1_CI_NPZ,
    n_steps: int = COST_WINDOW_STEPS,
    half_window_steps: int = COST_HALF_WINDOW_STEPS,
    deltat_ms: float = 10.0,
) -> np.ndarray:
    """Resample one fig1 trace to ``n_steps`` centred at 0 ms (10 ms spacing).

    Digitized ``time_ms`` in the npz is -450..+450 ms relative to bar centre.
    """
    key = f"{trace_id}|{n_steps}|{half_window_steps}|{deltat_ms}"
    if key in _TRACE_CACHE:
        return _TRACE_CACHE[key]

    with np.load(npz_path) as d:
        t_key, v_key = f"{trace_id}__time_ms", f"{trace_id}__vm_mv"
        if t_key not in d.files:
            raise KeyError(f"missing trace {trace_id!r} in {npz_path}")
        time_ms = np.asarray(d[t_key], dtype=np.float64)
        vm_mv = np.asarray(d[v_key], dtype=np.float64)

    rel_ms = (np.arange(n_steps, dtype=np.float64) - half_window_steps) * deltat_ms
    trace = np.interp(rel_ms, time_ms, vm_mv, left=vm_mv[0], right=vm_mv[-1])
    _TRACE_CACHE[key] = trace
    return trace


def load_fig1_traces(
    npz_path: Path = FIG1_CI_NPZ,
    n_steps: int = COST_WINDOW_STEPS,
    half_window_steps: int = COST_HALF_WINDOW_STEPS,
    deltat_ms: float = 10.0,
) -> Dict[str, np.ndarray]:
    """All fig1 traces resampled to the per-column training window."""
    return {
        tid: load_fig1_trace(tid, npz_path, n_steps, half_window_steps, deltat_ms)
        for tid in _fig1_trace_ids(npz_path)
    }


from .tiling import unit_type_names


def col2subtype(C, u: int, v: int, subtype: str, names: Optional[np.ndarray] = None) -> np.ndarray:
    """Unit indices of ``subtype`` (e.g. ``T4a``) on column ``(u, v)``."""
    if names is None:
        names = unit_type_names(C)
    return np.where(
        (C.u == int(u)) & (C.v == int(v)) & (names == subtype),
    )[0]


def _borst_hex_columns() -> List[HexColumn]:
    """Five Borst columns on one horizontal row with 5 deg spacing."""
    cols: List[HexColumn] = []
    spacing_deg = 5.0
    for col in range(ml.nofcols):
        k = float(col - ml.CENTER_COL)
        v = (spacing_deg / DEFAULT_KERNEL_SIZE) * k
        u = -0.5 * v
        x, y = hex_to_pixel(u, v, DEFAULT_KERNEL_SIZE)
        cols.append(HexColumn(u=int(col), v=0, x=float(x), y=float(y), hex_xy=hex_vertices(x, y)))
    return cols


def _borst_moving_bar_specs(*, contrasts=("bright", "dark")):
    return gruntman_moving_bar_specs(directions=("right", "left"), contrasts=contrasts)


def _moving_bar_peak_kwargs(
    contrasts: Sequence[str],
    *,
    i_bright_bar: Optional[float] = None,
    i_dark_bar: Optional[float] = None,
) -> dict:
    """Peak-current kwargs for specs that only include the relevant contrasts."""
    contrast_set = frozenset(contrasts)
    kw = {}
    if "bright" in contrast_set and i_bright_bar is not None:
        kw["i_bright_bar"] = float(i_bright_bar)
    if "dark" in contrast_set and i_dark_bar is not None:
        kw["i_dark_bar"] = float(i_dark_bar)
    return kw


def build_moving_bar_target(
    C,
    device: Optional[str] = None,
    t_on: int = T_ON,
    deltat_ms: float = 10.0,
    fig1_path: Path = FIG1_CI_NPZ,
    use_cache: bool = True,
    center_column: bool = False,
    i_baseline: Optional[float] = None,
    i_bright_bar: Optional[float] = None,
    i_dark_bar: Optional[float] = None,
    contrasts: Sequence[str] = ("bright", "dark"),
) -> MovingBarTarget:
    """Build moving-bar stimulus + fig1 targets for photo columns × T4/T5 subtypes.

    ``center_column=True`` restricts cost to the hex centre column ``(u,v)=(0,0)``.
    Stimulus still drives all photoreceptor columns.
    """
    device = device or C.device
    side = normalize_side(C.meta.get("side", "right"))

    specs = gruntman_moving_bar_specs(contrasts=tuple(contrasts))
    contrast_set = frozenset(contrasts)
    stim = build_moving_bar_signals(
        C, specs=specs, t_on=t_on, deltat_ms=deltat_ms, device=device, use_cache=use_cache,
        network_json=getattr(C, "source_json", None),
        i_baseline=I_BASELINE if i_baseline is None else float(i_baseline),
        **_moving_bar_peak_kwargs(
            contrast_set,
            i_bright_bar=i_bright_bar,
            i_dark_bar=i_dark_bar,
        ),
    )
    maxtime = int(stim.info["maxtime"])
    field_deg = stim.info["field_deg"]
    fig1 = load_fig1_traces(fig1_path, deltat_ms=deltat_ms)

    present = [st for st in READOUT_SUBTYPES if st in set(C.type_names)]
    if not present:
        raise ValueError("network has no T4a–d / T5a–d subtypes for moving-bar target")

    type_names = unit_type_names(C)
    cols = cost_photo_columns(C, center_column=center_column)
    center_col = cols[0] if center_column else None

    r_batch, r_unit, r_target, r_weight, r_t0, r_pd_nd = [], [], [], [], [], []
    skipped_orthogonal = 0
    for b, spec in enumerate(stim.specs):
        for col in cols:
            t_center = column_bar_center_step(
                col.x, col.y, spec, field_deg, t_on=t_on, deltat_ms=deltat_ms,
            )
            t0 = t_center - COST_HALF_WINDOW_STEPS
            if t0 < 0 or t_center + COST_HALF_WINDOW_STEPS > maxtime:
                raise ValueError(
                    f"cost window out of range for column ({col.u},{col.v}) "
                    f"spec={spec.name}: t_center={t_center}, maxtime={maxtime}"
                )
            for subtype in present:
                pref = motion_preference(side, subtype, spec.direction, spec.contrast)
                if pref is None:
                    skipped_orthogonal += 1
                    continue
                trace_id = fig1_key_for_stimulus(side, subtype, spec)
                if trace_id not in fig1:
                    raise KeyError(f"fig1 trace missing: {trace_id}")
                units = col2subtype(C, col.u, col.v, subtype, type_names)
                if len(units) == 0:
                    continue
                target = fig1[trace_id]
                pd_nd_idx = _pd_nd_index(pref.pd_nd)
                for uidx in units:
                    r_batch.append(b)
                    r_unit.append(int(uidx))
                    r_target.append(target)
                    r_weight.append(1.0)
                    r_t0.append(t0)
                    r_pd_nd.append(pd_nd_idx)

    if not r_batch:
        raise ValueError("no moving-bar cost cells (check subtypes and photo columns)")

    data = torch.tensor(np.asarray(r_target), dtype=torch.float64, device=device)
    cost_weight = torch.tensor(np.asarray(r_weight), dtype=torch.float64, device=device)
    readout_batch = torch.tensor(np.asarray(r_batch), dtype=torch.long, device=device)
    readout_unit = torch.tensor(np.asarray(r_unit), dtype=torch.long, device=device)
    cost_t0 = torch.tensor(np.asarray(r_t0), dtype=torch.long, device=device)
    cost_pd_nd = torch.tensor(np.asarray(r_pd_nd), dtype=torch.long, device=device)

    power = torch.sum(cost_weight[:, None] * data ** 2)
    if float(power) == 0.0:
        power = torch.tensor(1.0, dtype=torch.float64, device=device)

    info = {
        **stim.info,
        "n_cost": int(data.shape[0]),
        "n_cost_pd": int((cost_pd_nd == PD_IDX).sum().item()),
        "n_cost_nd": int((cost_pd_nd == ND_IDX).sum().item()),
        "n_batch": stim.info["n_batch"],
        "n_cost_columns": len(cols),
        "center_column": bool(center_column),
        "cost_column_uv": (int(center_col.u), int(center_col.v)) if center_col else None,
        "side": side,
        "present_subtypes": present,
        "skipped_orthogonal": skipped_orthogonal,
        "fig1_path": str(fig1_path),
        "cost_window_steps": COST_WINDOW_STEPS,
    }
    return MovingBarTarget(
        signal=stim.signal,
        data=data,
        power=power,
        cost_weight=cost_weight,
        cost_t0=cost_t0,
        readout_batch=readout_batch,
        readout_unit=readout_unit,
        cost_pd_nd=cost_pd_nd,
        n_batch=stim.info["n_batch"],
        maxtime=maxtime,
        info=info,
    )


def build_borst_moving_bar_target(
    device: Optional[str] = None,
    t_on: int = T_ON,
    deltat_ms: float = 10.0,
    fig1_path: Path = FIG1_CI_NPZ,
    center_column: bool = False,
    readout_subtypes: Optional[Sequence[str]] = None,
    i_baseline: float = I_BASELINE,
    i_bright_bar: Optional[float] = None,
    i_dark_bar: Optional[float] = None,
    contrasts: Sequence[str] = ("bright", "dark"),
) -> MovingBarTarget:
    """Build Borst 5-column horizontal moving-bar target for T4/T5 subtypes."""
    device = device or "cpu"
    specs = _borst_moving_bar_specs(contrasts=tuple(contrasts))
    cols = _borst_hex_columns()
    field_deg = field_bounds(cols)
    maxtime = moving_bar_maxtime(specs, field_deg, t_on=t_on, deltat_ms=deltat_ms)
    sweep_end = moving_bar_sweep_end_step(specs, field_deg, t_on=t_on, deltat_ms=deltat_ms)
    fig1 = load_fig1_traces(fig1_path, deltat_ms=deltat_ms)
    subtype_pool = tuple(readout_subtypes) if readout_subtypes is not None else BORST_READOUT_SUBTYPES
    present = [st for st in subtype_pool if st in set(ml.ctype.tolist())]
    if not present:
        raise ValueError(
            f"Borst path has no moving-bar readout subtypes in ctype "
            f"(requested {list(subtype_pool)!r})",
        )

    contrast_set = frozenset(contrasts)
    col_kw = dict(
        i_baseline=i_baseline,
    )
    if "bright" in contrast_set:
        col_kw["i_bright_bar"] = I_BRIGHT if i_bright_bar is None else float(i_bright_bar)
    if "dark" in contrast_set:
        col_kw["i_dark_bar"] = I_DARK if i_dark_bar is None else float(i_dark_bar)
    column_current = build_batched_column_current(
        cols, specs, maxtime, t_on=t_on, deltat_ms=deltat_ms,
        **col_kw,
    )
    n_units = ml.n_state_units()
    signal = torch.zeros((len(specs), maxtime, n_units), dtype=torch.float64, device=device)
    for col in range(ml.nofcols):
        pr = ml.photoreceptor_slice(col)
        signal[:, :, pr] = torch.tensor(column_current[:, :, col], dtype=torch.float64, device=device)[:, :, None]

    cost_cols = [cols[ml.CENTER_COL]] if center_column else cols
    cost_col_ids = [ml.CENTER_COL] if center_column else list(range(ml.nofcols))

    r_batch, r_unit, r_target, r_weight, r_t0, r_pd_nd = [], [], [], [], [], []
    skipped_orthogonal = 0
    for b, spec in enumerate(specs):
        for col_id, col in zip(cost_col_ids, cost_cols):
            t_center = column_bar_center_step(
                col.x, col.y, spec, field_deg, t_on=t_on, deltat_ms=deltat_ms,
            )
            t0 = t_center - COST_HALF_WINDOW_STEPS
            if t0 < 0 or t_center + COST_HALF_WINDOW_STEPS > maxtime:
                raise ValueError(
                    f"cost window out of range for Borst column {col_id} "
                    f"spec={spec.name}: t_center={t_center}, maxtime={maxtime}"
                )
            for subtype in present:
                pref = motion_preference("right", subtype, spec.direction, spec.contrast)
                if pref is None:
                    skipped_orthogonal += 1
                    continue
                trace_id = fig1_key_for_stimulus("right", subtype, spec)
                if trace_id not in fig1:
                    raise KeyError(f"fig1 trace missing: {trace_id}")
                uidx = ml.unit_index(col_id, ml.type_index(subtype))
                r_batch.append(b)
                r_unit.append(int(uidx))
                r_target.append(fig1[trace_id])
                r_weight.append(1.0)
                r_t0.append(t0)
                r_pd_nd.append(_pd_nd_index(pref.pd_nd))

    if not r_batch:
        raise ValueError("no Borst moving-bar cost cells")

    data = torch.tensor(np.asarray(r_target), dtype=torch.float64, device=device)
    cost_weight = torch.tensor(np.asarray(r_weight), dtype=torch.float64, device=device)
    readout_batch = torch.tensor(np.asarray(r_batch), dtype=torch.long, device=device)
    readout_unit = torch.tensor(np.asarray(r_unit), dtype=torch.long, device=device)
    cost_t0 = torch.tensor(np.asarray(r_t0), dtype=torch.long, device=device)
    cost_pd_nd = torch.tensor(np.asarray(r_pd_nd), dtype=torch.long, device=device)
    power = torch.sum(cost_weight[:, None] * data ** 2)
    if float(power) == 0.0:
        power = torch.tensor(1.0, dtype=torch.float64, device=device)

    info = {
        "n_cost": int(data.shape[0]),
        "n_cost_pd": int((cost_pd_nd == PD_IDX).sum().item()),
        "n_cost_nd": int((cost_pd_nd == ND_IDX).sum().item()),
        "n_batch": len(specs),
        "n_cost_columns": len(cost_cols),
        "center_column": bool(center_column),
        "cost_column_uv": (ml.CENTER_COL, 0) if center_column else None,
        "side": "right",
        "present_subtypes": present,
        "skipped_orthogonal": skipped_orthogonal,
        "fig1_path": str(fig1_path),
        "cost_window_steps": COST_WINDOW_STEPS,
        "maxtime": maxtime,
        "t_on": t_on,
        "field_deg": field_deg,
        "sweep_steps": sweep_end - t_on,
        "sweep_time_s": (sweep_end - t_on) * (deltat_ms / 1000.0),
        "spec_names": [s.name for s in specs],
        "n_photo_columns": ml.nofcols,
        "mode": "borst",
        "i_baseline": float(i_baseline),
    }
    if "bright" in contrast_set:
        info["i_bright_bar"] = col_kw["i_bright_bar"]
    if "dark" in contrast_set:
        info["i_dark_bar"] = col_kw["i_dark_bar"]
    return MovingBarTarget(
        signal=signal,
        data=data,
        power=power,
        cost_weight=cost_weight,
        cost_t0=cost_t0,
        readout_batch=readout_batch,
        readout_unit=readout_unit,
        cost_pd_nd=cost_pd_nd,
        n_batch=len(specs),
        maxtime=maxtime,
        info=info,
    )
