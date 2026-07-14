"""Conductance / Vm budget for cell responses (cost-extent average or hex).

CLI matches ``analysis.cell_trace`` shared flags; budget walks are dynamics-only.
Reuse loaders/forwards from ``analysis.cell_trace`` (do not fork).

Speed / agent contract
----------------------
Per ``--run``:

  * one ``plot_trained.load_best``
  * one spot forward per distinct ``spot_*`` target
  * one moving-bar forward per distinct ``moving_bar_*`` target
  * unit-level ``update_Vm`` budget walks once per (target, bar-spec or spot)

**Do not** re-invoke once per cell / spec.

Modes
-----
* Omit ``--x`` / ``--y``: cost-extent **average** (plot totals).
* Exactly one ``--x`` and one ``--y``: **hex** mode (moving_bar only; one cell).
* Multiple x/y: rejected.

Examples
--------
  cd temporal_filtering/SimulationCode
  ../.venv/bin/python -m analysis.cell_dynamics \\
    --run /abs/path/to/run --cell Mi4,Mi9 \\
    --target spot_bright,moving_bar_bright --spec right_bright_w1

  ../.venv/bin/python -m analysis.cell_dynamics \\
    --run /abs/path/to/run --cell L3 --target moving_bar_bright \\
    --spec right_bright_w1 --x -2 --y -1
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

import FiveCol_MedSim_Pytorch as fc
import plot_trained
from analysis.cell_trace import (
    add_shared_cli,
    extract_moving_bar_bundle,
    extract_spot_bundle,
    parse_shared_cli,
    specs_for_cell,
)
from train import parse_comma_list

@dataclass
class StepBudget:
    rel: int
    ti: int
    slice_mV: float
    vm_d_mV: float
    dVm_mV: float
    signal: float
    g_exc_nS: float
    g_inh_nS: float
    g_Ih_on_nS: float
    g_Ih_off_nS: float
    num_exc: float
    num_inh: float
    num_leak: float
    num_ihoff: float
    num_cdt: float
    num_sig: float
    den: float
    n_units: int = 1


def _equilibrate(session, p, signal_batch: torch.Tensor, t_on: int):
    import FiveCol_MedSim_Pytorch as fc

    backend = session.backend
    B, T, N = signal_batch.shape
    dev = backend.conn.node_type.device
    dtype = session.sim_dtype
    u_on = torch.zeros((B, N), dtype=dtype, device=dev)
    u_off = torch.zeros((B, N), dtype=dtype, device=dev)
    Vm = backend.e_leak.expand(B, N).clone()
    for t in range(1, min(t_on, T)):
        Vm, u_on, u_off = fc.update_Vm(
            Vm, u_on, u_off,
            p["inp_gain"], p["out_gain"], p["Ih_gmax"], p["Ih_gmax_off"],
            p["Ih_midv"], p["Ih_slope"], p["tau_midv"],
            p["Ih_midv_off"], p["Ih_slope_off"], p["tau_midv_off"],
            signal_batch[:, t - 1], backend,
        )
    return Vm, u_on, u_off


def _budget_units(
    Vm_pre: torch.Tensor,
    u_on: torch.Tensor,
    u_off: torch.Tensor,
    sig_t: torch.Tensor,
    units: np.ndarray,
    p,
    conn,
    backend,
    *,
    vm_ref: np.ndarray,
) -> dict[str, np.ndarray]:
    """Per-unit conductance budget at one pre-step (batch dim = 1)."""
    import FiveCol_MedSim_Pytorch as fc

    units = np.asarray(units, dtype=np.int64)
    with torch.no_grad():
        g_Ih_on = (u_on[0] * p["Ih_gmax"] * fc.Ih_gain).cpu().numpy()
        g_Ih_off = (u_off[0] * p["Ih_gmax_off"] * fc.Ih_gain).cpu().numpy()
        rect = fc.rectsyn(Vm_pre[0], fc.trld)
        g_exc_all, g_inh_all = conn.exc_inh_drive(rect * p["out_gain"])
        g_exc = (g_exc_all * p["inp_gain"]).cpu().numpy()
        g_inh = (g_inh_all * p["inp_gain"]).cpu().numpy()
        Vm_u = Vm_pre[0].cpu().numpy()
        sig_v = sig_t[0].cpu().numpy()
        e_leak = backend.e_leak.cpu().numpy()

    u = units
    out = {
        "Vm_u": Vm_u[u],
        "vm_d": Vm_u[u] - vm_ref[u],
        "g_exc": g_exc[u],
        "g_inh": g_inh[u],
        "g_Ih_on": g_Ih_on[u],
        "g_Ih_off": g_Ih_off[u],
        "signal": sig_v[u],
        "num_exc": g_exc[u] * fc.E_exc,
        "num_inh": g_inh[u] * fc.E_inh,
        "num_leak": fc.g_leak * e_leak[u],
        "num_ihoff": g_Ih_off[u] * fc.E_IH_OFF,
        "num_cdt": fc.cdt * Vm_u[u],
        "num_sig": sig_v[u],
        "den": g_exc[u] + g_inh[u] + g_Ih_on[u] + g_Ih_off[u] + fc.g_leak + fc.cdt,
    }
    return out


def _mean_budget_row(
    *,
    rel: int,
    ti: int,
    overlay_val: float,
    bud: dict[str, np.ndarray],
    Vm_post: np.ndarray,
) -> StepBudget:
    n = int(bud["vm_d"].size)
    if n == 0:
        raise ValueError("empty unit set for mean budget")
    vm_d = float(np.mean(bud["vm_d"]))
    dVm = float(np.mean(Vm_post - bud["Vm_u"]))
    return StepBudget(
        rel=rel,
        ti=ti,
        slice_mV=float(overlay_val),
        vm_d_mV=vm_d,
        dVm_mV=dVm,
        signal=float(np.mean(bud["signal"])),
        g_exc_nS=float(np.mean(bud["g_exc"])),
        g_inh_nS=float(np.mean(bud["g_inh"])),
        g_Ih_on_nS=float(np.mean(bud["g_Ih_on"])),
        g_Ih_off_nS=float(np.mean(bud["g_Ih_off"])),
        num_exc=float(np.mean(bud["num_exc"])),
        num_inh=float(np.mean(bud["num_inh"])),
        num_leak=float(np.mean(bud["num_leak"])),
        num_ihoff=float(np.mean(bud["num_ihoff"])),
        num_cdt=float(np.mean(bud["num_cdt"])),
        num_sig=float(np.mean(bud["num_sig"])),
        den=float(np.mean(bud["den"])),
        n_units=n,
    )


def _first_nonzero_rel(trace: np.ndarray, *, eps: float = 1e-6) -> int | None:
    idx = np.where(np.abs(trace) > eps)[0]
    return int(idx[0]) if idx.size else None


def _auto_rel_window(trace: np.ndarray, *, before: int = 2, after: int = 8) -> tuple[int, int]:
    onset = _first_nonzero_rel(trace)
    if onset is None:
        return 0, min(10, trace.size - 1)
    return max(0, onset - before), min(trace.size - 1, onset + after)


def _post_peak_rel(
    overlay: np.ndarray,
    before_steps: int | None,
    *,
    horizon: int | None = 40,
) -> int:
    """Index of largest |overlay| after onset (optionally capped by ``horizon``)."""
    arr = np.asarray(overlay, dtype=float)
    if before_steps is not None and 0 < before_steps < arr.size:
        stop = arr.size
        if horizon is not None:
            stop = min(stop, before_steps + int(horizon))
        post = arr[before_steps:stop]
        if post.size == 0:
            return int(before_steps)
        return int(before_steps + int(np.argmax(np.abs(post))))
    stop = arr.size if horizon is None else min(arr.size, int(horizon))
    return int(np.argmax(np.abs(arr[:stop])))

def _polarity(v: float, *, eps: float = 1e-3) -> str:
    if v > eps:
        return "+"
    if v < -eps:
        return "-"
    return "0"


def _dominant_drive(step: StepBudget) -> str:
    """Which synaptic numerator push is larger in magnitude at this step."""
    if abs(step.num_exc) >= abs(step.num_inh):
        return "exc" if abs(step.num_exc) > 1e-9 else "none"
    return "inh"


def _unit_params(p, backend, unit: int) -> dict[str, float]:
    return {
        "inp_gain": float(p["inp_gain"][unit]),
        "out_gain": float(p["out_gain"][unit]),
        "Ih_gmax": float(p["Ih_gmax"][unit]),
        "Ih_gmax_off": float(p["Ih_gmax_off"][unit]),
        "e_leak_mV": float(backend.e_leak[unit]),
    }


def _globals():
    import FiveCol_MedSim_Pytorch as fc

    return {
        "E_exc": fc.E_exc,
        "E_inh": fc.E_inh,
        "E_IH_OFF": fc.E_IH_OFF,
        "g_leak_nS": fc.g_leak,
        "cdt": fc.cdt,
        "deltat_ms": fc.deltat,
        "trld_mV": fc.trld,
        "t_on": fc.t_on,
    }


# ---------------------------------------------------------------------------
# Average bar budgets (cost-extent)
# ---------------------------------------------------------------------------


def _bar_cost_units(session, cell: str, *, bi: int, t0_bn: np.ndarray, cost_extent) -> np.ndarray:
    from network.moving_bar_target import moving_bar_cost_columns
    from plot.moving_bar import _network_uv_np

    C = session.backend.network
    if cell not in C.type_names:
        raise SystemExit(f"unknown cell type {cell!r}; known: {list(C.type_names)}")
    ti = C.type_names.index(cell)
    u_np, v_np = _network_uv_np(C)
    type_ids = np.asarray(C.node_type.cpu().numpy(), dtype=np.int64)
    cols = moving_bar_cost_columns(C, cost_extent=cost_extent)
    col_uv = {(int(c.u), int(c.v)) for c in cols}
    mask = (type_ids == ti) & np.array(
        [(int(u), int(v)) in col_uv for u, v in zip(u_np, v_np)],
        dtype=bool,
    )
    units = np.where(mask)[0]
    valid = t0_bn[bi, units] >= 0
    units = units[valid]
    if units.size == 0:
        raise SystemExit(f"no valid {cell} units in cost_extent for bar aggregation")
    return units.astype(np.int64)


def analyze_bar_average(
    session,
    *,
    p,
    bundle,
    cells: list[str],
    target: str,
    spec: str,
    trace_kind: str,
    rel_start: int | None,
    rel_stop: int | None,
) -> dict[str, dict[str, Any]]:
    """One bar forward + one Vm walk; mean budget per cell over cost-extent units."""
    import FiveCol_MedSim_Pytorch as fc
    from plot import moving_bar as mb_plot

    if target not in fc.MOVING_BAR_TARGETS:
        raise SystemExit(f"unsupported target {target!r}")
    pack = session.pack_for(target)
    if bundle.traces.t0_bn is None:
        raise SystemExit("moving_bar bundle missing t0_bn")
    specs = mb_plot._bar_specs_for_session(session, target)
    try:
        bi = next(i for i, s in enumerate(specs) if s.name == spec)
    except StopIteration:
        raise SystemExit(f"spec {spec!r} not in {[s.name for s in specs]}")
    conn = session.backend.conn
    signal = pack.signal[bi:bi + 1]
    B, T, N = signal.shape
    t_on = fc.t_on
    t0_bn = np.asarray(bundle.traces.t0_bn)
    before_steps = int(bundle.traces.before_steps[spec])
    trace_len = T - t_on

    overlays: dict[str, np.ndarray] = {}
    unit_sets: dict[str, np.ndarray] = {}
    for cell in cells:
        key = (cell, spec)
        if key not in bundle.traces.model_mean:
            avail = sorted(s for c, s in bundle.traces.model_mean if c == cell)
            raise SystemExit(f"spec {spec!r} not found for {cell}; available: {avail}")
        overlays[cell] = np.asarray(bundle.traces.model_mean[key], dtype=float)
        unit_sets[cell] = _bar_cost_units(
            session, cell, bi=bi, t0_bn=t0_bn, cost_extent=pack.cost_extent,
        )

    # shared rel window from first overlay (same alignment length for all)
    ref_overlay = overlays[cells[0]]
    if rel_start is not None and rel_stop is not None:
        rel_lo, rel_hi = rel_start, rel_stop
    else:
        # post-onset window around |peak| (shape question sits after bar arrival)
        peak = _post_peak_rel(ref_overlay, before_steps)
        rel_lo = max(0, peak - 4)
        rel_hi = min(ref_overlay.size - 1, peak + 8)

    all_units = np.unique(np.concatenate([unit_sets[c] for c in cells]))
    Vm, u_on, u_off = _equilibrate(session, p, signal, t_on)
    vm_ref = Vm[0].detach().cpu().numpy().copy()
    local_t0 = t0_bn[bi]  # (N,)

    # acc[cell][rel] -> list of per-unit budget vectors… store running sum
    sums: dict[str, dict[int, dict[str, float]]] = {c: {} for c in cells}
    counts: dict[str, dict[int, int]] = {c: {} for c in cells}
    overlay_at: dict[str, dict[int, float]] = {
        c: {rel: float(overlays[c][rel]) for rel in range(rel_lo, rel_hi + 1)}
        for c in cells
    }
    dVm_sums: dict[str, dict[int, float]] = {c: {} for c in cells}

    for ti in range(trace_len):
        t_global = t_on + ti
        sig_t = signal[:, t_global - 1]
        # Overlay index: windows use abs time ``t0_bn + rel`` (t0 = window start;
        # stimulus arrives at rel=before_steps).
        rel_u = t_global - local_t0[all_units]
        in_win = (rel_u >= rel_lo) & (rel_u <= rel_hi)
        if not np.any(in_win):
            Vm, u_on, u_off = fc.update_Vm(
                Vm, u_on, u_off,
                p["inp_gain"], p["out_gain"], p["Ih_gmax"], p["Ih_gmax_off"],
                p["Ih_midv"], p["Ih_slope"], p["tau_midv"],
                p["Ih_midv_off"], p["Ih_slope_off"], p["tau_midv_off"],
                sig_t, session.backend,
            )
            continue
        active = all_units[in_win]
        active_rel = rel_u[in_win].astype(np.int64)
        bud = _budget_units(
            Vm, u_on, u_off, sig_t, active, p, conn, session.backend, vm_ref=vm_ref,
        )
        Vm_pre_u = bud["Vm_u"].copy()
        Vm, u_on, u_off = fc.update_Vm(
            Vm, u_on, u_off,
            p["inp_gain"], p["out_gain"], p["Ih_gmax"], p["Ih_gmax_off"],
            p["Ih_midv"], p["Ih_slope"], p["tau_midv"],
            p["Ih_midv_off"], p["Ih_slope_off"], p["tau_midv_off"],
            sig_t, session.backend,
        )
        Vm_post = Vm[0, torch.as_tensor(active, device=Vm.device)].detach().cpu().numpy()
        dVm_u = Vm_post - Vm_pre_u

        for cell in cells:
            uset = set(unit_sets[cell].tolist())
            for i, u in enumerate(active.tolist()):
                if u not in uset:
                    continue
                rel = int(active_rel[i])
                acc = sums[cell].setdefault(
                    rel,
                    {k: 0.0 for k in (
                        "vm_d", "signal", "g_exc", "g_inh", "g_Ih_on", "g_Ih_off",
                        "num_exc", "num_inh", "num_leak", "num_ihoff", "num_cdt",
                        "num_sig", "den",
                    )},
                )
                acc["vm_d"] += float(bud["vm_d"][i])
                acc["signal"] += float(bud["signal"][i])
                acc["g_exc"] += float(bud["g_exc"][i])
                acc["g_inh"] += float(bud["g_inh"][i])
                acc["g_Ih_on"] += float(bud["g_Ih_on"][i])
                acc["g_Ih_off"] += float(bud["g_Ih_off"][i])
                acc["num_exc"] += float(bud["num_exc"][i])
                acc["num_inh"] += float(bud["num_inh"][i])
                acc["num_leak"] += float(bud["num_leak"][i])
                acc["num_ihoff"] += float(bud["num_ihoff"][i])
                acc["num_cdt"] += float(bud["num_cdt"][i])
                acc["num_sig"] += float(bud["num_sig"][i])
                acc["den"] += float(bud["den"][i])
                dVm_sums[cell][rel] = dVm_sums[cell].get(rel, 0.0) + float(dVm_u[i])
                counts[cell][rel] = counts[cell].get(rel, 0) + 1

    reports: dict[str, dict[str, Any]] = {}
    for cell in cells:
        steps: list[StepBudget] = []
        for rel in range(rel_lo, rel_hi + 1):
            n = counts[cell].get(rel, 0)
            if n == 0:
                continue
            acc = sums[cell][rel]
            steps.append(StepBudget(
                rel=rel,
                ti=rel,  # mean absolute ti not well-defined under alignment
                slice_mV=overlay_at[cell][rel],
                vm_d_mV=acc["vm_d"] / n,
                dVm_mV=dVm_sums[cell][rel] / n,
                signal=acc["signal"] / n,
                g_exc_nS=acc["g_exc"] / n,
                g_inh_nS=acc["g_inh"] / n,
                g_Ih_on_nS=acc["g_Ih_on"] / n,
                g_Ih_off_nS=acc["g_Ih_off"] / n,
                num_exc=acc["num_exc"] / n,
                num_inh=acc["num_inh"] / n,
                num_leak=acc["num_leak"] / n,
                num_ihoff=acc["num_ihoff"] / n,
                num_cdt=acc["num_cdt"] / n,
                num_sig=acc["num_sig"] / n,
                den=acc["den"] / n,
                n_units=n,
            ))
        overlay = overlays[cell]
        peak_rel = _post_peak_rel(overlay, before_steps)
        peak_step = next((s for s in steps if s.rel == peak_rel), steps[len(steps) // 2] if steps else None)
        reports[cell] = {
            "mode": "average",
            "cell": cell,
            "n_units": int(unit_sets[cell].size),
            "target": target,
            "spec": spec,
            "trace_kind": trace_kind,
            "before_steps": before_steps,
            "resp_peak_rel": peak_rel,
            "overlay_peak_mV": float(overlay[peak_rel]),
            "overlay_polarity": _polarity(float(overlay[peak_rel])),
            "rel_window": [rel_lo, rel_hi],
            "overlay_onset_rel": _first_nonzero_rel(overlay),
            "params": _unit_params(p, session.backend, int(unit_sets[cell][0])),
            "globals": _globals(),
            "steps": [asdict(s) for s in steps],
            "peak_step": asdict(peak_step) if peak_step is not None else None,
            "peak_drive": _dominant_drive(peak_step) if peak_step is not None else None,
            "overlay": overlay.tolist(),
        }
    return reports


# ---------------------------------------------------------------------------
# Average spot budgets (center-bin / stim-on-column)
# ---------------------------------------------------------------------------


def analyze_spot_average(
    session_one,
    *,
    p,
    bundle,
    ref_on,
    cells: list[str],
    target: str,
    trace_kind: str,
    abs_start: int | None,
    abs_stop: int | None,
) -> dict[str, dict[str, Any]]:
    """One spot forward + Vm walks over batches; mean center-bin budget per cell."""
    import FiveCol_MedSim_Pytorch as fc
    from network.spot_target import (
        spotting_from_opts,
        spot_stimulus_batches,
        resolve_spot_cost_radii,
        spot_cost_unit_radius_layout,
    )
    from plot import spot as spot_plot
    from plot.spot import _readout_duv_from_batches

    if target not in fc.SPOT_TARGETS:
        raise SystemExit(f"unsupported target {target!r}")
    pack = session_one.primary_pack
    conn = session_one.backend.conn
    C = session_one.backend.network
    t_on = fc.t_on

    opts = dict((session_one.train_opts or {}).get(f"{pack.name}_stimulus_opts") or {})
    spotting = spotting_from_opts(C, stimulus_opts=opts)
    batches = spot_stimulus_batches(spotting)
    cost_radii = resolve_spot_cost_radii(stimulus_opts=opts)
    batch_idx, unit_idx, _radius, type_idx, stim_u, stim_v = spot_cost_unit_radius_layout(
        C, batches, cost_radii, pack.cost_extent,
    )
    du, dv = _readout_duv_from_batches(C, batch_idx, unit_idx, stim_u=stim_u, stim_v=stim_v)
    batch_idx = np.asarray(batch_idx, dtype=np.int64)
    unit_idx = np.asarray(unit_idx, dtype=np.int64)
    type_idx = np.asarray(type_idx, dtype=np.int64)
    du = np.asarray(du, dtype=np.int64)
    dv = np.asarray(dv, dtype=np.int64)
    center_row = (du == 0) & (dv == 0)

    overlays: dict[str, np.ndarray] = {}
    type_i: dict[str, int] = {}
    for cell in cells:
        if cell not in C.type_names:
            raise SystemExit(f"unknown cell {cell!r}")
        type_i[cell] = C.type_names.index(cell)
        cell_on = next((c for c in bundle.cells_on if c["name"] == cell), None)
        if cell_on is None:
            raise SystemExit(f"cell {cell!r} missing from spot bundle")
        imp, _rf = spot_plot.scale_curve(cell_on["cube"], spot_plot.CENTER_BIN)
        overlays[cell] = np.asarray(imp, dtype=float)

    sig = pack.signal if pack.signal.dim() == 3 else pack.signal.unsqueeze(0)
    B, T, N = sig.shape
    if abs_start is not None and abs_stop is not None:
        t_lo, t_hi = abs_start, abs_stop
    else:
        # around |peak| of first cell after t_on
        peak = _post_peak_rel(overlays[cells[0]], t_on)
        t_lo = max(t_on, peak - 4)
        t_hi = min(T - 1, peak + 8)

    sums: dict[str, dict[int, dict[str, float]]] = {c: {} for c in cells}
    counts: dict[str, dict[int, int]] = {c: {} for c in cells}
    dVm_sums: dict[str, dict[int, float]] = {c: {} for c in cells}

    for b in range(B):
        row_mask = center_row & (batch_idx == b)
        if not np.any(row_mask):
            continue
        # units requested for any cell on this batch at center
        want: dict[str, np.ndarray] = {}
        for cell in cells:
            m = row_mask & (type_idx == type_i[cell])
            if np.any(m):
                want[cell] = np.unique(unit_idx[m])
        if not want:
            continue
        all_u = np.unique(np.concatenate(list(want.values())))
        signal_b = sig[b:b + 1]
        Vm, u_on, u_off = _equilibrate(session_one, p, signal_b, t_on)
        vm_ref = Vm[0].detach().cpu().numpy().copy()
        for t_global in range(max(1, t_lo), t_hi + 1):
            sig_t = signal_b[:, t_global - 1]
            bud = _budget_units(
                Vm, u_on, u_off, sig_t, all_u, p, conn, session_one.backend, vm_ref=vm_ref,
            )
            Vm_pre = bud["Vm_u"].copy()
            Vm, u_on, u_off = fc.update_Vm(
                Vm, u_on, u_off,
                p["inp_gain"], p["out_gain"], p["Ih_gmax"], p["Ih_gmax_off"],
                p["Ih_midv"], p["Ih_slope"], p["tau_midv"],
                p["Ih_midv_off"], p["Ih_slope_off"], p["tau_midv_off"],
                sig_t, session_one.backend,
            )
            Vm_post = Vm[0, torch.as_tensor(all_u, device=Vm.device)].detach().cpu().numpy()
            dVm_u = Vm_post - Vm_pre
            u_to_i = {int(u): i for i, u in enumerate(all_u.tolist())}
            for cell, us in want.items():
                for u in us.tolist():
                    i = u_to_i[int(u)]
                    acc = sums[cell].setdefault(
                        t_global,
                        {k: 0.0 for k in (
                            "vm_d", "signal", "g_exc", "g_inh", "g_Ih_on", "g_Ih_off",
                            "num_exc", "num_inh", "num_leak", "num_ihoff", "num_cdt",
                            "num_sig", "den",
                        )},
                    )
                    acc["vm_d"] += float(bud["vm_d"][i])
                    acc["signal"] += float(bud["signal"][i])
                    acc["g_exc"] += float(bud["g_exc"][i])
                    acc["g_inh"] += float(bud["g_inh"][i])
                    acc["g_Ih_on"] += float(bud["g_Ih_on"][i])
                    acc["g_Ih_off"] += float(bud["g_Ih_off"][i])
                    acc["num_exc"] += float(bud["num_exc"][i])
                    acc["num_inh"] += float(bud["num_inh"][i])
                    acc["num_leak"] += float(bud["num_leak"][i])
                    acc["num_ihoff"] += float(bud["num_ihoff"][i])
                    acc["num_cdt"] += float(bud["num_cdt"][i])
                    acc["num_sig"] += float(bud["num_sig"][i])
                    acc["den"] += float(bud["den"][i])
                    dVm_sums[cell][t_global] = dVm_sums[cell].get(t_global, 0.0) + float(dVm_u[i])
                    counts[cell][t_global] = counts[cell].get(t_global, 0) + 1

    reports: dict[str, dict[str, Any]] = {}
    for cell in cells:
        steps: list[StepBudget] = []
        overlay = overlays[cell]
        for t_global in range(t_lo, t_hi + 1):
            n = counts[cell].get(t_global, 0)
            if n == 0:
                continue
            acc = sums[cell][t_global]
            steps.append(StepBudget(
                rel=t_global,
                ti=t_global - t_on,
                slice_mV=float(overlay[t_global]) if t_global < overlay.size else float("nan"),
                vm_d_mV=acc["vm_d"] / n,
                dVm_mV=dVm_sums[cell][t_global] / n,
                signal=acc["signal"] / n,
                g_exc_nS=acc["g_exc"] / n,
                g_inh_nS=acc["g_inh"] / n,
                g_Ih_on_nS=acc["g_Ih_on"] / n,
                g_Ih_off_nS=acc["g_Ih_off"] / n,
                num_exc=acc["num_exc"] / n,
                num_inh=acc["num_inh"] / n,
                num_leak=acc["num_leak"] / n,
                num_ihoff=acc["num_ihoff"] / n,
                num_cdt=acc["num_cdt"] / n,
                num_sig=acc["num_sig"] / n,
                den=acc["den"] / n,
                n_units=n,
            ))
        peak_t = _post_peak_rel(overlay, t_on)
        peak_step = next((s for s in steps if s.rel == peak_t), steps[len(steps) // 2] if steps else None)
        n_center = int(np.sum(center_row & (type_idx == type_i[cell])))
        reports[cell] = {
            "mode": "average",
            "cell": cell,
            "n_units": n_center,
            "target": target,
            "spec": None,
            "trace_kind": trace_kind,
            "before_steps": t_on,
            "resp_peak_rel": peak_t,
            "overlay_peak_mV": float(overlay[peak_t]),
            "overlay_polarity": _polarity(float(overlay[peak_t])),
            "rel_window": [t_lo, t_hi],
            "overlay_onset_rel": _first_nonzero_rel(overlay[t_on:]) if overlay.size > t_on else None,
            "params": _unit_params(p, session_one.backend, int(unit_idx[center_row & (type_idx == type_i[cell])][0])),
            "globals": _globals(),
            "steps": [asdict(s) for s in steps],
            "peak_step": asdict(peak_step) if peak_step is not None else None,
            "peak_drive": _dominant_drive(peak_step) if peak_step is not None else None,
            "overlay": overlay.tolist(),
            "ref_peak_mV": float(
                spot_plot.scale_curve(ref_on[cell], spot_plot.CENTER_BIN)[0][peak_t]
            ) if cell in ref_on else None,
        }
    return reports


# ---------------------------------------------------------------------------
# Hex-mode bar (kept; uses shared cache)
# ---------------------------------------------------------------------------


def _units_at_hex(session, cell: str, *, at_x: float, at_y: float, cost_extent: int):
    from network.moving_bar_target import moving_bar_cost_columns
    from plot.moving_bar import _network_uv_np
    from plot.utils import filter_sti_columns

    C = session.backend.network
    if C is None:
        raise SystemExit("hex mode requires a network backend")
    cols = filter_sti_columns(
        moving_bar_cost_columns(C, cost_extent=cost_extent),
        at_x=at_x,
        at_y=at_y,
    )
    if not cols:
        raise SystemExit(f"no column at x={at_x!r} y={at_y!r} within cost_extent={cost_extent}")
    if len(cols) > 1:
        raise SystemExit(f"multiple columns at x={at_x!r} y={at_y!r}; pick a unique hex")
    col = cols[0]
    if cell not in C.type_names:
        raise SystemExit(f"unknown cell type {cell!r}")
    ti = C.type_names.index(cell)
    u_np, v_np = _network_uv_np(C)
    type_ids = np.asarray(C.node_type.cpu().numpy(), dtype=np.int64)
    mask = (u_np == int(col.u)) & (v_np == int(col.v)) & (type_ids == ti)
    units = np.where(mask)[0].tolist()
    if not units:
        raise SystemExit(f"no {cell} unit at hex ({at_x},{at_y})")
    return col, units


def _rel_to_ti(local_t0: int, t_on: int, rel: int) -> int:
    return local_t0 - t_on + rel


def analyze_bar_hex(
    session,
    *,
    p,
    bundle,
    cell: str,
    target: str,
    spec: str,
    at_x: float,
    at_y: float,
    unit: int | None,
    trace_kind: str,
    rel_start: int | None,
    rel_stop: int | None,
) -> dict[str, Any]:
    import FiveCol_MedSim_Pytorch as fc
    from plot import moving_bar as mb_plot
    from plot.utils import slice_xy_label

    pack = session.pack_for(target)
    slice_label = slice_xy_label(at_x, at_y)
    if bundle.slice_overlay is None or slice_label not in bundle.slice_overlay:
        raise SystemExit(f"no slice overlay for {slice_label!r}")
    wt = bundle.slice_overlay[slice_label]
    key = (cell, spec)
    if key not in wt.model_mean:
        avail = sorted(s for c, s in wt.model_mean if c == cell)
        raise SystemExit(f"spec {spec!r} not found for {cell}; available: {avail}")
    overlay = np.asarray(wt.model_mean[key], dtype=float)
    if rel_start is not None and rel_stop is not None:
        rel_lo, rel_hi = rel_start, rel_stop
    else:
        before = int(bundle.traces.before_steps[spec])
        peak = _post_peak_rel(overlay, before)
        rel_lo, rel_hi = max(0, peak - 4), min(overlay.size - 1, peak + 8)

    col, units = _units_at_hex(session, cell, at_x=at_x, at_y=at_y, cost_extent=pack.cost_extent)
    if unit is None:
        if len(units) > 1:
            raise SystemExit(f"multiple {cell} at ({at_x},{at_y}): {units}; pass --unit")
        unit = units[0]
    elif unit not in units:
        raise SystemExit(f"unit {unit} not in {units}")

    specs = mb_plot._bar_specs_for_session(session, target)
    bi = next(i for i, s in enumerate(specs) if s.name == spec)
    signal = pack.signal[bi:bi + 1]
    t_on = fc.t_on
    Vm, u_on, u_off = _equilibrate(session, p, signal, t_on)
    vm_ref = Vm[0].detach().cpu().numpy().copy()
    local_t0 = int(bundle.traces.t0_bn[bi, unit])
    before_steps = int(bundle.traces.before_steps[spec])
    trace_len = signal.shape[1] - t_on
    steps: list[StepBudget] = []
    for rel in range(rel_lo, rel_hi + 1):
        ti = _rel_to_ti(local_t0, t_on, rel)
        if ti < 0 or ti >= trace_len:
            continue
        t_global = t_on + ti
        sig_t = signal[:, t_global - 1]
        bud = _budget_units(
            Vm, u_on, u_off, sig_t, np.array([unit]), p, session.backend.conn,
            session.backend, vm_ref=vm_ref,
        )
        Vm_pre = float(bud["Vm_u"][0])
        Vm, u_on, u_off = fc.update_Vm(
            Vm, u_on, u_off,
            p["inp_gain"], p["out_gain"], p["Ih_gmax"], p["Ih_gmax_off"],
            p["Ih_midv"], p["Ih_slope"], p["tau_midv"],
            p["Ih_midv_off"], p["Ih_slope_off"], p["tau_midv_off"],
            sig_t, session.backend,
        )
        dVm = float(Vm[0, unit] - Vm_pre)
        steps.append(_mean_budget_row(
            rel=rel, ti=ti, overlay_val=float(overlay[rel]), bud=bud,
            Vm_post=np.array([Vm_pre + dVm]),
        ))
    peak_rel = _post_peak_rel(overlay, before_steps)
    peak_step = next((s for s in steps if s.rel == peak_rel), None)
    return {
        "mode": "hex",
        "cell": cell,
        "unit": int(unit),
        "hex": {"x": at_x, "y": at_y},
        "uv": {"u": int(col.u), "v": int(col.v)},
        "n_units": 1,
        "target": target,
        "spec": spec,
        "trace_kind": trace_kind,
        "before_steps": before_steps,
        "resp_peak_rel": peak_rel,
        "overlay_peak_mV": float(overlay[peak_rel]),
        "overlay_polarity": _polarity(float(overlay[peak_rel])),
        "rel_window": [rel_lo, rel_hi],
        "params": _unit_params(p, session.backend, unit),
        "globals": _globals(),
        "steps": [asdict(s) for s in steps],
        "peak_step": asdict(peak_step) if peak_step is not None else None,
        "peak_drive": _dominant_drive(peak_step) if peak_step is not None else None,
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def _print_report(report: dict[str, Any]) -> None:
    mode = report.get("mode", "?")
    hdr = f"cell={report['cell']} mode={mode} n_units={report.get('n_units', '?')}"
    if mode == "hex":
        hdr += (
            f" unit=#{report['unit']} hex=({report['hex']['x']},{report['hex']['y']}) "
            f"uv=({report['uv']['u']},{report['uv']['v']})"
        )
    print(hdr)
    print(
        f"target={report['target']} spec={report.get('spec')} "
        f"trace_kind={report['trace_kind']}"
    )
    print(
        f"overlay_peak={report['overlay_peak_mV']:+.4f} mV "
        f"polarity={report['overlay_polarity']}  "
        f"resp_peak_rel={report['resp_peak_rel']}  "
        f"before_steps={report.get('before_steps')}  "
        f"peak_drive={report.get('peak_drive')}"
    )
    print("trained:", "  ".join(f"{k}={v:.6g}" for k, v in report["params"].items()))

    print("\nrel  n  overlay   vm_d    dVm    sig   g_inh  g_Ih_off  g_exc  num_inh  num_exc")
    for s in report["steps"]:
        print(
            f"{s['rel']:4d} {s.get('n_units', 1):3d} {s['slice_mV']:+8.4f} "
            f"{s['vm_d_mV']:+8.4f} {s['dVm_mV']:+8.4f} "
            f"{s['signal']:5.1f} {s['g_inh_nS']:.4f} {s['g_Ih_off_nS']:.4f} "
            f"{s['g_exc_nS']:.4f} {s['num_inh']:+8.2f} {s['num_exc']:+8.2f}"
        )

    ps = report.get("peak_step")
    if ps is not None:
        num = (
            ps["num_exc"] + ps["num_inh"] + ps["num_leak"]
            + ps["num_ihoff"] + ps["num_cdt"] + ps["num_sig"]
        )
        print(f"\nNumerator at peak rel={ps['rel']} (num={num:.2f}):")
        for name, val in [
            ("inh", ps["num_inh"]), ("ihoff", ps["num_ihoff"]), ("leak", ps["num_leak"]),
            ("cdt", ps["num_cdt"]), ("sig", ps["num_sig"]), ("exc", ps["num_exc"]),
        ]:
            pct = 100.0 * val / num if num else 0.0
            print(f"  {name:5s} {val:+9.2f} ({pct:.0f}%)")


def _print_polarity_compare(
    spot_reports: dict[str, dict[str, Any]],
    bar_reports: dict[str, dict[str, Any]],
) -> None:
    cells = sorted(set(spot_reports) & set(bar_reports))
    if not cells:
        return
    print("\n======== SPOT vs BAR polarity (cost-extent averages) ========")
    print(
        f"{'cell':6s} {'spot_peak':>10s} {'spot_pol':>8s} {'spot_drv':>8s} "
        f"{'bar_peak':>10s} {'bar_pol':>8s} {'bar_drv':>8s}  note"
    )
    for cell in cells:
        s = spot_reports[cell]
        b = bar_reports[cell]
        flip = s["overlay_polarity"] != b["overlay_polarity"] and "0" not in (
            s["overlay_polarity"], b["overlay_polarity"],
        )
        note = "FLIP" if flip else "same"
        if flip and s.get("peak_drive") and b.get("peak_drive"):
            if s["peak_drive"] != b["peak_drive"]:
                note += f" (drive {s['peak_drive']}→{b['peak_drive']})"
            else:
                note += f" (same drive={s['peak_drive']}; see num terms)"
        print(
            f"{cell:6s} {s['overlay_peak_mV']:+10.4f} {s['overlay_polarity']:>8s} "
            f"{str(s.get('peak_drive')):>8s} "
            f"{b['overlay_peak_mV']:+10.4f} {b['overlay_polarity']:>8s} "
            f"{str(b.get('peak_drive')):>8s}  {note}"
        )
        # short diagnosis at peak
        sps, bps = s.get("peak_step"), b.get("peak_step")
        if sps and bps:
            print(
                f"       spot@peak: g_exc={sps['g_exc_nS']:.4f} g_inh={sps['g_inh_nS']:.4f} "
                f"num_exc={sps['num_exc']:+.1f} num_inh={sps['num_inh']:+.1f} "
                f"vm_d={sps['vm_d_mV']:+.3f}"
            )
            print(
                f"       bar @peak: g_exc={bps['g_exc_nS']:.4f} g_inh={bps['g_inh_nS']:.4f} "
                f"num_exc={bps['num_exc']:+.1f} num_inh={bps['num_inh']:+.1f} "
                f"vm_d={bps['vm_d_mV']:+.3f}"
            )




def main() -> None:
    if __package__ is None:
        raise SystemExit(
            "run as a module from SimulationCode/: "
            "../.venv/bin/python -m analysis.cell_dynamics ..."
        )

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_shared_cli(ap)
    ap.add_argument("--unit", type=int, default=None, help="hex-mode unit index")
    ap.add_argument("--rel", default=None, help="rel/abs START,STOP inclusive window")
    ap.add_argument("--json", action="store_true", help="print JSON to stdout")
    args = ap.parse_args()
    cli = parse_shared_cli(args)

    hex_mode = False
    if cli.x_list is not None and cli.y_list is not None:
        if len(cli.x_list) != 1 or len(cli.y_list) != 1:
            raise SystemExit(
                "hex mode needs exactly one --x and one --y; "
                "omit both for cost-extent averages"
            )
        hex_mode = True
        if any(t in fc.SPOT_TARGETS for t in cli.targets):
            raise SystemExit("hex mode is moving_bar-only; omit --x/--y for spot")
        if len(cli.cells) != 1:
            raise SystemExit("hex mode supports one --cell")
    elif cli.x_list is not None or cli.y_list is not None:
        raise SystemExit("pass both --x and --y for hex mode, or neither for averages")

    rel_start = rel_stop = None
    if args.rel is not None:
        parts = parse_comma_list(args.rel)
        if len(parts) != 2:
            raise SystemExit("--rel must be START,STOP")
        rel_start, rel_stop = int(parts[0]), int(parts[1])

    for run_i, run_arg in enumerate(args.run):
        run_dir = plot_trained.resolve_run_dir(run_arg)
        session, z, best_i, best_cost = plot_trained.load_best(run_dir)
        p = fc.assign_params(z, list(session.schema), session.backend)

        spot_cache: dict[str, tuple] = {}
        bar_cache: dict[str, object] = {}
        spot_by_cell: dict[str, dict[str, Any]] = {}
        bar_by_cell: dict[str, dict[str, Any]] = {}
        all_reports: list[dict[str, Any]] = []

        if not args.json:
            print(f"== RUN {run_i}: {run_dir} ==")
            print(
                f"best_i={best_i}  best_cost={best_cost:.6g}  "
                f"mode={'hex' if hex_mode else 'average'}"
            )

        for target in cli.targets:
            if target in fc.SPOT_TARGETS:
                if target not in spot_cache:
                    spot_cache[target] = extract_spot_bundle(
                        session, z, target=target, trace_kind=args.trace_kind,
                        x_list=None, y_list=None,
                    )
                session_one, bundle, ref_on = spot_cache[target]
                reports = analyze_spot_average(
                    session_one,
                    p=p,
                    bundle=bundle,
                    ref_on=ref_on,
                    cells=cli.cells,
                    target=target,
                    trace_kind=args.trace_kind,
                    abs_start=rel_start,
                    abs_stop=rel_stop,
                )
                for cell, rep in reports.items():
                    spot_by_cell[cell] = rep
                    all_reports.append(rep)
                    if not args.json:
                        print("")
                        _print_report(rep)
            else:
                hx = cli.x_list[0] if hex_mode else None
                hy = cli.y_list[0] if hex_mode else None
                bar_key = target
                if hex_mode:
                    bar_key = (target, float(hx), float(hy))
                if bar_key not in bar_cache:
                    bar_cache[bar_key] = extract_moving_bar_bundle(
                        session, z, target=target, trace_kind=args.trace_kind,
                        x_list=cli.x_list if hex_mode else None,
                        y_list=cli.y_list if hex_mode else None,
                    )
                bundle = bar_cache[bar_key]
                if hex_mode:
                    cell = cli.cells[0]
                    for spec in specs_for_cell(bundle, cell, cli.specs_req):
                        rep = analyze_bar_hex(
                            session,
                            p=p,
                            bundle=bundle,
                            cell=cell,
                            target=target,
                            spec=spec,
                            at_x=float(hx),
                            at_y=float(hy),
                            unit=args.unit,
                            trace_kind=args.trace_kind,
                            rel_start=rel_start,
                            rel_stop=rel_stop,
                        )
                        bar_by_cell[cell] = rep
                        all_reports.append(rep)
                        if not args.json:
                            print("")
                            _print_report(rep)
                else:
                    specs_ordered: list[str] = []
                    for cell in cli.cells:
                        for spec in specs_for_cell(bundle, cell, cli.specs_req):
                            if spec not in specs_ordered:
                                specs_ordered.append(spec)
                    for spec in specs_ordered:
                        cells_for_spec = [
                            c for c in cli.cells
                            if spec in specs_for_cell(bundle, c, None)
                        ]
                        reports = analyze_bar_average(
                            session,
                            p=p,
                            bundle=bundle,
                            cells=cells_for_spec,
                            target=target,
                            spec=spec,
                            trace_kind=args.trace_kind,
                            rel_start=rel_start,
                            rel_stop=rel_stop,
                        )
                        for c, rep in reports.items():
                            bar_by_cell[c] = rep
                            all_reports.append(rep)
                            if not args.json:
                                print("")
                                _print_report(rep)

        if not args.json and spot_by_cell and bar_by_cell:
            _print_polarity_compare(spot_by_cell, bar_by_cell)

        if args.json:
            print(json.dumps(
                {"run": run_dir, "best_i": best_i, "best_cost": best_cost,
                 "reports": all_reports},
                indent=2,
            ))


if __name__ == "__main__":
    main()
