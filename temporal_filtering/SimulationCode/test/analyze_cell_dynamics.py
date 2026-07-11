"""
Per-step conductance budget for one cell at a moving-bar slice column.

Reconstructs overlay traces (Vm - Vm_ref) and, for the selected stimulus spec,
steps ``update_Vm`` from an equilibrated pre-stimulus state to report:
  - trained synaptic / Ih parameters at the unit
  - overlay vs single-step dVm alignment (rel index -> trace ti)
  - g_exc, g_inh, g_Ih_on/off, signal, and numerator term shares
  - presynaptic g_inh breakdown by cell type
  - optional counterfactual: hold signal at i_baseline on ti=1
  - with ``--upstream``: per-presynaptic (cell type, column) exc/inh drive vs rel,
    each partner's bar ``rel`` in the target window, and peak-timing table
  - with ``--compare-x`` / ``--compare-y``: same upstream table for a second hex
    and a side-by-side peak-rel diff (all rel indices are target-local)

Examples:
  temporal_filtering/.venv/bin/python temporal_filtering/SimulationCode/test/analyze_cell_dynamics.py \\
    --run /abs/path/to/run --cell L3 --target moving_bar_bright --spec right_bright_w1 \\
    --x -2 --y -1

  temporal_filtering/.venv/bin/python temporal_filtering/SimulationCode/test/analyze_cell_dynamics.py \\
    --run /abs/path/to/run --cell Mi4 --target moving_bar_bright --spec right_bright_w1 \\
    --x 1 --y 0.5 --trace-kind model --upstream --compare-x 0 --compare-y -2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch


def _add_simulation_code_to_syspath() -> None:
    sim_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, sim_dir)


def _load_best(outdir: str):
    import plot_trained

    params_path, _ = plot_trained.find_training_params(outdir)
    params = np.load(params_path)
    model = plot_trained.resolve_model(outdir)
    session = plot_trained.load_session(outdir, model=model)
    best_i = None
    best_i_path = os.path.join(outdir, "data", "best_i.txt")
    if os.path.isfile(best_i_path):
        s = open(best_i_path).read().strip()
        if s:
            best_i = int(s)
    best, best_cost, best_i = plot_trained.select_best(params, session, best_i=best_i)
    z = torch.tensor(best, dtype=session.sim_dtype, device=session.device)
    return session, z, int(best_i), float(best_cost)


def _parse_rel_window(text: str, *, n: int) -> tuple[int, int]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 2:
        raise SystemExit("--rel must be START,STOP (comma-separated, inclusive)")
    start, stop = int(parts[0]), int(parts[1])
    if start < 0 or stop >= n or start > stop:
        raise SystemExit(f"--rel window [{start},{stop}] out of range for trace length {n}")
    return start, stop


def _rel_to_ti(local_t0: int, t_on: int, rel: int) -> int:
    return local_t0 - t_on + rel


def _units_at_hex(session, cell: str, *, at_x: float, at_y: float, cost_extent: int) -> tuple[Any, list[int]]:
    from network.moving_bar_target import moving_bar_cost_columns
    from plot.moving_bar import _network_uv_np
    from plot.utils import filter_sti_columns

    C = session.backend.network
    if C is None:
        raise SystemExit("analyze_cell_dynamics requires a network backend")
    cols = filter_sti_columns(
        moving_bar_cost_columns(C, cost_extent=cost_extent),
        at_x=at_x,
        at_y=at_y,
    )
    if not cols:
        raise SystemExit(f"no column at x={at_x!r} y={at_y!r} within cost_extent={cost_extent}")
    if len(cols) > 1:
        raise SystemExit(f"multiple columns at x={at_x!r} y={at_y!r}; pick a unique hex coordinate")
    col = cols[0]
    if cell not in C.type_names:
        raise SystemExit(f"unknown cell type {cell!r}; known: {list(C.type_names)}")
    ti = C.type_names.index(cell)
    u_np, v_np = _network_uv_np(C)
    type_ids = np.asarray(C.node_type.cpu().numpy(), dtype=np.int64)
    mask = (u_np == int(col.u)) & (v_np == int(col.v)) & (type_ids == ti)
    units = np.where(mask)[0].tolist()
    if not units:
        raise SystemExit(f"no {cell} unit at hex ({at_x},{at_y}) uv=({col.u},{col.v})")
    return col, units


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


def _budget_pre_step(
    Vm_pre: torch.Tensor,
    u_on: torch.Tensor,
    u_off: torch.Tensor,
    sig_t: torch.Tensor,
    unit: int,
    p,
    conn,
    backend,
    *,
    vm_ref_u: float,
) -> dict[str, float]:
    import FiveCol_MedSim_Pytorch as fc

    u = unit
    e_leak_u = float(backend.e_leak[u])
    with torch.no_grad():
        g_Ih_on = float(u_on[0, u] * p["Ih_gmax"][u] * fc.Ih_gain)
        g_Ih_off = float(u_off[0, u] * p["Ih_gmax_off"][u] * fc.Ih_gain)
        rect = fc.rectsyn(Vm_pre[0], fc.trld)
        g_exc_all, g_inh_all = conn.exc_inh_drive(rect * p["out_gain"])
        g_exc = float(g_exc_all[u] * p["inp_gain"][u])
        g_inh = float(g_inh_all[u] * p["inp_gain"][u])
        Vm_u = float(Vm_pre[0, u])
        sig_v = float(sig_t[0, u])
        num_exc = g_exc * fc.E_exc
        num_inh = g_inh * fc.E_inh
        num_leak = fc.g_leak * e_leak_u
        num_ihoff = g_Ih_off * fc.E_IH_OFF
        num_cdt = fc.cdt * Vm_u
        num_sig = sig_v
        den = g_exc + g_inh + g_Ih_on + g_Ih_off + fc.g_leak + fc.cdt
    return {
        "Vm_u": Vm_u,
        "vm_d": Vm_u - vm_ref_u,
        "g_exc": g_exc,
        "g_inh": g_inh,
        "g_Ih_on": g_Ih_on,
        "g_Ih_off": g_Ih_off,
        "signal": sig_v,
        "num_exc": num_exc,
        "num_inh": num_inh,
        "num_leak": num_leak,
        "num_ihoff": num_ihoff,
        "num_cdt": num_cdt,
        "num_sig": num_sig,
        "den": den,
    }


def _presynaptic_g_inh(Vm: torch.Tensor, unit: int, conn, p, C) -> dict[str, float]:
    import FiveCol_MedSim_Pytorch as fc

    u = unit
    with torch.no_grad():
        drive = fc.rectsyn(
            Vm[0].index_select(0, conn.src_idx), fc.trld,
        ) * p["out_gain"].index_select(0, conn.src_idx)
        inh_edges = conn.w_inh * drive
        mask = conn.tar_idx == u
        by_type: dict[str, float] = defaultdict(float)
        src = conn.src_idx[mask].cpu().numpy()
        weights = (inh_edges[mask] * p["inp_gain"][u]).cpu().numpy()
        type_ids = C.node_type.cpu().numpy()
        for s, w in zip(src, weights):
            by_type[C.type_names[int(type_ids[s])]] += float(w)
    return dict(by_type)


def _unit_xy_table(C) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from column_mapper import uv_to_xy
    from plot.moving_bar import _network_uv_np

    u_np, v_np = _network_uv_np(C)
    xy = np.array([uv_to_xy(int(u), int(v)) for u, v in zip(u_np, v_np)], dtype=float)
    return u_np, v_np, xy


def _partner_label(type_name: str, x: float, y: float) -> str:
    from plot.utils import slice_xy_label

    return f"{type_name}@{slice_xy_label(x, y)}"


def _incoming_partner_edges(conn, unit: int, C, xy: np.ndarray) -> list[dict[str, Any]]:
    """Static incoming edges grouped by (source_type, column)."""
    mask = (conn.tar_idx == unit).cpu().numpy()
    src = conn.src_idx[mask].cpu().numpy()
    w_exc = conn.w_exc[mask].cpu().numpy()
    w_inh = conn.w_inh[mask].cpu().numpy()
    type_ids = C.node_type.cpu().numpy()
    type_names = list(C.type_names)
    groups: dict[tuple[str, float, float], dict[str, Any]] = {}
    for s, we, wi in zip(src, w_exc, w_inh):
        tn = type_names[int(type_ids[s])]
        x, y = float(xy[s, 0]), float(xy[s, 1])
        key = (tn, x, y)
        row = groups.setdefault(
            key,
            {
                "type": tn,
                "col": {"x": x, "y": y},
                "source_units": [],
                "n_syn_exc": 0.0,
                "n_syn_inh": 0.0,
            },
        )
        row["source_units"].append(int(s))
        row["n_syn_exc"] += float(we)
        row["n_syn_inh"] += float(wi)
    out = list(groups.values())
    out.sort(key=lambda r: -(r["n_syn_exc"] + r["n_syn_inh"]))
    return out


def _bar_rel_by_column(
    *,
    bi: int,
    local_t0: int,
    column_current: np.ndarray,
    sti_cols,
    i_baseline: float,
) -> dict[tuple[float, float], int]:
    from network.moving_bar_target import column_first_stim_step

    out: dict[tuple[float, float], int] = {}
    for j, col in enumerate(sti_cols):
        t_first = int(column_first_stim_step(column_current[bi, :, j], i_baseline=i_baseline))
        out[(float(col.x), float(col.y))] = int(t_first - local_t0)
    return out


def _partner_drive_row(
    Vm_pre: torch.Tensor,
    unit: int,
    partner: dict[str, Any],
    p,
    conn,
) -> dict[str, float]:
    import FiveCol_MedSim_Pytorch as fc

    inp = float(p["inp_gain"][unit])
    src_set = set(partner["source_units"])
    g_exc = 0.0
    g_inh = 0.0
    with torch.no_grad():
        mask = (conn.tar_idx == unit).cpu().numpy()
        src_all = conn.src_idx[mask].cpu().numpy()
        w_exc = conn.w_exc[mask].cpu().numpy()
        w_inh = conn.w_inh[mask].cpu().numpy()
        for s, we, wi in zip(src_all, w_exc, w_inh):
            si = int(s)
            if si not in src_set:
                continue
            drive = float(fc.rectsyn(Vm_pre[0, si], fc.trld) * p["out_gain"][si])
            g_exc += float(we) * drive
            g_inh += float(wi) * drive
    g_exc *= inp
    g_inh *= inp
    return {
        "g_exc_nS": g_exc,
        "g_inh_nS": g_inh,
        "num_exc": g_exc * fc.E_exc,
        "num_inh": g_inh * fc.E_inh,
    }


def _summarize_upstream_partner(
    label: str,
    partner: dict[str, Any],
    *,
    bar_rel: int | None,
    before_steps: int,
    resp_peak_rel: int,
    trace: list[dict[str, Any]],
) -> dict[str, Any]:
    if not trace:
        return {"label": label, "partner": partner, "bar_rel": bar_rel}
    rels = np.array([int(t["rel"]) for t in trace], dtype=int)
    num_exc = np.array([float(t["num_exc"]) for t in trace], dtype=float)
    num_inh = np.array([float(t["num_inh"]) for t in trace], dtype=float)
    g_exc = np.array([float(t["g_exc_nS"]) for t in trace], dtype=float)
    g_inh = np.array([float(t["g_inh_nS"]) for t in trace], dtype=float)

    def _pick(arr, rel):
        hit = np.where(rels == rel)[0]
        if hit.size == 0:
            return None
        return float(arr[hit[0]])

    peak_exc_rel = int(rels[np.argmax(num_exc)])
    peak_inh_rel = int(rels[np.argmax(num_inh)])
    return {
        "label": label,
        "partner": partner,
        "bar_rel": bar_rel,
        "peak_num_exc_rel": peak_exc_rel,
        "peak_num_exc": float(num_exc.max()),
        "peak_num_inh_rel": peak_inh_rel,
        "peak_num_inh": float(num_inh.max()),
        "at_target_bar_rel": {
            "rel": before_steps,
            "num_exc": _pick(num_exc, before_steps),
            "num_inh": _pick(num_inh, before_steps),
            "g_exc_nS": _pick(g_exc, before_steps),
            "g_inh_nS": _pick(g_inh, before_steps),
        },
        "at_resp_peak_rel": {
            "rel": resp_peak_rel,
            "num_exc": _pick(num_exc, resp_peak_rel),
            "num_inh": _pick(num_inh, resp_peak_rel),
            "g_exc_nS": _pick(g_exc, resp_peak_rel),
            "g_inh_nS": _pick(g_inh, resp_peak_rel),
        },
        "trace": trace,
    }


def _build_upstream_report(
    *,
    partners: list[dict[str, Any]],
    bar_rel_map: dict[tuple[float, float], int],
    before_steps: int,
    resp_peak_rel: int,
    rel_lo: int,
    rel_hi: int,
    partner_traces: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    summaries = []
    for partner in partners:
        col = partner["col"]
        key_xy = (float(col["x"]), float(col["y"]))
        label = _partner_label(partner["type"], col["x"], col["y"])
        summaries.append(
            _summarize_upstream_partner(
                label,
                partner,
                bar_rel=bar_rel_map.get(key_xy),
                before_steps=before_steps,
                resp_peak_rel=resp_peak_rel,
                trace=partner_traces.get(label, []),
            )
        )
    summaries.sort(
        key=lambda s: max(
            abs(s.get("peak_num_exc", 0.0) or 0.0),
            abs(s.get("peak_num_inh", 0.0) or 0.0),
        ),
        reverse=True,
    )
    return {
        "before_steps": before_steps,
        "resp_peak_rel": resp_peak_rel,
        "rel_window": [rel_lo, rel_hi],
        "partners": summaries,
    }


def _unit_params(p, backend, unit: int) -> dict[str, float]:
    return {
        "inp_gain": float(p["inp_gain"][unit]),
        "out_gain": float(p["out_gain"][unit]),
        "Ih_gmax": float(p["Ih_gmax"][unit]),
        "Ih_gmax_off": float(p["Ih_gmax_off"][unit]),
        "e_leak_mV": float(backend.e_leak[unit]),
    }


def analyze_moving_bar_dynamics(
    session,
    z,
    *,
    cell: str,
    target: str,
    spec: str,
    at_x: float,
    at_y: float,
    trace_kind: str,
    unit: int | None,
    rel_start: int | None,
    rel_stop: int | None,
    counterfactual: bool,
    upstream: bool = False,
) -> dict[str, Any]:
    import FiveCol_MedSim_Pytorch as fc
    import plot_trained
    from plot import moving_bar as mb_plot
    from plot.utils import slice_xy_label

    if not target.startswith("moving_bar_"):
        raise SystemExit(f"unsupported target {target!r}; expected moving_bar_*")
    pack = session.pack_for(target)
    one = plot_trained._session_for_target(session, target)
    bundle = mb_plot.moving_bar_trace_bundle(
        one, z, target,
        at_x_list=[at_x], at_y_list=[at_y],
        trace_kind=trace_kind,
    )
    if bundle.traces.t0_bn is None:
        raise SystemExit("moving_bar bundle missing t0_bn; update plot.moving_bar")
    slice_label = slice_xy_label(at_x, at_y)
    if bundle.slice_overlay is None or slice_label not in bundle.slice_overlay:
        raise SystemExit(f"no slice overlay for {slice_label!r}")
    wt = bundle.slice_overlay[slice_label]
    key = (cell, spec)
    if key not in wt.model_mean:
        avail = sorted(k for k in wt.model_mean if k[0] == cell)
        raise SystemExit(f"spec {spec!r} not found for {cell}; available: {avail}")
    overlay = np.asarray(wt.model_mean[key], dtype=float)
    if rel_start is not None and rel_stop is not None:
        rel_lo, rel_hi = rel_start, rel_stop
        if rel_lo < 0 or rel_hi >= overlay.size or rel_lo > rel_hi:
            raise SystemExit(f"--rel [{rel_lo},{rel_hi}] out of range for length {overlay.size}")
    elif upstream:
        rel_lo, rel_hi = 0, overlay.size - 1
    else:
        rel_lo, rel_hi = _auto_rel_window(overlay)

    col, units = _units_at_hex(session, cell, at_x=at_x, at_y=at_y, cost_extent=pack.cost_extent)
    if unit is None:
        if len(units) > 1:
            raise SystemExit(
                f"multiple {cell} units at ({at_x},{at_y}): {units}; pass --unit",
            )
        unit = units[0]
    elif unit not in units:
        raise SystemExit(f"unit {unit} is not {cell} at ({at_x},{at_y}); candidates: {units}")

    specs = mb_plot._bar_specs_for_session(session, target)
    try:
        bi = next(i for i, s in enumerate(specs) if s.name == spec)
    except StopIteration:
        raise SystemExit(f"spec {spec!r} not in session specs: {[s.name for s in specs]}")

    p = fc.assign_params(z, list(session.schema), session.backend)
    conn = session.backend.conn
    signal = pack.signal[bi:bi + 1]
    B, T, N = signal.shape
    t_on = fc.t_on
    Vm, u_on, u_off = _equilibrate(session, p, signal, t_on)
    vm_ref_u = float(Vm[0, unit])
    local_t0 = int(bundle.traces.t0_bn[bi, unit])
    before_steps = int(bundle.traces.before_steps[spec])
    trace_len = T - t_on
    overlay_peak_rel = int(np.argmax(overlay))

    C = session.backend.network
    partners: list[dict[str, Any]] = []
    bar_rel_map: dict[tuple[float, float], int] = {}
    partner_traces: dict[str, list[dict[str, Any]]] = {}
    if upstream:
        if C is None:
            raise SystemExit("--upstream requires a network backend")
        from network.moving_bar_target import build_moving_bar_signals, sti_columns

        _, _, xy = _unit_xy_table(C)
        partners = _incoming_partner_edges(conn, unit, C, xy)
        i_baseline = fc.session_moving_bar_i_baseline(session.train_opts)
        stim = build_moving_bar_signals(
            C,
            specs=specs,
            maxtime=int(session.maxtime),
            t_on=t_on,
            deltat_ms=fc.deltat,
            device=C.node_type.device,
            i_baseline=i_baseline,
        )
        sti_cols = sti_columns(C)
        bar_rel_map = _bar_rel_by_column(
            bi=bi,
            local_t0=local_t0,
            column_current=stim.column_current,
            sti_cols=sti_cols,
            i_baseline=i_baseline,
        )
        for partner in partners:
            label = _partner_label(partner["type"], partner["col"]["x"], partner["col"]["y"])
            partner["bar_rel"] = bar_rel_map.get(
                (float(partner["col"]["x"]), float(partner["col"]["y"])),
            )
            partner_traces[label] = []

    steps: list[StepBudget] = []
    for rel in range(rel_lo, rel_hi + 1):
        ti = _rel_to_ti(local_t0, t_on, rel)
        if ti < 0 or ti >= trace_len:
            continue
        t_global = t_on + ti
        sig_t = signal[:, t_global - 1]
        pre = _budget_pre_step(
            Vm, u_on, u_off, sig_t, unit, p, conn, session.backend, vm_ref_u=vm_ref_u,
        )
        if upstream:
            for partner in partners:
                label = _partner_label(partner["type"], partner["col"]["x"], partner["col"]["y"])
                drive = _partner_drive_row(Vm, unit, partner, p, conn)
                partner_traces[label].append({"rel": rel, **drive})
        Vm_pre = Vm.clone()
        Vm, u_on, u_off = fc.update_Vm(
            Vm, u_on, u_off,
            p["inp_gain"], p["out_gain"], p["Ih_gmax"], p["Ih_gmax_off"],
            p["Ih_midv"], p["Ih_slope"], p["tau_midv"],
            p["Ih_midv_off"], p["Ih_slope_off"], p["tau_midv_off"],
            sig_t, session.backend,
        )
        dVm = float(Vm[0, unit] - pre["Vm_u"])
        steps.append(StepBudget(
            rel=rel,
            ti=ti,
            slice_mV=float(overlay[rel]),
            vm_d_mV=pre["vm_d"],
            dVm_mV=dVm,
            signal=pre["signal"],
            g_exc_nS=pre["g_exc"],
            g_inh_nS=pre["g_inh"],
            g_Ih_on_nS=pre["g_Ih_on"],
            g_Ih_off_nS=pre["g_Ih_off"],
            num_exc=pre["num_exc"],
            num_inh=pre["num_inh"],
            num_leak=pre["num_leak"],
            num_ihoff=pre["num_ihoff"],
            num_cdt=pre["num_cdt"],
            num_sig=pre["num_sig"],
            den=pre["den"],
        ))

    presynaptic = _presynaptic_g_inh(Vm, unit, conn, p, session.backend.network)

    upstream_report = None
    if upstream:
        upstream_report = _build_upstream_report(
            partners=partners,
            bar_rel_map=bar_rel_map,
            before_steps=before_steps,
            resp_peak_rel=overlay_peak_rel,
            rel_lo=rel_lo,
            rel_hi=rel_hi,
            partner_traces=partner_traces,
        )

    cf = None
    if counterfactual and steps:
        import FiveCol_MedSim_Pytorch as fc

        cf_ti = steps[min(1, len(steps) - 1)].ti
        Vm_cf, uo_cf, uf_cf = _equilibrate(session, p, signal, t_on)
        for ti in range(cf_ti):
            Vm_cf, uo_cf, uf_cf = fc.update_Vm(
                Vm_cf, uo_cf, uf_cf,
                p["inp_gain"], p["out_gain"], p["Ih_gmax"], p["Ih_gmax_off"],
                p["Ih_midv"], p["Ih_slope"], p["tau_midv"],
                p["Ih_midv_off"], p["Ih_slope_off"], p["tau_midv_off"],
                signal[:, t_on + ti - 1], session.backend,
            )
        Vm_u = float(Vm_cf[0, unit])
        sig_actual = signal[:, t_on + cf_ti - 1]
        i_hold = float(signal[0, t_on + cf_ti - 2, unit].item()) if cf_ti > 0 else float(signal[0, t_on - 1, unit].item())
        Vm_act, _, _ = fc.update_Vm(
            Vm_cf.clone(), uo_cf.clone(), uf_cf.clone(),
            p["inp_gain"], p["out_gain"], p["Ih_gmax"], p["Ih_gmax_off"],
            p["Ih_midv"], p["Ih_slope"], p["tau_midv"],
            p["Ih_midv_off"], p["Ih_slope_off"], p["tau_midv_off"],
            sig_actual, session.backend,
        )
        Vm_hold, _, _ = fc.update_Vm(
            Vm_cf.clone(), uo_cf.clone(), uf_cf.clone(),
            p["inp_gain"], p["out_gain"], p["Ih_gmax"], p["Ih_gmax_off"],
            p["Ih_midv"], p["Ih_slope"], p["tau_midv"],
            p["Ih_midv_off"], p["Ih_slope_off"], p["tau_midv_off"],
            torch.full((1, N), i_hold, dtype=session.sim_dtype, device=Vm.device),
            session.backend,
        )
        cf = {
            "ti": cf_ti,
            "signal_actual": float(sig_actual[0, unit].item()),
            "signal_hold": i_hold,
            "dVm_actual": float(Vm_act[0, unit] - Vm_u),
            "dVm_counterfactual_hold": float(Vm_hold[0, unit] - Vm_u),
        }

    return {
        "cell": cell,
        "unit": int(unit),
        "hex": {"x": at_x, "y": at_y},
        "uv": {"u": int(col.u), "v": int(col.v)},
        "target": target,
        "spec": spec,
        "trace_kind": trace_kind,
        "Vm_ref_mV": vm_ref_u,
        "local_t0": local_t0,
        "before_steps": before_steps,
        "resp_peak_rel": overlay_peak_rel,
        "rel_window": [rel_lo, rel_hi],
        "overlay_onset_rel": _first_nonzero_rel(overlay),
        "params": _unit_params(p, session.backend, unit),
        "globals": {
            "E_exc": fc.E_exc,
            "E_inh": fc.E_inh,
            "E_IH_OFF": fc.E_IH_OFF,
            "g_leak_nS": fc.g_leak,
            "cdt": fc.cdt,
            "deltat_ms": fc.deltat,
            "trld_mV": fc.trld,
            "t_on": t_on,
        },
        "steps": [asdict(s) for s in steps],
        "presynaptic_g_inh_nS": presynaptic,
        "upstream": upstream_report,
        "counterfactual": cf,
    }


def _first_nonzero_rel(trace: np.ndarray, *, eps: float = 1e-6) -> int | None:
    idx = np.where(np.abs(trace) > eps)[0]
    return int(idx[0]) if idx.size else None


def _auto_rel_window(trace: np.ndarray, *, before: int = 2, after: int = 8) -> tuple[int, int]:
    onset = _first_nonzero_rel(trace)
    if onset is None:
        return 0, min(10, trace.size - 1)
    return max(0, onset - before), min(trace.size - 1, onset + after)


def _print_report(report: dict[str, Any]) -> None:
    g = report["globals"]
    p = report["params"]
    print(f"cell={report['cell']} unit=#{report['unit']} hex=({report['hex']['x']},{report['hex']['y']}) "
          f"uv=({report['uv']['u']},{report['uv']['v']})")
    print(f"target={report['target']} spec={report['spec']} trace_kind={report['trace_kind']}")
    print(f"Vm_ref={report['Vm_ref_mV']:.4f} mV  local_t0={report['local_t0']}  "
          f"overlay_onset_rel={report['overlay_onset_rel']}  rel_window={report['rel_window']}")
    print("trained:", "  ".join(f"{k}={v:.6g}" for k, v in p.items()))
    print(f"globals: E_inh={g['E_inh']} E_IH_OFF={g['E_IH_OFF']} cdt={g['cdt']} t_on={g['t_on']}")

    print("\nrel  ti  slice    vm_d    dVm    sig   g_inh  g_Ih_off  g_exc")
    if not report["steps"]:
        print("  (no steps in window; widen --rel or check overlay_onset_rel / local_t0 mapping)")
    for s in report["steps"]:
        print(
            f"{s['rel']:3d} {s['ti']:3d} {s['slice_mV']:+8.4f} {s['vm_d_mV']:+8.4f} {s['dVm_mV']:+8.4f} "
            f"{s['signal']:5.1f} {s['g_inh_nS']:.4f} {s['g_Ih_off_nS']:.4f} {s['g_exc_nS']:.4f}",
        )

    if report["steps"]:
        s = report["steps"][min(1, len(report["steps"]) - 1)]
        num = s["num_exc"] + s["num_inh"] + s["num_leak"] + s["num_ihoff"] + s["num_cdt"] + s["num_sig"]
        print(f"\nNumerator terms at rel={s['rel']} ti={s['ti']} (num={num:.2f}):")
        for name, val in [
            ("inh", s["num_inh"]), ("ihoff", s["num_ihoff"]), ("leak", s["num_leak"]),
            ("cdt", s["num_cdt"]), ("sig", s["num_sig"]), ("exc", s["num_exc"]),
        ]:
            pct = 100.0 * val / num if num else 0.0
            print(f"  {name:5s} {val:+9.2f} ({pct:.0f}%)")

    pre = report["presynaptic_g_inh_nS"]
    total = sum(pre.values())
    print(f"\nPresynaptic g_inh (nS) total={total:.4f}:")
    for tn, v in sorted(pre.items(), key=lambda kv: -kv[1]):
        pct = 100.0 * v / total if total else 0.0
        print(f"  {tn:6s} {v:.4f} ({pct:.1f}%)")

    cf = report.get("counterfactual")
    if cf is not None:
        print(f"\nCounterfactual ti={cf['ti']} (signal drop vs held previous step):")
        print(f"  signal actual={cf['signal_actual']:.1f}  hold={cf['signal_hold']:.1f}")
        print(f"  dVm actual={cf['dVm_actual']:+.4f} mV")
        print(f"  dVm hold={cf['dVm_counterfactual_hold']:+.4f} mV")

    if report.get("upstream") is not None:
        _print_upstream_report(report)


def _fmt_opt(val: float | None, *, prec: int = 3) -> str:
    if val is None:
        return "   n/a"
    return f"{val:+8.{prec}f}"


def _print_upstream_report(report: dict[str, Any]) -> None:
    up = report["upstream"]
    before = up["before_steps"]
    peak = up["resp_peak_rel"]
    print(
        f"\nUpstream drive (target-local rel; rel={before} = bar at target column; "
        f"resp_peak_rel={peak}):"
    )
    print(
        "partner           n_syn+  n_syn-  bar_rel  peak_exc_rel  peak_num_exc  "
        "peak_inh_rel  peak_num_inh  exc@peak  inh@peak"
    )
    for row in up["partners"]:
        p = row["partner"]
        at_peak = row.get("at_resp_peak_rel") or {}
        print(
            f"{row['label']:<17} "
            f"{p['n_syn_exc']:7.2f} {p['n_syn_inh']:7.2f} "
            f"{row.get('bar_rel', 'n/a')!s:>7} "
            f"{row.get('peak_num_exc_rel', 'n/a')!s:>12} "
            f"{_fmt_opt(row.get('peak_num_exc'))} "
            f"{row.get('peak_num_inh_rel', 'n/a')!s:>12} "
            f"{_fmt_opt(row.get('peak_num_inh'))} "
            f"{_fmt_opt(at_peak.get('num_exc'))} "
            f"{_fmt_opt(at_peak.get('num_inh'))}"
        )


def _print_upstream_compare(report_a: dict[str, Any], report_b: dict[str, Any]) -> None:
    up_a = report_a.get("upstream")
    up_b = report_b.get("upstream")
    if up_a is None or up_b is None:
        return
    label_a = f"({report_a['hex']['x']},{report_a['hex']['y']})"
    label_b = f"({report_b['hex']['x']},{report_b['hex']['y']})"
    print(
        f"\nUpstream peak-rel compare: {label_a} resp_peak={up_a['resp_peak_rel']} "
        f"vs {label_b} resp_peak={up_b['resp_peak_rel']} "
        f"(Δresp_peak={up_b['resp_peak_rel'] - up_a['resp_peak_rel']:+d} rel)"
    )
    by_a = {row["label"]: row for row in up_a["partners"]}
    by_b = {row["label"]: row for row in up_b["partners"]}
    labels = sorted(set(by_a) | set(by_b))
    print(
        "partner           "
        f"{'A_bar':>6} {'B_bar':>6} {'Δbar':>5}  "
        f"{'A_exc':>6} {'B_exc':>6} {'Δexc':>5}  "
        f"{'A_inh':>6} {'B_inh':>6} {'Δinh':>5}"
    )
    for label in labels:
        row_a = by_a.get(label)
        row_b = by_b.get(label)
        bar_a = row_a.get("bar_rel") if row_a else None
        bar_b = row_b.get("bar_rel") if row_b else None
        d_bar = (bar_b - bar_a) if bar_a is not None and bar_b is not None else None
        pe_a = row_a.get("peak_num_exc_rel") if row_a else None
        pe_b = row_b.get("peak_num_exc_rel") if row_b else None
        pi_a = row_a.get("peak_num_inh_rel") if row_a else None
        pi_b = row_b.get("peak_num_inh_rel") if row_b else None

        def _cell(val, width: int = 6) -> str:
            return f"{val!s:>{width}}" if val is not None else f"{'n/a':>{width}}"

        print(
            f"{label:<17} "
            f"{_cell(bar_a)} {_cell(bar_b)} {_cell(d_bar, 5)}  "
            f"{_cell(pe_a)} {_cell(pe_b)} {_cell((pe_b - pe_a) if pe_a is not None and pe_b is not None else None, 5)}  "
            f"{_cell(pi_a)} {_cell(pi_b)} {_cell((pi_b - pi_a) if pi_a is not None and pi_b is not None else None, 5)}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="absolute run directory")
    ap.add_argument("--cell", required=True, help="cell type, e.g. L3, L4")
    ap.add_argument("--target", required=True, help="moving_bar_bright or moving_bar_dark")
    ap.add_argument("--spec", required=True, help="stimulus spec name, e.g. right_bright_w1")
    ap.add_argument("--x", type=float, required=True, help="hex slice x")
    ap.add_argument("--y", type=float, required=True, help="hex slice y")
    ap.add_argument("--trace-kind", default="vm", choices=("vm", "model"))
    ap.add_argument("--unit", type=int, default=None, help="unit index (required if multiple at column)")
    ap.add_argument("--rel", default=None, help="rel START,STOP inclusive overlay window")
    ap.add_argument("--counterfactual", action="store_true", help="compare ti=1 signal drop vs held baseline")
    ap.add_argument(
        "--upstream",
        action="store_true",
        help="per-presynaptic (type,column) exc/inh drive vs target-local rel",
    )
    ap.add_argument("--compare-x", type=float, default=None, help="second hex x for upstream compare")
    ap.add_argument("--compare-y", type=float, default=None, help="second hex y for upstream compare")
    ap.add_argument("--json", action="store_true", help="print JSON report to stdout")
    args = ap.parse_args()

    _add_simulation_code_to_syspath()

    run_dir = os.path.abspath(args.run)
    if not os.path.isdir(run_dir):
        raise SystemExit(f"run dir not found: {run_dir}")

    if (args.compare_x is None) ^ (args.compare_y is None):
        raise SystemExit("--compare-x and --compare-y must be passed together")
    if args.compare_x is not None and not args.upstream:
        raise SystemExit("--compare-x/--compare-y require --upstream")

    rel_start = rel_stop = None
    if args.rel is not None:
        parts = [p.strip() for p in args.rel.split(",") if p.strip()]
        if len(parts) != 2:
            raise SystemExit("--rel must be START,STOP")
        rel_start, rel_stop = int(parts[0]), int(parts[1])

    session, z, best_i, best_cost = _load_best(run_dir)
    common = dict(
        cell=args.cell,
        target=args.target,
        spec=args.spec,
        trace_kind=args.trace_kind,
        unit=args.unit,
        rel_start=rel_start,
        rel_stop=rel_stop,
        counterfactual=args.counterfactual,
        upstream=args.upstream,
    )
    report = analyze_moving_bar_dynamics(
        session, z,
        at_x=args.x,
        at_y=args.y,
        **common,
    )
    report["run"] = run_dir
    report["best_i"] = best_i
    report["best_cost"] = best_cost

    report_b = None
    if args.compare_x is not None and args.compare_y is not None:
        report_b = analyze_moving_bar_dynamics(
            session, z,
            at_x=args.compare_x,
            at_y=args.compare_y,
            **common,
        )
        report_b["run"] = run_dir
        report_b["best_i"] = best_i
        report_b["best_cost"] = best_cost

    if args.json:
        def _json_ready(rep: dict[str, Any]) -> dict[str, Any]:
            out = dict(rep)
            up = out.get("upstream")
            if up is not None:
                slim = dict(up)
                slim["partners"] = []
                for row in up["partners"]:
                    slim_row = {k: v for k, v in row.items() if k != "trace"}
                    slim["partners"].append(slim_row)
                out["upstream"] = slim
            return out

        payload = {"a": _json_ready(report)}
        if report_b is not None:
            payload["b"] = _json_ready(report_b)
        print(json.dumps(payload, indent=2))
    else:
        print(f"== RUN: {run_dir} ==")
        print(f"best_i={best_i}  best_cost={best_cost:.6g}")
        _print_report(report)
        if report_b is not None:
            print(f"\n== COMPARE hex ({args.compare_x},{args.compare_y}) ==")
            _print_report(report_b)
            _print_upstream_compare(report, report_b)


if __name__ == "__main__":
    main()
