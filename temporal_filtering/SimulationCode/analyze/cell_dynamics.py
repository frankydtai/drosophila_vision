"""Borst / v budget for cell responses (cost-extent average or hex).

CLI matches ``analyze.cell_trace`` shared flags (no ``--trace-kind``; v only).

Speed / agent contract
----------------------
Per ``--run``:

  * one ``plot_trained.load_best``
  * one batched v budget walk per distinct target (spot / bar average / bar hex)
  * v_post and budget share that walk (no separate plot forward)

``--rel START,STOP`` limits budget accumulate to that inclusive rel (or abs) window
after equilibrate; cheap updates before, break after.

Modes
-----
* Omit ``--x`` / ``--y``: cost-extent **average** (plot totals).
* Exactly one ``--x`` and one ``--y``: **hex** mode (moving_bar only; one cell).
* Multiple x/y: rejected.

``--plot`` writes PNGs under ``{run}/cell_dynamics/``. With multiple moving_bar
specs, one overlay PNG per cell (specs differ by linestyle). One spec → one PNG.

``--syn-strength SRC,TAR,VALUE,...`` overrides trained ``syn_strength`` for
type pairs before budget walks (flat comma list; length multiple of 3).

Examples
--------
  cd temporal_filtering/SimulationCode
  ../.venv/bin/python -m analyze.cell_dynamics \\
    --run /abs/path/to/run --cell Mi4,Mi9 \\
    --target spot_bright,moving_bar_bright --spec right_bright_w1

  ../.venv/bin/python -m analyze.cell_dynamics \\
    --run /abs/path/to/run --cell L3 --target moving_bar_bright \\
    --spec right_bright_w1 --x -2 --y -1

  ../.venv/bin/python -m analyze.cell_dynamics \\
    --run /abs/path/to/run --cell Mi4 --target spot_bright --plot

  ../.venv/bin/python -m analyze.cell_dynamics \\
    --run borst/27252028-... --cell T4a --target moving_bar_bright \\
    --spec left_bright_w4,right_bright_w4 --plot

  ../.venv/bin/python -m analyze.cell_dynamics \\
    --run borst/27252028-... --cell T4a --target moving_bar_bright \\
    --spec left_bright_w4,right_bright_w4 --syn-strength Mi4,T4a,2.0,Mi9,T4a,1.0 \\
    --plot --rel 0,176
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

import import_bootstrap  # noqa: F401
import training as fc
import figure.plot_runs as plot_trained
from analyze.cell_trace import add_shared_cli, parse_shared_cli
from task.moving_bar.data import (
    bar_specs_for_session,
    col2subtype,
    filter_requested_specs,
    moving_bar_row_specs,
    moving_bar_session_t0_grids,
    moving_bar_units_on_columns,
)
from task.moving_bar.input import (
    filter_sti_columns,
    moving_bar_cost_columns,
)
from task.spot.data import (
    resolve_spot_cost_radii,
    spot_center_bin_layout,
)
from task.spot.input import (
    spot_from_opts,
    spot_stimulus_batches,
)
from figure.readout import spot_ref_cubes
from figure.spot import CENTER_BIN
from figure.util import plot_sem_band
from connectome_io import parse_comma_list


# Budget-step fields plotted vs time (key, ylabel/legend).
_PLOT_PANELS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "v_post (mV)",
        [
            ("v_post_mV", "v_post"),
        ],
    ),
    (
        "conductance (nS)",
        [
            ("cdt_nS", "cdt"),
            ("g_exc_nS", "g_exc"),
            ("g_inh_nS", "g_inh"),
            ("g_leak_nS", "g_leak"),
            ("g_Ih_on_nS", "g_h_on"),
            ("g_Ih_off_nS", "g_h_off"),
        ],
    ),
    (
        "i (pA)",
        [
            ("num_cdt", "i_cdt"),
            ("num_exc", "i_exc"),
            ("num_inh", "i_inh"),
            ("num_leak", "i_leak"),
            ("num_ihon", "i_h_on"),
            ("num_ihoff", "i_h_off"),
            ("num_sig", "i_sig"),
        ],
    ),
    (
        "num / den",
        [
            ("num_over_den", "num/den"),
            ("num", "num"),
            ("den", "den"),
        ],
    ),
]

_PLOT_NCOLS = max(len(series) for _, series in _PLOT_PANELS)
_BLACK_TRACE_LABELS = frozenset({"num", "den"})
# Current row only: one ylim across columns.
_ROW_SHARED_YLIM = frozenset({2})


def _trace_ylabel(group_ylabel: str, label: str) -> str:
    if label == "num/den":
        return "num/den (mV)"
    if label == "num":
        return "num (pA)"
    if label == "den":
        return "den (nS)"
    if "(" in group_ylabel:
        unit = " " + group_ylabel[group_ylabel.index("("):]
        return f"{label}{unit}"
    return label


def _budget_axes_grid():
    """Return ``(n_rows, n_cols)`` grid: one row per ``_PLOT_PANELS`` group."""
    return len(_PLOT_PANELS), _PLOT_NCOLS


def _plot_colors():
    import matplotlib.pyplot as plt

    return plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _plot_trace_colors(colors: list[str]) -> dict[str, str]:
    """Map trace legend label → subplot color (column index within its row)."""
    out: dict[str, str] = {}
    for _group_ylabel, series in _PLOT_PANELS:
        for ci, (_key, label) in enumerate(series):
            if label in _BLACK_TRACE_LABELS:
                out[label] = "0.0"
            else:
                out[label] = colors[ci % len(colors)]
    return out


_FORMULA_G_TOKENS: list[tuple[str, str | None]] = [
    ("v_post", "v_post"),
    (" = (", None),
    ("cdt", "cdt"), ("·v_pre + ", None),
    ("E_exc·", None), ("g_exc", "g_exc"), (" + ", None),
    ("E_inh·", None), ("g_inh", "g_inh"), (" + ", None),
    ("E_leak·", None), ("g_leak", "g_leak"), (" + ", None),
    ("E_h_on·", None), ("g_h_on", "g_h_on"), (" + ", None),
    ("E_h_off·", None), ("g_h_off", "g_h_off"), (" + ", None),
    ("i_sig", "i_sig"),
    (") / (", None),
    ("cdt", "cdt"), (" + ", None),
    ("g_exc", "g_exc"), (" + ", None),
    ("g_inh", "g_inh"), (" + ", None),
    ("g_h_on", "g_h_on"), (" + ", None),
    ("g_h_off", "g_h_off"), (" + ", None),
    ("g_leak", "g_leak"),
    (")", None),
]

_FORMULA_TOKENS: list[tuple[str, str | None]] = [
    ("v_post", "v_post"),
    (" = (", None),
    ("i_cdt", "i_cdt"), (" + ", None),
    ("i_exc", "i_exc"), (" + ", None),
    ("i_inh", "i_inh"), (" + ", None),
    ("i_leak", "i_leak"), (" + ", None),
    ("i_h_on", "i_h_on"), (" + ", None),
    ("i_h_off", "i_h_off"), (" + ", None),
    ("i_sig", "i_sig"),
    (") / (", None),
    ("cdt", "cdt"), (" + ", None),
    ("g_exc", "g_exc"), (" + ", None),
    ("g_inh", "g_inh"), (" + ", None),
    ("g_h_on", "g_h_on"), (" + ", None),
    ("g_h_off", "g_h_off"), (" + ", None),
    ("g_leak", "g_leak"),
    (")", None),
]


def _g_e_note(label: str, *, e_leak_mV: float) -> str | None:
    """Reversal annotation for a conductance subplot (``E_exc=+10 mV`` …)."""
    notes = {
        "g_exc": f"E_exc={fc.E_exc:+g} mV",
        "g_inh": f"E_inh={fc.E_inh:+g} mV",
        "g_leak": f"E_leak={e_leak_mV:+g} mV",
        "g_h_on": f"E_h_on={fc.E_Ih:+g} mV",
        "g_h_off": f"E_h_off={fc.E_IH_OFF:+g} mV",
    }
    return notes.get(label)


def _add_budget_formula_row(
    fig,
    colors: list[str],
    tokens: list[tuple[str, str | None]],
    *,
    y: float,
    fontsize: int = 9,
) -> None:
    """One formula line; trace symbols use plot colors, operators/constants gray."""
    tc = _plot_trace_colors(colors)
    x = 0.02
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for text, label in tokens:
        color = tc[label] if label else "0.2"
        txt = fig.text(
            x, y, text, transform=fig.transFigure,
            ha="left", va="top", fontsize=fontsize, color=color,
        )
        bbox = txt.get_window_extent(renderer=renderer)
        x = inv.transform((bbox.x1, bbox.y0))[0] + 0.003


def _finish_budget_figure_layout(fig, title: str, colors: list[str]) -> None:
    fig.suptitle(title, fontsize=11, y=0.995)
    _add_budget_formula_row(fig, colors, _FORMULA_G_TOKENS, y=0.975, fontsize=8)
    _add_budget_formula_row(fig, colors, _FORMULA_TOKENS, y=0.955, fontsize=9)
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.88))

def _v_step_params(p):
    """Unpack ``assign_params`` fields for ``fc.update_v``."""
    return (
        p["in_gain"], p["out_gain"], p["syn_strength"], p["v_th"],
        p["Ih_gmax"], p["Ih_gmax_off"],
        p["Ih_midv"], p["Ih_slope"], p["tau_midv"],
        p["Ih_midv_off"], p["Ih_slope_off"], p["tau_midv_off"],
    )


def _equilibrate(session, p, signal_batch: torch.Tensor, t_on: int):
    backend = session.backend
    B, T, N = signal_batch.shape
    dev = backend.conn.node_type.device
    dtype = session.sim_dtype
    u_on = torch.zeros((B, N), dtype=dtype, device=dev)
    u_off = torch.zeros((B, N), dtype=dtype, device=dev)
    v = backend.e_leak.expand(B, N).clone()
    step_p = _v_step_params(p)
    for t in range(1, min(t_on, T)):
        v, u_on, u_off = fc.update_v(
            v, u_on, u_off, *step_p, signal_batch[:, t - 1], backend,
        )
    return v, u_on, u_off


def _budget_at_units(
    v_pre,
    v_post,
    g_exc,
    g_inh,
    g_Ih_on,
    g_Ih_off,
    sig_t,
    backend,
    units: np.ndarray,
    v_ref: np.ndarray,
    *,
    batch: int = 0,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Slice unit budget from a completed ``update_v(..., return_budget=True)`` step."""
    units = np.asarray(units, dtype=np.int64)
    with torch.no_grad():
        u = torch.as_tensor(units, device=v_pre.device, dtype=torch.long)
        b = int(batch)
        sig_u = sig_t[b, u] if sig_t.dim() > 1 else sig_t[u]
        packed = torch.stack(
            (
                v_pre[b, u],
                g_exc[b, u],
                g_inh[b, u],
                g_Ih_on[b, u],
                g_Ih_off[b, u],
                sig_u,
                backend.e_leak[u],
                v_post[b, u],
            ),
            dim=0,
        ).detach().cpu().numpy()
        v_pre_np = packed[0]
        ref = v_ref[b, units] if np.ndim(v_ref) == 2 else v_ref[units]
        terms = fc.v_budget_from_g(
            v_pre_np, packed[1], packed[2], packed[3], packed[4], packed[5], packed[6],
        )
        v_abs = packed[7]
        bud = {
            "v_pre_d": v_pre_np - ref,
            "v_abs": v_abs,
            "g_exc": packed[1],
            "g_inh": packed[2],
            "g_Ih_on": packed[3],
            "g_Ih_off": packed[4],
            "signal": packed[5],
            **terms,
            "num": (
                terms["num_exc"] + terms["num_inh"] + terms["num_leak"]
                + terms["num_ihon"] + terms["num_ihoff"] + terms["num_cdt"] + terms["num_sig"]
            ),
        }
        v_post_minus_pre_u = v_abs - v_pre_np
    return bud, v_post_minus_pre_u


def _log(msg: str) -> None:
    print(msg, flush=True)


_BUDGET_KEYS = (
    "v_pre_d", "v_abs", "signal", "g_exc", "g_inh", "g_Ih_on", "g_Ih_off",
    "num_exc", "num_inh", "num_leak", "num_ihon", "num_ihoff", "num_cdt", "num_sig",
    "num", "den",
)
_N_BUDGET_KEYS = len(_BUDGET_KEYS)
_I_V_ABS = _BUDGET_KEYS.index("v_abs")
_I_V_PRE_D = _BUDGET_KEYS.index("v_pre_d")

# Plot series key → budget accum key (None → SEM is identically 0).
_PLOT_KEY_BUDGET: dict[str, str | None] = {
    "v_post_mV": "v_abs",
    "cdt_nS": None,
    "g_exc_nS": "g_exc",
    "g_inh_nS": "g_inh",
    "g_leak_nS": None,
    "g_Ih_on_nS": "g_Ih_on",
    "g_Ih_off_nS": "g_Ih_off",
    "num_cdt": "num_cdt",
    "num_exc": "num_exc",
    "num_inh": "num_inh",
    "num_leak": "num_leak",
    "num_ihon": "num_ihon",
    "num_ihoff": "num_ihoff",
    "num_sig": "num_sig",
    "num_over_den": "v_abs",
    "num": "num",
    "den": "den",
    "signal": "signal",
}


def _bud_matrix(bud: dict[str, np.ndarray]) -> np.ndarray:
    """Stack budget keys to ``(n_units, n_keys)`` for vectorized accumulate."""
    return np.column_stack([bud[k] for k in _BUDGET_KEYS])


def _acc_dict_from_row(row: np.ndarray) -> dict[str, float]:
    return {k: float(row[i]) for i, k in enumerate(_BUDGET_KEYS)}


def _sem_from_sum_sumsq(sum_: float, sumsq: float, n: int) -> float:
    """Match ``figure.util.sem_from_traces`` (population std / sqrt(n))."""
    if n <= 1:
        return 0.0
    mean = sum_ / n
    var = sumsq / n - mean * mean
    if var <= 0.0:
        return 0.0
    return float(np.sqrt(var) / np.sqrt(n))


def _step_sem(
    acc: dict[str, float], accsq: dict[str, float], n: int,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for plot_key, bud_key in _PLOT_KEY_BUDGET.items():
        if bud_key is None:
            out[plot_key] = 0.0
        else:
            out[plot_key] = _sem_from_sum_sumsq(acc[bud_key], accsq[bud_key], n)
    return out


def _step_from_acc(
    *,
    rel: int,
    ti: int,
    v_post_val: float,
    acc: dict[str, float],
    accsq: dict[str, float],
    v_post_minus_pre_sum: float,
    n: int,
) -> dict[str, Any]:
    """One step dict from per-key sums over ``n`` units."""
    if n <= 0:
        raise ValueError("empty unit set for mean budget")
    num = acc["num"] / n
    den = acc["den"] / n
    return {
        "rel": rel,
        "ti": ti,
        "v_post_mV": float(v_post_val),
        "v_pre_d_mV": acc["v_pre_d"] / n,
        "v_post_minus_pre_mV": v_post_minus_pre_sum / n,
        "signal": acc["signal"] / n,
        "g_exc_nS": acc["g_exc"] / n,
        "g_inh_nS": acc["g_inh"] / n,
        "g_leak_nS": float(fc.g_leak),
        "cdt_nS": float(fc.cdt),
        "g_Ih_on_nS": acc["g_Ih_on"] / n,
        "g_Ih_off_nS": acc["g_Ih_off"] / n,
        "num_exc": acc["num_exc"] / n,
        "num_inh": acc["num_inh"] / n,
        "num_leak": acc["num_leak"] / n,
        "num_ihon": acc["num_ihon"] / n,
        "num_ihoff": acc["num_ihoff"] / n,
        "num_cdt": acc["num_cdt"] / n,
        "num_sig": acc["num_sig"] / n,
        "num": num,
        "den": den,
        # mean over units of (num_u/den_u); SEM matches ``sem_from_traces`` on those ratios
        "num_over_den": acc["v_abs"] / n,
        "sem": _step_sem(acc, accsq, n),
        "n_units": n,
    }


@dataclass
class _BudgetWalkBatch:
    """One signal batch row for the shared post-t_on v walk."""

    all_units: np.ndarray
    t0_u: np.ndarray
    win_len: int
    unit_to_cell: dict[int, str]
    units_by_cell: dict[str, np.ndarray]


@dataclass
class _BudgetWalkAccum:
    sums: list[dict[str, np.ndarray]]
    sumsq: list[dict[str, np.ndarray]]
    counts: list[dict[str, np.ndarray]]
    v_post_minus_pre_sums: list[dict[str, np.ndarray]]


def _walk_budget(
    session,
    p,
    signal: torch.Tensor,
    batches: list[_BudgetWalkBatch],
    cells: list[str],
    *,
    rel_start: int | None = None,
    rel_stop: int | None = None,
) -> _BudgetWalkAccum:
    """Equilibrate once; walk post-t_on; accumulate budget + absolute v_post.

    Shared by bar average, spot average, and bar hex. ``rel = t_global - t0_u``.
    v_post is mean absolute ``v_abs``; SEM uses sum / sumsq like ``sem_from_traces``.

    If ``rel_start``/``rel_stop`` are set, only open budget accumulate inside that
    inclusive rel window; cheap ``update_v`` before it; break after every unit
    has passed ``rel_stop``.
    """
    if not batches:
        raise SystemExit("budget walk requires at least one batch")
    if (rel_start is None) ^ (rel_stop is None):
        raise SystemExit("rel_start and rel_stop must both be set or both omitted")
    if rel_start is not None and rel_start > rel_stop:
        raise SystemExit(f"rel_start={rel_start} > rel_stop={rel_stop}")
    B, T, _N = signal.shape
    if B != len(batches):
        raise SystemExit(f"signal B={B} != len(batches)={len(batches)}")
    t_on = int(session.primary_pack.signal.shape[1] - session.primary_pack.data.shape[1])
    trace_len = T - t_on

    # Last absolute time that still needs a step for the requested rel window.
    t_last: int | None = None
    if rel_stop is not None:
        t_last = max(int(plan.t0_u.max()) + int(rel_stop) for plan in batches)

    v, u_on, u_off = _equilibrate(session, p, signal, t_on)
    v_ref = v.detach().cpu().numpy().copy()
    backend = session.backend
    step_p = _v_step_params(p)

    sums_b: list[dict[str, np.ndarray]] = []
    sumsq_b: list[dict[str, np.ndarray]] = []
    counts_b: list[dict[str, np.ndarray]] = []
    v_post_minus_pre_sums_b: list[dict[str, np.ndarray]] = []
    for plan in batches:
        wl = plan.win_len
        sums_b.append({c: np.zeros((wl, _N_BUDGET_KEYS), dtype=float) for c in cells})
        sumsq_b.append({c: np.zeros((wl, _N_BUDGET_KEYS), dtype=float) for c in cells})
        counts_b.append({c: np.zeros(wl, dtype=np.int64) for c in cells})
        v_post_minus_pre_sums_b.append({c: np.zeros(wl, dtype=float) for c in cells})

    unit_lookups = [_unit_cell_lookup(plan, cells) for plan in batches]

    for ti in range(trace_len):
        t_global = t_on + ti
        if t_last is not None and t_global > t_last:
            break
        sig_t = signal[:, t_global - 1]
        actives: list[tuple[np.ndarray, np.ndarray] | None] = []
        need_budget = False
        for plan in batches:
            au = plan.all_units
            rel_u = t_global - plan.t0_u
            in_win = (rel_u >= 0) & (rel_u < plan.win_len)
            if rel_start is not None:
                in_win = in_win & (rel_u >= rel_start) & (rel_u <= rel_stop)
            if np.any(in_win):
                need_budget = True
                actives.append((au[in_win], rel_u[in_win].astype(np.int64)))
            else:
                actives.append(None)

        if not need_budget:
            v, u_on, u_off = fc.update_v(
                v, u_on, u_off, *step_p, sig_t, backend,
            )
            continue

        with torch.no_grad():
            v_pre = v
            v, u_on, u_off, g_exc, g_inh, g_Ih_on, g_Ih_off = fc.update_v(
                v, u_on, u_off, *step_p, sig_t, backend, return_budget=True,
            )

        for b, plan in enumerate(batches):
            active_pack = actives[b]
            if active_pack is None:
                continue
            active, active_rel = active_pack
            bud, v_post_minus_pre_u = _budget_at_units(
                v_pre, v, g_exc, g_inh, g_Ih_on, g_Ih_off, sig_t,
                backend, active, v_ref, batch=b,
            )
            bud_mat = _bud_matrix(bud)
            lookup = unit_lookups[b]
            tags = lookup[active]
            for ci, cell in enumerate(cells):
                mask = tags == ci
                if not np.any(mask):
                    continue
                rels = active_rel[mask]
                chunk = bud_mat[mask]
                np.add.at(sums_b[b][cell], rels, chunk)
                np.add.at(sumsq_b[b][cell], rels, chunk * chunk)
                np.add.at(v_post_minus_pre_sums_b[b][cell], rels, v_post_minus_pre_u[mask])
                np.add.at(counts_b[b][cell], rels, 1)

    return _BudgetWalkAccum(
        sums=sums_b,
        sumsq=sumsq_b,
        counts=counts_b,
        v_post_minus_pre_sums=v_post_minus_pre_sums_b,
    )


def _v_post_from_accum(sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Mean absolute v_post from accumulated ``v_abs``."""
    n = np.maximum(counts, 1)
    v_post = sums[:, _I_V_ABS] / n
    v_post[counts == 0] = 0.0
    return v_post


def _v_post_d_from_accum(
    sums: np.ndarray, v_post_minus_pre_sums: np.ndarray, counts: np.ndarray,
) -> np.ndarray:
    """Mean ``v_post_d`` = v_post − v_ref = v_pre_d + v_post_minus_pre."""
    n = np.maximum(counts, 1)
    v_post_d = sums[:, _I_V_PRE_D] / n + v_post_minus_pre_sums / n
    v_post_d[counts == 0] = 0.0
    return v_post_d


def _dominant_drive_from_step(step: dict[str, Any] | None) -> str | None:
    if step is None:
        return None
    if abs(step["num_exc"]) >= abs(step["num_inh"]):
        return "exc" if abs(step["num_exc"]) > 1e-9 else "none"
    return "inh"


def _finalize_budget_report(
    *,
    cell: str,
    target: str,
    spec: str | None,
    mode: str,
    before_steps: int,
    units: np.ndarray,
    p,
    session,
    sums: np.ndarray,
    sumsq: np.ndarray,
    counts: np.ndarray,
    v_post_minus_pre_sums: np.ndarray,
    v_post: np.ndarray,
    rel_start: int | None,
    rel_stop: int | None,
    ti_mode: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one report dict from a single batch×cell accum row."""
    # Peak on |v_post_d| (= |v_post − v_ref| = |v_pre_d + v_post_minus_pre|).
    v_post_d = _v_post_d_from_accum(sums, v_post_minus_pre_sums, counts)
    if rel_start is not None and rel_stop is not None:
        rel_lo, rel_hi = rel_start, rel_stop
        seg = v_post_d[rel_lo:rel_hi + 1]
        peak_rel = rel_lo + int(np.argmax(np.abs(seg))) if seg.size else rel_lo
    else:
        peak_rel = _v_post_d_peak_rel(v_post_d, before_steps)
        rel_lo = max(0, peak_rel - 4)
        rel_hi = min(v_post_d.size - 1, peak_rel + 8)
    steps: list[dict[str, Any]] = []
    peak_step: dict[str, Any] | None = None
    for rel in range(rel_lo, rel_hi + 1):
        n = int(counts[rel])
        if n == 0:
            continue
        if ti_mode == "rel":
            ti = rel
        elif ti_mode == "abs_minus_before":
            ti = rel - before_steps
        else:
            raise ValueError(f"unknown ti_mode {ti_mode!r}")
        step = _step_from_acc(
            rel=rel, ti=ti, v_post_val=float(v_post[rel]),
            acc=_acc_dict_from_row(sums[rel]),
            accsq=_acc_dict_from_row(sumsq[rel]),
            v_post_minus_pre_sum=float(v_post_minus_pre_sums[rel]), n=n,
        )
        steps.append(step)
        if rel == peak_rel:
            peak_step = step
    if peak_step is None and steps:
        peak_step = steps[len(steps) // 2]
    if ti_mode == "abs_minus_before" and v_post_d.size > before_steps:
        onset = _first_nonzero_rel(v_post_d[before_steps:])
    else:
        onset = _first_nonzero_rel(v_post_d)
    report: dict[str, Any] = {
        "mode": mode,
        "cell": cell,
        "n_units": int(units.size),
        "target": target,
        "spec": spec,
        "before_steps": before_steps,
        "v_post_d_peak_rel": peak_rel,
        "v_post_d_peak_mV": float(v_post_d[peak_rel]),
        "v_post_d_polarity": _polarity(float(v_post_d[peak_rel])),
        "rel_window": [rel_lo, rel_hi],
        "v_post_d_onset_rel": onset,
        "params": _unit_params(p, session.backend, int(units[0])),
        "globals": _globals(session),
        "steps": steps,
        "peak_step": peak_step,
        "peak_drive": _dominant_drive_from_step(peak_step),
        "v_post": v_post.tolist(),
    }
    if extra:
        report.update(extra)
    return report


def _first_nonzero_rel(trace: np.ndarray, *, eps: float = 1e-6) -> int | None:
    idx = np.where(np.abs(trace) > eps)[0]
    return int(idx[0]) if idx.size else None


def _v_post_d_peak_rel(
    v_post_d: np.ndarray,
    before_steps: int | None,
    *,
    horizon: int | None = 40,
) -> int:
    """Index of largest |v_post_d| (= |v_post − v_ref|) after onset (optional ``horizon``)."""
    arr = np.asarray(v_post_d, dtype=float)
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


def _unit_params(p, backend, unit: int) -> dict[str, float]:
    return {
        "in_gain": float(p["in_gain"][unit]),
        "out_gain": float(p["out_gain"][unit]),
        "v_th_mV": float(p["v_th"][unit]),
        "Ih_gmax": float(p["Ih_gmax"][unit]),
        "Ih_gmax_off": float(p["Ih_gmax_off"][unit]),
        "e_leak_mV": float(backend.e_leak[unit]),
    }


def _globals(session):
    pack = session.primary_pack
    return {
        "E_exc": fc.E_exc,
        "E_inh": fc.E_inh,
        "E_Ih": fc.E_Ih,
        "E_IH_OFF": fc.E_IH_OFF,
        "g_leak_nS": fc.g_leak,
        "cdt": fc.cdt,
        "deltat_ms": fc.deltat,
        "t_on": int(pack.signal.shape[1] - pack.data.shape[1]),
    }


def _unit_to_cell_map(units_by_cell: dict[str, np.ndarray]) -> dict[int, str]:
    u2c: dict[int, str] = {}
    for cell, us in units_by_cell.items():
        for u in np.asarray(us, dtype=np.int64).ravel():
            u2c[int(u)] = cell
    return u2c


def _unit_cell_lookup(plan: _BudgetWalkBatch, cells: list[str]) -> np.ndarray:
    """Map unit id → index in ``cells`` (-1 if absent). Length ``max(unit_id)+1``."""
    cell_i = {c: i for i, c in enumerate(cells)}
    if plan.all_units.size == 0:
        return np.empty(0, dtype=np.int32)
    out = np.full(int(plan.all_units.max()) + 1, -1, dtype=np.int32)
    for u, cname in plan.unit_to_cell.items():
        ci = cell_i.get(cname)
        if ci is not None:
            out[int(u)] = ci
    return out


def _merge_walk_accum(
    accum: _BudgetWalkAccum,
    walk_batches: list[_BudgetWalkBatch],
    cells: list[str],
    win_len: int,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    """Sum per-cell accum rows across walk batches (spot multi-stimulus mean)."""
    sums = {c: np.zeros((win_len, _N_BUDGET_KEYS), dtype=float) for c in cells}
    sumsq = {c: np.zeros((win_len, _N_BUDGET_KEYS), dtype=float) for c in cells}
    counts = {c: np.zeros(win_len, dtype=np.int64) for c in cells}
    v_post_minus_pre_sums = {c: np.zeros(win_len, dtype=float) for c in cells}
    units_ref = {c: np.zeros(0, dtype=np.int64) for c in cells}
    for b, plan in enumerate(walk_batches):
        for cell in cells:
            if cell not in plan.units_by_cell:
                continue
            us = plan.units_by_cell[cell]
            if us.size == 0:
                continue
            if units_ref[cell].size == 0:
                units_ref[cell] = us
            sums[cell] += accum.sums[b][cell]
            sumsq[cell] += accum.sumsq[b][cell]
            counts[cell] += accum.counts[b][cell]
            v_post_minus_pre_sums[cell] += accum.v_post_minus_pre_sums[b][cell]
    return sums, sumsq, counts, v_post_minus_pre_sums, units_ref


def _make_walk_batch(
    units_by_cell: dict[str, np.ndarray],
    *,
    t0_bn_row: np.ndarray,
    win_len: int,
) -> _BudgetWalkBatch:
    """Build one walk batch; ``t0_u[i] = t0_bn_row[all_units[i]]``."""
    all_units = np.unique(np.concatenate([us for us in units_by_cell.values()]))
    return _BudgetWalkBatch(
        all_units=all_units,
        t0_u=np.asarray(t0_bn_row[all_units], dtype=np.int64),
        win_len=int(win_len),
        unit_to_cell=_unit_to_cell_map(units_by_cell),
        units_by_cell=units_by_cell,
    )


# ---------------------------------------------------------------------------
# Average bar budgets (cost-extent)
# ---------------------------------------------------------------------------


def _parse_syn_strength(
    syn_strength: str | None,
    session,
) -> dict[tuple[int, int], float]:
    """Parse ``--syn-strength SRC,TAR,VALUE,...`` (flat commas; len % 3 == 0)."""
    if not syn_strength:
        return {}
    if session.backend.network is None:
        raise SystemExit("--syn-strength requires a network backend")
    names = list(session.backend.network.type_names)
    name_to_i = {n: i for i, n in enumerate(names)}
    parts = parse_comma_list(syn_strength)
    if len(parts) % 3 != 0:
        raise SystemExit(
            "--syn-strength must be SRC,TAR,VALUE repeated (comma-separated; length multiple of 3)"
        )
    out: dict[tuple[int, int], float] = {}
    for i in range(0, len(parts), 3):
        src_name, tar_name, val_s = parts[i], parts[i + 1], parts[i + 2]
        if src_name not in name_to_i:
            raise SystemExit(f"--syn-strength unknown source type {src_name!r}")
        if tar_name not in name_to_i:
            raise SystemExit(f"--syn-strength unknown target type {tar_name!r}")
        try:
            val = float(val_s)
        except ValueError as exc:
            raise SystemExit(f"--syn-strength bad VALUE {val_s!r}") from exc
        out[(name_to_i[src_name], name_to_i[tar_name])] = val
    return out


def _apply_syn_strength(
    z: torch.Tensor,
    schema: list,
    session,
    edits: dict[tuple[int, int], float],
) -> torch.Tensor:
    """Return a copy of ``z`` with ``syn_strength`` overrides applied."""
    if not edits:
        return z
    names = list(session.backend.network.type_names)
    keys = session.backend.conn.pair_keys
    key_to_i = {k: i for i, k in enumerate(keys)}
    named = fc.z_to_unit_values(z, schema)
    if "syn_strength" not in named:
        raise SystemExit("schema missing syn_strength segment")
    arr = np.array(named["syn_strength"], dtype=np.float64, copy=True)
    for (src_i, tar_i), val in edits.items():
        pair = (src_i, tar_i)
        if pair not in key_to_i:
            raise SystemExit(
                f"no type pair {names[src_i]!r} -> {names[tar_i]!r} in connectome"
            )
        pair_i = key_to_i[pair]
        arr[pair_i] = val
        _log(f"syn_strength {names[src_i]} -> {names[tar_i]} = {val:g}")
    named["syn_strength"] = arr
    return fc.unit_values_to_z(named, schema, dtype=z.dtype, device=z.device)


def _bar_meta(session, target: str):
    """One-shot ``(specs, grids)`` for a moving-bar target."""
    specs = bar_specs_for_session(session, target)
    pack = session.pack_for(target)
    grids = moving_bar_session_t0_grids(
        session, specs, pack.cost_extent, int(session.maxtime),
        t_on=int(pack.signal.shape[1] - pack.data.shape[1]), deltat_ms=fc.deltat,
    )
    return specs, grids


def _bar_specs_requested(
    session,
    target: str,
    cells: list[str],
    requested: list[str] | None,
    *,
    specs=None,
    grids=None,
) -> list[str]:
    """Spec list for average-mode bar without a full forward bundle."""
    if specs is None or grids is None:
        specs, grids = _bar_meta(session, target)
    all_specs = [s.name for s in specs]
    try:
        if requested is not None:
            return filter_requested_specs(all_specs, requested)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    row = moving_bar_row_specs(session, target, grids.side)
    out: list[str] = []
    for cell in cells:
        for s in row.get(cell, all_specs):
            if s in all_specs and s not in out:
                out.append(s)
    return out or list(all_specs)


def _resolve_bar_spec_signal(
    session,
    target: str,
    spec_names: list[str],
    *,
    specs=None,
    grids=None,
):
    """Validate specs; return ``(pack, specs, grids, bis, signal, t0_bn)``."""
    if target not in fc.MOVING_BAR_TARGETS:
        raise SystemExit(f"unsupported target {target!r}")
    if not spec_names:
        raise SystemExit("bar budget walk requires at least one spec")
    pack = session.pack_for(target)
    if specs is None or grids is None:
        specs, grids = _bar_meta(session, target)
    name_to_bi = {s.name: i for i, s in enumerate(specs)}
    missing = [s for s in spec_names if s not in name_to_bi]
    if missing:
        raise SystemExit(f"spec(s) {missing} not in {[s.name for s in specs]}")
    bis = [name_to_bi[s] for s in spec_names]
    return pack, specs, grids, bis, pack.signal[bis], np.asarray(grids.t0_bn)


def _analyze_budget_walk(
    session,
    *,
    p,
    cells: list[str],
    target: str,
    signal: torch.Tensor,
    walk_batches: list[_BudgetWalkBatch],
    before_steps: list[int],
    batch_specs: list[str | None],
    rel_start: int | None,
    rel_stop: int | None,
    mode: str,
    ti_mode: str,
    merge_batches: bool = False,
    extra: dict[str, Any] | None = None,
    extra_for_cell=None,
    n_units_for_cell=None,
):
    """Shared spot/bar: ``_walk_budget`` → finalize reports.

    * ``merge_batches=False`` (bar): ``reports[spec][cell]``; ``batch_specs`` are str.
    * ``merge_batches=True`` (spot): sum across batches → ``reports[cell]``.
    """
    if not walk_batches:
        raise SystemExit("budget walk requires at least one batch")
    if len(before_steps) != len(walk_batches) or len(batch_specs) != len(walk_batches):
        raise SystemExit("before_steps/batch_specs length must match walk_batches")

    accum = _walk_budget(
        session, p, signal, walk_batches, cells,
        rel_start=rel_start, rel_stop=rel_stop,
    )

    def _one_report(
        *,
        cell: str,
        spec: str | None,
        before: int,
        units: np.ndarray,
        sums: np.ndarray,
        sumsq: np.ndarray,
        counts: np.ndarray,
        v_post_minus_pre_sums: np.ndarray,
    ) -> dict[str, Any]:
        if units.size == 0:
            raise SystemExit(f"no units for cell {cell!r}")
        v_post = _v_post_from_accum(sums, counts)
        v_post_d = _v_post_d_from_accum(sums, v_post_minus_pre_sums, counts)
        cell_extra = dict(extra) if extra else {}
        if extra_for_cell is not None:
            cell_extra.update(extra_for_cell(cell, v_post, v_post_d) or {})
        report = _finalize_budget_report(
            cell=cell,
            target=target,
            spec=spec,
            mode=mode,
            before_steps=before,
            units=units,
            p=p,
            session=session,
            sums=sums,
            sumsq=sumsq,
            counts=counts,
            v_post_minus_pre_sums=v_post_minus_pre_sums,
            v_post=v_post,
            rel_start=rel_start,
            rel_stop=rel_stop,
            ti_mode=ti_mode,
            extra=cell_extra or None,
        )
        if n_units_for_cell is not None:
            report["n_units"] = int(n_units_for_cell(cell))
        return report

    if merge_batches:
        win_len = walk_batches[0].win_len
        sums, sumsq, counts, v_post_minus_pre_sums, units_ref = _merge_walk_accum(
            accum, walk_batches, cells, win_len,
        )
        before = int(before_steps[0])
        return {
            cell: _one_report(
                cell=cell,
                spec=None,
                before=before,
                units=units_ref[cell],
                sums=sums[cell],
                sumsq=sumsq[cell],
                counts=counts[cell],
                v_post_minus_pre_sums=v_post_minus_pre_sums[cell],
            )
            for cell in cells
        }

    out: dict[str, dict[str, dict[str, Any]]] = {}
    for b, spec in enumerate(batch_specs):
        if spec is None:
            raise SystemExit("non-merge budget walk requires batch_specs as str")
        out[spec] = {}
        for cell in cells:
            out[spec][cell] = _one_report(
                cell=cell,
                spec=spec,
                before=int(before_steps[b]),
                units=walk_batches[b].units_by_cell[cell],
                sums=accum.sums[b][cell],
                sumsq=accum.sumsq[b][cell],
                counts=accum.counts[b][cell],
                v_post_minus_pre_sums=accum.v_post_minus_pre_sums[b][cell],
            )
    return out


def _analyze_bar_walk(
    session,
    *,
    p,
    cells: list[str],
    target: str,
    spec_names: list[str],
    rel_start: int | None,
    rel_stop: int | None,
    units_for_bi,
    mode: str,
    specs=None,
    grids=None,
    extra: dict[str, Any] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Bar prep: resolve signal/specs → ``_analyze_budget_walk`` (no merge)."""
    pack, specs, grids, bis, signal, t0_bn = _resolve_bar_spec_signal(
        session, target, spec_names, specs=specs, grids=grids,
    )
    before_b: list[int] = []
    walk_batches: list[_BudgetWalkBatch] = []
    for bi, spec in zip(bis, spec_names):
        usets = units_for_bi(bi, spec, pack=pack, t0_bn=t0_bn)
        before = int(grids.before_steps[spec])
        after = int(grids.after_steps[spec])
        before_b.append(before)
        walk_batches.append(
            _make_walk_batch(usets, t0_bn_row=t0_bn[bi], win_len=before + after + 1),
        )
    return _analyze_budget_walk(
        session,
        p=p,
        cells=cells,
        target=target,
        signal=signal,
        walk_batches=walk_batches,
        before_steps=before_b,
        batch_specs=list(spec_names),
        rel_start=rel_start,
        rel_stop=rel_stop,
        mode=mode,
        ti_mode="rel",
        merge_batches=False,
        extra=extra,
    )


def analyze_bar_average(
    session,
    *,
    p,
    cells: list[str],
    target: str,
    spec_names: list[str],
    rel_start: int | None,
    rel_stop: int | None,
    specs=None,
    grids=None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """One batched v walk over all requested specs; mean budget per cell.

    Returns ``reports[spec][cell]``. v_post + budget share ``_walk_budget``.
    """
    cols_holder: list = []

    def units_for_bi(bi, spec, *, pack, t0_bn):
        C = session.backend.network
        if not cols_holder:
            cols_holder.append(moving_bar_cost_columns(C, cost_extent=pack.cost_extent))
        cols = cols_holder[0]
        out: dict[str, np.ndarray] = {}
        for cell in cells:
            try:
                units = moving_bar_units_on_columns(C, cell, cols)
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
            units = units[t0_bn[bi, units] >= 0]
            if units.size == 0:
                raise SystemExit(f"no valid {cell} units in cost_extent for bar aggregation")
            out[cell] = units
        return out

    return _analyze_bar_walk(
        session,
        p=p,
        cells=cells,
        target=target,
        spec_names=spec_names,
        rel_start=rel_start,
        rel_stop=rel_stop,
        units_for_bi=units_for_bi,
        mode="average",
        specs=specs,
        grids=grids,
    )


# ---------------------------------------------------------------------------
# Average spot budgets (center-bin / stim-on-column)
# ---------------------------------------------------------------------------


def _spot_session_layout(session_one, cells: list[str]):
    """Session-scoped center-bin layout for spot budget walks."""
    pack = session_one.primary_pack
    C = session_one.backend.network
    if C is None:
        raise SystemExit("spot average requires a network backend")
    opts = dict((session_one.train_opts or {}).get(f"{pack.name}_stimulus_opts") or {})
    spot = spot_from_opts(C, stimulus_opts=opts)
    (
        batch_idx, unit_idx, _radius, type_idx, _stim_u, _stim_v, _du, _dv, center_row,
    ) = spot_center_bin_layout(
        C,
        spot_stimulus_batches(spot),
        resolve_spot_cost_radii(stimulus_opts=opts),
        pack.cost_extent,
    )
    type_i: dict[str, int] = {}
    for cell in cells:
        if cell not in C.type_names:
            raise SystemExit(f"unknown cell {cell!r}")
        type_i[cell] = C.type_names.index(cell)
    return pack, batch_idx, unit_idx, type_idx, center_row, type_i


def analyze_spot_average(
    session_one,
    *,
    p,
    cells: list[str],
    target: str,
    abs_start: int | None,
    abs_stop: int | None,
) -> dict[str, dict[str, Any]]:
    """One batched v walk over spot stimulus rows; mean center-bin budget."""
    if target not in fc.SPOT_TARGETS:
        raise SystemExit(f"unsupported target {target!r}")
    pack, batch_idx, unit_idx, type_idx, center_row, type_i = _spot_session_layout(
        session_one, cells,
    )
    t_on = int(pack.signal.shape[1] - pack.data.shape[1])

    sig = pack.signal if pack.signal.dim() == 3 else pack.signal.unsqueeze(0)
    B_all, T, _N = sig.shape
    t0_abs = np.zeros(_N, dtype=np.int64)
    win_len = int(abs_stop) + 1 if abs_stop is not None else T

    walk_batches: list[_BudgetWalkBatch] = []
    sig_rows: list[int] = []
    for b in range(B_all):
        row_mask = center_row & (batch_idx == b)
        if not np.any(row_mask):
            continue
        usets: dict[str, np.ndarray] = {}
        for cell in cells:
            m = row_mask & (type_idx == type_i[cell])
            if np.any(m):
                usets[cell] = np.unique(unit_idx[m])
        if not usets:
            continue
        walk_batches.append(
            _make_walk_batch(usets, t0_bn_row=t0_abs, win_len=win_len),
        )
        sig_rows.append(b)

    if not walk_batches:
        raise SystemExit("no center-bin units for requested cells in spot layout")

    ref_on = spot_ref_cubes(
        session_one, pack.name, dark=(pack.name == "spot_dark"),
    )

    def extra_for_cell(
        cell: str, v_post: np.ndarray, v_post_d: np.ndarray,
    ) -> dict[str, Any]:
        del v_post  # peak time from |v_post_d|; absolute series unused here
        extra: dict[str, Any] = {"ref_peak_mV": None}
        if cell in ref_on:
            peak_probe = _v_post_d_peak_rel(v_post_d, t_on)
            ref_cube = np.asarray(ref_on[cell], dtype=float)
            if peak_probe < ref_cube.shape[1]:
                extra["ref_peak_mV"] = float(ref_cube[CENTER_BIN, peak_probe])
        return extra

    def n_units_for_cell(cell: str) -> int:
        # Total center readouts across layout (matches prior semantics).
        return int(np.sum(center_row & (type_idx == type_i[cell])))

    n_b = len(walk_batches)
    return _analyze_budget_walk(
        session_one,
        p=p,
        cells=cells,
        target=target,
        signal=sig[sig_rows],
        walk_batches=walk_batches,
        before_steps=[t_on] * n_b,
        batch_specs=[None] * n_b,
        rel_start=abs_start,
        rel_stop=abs_stop,
        mode="average",
        ti_mode="abs_minus_before",
        merge_batches=True,
        extra_for_cell=extra_for_cell,
        n_units_for_cell=n_units_for_cell,
    )


# ---------------------------------------------------------------------------
# Hex-mode bar (single column; same walk as average)
# ---------------------------------------------------------------------------


def _units_at_hex(session, cell: str, *, at_x: float, at_y: float, cost_extent: int):
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
    units = col2subtype(C, int(col.u), int(col.v), cell).tolist()
    if not units:
        raise SystemExit(f"no {cell} unit at hex ({at_x},{at_y})")
    return col, units


def analyze_bar_hex(
    session,
    *,
    p,
    cell: str,
    target: str,
    spec_names: list[str],
    at_x: float,
    at_y: float,
    unit: int | None,
    rel_start: int | None,
    rel_stop: int | None,
    specs=None,
    grids=None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """One batched v walk over specs at one hex; returns ``reports[spec][cell]``."""
    pack = session.pack_for(target)
    col, units = _units_at_hex(
        session, cell, at_x=at_x, at_y=at_y, cost_extent=pack.cost_extent,
    )
    if unit is None:
        if len(units) > 1:
            raise SystemExit(f"multiple {cell} at ({at_x},{at_y}): {units}; pass --unit")
        unit = units[0]
    elif unit not in units:
        raise SystemExit(f"unit {unit} not in {units}")
    unit_arr = np.asarray([unit], dtype=np.int64)
    usets = {cell: unit_arr}

    def units_for_bi(bi, spec, *, pack, t0_bn):
        if int(t0_bn[bi, unit]) < 0:
            raise SystemExit(f"no t0 for unit {unit} on spec {spec!r}")
        return usets

    return _analyze_bar_walk(
        session,
        p=p,
        cells=[cell],
        target=target,
        spec_names=spec_names,
        rel_start=rel_start,
        rel_stop=rel_stop,
        units_for_bi=units_for_bi,
        mode="hex",
        specs=specs,
        grids=grids,
        extra={
            "unit": int(unit),
            "hex": {"x": at_x, "y": at_y},
            "uv": {"u": int(col.u), "v": int(col.v)},
        },
    )



# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_filename(report: dict[str, Any]) -> str:
    parts = [report["cell"], report["target"], report.get("mode", "average")]
    if report.get("spec"):
        parts.append(str(report["spec"]))
    if report.get("mode") == "hex":
        hx = report["hex"]
        parts.append(f"x{hx['x']}_y{hx['y']}")
    parts.append("v")
    return "_".join(parts) + ".png"


def _overlay_plot_filename(reports: list[dict[str, Any]]) -> str:
    r0 = reports[0]
    specs = "_".join(str(r["spec"]) for r in reports)
    return f"{r0['cell']}_{r0['target']}_overlay_{specs}_v.png"


def _budget_figure(title: str):
    """Shared grid figure: rows = panel groups, cols = traces within a row."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from figure.util import save_figure

    n_rows, n_cols = _budget_axes_grid()
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.6 * n_cols, 2.2 * n_rows),
        sharex=True,
        constrained_layout=True,
    )
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = axes[:, np.newaxis]
    return fig, axes, save_figure


def _hide_unused_axes(axes) -> None:
    n_rows, n_cols = axes.shape
    for ri, (_group_ylabel, series) in enumerate(_PLOT_PANELS):
        for ci in range(len(series), n_cols):
            axes[ri, ci].set_visible(False)


def _visible_axes(axes):
    return [ax for ax in axes.ravel() if ax.get_visible()]


def _style_budget_ax(
    ax, ylabel: str, *, legend_fontsize: int, legend_ncol: int, show_legend: bool = True,
) -> None:
    ax.set_ylabel(ylabel, fontsize=8)
    if show_legend:
        ax.legend(loc="best", fontsize=legend_fontsize, ncol=legend_ncol)
    ax.tick_params(labelsize=7)
    ax.grid(True, axis="y", alpha=0.3)


def _shared_row_ylim(
    curves: list[np.ndarray],
    *,
    floor_zero: bool = False,
    margin_frac: float = 0.06,
) -> tuple[float, float]:
    """Tight row ylim from data min/max + small relative pad (not symmetric)."""
    chunks: list[np.ndarray] = []
    for c in curves:
        v = np.asarray(c, dtype=float).ravel()
        v = v[np.isfinite(v)]
        if v.size:
            chunks.append(v)
    if not chunks:
        return -1.0, 1.0
    all_v = np.concatenate(chunks)
    ylo = float(np.min(all_v))
    yhi = float(np.max(all_v))
    if floor_zero and ylo >= 0.0:
        ylo = 0.0
    span = yhi - ylo
    pad = max(span * margin_frac, abs(yhi) * 0.02, 1e-3) if span > 0.0 else max(abs(yhi) * 0.05, 1e-3)
    return ylo - pad, yhi + pad


def _apply_shared_row_ylim(
    axes,
    row_curves: dict[int, list[np.ndarray]],
) -> None:
    """One tight data-driven ylim per row in ``_ROW_SHARED_YLIM``."""
    for ri, curves in row_curves.items():
        if not curves:
            continue
        _, series = _PLOT_PANELS[ri]
        ylo, yhi = _shared_row_ylim(curves)
        for ci in range(len(series)):
            axes[ri, ci].set_ylim(ylo, yhi)


def _save_budget_figure(fig, axes, *, before_steps, out_path, save_figure) -> None:
    _hide_unused_axes(axes)
    last_row = axes.shape[0] - 1
    for ci in range(axes.shape[1]):
        ax = axes[last_row, ci]
        if ax.get_visible():
            ax.set_xlabel("rel (spot: abs t; bar: aligned)", fontsize=8)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    save_figure(fig, out_path)
    print(f"wrote {out_path}")


def _plot_budget_reports(
    reports: list[dict[str, Any]],
    out_path: str,
    *,
    title: str,
) -> None:
    """Shared grid PNG for one or more reports (overlay = linestyle per report)."""
    if not reports:
        raise SystemExit("no reports to plot")
    for rep in reports:
        if not rep.get("steps"):
            raise SystemExit(f"no steps to plot for {rep.get('cell')} spec={rep.get('spec')!r}")

    fig, axes, save_figure = _budget_figure(title)
    colors = _plot_colors()
    linestyles = ("-", "--", "-.", ":")
    overlay = len(reports) > 1
    e_leak_mV = float(reports[0].get("params", {}).get("e_leak_mV", fc.E_LEAK_REST))
    row_curves: dict[int, list[np.ndarray]] = {ri: [] for ri in _ROW_SHARED_YLIM}

    for ri, (group_ylabel, series) in enumerate(_PLOT_PANELS):
        for ci, (key, label) in enumerate(series):
            ax = axes[ri, ci]
            color = (
                "0.0" if label in _BLACK_TRACE_LABELS
                else colors[ci % len(colors)]
            )
            show_legend = overlay and ri == 0 and ci == 0
            for si, rep in enumerate(reports):
                ls = linestyles[si % len(linestyles)] if overlay else "-"
                rel = np.asarray([s["rel"] for s in rep["steps"]], dtype=float)
                y = np.asarray([s[key] for s in rep["steps"]], dtype=float)
                sem = np.asarray(
                    [float(s.get("sem", {}).get(key, 0.0)) for s in rep["steps"]],
                    dtype=float,
                )
                if ri in _ROW_SHARED_YLIM:
                    row_curves[ri].append(y)
                    if np.any(sem):
                        row_curves[ri].append(y + sem)
                        row_curves[ri].append(y - sem)
                plot_sem_band(ax, rel, y, sem, color=color, alpha=0.3)
                ax.plot(
                    rel, y,
                    label=str(rep["spec"]) if show_legend else "_nolegend_",
                    color=color,
                    linestyle=ls,
                    linewidth=1.4,
                )
            e_note = _g_e_note(label, e_leak_mV=e_leak_mV)
            if e_note is not None:
                ax.set_title(e_note, fontsize=8)
            _style_budget_ax(
                ax, _trace_ylabel(group_ylabel, label),
                legend_fontsize=6 if overlay else 7,
                legend_ncol=1,
                show_legend=show_legend,
            )
    _apply_shared_row_ylim(axes, row_curves)
    _finish_budget_figure_layout(fig, title, colors)
    _save_budget_figure(
        fig, axes, before_steps=reports[0].get("before_steps"),
        out_path=out_path, save_figure=save_figure,
    )


def plot_report(report: dict[str, Any], out_path: str) -> None:
    """Write one multi-panel PNG: budget series vs rel (x-axis)."""
    title = (
        f"{report['cell']}  {report['target']}"
        + (f"  {report['spec']}" if report.get("spec") else "")
        + f"  mode={report.get('mode')}  n={report.get('n_units')}"
    )
    if report.get("mode") == "hex":
        title += f"  hex=({report['hex']['x']},{report['hex']['y']})"
    _plot_budget_reports([report], out_path, title=title)


def plot_reports_overlay(reports: list[dict[str, Any]], out_path: str) -> None:
    """One grid PNG: specs share color per subplot, differ by linestyle."""
    if not reports:
        raise SystemExit("no reports to overlay")
    r0 = reports[0]
    spec_list = ",".join(str(r["spec"]) for r in reports)
    title = (
        f"{r0['cell']}  {r0['target']}  overlay=[{spec_list}]"
        f"  mode={r0.get('mode')}  n={r0.get('n_units')}"
    )
    _plot_budget_reports(reports, out_path, title=title)


def _emit_report(
    report: dict[str, Any],
    *,
    run_dir: str,
    do_print: bool,
    do_plot: bool,
) -> None:
    if do_print:
        print("")
        _print_report(report)
    if do_plot:
        out = os.path.join(run_dir, "cell_dynamics", _plot_filename(report))
        plot_report(report, out)


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
    print(f"target={report['target']} spec={report.get('spec')}")
    print(
        f"v_post_d_peak={report['v_post_d_peak_mV']:+.4f} mV "
        f"v_post_d_polarity={report['v_post_d_polarity']}  "
        f"v_post_d_peak_rel={report['v_post_d_peak_rel']}  "
        f"before_steps={report.get('before_steps')}  "
        f"peak_drive={report.get('peak_drive')}"
    )
    print("trained:", "  ".join(f"{k}={v:.6g}" for k, v in report["params"].items()))

    print("\nrel  n  v_post  v_pre_d  v_post_minus_pre  sig   g_inh  g_Ih_off  g_exc  num_inh  num_exc")
    for s in report["steps"]:
        print(
            f"{s['rel']:4d} {s.get('n_units', 1):3d} {s['v_post_mV']:+8.4f} "
            f"{s['v_pre_d_mV']:+8.4f} {s['v_post_minus_pre_mV']:+8.4f} "
            f"{s['signal']:5.1f} {s['g_inh_nS']:.4f} {s['g_Ih_off_nS']:.4f} "
            f"{s['g_exc_nS']:.4f} {s['num_inh']:+8.2f} {s['num_exc']:+8.2f}"
        )

    ps = report.get("peak_step")
    if ps is not None:
        num = (
            ps["num_exc"] + ps["num_inh"] + ps["num_leak"]
            + ps["num_ihon"] + ps["num_ihoff"] + ps["num_cdt"] + ps["num_sig"]
        )
        print(f"\nNumerator at peak rel={ps['rel']} (num={num:.2f}):")
        for name, val in [
            ("i_cdt", ps["num_cdt"]), ("i_exc", ps["num_exc"]), ("i_inh", ps["num_inh"]),
            ("i_leak", ps["num_leak"]), ("i_h_on", ps["num_ihon"]), ("i_h_off", ps["num_ihoff"]),
            ("i_sig", ps["num_sig"]),
        ]:
            pct = 100.0 * val / num if num else 0.0
            print(f"  {name:8s} {val:+9.2f} ({pct:.0f}%)")


def _print_polarity_compare(
    spot_reports: dict[str, dict[str, Any]],
    bar_reports: dict[str, dict[str, Any]],
) -> None:
    cells = sorted(set(spot_reports) & set(bar_reports))
    if not cells:
        return
    print("\n======== SPOT vs BAR polarity (cost-extent averages) ========")
    print(
        f"{'cell':6s} {'spot_post_d':>11s} {'spot_pol':>8s} {'spot_drv':>8s} "
        f"{'bar_post_d':>10s} {'bar_pol':>8s} {'bar_drv':>8s}  note"
    )
    for cell in cells:
        s = spot_reports[cell]
        b = bar_reports[cell]
        flip = s["v_post_d_polarity"] != b["v_post_d_polarity"] and "0" not in (
            s["v_post_d_polarity"], b["v_post_d_polarity"],
        )
        note = "FLIP" if flip else "same"
        if flip and s.get("peak_drive") and b.get("peak_drive"):
            if s["peak_drive"] != b["peak_drive"]:
                note += f" (drive {s['peak_drive']}→{b['peak_drive']})"
            else:
                note += f" (same drive={s['peak_drive']}; see num terms)"
        print(
            f"{cell:6s} {s['v_post_d_peak_mV']:+11.4f} {s['v_post_d_polarity']:>8s} "
            f"{str(s.get('peak_drive')):>8s} "
            f"{b['v_post_d_peak_mV']:+10.4f} {b['v_post_d_polarity']:>8s} "
            f"{str(b.get('peak_drive')):>8s}  {note}"
        )
        # short diagnosis at peak
        sps, bps = s.get("peak_step"), b.get("peak_step")
        if sps and bps:
            print(
                f"       spot@peak: g_exc={sps['g_exc_nS']:.4f} g_inh={sps['g_inh_nS']:.4f} "
                f"num_exc={sps['num_exc']:+.1f} num_inh={sps['num_inh']:+.1f} "
                f"v_pre_d={sps['v_pre_d_mV']:+.3f}"
            )
            print(
                f"       bar @peak: g_exc={bps['g_exc_nS']:.4f} g_inh={bps['g_inh_nS']:.4f} "
                f"num_exc={bps['num_exc']:+.1f} num_inh={bps['num_inh']:+.1f} "
                f"v_pre_d={bps['v_pre_d_mV']:+.3f}"
            )




def main() -> None:
    if __package__ is None:
        raise SystemExit(
            "run as a module from SimulationCode/: "
            "../.venv/bin/python -m analyze.cell_dynamics ..."
        )

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_shared_cli(ap)
    ap.add_argument("--unit", type=int, default=None, help="hex-mode unit index")
    ap.add_argument("--rel", default=None, help="rel/abs START,STOP inclusive window")
    ap.add_argument(
        "--plot",
        action="store_true",
        help="save budget time-series PNGs under {run}/cell_dynamics/",
    )
    ap.add_argument(
        "--syn-strength",
        default=None,
        metavar="SRC,TAR,VALUE,...",
        help="override syn_strength; flat SRC,TAR,VALUE triples (comma-separated)",
    )
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
        _log(f"load_best {run_dir} ...")
        session, z, best_i, best_cost = plot_trained.load_best(run_dir)
        schema = list(session.schema)
        syn_strength_edits = _parse_syn_strength(args.syn_strength, session)
        z_t = torch.tensor(np.asarray(z, dtype=np.float64), dtype=torch.float64, device=session.device)
        z_t = _apply_syn_strength(z_t, schema, session, syn_strength_edits)
        p = fc.assign_params(z_t, schema, session.backend)

        spot_session_cache: dict[str, object] = {}
        bar_meta_cache: dict[str, tuple] = {}
        spot_by_cell: dict[str, dict[str, Any]] = {}
        bar_by_cell: dict[str, dict[str, Any]] = {}
        all_reports: list[dict[str, Any]] = []

        if not args.json:
            _log(f"== RUN {run_i}: {run_dir} ==")
            _log(
                f"best_i={best_i}  best_cost={best_cost:.6g}  "
                f"mode={'hex' if hex_mode else 'average'}"
            )

        for target in cli.targets:
            if target in fc.SPOT_TARGETS:
                if target not in spot_session_cache:
                    spot_session_cache[target] = plot_trained.session_for_target(
                        session, target,
                    )
                session_one = spot_session_cache[target]
                _log(f"budget walk {target} (spot; batched) ...")
                reports = analyze_spot_average(
                    session_one,
                    p=p,
                    cells=cli.cells,
                    target=target,
                    abs_start=rel_start,
                    abs_stop=rel_stop,
                )
                for cell, rep in reports.items():
                    spot_by_cell[cell] = rep
                    all_reports.append(rep)
                    _emit_report(
                        rep,
                        run_dir=run_dir,
                        do_print=not args.json,
                        do_plot=args.plot,
                    )
            else:
                hx = cli.x_list[0] if hex_mode else None
                hy = cli.y_list[0] if hex_mode else None
                if target not in bar_meta_cache:
                    bar_meta_cache[target] = _bar_meta(session, target)
                specs, grids = bar_meta_cache[target]
                cells_bar = [cli.cells[0]] if hex_mode else cli.cells
                specs_ordered = _bar_specs_requested(
                    session, target, cells_bar, cli.specs_req,
                    specs=specs, grids=grids,
                )
                multi_spec_plot = args.plot and len(specs_ordered) > 1
                if hex_mode:
                    _log(
                        f"budget walk {target} specs={specs_ordered} "
                        f"hex=({hx},{hy}) (batched, no full forward) ..."
                    )
                    reports_by_spec = analyze_bar_hex(
                        session,
                        p=p,
                        cell=cells_bar[0],
                        target=target,
                        spec_names=specs_ordered,
                        at_x=float(hx),
                        at_y=float(hy),
                        unit=args.unit,
                        rel_start=rel_start,
                        rel_stop=rel_stop,
                        specs=specs,
                        grids=grids,
                    )
                else:
                    _log(
                        f"budget walk {target} specs={specs_ordered} "
                        f"(batched, no full forward) ..."
                    )
                    reports_by_spec = analyze_bar_average(
                        session,
                        p=p,
                        cells=cells_bar,
                        target=target,
                        spec_names=specs_ordered,
                        rel_start=rel_start,
                        rel_stop=rel_stop,
                        specs=specs,
                        grids=grids,
                    )
                overlay_by_cell: dict[str, list[dict[str, Any]]] = {
                    c: [] for c in cells_bar
                }
                for spec in specs_ordered:
                    for c, rep in reports_by_spec[spec].items():
                        bar_by_cell[c] = rep
                        all_reports.append(rep)
                        if multi_spec_plot:
                            overlay_by_cell[c].append(rep)
                        _emit_report(
                            rep,
                            run_dir=run_dir,
                            do_print=not args.json,
                            do_plot=args.plot and not multi_spec_plot,
                        )
                if multi_spec_plot:
                    for c in cells_bar:
                        reps = overlay_by_cell[c]
                        out = os.path.join(
                            run_dir, "cell_dynamics", _overlay_plot_filename(reps),
                        )
                        plot_reports_overlay(reps, out)

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
