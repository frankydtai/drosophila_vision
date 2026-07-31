#!/usr/bin/env python
"""FiveCol V budget for cell responses (like vision ``analyze.cell_dynamics``).

Loads ``best_parameter.npy`` from a ``FiveCol_Parameter/`` run, walks the Borst
``update_Vm`` membrane equation for selected cell types, prints a budget report,
and optionally saves multi-panel PNGs under ``{run}/cell_dynamics/``.

Stimulus is the built-in photoreceptor current pulse (``signal``). Column filter
``--x`` matches ``cell_syn.py`` (``-2..2``; omit = all five columns).

FiveCol has a single Ih channel → ``g_Ih_off`` / ``i_h_off`` are identically 0.

Examples
--------
  cd SimulationCode
  ../vision/.venv/bin/python cell_dynamics.py --run with_Ih --cell L3 --plot
  ../vision/.venv/bin/python cell_dynamics.py --run with_Ih --cell L3,Mi1 --x=0 --plot
  ../vision/.venv/bin/python cell_dynamics.py --run with_Ih --cell L3 --rel 0,100 --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from FiveCol_MedSim_Pytorch import (  # noqa: E402
    E_Ih,
    E_exc,
    E_inh,
    E_leak,
    Ih_gain,
    M_exc,
    M_inh,
    capac,
    cdt,
    deltat,
    device,
    g_leak,
    nofcells,
    rectsyn,
    signal,
    trld,
)
from cell_syn import (  # noqa: E402
    _instance_ids,
    parse_x_list,
)
from plot_run import (  # noqa: E402
    DEFAULT_RUN,
    load_best_z,
    resolve_run_dir,
)
from run_local_cpu_test import assign_params  # noqa: E402

T_ONSET = 50
T_MAX = 200
TARGET_NAME = "pr_pulse"

_PLOT_PANELS: list[tuple[str, list[tuple[str, str]]]] = [
    ("v_post (mV)", [("v_post_mV", "v_post")]),
    (
        "conductance (nS)",
        [
            ("cdt_nS", "cdt"),
            ("g_exc_nS", "g_exc"),
            ("g_inh_nS", "g_inh"),
            ("g_leak_nS", "g_leak"),
            ("g_Ih_on_nS", "g_h"),
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
            ("num_ihon", "i_h"),
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
_PLOT_NCOLS = max(len(s) for _, s in _PLOT_PANELS)
_ROW_SHARED_YLIM = frozenset({2})
_BLACK_TRACE_LABELS = frozenset({"num", "den"})

# Colored formula tokens (label → subplot color); None = gray operator/constant.
_FORMULA_G_TOKENS: list[tuple[str, str | None]] = [
    ("v_post", "v_post"),
    (" = (", None),
    ("cdt", "cdt"), ("·v_pre + ", None),
    ("E_exc·", None), ("g_exc", "g_exc"), (" + ", None),
    ("E_inh·", None), ("g_inh", "g_inh"), (" + ", None),
    ("E_leak·", None), ("g_leak", "g_leak"), (" + ", None),
    ("E_Ih·", None), ("g_h", "g_h"), (" + ", None),
    ("E_Ih·", None), ("g_h_off", "g_h_off"), (" + ", None),
    ("i_sig", "i_sig"),
    (") / (", None),
    ("cdt", "cdt"), (" + ", None),
    ("g_exc", "g_exc"), (" + ", None),
    ("g_inh", "g_inh"), (" + ", None),
    ("g_h", "g_h"), (" + ", None),
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
    ("i_h", "i_h"), (" + ", None),
    ("i_h_off", "i_h_off"), (" + ", None),
    ("i_sig", "i_sig"),
    (") / (", None),
    ("cdt", "cdt"), (" + ", None),
    ("g_exc", "g_exc"), (" + ", None),
    ("g_inh", "g_inh"), (" + ", None),
    ("g_h", "g_h"), (" + ", None),
    ("g_h_off", "g_h_off"), (" + ", None),
    ("g_leak", "g_leak"),
    (")", None),
]


def _log(msg: str) -> None:
    print(msg, flush=True)


def _parse_comma_list(token: str) -> List[str]:
    return [p.strip() for p in str(token).split(",") if p.strip()]


def _plot_colors() -> List[str]:
    return list(plt.rcParams["axes.prop_cycle"].by_key()["color"])


def _plot_trace_colors(colors: List[str]) -> Dict[str, str]:
    """Map legend label → subplot color (column index within its row)."""
    out: Dict[str, str] = {}
    for _group, series in _PLOT_PANELS:
        for ci, (_key, label) in enumerate(series):
            if label in _BLACK_TRACE_LABELS:
                out[label] = "0.0"
            else:
                out[label] = colors[ci % len(colors)]
    return out


def _g_e_note(label: str, *, e_leak_mV: float) -> str | None:
    notes = {
        "g_exc": f"E_exc={E_exc:+g} mV",
        "g_inh": f"E_inh={E_inh:+g} mV",
        "g_leak": f"E_leak={e_leak_mV:+g} mV",
        "g_h": f"E_Ih={E_Ih:+g} mV",
        "g_h_off": f"E_Ih={E_Ih:+g} mV (unused)",
    }
    return notes.get(label)


def _add_budget_formula_row(
    fig,
    colors: List[str],
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
            x,
            y,
            text,
            transform=fig.transFigure,
            ha="left",
            va="top",
            fontsize=fontsize,
            color=color,
        )
        bbox = txt.get_window_extent(renderer=renderer)
        x = inv.transform((bbox.x1, bbox.y0))[0] + 0.003


def _finish_budget_figure_layout(fig, title: str, colors: List[str]) -> None:
    fig.suptitle(title, fontsize=11, y=0.995)
    _add_budget_formula_row(fig, colors, _FORMULA_G_TOKENS, y=0.975, fontsize=8)
    _add_budget_formula_row(fig, colors, _FORMULA_TOKENS, y=0.955, fontsize=9)
    fig.subplots_adjust(top=0.88, hspace=0.45, wspace=0.45)


def _ctype() -> np.ndarray:
    return np.asarray(np.load(HERE / "Circuits" / "ctype.npy", allow_pickle=True), dtype=str)


def _resolve_cells(tokens: Sequence[str], ctype: np.ndarray) -> List[str]:
    known = set(ctype)
    out: List[str] = []
    for tok in tokens:
        if tok not in known:
            raise SystemExit(f"unknown cell type {tok!r}")
        if tok not in out:
            out.append(tok)
    return out


def _unit_ids(cell: str, ctype: np.ndarray, xs: Optional[Sequence[int]]) -> np.ndarray:
    name_to_idx = {str(n): i for i, n in enumerate(ctype)}
    return np.asarray(
        _instance_ids(name_to_idx[cell], nofcells, xs=xs), dtype=np.int64
    )


def v_budget_from_g(
    v_pre: np.ndarray,
    g_exc: np.ndarray,
    g_inh: np.ndarray,
    g_Ih: np.ndarray,
    sig: np.ndarray,
    e_leak_u: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Numerator / denom terms matching FiveCol ``update_Vm`` (numpy)."""
    g_Ih_off = np.zeros_like(g_Ih)
    return {
        "num_exc": g_exc * E_exc,
        "num_inh": g_inh * E_inh,
        "num_leak": g_leak * e_leak_u,
        "num_ihon": g_Ih * E_Ih,
        "num_ihoff": g_Ih_off * E_Ih,
        "num_cdt": float(cdt) * v_pre,
        "num_sig": sig,
        "den": g_exc + g_inh + g_Ih + g_Ih_off + g_leak + float(cdt),
        "g_Ih_off": g_Ih_off,
    }


@torch.no_grad()
def _step_budget(
    vm: torch.Tensor,
    u: torch.Tensor,
    inp_gain: torch.Tensor,
    out_gain: torch.Tensor,
    ih_gmax: torch.Tensor,
    ih_midv: torch.Tensor,
    ih_slope: torch.Tensor,
    tau_midv: torch.Tensor,
    sig_t: torch.Tensor,
    units: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, np.ndarray]]:
    """One ``update_Vm`` step; return new state + per-unit budget arrays."""
    v_pre = vm
    ih_ss = 1.0 / (1.0 + torch.exp((ih_midv - vm) * ih_slope))
    tau = (
        1.5
        / (torch.exp(-0.1 * (vm - tau_midv)) + torch.exp(0.1 * (vm - tau_midv)))
        * 1000.0
        + 100.0
    )
    u = deltat / tau * (ih_ss - u) + u
    g_Ih = u * ih_gmax * Ih_gain
    drive = rectsyn(vm, trld) * out_gain
    g_exc = torch.mv(M_exc, drive) * inp_gain
    g_inh = torch.mv(M_inh, drive) * inp_gain
    num = (
        g_exc * E_exc
        + g_inh * E_inh
        + g_leak * E_leak
        + E_Ih * g_Ih
        + cdt * vm
        + sig_t
    )
    den = g_exc + g_inh + g_Ih + g_leak + cdt
    vm_post = num / den

    u_idx = units
    v_pre_u = v_pre[u_idx].detach().cpu().numpy()
    g_exc_u = g_exc[u_idx].detach().cpu().numpy()
    g_inh_u = g_inh[u_idx].detach().cpu().numpy()
    g_Ih_u = g_Ih[u_idx].detach().cpu().numpy()
    sig_u = sig_t[u_idx].detach().cpu().numpy()
    e_leak_u = E_leak[u_idx].detach().cpu().numpy()
    v_post_u = vm_post[u_idx].detach().cpu().numpy()
    terms = v_budget_from_g(v_pre_u, g_exc_u, g_inh_u, g_Ih_u, sig_u, e_leak_u)
    num_tot = (
        terms["num_exc"]
        + terms["num_inh"]
        + terms["num_leak"]
        + terms["num_ihon"]
        + terms["num_ihoff"]
        + terms["num_cdt"]
        + terms["num_sig"]
    )
    bud = {
        "v_pre": v_pre_u,
        "v_abs": v_post_u,
        "g_exc": g_exc_u,
        "g_inh": g_inh_u,
        "g_Ih_on": g_Ih_u,
        "g_Ih_off": terms["g_Ih_off"],
        "signal": sig_u,
        "num_exc": terms["num_exc"],
        "num_inh": terms["num_inh"],
        "num_leak": terms["num_leak"],
        "num_ihon": terms["num_ihon"],
        "num_ihoff": terms["num_ihoff"],
        "num_cdt": terms["num_cdt"],
        "num_sig": terms["num_sig"],
        "num": num_tot,
        "den": terms["den"],
    }
    return vm_post, u, bud


def _mean_step(bud: Dict[str, np.ndarray], v_onset: np.ndarray) -> Dict[str, float]:
    n = bud["v_abs"].shape[0]
    v_abs = float(np.mean(bud["v_abs"]))
    v_pre = float(np.mean(bud["v_pre"]))
    v_onset_m = float(np.mean(v_onset))
    num = float(np.mean(bud["num"]))
    den = float(np.mean(bud["den"]))
    return {
        "n_units": n,
        "v_post_mV": v_abs,
        "v_pre_d_mV": v_pre - v_onset_m,
        "v_post_minus_pre_mV": v_abs - v_pre,
        "signal": float(np.mean(bud["signal"])),
        "g_exc_nS": float(np.mean(bud["g_exc"])),
        "g_inh_nS": float(np.mean(bud["g_inh"])),
        "g_leak_nS": float(g_leak),
        "g_Ih_on_nS": float(np.mean(bud["g_Ih_on"])),
        "g_Ih_off_nS": 0.0,
        "cdt_nS": float(cdt),
        "num_exc": float(np.mean(bud["num_exc"])),
        "num_inh": float(np.mean(bud["num_inh"])),
        "num_leak": float(np.mean(bud["num_leak"])),
        "num_ihon": float(np.mean(bud["num_ihon"])),
        "num_ihoff": 0.0,
        "num_cdt": float(np.mean(bud["num_cdt"])),
        "num_sig": float(np.mean(bud["num_sig"])),
        "num": num,
        "den": den,
        "num_over_den": num / den if den else 0.0,
    }


def _dominant_drive(step: Dict[str, float]) -> str:
    keys = [
        ("i_exc", step["num_exc"]),
        ("i_inh", step["num_inh"]),
        ("i_leak", step["num_leak"]),
        ("i_h", step["num_ihon"]),
        ("i_cdt", step["num_cdt"]),
        ("i_sig", step["num_sig"]),
    ]
    name, _ = max(keys, key=lambda kv: abs(kv[1]))
    return name


@torch.no_grad()
def walk_budget(
    z: torch.Tensor,
    units: np.ndarray,
    *,
    rel_start: Optional[int] = None,
    rel_stop: Optional[int] = None,
) -> List[Dict[str, float]]:
    """Equilibrate to ``T_ONSET``, then record mean budget for ``units``."""
    inp_gain, out_gain, ih_gmax, ih_midv, ih_slope, tau_midv = assign_params(z)
    u = torch.zeros(325, dtype=torch.float64, device=device)
    vm = E_leak.clone()
    units_t = torch.as_tensor(units, dtype=torch.long, device=device)

    for t in range(1, T_ONSET):
        vm, u, _ = _step_budget(
            vm, u, inp_gain, out_gain, ih_gmax, ih_midv, ih_slope, tau_midv,
            signal[t - 1], units_t,
        )

    v_onset = vm[units_t].detach().cpu().numpy().copy()
    steps: List[Dict[str, float]] = []

    for t in range(T_ONSET, T_MAX):
        rel = t - T_ONSET
        if rel_start is not None and rel < rel_start:
            vm, u, _ = _step_budget(
                vm, u, inp_gain, out_gain, ih_gmax, ih_midv, ih_slope, tau_midv,
                signal[t - 1], units_t,
            )
            continue
        if rel_stop is not None and rel > rel_stop:
            break
        vm, u, bud = _step_budget(
            vm, u, inp_gain, out_gain, ih_gmax, ih_midv, ih_slope, tau_midv,
            signal[t - 1], units_t,
        )
        step = _mean_step(bud, v_onset)
        step["t"] = t
        step["rel"] = rel
        steps.append(step)
    return steps


def _unit_params(z_np: np.ndarray, cell: str, ctype: np.ndarray) -> Dict[str, float]:
    ti = int(np.where(ctype == cell)[0][0])
    ih = 0.0
    if 8 <= ti <= 12:
        ih = float(z_np[130 + (ti - 8)])
    return {
        "in_gain": float(z_np[ti]),
        "out_gain": float(z_np[65 + ti]),
        "Ih_gmax": ih,
        "Ih_midv": float(z_np[135]),
        "Ih_slope": float(z_np[136]),
        "tau_midv": float(z_np[137]),
        "e_leak_mV": float(E_leak[ti].item() if ti < E_leak.numel() else -50.0),
        "cdt_nS": float(cdt),
        "g_leak_nS": float(g_leak),
        "capac_pF": float(capac),
        "delta_ms": float(deltat),
    }


def analyze_cell(
    z: torch.Tensor,
    z_np: np.ndarray,
    cell: str,
    ctype: np.ndarray,
    xs: Optional[Sequence[int]],
    *,
    rel_start: Optional[int] = None,
    rel_stop: Optional[int] = None,
) -> Dict[str, Any]:
    units = _unit_ids(cell, ctype, xs)
    steps = walk_budget(z, units, rel_start=rel_start, rel_stop=rel_stop)
    if not steps:
        raise SystemExit(f"no steps for {cell} (check --rel)")

    for s in steps:
        s["v_post_d_mV"] = s["v_pre_d_mV"] + s["v_post_minus_pre_mV"]

    peak_i = int(np.argmax(np.abs([s["v_post_d_mV"] for s in steps])))
    peak_step = steps[peak_i]
    peak_val = float(peak_step["v_post_d_mV"])
    polarity = "up" if peak_val > 1e-3 else ("down" if peak_val < -1e-3 else "flat")

    mode = "column" if xs is not None else "all_columns"
    return {
        "cell": cell,
        "target": TARGET_NAME,
        "spec": None,
        "mode": mode,
        "n_units": int(units.size),
        "xs": list(xs) if xs is not None else list(range(-2, 3)),
        "params": _unit_params(z_np, cell, ctype),
        "steps": steps,
        "v_post_d_peak_mV": peak_val,
        "v_post_d_polarity": polarity,
        "v_post_d_peak_rel": int(peak_step["rel"]),
        "before_t": T_ONSET,
        "peak_drive": _dominant_drive(peak_step),
        "peak_step": peak_step,
    }


def _print_report(report: Dict[str, Any]) -> None:
    print(
        f"cell={report['cell']} mode={report['mode']} n_units={report['n_units']} "
        f"x={report['xs']}"
    )
    print(f"target={report['target']}")
    print(
        f"v_post_d_peak={report['v_post_d_peak_mV']:+.4f} mV "
        f"v_post_d_polarity={report['v_post_d_polarity']}  "
        f"v_post_d_peak_rel={report['v_post_d_peak_rel']}  "
        f"before_t={report.get('before_t')}  "
        f"peak_drive={report.get('peak_drive')}"
    )
    print("trained:", "  ".join(f"{k}={v:.6g}" for k, v in report["params"].items()))
    print(
        "\nrel  n  v_post  v_pre_d  v_post_minus_pre  sig   g_inh  g_Ih  g_exc  "
        "num_inh  num_exc"
    )
    for s in report["steps"]:
        print(
            f"{s['rel']:4d} {s.get('n_units', 1):3d} {s['v_post_mV']:+8.4f} "
            f"{s['v_pre_d_mV']:+8.4f} {s['v_post_minus_pre_mV']:+8.4f} "
            f"{s['signal']:5.1f} {s['g_inh_nS']:.4f} {s['g_Ih_on_nS']:.4f} "
            f"{s['g_exc_nS']:.4f} {s['num_inh']:+8.2f} {s['num_exc']:+8.2f}"
        )
    ps = report.get("peak_step")
    if ps is not None:
        num = ps["num"]
        print(f"\nNumerator at peak rel={ps['rel']} (num={num:.2f}):")
        for name, val in [
            ("i_cdt", ps["num_cdt"]),
            ("i_exc", ps["num_exc"]),
            ("i_inh", ps["num_inh"]),
            ("i_leak", ps["num_leak"]),
            ("i_h", ps["num_ihon"]),
            ("i_h_off", ps["num_ihoff"]),
            ("i_sig", ps["num_sig"]),
        ]:
            pct = 100.0 * val / num if num else 0.0
            print(f"  {name:8s} {val:+9.2f} ({pct:.0f}%)")


def _plot_filename(report: Dict[str, Any]) -> str:
    xs = report.get("xs") or []
    xnote = "x" + "_".join(str(v) for v in xs) if xs else "allx"
    return f"{report['cell']}_{report['target']}_{xnote}.png"


def plot_report(report: Dict[str, Any], out_path: Path) -> None:
    title = (
        f"{report['cell']}  {report['target']}  mode={report['mode']}  "
        f"n={report['n_units']}  x={report['xs']}"
    )
    fig, axes = plt.subplots(
        len(_PLOT_PANELS),
        _PLOT_NCOLS,
        figsize=(2.6 * _PLOT_NCOLS, 2.4 * len(_PLOT_PANELS)),
        squeeze=False,
        sharex=True,
    )
    colors = _plot_colors()
    e_leak_mV = float(report.get("params", {}).get("e_leak_mV", -50.0))
    rel = np.asarray([s["rel"] for s in report["steps"]], dtype=float)
    row_curves: Dict[int, List[np.ndarray]] = {ri: [] for ri in _ROW_SHARED_YLIM}

    for ri, (_group_ylabel, series) in enumerate(_PLOT_PANELS):
        for ci in range(_PLOT_NCOLS):
            ax = axes[ri, ci]
            if ci >= len(series):
                ax.set_visible(False)
                continue
            key, label = series[ci]
            y = np.asarray([s[key] for s in report["steps"]], dtype=float)
            color = (
                "0.0" if label in _BLACK_TRACE_LABELS else colors[ci % len(colors)]
            )
            ax.plot(rel, y, color=color, linewidth=1.4, label=label)
            ax.set_ylabel(label, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, axis="y", alpha=0.3)
            e_note = _g_e_note(label, e_leak_mV=e_leak_mV)
            if e_note is not None:
                ax.set_title(e_note, fontsize=8)
            if ri in _ROW_SHARED_YLIM:
                row_curves[ri].append(y)
            if ri == len(_PLOT_PANELS) - 1:
                ax.set_xlabel("rel (t - t_onset)", fontsize=8)

    for ri in _ROW_SHARED_YLIM:
        curves = row_curves.get(ri) or []
        if not curves:
            continue
        ymin = min(float(np.min(c)) for c in curves)
        ymax = max(float(np.max(c)) for c in curves)
        pad = 0.05 * (ymax - ymin + 1e-9)
        for ci in range(len(_PLOT_PANELS[ri][1])):
            axes[ri, ci].set_ylim(ymin - pad, ymax + pad)

    _finish_budget_figure_layout(fig, title, colors)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--run",
        default=DEFAULT_RUN,
        help=f"run folder under FiveCol_Parameter/ (default: {DEFAULT_RUN})",
    )
    ap.add_argument(
        "--cell",
        required=True,
        metavar="CELL[,CELL...]",
        help="comma-separated cell types (e.g. L3,Mi1)",
    )
    ap.add_argument(
        "--x",
        default=None,
        metavar="X[,X...]",
        help="column x in -2..2 (comma-separated). Default: all five columns",
    )
    ap.add_argument(
        "--rel",
        default=None,
        metavar="START,STOP",
        help="inclusive rel window after t_onset=50 (e.g. 0,100)",
    )
    ap.add_argument(
        "--plot",
        action="store_true",
        help="save budget PNGs under {run}/cell_dynamics/",
    )
    ap.add_argument("--json", action="store_true", help="print JSON reports to stdout")
    args = ap.parse_args(argv)

    run_dir = resolve_run_dir(args.run)
    z_np = load_best_z(run_dir)
    z = torch.tensor(z_np, dtype=torch.float64, device=device)
    ctype = _ctype()
    cells = _resolve_cells(_parse_comma_list(args.cell), ctype)

    xs: Optional[List[int]] = None
    if args.x is not None:
        try:
            xs = parse_x_list(args.x)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    rel_start = rel_stop = None
    if args.rel is not None:
        parts = _parse_comma_list(args.rel)
        if len(parts) != 2:
            raise SystemExit("--rel must be START,STOP")
        rel_start, rel_stop = int(parts[0]), int(parts[1])

    _log(f"run={run_dir}")
    _log(f"device={device}  cells={cells}  x={xs if xs is not None else 'all'}")

    reports: List[Dict[str, Any]] = []
    for cell in cells:
        rep = analyze_cell(
            z, z_np, cell, ctype, xs, rel_start=rel_start, rel_stop=rel_stop
        )
        reports.append(rep)
        if args.json:
            # peak_step is nested in steps; drop non-serializable if any
            print(json.dumps(rep, default=float))
        else:
            print("")
            _print_report(rep)
        if args.plot:
            out = run_dir / "cell_dynamics" / _plot_filename(rep)
            plot_report(rep, out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
