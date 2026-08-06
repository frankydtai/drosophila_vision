"""Borst / hp_lp v component analysis."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

import import_bootstrap  # noqa: F401
import training
from param_defaults import DEFAULT_RUN_PATH
import figure.plot_run as plot_trained
from figure.readout import contrast_for_task
from figure.spot import pack_spot_cost_radii, resolve_spot_gt_cubes
from figure.util import plot_sem_band
from import_bootstrap import parse_bool, parse_comma_list
from network.construction import col2gt
from task.moving_bar.gt import (
    bar_specs_for_session,
    filter_requested_specs,
    moving_bar_nodes_on_hexes,
    moving_bar_row_specs,
    moving_bar_session_t0_grids,
)
from task.moving_bar.input import (
    filter_sti_hexes,
    moving_bar_cost_hexes,
)
from task.spot.gt import build_spot_center_readout
from task.spot.input import (
    spot_from_opts,
    spot_stimulus_batches,
)
from training.implement import parse_tasks, stimulus_timing_kwargs_from_args

__doc__ = """Borst / hp_lp v component analysis.

Consumers (CLI or ``import analyze.cell_dynamics``) must reuse this module's
forward helpers. Do not re-implement spot/bar readout + step loops in test/.

Time axis (read this before ``--ms-shown`` / ``TimeWindow``)
------------------------------------------------------------
Two *different* knobs — do not mix them:

1. **Stimulus length** (``--ms-pre`` / ``--ms-pulse`` / ``--ms-response`` /
   ``--ms-post`` / ``--delta-ms``): rebuilds the session stimulus (via
   ``figure.plot_run.maybe_override_stimulus_timing``). Unset = keep the run's
   train opts. These change *how long* pre/pulse/response *are*, not which
   slice of an existing trace you plot.

2. **Analyze / plot window** (``--ms-shown START,STOP`` or ``--t-rel START:STOP``,
   mutually exclusive): which inclusive slice of the forward to accumulate and
   report. Default if both omitted: absolute ms ``0`` .. last sample.

``--ms-shown`` is **absolute aligned ms**, never "ms before onset":

* **spot**: aligned ``t = 0`` is trial start. Stimulus onset is at
  ``t_onset = ms_to_t(ms_pre)`` (e.g. ``ms_pre=1000``, ``delta_ms=5`` →
  ``t_onset=200`` ↔ **1000 ms**). Pre-stimulus is therefore
  ``--ms-shown 0,1000`` (or ``0,ms_pre``), **not** ``-1000,0``.
  Negative START is wrong for spot (aligned index goes negative; accum window
  collapses).
* **moving_bar**: aligned ``t = 0`` is bar ``t0`` at the node (crossing), so
  ``--ms-shown`` is ms relative to that ``t0`` (negative START is valid).

``--t-rel START:STOP`` is **t-index offsets from the |v_post_d| peak** (not from
onset, not absolute ms). Example: ``--t-rel -5:15``.

``TimeWindow(kind="ms", start, stop)`` uses the same absolute-aligned-ms rule as
``--ms-shown``. ``kind="t_rel"`` matches ``--t-rel``. Spot R0-average API:
``analyze_spot_average(..., time_window=TimeWindow("ms", 0, ms_pre), radius=0)``.

Programmatic reuse
------------------
* Spot R0 / R1 average: ``analyze_spot_average`` (omit hex; ``radius=0|1``).
* Spot one hex: ``analyze_spot_hex``.
* Bar average / hex: ``analyze_bar_average`` / ``analyze_bar_hex``.
* Load run: ``figure.plot_run.load_best`` + ``assign_params``; do not invent a
  second forward path.

CLI
---
``CELL,...`` ``--run`` ``--task`` ``--spec`` ``--x`` ``--y``. Pass comma lists in
one process (do not re-invoke once per cell/spec).

Per ``--run``: one ``load_best``; one batched v component forward per distinct task.

* Omit ``--x`` / ``--y``: cost-extent **average** (optional ``--radius 0|1``).
* Exactly one ``--x`` and one ``--y``: **hex** (spot or moving_bar; one cell).
  Incompatible with ``--radius`` (hex is stim-on only).
* Multiple x/y: rejected.

``--plot true|false``: PNGs under ``{run}/cell_dynamics/`` (default true);
  per-t step table prints only when ``--plot false``.
``--radius 0|1``: spot average Euclidean readout radius (default 0 = stim-on hex; 1 = neighbors).
  Average only; PNGs for ``--radius 1`` get ``_radius1`` in the filename.
``--param NAME=VALUE`` / ``NAME.NODE=VALUE``: via ``figure.plot_run`` — overwrite
  any schema param before forward (``NODE`` = cell, ``SRC:TAR`` pair, or ``eN``;
  omit / ``all`` = every node). Each edit appends ``_NAME_NODE_VALUE`` (``:`` in
  NODE → ``_``; no NODE when omitted / ``all``) to PNG stems, in CLI order,
  after timing suffixes.

``--euler im|ex``: optional membrane Euler override (default: keep run
``train_opts.euler``). Re-opens the session with timing overrides when set;
PNG stem gets ``_im`` / ``_ex``. Component formula rows follow
implicit vs explicit.

Examples
--------
  # Spot R0 average, pre window only (ms_pre=1000 on the run → 0..1000 ms):
  ../.venv/bin/python analyze/cell_dynamics.py \\
    L1,L2,Mi1 --run hp_lp/28693664-... \\
    --task spot_bright --radius 0 --ms-shown 0,1000 --plot false

  ../.venv/bin/python analyze/cell_dynamics.py \\
    Mi4,Mi9 --run /abs/path/to/run \\
    --task spot_bright,moving_bar_bright --spec right_bright_w1

  ../.venv/bin/python analyze/cell_dynamics.py \\
    L3 --run /abs/path/to/run --task spot_bright --x 1 --y 0

  ../.venv/bin/python analyze/cell_dynamics.py \\
    L3 --run /abs/path/to/run --task moving_bar_bright \\
    --spec right_bright_w1 --x -2 --y -1

  ../.venv/bin/python analyze/cell_dynamics.py \\
    T4a --run borst/27252028-... --task moving_bar_bright \\
    --spec left_bright_w4,right_bright_w4 \\
    --param syn_strength_cell.Mi4:T4a=2.0 syn_strength_cell.Mi9:T4a=1.0 \\
    --t-rel -5:15

  ../.venv/bin/python analyze/cell_dynamics.py \\
    L3 --run /abs/path/to/run --task spot_bright --param tau_hp.L3=500
"""



def _parse_t_range(token: str, *, flag: str) -> tuple[int, int]:
    """Parse ``START:STOP`` in t indices (colon; one token)."""
    parts = str(token).split(":")
    if len(parts) != 2 or parts[0] == "" or parts[1] == "":
        raise SystemExit(f"{flag} must be START:STOP")
    start, stop = int(parts[0]), int(parts[1])
    if start > stop:
        raise SystemExit(f"{flag} START={start} > STOP={stop}")
    return start, stop


@dataclass(frozen=True)
class TimeWindow:
    """Inclusive analyze window for component forward / finalize.

    ``kind="ms"``
        Absolute **aligned** ms (same as ``--ms-shown``). Spot: ``0`` = trial
        start, onset ≈ ``ms_pre``; pre = ``TimeWindow("ms", 0, ms_pre)``.
        Never use negative ``start`` for spot "pre" (that is not onset-relative).
        Bar: ``0`` = bar ``t0`` at the node (negative ``start`` OK).
    ``kind="t_rel"``
        Integer t offsets from the |v_post_d| peak (same as ``--t-rel``).

    Not stimulus-length overrides (``--ms-pre`` / …); those rebuild the session.
    """

    kind: str  # "t_rel" | "ms"
    start: float
    stop: float

    def __post_init__(self) -> None:
        if self.kind not in ("t_rel", "ms"):
            raise ValueError(f"TimeWindow.kind must be t_rel|ms; got {self.kind!r}")

    def forward_t_start(self, *, delta_ms: float) -> int | None:
        if self.kind != "ms":
            return None
        return training.ms_to_t(self.start, delta_ms=delta_ms)

    def forward_t_stop(self, *, delta_ms: float) -> int | None:
        if self.kind != "ms":
            return None
        return training.ms_to_t(self.stop, delta_ms=delta_ms)

    def aligned_win_len(self, T: int, *, delta_ms: float) -> int:
        """Buffer length for spot (``t0=0``): indices ``0 .. stop`` inclusive.

        For ``kind="ms"``, length is ``ms_to_t(stop) + 1``. ``stop`` must be
        non-negative for a useful buffer (spot pre → ``stop=ms_pre``).
        """
        if self.kind != "ms":
            return T
        return training.ms_to_t(self.stop, delta_ms=delta_ms) + 1


@dataclass(frozen=True)
class SharedCli:
    """Parsed shared CLI."""

    cells: list[str]
    tasks: list[str]
    specs_req: list[str] | None
    xs: list | None
    ys: list | None


def add_shared_cli(
    ap: argparse.ArgumentParser,
    *,
    default_run: str | None = None,
) -> None:
    """Register positional cell + ``--run/--task/--spec/--x/--y``."""
    ap.add_argument(
        "cell",
        metavar="CELL,...",
        help="comma-separated cells, e.g. Mi4,Mi9",
    )
    run_kw: dict = {
        "action": "append",
        "default": None,
        "help": (
            "run directory (absolute, or relative to PARAMETER_DIR via "
            "plot_trained.resolve_run_dir)"
            + (f"; default: {default_run}" if default_run else "")
        ),
    }
    if default_run is None:
        run_kw["required"] = True
    ap.add_argument("--run", **run_kw)
    ap.add_argument(
        "--task",
        default="spot_bright",
        metavar="TASK,...",
        help="comma-separated tasks (spot_* / moving_bar_* or TASK_ALIASES)",
    )
    ap.add_argument(
        "--spec",
        default=None,
        metavar="SPEC,...",
        help="comma-separated moving_bar stimulus specs; omit = all available",
    )
    ap.add_argument(
        "--x",
        default=None,
        metavar="X,...",
        help="optional comma-separated x; component hex needs exactly one with --y",
    )
    ap.add_argument(
        "--y",
        default=None,
        metavar="Y,...",
        help="optional comma-separated y; component hex needs exactly one with --x",
    )


def parse_shared_cli(args: argparse.Namespace) -> SharedCli:
    cells = parse_comma_list(args.cell)
    if not cells:
        raise SystemExit("cell is required")
    tasks = parse_tasks(args.task)
    if not tasks:
        raise SystemExit("--task is required")
    specs_req = parse_comma_list(args.spec) if args.spec is not None else None
    for t in tasks:
        if t not in training.SPOT_TASKS and t not in training.MOVING_BAR_TASKS:
            raise SystemExit(
                f"unsupported task {t!r}; expected spot_* or moving_bar_* "
                f"(after TASK_ALIASES expansion)"
            )
    xs = plot_trained.parse_axis_slices(args.x)
    ys = plot_trained.parse_axis_slices(args.y)
    return SharedCli(
        cells=cells,
        tasks=tasks,
        specs_req=specs_req,
        xs=xs,
        ys=ys,
    )


# Component-step fields plotted vs time (key, ylabel/legend).
_BORST_PLOT_PANELS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "v_post (mV)",
        [
            ("v_post", "v_post"),
        ],
    ),
    (
        "conductance (nS)",
        [
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
            ("num_v", "num_v"),
            ("num_exc", "i_exc"),
            ("num_inh", "i_inh"),
            ("num_leak", "i_leak"),
            ("num_ihon", "i_h_on"),
            ("num_ihoff", "i_h_off"),
            ("i_sti", "i_sti"),
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

_HP_LP_PLOT_PANELS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "v_post (mV)",
        [
            ("v_post", "v_post"),
        ],
    ),
    (
        "v_in (mV)",
        [
            ("v_in", "v_in"),
            ("v_in_exc", "v_in_exc"),
            ("v_in_inh", "-v_in_inh"),
            ("v_sti", "v_sti"),
        ],
    ),
    (
        "HP state",
        [
            ("v_tot", "v_tot"),
            ("v_slow", "v_slow"),
            ("v_hp", "v_hp"),
        ],
    ),
    (
        "dv (mV)",
        [
            ("dv_leak", "dv_leak"),
            ("dv_hp", "dv_hp"),
            ("dv_leak_plus_hp", "dv_leak+dv_hp"),
        ],
    ),
]

_BORST_COMPONENT_KEYS = (
    "v_pre_d", "v_abs", "i_sti", "g_exc", "g_inh", "g_Ih_on", "g_Ih_off",
    "num_exc", "num_inh", "num_leak", "num_ihon", "num_ihoff", "num_v",
    "num", "den",
)
_HP_LP_COMPONENT_KEYS = (
    "v_pre_d", "v_abs", "i_sti", "v_sti", "v_in", "v_in_exc", "v_in_inh",
    "v_slow", "v_tot", "v_hp", "dv_leak", "dv_hp",
)

_BORST_PLOT_KEY_COMPONENT: dict[str, str | None] = {
    "v_post": "v_abs",
    "g_exc_nS": "g_exc",
    "g_inh_nS": "g_inh",
    "g_leak_nS": None,
    "g_Ih_on_nS": "g_Ih_on",
    "g_Ih_off_nS": "g_Ih_off",
    "num_v": "num_v",
    "num_exc": "num_exc",
    "num_inh": "num_inh",
    "num_leak": "num_leak",
    "num_ihon": "num_ihon",
    "num_ihoff": "num_ihoff",
    "i_sti": "i_sti",
    "num_over_den": "v_abs",
    "num": "num",
    "den": "den",
}
_HP_LP_PLOT_KEY_COMPONENT: dict[str, str | None] = {
    "v_post": "v_abs",
    "v_in": "v_in",
    "v_in_exc": "v_in_exc",
    "v_in_inh": "v_in_inh",
    "v_sti": "v_sti",
    "v_slow": "v_slow",
    "v_tot": "v_tot",
    "v_hp": "v_hp",
    "dv_leak": "dv_leak",
    "dv_hp": "dv_hp",
    "dv_leak_plus_hp": None,
}

_BORST_FORMULA_G_IMPLICIT: list[tuple[str, str | None]] = [
    ("v_post", "v_post"),
    (" = (v_pre + ", None),
    ("dt_over_c", None), ("·(i_sti + ", None),
    ("E_exc·", None), ("g_exc", "g_exc"), (" + ", None),
    ("E_inh·", None), ("g_inh", "g_inh"), (" + ", None),
    ("E_leak·", None), ("g_leak", "g_leak"), (" + ", None),
    ("E_h_on·", None), ("g_h_on", "g_h_on"), (" + ", None),
    ("E_h_off·", None), ("g_h_off", "g_h_off"),
    (")) / (1 + ", None),
    ("dt_over_c", None), ("·(", None),
    ("g_exc", "g_exc"), (" + ", None),
    ("g_inh", "g_inh"), (" + ", None),
    ("g_h_on", "g_h_on"), (" + ", None),
    ("g_h_off", "g_h_off"), (" + ", None),
    ("g_leak", "g_leak"),
    ("))", None),
]

_BORST_FORMULA_I_IMPLICIT: list[tuple[str, str | None]] = [
    ("v_post", "v_post"),
    (" = (", None),
    ("num_v", "num_v"), (" + ", None),
    ("dt_over_c", None), ("·(", None),
    ("i_sti", "i_sti"), (" + ", None),
    ("i_exc", "i_exc"), (" + ", None),
    ("i_inh", "i_inh"), (" + ", None),
    ("i_leak", "i_leak"), (" + ", None),
    ("i_h_on", "i_h_on"), (" + ", None),
    ("i_h_off", "i_h_off"),
    (")) / (", None),
    ("den", "den"),
    (")", None),
]

_BORST_FORMULA_G_EXPLICIT: list[tuple[str, str | None]] = [
    ("v_post", "v_post"),
    (" = v_pre + ", None),
    ("dt_over_c", None), ("·(i_sti + ", None),
    ("E_exc·", None), ("g_exc", "g_exc"), (" + ", None),
    ("E_inh·", None), ("g_inh", "g_inh"), (" + ", None),
    ("E_leak·", None), ("g_leak", "g_leak"), (" + ", None),
    ("E_h_on·", None), ("g_h_on", "g_h_on"), (" + ", None),
    ("E_h_off·", None), ("g_h_off", "g_h_off"),
    (" − (", None),
    ("g_exc", "g_exc"), (" + ", None),
    ("g_inh", "g_inh"), (" + ", None),
    ("g_h_on", "g_h_on"), (" + ", None),
    ("g_h_off", "g_h_off"), (" + ", None),
    ("g_leak", "g_leak"),
    (")·v_pre)", None),
]

_BORST_FORMULA_I_EXPLICIT: list[tuple[str, str | None]] = [
    ("v_post", "v_post"),
    (" = ", None),
    ("num_v", "num_v"), (" + ", None),
    ("dt_over_c", None), ("·(", None),
    ("i_sti", "i_sti"), (" + ", None),
    ("i_exc", "i_exc"), (" + ", None),
    ("i_inh", "i_inh"), (" + ", None),
    ("i_leak", "i_leak"), (" + ", None),
    ("i_h_on", "i_h_on"), (" + ", None),
    ("i_h_off", "i_h_off"),
    (")", None),
]

_HP_LP_FORMULA_G_IMPLICIT: list[tuple[str, str | None]] = [
    ("v_post", "v_post"),
    (" = (v_pre + (dt/τ_lp)·(v_rest + G·(v_rest + ", None),
    ("v_in", "v_in"), (" + ", None),
    ("i_sti/g_in", "v_sti"), (" − ", None),
    ("v_slow", "v_slow"), ("))) / (1 + dt/τ_lp)", None),
]

_HP_LP_FORMULA_I_IMPLICIT: list[tuple[str, str | None]] = [
    ("v_post", "v_post"), (" = v_pre + ", None),
    ("dv_leak", "dv_leak"), (" + ", None),
    ("dv_hp", "dv_hp"),
]

_HP_LP_FORMULA_G_EXPLICIT: list[tuple[str, str | None]] = [
    ("v_post", "v_post"),
    (" = v_pre + (dt/τ_lp)[-(v_pre−v_rest) + G·(v_rest + ", None),
    ("v_in", "v_in"), (" + ", None),
    ("i_sti/g_in", "v_sti"), (" − ", None),
    ("v_slow", "v_slow"), (")]", None),
]

_HP_LP_FORMULA_I_EXPLICIT: list[tuple[str, str | None]] = [
    ("v_post", "v_post"), (" = v_pre + ", None),
    ("dv_leak", "dv_leak"), (" + ", None),
    ("dv_hp", "dv_hp"),
]

_BLACK_TRACE_LABELS = frozenset({"num", "den", "v_tot", "v_slow", "v_hp", "dv_hp"})
# Label → reuse another label's plot color (same row cycle).
_TRACE_COLOR_MATCH: dict[str, str] = {
    "dv_leak+dv_hp": "dv_leak",
}
# Shared ylim across columns: borst current row; hp_lp HP-state + dv rows.
_BORST_ROW_SHARED_YLIM = frozenset({2})
_HP_LP_ROW_SHARED_YLIM = frozenset({2, 3})


@dataclass(frozen=True)
class _ComponentSpec:
    model: str
    keys: tuple[str, ...]
    plot_panels: list[tuple[str, list[tuple[str, str]]]]
    plot_key_component: dict[str, str | None]
    formula_g: list[tuple[str, str | None]]
    formula_i: list[tuple[str, str | None]]
    row_shared_ylim: frozenset[int]

    @property
    def n_keys(self) -> int:
        return len(self.keys)

    @property
    def i_v_abs(self) -> int:
        return self.keys.index("v_abs")

    @property
    def i_v_pre_d(self) -> int:
        return self.keys.index("v_pre_d")

    @property
    def n_plot_cols(self) -> int:
        return max(len(series) for _, series in self.plot_panels)


def _component_spec(model: str, euler: str) -> _ComponentSpec:
    euler = training.expand_euler(euler)
    if model == "borst":
        if euler == "implicit":
            formula_g, formula_i = _BORST_FORMULA_G_IMPLICIT, _BORST_FORMULA_I_IMPLICIT
        else:
            formula_g, formula_i = _BORST_FORMULA_G_EXPLICIT, _BORST_FORMULA_I_EXPLICIT
        return _ComponentSpec(
            model="borst",
            keys=_BORST_COMPONENT_KEYS,
            plot_panels=_BORST_PLOT_PANELS,
            plot_key_component=_BORST_PLOT_KEY_COMPONENT,
            formula_g=formula_g,
            formula_i=formula_i,
            row_shared_ylim=_BORST_ROW_SHARED_YLIM,
        )
    if model == "hp_lp":
        if euler == "implicit":
            formula_g, formula_i = _HP_LP_FORMULA_G_IMPLICIT, _HP_LP_FORMULA_I_IMPLICIT
        else:
            formula_g, formula_i = _HP_LP_FORMULA_G_EXPLICIT, _HP_LP_FORMULA_I_EXPLICIT
        return _ComponentSpec(
            model="hp_lp",
            keys=_HP_LP_COMPONENT_KEYS,
            plot_panels=_HP_LP_PLOT_PANELS,
            plot_key_component=_HP_LP_PLOT_KEY_COMPONENT,
            formula_g=formula_g,
            formula_i=formula_i,
            row_shared_ylim=_HP_LP_ROW_SHARED_YLIM,
        )
    raise SystemExit(f"cell_dynamics supports borst|hp_lp; got {model!r}")


def _trace_ylabel(panel_ylabel: str, label: str) -> str:
    if label == "num/den":
        return "num/den (mV)"
    if label == "num":
        return "num (pA)"
    if label == "den":
        return "den (nS)"
    if "(" in panel_ylabel:
        node = " " + panel_ylabel[panel_ylabel.index("("):]
        return f"{label}{node}"
    return label


def _component_axes_grid(spec: _ComponentSpec):
    """Return ``(n_rows, n_hexes)`` grid: one row per plot panel."""
    return len(spec.plot_panels), spec.n_plot_cols


def _plot_colors():
    import matplotlib.pyplot as plt

    return plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _plot_trace_colors(colors: list[str], spec: _ComponentSpec) -> dict[str, str]:
    """Map trace legend label → subplot color (hex index within its row)."""
    out: dict[str, str] = {}
    for _panel_ylabel, series in spec.plot_panels:
        for ci, (_key, label) in enumerate(series):
            if label in _BLACK_TRACE_LABELS:
                out[label] = "0.0"
            else:
                out[label] = colors[ci % len(colors)]
    for label, src in _TRACE_COLOR_MATCH.items():
        if label in out and src in out:
            out[label] = out[src]
    return out


def _g_e_note(label: str, *, e_leak_mV: float, globs: dict[str, Any]) -> str | None:
    """Reversal annotation for a conductance subplot (``E_exc=+10 mV`` …)."""
    if label == "g_leak":
        return f"E_leak={e_leak_mV:+g} mV"
    notes = {
        "g_exc": "E_exc",
        "g_inh": "E_inh",
        "g_h_on": "E_Ih",
        "g_h_off": "E_IH_OFF",
    }
    gkey = notes.get(label)
    if gkey is None or gkey not in globs:
        return None
    pretty = {"E_Ih": "E_h_on", "E_IH_OFF": "E_h_off"}.get(gkey, gkey)
    return f"{pretty}={globs[gkey]:+g} mV"


def _add_component_formula_row(
    fig,
    colors: list[str],
    tokens: list[tuple[str, str | None]],
    spec: _ComponentSpec,
    *,
    y: float,
    fontsize: int = 9,
) -> None:
    """One formula line; trace symbols use plot colors, operators/constants gray."""
    tc = _plot_trace_colors(colors, spec)
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


def _finish_component_figure(
    fig, title: str, colors: list[str], spec: _ComponentSpec,
) -> None:
    fig.suptitle(title, fontsize=11, y=0.995)
    _add_component_formula_row(fig, colors, spec.formula_g, spec, y=0.975, fontsize=8)
    _add_component_formula_row(fig, colors, spec.formula_i, spec, y=0.955, fontsize=9)
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.88))


def _model_driver(session):
    from neuron.forward import MODEL_DRIVERS

    try:
        return MODEL_DRIVERS[session.model]
    except KeyError as exc:
        raise SystemExit(
            f"cell_dynamics supports {tuple(MODEL_DRIVERS)}; got {session.model!r}"
        ) from exc


def _prepare_drive(session, p, i_sti: torch.Tensor) -> torch.Tensor:
    """Model ``prepare_i_sti`` + spot ``a_sti_r`` on a ``(B, T, N)`` pack ``i_sti``."""
    from neuron.forward import apply_a_sti_r

    pack = session.primary_readout
    drive = _model_driver(session).prepare_i_sti(session, p, i_sti, pack)
    return apply_a_sti_r(drive, p, pack)


def _equilibrate(session, p, i_sti_batch: torch.Tensor, t_onset: int):
    """Equilibrate to ``t_onset``; returns ``(v, state)`` where state is model-specific."""
    _component_spec(session.model, session.euler)  # validate early
    B, T, _N = i_sti_batch.shape
    drv = _model_driver(session)
    state, v = drv.pre_steady(session, p, B, i_sti=i_sti_batch)
    for t in range(1, min(t_onset, T)):
        state, v = drv.step(state, v, p, i_sti_batch[:, t - 1], session)
    return v, state


def _component_at_nodes_borst(
    v_pre,
    v_post,
    g_exc,
    g_inh,
    g_Ih_on,
    g_Ih_off,
    sig_t,
    backend,
    nodes: np.ndarray,
    v_onset: np.ndarray,
    *,
    batch: int = 0,
    delta_ms: float,
    capac: float,
    g_leak: float,
    E_exc: float,
    E_inh: float,
    E_Ih: float,
    E_LEAK_REST: float,
    euler: str,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Slice node component from a completed ``update_v(..., return_component=True)`` step."""
    nodes = np.asarray(nodes, dtype=np.int64)
    with torch.no_grad():
        u = torch.as_tensor(nodes, device=v_pre.device, dtype=torch.long)
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
        ref = v_onset[b, nodes] if np.ndim(v_onset) == 2 else v_onset[nodes]
        terms = training.v_component_from_g(
            v_pre_np, packed[1], packed[2], packed[3], packed[4], packed[5], packed[6],
            delta_ms=delta_ms, capac=capac, g_leak=g_leak,
            E_exc=E_exc, E_inh=E_inh, E_Ih=E_Ih, E_LEAK_REST=E_LEAK_REST,
            euler=euler,
        )
        v_abs = packed[7]
        component = {
            "v_pre_d": v_pre_np - ref,
            "v_abs": v_abs,
            "g_exc": packed[1],
            "g_inh": packed[2],
            "g_Ih_on": packed[3],
            "g_Ih_off": packed[4],
            **terms,
            "num": terms["num"],
        }
        v_post_minus_pre_u = v_abs - v_pre_np
    return component, v_post_minus_pre_u


def _component_at_nodes_hp_lp(
    v_pre,
    v_post,
    component_t: dict[str, torch.Tensor],
    nodes: np.ndarray,
    v_onset: np.ndarray,
    *,
    batch: int = 0,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Slice hp_lp ODE component tensors at ``nodes``."""
    nodes = np.asarray(nodes, dtype=np.int64)
    with torch.no_grad():
        u = torch.as_tensor(nodes, device=v_pre.device, dtype=torch.long)
        b = int(batch)
        v_pre_np = v_pre[b, u].detach().cpu().numpy()
        v_abs = v_post[b, u].detach().cpu().numpy()
        ref = v_onset[b, nodes] if np.ndim(v_onset) == 2 else v_onset[nodes]
        component = {
            "v_pre_d": v_pre_np - ref,
            "v_abs": v_abs,
        }
        for k in (
            "i_sti", "v_sti", "v_in", "v_in_exc", "v_in_inh", "v_slow", "v_tot", "v_hp",
            "dv_leak", "dv_hp",
        ):
            t = component_t[k]
            component[k] = (t[b, u] if t.dim() > 1 else t[u]).detach().cpu().numpy()
        v_post_minus_pre_u = v_abs - v_pre_np
    return component, v_post_minus_pre_u


def _log(msg: str) -> None:
    print(msg, flush=True)


def _component_matrix(component: dict[str, np.ndarray], keys: tuple[str, ...]) -> np.ndarray:
    """Stack component keys to ``(n_nodes, n_keys)`` for vectorized accumulate."""
    return np.column_stack([component[k] for k in keys])


def _acc_dict_from_row(row: np.ndarray, keys: tuple[str, ...]) -> dict[str, float]:
    return {k: float(row[i]) for i, k in enumerate(keys)}


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
    acc: dict[str, float], accsq: dict[str, float], n: int, spec: _ComponentSpec,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for plot_key, component_key in spec.plot_key_component.items():
        if component_key is None:
            out[plot_key] = 0.0
        else:
            out[plot_key] = _sem_from_sum_sumsq(acc[component_key], accsq[component_key], n)
    return out


def _step_from_acc(
    *,
    t: int,
    t_rel: int,
    ti: int,
    v_post_val: float,
    acc: dict[str, float],
    accsq: dict[str, float],
    v_post_minus_pre_sum: float,
    n: int,
    spec: _ComponentSpec,
    g_leak: float = 0.0,
    dt_over_c: float = 0.0,
) -> dict[str, Any]:
    """One step dict from per-key sums over ``n`` nodes."""
    if n <= 0:
        raise ValueError("empty node set for mean component")
    base = {
        "t": t,
        "t_rel": t_rel,
        "ti": ti,
        "v_post": float(v_post_val),
        "v_pre_d": acc["v_pre_d"] / n,
        "v_post_minus_pre": v_post_minus_pre_sum / n,
        "i_sti": acc["i_sti"] / n,
        "sem": _step_sem(acc, accsq, n, spec),
        "n_nodes": n,
    }
    if spec.model == "borst":
        num = acc["num"] / n
        den = acc["den"] / n
        base.update({
            "g_exc_nS": acc["g_exc"] / n,
            "g_inh_nS": acc["g_inh"] / n,
            "g_leak_nS": float(g_leak),
            "dt_over_c": float(dt_over_c),
            "g_Ih_on_nS": acc["g_Ih_on"] / n,
            "g_Ih_off_nS": acc["g_Ih_off"] / n,
            "num_exc": acc["num_exc"] / n,
            "num_inh": acc["num_inh"] / n,
            "num_leak": acc["num_leak"] / n,
            "num_ihon": acc["num_ihon"] / n,
            "num_ihoff": acc["num_ihoff"] / n,
            "num_v": acc["num_v"] / n,
            "num": num,
            "den": den,
            "num_over_den": acc["v_abs"] / n,
        })
        return base
    base.update({
        "v_in": acc["v_in"] / n,
        "v_in_exc": acc["v_in_exc"] / n,
        "v_in_inh": -acc["v_in_inh"] / n,
        "v_sti": acc["v_sti"] / n,
        "v_slow": acc["v_slow"] / n,
        "v_tot": acc["v_tot"] / n,
        "v_hp": acc["v_hp"] / n,
        "dv_leak": acc["dv_leak"] / n,
        "dv_hp": acc["dv_hp"] / n,
        "dv_leak_plus_hp": (acc["dv_leak"] + acc["dv_hp"]) / n,
    })
    return base


@dataclass
class _ComponentForwardBatch:
    """One i_sti batch row for the shared full-T component forward."""

    all_nodes: np.ndarray
    t0_u: np.ndarray
    win_len: int
    node_to_cell: dict[int, str]
    nodes_by_cell: dict[str, np.ndarray]


@dataclass
class _ComponentForwardAccum:
    sums: list[dict[str, np.ndarray]]
    sumsq: list[dict[str, np.ndarray]]
    counts: list[dict[str, np.ndarray]]
    v_post_minus_pre_sums: list[dict[str, np.ndarray]]
    spec: _ComponentSpec


def _forward_component(
    session,
    p,
    i_sti: torch.Tensor,
    batches: list[_ComponentForwardBatch],
    cells: list[str],
    *,
    t_start: int | None = None,
    t_stop: int | None = None,
) -> _ComponentForwardAccum:
    """Full-T step from t=0 (same loop as ``forward_full``); accumulate component.

    Shared by bar/spot average and bar/spot hex.
    ``v_onset`` matches ``forward_full`` (``v`` at ``t_onset - 1``). Aligned index
    ``t = t_global - t0_u``. v_post is mean absolute ``v_abs``; SEM uses sum /
    sumsq like ``sem_from_traces``.

    If ``t_start``/``t_stop`` are set (from ``--ms-shown`` via ``ms_to_t``), only
    accumulate inside that inclusive aligned window; cheap steps outside it;
    break after every node has passed ``t_stop``.
    """
    if not batches:
        raise SystemExit("component forward requires at least one batch")
    if (t_start is None) ^ (t_stop is None):
        raise SystemExit("t_start and t_stop must both be set or both omitted")
    if t_start is not None and t_start > t_stop:
        raise SystemExit(f"t_start={t_start} > t_stop={t_stop}")
    B, T, _N = i_sti.shape
    if B != len(batches):
        raise SystemExit(f"i_sti B={B} != len(batches)={len(batches)}")

    spec = _component_spec(session.model, session.euler)
    drive = _prepare_drive(session, p, i_sti)
    t_onset = training.pack_t_onset(session.primary_readout)

    t_last: int | None = None
    if t_stop is not None:
        t_last = max(int(plan.t0_u.max()) + int(t_stop) for plan in batches)

    # Same ref as forward_full: v at t_onset-1, then restart so pre is stepped+accum'd.
    v_at_onset, _ = _equilibrate(session, p, drive, t_onset)
    v_onset = v_at_onset.detach().cpu().numpy().copy()
    backend = session.backend
    n_keys = spec.n_keys
    drv = _model_driver(session)
    state, v = drv.pre_steady(session, p, B, i_sti=drive)

    sums_b: list[dict[str, np.ndarray]] = []
    sumsq_b: list[dict[str, np.ndarray]] = []
    counts_b: list[dict[str, np.ndarray]] = []
    v_post_minus_pre_sums_b: list[dict[str, np.ndarray]] = []
    for plan in batches:
        wl = plan.win_len
        sums_b.append({c: np.zeros((wl, n_keys), dtype=float) for c in cells})
        sumsq_b.append({c: np.zeros((wl, n_keys), dtype=float) for c in cells})
        counts_b.append({c: np.zeros(wl, dtype=np.int64) for c in cells})
        v_post_minus_pre_sums_b.append({c: np.zeros(wl, dtype=float) for c in cells})

    node_lookups = [_node_cell_lookup(plan, cells) for plan in batches]

    for t_global in range(1, T):
        if t_last is not None and t_global > t_last:
            break
        sig_t = drive[:, t_global - 1]
        actives: list[tuple[np.ndarray, np.ndarray] | None] = []
        need_component = False
        for plan in batches:
            au = plan.all_nodes
            t_u = t_global - plan.t0_u
            in_win = (t_u >= 0) & (t_u < plan.win_len)
            if t_start is not None:
                in_win = in_win & (t_u >= t_start) & (t_u <= t_stop)
            if np.any(in_win):
                need_component = True
                actives.append((au[in_win], t_u[in_win].astype(np.int64)))
            else:
                actives.append(None)

        if not need_component:
            state, v = drv.step(state, v, p, sig_t, session)
            continue

        with torch.no_grad():
            v_pre = v
            if spec.model == "borst":
                state, v, (g_exc, g_inh, g_Ih_on, g_Ih_off) = drv.step(
                    state, v, p, sig_t, session, return_component=True,
                )
            else:
                state, v, component_t = drv.step(
                    state, v, p, sig_t, session, return_component=True,
                )

        for b, plan in enumerate(batches):
            active_pack = actives[b]
            if active_pack is None:
                continue
            active, active_t = active_pack
            if spec.model == "borst":
                component, v_post_minus_pre_u = _component_at_nodes_borst(
                    v_pre, v, g_exc, g_inh, g_Ih_on, g_Ih_off, sig_t,
                    backend, active, v_onset, batch=b,
                    delta_ms=session.delta_ms, capac=session.capac, g_leak=session.g_leak,
                    E_exc=session.E_exc, E_inh=session.E_inh, E_Ih=session.E_Ih,
                    E_LEAK_REST=session.E_LEAK_REST, euler=session.euler,
                )
            else:
                component, v_post_minus_pre_u = _component_at_nodes_hp_lp(
                    v_pre, v, component_t, active, v_onset, batch=b,
                )
            component_mat = _component_matrix(component, spec.keys)
            lookup = node_lookups[b]
            tags = lookup[active]
            for ci, cell in enumerate(cells):
                mask = tags == ci
                if not np.any(mask):
                    continue
                ts = active_t[mask]
                chunk = component_mat[mask]
                np.add.at(sums_b[b][cell], ts, chunk)
                np.add.at(sumsq_b[b][cell], ts, chunk * chunk)
                np.add.at(v_post_minus_pre_sums_b[b][cell], ts, v_post_minus_pre_u[mask])
                np.add.at(counts_b[b][cell], ts, 1)

    return _ComponentForwardAccum(
        sums=sums_b,
        sumsq=sumsq_b,
        counts=counts_b,
        v_post_minus_pre_sums=v_post_minus_pre_sums_b,
        spec=spec,
    )


def _v_post_from_accum(
    sums: np.ndarray, counts: np.ndarray, spec: _ComponentSpec,
) -> np.ndarray:
    """Mean absolute v_post from accumulated ``v_abs``."""
    n = np.maximum(counts, 1)
    v_post = sums[:, spec.i_v_abs] / n
    v_post[counts == 0] = 0.0
    return v_post


def _v_post_d_from_accum(
    sums: np.ndarray, v_post_minus_pre_sums: np.ndarray, counts: np.ndarray,
    spec: _ComponentSpec,
) -> np.ndarray:
    """Mean ``v_post_d`` = v_post − v_onset = v_pre_d + v_post_minus_pre."""
    n = np.maximum(counts, 1)
    v_post_d = sums[:, spec.i_v_pre_d] / n + v_post_minus_pre_sums / n
    v_post_d[counts == 0] = 0.0
    return v_post_d


def _dominant_drive_from_step(
    step: dict[str, Any] | None, *, model: str,
) -> str | None:
    if step is None:
        return None
    if model == "hp_lp":
        if abs(step["v_in_exc"]) >= abs(step["v_in_inh"]):
            return "exc" if abs(step["v_in_exc"]) > 1e-9 else "none"
        return "inh"
    if abs(step["num_exc"]) >= abs(step["num_inh"]):
        return "exc" if abs(step["num_exc"]) > 1e-9 else "none"
    return "inh"


def _finalize_component_report(
    *,
    cell: str,
    task: str,
    spec: str | None,
    mode: str,
    before_t: int,
    nodes: np.ndarray,
    p,
    session,
    sums: np.ndarray,
    sumsq: np.ndarray,
    counts: np.ndarray,
    v_post_minus_pre_sums: np.ndarray,
    v_post: np.ndarray,
    time_window: TimeWindow,
    ti_mode: str,
    component_spec: _ComponentSpec,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one report dict from a single batch×cell accum row."""
    # Peak on |v_post_d| (= |v_post − v_onset| = |v_pre_d + v_post_minus_pre|).
    v_post_d = _v_post_d_from_accum(sums, v_post_minus_pre_sums, counts, component_spec)
    n = int(v_post_d.size)
    if time_window.kind == "ms":
        dt = float(session.delta_ms)
        t_lo = training.ms_to_t(time_window.start, delta_ms=dt)
        t_hi = training.ms_to_t(time_window.stop, delta_ms=dt)
        if t_lo < 0 or t_hi >= n or t_lo > t_hi:
            raise SystemExit(
                f"--ms-shown {time_window.start:g},{time_window.stop:g} "
                f"(t={t_lo}:{t_hi}) out of range for accum length {n}"
            )
        seg = v_post_d[t_lo:t_hi + 1]
        peak_t = t_lo + int(np.argmax(np.abs(seg))) if seg.size else t_lo
    else:
        peak_t = _v_post_d_peak_t_rel(v_post_d, before_t)
        t_lo = max(0, peak_t + int(time_window.start))
        t_hi = min(n - 1, peak_t + int(time_window.stop))
    steps: list[dict[str, Any]] = []
    peak_step: dict[str, Any] | None = None
    for t in range(t_lo, t_hi + 1):
        n_nodes = int(counts[t])
        if n_nodes == 0:
            continue
        if ti_mode == "t_rel":
            ti = t
        elif ti_mode == "abs_minus_before":
            ti = t - before_t
        else:
            raise ValueError(f"unknown ti_mode {ti_mode!r}")
        t_rel = t - peak_t
        step = _step_from_acc(
            t=t, t_rel=t_rel, ti=ti, v_post_val=float(v_post[t]),
            acc=_acc_dict_from_row(sums[t], component_spec.keys),
            accsq=_acc_dict_from_row(sumsq[t], component_spec.keys),
            v_post_minus_pre_sum=float(v_post_minus_pre_sums[t]), n=n_nodes,
            spec=component_spec,
            g_leak=float(session.g_leak),
            dt_over_c=float(training.membrane_dt_over_c(session.capac, session.delta_ms)),
        )
        steps.append(step)
        if t == peak_t:
            peak_step = step
    if peak_step is None and steps:
        peak_step = steps[len(steps) // 2]
    if ti_mode == "abs_minus_before" and v_post_d.size > before_t:
        onset = _first_nonzero_t_rel(v_post_d[before_t:])
    else:
        onset = _first_nonzero_t_rel(v_post_d)
    report: dict[str, Any] = {
        "mode": mode,
        "model": component_spec.model,
        "cell": cell,
        "n_nodes": int(nodes.size),
        "task": task,
        "spec": spec,
        "before_t": before_t,
        "time_window_kind": time_window.kind,
        "time_window": [time_window.start, time_window.stop],
        "t_window": [t_lo, t_hi],
        "v_post_d_peak_t": peak_t,
        "v_post_d_peak": float(v_post_d[peak_t]),
        "v_post_d_polarity": _polarity(float(v_post_d[peak_t])),
        "v_post_d_onset_t": onset,
        "params": _node_params(p, session, int(nodes[0])),
        "globals": _globals(session),
        "steps": steps,
        "peak_step": peak_step,
        "peak_drive": _dominant_drive_from_step(
            peak_step, model=component_spec.model,
        ),
        "v_post": v_post.tolist(),
    }
    if extra:
        report.update(extra)
    return report


def _first_nonzero_t_rel(trace: np.ndarray, *, eps: float = 1e-6) -> int | None:
    idx = np.where(np.abs(trace) > eps)[0]
    return int(idx[0]) if idx.size else None


def _v_post_d_peak_t_rel(
    v_post_d: np.ndarray,
    before_t: int | None,
    *,
    horizon: int | None = 40,
) -> int:
    """Index of largest |v_post_d| (= |v_post − v_onset|) after onset (optional ``horizon``)."""
    arr = np.asarray(v_post_d, dtype=float)
    if before_t is not None and 0 < before_t < arr.size:
        stop = arr.size
        if horizon is not None:
            stop = min(stop, before_t + int(horizon))
        post = arr[before_t:stop]
        if post.size == 0:
            return int(before_t)
        return int(before_t + int(np.argmax(np.abs(post))))
    stop = arr.size if horizon is None else min(arr.size, int(horizon))
    return int(np.argmax(np.abs(arr[:stop])))


def _polarity(v: float, *, eps: float = 1e-3) -> str:
    if v > eps:
        return "+"
    if v < -eps:
        return "-"
    return "0"


def _scalar_param_at_node(p, key: str, session, node: int) -> float:
    raw = p[key]
    backend = session.backend
    if torch.is_tensor(raw):
        if raw.dim() == 0:
            return float(raw.item())
        if backend.network is not None:
            ci = int(backend.network.node_cell[node])
        else:
            ci = int(node) % int(backend.n_cells)
        return float(raw[ci])
    return float(raw)


def _node_params(p, session, node: int) -> dict[str, float]:
    backend = session.backend
    for key in ("a_gt", "bias_gt"):
        if key not in p:
            raise SystemExit(f"params missing {key}")
    a_gt = _scalar_param_at_node(p, "a_gt", session, node)
    bias_gt = _scalar_param_at_node(p, "bias_gt", session, node)
    if session.model == "hp_lp":
        return {
            "a_in": float(p["a_in"][node]),
            "a_out": float(p["a_out"][node]),
            "bias_out": float(p["bias_out"][node]),
            "a_gt": a_gt,
            "bias_gt": bias_gt,
            "v_rest_mV": float(p["v_rest"][node]),
            "tau_lp_ms": float(p["tau_lp"][node]),
            "tau_hp_ms": float(p["tau_hp"][node]),
            "a_slow": float(p["a_slow"][node]),
        }
    from neuron.schema import borst_ih_off_kwargs

    ih_off = (session.train_opts or {})["ih_off"]
    gmax_off, *_rest = borst_ih_off_kwargs(p, ih_off)
    return {
        "a_in": float(p["a_in"][node]),
        "a_out": float(p["a_out"][node]),
        "a_gt": a_gt,
        "bias_gt": bias_gt,
        "v_th_mV": float(p["v_th"][node]),
        "Ih_gmax": float(p["Ih_gmax"][node]),
        "Ih_gmax_off": float(gmax_off[node]),
        "e_leak_mV": float(backend.e_leak[node]),
    }


def _globals(session):
    pack = session.primary_readout
    t_onset = training.pack_t_onset(pack)
    if session.model == "hp_lp":
        return {
            "delta_ms": float(session.delta_ms),
            "state_clamp": float(session.STATE_CLAMP),
            "g_in_nS": float(session.g_in),
            "euler": str(session.euler),
            "t_onset": t_onset,
        }
    return {
        "E_exc": float(session.E_exc),
        "E_inh": float(session.E_inh),
        "E_Ih": float(session.E_Ih),
        "E_IH_OFF": float(training.e_ih_off(session.E_LEAK_REST, session.E_Ih)),
        "g_leak_nS": float(session.g_leak),
        "dt_over_c": float(training.membrane_dt_over_c(session.capac, session.delta_ms)),
        "delta_ms": float(session.delta_ms),
        "euler": str(session.euler),
        "t_onset": t_onset,
    }


def _node_to_cell_map(nodes_by_cell: dict[str, np.ndarray]) -> dict[int, str]:
    u2c: dict[int, str] = {}
    for cell, us in nodes_by_cell.items():
        for u in np.asarray(us, dtype=np.int64).ravel():
            u2c[int(u)] = cell
    return u2c


def _node_cell_lookup(plan: _ComponentForwardBatch, cells: list[str]) -> np.ndarray:
    """Map node id → index in ``cells`` (-1 if absent). Length ``max(node_id)+1``."""
    cell_i = {c: i for i, c in enumerate(cells)}
    if plan.all_nodes.size == 0:
        return np.empty(0, dtype=np.int32)
    out = np.full(int(plan.all_nodes.max()) + 1, -1, dtype=np.int32)
    for u, cname in plan.node_to_cell.items():
        ci = cell_i.get(cname)
        if ci is not None:
            out[int(u)] = ci
    return out


def _merge_forward_accum(
    accum: _ComponentForwardAccum,
    forward_batches: list[_ComponentForwardBatch],
    cells: list[str],
    win_len: int,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    """Sum per-cell accum rows across run batches (spot multi-stimulus mean)."""
    n_keys = accum.spec.n_keys
    sums = {c: np.zeros((win_len, n_keys), dtype=float) for c in cells}
    sumsq = {c: np.zeros((win_len, n_keys), dtype=float) for c in cells}
    counts = {c: np.zeros(win_len, dtype=np.int64) for c in cells}
    v_post_minus_pre_sums = {c: np.zeros(win_len, dtype=float) for c in cells}
    nodes_ref = {c: np.zeros(0, dtype=np.int64) for c in cells}
    for b, plan in enumerate(forward_batches):
        for cell in cells:
            if cell not in plan.nodes_by_cell:
                continue
            us = plan.nodes_by_cell[cell]
            if us.size == 0:
                continue
            if nodes_ref[cell].size == 0:
                nodes_ref[cell] = us
            sums[cell] += accum.sums[b][cell]
            sumsq[cell] += accum.sumsq[b][cell]
            counts[cell] += accum.counts[b][cell]
            v_post_minus_pre_sums[cell] += accum.v_post_minus_pre_sums[b][cell]
    return sums, sumsq, counts, v_post_minus_pre_sums, nodes_ref


def _make_forward_batch(
    nodes_by_cell: dict[str, np.ndarray],
    *,
    t0_bn_row: np.ndarray,
    win_len: int,
) -> _ComponentForwardBatch:
    """Build one run batch; ``t0_u[i] = t0_bn_row[all_nodes[i]]``."""
    all_nodes = np.unique(np.concatenate([us for us in nodes_by_cell.values()]))
    return _ComponentForwardBatch(
        all_nodes=all_nodes,
        t0_u=np.asarray(t0_bn_row[all_nodes], dtype=np.int64),
        win_len=int(win_len),
        node_to_cell=_node_to_cell_map(nodes_by_cell),
        nodes_by_cell=nodes_by_cell,
    )


# ---------------------------------------------------------------------------
# Average bar components (cost-extent)
# ---------------------------------------------------------------------------


def _bar_meta(session, task: str):
    """One-shot ``(specs, grids)`` for a moving-bar task."""
    specs = bar_specs_for_session(session, task)
    pack = session.pack_for(task)
    grids = moving_bar_session_t0_grids(
        session, specs, pack.cost_extent, int(session.n_t),
        t_onset=training.pack_t_onset(pack),
        delta_ms=float(session.delta_ms),
    )
    return specs, grids


def _bar_specs_requested(
    session,
    task: str,
    cells: list[str],
    requested: list[str] | None,
    *,
    specs=None,
    grids=None,
) -> list[str]:
    """Spec list for average-mode bar without a full forward bundle."""
    if specs is None or grids is None:
        specs, grids = _bar_meta(session, task)
    all_specs = [s.name for s in specs]
    try:
        if requested is not None:
            return filter_requested_specs(all_specs, requested)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    row = moving_bar_row_specs(session, task, grids.side)
    out: list[str] = []
    for cell in cells:
        for s in row.get(cell, all_specs):
            if s in all_specs and s not in out:
                out.append(s)
    return out or list(all_specs)


def _resolve_bar_spec_i_sti(
    session,
    task: str,
    spec_names: list[str],
    *,
    specs=None,
    grids=None,
):
    """Validate specs; return ``(pack, specs, grids, bis, i_sti, t0_bn)``."""
    if task not in training.MOVING_BAR_TASKS:
        raise SystemExit(f"unsupported task {task!r}")
    if not spec_names:
        raise SystemExit("bar component forward requires at least one spec")
    pack = session.pack_for(task)
    if specs is None or grids is None:
        specs, grids = _bar_meta(session, task)
    name_to_bi = {s.name: i for i, s in enumerate(specs)}
    missing = [s for s in spec_names if s not in name_to_bi]
    if missing:
        raise SystemExit(f"spec(s) {missing} not in {[s.name for s in specs]}")
    bis = [name_to_bi[s] for s in spec_names]
    return pack, specs, grids, bis, pack.i_sti[bis], np.asarray(grids.t0_bn)


def _analyze_component_forward(
    session,
    *,
    p,
    cells: list[str],
    task: str,
    i_sti: torch.Tensor,
    forward_batches: list[_ComponentForwardBatch],
    before_t: list[int],
    batch_specs: list[str | None],
    time_window: TimeWindow,
    mode: str,
    ti_mode: str,
    merge_batches: bool = False,
    extra: dict[str, Any] | None = None,
    extra_for_cell=None,
    n_nodes_for_cell=None,
):
    """Shared spot/bar: ``_forward_component`` → finalize reports.

    * ``merge_batches=False`` (bar): ``reports[spec][cell]``; ``batch_specs`` are str.
    * ``merge_batches=True`` (spot): sum across batches → ``reports[cell]``.
    """
    if not forward_batches:
        raise SystemExit("component forward requires at least one batch")
    if len(before_t) != len(forward_batches) or len(batch_specs) != len(forward_batches):
        raise SystemExit("before_t/batch_specs length must match forward_batches")

    dt = float(session.delta_ms)
    accum = _forward_component(
        session, p, i_sti, forward_batches, cells,
        t_start=time_window.forward_t_start(delta_ms=dt),
        t_stop=time_window.forward_t_stop(delta_ms=dt),
    )
    component_spec = accum.spec

    def _one_report(
        *,
        cell: str,
        spec: str | None,
        before: int,
        nodes: np.ndarray,
        sums: np.ndarray,
        sumsq: np.ndarray,
        counts: np.ndarray,
        v_post_minus_pre_sums: np.ndarray,
    ) -> dict[str, Any]:
        if nodes.size == 0:
            raise SystemExit(f"no nodes for cell {cell!r}")
        v_post = _v_post_from_accum(sums, counts, component_spec)
        v_post_d = _v_post_d_from_accum(
            sums, v_post_minus_pre_sums, counts, component_spec,
        )
        cell_extra = dict(extra) if extra else {}
        if extra_for_cell is not None:
            cell_extra.update(extra_for_cell(cell, v_post, v_post_d) or {})
        report = _finalize_component_report(
            cell=cell,
            task=task,
            spec=spec,
            mode=mode,
            before_t=before,
            nodes=nodes,
            p=p,
            session=session,
            sums=sums,
            sumsq=sumsq,
            counts=counts,
            v_post_minus_pre_sums=v_post_minus_pre_sums,
            v_post=v_post,
            time_window=time_window,
            ti_mode=ti_mode,
            component_spec=component_spec,
            extra=cell_extra or None,
        )
        if n_nodes_for_cell is not None:
            report["n_nodes"] = int(n_nodes_for_cell(cell))
        return report

    if merge_batches:
        win_len = forward_batches[0].win_len
        sums, sumsq, counts, v_post_minus_pre_sums, nodes_ref = _merge_forward_accum(
            accum, forward_batches, cells, win_len,
        )
        before = int(before_t[0])
        return {
            cell: _one_report(
                cell=cell,
                spec=None,
                before=before,
                nodes=nodes_ref[cell],
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
            raise SystemExit("non-merge component forward requires batch_specs as str")
        out[spec] = {}
        for cell in cells:
            out[spec][cell] = _one_report(
                cell=cell,
                spec=spec,
                before=int(before_t[b]),
                nodes=forward_batches[b].nodes_by_cell[cell],
                sums=accum.sums[b][cell],
                sumsq=accum.sumsq[b][cell],
                counts=accum.counts[b][cell],
                v_post_minus_pre_sums=accum.v_post_minus_pre_sums[b][cell],
            )
    return out


def _analyze_bar_forward(
    session,
    *,
    p,
    cells: list[str],
    task: str,
    spec_names: list[str],
    time_window: TimeWindow,
    nodes_for_bi,
    mode: str,
    specs=None,
    grids=None,
    extra: dict[str, Any] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Bar prep: resolve i_sti/specs → ``_analyze_component_forward`` (no merge)."""
    pack, specs, grids, bis, i_sti, t0_bn = _resolve_bar_spec_i_sti(
        session, task, spec_names, specs=specs, grids=grids,
    )
    before_b: list[int] = []
    forward_batches: list[_ComponentForwardBatch] = []
    for bi, spec in zip(bis, spec_names):
        usets = nodes_for_bi(bi, spec, pack=pack, t0_bn=t0_bn)
        before = int(grids.before_t[spec])
        after = int(grids.after_t[spec])
        before_b.append(before)
        forward_batches.append(
            _make_forward_batch(usets, t0_bn_row=t0_bn[bi], win_len=before + after + 1),
        )
    return _analyze_component_forward(
        session,
        p=p,
        cells=cells,
        task=task,
        i_sti=i_sti,
        forward_batches=forward_batches,
        before_t=before_b,
        batch_specs=list(spec_names),
        time_window=time_window,
        mode=mode,
        ti_mode="t_rel",
        merge_batches=False,
        extra=extra,
    )


def analyze_bar_average(
    session,
    *,
    p,
    cells: list[str],
    task: str,
    spec_names: list[str],
    time_window: TimeWindow,
    specs=None,
    grids=None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """One batched v forward over all requested specs; mean component per cell.

    Returns ``reports[spec][cell]``. v_post + component share ``_forward_component``.
    """
    cols_holder: list = []

    def nodes_for_bi(bi, spec, *, pack, t0_bn):
        C = session.backend.network
        if not cols_holder:
            cols_holder.append(moving_bar_cost_hexes(C, cost_extent=pack.cost_extent))
        hexes = cols_holder[0]
        out: dict[str, np.ndarray] = {}
        for cell in cells:
            try:
                nodes = moving_bar_nodes_on_hexes(C, cell, hexes)
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
            nodes = nodes[t0_bn[bi, nodes] >= 0]
            if nodes.size == 0:
                raise SystemExit(f"no valid {cell} nodes in cost_extent for bar aggregation")
            out[cell] = nodes
        return out

    return _analyze_bar_forward(
        session,
        p=p,
        cells=cells,
        task=task,
        spec_names=spec_names,
        time_window=time_window,
        nodes_for_bi=nodes_for_bi,
        mode="average",
        specs=specs,
        grids=grids,
    )


# ---------------------------------------------------------------------------
# Average spot components (Euclidean radius)
# ---------------------------------------------------------------------------


def _spot_session_readout(session_one, cells: list[str]):
    """Session-scoped spot cost readout (all radii) for component forward."""
    pack = session_one.primary_readout
    C = session_one.backend.network
    if C is None:
        raise SystemExit("spot average requires a network backend")
    opts = dict((session_one.train_opts or {}).get(f"{pack.name}_stimulus_opts") or {})
    spot = spot_from_opts(C, stimulus_opts=opts)
    (
        batch_idx, node_idx, radii, type_idx, _stim_u, _stim_v, _du, _dv, _center_row,
    ) = build_spot_center_readout(
        C,
        spot_stimulus_batches(spot),
        pack_spot_cost_radii(pack),
        pack.cost_extent,
    )
    type_i: dict[str, int] = {}
    for cell in cells:
        if cell not in C.cell_names:
            raise SystemExit(f"unknown cell {cell!r}")
        type_i[cell] = C.cell_names.index(cell)
    return pack, batch_idx, node_idx, radii, type_idx, type_i


def _spot_radius_row(radii: np.ndarray, radius: int) -> np.ndarray:
    """True for cost-readout rows at Euclidean ``radius``."""
    return np.isclose(np.asarray(radii, dtype=np.float64), float(radius))


def _gt_affine_for_cell(p, session, cell: str) -> tuple[float, float]:
    """``(a_gt, effective_bias)`` for one cell type name (matches cost)."""
    names = [str(n) for n in session.backend.network.cell_names]
    ci = names.index(str(cell))
    gs, gb = p["a_gt"], p["bias_gt"]
    scale = float(gs[ci] if torch.is_tensor(gs) and gs.dim() > 0 else gs)
    bias = float(gb[ci] if torch.is_tensor(gb) and gb.dim() > 0 else gb)
    if "v_th" in p:
        vt = p["v_th"]
        bias = bias + float(vt[ci] if torch.is_tensor(vt) and vt.dim() > 0 else vt)
    return scale, bias


def _spot_gt_v_post_extra(
    *,
    cell: str,
    gt_on: dict,
    radius: int,
    v_post: np.ndarray,
    v_post_d: np.ndarray,
    t_onset: int,
    a_gt: float,
    bias_gt: float,
) -> dict[str, Any]:
    """GT on absolute ``v`` axis: ``a_gt * gt + effective_bias`` (matches cost)."""
    extra: dict[str, Any] = {"gt_peak": None, "gt_v_post": None, "radius": radius}
    if cell not in gt_on:
        return extra
    gt_cube = np.asarray(gt_on[cell], dtype=float)
    if radius < 0 or radius >= gt_cube.shape[0]:
        return extra
    gt_row = gt_cube[radius]
    gt_aff = float(a_gt) * gt_row + float(bias_gt)
    peak_probe = _v_post_d_peak_t_rel(v_post_d, t_onset)
    if 0 <= peak_probe < gt_aff.shape[0]:
        extra["gt_peak"] = float(gt_aff[peak_probe])
    mask = np.isfinite(v_post) & np.isfinite(v_post_d)
    if not np.any(mask):
        return extra
    extra["gt_v_post"] = gt_aff.tolist()
    return extra


def analyze_spot_average(
    session_one,
    *,
    p,
    cells: list[str],
    task: str,
    time_window: TimeWindow,
    radius: int = 0,
) -> dict[str, dict[str, Any]]:
    """One batched v forward over spot stimulus rows; mean at Euclidean ``radius``.

    ``time_window`` is absolute aligned ms for spot (``0`` = trial start). Pre
    only: ``TimeWindow("ms", 0, ms_pre)``. Do not pass negative ``start`` for
    "before onset" — that confuses stimulus length with the analyze window.
    """
    if task not in training.SPOT_TASKS:
        raise SystemExit(f"unsupported task {task!r}")
    radius = int(radius)
    pack, batch_idx, node_idx, radii, type_idx, type_i = _spot_session_readout(
        session_one, cells,
    )
    radius_row = _spot_radius_row(radii, radius)
    t_onset = training.pack_t_onset(pack)

    i_sti = pack.i_sti if pack.i_sti.dim() == 3 else pack.i_sti.unsqueeze(0)
    B_all, T, _N = i_sti.shape
    t0_abs = np.zeros(_N, dtype=np.int64)
    win_len = time_window.aligned_win_len(T, delta_ms=float(session_one.delta_ms))

    forward_batches: list[_ComponentForwardBatch] = []
    i_sti_rows: list[int] = []
    for b in range(B_all):
        row_mask = radius_row & (batch_idx == b)
        if not np.any(row_mask):
            continue
        usets: dict[str, np.ndarray] = {}
        for cell in cells:
            m = row_mask & (type_idx == type_i[cell])
            if np.any(m):
                usets[cell] = np.unique(node_idx[m])
        if not usets:
            continue
        forward_batches.append(
            _make_forward_batch(usets, t0_bn_row=t0_abs, win_len=win_len),
        )
        i_sti_rows.append(b)

    if not forward_batches:
        raise SystemExit(
            f"no spot nodes at radius={radius} for requested cells in spot readout"
        )

    gt_by_contrast = resolve_spot_gt_cubes(
        {contrast_for_task(pack.name): session_one},
    )
    contrast = contrast_for_task(pack.name)
    gt_on = gt_by_contrast.get(contrast) or {}

    def extra_for_cell(
        cell: str, v_post: np.ndarray, v_post_d: np.ndarray,
    ) -> dict[str, Any]:
        a_gt, bias_gt = _gt_affine_for_cell(p, session_one, cell)
        return _spot_gt_v_post_extra(
            cell=cell,
            gt_on=gt_on,
            radius=radius,
            v_post=v_post,
            v_post_d=v_post_d,
            t_onset=t_onset,
            a_gt=a_gt,
            bias_gt=bias_gt,
        )

    def n_nodes_for_cell(cell: str) -> int:
        return int(np.sum(radius_row & (type_idx == type_i[cell])))

    n_b = len(forward_batches)
    return _analyze_component_forward(
        session_one,
        p=p,
        cells=cells,
        task=task,
        i_sti=i_sti[i_sti_rows],
        forward_batches=forward_batches,
        before_t=[t_onset] * n_b,
        batch_specs=[None] * n_b,
        time_window=time_window,
        mode="average",
        ti_mode="abs_minus_before",
        merge_batches=True,
        extra={"radius": radius},
        extra_for_cell=extra_for_cell,
        n_nodes_for_cell=n_nodes_for_cell,
    )


# ---------------------------------------------------------------------------
# Hex mode (single hex; same run as average)
# ---------------------------------------------------------------------------


def _nodes_at_hex(session, cell: str, *, at_x: float, at_y: float, cost_extent: int):
    C = session.backend.network
    if C is None:
        raise SystemExit("hex mode requires a network backend")
    hexes = filter_sti_hexes(
        moving_bar_cost_hexes(C, cost_extent=cost_extent),
        at_x=at_x,
        at_y=at_y,
    )
    if not hexes:
        raise SystemExit(f"no hex at x={at_x!r} y={at_y!r} within cost_extent={cost_extent}")
    if len(hexes) > 1:
        raise SystemExit(f"multiple hexes at x={at_x!r} y={at_y!r}; pick a unique hex")
    col = hexes[0]
    if cell not in C.cell_names:
        raise SystemExit(f"unknown cell {cell!r}")
    nodes = col2gt(C, int(col.u), int(col.v), cell).tolist()
    if not nodes:
        raise SystemExit(f"no {cell} node at hex ({at_x},{at_y})")
    return col, nodes


def _resolve_hex_node(
    session,
    cell: str,
    *,
    at_x: float,
    at_y: float,
    cost_extent: int,
    node: int | None,
):
    """Resolve ``(col, node_id)`` for hex mode; ``--node`` if multiple at hex."""
    col, nodes = _nodes_at_hex(
        session, cell, at_x=at_x, at_y=at_y, cost_extent=cost_extent,
    )
    if node is None:
        if len(nodes) > 1:
            raise SystemExit(f"multiple {cell} at ({at_x},{at_y}): {nodes}; pass --node")
        node = nodes[0]
    elif node not in nodes:
        raise SystemExit(f"node {node} not in {nodes}")
    return col, int(node)


def analyze_spot_hex(
    session_one,
    *,
    p,
    cell: str,
    task: str,
    at_x: float,
    at_y: float,
    node: int | None,
    time_window: TimeWindow,
) -> dict[str, dict[str, Any]]:
    """One batched v forward at one hex; stim-on (radius 0) rows for that node only."""
    if task not in training.SPOT_TASKS:
        raise SystemExit(f"unsupported task {task!r}")
    pack, batch_idx, node_idx, radii, type_idx, type_i = _spot_session_readout(
        session_one, [cell],
    )
    radius_row = _spot_radius_row(radii, 0)
    col, node = _resolve_hex_node(
        session_one, cell, at_x=at_x, at_y=at_y,
        cost_extent=pack.cost_extent, node=node,
    )
    t_onset = training.pack_t_onset(pack)

    i_sti = pack.i_sti if pack.i_sti.dim() == 3 else pack.i_sti.unsqueeze(0)
    B_all, T, _N = i_sti.shape
    t0_abs = np.zeros(_N, dtype=np.int64)
    win_len = time_window.aligned_win_len(T, delta_ms=float(session_one.delta_ms))
    node_arr = np.asarray([node], dtype=np.int64)
    type_cell = type_i[cell]

    forward_batches: list[_ComponentForwardBatch] = []
    i_sti_rows: list[int] = []
    for b in range(B_all):
        row_mask = (
            radius_row
            & (batch_idx == b)
            & (type_idx == type_cell)
            & (node_idx == node)
        )
        if not np.any(row_mask):
            continue
        forward_batches.append(
            _make_forward_batch({cell: node_arr}, t0_bn_row=t0_abs, win_len=win_len),
        )
        i_sti_rows.append(b)

    if not forward_batches:
        raise SystemExit(
            f"no stim-on spot rows for {cell} node {node} at hex ({at_x},{at_y})"
        )

    gt_by_contrast = resolve_spot_gt_cubes(
        {contrast_for_task(pack.name): session_one},
    )
    contrast = contrast_for_task(pack.name)
    gt_on = gt_by_contrast.get(contrast) or {}

    def extra_for_cell(
        cell_name: str, v_post: np.ndarray, v_post_d: np.ndarray,
    ) -> dict[str, Any]:
        a_gt, bias_gt = _gt_affine_for_cell(p, session_one, cell_name)
        return _spot_gt_v_post_extra(
            cell=cell_name,
            gt_on=gt_on,
            radius=0,
            v_post=v_post,
            v_post_d=v_post_d,
            t_onset=t_onset,
            a_gt=a_gt,
            bias_gt=bias_gt,
        )

    n_b = len(forward_batches)
    return _analyze_component_forward(
        session_one,
        p=p,
        cells=[cell],
        task=task,
        i_sti=i_sti[i_sti_rows],
        forward_batches=forward_batches,
        before_t=[t_onset] * n_b,
        batch_specs=[None] * n_b,
        time_window=time_window,
        mode="hex",
        ti_mode="abs_minus_before",
        merge_batches=True,
        extra={
            "node": node,
            "hex": {"x": at_x, "y": at_y},
            "uv": {"u": int(col.u), "v": int(col.v)},
        },
        extra_for_cell=extra_for_cell,
        n_nodes_for_cell=lambda _c: 1,
    )


def analyze_bar_hex(
    session,
    *,
    p,
    cell: str,
    task: str,
    spec_names: list[str],
    at_x: float,
    at_y: float,
    node: int | None,
    time_window: TimeWindow,
    specs=None,
    grids=None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """One batched v forward over specs at one hex; returns ``reports[spec][cell]``."""
    pack = session.pack_for(task)
    col, node = _resolve_hex_node(
        session, cell, at_x=at_x, at_y=at_y,
        cost_extent=pack.cost_extent, node=node,
    )
    node_arr = np.asarray([node], dtype=np.int64)
    usets = {cell: node_arr}

    def nodes_for_bi(bi, spec, *, pack, t0_bn):
        if int(t0_bn[bi, node]) < 0:
            raise SystemExit(f"no t0 for node {node} on spec {spec!r}")
        return usets

    return _analyze_bar_forward(
        session,
        p=p,
        cells=[cell],
        task=task,
        spec_names=spec_names,
        time_window=time_window,
        nodes_for_bi=nodes_for_bi,
        mode="hex",
        specs=specs,
        grids=grids,
        extra={
            "node": node,
            "hex": {"x": at_x, "y": at_y},
            "uv": {"u": int(col.u), "v": int(col.v)},
        },
    )



# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_filename(report: dict[str, Any], *, file_suffix: str = "", html: bool = False) -> str:
    from figure.util import plot_file_ext

    parts = [report["cell"], report["task"], "v", report.get("mode", "average")]
    if report.get("spec"):
        parts.append(str(report["spec"]))
    if report.get("mode") == "hex":
        hx = report["hex"]
        parts.append(f"x{hx['x']}_y{hx['y']}")
    radius = report.get("radius")
    if radius is not None and int(radius) != 0:
        parts.append(f"radius{int(radius)}")
    return "_".join(parts) + f"{file_suffix}{plot_file_ext(html=html)}"


def _overlay_plot_filename(
    reports: list[dict[str, Any]], *, file_suffix: str = "", html: bool = False,
) -> str:
    from figure.util import plot_file_ext

    r0 = reports[0]
    specs = "_".join(str(r["spec"]) for r in reports)
    return (
        f"{r0['cell']}_{r0['task']}_v_overlay_{specs}"
        f"{file_suffix}{plot_file_ext(html=html)}"
    )


def _component_figure(title: str, spec: _ComponentSpec):
    """Shared grid figure: rows = plot panels, cols = traces within a row."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from figure.util import save_figure

    n_rows, n_hexes = _component_axes_grid(spec)
    fig, axes = plt.subplots(
        n_rows, n_hexes,
        figsize=(2.6 * n_hexes, 2.2 * n_rows),
        sharex=True,
        constrained_layout=True,
    )
    if n_rows == 1 and n_hexes == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_hexes == 1:
        axes = axes[:, np.newaxis]
    return fig, axes, save_figure


def _hide_unused_axes(axes, spec: _ComponentSpec) -> None:
    n_rows, n_hexes = axes.shape
    for ri, (_panel_ylabel, series) in enumerate(spec.plot_panels):
        for ci in range(len(series), n_hexes):
            axes[ri, ci].set_visible(False)


def _visible_axes(axes):
    return [ax for ax in axes.ravel() if ax.get_visible()]


def _style_component_ax(
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
    spec: _ComponentSpec,
) -> None:
    """One tight data-driven ylim per row in ``spec.row_shared_ylim``."""
    for ri, curves in row_curves.items():
        if not curves:
            continue
        _, series = spec.plot_panels[ri]
        ylo, yhi = _shared_row_ylim(curves)
        for ci in range(len(series)):
            axes[ri, ci].set_ylim(ylo, yhi)


def _save_component_figure(
    fig, axes, *, xlabel: str, out_path, save_figure, spec: _ComponentSpec,
) -> None:
    _hide_unused_axes(axes, spec)
    last_row = axes.shape[0] - 1
    for ci in range(axes.shape[1]):
        ax = axes[last_row, ci]
        if ax.get_visible():
            ax.set_xlabel(xlabel, fontsize=8)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    save_figure(fig, out_path)
    print(f"wrote {out_path}")


def _plot_component_reports(
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

    models = {str(r.get("model", "borst")) for r in reports}
    if len(models) != 1:
        raise SystemExit(f"overlay requires one model; got {sorted(models)}")
    kinds = {str(r.get("time_window_kind", "t_rel")) for r in reports}
    if len(kinds) != 1:
        raise SystemExit(f"overlay requires one time_window_kind; got {sorted(kinds)}")
    eulers = {
        training.expand_euler((r.get("globals") or {}).get("euler", "implicit"))
        for r in reports
    }
    if len(eulers) != 1:
        raise SystemExit(f"overlay requires one euler; got {sorted(eulers)}")
    spec = _component_spec(models.pop(), eulers.pop())

    fig, axes, save_figure = _component_figure(title, spec)
    colors = _plot_colors()
    linestyles = ("-", "--", "-.", ":")
    overlay = len(reports) > 1
    e_leak_mV = float(reports[0].get("params", {}).get("e_leak_mV", 0.0))
    globs = reports[0].get("globals") or {}
    delta_ms = float(globs.get("delta_ms", training.DELTA_MS))
    row_curves: dict[int, list[np.ndarray]] = {ri: [] for ri in spec.row_shared_ylim}
    tc = _plot_trace_colors(colors, spec)

    for ri, (panel_ylabel, series) in enumerate(spec.plot_panels):
        for ci, (key, label) in enumerate(series):
            ax = axes[ri, ci]
            color = tc[label]
            show_legend = overlay and ri == 0 and ci == 0
            drew_gt = False
            for si, rep in enumerate(reports):
                ls = linestyles[si % len(linestyles)] if overlay else "-"
                ts = np.asarray([s["t"] for s in rep["steps"]], dtype=int)
                xs = ts.astype(float) * delta_ms
                y = np.asarray([s[key] for s in rep["steps"]], dtype=float)
                sem = np.asarray(
                    [float(s.get("sem", {}).get(key, 0.0)) for s in rep["steps"]],
                    dtype=float,
                )
                if ri in spec.row_shared_ylim:
                    row_curves[ri].append(y)
                    if np.any(sem):
                        row_curves[ri].append(y + sem)
                        row_curves[ri].append(y - sem)
                plot_sem_band(ax, xs, y, sem, color=color, alpha=0.3)
                model_label = (
                    str(rep["spec"]) if show_legend
                    else ("v_post" if key == "v_post" and rep.get("gt_v_post") else "_nolegend_")
                )
                ax.plot(
                    xs, y,
                    label=model_label,
                    color=color,
                    linestyle=ls,
                    linewidth=1.4,
                )
                if key == "v_post" and rep.get("gt_v_post") is not None:
                    gt_full = np.asarray(rep["gt_v_post"], dtype=float)
                    valid = (ts >= 0) & (ts < gt_full.shape[0])
                    if np.any(valid):
                        y_gt = gt_full[ts[valid]]
                        xs_gt = xs[valid]
                        if ri in spec.row_shared_ylim:
                            row_curves[ri].append(y_gt)
                        ax.plot(
                            xs_gt, y_gt,
                            color="k",
                            linestyle="--",
                            linewidth=1.2,
                            label="gt" if (show_legend or not overlay) else "_nolegend_",
                        )
                        drew_gt = True
            e_note = _g_e_note(label, e_leak_mV=e_leak_mV, globs=globs)
            if e_note is not None:
                ax.set_title(e_note, fontsize=8)
            _style_component_ax(
                ax, _trace_ylabel(panel_ylabel, label),
                legend_fontsize=6 if overlay else 7,
                legend_ncol=1,
                show_legend=show_legend or drew_gt,
            )
    _apply_shared_row_ylim(axes, row_curves, spec)
    _finish_component_figure(fig, title, colors, spec)
    _save_component_figure(
        fig, axes, xlabel="t (ms)",
        out_path=out_path, save_figure=save_figure, spec=spec,
    )


def plot_report(report: dict[str, Any], out_path: str) -> None:
    """Write one multi-panel PNG: component series vs ``t`` in ms."""
    title = (
        f"{report['cell']}  {report['task']}"
        + (f"  {report['spec']}" if report.get("spec") else "")
        + f"  mode={report.get('mode')}  n={report.get('n_nodes')}"
    )
    if report.get("mode") == "hex":
        title += f"  hex=({report['hex']['x']},{report['hex']['y']})"
    _plot_component_reports([report], out_path, title=title)


def plot_reports_overlay(reports: list[dict[str, Any]], out_path: str) -> None:
    """One grid PNG: specs share color per subplot, differ by linestyle."""
    if not reports:
        raise SystemExit("no reports to overlay")
    r0 = reports[0]
    specs_csv = ",".join(str(r["spec"]) for r in reports)
    title = (
        f"{r0['cell']}  {r0['task']}  overlay=[{specs_csv}]"
        f"  mode={r0.get('mode')}  n={r0.get('n_nodes')}"
    )
    _plot_component_reports(reports, out_path, title=title)


def _emit_report(
    report: dict[str, Any],
    *,
    run_dir: str,
    do_print: bool,
    do_plot: bool,
    file_suffix: str = "",
    html: bool = False,
) -> None:
    if do_print:
        print("")
        # Full per-t table only when not plotting (--plot false).
        _print_report(report, print_steps=not do_plot)
    if do_plot:
        out = os.path.join(
            run_dir,
            "cell_dynamics",
            _plot_filename(report, file_suffix=file_suffix, html=html),
        )
        plot_report(report, out)


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def _print_report(report: dict[str, Any], *, print_steps: bool = True) -> None:
    mode = report.get("mode", "?")
    model = report.get("model", "borst")
    kind = report.get("time_window_kind", "t_rel")
    x_key = "t_rel" if kind == "t_rel" else "t"
    hdr = (
        f"cell={report['cell']} model={model} mode={mode} "
        f"n_nodes={report.get('n_nodes', '?')}"
    )
    if mode == "hex":
        hdr += (
            f" node=#{report['node']} hex=({report['hex']['x']},{report['hex']['y']}) "
            f"uv=({report['uv']['u']},{report['uv']['v']})"
        )
    print(hdr)
    print(f"task={report['task']} spec={report.get('spec')}")
    tw = report.get("time_window")
    print(
        f"v_post_d_peak={report['v_post_d_peak']:+.4f} mV "
        f"v_post_d_polarity={report['v_post_d_polarity']}  "
        f"v_post_d_peak_t={report['v_post_d_peak_t']}  "
        f"before_t={report.get('before_t')}  "
        f"peak_drive={report.get('peak_drive')}  "
        f"{kind}={tw[0]}:{tw[1]}"
    )
    print("trained:", "  ".join(f"{k}={v:.6g}" for k, v in report["params"].items()))

    if model == "hp_lp":
        if print_steps:
            print(
                f"\n{x_key}  n  v_post  v_pre  v_post_minus_pre  i_sti "
                "v_in  v_in_exc -v_in_inh  dv_leak  dv_hp"
            )
            for s in report["steps"]:
                v_pre = float(s["v_post"]) - float(s["v_post_minus_pre"])
                print(
                    f"{s[x_key]:4d} {s.get('n_nodes', 1):3d} {s['v_post']:+8.4f} "
                    f"{v_pre:+8.4f} {s['v_post_minus_pre']:+8.4f} "
                    f"{s['i_sti']:+6.3f} {s['v_in']:+6.3f} {s['v_in_exc']:+7.3f} "
                    f"{s['v_in_inh']:+7.3f} {s['dv_leak']:+8.4f} {s['dv_hp']:+7.4f}"
                )
        ps = report.get("peak_step")
        if ps is not None:
            print(f"\nHP/LP terms at peak {x_key}={ps[x_key]}:")
            for name, val in [
                ("v_in", ps["v_in"]), ("v_in_exc", ps["v_in_exc"]), ("-v_in_inh", ps["v_in_inh"]),
                ("v_slow", ps["v_slow"]), ("v_tot", ps["v_tot"]), ("v_hp", ps["v_hp"]),
                ("dv_leak", ps["dv_leak"]), ("dv_hp", ps["dv_hp"]),
            ]:
                print(f"  {name:8s} {val:+9.4f}")
        if "cost" in report and "best_cost" in report:
            print(f"best_cost={report['best_cost']:.4f}  cost={report['cost']:.4f}")
        return

    if print_steps:
        print(
            f"\n{x_key}  n  v_post  v_pre  v_post_minus_pre  i_sti "
            "g_inh  g_Ih_off  g_exc  num_inh  num_exc"
        )
        for s in report["steps"]:
            v_pre = float(s["v_post"]) - float(s["v_post_minus_pre"])
            print(
                f"{s[x_key]:4d} {s.get('n_nodes', 1):3d} {s['v_post']:+8.4f} "
                f"{v_pre:+8.4f} {s['v_post_minus_pre']:+8.4f} "
                f"{s['i_sti']:5.1f} {s['g_inh_nS']:.4f} {s['g_Ih_off_nS']:.4f} "
                f"{s['g_exc_nS']:.4f} {s['num_inh']:+8.2f} {s['num_exc']:+8.2f}"
            )

    ps = report.get("peak_step")
    if ps is not None:
        num = float(ps["num"])
        dt_over_c = float(ps.get("dt_over_c", 0.0))
        print(f"\nNumerator at peak {x_key}={ps[x_key]} (num={num:.2f}):")
        for name, val in [
            ("num_v", ps["num_v"]),
            ("dt_over_c*i_sti", dt_over_c * ps["i_sti"]),
            ("dt_over_c*i_exc", dt_over_c * ps["num_exc"]),
            ("dt_over_c*i_inh", dt_over_c * ps["num_inh"]),
            ("dt_over_c*i_leak", dt_over_c * ps["num_leak"]),
            ("dt_over_c*i_h_on", dt_over_c * ps["num_ihon"]),
            ("dt_over_c*i_h_off", dt_over_c * ps["num_ihoff"]),
        ]:
            pct = 100.0 * val / num if num else 0.0
            print(f"  {name:20s} {val:+9.2f} ({pct:.0f}%)")
    if "cost" in report and "best_cost" in report:
        print(f"best_cost={report['best_cost']:.4f}  cost={report['cost']:.4f}")


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
            f"{cell:6s} {s['v_post_d_peak']:+11.4f} {s['v_post_d_polarity']:>8s} "
            f"{str(s.get('peak_drive')):>8s} "
            f"{b['v_post_d_peak']:+10.4f} {b['v_post_d_polarity']:>8s} "
            f"{str(b.get('peak_drive')):>8s}  {note}"
        )
        sps, bps = s.get("peak_step"), b.get("peak_step")
        if not (sps and bps):
            continue
        model = s.get("model", "borst")
        if model == "hp_lp":
            print(
                f"       spot@peak: v_in_exc={sps['v_in_exc']:+.4f} -v_in_inh={sps['v_in_inh']:+.4f} "
                f"dv_hp={sps['dv_hp']:+.4f} dv_leak={sps['dv_leak']:+.4f} "
                f"v_pre={float(sps['v_post']) - float(sps['v_post_minus_pre']):+.3f}"
            )
            print(
                f"       bar @peak: v_in_exc={bps['v_in_exc']:+.4f} -v_in_inh={bps['v_in_inh']:+.4f} "
                f"dv_hp={bps['dv_hp']:+.4f} dv_leak={bps['dv_leak']:+.4f} "
                f"v_pre={float(bps['v_post']) - float(bps['v_post_minus_pre']):+.3f}"
            )
        else:
            print(
                f"       spot@peak: g_exc={sps['g_exc_nS']:.4f} g_inh={sps['g_inh_nS']:.4f} "
                f"num_exc={sps['num_exc']:+.1f} num_inh={sps['num_inh']:+.1f} "
                f"v_pre={float(sps['v_post']) - float(sps['v_post_minus_pre']):+.3f}"
            )
            print(
                f"       bar @peak: g_exc={bps['g_exc_nS']:.4f} g_inh={bps['g_inh_nS']:.4f} "
                f"num_exc={bps['num_exc']:+.1f} num_inh={bps['num_inh']:+.1f} "
                f"v_pre={float(bps['v_post']) - float(bps['v_post_minus_pre']):+.3f}"
            )




def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_shared_cli(ap, default_run=DEFAULT_RUN_PATH)
    plot_trained.add_plot_timing_arguments(ap)
    plot_trained.add_plot_euler_argument(ap)
    plot_trained.add_param_argument(ap)
    ap.add_argument("--node", type=int, default=None, help="hex-mode node index")
    ap.add_argument(
        "--radius",
        type=int,
        choices=(0, 1),
        default=0,
        help=(
            "spot average Euclidean readout radius (0=stim-on, 1=neighbors); "
            "average mode only (not with --x/--y); PNG gets _radius1 when 1"
        ),
    )
    t_group = ap.add_mutually_exclusive_group()
    t_group.add_argument(
        "--t-rel",
        default=None,
        metavar="START:STOP",
        help=(
            "t-index window relative to |v_post_d| peak (not onset, not abs ms); "
            "mutually exclusive with --ms-shown; default without either: 0..last ms"
        ),
    )
    plot_trained.add_ms_shown_argument(t_group)
    # --ms-shown: absolute aligned ms (spot 0=trial start; pre = 0,ms_pre).
    # --ms-pre/…: stimulus length overrides (rebuild session). Do not confuse.
    ap.add_argument(
        "--plot",
        type=parse_bool,
        default=True,
        metavar="true|false",
        help=(
            "save component figures under {run}/cell_dynamics/ (default: true); "
            "per-t step table prints only when false"
        ),
    )
    ap.add_argument(
        "--html",
        action="store_true",
        help="save interactive plotly HTML (hover x/y) instead of PNG",
    )
    ap.add_argument("--json", action="store_true", help="print JSON to stdout")
    args = ap.parse_args()
    if not args.run:
        args.run = [DEFAULT_RUN_PATH]
    cli = parse_shared_cli(args)

    if args.radius != 0 and not any(t in training.SPOT_TASKS for t in cli.tasks):
        raise SystemExit("--radius requires a spot task")

    hex_mode = False
    if cli.xs is not None and cli.ys is not None:
        if len(cli.xs) != 1 or len(cli.ys) != 1:
            raise SystemExit(
                "hex mode needs exactly one --x and one --y; "
                "omit both for cost-extent averages"
            )
        hex_mode = True
        if len(cli.cells) != 1:
            raise SystemExit("hex mode supports one cell")
    elif cli.xs is not None or cli.ys is not None:
        raise SystemExit("pass both --x and --y for hex mode, or neither for averages")

    if hex_mode and args.radius != 0:
        raise SystemExit(
            "--radius is average-only; omit --x/--y, or omit --radius for hex mode"
        )

    if args.t_rel is not None:
        t_lo, t_hi = _parse_t_range(args.t_rel, flag="--t-rel")
        time_window = TimeWindow(kind="t_rel", start=t_lo, stop=t_hi)
        ms_range = None
        use_ms = False
    elif args.ms_shown is not None:
        try:
            ms_range = plot_trained.parse_ms_shown_range(args.ms_shown)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        use_ms = True
    else:
        ms_range = None  # default: 0 .. last sample
        use_ms = True

    timing_kw = stimulus_timing_kwargs_from_args(args)
    param_edits = plot_trained.parse_param_tokens(args.param)

    for run_i, run_arg in enumerate(args.run):
        run_dir = plot_trained.resolve_run_dir(run_arg)
        _log(f"load_best {run_dir} ...")
        session, z, best_cost = plot_trained.load_best(run_dir)
        session, z, timing_changed = plot_trained.maybe_override_stimulus_timing(
            run_dir=run_dir,
            session=session,
            z=z,
            **timing_kw,
            euler=args.euler,
        )
        file_suffix = (
            plot_trained.stimulus_timing_filename_suffix(
                **timing_changed,
            )
            + plot_trained.euler_filename_suffix(args.euler)
            + plot_trained.param_filename_suffix(param_edits)
        )
        if use_ms:
            lo, hi = (
                (0.0, (int(session.n_t) - 1) * float(session.delta_ms))
                if ms_range is None else ms_range
            )
            time_window = TimeWindow(kind="ms", start=lo, stop=hi)
        schema = list(session.schema)
        z_t = torch.tensor(np.asarray(z, dtype=np.float64), dtype=torch.float64, device=session.device)
        z_t, schema = plot_trained.apply_param_overrides(
            z_t, schema, session, param_edits,
        )
        session = session.with_schema(schema)
        p = training.assign_params(z_t, schema, session.backend)
        cost = float(training.calc_cost(z_t, session).item())

        spot_session_cache: dict[str, object] = {}
        bar_meta_cache: dict[str, tuple] = {}
        spot_by_cell: dict[str, dict[str, Any]] = {}
        bar_by_cell: dict[str, dict[str, Any]] = {}
        all_reports: list[dict[str, Any]] = []

        if not args.json:
            _log(f"== RUN {run_i}: {run_dir} ==")
            _log(
                f"best_cost={best_cost:.6g}  cost={cost:.6g}  "
                f"mode={'hex' if hex_mode else 'average'}  "
                f"radius={args.radius}  "
                f"{time_window.kind}={time_window.start}:{time_window.stop}"
            )

        for task in cli.tasks:
            if task in training.SPOT_TASKS:
                if task not in spot_session_cache:
                    spot_session_cache[task] = plot_trained.session_for_task(
                        session, task,
                    )
                session_one = spot_session_cache[task]
                if hex_mode:
                    hx = cli.xs[0]
                    hy = cli.ys[0]
                    _log(
                        f"component forward {task} "
                        f"(spot hex=({hx},{hy}); batched) ..."
                    )
                    reports = analyze_spot_hex(
                        session_one,
                        p=p,
                        cell=cli.cells[0],
                        task=task,
                        at_x=float(hx),
                        at_y=float(hy),
                        node=args.node,
                        time_window=time_window,
                    )
                else:
                    _log(
                        f"component forward {task} "
                        f"(spot radius={args.radius}; batched) ..."
                    )
                    reports = analyze_spot_average(
                        session_one,
                        p=p,
                        cells=cli.cells,
                        task=task,
                        time_window=time_window,
                        radius=args.radius,
                    )
                for cell, rep in reports.items():
                    rep["best_cost"] = best_cost
                    rep["cost"] = cost
                    spot_by_cell[cell] = rep
                    all_reports.append(rep)
                    _emit_report(
                        rep,
                        run_dir=run_dir,
                        do_print=not args.json,
                        do_plot=args.plot,
                        file_suffix=file_suffix,
                        html=args.html,
                    )
            else:
                hx = cli.xs[0] if hex_mode else None
                hy = cli.ys[0] if hex_mode else None
                if task not in bar_meta_cache:
                    bar_meta_cache[task] = _bar_meta(session, task)
                specs, grids = bar_meta_cache[task]
                cells_bar = [cli.cells[0]] if hex_mode else cli.cells
                specs_ordered = _bar_specs_requested(
                    session, task, cells_bar, cli.specs_req,
                    specs=specs, grids=grids,
                )
                multi_spec_plot = args.plot and len(specs_ordered) > 1
                if hex_mode:
                    _log(
                        f"component forward {task} specs={specs_ordered} "
                        f"hex=({hx},{hy}) (batched, no full forward) ..."
                    )
                    reports_by_spec = analyze_bar_hex(
                        session,
                        p=p,
                        cell=cells_bar[0],
                        task=task,
                        spec_names=specs_ordered,
                        at_x=float(hx),
                        at_y=float(hy),
                        node=args.node,
                        time_window=time_window,
                        specs=specs,
                        grids=grids,
                    )
                else:
                    _log(
                        f"component forward {task} specs={specs_ordered} "
                        f"(batched, no full forward) ..."
                    )
                    reports_by_spec = analyze_bar_average(
                        session,
                        p=p,
                        cells=cells_bar,
                        task=task,
                        spec_names=specs_ordered,
                        time_window=time_window,
                        specs=specs,
                        grids=grids,
                    )
                overlay_by_cell: dict[str, list[dict[str, Any]]] = {
                    c: [] for c in cells_bar
                }
                for spec in specs_ordered:
                    for c, rep in reports_by_spec[spec].items():
                        rep["best_cost"] = best_cost
                        rep["cost"] = cost
                        bar_by_cell[c] = rep
                        all_reports.append(rep)
                        if multi_spec_plot:
                            overlay_by_cell[c].append(rep)
                        _emit_report(
                            rep,
                            run_dir=run_dir,
                            do_print=not args.json,
                            do_plot=args.plot and not multi_spec_plot,
                            file_suffix=file_suffix,
                            html=args.html,
                        )
                if multi_spec_plot:
                    for c in cells_bar:
                        reps = overlay_by_cell[c]
                        out = os.path.join(
                            run_dir,
                            "cell_dynamics",
                            _overlay_plot_filename(
                                reps, file_suffix=file_suffix, html=args.html,
                            ),
                        )
                        plot_reports_overlay(reps, out)

        if not args.json and spot_by_cell and bar_by_cell:
            _print_polarity_compare(spot_by_cell, bar_by_cell)

        if args.json:
            print(json.dumps(
                {
                    "run": run_dir,
                    "best_cost": best_cost,
                    "cost": cost,
                    "reports": all_reports,
                },
                indent=2,
            ))


if __name__ == "__main__":
    main()
