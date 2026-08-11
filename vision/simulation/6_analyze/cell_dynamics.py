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
import train
from param_defaults import DEFAULT_RUN_PATH
import figure.plot as plot_trained
from figure.gt import contrast_for_task
from figure.spot import pack_spot_cost_radii, resolve_spot_gt_cubes
from figure.util import (
    filter_plot_token,
    gt_affine_scalars_for_cell,
    plot_std_band,
    session_filter_plot_token,
)
from import_bootstrap import parse_bool, parse_comma_list
from neuron.filter_ca import filter_ca
from network.construction import hex2gt
from task.moving_bar.gt import (
    bar_specs_for_session,
    filter_requested_specs,
    moving_bar_nodes_on_hexes,
    moving_bar_specs_by_cell,
    moving_bar_session_t0_grids,
)
from task.moving_bar.input import (
    filter_sti_hexes,
    moving_bar_cost_hexes,
)
from task.spot.readout import build_spot_center_readout
from task.spot.input import (
    spot_from_opts,
    spot_stimulus_batches,
)
from train.cli import parse_tasks, stimulus_timing_kwargs_from_args

__doc__ = """Borst / hp_lp v component analysis.

Consumers (CLI or ``import analyze.cell_dynamics``) must reuse this module's
forward helpers. Do not re-implement spot/bar readout + step loops in test/.

Time axis (read this before ``--ms-shown`` / ``TimeWindow``)
------------------------------------------------------------
Two *different* knobs — do not mix them:

1. **Stimulus length** (``--ms-pre`` / ``--ms-spot`` / ``--ms-response`` /
   ``--ms-post`` / ``--delta-ms``): rebuilds the session stimulus (via
   ``figure.plot.maybe_override_stimulus_timing``). Unset = keep the run's
   train opts. These change *how long* pre/spot/response *are*, not which
   slice of an existing trace you plot.

2. **Analyze / plot window** (``--ms-shown START,STOP`` or ``--t-rel START:STOP``,
   mutually exclusive): which inclusive slice of the forward to accumulate and
   report. Default if both omitted: absolute ms ``0`` .. last sample.

``--ms-shown`` is **absolute aligned ms**, never "ms before onset":

* **spot**: aligned ``t = 0`` is trial start. Stimulus onset is at
  ``t_onset = t_from_ms(ms_pre, delta_ms=delta_ms_pre)`` (e.g. ``ms_pre=1000``,
  ``delta_ms_pre=5`` → ``t_onset=200`` ↔ **1000 ms**). Pre-stimulus is therefore
  ``--ms-shown 0,1000`` (or ``0,ms_pre``), **not** ``-1000,0``.
  Negative START is wrong for spot (aligned index goes negative; accumulate window
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
* Load run: ``figure.plot.load_best`` + ``assign_params``; do not invent a
  second forward path.

CLI
---
``CELL,...`` ``--run`` ``--task`` ``--spec`` ``--x`` ``--y``. Pass comma lists in
one process (do not re-invoke once per cell/spec).

Per ``--run``: one ``load_best``; one batched v component forward per distinct task.

* Omit ``--x`` / ``--y``: cost-radius **average** (optional ``--radius 0|1``).
* Exactly one ``--x`` and one ``--y``: **hex** (spot or moving_bar; one cell).
  Incompatible with ``--radius`` (hex is stim-on only).
* Multiple x/y: rejected.

``--plot true|false``: PNGs under ``{run}/cell_dynamics/`` (default true).
  With ``filter=ca``, first plot row is ``v_post`` / ``v_ca`` / ``ca``; with
  ``filter=none``, first row is ``v_post`` only (no Ca). Train GT is named
  ``gt_v`` or ``gt_ca`` from the run's train ``filter`` and overlays only on
  ``v_post`` or ``ca`` respectively (not the analyze ``--filter``). Per-t step
  table prints only when ``--plot false``. With analyze ``filter=ca``, the first
  three table columns are ``ca`` / ``ca_pre`` / ``ca_post_minus_pre``
  (else ``v_post`` / ``v_pre`` / ``v_post_minus_pre``); membrane component
  columns stay ``v_*``.
``--radius 0|1``: spot average Euclidean readout radius (default 0 = stim-on hex; 1 = neighbors).
  Average only; PNGs for ``--radius 1`` get ``_radius1`` in the filename.
``--param NAME=VALUE`` / ``NAME.NODE=VALUE``: via ``figure.plot`` — overwrite
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
        return train.t_from_ms(self.start, delta_ms=delta_ms)

    def forward_t_stop(self, *, delta_ms: float) -> int | None:
        if self.kind != "ms":
            return None
        return train.t_from_ms(self.stop, delta_ms=delta_ms)

    def aligned_n_t(self, n_t: int, *, delta_ms: float) -> int:
        """Buffer length for spot (``t0=0``): indices ``0 .. stop`` inclusive.

        For ``kind="ms"``, length is ``t_from_ms(stop) + 1``. ``stop`` must be
        non-negative for a useful buffer (spot pre → ``stop=ms_pre``).
        """
        if self.kind != "ms":
            return n_t
        return train.t_from_ms(self.stop, delta_ms=delta_ms) + 1


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
    for task in tasks:
        if task not in train.SPOT_TASKS and task not in train.MOVING_BAR_TASKS:
            raise SystemExit(
                f"unsupported task {task!r}; expected spot_* or moving_bar_* "
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
            ("g_h_nS", "g_h"),
            ("g_h_rev_nS", "g_h_rev"),
        ],
    ),
    (
        "i (pA)",
        [
            ("num_v", "num_v"),
            ("num_exc", "i_exc"),
            ("num_inh", "i_inh"),
            ("num_leak", "i_leak"),
            ("num_i_h", "i_h"),
            ("num_i_h_rev", "i_h_rev"),
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
        "v_syn (mV)",
        [
            ("v_syn", "v_syn"),
            ("v_syn_exc", "v_syn_exc"),
            ("v_syn_inh", "-v_syn_inh"),
            ("v_sti", "v_sti"),
        ],
    ),
    (
        "HP state",
        [
            ("v_in", "v_in"),
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

# First plot row when ``filter=ca`` (replaces the single ``v_post`` series).
_CA_PLOT_ROW0: tuple[str, list[tuple[str, str]]] = (
    "v / Ca",
    [
        ("v_post", "v_post"),
        ("v_ca", "v_ca"),
        ("ca", "ca"),
    ],
)

_BORST_COMPONENT_KEYS = (
    "v_pre_d", "v_abs", "i_sti", "g_exc", "g_inh", "g_h", "g_h_rev",
    "num_exc", "num_inh", "num_leak", "num_i_h", "num_i_h_rev", "num_v",
    "num", "den",
)
_HP_LP_COMPONENT_KEYS = (
    "v_pre_d", "v_abs", "i_sti", "v_sti", "v_syn", "v_syn_exc", "v_syn_inh",
    "v_slow", "v_in", "v_hp", "dv_leak", "dv_hp",
)

_BORST_PLOT_KEY_COMPONENT: dict[str, str | None] = {
    "v_post": "v_abs",
    "v_ca": None,
    "ca": None,
    "g_exc_nS": "g_exc",
    "g_inh_nS": "g_inh",
    "g_leak_nS": None,
    "g_h_nS": "g_h",
    "g_h_rev_nS": "g_h_rev",
    "num_v": "num_v",
    "num_exc": "num_exc",
    "num_inh": "num_inh",
    "num_leak": "num_leak",
    "num_i_h": "num_i_h",
    "num_i_h_rev": "num_i_h_rev",
    "i_sti": "i_sti",
    "num_over_den": "v_abs",
    "num": "num",
    "den": "den",
}
_HP_LP_PLOT_KEY_COMPONENT: dict[str, str | None] = {
    "v_post": "v_abs",
    "v_ca": None,
    "ca": None,
    "v_syn": "v_syn",
    "v_syn_exc": "v_syn_exc",
    "v_syn_inh": "v_syn_inh",
    "v_sti": "v_sti",
    "v_slow": "v_slow",
    "v_in": "v_in",
    "v_hp": "v_hp",
    "dv_leak": "dv_leak",
    "dv_hp": "dv_hp",
    "dv_leak_plus_hp": None,
}

_BORST_FORMULA_G_IMPLICIT: list[tuple[str, str | None]] = [
    ("v_post", "v_post"),
    (" = (v_pre + ", None),
    ("dt_over_c", None), ("·(i_sti + ", None),
    ("e_exc·", None), ("g_exc", "g_exc"), (" + ", None),
    ("e_inh·", None), ("g_inh", "g_inh"), (" + ", None),
    ("E_leak·", None), ("g_leak", "g_leak"), (" + ", None),
    ("E_h·", None), ("g_h", "g_h"), (" + ", None),
    ("E_h_rev·", None), ("g_h_rev", "g_h_rev"),
    (")) / (1 + ", None),
    ("dt_over_c", None), ("·(", None),
    ("g_exc", "g_exc"), (" + ", None),
    ("g_inh", "g_inh"), (" + ", None),
    ("g_h", "g_h"), (" + ", None),
    ("g_h_rev", "g_h_rev"), (" + ", None),
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
    ("i_h", "i_h"), (" + ", None),
    ("i_h_rev", "i_h_rev"),
    (")) / (", None),
    ("den", "den"),
    (")", None),
]

_BORST_FORMULA_G_EXPLICIT: list[tuple[str, str | None]] = [
    ("v_post", "v_post"),
    (" = v_pre + ", None),
    ("dt_over_c", None), ("·(i_sti + ", None),
    ("e_exc·", None), ("g_exc", "g_exc"), (" + ", None),
    ("e_inh·", None), ("g_inh", "g_inh"), (" + ", None),
    ("E_leak·", None), ("g_leak", "g_leak"), (" + ", None),
    ("E_h·", None), ("g_h", "g_h"), (" + ", None),
    ("E_h_rev·", None), ("g_h_rev", "g_h_rev"),
    (" − (", None),
    ("g_exc", "g_exc"), (" + ", None),
    ("g_inh", "g_inh"), (" + ", None),
    ("g_h", "g_h"), (" + ", None),
    ("g_h_rev", "g_h_rev"), (" + ", None),
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
    ("i_h", "i_h"), (" + ", None),
    ("i_h_rev", "i_h_rev"),
    (")", None),
]

_HP_LP_FORMULA_G_IMPLICIT: list[tuple[str, str | None]] = [
    ("v_post", "v_post"),
    (" = (v_pre + (dt/τ_lp)·(e_leak + G·(e_leak + ", None),
    ("v_syn", "v_syn"), (" + ", None),
    ("i_sti/g_leak", "v_sti"), (" − ", None),
    ("v_slow", "v_slow"), ("))) / (1 + dt/τ_lp)", None),
]

_HP_LP_FORMULA_I_IMPLICIT: list[tuple[str, str | None]] = [
    ("v_post", "v_post"), (" = v_pre + ", None),
    ("dv_leak", "dv_leak"), (" + ", None),
    ("dv_hp", "dv_hp"),
]

_HP_LP_FORMULA_G_EXPLICIT: list[tuple[str, str | None]] = [
    ("v_post", "v_post"),
    (" = v_pre + (dt/τ_lp)[-(v_pre−e_leak) + G·(e_leak + ", None),
    ("v_syn", "v_syn"), (" + ", None),
    ("i_sti/g_leak", "v_sti"), (" − ", None),
    ("v_slow", "v_slow"), (")]", None),
]

_HP_LP_FORMULA_I_EXPLICIT: list[tuple[str, str | None]] = [
    ("v_post", "v_post"), (" = v_pre + ", None),
    ("dv_leak", "dv_leak"), (" + ", None),
    ("dv_hp", "dv_hp"),
]

_BLACK_TRACE_LABELS = frozenset({"num", "den", "v_in", "v_slow", "v_hp", "dv_hp"})
# Label → reuse another label's plot color (same row cycle).
_TRACE_COLOR_MATCH: dict[str, str] = {
    "dv_leak+dv_hp": "dv_leak",
    "v_ca": "v_post",
    "ca": "v_post",
}
# Report GT key → plot panel key (train GT kind; never mix).
_GT_PLOT_PANEL: dict[str, str] = {
    "gt_v": "v_post",
    "gt_ca": "ca",
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
    def n_col(self) -> int:
        return max(len(series) for _, series in self.plot_panels)


def _component_spec(model: str, euler: str, *, filter: str = "v") -> _ComponentSpec:
    """Build plot/component spec. ``filter`` is plot token ``v``|``ca`` (row-0 Ca cols)."""
    euler = train.expand_euler(euler)
    use_ca = str(filter) == "ca"
    if model == "borst":
        if euler == "implicit":
            formula_g, formula_i = _BORST_FORMULA_G_IMPLICIT, _BORST_FORMULA_I_IMPLICIT
        else:
            formula_g, formula_i = _BORST_FORMULA_G_EXPLICIT, _BORST_FORMULA_I_EXPLICIT
        panels = list(_BORST_PLOT_PANELS)
        if use_ca:
            panels[0] = _CA_PLOT_ROW0
        return _ComponentSpec(
            model="borst",
            keys=_BORST_COMPONENT_KEYS,
            plot_panels=panels,
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
        panels = list(_HP_LP_PLOT_PANELS)
        if use_ca:
            panels[0] = _CA_PLOT_ROW0
        return _ComponentSpec(
            model="hp_lp",
            keys=_HP_LP_COMPONENT_KEYS,
            plot_panels=panels,
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


def _plot_trace_colors(colors: list[str], spec: _ComponentSpec) -> dict[str, str]:
    """Map trace legend label → subplot color (hex index within its row)."""
    out: dict[str, str] = {}
    for _panel_ylabel, series in spec.plot_panels:
        for series_idx, (_key, label) in enumerate(series):
            if label in _BLACK_TRACE_LABELS:
                out[label] = "0.0"
            else:
                out[label] = colors[series_idx % len(colors)]
    for label, src in _TRACE_COLOR_MATCH.items():
        if label in out and src in out:
            out[label] = out[src]
    return out


def _g_e_note(label: str, *, e_leak_mV: float, globs: dict[str, Any], params: dict[str, Any] | None = None) -> str | None:
    """Reversal annotation for a conductance subplot (``e_exc=+10 mV`` …)."""
    if label == "g_leak":
        return f"E_leak={e_leak_mV:+g} mV"
    notes = {
        "g_exc": "e_exc",
        "g_inh": "e_inh",
        "g_h": "e_h",
        "g_h_rev": "e_h_rev",
    }
    gkey = notes.get(label)
    if gkey is None:
        return None
    src = globs
    if gkey not in src and params is not None and gkey in params:
        src = params
    if gkey not in src:
        return None
    pretty = {"e_h": "E_h", "e_h_rev": "E_h_rev"}.get(gkey, gkey)
    return f"{pretty}={src[gkey]:+g} mV"


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


def _prepare_drive(session, params, i_sti: torch.Tensor) -> torch.Tensor:
    """Model ``prepare_i_sti`` + spot ``a_sti_radius`` on a ``(B, T, N)`` pack ``i_sti``."""
    from neuron.forward import inject_a_sti_radius

    pack = session.primary_readout
    drive = _model_driver(session).prepare_i_sti(session, params, i_sti, pack)
    return inject_a_sti_radius(drive, params, pack)


def _equilibrate(session, params, i_sti_batch: torch.Tensor, t_onset: int):
    """Equilibrate to ``t_onset``; returns ``(v, state)`` where state is model-specific."""
    _component_spec(session.model, session.euler)  # validate early
    n_batch, n_t, n_nodes = i_sti_batch.shape
    drv = _model_driver(session)
    state, v = drv.pre_steady(session, params, n_batch, i_sti=i_sti_batch)
    for t in range(1, min(t_onset, n_t)):
        state, v = drv.step(
            state, v, params, i_sti_batch[:, t - 1], session,
            delta_ms=train.step_delta_ms(session, t, t_onset),
        )
    return v, state


def _component_at_nodes_borst(
    v_pre,
    v_post,
    g_exc,
    g_inh,
    g_h,
    g_h_rev,
    sig_t,
    e_leak,
    nodes: np.ndarray,
    v_onset: np.ndarray,
    *,
    batch_idx: int = 0,
    delta_ms: float,
    cap: float,
    g_leak: float,
    e_exc: float,
    e_inh: float,
    e_h: float,
    euler: str,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Slice node component from a completed ``update_v(..., return_component=True)`` step."""
    nodes = np.asarray(nodes, dtype=np.int64)
    with torch.no_grad():
        node_idx = torch.as_tensor(nodes, device=v_pre.device, dtype=torch.long)
        sig_active = (
            sig_t[batch_idx, node_idx] if sig_t.dim() > 1 else sig_t[node_idx]
        )
        packed = torch.stack(
            (
                v_pre[batch_idx, node_idx],
                g_exc[batch_idx, node_idx],
                g_inh[batch_idx, node_idx],
                g_h[batch_idx, node_idx],
                g_h_rev[batch_idx, node_idx],
                sig_active,
                e_leak[node_idx],
                v_post[batch_idx, node_idx],
            ),
            dim=0,
        ).detach().cpu().numpy()
        v_pre_np = packed[0]
        ref = (
            v_onset[batch_idx, nodes] if np.ndim(v_onset) == 2 else v_onset[nodes]
        )
        terms = train.v_component_from_g(
            v_pre_np, packed[1], packed[2], packed[3], packed[4], packed[5], packed[6],
            delta_ms=delta_ms, cap=cap, g_leak=g_leak,
            e_exc=e_exc, e_inh=e_inh, e_h=e_h,
            euler=euler,
        )
        v_abs = packed[7]
        component = {
            "v_pre_d": v_pre_np - ref,
            "v_abs": v_abs,
            "g_exc": packed[1],
            "g_inh": packed[2],
            "g_h": packed[3],
            "g_h_rev": packed[4],
            **terms,
            "num": terms["num"],
        }
        v_post_minus_pre_active = v_abs - v_pre_np
    return component, v_post_minus_pre_active


def _component_at_nodes_hp_lp(
    v_pre,
    v_post,
    component_t: dict[str, torch.Tensor],
    nodes: np.ndarray,
    v_onset: np.ndarray,
    *,
    batch_idx: int = 0,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Slice hp_lp ODE component tensors at ``nodes``."""
    nodes = np.asarray(nodes, dtype=np.int64)
    with torch.no_grad():
        node_idx = torch.as_tensor(nodes, device=v_pre.device, dtype=torch.long)
        v_pre_np = v_pre[batch_idx, node_idx].detach().cpu().numpy()
        v_abs = v_post[batch_idx, node_idx].detach().cpu().numpy()
        ref = (
            v_onset[batch_idx, nodes] if np.ndim(v_onset) == 2 else v_onset[nodes]
        )
        component = {
            "v_pre_d": v_pre_np - ref,
            "v_abs": v_abs,
        }
        for component_key in (
            "i_sti", "v_sti", "v_syn", "v_syn_exc", "v_syn_inh", "v_slow", "v_in", "v_hp",
            "dv_leak", "dv_hp",
        ):
            component_arr = component_t[component_key]
            component[component_key] = (
                component_arr[batch_idx, node_idx]
                if component_arr.dim() > 1 else component_arr[node_idx]
            ).detach().cpu().numpy()
        v_post_minus_pre_active = v_abs - v_pre_np
    return component, v_post_minus_pre_active


def _log(msg: str) -> None:
    print(msg, flush=True)


def _component_matrix(component: dict[str, np.ndarray], keys: tuple[str, ...]) -> np.ndarray:
    """Stack component keys to ``(n_nodes, n_keys)`` for vectorized accumulate."""
    return np.column_stack([component[component_key] for component_key in keys])


def _sums_dict_from_vec(vec: np.ndarray, keys: tuple[str, ...]) -> dict[str, float]:
    return {
        component_key: float(vec[key_idx])
        for key_idx, component_key in enumerate(keys)
    }


def _std_from_sum_and_sum_sq(sum_: float, sum_sq: float, n_nodes: int) -> float:
    """Match ``figure.util.std_from_traces`` (population std)."""
    if n_nodes <= 1:
        return 0.0
    mean = sum_ / n_nodes
    var = sum_sq / n_nodes - mean * mean
    if var <= 0.0:
        return 0.0
    return float(np.sqrt(var))


def _step_std(
    sums: dict[str, float], sum_sqs: dict[str, float], n_nodes: int, spec: _ComponentSpec,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for plot_key, component_key in spec.plot_key_component.items():
        if component_key is None:
            out[plot_key] = 0.0
        else:
            out[plot_key] = _std_from_sum_and_sum_sq(
                sums[component_key], sum_sqs[component_key], n_nodes,
            )
    return out


def _step_from_sums(
    *,
    t: int,
    t_rel: int,
    ti: int,
    v_post_val: float,
    sums: dict[str, float],
    sum_sqs: dict[str, float],
    v_post_minus_pre_sum: float,
    n_nodes: int,
    spec: _ComponentSpec,
    g_leak: float = 0.0,
    dt_over_c: float = 0.0,
) -> dict[str, Any]:
    """One step dict from per-key sums over ``n_nodes`` nodes."""
    if n_nodes <= 0:
        raise ValueError("empty node set for mean component")
    base = {
        "t": t,
        "t_rel": t_rel,
        "ti": ti,
        "v_post": float(v_post_val),
        "v_pre_d": sums["v_pre_d"] / n_nodes,
        "v_post_minus_pre": v_post_minus_pre_sum / n_nodes,
        "i_sti": sums["i_sti"] / n_nodes,
        "std": _step_std(sums, sum_sqs, n_nodes, spec),
        "n_nodes": n_nodes,
    }
    if spec.model == "borst":
        num = sums["num"] / n_nodes
        den = sums["den"] / n_nodes
        base.update({
            "g_exc_nS": sums["g_exc"] / n_nodes,
            "g_inh_nS": sums["g_inh"] / n_nodes,
            "g_leak_nS": float(g_leak),
            "dt_over_c": float(dt_over_c),
            "g_h_nS": sums["g_h"] / n_nodes,
            "g_h_rev_nS": sums["g_h_rev"] / n_nodes,
            "num_exc": sums["num_exc"] / n_nodes,
            "num_inh": sums["num_inh"] / n_nodes,
            "num_leak": sums["num_leak"] / n_nodes,
            "num_i_h": sums["num_i_h"] / n_nodes,
            "num_i_h_rev": sums["num_i_h_rev"] / n_nodes,
            "num_v": sums["num_v"] / n_nodes,
            "num": num,
            "den": den,
            "num_over_den": sums["v_abs"] / n_nodes,
        })
        return base
    base.update({
        "v_syn": sums["v_syn"] / n_nodes,
        "v_syn_exc": sums["v_syn_exc"] / n_nodes,
        "v_syn_inh": -sums["v_syn_inh"] / n_nodes,
        "v_sti": sums["v_sti"] / n_nodes,
        "v_slow": sums["v_slow"] / n_nodes,
        "v_in": sums["v_in"] / n_nodes,
        "v_hp": sums["v_hp"] / n_nodes,
        "dv_leak": sums["dv_leak"] / n_nodes,
        "dv_hp": sums["dv_hp"] / n_nodes,
        "dv_leak_plus_hp": (sums["dv_leak"] + sums["dv_hp"]) / n_nodes,
    })
    return base


@dataclass
class _ComponentForwardBatch:
    """One i_sti batch row for the shared full-T component forward."""

    all_nodes: np.ndarray
    node_t0: np.ndarray
    n_t_aligned: int
    cell_from_node: dict[int, str]
    nodes_by_cell: dict[str, np.ndarray]


@dataclass
class _ComponentForwardSums:
    sums: list[dict[str, np.ndarray]]
    sum_sqs: list[dict[str, np.ndarray]]
    n_nodes: list[dict[str, np.ndarray]]
    v_post_minus_pre_sums: list[dict[str, np.ndarray]]
    spec: _ComponentSpec
    v_ca_sums: list[dict[str, np.ndarray]] | None = None
    ca_sums: list[dict[str, np.ndarray]] | None = None
    ca_pre_sums: list[dict[str, np.ndarray]] | None = None


def _forward_component(
    session,
    params,
    i_sti: torch.Tensor,
    batches: list[_ComponentForwardBatch],
    cells: list[str],
    *,
    t_start: int | None = None,
    t_stop: int | None = None,
) -> _ComponentForwardSums:
    """Full-T step from t=0 (same loop as ``forward_full``); accumulate component.

    Shared by bar/spot average and bar/spot hex.
    ``v_onset`` matches ``forward_full`` (``v`` at ``t_onset - 1``). Aligned index
    ``t = t_global - node_t0``. v_post is mean absolute ``v_abs``; STD uses sum /
    sum_sqs like ``std_from_traces``. When ``filter=ca``, also track ``v_ca`` /
    ``ca`` via ``v_ca_from_v`` + ``filter_ca`` (same as ``forward_ca``).

    If ``t_start``/``t_stop`` are set (from ``--ms-shown`` via ``t_from_ms``), only
    accumulate inside that inclusive aligned window; cheap steps outside it;
    break after every node has passed ``t_stop``.
    """
    if not batches:
        raise SystemExit("component forward requires at least one batch")
    if (t_start is None) ^ (t_stop is None):
        raise SystemExit("t_start and t_stop must both be set or both omitted")
    if t_start is not None and t_start > t_stop:
        raise SystemExit(f"t_start={t_start} > t_stop={t_stop}")
    n_batch, n_t, n_nodes = i_sti.shape
    if n_batch != len(batches):
        raise SystemExit(f"i_sti n_batch={n_batch} != len(batches)={len(batches)}")

    spec = _component_spec(session.model, session.euler)
    drive = _prepare_drive(session, params, i_sti)
    t_onset = train.pack_t_onset(session.primary_readout)

    t_last: int | None = None
    if t_stop is not None:
        t_last = max(int(plan.node_t0.max()) + int(t_stop) for plan in batches)

    # Same ref as forward_full: v at t_onset-1, then restart so pre is stepped+accumulated.
    v_at_onset, _ = _equilibrate(session, params, drive, t_onset)
    v_onset = v_at_onset.detach().cpu().numpy().copy()
    n_keys = spec.n_keys
    drv = _model_driver(session)
    state, v = drv.pre_steady(session, params, n_batch, i_sti=drive)

    use_ca = session_filter_plot_token(session) == "ca"
    ca = train.v_ca_from_v(v, params, session) if use_ca else None
    tau_ca = (
        torch.clamp(params["tau_ca"], min=float(session.delta_ms)) if use_ca else None
    )

    batch_sums: list[dict[str, np.ndarray]] = []
    batch_sum_sqs: list[dict[str, np.ndarray]] = []
    batch_n_nodes: list[dict[str, np.ndarray]] = []
    batch_v_post_minus_pre_sums: list[dict[str, np.ndarray]] = []
    batch_v_ca_sums: list[dict[str, np.ndarray]] | None = [] if use_ca else None
    batch_ca_sums: list[dict[str, np.ndarray]] | None = [] if use_ca else None
    batch_ca_pre_sums: list[dict[str, np.ndarray]] | None = [] if use_ca else None
    for plan in batches:
        n_t_aligned = plan.n_t_aligned
        batch_sums.append({cell: np.zeros((n_t_aligned, n_keys), dtype=float) for cell in cells})
        batch_sum_sqs.append({cell: np.zeros((n_t_aligned, n_keys), dtype=float) for cell in cells})
        batch_n_nodes.append({cell: np.zeros(n_t_aligned, dtype=np.int64) for cell in cells})
        batch_v_post_minus_pre_sums.append({cell: np.zeros(n_t_aligned, dtype=float) for cell in cells})
        if use_ca:
            batch_v_ca_sums.append({cell: np.zeros(n_t_aligned, dtype=float) for cell in cells})
            batch_ca_sums.append({cell: np.zeros(n_t_aligned, dtype=float) for cell in cells})
            batch_ca_pre_sums.append({cell: np.zeros(n_t_aligned, dtype=float) for cell in cells})

    node_lookups = [_node_cell_lookup(plan, cells) for plan in batches]

    for t_global in range(1, n_t):
        if t_last is not None and t_global > t_last:
            break
        sig_t = drive[:, t_global - 1]
        actives: list[tuple[np.ndarray, np.ndarray] | None] = []
        need_component = False
        for plan in batches:
            all_nodes = plan.all_nodes
            node_aligned_t = t_global - plan.node_t0
            in_aligned_t = (node_aligned_t >= 0) & (node_aligned_t < plan.n_t_aligned)
            if t_start is not None:
                in_aligned_t = (
                    in_aligned_t
                    & (node_aligned_t >= t_start)
                    & (node_aligned_t <= t_stop)
                )
            if np.any(in_aligned_t):
                need_component = True
                actives.append((
                    all_nodes[in_aligned_t], node_aligned_t[in_aligned_t].astype(np.int64),
                ))
            else:
                actives.append(None)

        step_dt = train.step_delta_ms(session, t_global, t_onset)
        if not need_component:
            state, v = drv.step(
                state, v, params, sig_t, session,
                delta_ms=step_dt,
            )
            if ca is not None:
                ca = filter_ca(
                    ca, train.v_ca_from_v(v, params, session),
                    delta_ms=step_dt, tau_ca=tau_ca,
                )
            continue

        with torch.no_grad():
            v_pre = v
            ca_pre = ca
            if spec.model == "borst":
                state, v, (g_exc, g_inh, g_h, g_h_rev) = drv.step(
                    state, v, params, sig_t, session,
                    delta_ms=step_dt, return_component=True,
                )
            else:
                state, v, component_t = drv.step(
                    state, v, params, sig_t, session,
                    delta_ms=step_dt, return_component=True,
                )
            v_ca = None
            if ca is not None:
                v_ca = train.v_ca_from_v(v, params, session)
                ca = filter_ca(
                    ca_pre, v_ca, delta_ms=step_dt, tau_ca=tau_ca,
                )

        for batch_idx, plan in enumerate(batches):
            active_pack = actives[batch_idx]
            if active_pack is None:
                continue
            active_node_idx, active_t = active_pack
            if spec.model == "borst":
                component, v_post_minus_pre_active = _component_at_nodes_borst(
                    v_pre, v, g_exc, g_inh, g_h, g_h_rev, sig_t,
                    params["e_leak"], active_node_idx, v_onset, batch_idx=batch_idx,
                    delta_ms=step_dt, cap=session.cap, g_leak=session.g_leak,
                    e_exc=session.e_exc, e_inh=session.e_inh, e_h=session.e_h,
                    euler=session.euler,
                )
            else:
                component, v_post_minus_pre_active = _component_at_nodes_hp_lp(
                    v_pre, v, component_t, active_node_idx, v_onset, batch_idx=batch_idx,
                )
            component_mat = _component_matrix(component, spec.keys)
            lookup = node_lookups[batch_idx]
            tags = lookup[active_node_idx]
            v_ca_active = ca_post_active = ca_pre_active = None
            if ca is not None:
                active_node_idx_t = torch.as_tensor(
                    active_node_idx, device=ca.device, dtype=torch.long,
                )
                v_ca_active = v_ca[batch_idx, active_node_idx_t].detach().cpu().numpy()
                ca_post_active = ca[batch_idx, active_node_idx_t].detach().cpu().numpy()
                ca_pre_active = ca_pre[batch_idx, active_node_idx_t].detach().cpu().numpy()
            for cell_idx, cell in enumerate(cells):
                mask = tags == cell_idx
                if not np.any(mask):
                    continue
                ts = active_t[mask]
                chunk = component_mat[mask]
                np.add.at(batch_sums[batch_idx][cell], ts, chunk)
                np.add.at(batch_sum_sqs[batch_idx][cell], ts, chunk * chunk)
                np.add.at(
                    batch_v_post_minus_pre_sums[batch_idx][cell], ts, v_post_minus_pre_active[mask],
                )
                np.add.at(batch_n_nodes[batch_idx][cell], ts, 1)
                if v_ca_active is not None:
                    np.add.at(batch_v_ca_sums[batch_idx][cell], ts, v_ca_active[mask])
                    np.add.at(batch_ca_sums[batch_idx][cell], ts, ca_post_active[mask])
                    np.add.at(batch_ca_pre_sums[batch_idx][cell], ts, ca_pre_active[mask])

    return _ComponentForwardSums(
        sums=batch_sums,
        sum_sqs=batch_sum_sqs,
        n_nodes=batch_n_nodes,
        v_post_minus_pre_sums=batch_v_post_minus_pre_sums,
        spec=spec,
        v_ca_sums=batch_v_ca_sums,
        ca_sums=batch_ca_sums,
        ca_pre_sums=batch_ca_pre_sums,
    )


def _v_post_from_sums(
    sums: np.ndarray, n_nodes: np.ndarray, spec: _ComponentSpec,
) -> np.ndarray:
    """Mean absolute v_post from accumulated ``v_abs``."""
    n_nodes_div = np.maximum(n_nodes, 1)
    v_post = sums[:, spec.i_v_abs] / n_nodes_div
    v_post[n_nodes == 0] = 0.0
    return v_post


def _v_post_d_from_sums(
    sums: np.ndarray, v_post_minus_pre_sums: np.ndarray, n_nodes: np.ndarray,
    spec: _ComponentSpec,
) -> np.ndarray:
    """Mean ``v_post_d`` = v_post − v_onset = v_pre_d + v_post_minus_pre."""
    n_nodes_div = np.maximum(n_nodes, 1)
    v_post_d = sums[:, spec.i_v_pre_d] / n_nodes_div + v_post_minus_pre_sums / n_nodes_div
    v_post_d[n_nodes == 0] = 0.0
    return v_post_d


def _dominant_drive_from_step(
    step: dict[str, Any] | None, *, model: str,
) -> str | None:
    if step is None:
        return None
    if model == "hp_lp":
        if abs(step["v_syn_exc"]) >= abs(step["v_syn_inh"]):
            return "exc" if abs(step["v_syn_exc"]) > 1e-9 else "none"
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
    params,
    session,
    sums: np.ndarray,
    sum_sqs: np.ndarray,
    n_nodes: np.ndarray,
    v_post_minus_pre_sums: np.ndarray,
    v_post: np.ndarray,
    time_window: TimeWindow,
    ti_mode: str,
    component_spec: _ComponentSpec,
    extra: dict[str, Any] | None = None,
    v_ca_sums: np.ndarray | None = None,
    ca_sums: np.ndarray | None = None,
    ca_pre_sums: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build one report dict from a single batch×cell sums row."""
    # Peak on |v_post_d| (= |v_post − v_onset| = |v_pre_d + v_post_minus_pre|).
    v_post_d = _v_post_d_from_sums(sums, v_post_minus_pre_sums, n_nodes, component_spec)
    n_t = int(v_post_d.size)
    if time_window.kind == "ms":
        dt = float(session.delta_ms)
        t_lo = train.t_from_ms(time_window.start, delta_ms=dt)
        t_hi = train.t_from_ms(time_window.stop, delta_ms=dt)
        if t_lo < 0 or t_hi >= n_t or t_lo > t_hi:
            raise SystemExit(
                f"--ms-shown {time_window.start:g},{time_window.stop:g} "
                f"(t={t_lo}:{t_hi}) out of range for sums length {n_t}"
            )
        seg = v_post_d[t_lo:t_hi + 1]
        peak_t = t_lo + int(np.argmax(np.abs(seg))) if seg.size else t_lo
    else:
        peak_t = _v_post_d_peak_t_rel(v_post_d, before_t)
        t_lo = max(0, peak_t + int(time_window.start))
        t_hi = min(n_t - 1, peak_t + int(time_window.stop))
    steps: list[dict[str, Any]] = []
    peak_step: dict[str, Any] | None = None
    for t in range(t_lo, t_hi + 1):
        n_nodes_at_t = int(n_nodes[t])
        if n_nodes_at_t == 0:
            continue
        if ti_mode == "t_rel":
            ti = t
        elif ti_mode == "abs_minus_before":
            ti = t - before_t
        else:
            raise ValueError(f"unknown ti_mode {ti_mode!r}")
        t_rel = t - peak_t
        step = _step_from_sums(
            t=t, t_rel=t_rel, ti=ti, v_post_val=float(v_post[t]),
            sums=_sums_dict_from_vec(sums[t], component_spec.keys),
            sum_sqs=_sums_dict_from_vec(sum_sqs[t], component_spec.keys),
            v_post_minus_pre_sum=float(v_post_minus_pre_sums[t]),
            n_nodes=n_nodes_at_t,
            spec=component_spec,
            g_leak=float(session.g_leak),
            dt_over_c=float(train.membrane_dt_over_c(session.cap, session.delta_ms)),
        )
        if v_ca_sums is not None:
            step["v_ca"] = float(v_ca_sums[t] / n_nodes_at_t)
        if ca_sums is not None and ca_pre_sums is not None:
            ca = float(ca_sums[t] / n_nodes_at_t)
            ca_pre = float(ca_pre_sums[t] / n_nodes_at_t)
            step["ca"] = ca
            step["ca_pre"] = ca_pre
            step["ca_post_minus_pre"] = ca - ca_pre
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
        "filter": session_filter_plot_token(session),
        "before_t": before_t,
        "time_window_kind": time_window.kind,
        "time_window": [time_window.start, time_window.stop],
        "t_window": [t_lo, t_hi],
        "v_post_d_peak_t": peak_t,
        "v_post_d_peak": float(v_post_d[peak_t]),
        "v_post_d_polarity": _polarity(float(v_post_d[peak_t])),
        "v_post_d_onset_t": onset,
        "params": _node_params(params, session, int(nodes[0])),
        "globals": _globals(session),
        "steps": steps,
        "peak_step": peak_step,
        "peak_drive": _dominant_drive_from_step(
            peak_step, model=component_spec.model,
        ),
        "v_post": v_post.tolist(),
    }
    if v_ca_sums is not None:
        n_nodes_div = np.maximum(n_nodes, 1)
        v_ca_trace = v_ca_sums / n_nodes_div
        v_ca_trace[n_nodes == 0] = 0.0
        report["v_ca"] = v_ca_trace.tolist()
    if ca_sums is not None:
        n_nodes_div = np.maximum(n_nodes, 1)
        ca_trace = ca_sums / n_nodes_div
        ca_trace[n_nodes == 0] = 0.0
        report["ca"] = ca_trace.tolist()
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


def _scalar_param_at_node(params, key: str, session, node: int) -> float:
    raw = params[key]
    backend = session.backend
    if torch.is_tensor(raw):
        if raw.dim() == 0:
            return float(raw.item())
        if backend.network is not None:
            cell_idx = int(backend.network.node_cell[node])
        else:
            cell_idx = int(node) % int(backend.n_cells)
        return float(raw[cell_idx])
    return float(raw)


def _node_params(params, session, node: int) -> dict[str, float]:
    backend = session.backend
    for key in ("a_gt", "bias_gt"):
        if key not in params:
            raise SystemExit(f"params missing {key}")
    a_gt = _scalar_param_at_node(params, "a_gt", session, node)
    bias_gt = _scalar_param_at_node(params, "bias_gt", session, node)
    if session.model == "hp_lp":
        return {
            "a_in": float(params["a_in"][node]),
            "a_out": float(params["a_out"][node]),
            "v_th": float(params["v_th"][node]),
            "a_gt": a_gt,
            "bias_gt": bias_gt,
            "e_leak_mV": float(params["e_leak"][node]),
            "tau_lp_ms": float(params["tau_lp"][node]),
            "tau_hp_ms": float(params["tau_hp"][node]),
            "a_h": float(params["a_h"][node]),
        }
    from neuron.schema import borst_i_h_rev_kwargs

    i_h_rev = (session.train_opts or {})["i_h_rev"]
    a_h_rev, *_rest = borst_i_h_rev_kwargs(params, i_h_rev)
    e_leak = float(params["e_leak"][node])
    return {
        "a_in": float(params["a_in"][node]),
        "a_out": float(params["a_out"][node]),
        "a_gt": a_gt,
        "bias_gt": bias_gt,
        "v_th": float(params["v_th"][node]),
        "h_g_max": float(session.h_g_max),
        "a_h": float(params["a_h"][node]),
        "a_h_rev": float(a_h_rev[node]),
        "e_leak_mV": e_leak,
        "e_h_rev": float(train.e_h_rev(e_leak, session.e_h)),
    }


def _globals(session):
    pack = session.primary_readout
    t_onset = train.pack_t_onset(pack)
    if session.model == "hp_lp":
        return {
            "delta_ms": float(session.delta_ms),
            "state_clamp": float(session.STATE_CLAMP),
            "g_leak_nS": float(session.g_leak),
            "euler": str(session.euler),
            "t_onset": t_onset,
        }
    return {
        "e_exc": float(session.e_exc),
        "e_inh": float(session.e_inh),
        "e_h": float(session.e_h),
        "g_leak_nS": float(session.g_leak),
        "dt_over_c": float(train.membrane_dt_over_c(session.cap, session.delta_ms)),
        "delta_ms": float(session.delta_ms),
        "euler": str(session.euler),
        "t_onset": t_onset,
    }


def cell_from_node(nodes_by_cell: dict[str, np.ndarray]) -> dict[int, str]:
    out: dict[int, str] = {}
    for cell, node_indices in nodes_by_cell.items():
        for node_idx in np.asarray(node_indices, dtype=np.int64).ravel():
            out[int(node_idx)] = cell
    return out


def _node_cell_lookup(plan: _ComponentForwardBatch, cells: list[str]) -> np.ndarray:
    """Map node id → index in ``cells`` (-1 if absent). Length ``max(node_id)+1``."""
    cell_idx_from_cell = {cell: cell_idx for cell_idx, cell in enumerate(cells)}
    if plan.all_nodes.size == 0:
        return np.empty(0, dtype=np.int32)
    out = np.full(int(plan.all_nodes.max()) + 1, -1, dtype=np.int32)
    for node_id, cname in plan.cell_from_node.items():
        cell_idx = cell_idx_from_cell.get(cname)
        if cell_idx is not None:
            out[int(node_id)] = cell_idx
    return out


def _merge_forward_sums(
    forward_sums: _ComponentForwardSums,
    forward_batches: list[_ComponentForwardBatch],
    cells: list[str],
    n_t_aligned: int,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray] | None,
    dict[str, np.ndarray] | None,
    dict[str, np.ndarray] | None,
]:
    """Sum per-cell sums rows across run batches (spot multi-stimulus mean)."""
    n_keys = forward_sums.spec.n_keys
    sums = {cell: np.zeros((n_t_aligned, n_keys), dtype=float) for cell in cells}
    sum_sqs = {cell: np.zeros((n_t_aligned, n_keys), dtype=float) for cell in cells}
    n_nodes = {cell: np.zeros(n_t_aligned, dtype=np.int64) for cell in cells}
    v_post_minus_pre_sums = {cell: np.zeros(n_t_aligned, dtype=float) for cell in cells}
    nodes_ref = {cell: np.zeros(0, dtype=np.int64) for cell in cells}
    v_ca_sums = (
        {cell: np.zeros(n_t_aligned, dtype=float) for cell in cells}
        if forward_sums.v_ca_sums is not None else None
    )
    ca_sums = (
        {cell: np.zeros(n_t_aligned, dtype=float) for cell in cells}
        if forward_sums.ca_sums is not None else None
    )
    ca_pre_sums = (
        {cell: np.zeros(n_t_aligned, dtype=float) for cell in cells}
        if forward_sums.ca_pre_sums is not None else None
    )
    for batch_idx, plan in enumerate(forward_batches):
        for cell in cells:
            if cell not in plan.nodes_by_cell:
                continue
            us = plan.nodes_by_cell[cell]
            if us.size == 0:
                continue
            if nodes_ref[cell].size == 0:
                nodes_ref[cell] = us
            sums[cell] += forward_sums.sums[batch_idx][cell]
            sum_sqs[cell] += forward_sums.sum_sqs[batch_idx][cell]
            n_nodes[cell] += forward_sums.n_nodes[batch_idx][cell]
            v_post_minus_pre_sums[cell] += forward_sums.v_post_minus_pre_sums[batch_idx][cell]
            if v_ca_sums is not None:
                v_ca_sums[cell] += forward_sums.v_ca_sums[batch_idx][cell]
                ca_sums[cell] += forward_sums.ca_sums[batch_idx][cell]
                ca_pre_sums[cell] += forward_sums.ca_pre_sums[batch_idx][cell]
    return (
        sums, sum_sqs, n_nodes, v_post_minus_pre_sums, nodes_ref,
        v_ca_sums, ca_sums, ca_pre_sums,
    )


def _build_forward_batch(
    nodes_by_cell: dict[str, np.ndarray],
    *,
    t0_bn_row: np.ndarray,
    n_t_aligned: int,
) -> _ComponentForwardBatch:
    """Build one run batch; ``node_t0[i] = t0_bn_row[all_nodes[i]]``."""
    all_nodes = np.unique(np.concatenate([ids for ids in nodes_by_cell.values()]))
    return _ComponentForwardBatch(
        all_nodes=all_nodes,
        node_t0=np.asarray(t0_bn_row[all_nodes], dtype=np.int64),
        n_t_aligned=int(n_t_aligned),
        cell_from_node=cell_from_node(nodes_by_cell),
        nodes_by_cell=nodes_by_cell,
    )


# ---------------------------------------------------------------------------
# Average bar components (cost-radius)
# ---------------------------------------------------------------------------


def _bar_meta(session, task: str):
    """One-shot ``(specs, grids)`` for a moving-bar task."""
    specs = bar_specs_for_session(session, task)
    pack = session.pack_for(task)
    grids = moving_bar_session_t0_grids(
        session, specs, pack.cost_radius, int(session.n_t),
        t_onset=train.pack_t_onset(pack),
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
    """Spec list for average-mode bar without a full forward readout."""
    if specs is None or grids is None:
        specs, grids = _bar_meta(session, task)
    all_specs = [spec.name for spec in specs]
    try:
        if requested is not None:
            return filter_requested_specs(all_specs, requested)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    specs_by_cell = moving_bar_specs_by_cell(session, task, grids.side)
    out: list[str] = []
    for cell in cells:
        for spec_name in specs_by_cell.get(cell, all_specs):
            if spec_name in all_specs and spec_name not in out:
                out.append(spec_name)
    return out or list(all_specs)


def _resolve_bar_spec_i_sti(
    session,
    task: str,
    spec_names: list[str],
    *,
    specs=None,
    grids=None,
):
    """Validate specs; return ``(pack, specs, grids, spec_batch_indices, i_sti, t0_bn)``."""
    if task not in train.MOVING_BAR_TASKS:
        raise SystemExit(f"unsupported task {task!r}")
    if not spec_names:
        raise SystemExit("bar component forward requires at least one spec")
    pack = session.pack_for(task)
    if specs is None or grids is None:
        specs, grids = _bar_meta(session, task)
    batch_idx_from_spec_name = {
        spec.name: spec_idx for spec_idx, spec in enumerate(specs)
    }
    missing = [spec_name for spec_name in spec_names if spec_name not in batch_idx_from_spec_name]
    if missing:
        raise SystemExit(
            f"spec(s) {missing} not in {[spec.name for spec in specs]}"
        )
    spec_batch_indices = [batch_idx_from_spec_name[spec_name] for spec_name in spec_names]
    return pack, specs, grids, spec_batch_indices, pack.i_sti[spec_batch_indices], np.asarray(grids.t0_bn)


def _analyze_component_forward(
    session,
    *,
    params,
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
    forward_sums = _forward_component(
        session, params, i_sti, forward_batches, cells,
        t_start=time_window.forward_t_start(delta_ms=dt),
        t_stop=time_window.forward_t_stop(delta_ms=dt),
    )
    component_spec = forward_sums.spec

    def _one_report(
        *,
        cell: str,
        spec: str | None,
        before: int,
        nodes: np.ndarray,
        sums: np.ndarray,
        sum_sqs: np.ndarray,
        n_nodes: np.ndarray,
        v_post_minus_pre_sums: np.ndarray,
        v_ca_sums: np.ndarray | None = None,
        ca_sums: np.ndarray | None = None,
        ca_pre_sums: np.ndarray | None = None,
    ) -> dict[str, Any]:
        if nodes.size == 0:
            raise SystemExit(f"no nodes for cell {cell!r}")
        v_post = _v_post_from_sums(sums, n_nodes, component_spec)
        v_post_d = _v_post_d_from_sums(
            sums, v_post_minus_pre_sums, n_nodes, component_spec,
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
            params =params,
            session=session,
            sums=sums,
            sum_sqs=sum_sqs,
            n_nodes=n_nodes,
            v_post_minus_pre_sums=v_post_minus_pre_sums,
            v_post=v_post,
            time_window=time_window,
            ti_mode=ti_mode,
            component_spec=component_spec,
            extra=cell_extra or None,
            v_ca_sums=v_ca_sums,
            ca_sums=ca_sums,
            ca_pre_sums=ca_pre_sums,
        )
        if n_nodes_for_cell is not None:
            report["n_nodes"] = int(n_nodes_for_cell(cell))
        return report

    if merge_batches:
        n_t_aligned = forward_batches[0].n_t_aligned
        (
            sums, sum_sqs, n_nodes, v_post_minus_pre_sums, nodes_ref,
            v_ca_sums, ca_sums, ca_pre_sums,
        ) = _merge_forward_sums(
            forward_sums, forward_batches, cells, n_t_aligned,
        )
        before = int(before_t[0])
        return {
            cell: _one_report(
                cell=cell,
                spec=None,
                before=before,
                nodes=nodes_ref[cell],
                sums=sums[cell],
                sum_sqs=sum_sqs[cell],
                n_nodes=n_nodes[cell],
                v_post_minus_pre_sums=v_post_minus_pre_sums[cell],
                v_ca_sums=None if v_ca_sums is None else v_ca_sums[cell],
                ca_sums=None if ca_sums is None else ca_sums[cell],
                ca_pre_sums=None if ca_pre_sums is None else ca_pre_sums[cell],
            )
            for cell in cells
        }

    out: dict[str, dict[str, dict[str, Any]]] = {}
    for batch_idx, spec in enumerate(batch_specs):
        if spec is None:
            raise SystemExit("non-merge component forward requires batch_specs as str")
        out[spec] = {}
        for cell in cells:
            out[spec][cell] = _one_report(
                cell=cell,
                spec=spec,
                before=int(before_t[batch_idx]),
                nodes=forward_batches[batch_idx].nodes_by_cell[cell],
                sums=forward_sums.sums[batch_idx][cell],
                sum_sqs=forward_sums.sum_sqs[batch_idx][cell],
                n_nodes=forward_sums.n_nodes[batch_idx][cell],
                v_post_minus_pre_sums=forward_sums.v_post_minus_pre_sums[batch_idx][cell],
                v_ca_sums=(
                    None if forward_sums.v_ca_sums is None else forward_sums.v_ca_sums[batch_idx][cell]
                ),
                ca_sums=(
                    None if forward_sums.ca_sums is None else forward_sums.ca_sums[batch_idx][cell]
                ),
                ca_pre_sums=(
                    None if forward_sums.ca_pre_sums is None
                    else forward_sums.ca_pre_sums[batch_idx][cell]
                ),
            )
    return out


def _analyze_bar_forward(
    session,
    *,
    params,
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
    pack, specs, grids, spec_batch_indices, i_sti, t0_bn = _resolve_bar_spec_i_sti(
        session, task, spec_names, specs=specs, grids=grids,
    )
    before_by_batch: list[int] = []
    forward_batches: list[_ComponentForwardBatch] = []
    for spec_batch_idx, spec in zip(spec_batch_indices, spec_names):
        spec_nodes_by_cell = nodes_for_bi(spec_batch_idx, spec, pack=pack, t0_bn=t0_bn)
        before = int(grids.before_t[spec])
        after = int(grids.after_t[spec])
        before_by_batch.append(before)
        forward_batches.append(
            _build_forward_batch(
                spec_nodes_by_cell,
                t0_bn_row=t0_bn[spec_batch_idx],
                n_t_aligned=before + after + 1,
            ),
        )
    return _analyze_component_forward(
        session,
        params =params,
        cells=cells,
        task=task,
        i_sti=i_sti,
        forward_batches=forward_batches,
        before_t=before_by_batch,
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
    params,
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
        connectome = session.backend.network
        if not cols_holder:
            cols_holder.append(moving_bar_cost_hexes(connectome, cost_radius=pack.cost_radius))
        hexes = cols_holder[0]
        out: dict[str, np.ndarray] = {}
        for cell in cells:
            try:
                nodes = moving_bar_nodes_on_hexes(connectome, cell, hexes)
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
            nodes = nodes[t0_bn[bi, nodes] >= 0]
            if nodes.size == 0:
                raise SystemExit(f"no valid {cell} nodes in cost_radius for bar")
            out[cell] = nodes
        return out

    return _analyze_bar_forward(
        session,
        params =params,
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
    connectome = session_one.backend.network
    if connectome is None:
        raise SystemExit("spot average requires a network backend")
    opts = dict((session_one.train_opts or {}).get(f"{pack.name}_stimulus_opts") or {})
    spot = spot_from_opts(connectome, stimulus_opts=opts)
    (
        batch_idx, node_idx, radii, type_idx, _stim_u, _stim_v, _du, _dv, _center_entry_mask,
    ) = build_spot_center_readout(
        connectome,
        spot_stimulus_batches(spot),
        pack_spot_cost_radii(pack),
        pack.cost_radius,
    )
    type_idx_from_cell_name: dict[str, int] = {}
    for cell in cells:
        if cell not in connectome.cells:
            raise SystemExit(f"unknown cell {cell!r}")
        type_idx_from_cell_name[cell] = connectome.cells.index(cell)
    return pack, batch_idx, node_idx, radii, type_idx, type_idx_from_cell_name


def _spot_radius_entry_mask(radii: np.ndarray, radius: int) -> np.ndarray:
    """True for cost-readout entries at Euclidean ``radius``."""
    return np.isclose(np.asarray(radii, dtype=np.float64), float(radius))


def _spot_gt_extra(
    *,
    cell: str,
    gt_on: dict,
    radius: int,
    v_post: np.ndarray,
    v_post_d: np.ndarray,
    t_onset: int,
    a_gt: float,
    bias_gt: float,
    gt_key: str,
) -> dict[str, Any]:
    """Affine GT on readout axis: ``a_gt * gt + bias``; key is ``gt_v`` or ``gt_ca``."""
    if gt_key not in _GT_PLOT_PANEL:
        raise SystemExit(f"unknown gt_key {gt_key!r}; expected gt_v|gt_ca")
    extra: dict[str, Any] = {"gt_peak": None, gt_key: None, "radius": radius}
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
    extra[gt_key] = gt_aff.tolist()
    return extra


def _spot_extra_for_cell_fn(
    session_one, params, pack, *, radius: int, t_onset: int, train_filter,
):
    """Build ``extra_for_cell`` with train GT named ``gt_v`` / ``gt_ca``."""
    from param_defaults import BIAS_GT_FROM_V_ONSET, PARAM_BOXES
    contrast = contrast_for_task(pack.name)
    train_filter = train.expand_filter(train_filter)
    gt_key = f"gt_{filter_plot_token(train_filter)}"
    gt_on = resolve_spot_gt_cubes(
        {contrast: session_one}, filter=train_filter,
    ).get(contrast) or {}
    opts = session_one.train_opts or {}
    from_onset = bool(opts.get("bias_gt_from_v_onset", BIAS_GT_FROM_V_ONSET))
    lo = float(PARAM_BOXES["bias_gt"]["lo"])
    hi = float(PARAM_BOXES["bias_gt"]["hi"])
    cell_names = [str(cell_name) for cell_name in session_one.backend.network.cells]

    def extra_for_cell(
        cell: str, v_post: np.ndarray, v_post_d: np.ndarray,
    ) -> dict[str, Any]:
        if from_onset and 0 <= int(t_onset) < len(v_post):
            cell_idx = cell_names.index(str(cell))
            val = float(np.clip(float(v_post[int(t_onset)]), lo, hi))
            bg = params["bias_gt"]
            if torch.is_tensor(bg):
                if bg.dim() == 0:
                    params["bias_gt"] = bg.new_tensor(val)
                else:
                    params["bias_gt"] = bg.clone()
                    params["bias_gt"][cell_idx] = val
            else:
                params["bias_gt"] = val
        a_gt, bias_gt = gt_affine_scalars_for_cell(
            params, cell, session_one.backend, session=session_one,
        )
        return _spot_gt_extra(
            cell=cell,
            gt_on=gt_on,
            radius=radius,
            v_post=v_post,
            v_post_d=v_post_d,
            t_onset=t_onset,
            a_gt=a_gt,
            bias_gt=bias_gt,
            gt_key=gt_key,
        )

    return extra_for_cell


def analyze_spot_average(
    session_one,
    *,
    params,
    cells: list[str],
    task: str,
    time_window: TimeWindow,
    radius: int = 0,
    train_filter="none",
) -> dict[str, dict[str, Any]]:
    """One batched v forward over spot stimulus rows; mean at Euclidean ``radius``.

    ``time_window`` is absolute aligned ms for spot (``0`` = trial start). Pre
    only: ``TimeWindow("ms", 0, ms_pre)``. Do not pass negative ``start`` for
    "before onset" — that confuses stimulus length with the analyze window.
    ``train_filter`` is the run's train ``filter`` (names GT ``gt_v``/``gt_ca``).
    """
    if task not in train.SPOT_TASKS:
        raise SystemExit(f"unsupported task {task!r}")
    radius = int(radius)
    pack, batch_idx, node_idx, radii, type_idx, type_idx_from_cell_name = _spot_session_readout(
        session_one, cells,
    )
    radius_entry_mask = _spot_radius_entry_mask(radii, radius)
    t_onset = train.pack_t_onset(pack)

    i_sti = pack.i_sti if pack.i_sti.dim() == 3 else pack.i_sti.unsqueeze(0)
    n_stimulus_batch, n_t, n_nodes = i_sti.shape
    t0_abs = np.zeros(n_nodes, dtype=np.int64)
    n_t_aligned = time_window.aligned_n_t(n_t, delta_ms=float(session_one.delta_ms))

    forward_batches: list[_ComponentForwardBatch] = []
    i_sti_rows: list[int] = []
    for stimulus_batch_idx in range(n_stimulus_batch):
        entry_mask = radius_entry_mask & (batch_idx == stimulus_batch_idx)
        if not np.any(entry_mask):
            continue
        nodes_by_cell: dict[str, np.ndarray] = {}
        for cell in cells:
            cell_entry_mask = entry_mask & (type_idx == type_idx_from_cell_name[cell])
            if np.any(cell_entry_mask):
                nodes_by_cell[cell] = np.unique(node_idx[cell_entry_mask])
        if not nodes_by_cell:
            continue
        forward_batches.append(
            _build_forward_batch(nodes_by_cell, t0_bn_row=t0_abs, n_t_aligned=n_t_aligned),
        )
        i_sti_rows.append(stimulus_batch_idx)

    if not forward_batches:
        raise SystemExit(
            f"no spot nodes at radius={radius} for requested cells in spot readout"
        )

    n_batch = len(forward_batches)
    return _analyze_component_forward(
        session_one,
        params =params,
        cells=cells,
        task=task,
        i_sti=i_sti[i_sti_rows],
        forward_batches=forward_batches,
        before_t=[t_onset] * n_batch,
        batch_specs=[None] * n_batch,
        time_window=time_window,
        mode="average",
        ti_mode="abs_minus_before",
        merge_batches=True,
        extra={"radius": radius},
        extra_for_cell=_spot_extra_for_cell_fn(
            session_one, params, pack, radius=radius, t_onset=t_onset,
            train_filter=train_filter,
        ),
        n_nodes_for_cell=lambda cell: int(
            np.sum(radius_entry_mask & (type_idx == type_idx_from_cell_name[cell]))
        ),
    )


# ---------------------------------------------------------------------------
# Hex mode (single hex; same run as average)
# ---------------------------------------------------------------------------


def _nodes_at_hex(session, cell: str, *, at_x: float, at_y: float, cost_radius: int):
    connectome = session.backend.network
    if connectome is None:
        raise SystemExit("hex mode requires a network backend")
    hexes = filter_sti_hexes(
        moving_bar_cost_hexes(connectome, cost_radius=cost_radius),
        at_x=at_x,
        at_y=at_y,
    )
    if not hexes:
        raise SystemExit(f"no hex at x={at_x!r} y={at_y!r} within cost_radius={cost_radius}")
    if len(hexes) > 1:
        raise SystemExit(f"multiple hexes at x={at_x!r} y={at_y!r}; pick a unique hex")
    hex = hexes[0]
    if cell not in connectome.cells:
        raise SystemExit(f"unknown cell {cell!r}")
    nodes = hex2gt(connectome, int(hex.u), int(hex.v), cell).tolist()
    if not nodes:
        raise SystemExit(f"no {cell} node at hex ({at_x},{at_y})")
    return hex, nodes


def _resolve_hex_node(
    session,
    cell: str,
    *,
    at_x: float,
    at_y: float,
    cost_radius: int,
    node: int | None,
):
    """Resolve ``(hex, node_id)`` for hex mode; ``--node`` if multiple at hex."""
    hex, nodes = _nodes_at_hex(
        session, cell, at_x=at_x, at_y=at_y, cost_radius=cost_radius,
    )
    if node is None:
        if len(nodes) > 1:
            raise SystemExit(f"multiple {cell} at ({at_x},{at_y}): {nodes}; pass --node")
        node = nodes[0]
    elif node not in nodes:
        raise SystemExit(f"node {node} not in {nodes}")
    return hex, int(node)


def analyze_spot_hex(
    session_one,
    *,
    params,
    cell: str,
    task: str,
    at_x: float,
    at_y: float,
    node: int | None,
    time_window: TimeWindow,
    train_filter="none",
) -> dict[str, dict[str, Any]]:
    """One batched v forward at one hex; stim-on (radius 0) rows for that node only."""
    if task not in train.SPOT_TASKS:
        raise SystemExit(f"unsupported task {task!r}")
    pack, batch_idx, node_idx, radii, type_idx, type_idx_from_cell_name = _spot_session_readout(
        session_one, [cell],
    )
    radius_entry_mask = _spot_radius_entry_mask(radii, 0)
    hex, node = _resolve_hex_node(
        session_one, cell, at_x=at_x, at_y=at_y,
        cost_radius=pack.cost_radius, node=node,
    )
    t_onset = train.pack_t_onset(pack)

    i_sti = pack.i_sti if pack.i_sti.dim() == 3 else pack.i_sti.unsqueeze(0)
    n_stimulus_batch, n_t, n_nodes = i_sti.shape
    t0_abs = np.zeros(n_nodes, dtype=np.int64)
    n_t_aligned = time_window.aligned_n_t(n_t, delta_ms=float(session_one.delta_ms))
    node_arr = np.asarray([node], dtype=np.int64)
    type_cell = type_idx_from_cell_name[cell]

    forward_batches: list[_ComponentForwardBatch] = []
    i_sti_rows: list[int] = []
    for stimulus_batch_idx in range(n_stimulus_batch):
        entry_mask = (
            radius_entry_mask
            & (batch_idx == stimulus_batch_idx)
            & (type_idx == type_cell)
            & (node_idx == node)
        )
        if not np.any(entry_mask):
            continue
        forward_batches.append(
            _build_forward_batch({cell: node_arr}, t0_bn_row=t0_abs, n_t_aligned=n_t_aligned),
        )
        i_sti_rows.append(stimulus_batch_idx)

    if not forward_batches:
        raise SystemExit(
            f"no stim-on spot rows for {cell} node {node} at hex ({at_x},{at_y})"
        )

    n_batch = len(forward_batches)
    return _analyze_component_forward(
        session_one,
        params =params,
        cells=[cell],
        task=task,
        i_sti=i_sti[i_sti_rows],
        forward_batches=forward_batches,
        before_t=[t_onset] * n_batch,
        batch_specs=[None] * n_batch,
        time_window=time_window,
        mode="hex",
        ti_mode="abs_minus_before",
        merge_batches=True,
        extra={
            "node": node,
            "hex": {"x": at_x, "y": at_y},
            "uv": {"u": int(hex.u), "v": int(hex.v)},
        },
        extra_for_cell=_spot_extra_for_cell_fn(
            session_one, params, pack, radius=0, t_onset=t_onset,
            train_filter=train_filter,
        ),
        n_nodes_for_cell=lambda _c: 1,
    )


def analyze_bar_hex(
    session,
    *,
    params,
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
    hex, node = _resolve_hex_node(
        session, cell, at_x=at_x, at_y=at_y,
        cost_radius=pack.cost_radius, node=node,
    )
    node_arr = np.asarray([node], dtype=np.int64)
    nodes_by_cell = {cell: node_arr}

    def nodes_for_bi(bi, spec, *, pack, t0_bn):
        if int(t0_bn[bi, node]) < 0:
            raise SystemExit(f"no t0 for node {node} on spec {spec!r}")
        return nodes_by_cell

    return _analyze_bar_forward(
        session,
        params =params,
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
            "uv": {"u": int(hex.u), "v": int(hex.v)},
        },
    )



# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_filename(report: dict[str, Any], *, file_suffix: str = "", html: bool = False) -> str:
    from figure.util import plot_file_ext

    filter_token = str(report.get("filter") or "v")
    parts = [report["cell"], report["task"], filter_token, report.get("mode", "average")]
    if report.get("spec"):
        parts.append(str(report["spec"]))
    if report.get("mode") == "hex":
        hex = report["hex"]
        parts.append(f"x{hex['x']}_y{hex['y']}")
    radius = report.get("radius")
    if radius is not None and int(radius) != 0:
        parts.append(f"radius{int(radius)}")
    return "_".join(parts) + f"{file_suffix}{plot_file_ext(html=html)}"


def _overlay_plot_filename(
    reports: list[dict[str, Any]], *, file_suffix: str = "", html: bool = False,
) -> str:
    from figure.util import plot_file_ext

    first_report = reports[0]
    filter_token = str(first_report.get("filter") or "v")
    specs = "_".join(str(one_report["spec"]) for one_report in reports)
    return (
        f"{first_report['cell']}_{first_report['task']}_{filter_token}_overlay_{specs}"
        f"{file_suffix}{plot_file_ext(html=html)}"
    )


def _component_figure(title: str, spec: _ComponentSpec):
    """Shared grid figure: rows = plot panels, cols = traces within a row."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from figure.util import save_figure

    n_row, n_col = len(spec.plot_panels), spec.n_col
    fig, axes = plt.subplots(
        n_row, n_col,
        figsize=(2.6 * n_col, 2.2 * n_row),
        sharex=True,
        constrained_layout=True,
    )
    if n_row == 1 and n_col == 1:
        axes = np.array([[axes]])
    elif n_row == 1:
        axes = np.array([axes])
    elif n_col == 1:
        axes = axes[:, np.newaxis]
    return fig, axes, save_figure


def _hide_unused_axes(axes, spec: _ComponentSpec) -> None:
    n_row, n_col = axes.shape
    for row_idx, (_panel_ylabel, series) in enumerate(spec.plot_panels):
        for col_idx in range(len(series), n_col):
            axes[row_idx, col_idx].set_visible(False)


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
    for curve in curves:
        val = np.asarray(curve, dtype=float).ravel()
        val = val[np.isfinite(val)]
        if val.size:
            chunks.append(val)
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
    for row_idx, curves in row_curves.items():
        if not curves:
            continue
        _, series = spec.plot_panels[row_idx]
        ylo, yhi = _shared_row_ylim(curves)
        for col_idx in range(len(series)):
            axes[row_idx, col_idx].set_ylim(ylo, yhi)


def _save_component_figure(
    fig, axes, *, xlabel: str, out_path, save_figure, spec: _ComponentSpec,
) -> None:
    _hide_unused_axes(axes, spec)
    last_row = axes.shape[0] - 1
    for col_idx in range(axes.shape[1]):
        ax = axes[last_row, col_idx]
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

    models = {str(one_report.get("model", "borst")) for one_report in reports}
    if len(models) != 1:
        raise SystemExit(f"overlay requires one model; got {sorted(models)}")
    kinds = {str(one_report.get("time_window_kind", "t_rel")) for one_report in reports}
    if len(kinds) != 1:
        raise SystemExit(f"overlay requires one time_window_kind; got {sorted(kinds)}")
    eulers = {
        train.expand_euler((one_report.get("globals") or {}).get("euler", "implicit"))
        for one_report in reports
    }
    if len(eulers) != 1:
        raise SystemExit(f"overlay requires one euler; got {sorted(eulers)}")
    filters = {str(one_report.get("filter", "v")) for one_report in reports}
    if len(filters) != 1:
        raise SystemExit(f"overlay requires one filter; got {sorted(filters)}")
    spec = _component_spec(models.pop(), eulers.pop(), filter=filters.pop())

    fig, axes, save_figure = _component_figure(title, spec)
    import matplotlib.pyplot as plt

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    linestyles = ("-", "--", "-.", ":")
    overlay = len(reports) > 1
    e_leak_mV = float(reports[0].get("params", {}).get("e_leak_mV", 0.0))
    globs = reports[0].get("globals") or {}
    params0 = reports[0].get("params") or {}
    delta_ms = float(globs.get("delta_ms", train.DELTA_MS))
    row_curves: dict[int, list[np.ndarray]] = {
        row_idx: [] for row_idx in spec.row_shared_ylim
    }
    tc = _plot_trace_colors(colors, spec)

    for row_idx, (panel_ylabel, series) in enumerate(spec.plot_panels):
        for col_idx, (key, label) in enumerate(series):
            ax = axes[row_idx, col_idx]
            color = tc[label]
            show_legend = overlay and row_idx == 0 and col_idx == 0
            drew_gt = False
            for report_idx, rep in enumerate(reports):
                ls = linestyles[report_idx % len(linestyles)] if overlay else "-"
                ts = np.asarray([step["t"] for step in rep["steps"]], dtype=int)
                xs = ts.astype(float) * delta_ms
                y = np.asarray([step[key] for step in rep["steps"]], dtype=float)
                std = np.asarray(
                    [
                        float(step.get("std", {}).get(key, 0.0))
                        for step in rep["steps"]
                    ],
                    dtype=float,
                )
                if row_idx in spec.row_shared_ylim:
                    row_curves[row_idx].append(y)
                    if np.any(std):
                        row_curves[row_idx].append(y + std)
                        row_curves[row_idx].append(y - std)
                plot_std_band(ax, xs, y, std, color=color, alpha=0.3)
                gt_key = next(
                    (gk for gk, pk in _GT_PLOT_PANEL.items() if pk == key and rep.get(gk) is not None),
                    None,
                )
                model_label = (
                    str(rep["spec"]) if show_legend
                    else (key if gt_key is not None else "_nolegend_")
                )
                ax.plot(
                    xs, y,
                    label=model_label,
                    color=color,
                    linestyle=ls,
                    linewidth=1.4,
                )
                if gt_key is not None:
                    gt_full = np.asarray(rep[gt_key], dtype=float)
                    t_onset = int(rep.get("before_t") or 0)
                    time_idx = ts - t_onset
                    valid = (time_idx >= 0) & (time_idx < gt_full.shape[0])
                    if np.any(valid):
                        y_gt = gt_full[time_idx[valid]]
                        xs_gt = xs[valid]
                        if row_idx in spec.row_shared_ylim:
                            row_curves[row_idx].append(y_gt)
                        ax.plot(
                            xs_gt, y_gt,
                            color="k",
                            linestyle="--",
                            linewidth=1.2,
                            label="gt" if (show_legend or not overlay) else "_nolegend_",
                        )
                        drew_gt = True
            e_note = _g_e_note(label, e_leak_mV=e_leak_mV, globs=globs, params=params0)
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
    first_report = reports[0]
    specs_csv = ",".join(str(one_report["spec"]) for one_report in reports)
    title = (
        f"{first_report['cell']}  {first_report['task']}  overlay=[{specs_csv}]"
        f"  mode={first_report.get('mode')}  n={first_report.get('n_nodes')}"
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
    use_ca = report.get("filter") == "ca"
    state_hdr = (
        "ca  ca_pre  ca_post_minus_pre"
        if use_ca else "v_post  v_pre  v_post_minus_pre"
    )
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

    def _state_cols(step: dict[str, Any]) -> str:
        if use_ca:
            return (
                f"{step['ca']:+8.4f} {step['ca_pre']:+8.4f} "
                f"{step['ca_post_minus_pre']:+8.4f}"
            )
        v_pre = float(step["v_post"]) - float(step["v_post_minus_pre"])
        return (
            f"{step['v_post']:+8.4f} {v_pre:+8.4f} {step['v_post_minus_pre']:+8.4f}"
        )

    if model == "hp_lp":
        if print_steps:
            print(
                f"\n{x_key}  n  {state_hdr}  i_sti "
                "v_syn  v_syn_exc -v_syn_inh  dv_leak  dv_hp"
            )
            for step in report["steps"]:
                print(
                    f"{step[x_key]:4d} {step.get('n_nodes', 1):3d} {_state_cols(step)} "
                    f"{step['i_sti']:+6.3f} {step['v_syn']:+6.3f} {step['v_syn_exc']:+7.3f} "
                    f"{step['v_syn_inh']:+7.3f} {step['dv_leak']:+8.4f} {step['dv_hp']:+7.4f}"
                )
        peak_step = report.get("peak_step")
        if peak_step is not None:
            print(f"\nHP/LP terms at peak {x_key}={peak_step[x_key]}:")
            for name, val in [
                ("v_syn", peak_step["v_syn"]), ("v_syn_exc", peak_step["v_syn_exc"]),
                ("-v_syn_inh", peak_step["v_syn_inh"]),
                ("v_slow", peak_step["v_slow"]), ("v_in", peak_step["v_in"]),
                ("v_hp", peak_step["v_hp"]),
                ("dv_leak", peak_step["dv_leak"]), ("dv_hp", peak_step["dv_hp"]),
            ]:
                print(f"  {name:8s} {val:+9.4f}")
        if "cost" in report and "best_cost" in report:
            print(f"best_cost={report['best_cost']:.4f}  cost={report['cost']:.4f}")
        return

    if print_steps:
        print(
            f"\n{x_key}  n  {state_hdr}  i_sti "
            "g_inh  g_h_rev  g_exc  num_inh  num_exc"
        )
        for step in report["steps"]:
            print(
                f"{step[x_key]:4d} {step.get('n_nodes', 1):3d} {_state_cols(step)} "
                f"{step['i_sti']:5.1f} {step['g_inh_nS']:.4f} {step['g_h_rev_nS']:.4f} "
                f"{step['g_exc_nS']:.4f} {step['num_inh']:+8.2f} {step['num_exc']:+8.2f}"
            )

    peak_step = report.get("peak_step")
    if peak_step is not None:
        num = float(peak_step["num"])
        dt_over_c = float(peak_step.get("dt_over_c", 0.0))
        print(f"\nNumerator at peak {x_key}={peak_step[x_key]} (num={num:.2f}):")
        for name, val in [
            ("num_v", peak_step["num_v"]),
            ("dt_over_c*i_sti", dt_over_c * peak_step["i_sti"]),
            ("dt_over_c*i_exc", dt_over_c * peak_step["num_exc"]),
            ("dt_over_c*i_inh", dt_over_c * peak_step["num_inh"]),
            ("dt_over_c*i_leak", dt_over_c * peak_step["num_leak"]),
            ("dt_over_c*i_h", dt_over_c * peak_step["num_i_h"]),
            ("dt_over_c*i_h_rev", dt_over_c * peak_step["num_i_h_rev"]),
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
    print("\n======== SPOT vs BAR polarity (cost-radius averages) ========")
    print(
        f"{'cell':6s} {'spot_post_d':>11s} {'spot_pol':>8s} {'spot_drv':>8s} "
        f"{'bar_post_d':>10s} {'bar_pol':>8s} {'bar_drv':>8s}  note"
    )
    for cell in cells:
        spot_report = spot_reports[cell]
        bar_report = bar_reports[cell]
        flip = (
            spot_report["v_post_d_polarity"] != bar_report["v_post_d_polarity"]
            and "0" not in (
                spot_report["v_post_d_polarity"], bar_report["v_post_d_polarity"],
            )
        )
        note = "FLIP" if flip else "same"
        if flip and spot_report.get("peak_drive") and bar_report.get("peak_drive"):
            if spot_report["peak_drive"] != bar_report["peak_drive"]:
                note += f" (drive {spot_report['peak_drive']}→{bar_report['peak_drive']})"
            else:
                note += f" (same drive={spot_report['peak_drive']}; see num terms)"
        print(
            f"{cell:6s} {spot_report['v_post_d_peak']:+11.4f} "
            f"{spot_report['v_post_d_polarity']:>8s} "
            f"{str(spot_report.get('peak_drive')):>8s} "
            f"{bar_report['v_post_d_peak']:+10.4f} {bar_report['v_post_d_polarity']:>8s} "
            f"{str(bar_report.get('peak_drive')):>8s}  {note}"
        )
        spot_peak_step, bar_peak_step = (
            spot_report.get("peak_step"), bar_report.get("peak_step"),
        )
        if not (spot_peak_step and bar_peak_step):
            continue
        model = spot_report.get("model", "borst")
        if model == "hp_lp":
            print(
                f"       spot@peak: v_syn_exc={spot_peak_step['v_syn_exc']:+.4f} "
                f"-v_syn_inh={spot_peak_step['v_syn_inh']:+.4f} "
                f"dv_hp={spot_peak_step['dv_hp']:+.4f} "
                f"dv_leak={spot_peak_step['dv_leak']:+.4f} "
                f"v_pre={float(spot_peak_step['v_post']) - float(spot_peak_step['v_post_minus_pre']):+.3f}"
            )
            print(
                f"       bar @peak: v_syn_exc={bar_peak_step['v_syn_exc']:+.4f} "
                f"-v_syn_inh={bar_peak_step['v_syn_inh']:+.4f} "
                f"dv_hp={bar_peak_step['dv_hp']:+.4f} "
                f"dv_leak={bar_peak_step['dv_leak']:+.4f} "
                f"v_pre={float(bar_peak_step['v_post']) - float(bar_peak_step['v_post_minus_pre']):+.3f}"
            )
        else:
            print(
                f"       spot@peak: g_exc={spot_peak_step['g_exc_nS']:.4f} "
                f"g_inh={spot_peak_step['g_inh_nS']:.4f} "
                f"num_exc={spot_peak_step['num_exc']:+.1f} "
                f"num_inh={spot_peak_step['num_inh']:+.1f} "
                f"v_pre={float(spot_peak_step['v_post']) - float(spot_peak_step['v_post_minus_pre']):+.3f}"
            )
            print(
                f"       bar @peak: g_exc={bar_peak_step['g_exc_nS']:.4f} "
                f"g_inh={bar_peak_step['g_inh_nS']:.4f} "
                f"num_exc={bar_peak_step['num_exc']:+.1f} "
                f"num_inh={bar_peak_step['num_inh']:+.1f} "
                f"v_pre={float(bar_peak_step['v_post']) - float(bar_peak_step['v_post_minus_pre']):+.3f}"
            )




def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_shared_cli(ap, default_run=DEFAULT_RUN_PATH)
    plot_trained.add_plot_timing_arguments(ap)
    plot_trained.add_plot_euler_argument(ap)
    plot_trained.add_plot_filter_argument(ap)
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
            "per-t step table prints only when false "
            "(filter=ca → ca/ca_pre/ca_post_minus_pre)"
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

    if args.radius != 0 and not any(task in train.SPOT_TASKS for task in cli.tasks):
        raise SystemExit("--radius requires a spot task")

    hex_mode = False
    if cli.xs is not None and cli.ys is not None:
        if len(cli.xs) != 1 or len(cli.ys) != 1:
            raise SystemExit(
                "hex mode needs exactly one --x and one --y; "
                "omit both for cost-radius averages"
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

    for run_idx, run_arg in enumerate(args.run):
        run_dir = plot_trained.resolve_run_dir(run_arg)
        _log(f"load_best {run_dir} ...")
        session, z, best_cost = plot_trained.load_best(run_dir)
        train_opts = plot_trained.load_train_opts(run_dir) or {}
        train_filter = train.expand_filter(train_opts.get("filter", "none"))
        session, z, timing_changed = plot_trained.maybe_override_stimulus_timing(
            run_dir=run_dir,
            session=session,
            z=z,
            **timing_kw,
            euler=args.euler,
            filter=args.filter,
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
        z_t = torch.tensor(
            np.asarray(z, dtype=np.float64),
            dtype=session.sim_dtype,
            device=session.device,
        )
        z_t, schema = plot_trained.apply_param_overrides(
            z_t, schema, session, param_edits,
        )
        session = session.with_schema(schema)
        params = train.materialize_from_opts(
            train.assign_params(z_t, schema, session.backend), session,
        )
        cost = float(train.calc_cost(z_t, session).item())

        spot_session_cache: dict[str, object] = {}
        bar_meta_cache: dict[str, tuple] = {}
        spot_by_cell: dict[str, dict[str, Any]] = {}
        bar_by_cell: dict[str, dict[str, Any]] = {}
        all_reports: list[dict[str, Any]] = []

        if not args.json:
            _log(f"== RUN {run_idx}: {run_dir} ==")
            _log(
                f"best_cost={best_cost:.6g}  cost={cost:.6g}  "
                f"mode={'hex' if hex_mode else 'average'}  "
                f"radius={args.radius}  "
                f"{time_window.kind}={time_window.start}:{time_window.stop}"
            )

        for task in cli.tasks:
            if task in train.SPOT_TASKS:
                if task not in spot_session_cache:
                    spot_session_cache[task] = plot_trained.session_for_task(
                        session, task,
                    )
                session_one = spot_session_cache[task]
                if hex_mode:
                    at_x = cli.xs[0]
                    at_y = cli.ys[0]
                    _log(
                        f"component forward {task} "
                        f"(spot hex=({at_x},{at_y}); batched) ..."
                    )
                    reports = analyze_spot_hex(
                        session_one,
                        params =params,
                        cell=cli.cells[0],
                        task=task,
                        at_x=float(at_x),
                        at_y=float(at_y),
                        node=args.node,
                        time_window=time_window,
                        train_filter=train_filter,
                    )
                else:
                    _log(
                        f"component forward {task} "
                        f"(spot radius={args.radius}; batched) ..."
                    )
                    reports = analyze_spot_average(
                        session_one,
                        params =params,
                        cells=cli.cells,
                        task=task,
                        time_window=time_window,
                        radius=args.radius,
                        train_filter=train_filter,
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
                at_x = cli.xs[0] if hex_mode else None
                at_y = cli.ys[0] if hex_mode else None
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
                        f"hex=({at_x},{at_y}) (batched, no full forward) ..."
                    )
                    reports_by_spec = analyze_bar_hex(
                        session,
                        params =params,
                        cell=cells_bar[0],
                        task=task,
                        spec_names=specs_ordered,
                        at_x=float(at_x),
                        at_y=float(at_y),
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
                        params =params,
                        cells=cells_bar,
                        task=task,
                        spec_names=specs_ordered,
                        time_window=time_window,
                        specs=specs,
                        grids=grids,
                    )
                overlay_by_cell: dict[str, list[dict[str, Any]]] = {
                    cell: [] for cell in cells_bar
                }
                for spec in specs_ordered:
                    for cell, rep in reports_by_spec[spec].items():
                        rep["best_cost"] = best_cost
                        rep["cost"] = cost
                        bar_by_cell[cell] = rep
                        all_reports.append(rep)
                        if multi_spec_plot:
                            overlay_by_cell[cell].append(rep)
                        _emit_report(
                            rep,
                            run_dir=run_dir,
                            do_print=not args.json,
                            do_plot=args.plot and not multi_spec_plot,
                            file_suffix=file_suffix,
                            html=args.html,
                        )
                if multi_spec_plot:
                    for cell in cells_bar:
                        reps = overlay_by_cell[cell]
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
