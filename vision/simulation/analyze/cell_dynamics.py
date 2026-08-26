from __future__ import annotations

from config import (
    parse_cells,
    session_kwargs_from_cli,
)

import json
import os
import sys
import config as _config
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
import hydra
import train
import figure.plot as plot
from figure.spread import contrast_from_pack
from figure.spot import pack_spot_cost_radii, resolve_spot_gts
from figure.panel import (
    plot_std_band,
)
from figure.plot import filter_figure_token, session_filter_figure_token
from neuron.filter_ca import filter_ca
from neuron.schema import param_from_entry
from task.mbar.pack import (
    bar_specs_from_task,
    filter_requested_specs,
    nodes_from_hexes,
    mbar_specs_by_cell,
    mbar_session_t0_grids,
)
from task.sbar.pack import sbar_direction_active
from task.sbar.sti_spec import gruntman_sbar_specs
from task.sbar.sti_geo import sti_hexes_at_xy
from task.spread.pack import cost_sti_hexes
from task.spot.pack import build_spot_center_readout
from task.spot.sti_geo import (
    resolve_spot,
    spot_sti_bs,
)
from train.cost import node_vals_from_param

__doc__ = """Borst / hp_lp v component analysis.

Consumers (CLI or ``import analyze.cell_dynamics``) must reuse this module's
forward helpers. Do not re-implement spot/bar readout + step loops in scratch/.

Time axis (read this before ``ms_shown`` / ``TimeWindow``)
------------------------------------------------------------
Two *different* knobs — do not mix them:

1. **Stimulus length** (``ms_pre=50`` / ``ms_sti=160`` / …):
   rebuilds the session sti (via ``figure.plot.override_session``).
   Unset = keep the run's train opts. These change *how long*
   pre/spot/response *are*, not which slice of an existing trace you plot.

2. **Analyze / plot window** (``ms_shown=[START,STOP]`` or ``t_rel_start`` /
   ``t_rel_stop``, mutually exclusive): which inclusive slice of the forward
   to sum and report. Default if both omitted: absolute ms ``0`` .. last sample.

``ms_shown`` is **absolute aligned ms**, never "ms before onset":

* **spot** / **spread**: aligned ``t = 0`` is trial start. Stimulus onset is at
  ``t_onset = t_from_ms(ms_pre, delta_ms=delta_ms_pre)`` (e.g. ``ms_pre=1000``,
  ``delta_ms_pre=5`` → ``t_onset=200`` ↔ **1000 ms**). Pre-sti is therefore
  ``ms_shown=[0,1000]`` (or ``[0,ms_pre]``), **not** ``[-1000,0]``.
  Negative START is wrong for spot/spread (aligned index goes negative; sum window
  collapses).
* **mbar**: aligned ``t = 0`` is bar ``t0`` at the node (crossing), so
  ``ms_shown`` is ms relative to that ``t0`` (negative START is valid).
* **sbar**: aligned ``t = 0`` is trial start, as for spot. Stimulus onset is
  ``ms_pre``; negative ``ms_shown`` values are invalid.

``t_rel_start`` / ``t_rel_stop`` are **t-index offsets from the |v_post_d|
peak** (not from onset, not absolute ms). Example: ``t_rel_start=-5 t_rel_stop=15``.

``TimeWindow(kind="ms", start, stop)`` uses the same absolute-aligned-ms rule as
``ms_shown``. ``kind="t_rel"`` matches ``t_rel_start`` / ``t_rel_stop``.
Spot R0-average API:
``analyze_spot_average(..., time_window=TimeWindow("ms", 0, ms_pre), radius=0)``.

Programmatic reuse
------------------
* Spot R0 / R1 average: ``analyze_spot_average`` (omit hex; ``radius=0|1``).
* Spot one hex: ``analyze_spot_hex``.
* Spread average / hex: ``analyze_spread_average`` / ``analyze_spread_hex``.
* Bar average / hex: ``analyze_bar_average`` / ``analyze_bar_hex``.
* Load run: ``figure.plot.load_best`` + ``params_from_z``; do not invent a
  second forward path.

Hydra (from ``simulation/``)
----------------------------
``cells=[L1,L2]`` ``analyze_runs=hp_lp/...`` ``tasks=spot`` ``x=`` ``y=``.
Default run is ``RUN_PATH``. Multiple cells/specs use Hydra list syntax, e.g.
``spec=[left_bright_w4,right_bright_w4]``; multiple values must use a list.

* Omit ``x`` / ``y``: cost-radius **average** (optional ``radius=0|1``, spot only).
* Exactly one ``x`` and one ``y``: **hex** (spread, spot, sbar, or mbar; one cell).
  Incompatible with ``radius`` (hex is sti-on only).
* Multiple x/y: rejected.

``analyze_figure=true|false``: PNGs under ``{run}/cell_dynamics/`` (default true).
``param_vals.a_h.L1=0.8`` / ``euler=im|ex`` / ``filter=none|ca`` /
``ms_pre=...``: CLI session overrides (same Hydra keys as ``figure.plot``).
Unset = keep the run ``train_opts.json``.

Examples
--------
  ../.venv/bin/python -m analyze.cell_dynamics \\
    cells=[L1,L2,Mi1] tasks=spot contrasts=bright radius=0 \\
    ms_shown=[0,1000] analyze_figure=false

  ../.venv/bin/python -m analyze.cell_dynamics \\
    cells=L3 tasks=spot contrasts=bright x=1 y=0

  ../.venv/bin/python -m analyze.cell_dynamics \\
    cells=Mi4 tasks=spread contrasts=bright x=0 y=0

  ../.venv/bin/python -m analyze.cell_dynamics \\
    cells=T4a tasks=mbar contrasts=bright \\
    spec=[left_bright_w4,right_bright_w4] \\
    '+param_vals.syn_strength_cell={Mi4\\:T4a:2.0,Mi9\\:T4a:1.0}' \\
    t_rel_start=-5 t_rel_stop=15
"""



@dataclass(frozen=True)
class TimeWindow:
    """Inclusive analyze window for component forward / finalize.

    ``kind="ms"``
        Absolute **aligned** ms (same as ``ms_shown``). Spot: ``0`` = trial
        start, onset ≈ ``ms_pre``; pre = ``TimeWindow("ms", 0, ms_pre)``.
        Never use negative ``start`` for spot "pre" (that is not onset-relative).
        Bar: ``0`` = bar ``t0`` at the node (negative ``start`` OK).
    ``kind="t_rel"``
        Integer t offsets from the |v_post_d| peak (``t_rel_start`` /
        ``t_rel_stop``).

    Stimulus length uses CLI ``ms_pre`` / ``ms_sti`` / … (rebuilds the session),
    not this window.
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
    """Parsed shared Hydra analyze selection."""

    cells: list[str]
    tasks: list[str]
    contrasts: list[str]
    specs_req: list[str] | None
    mids_req: list[float] | None
    xs: list | None
    ys: list | None


def resolve_shared_cli() -> SharedCli:
    cells = parse_cells(_config.ANALYZE_CELL_DYNAMICS.get("cells"))
    if not cells:
        raise SystemExit("cells is required")
    tasks = list(_config.TRAIN_CONFIG["tasks"])
    contrasts = list(_config.TRAIN_CONFIG["contrasts"])
    specs_req = parse_cells(_config.ANALYZE_CELL_DYNAMICS.get("spec"))
    mids_raw = _config.ANALYZE_CELL_DYNAMICS.get("mid")
    mids_req = plot.parse_at_xs(mids_raw)
    xs = plot.parse_at_xs(_config.FIGURE_PLOT.get("x"))
    ys = plot.parse_at_xs(_config.FIGURE_PLOT.get("y"))
    return SharedCli(
        cells=cells,
        tasks=tasks,
        contrasts=contrasts,
        specs_req=specs_req,
        mids_req=mids_req,
        xs=xs,
        ys=ys,
    )


# Component-step fields plotted vs time (series color key, ylabel/legend).
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
        "HP",
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

# First plot row when ``filter=ca`` (replaces the single ``v_post`` color key).
_CA_PLOT_ROW0: tuple[str, list[tuple[str, str]]] = (
    "v / Ca",
    [
        ("v_post", "v_post"),
        ("v_ca", "v_ca"),
        ("ca", "ca"),
    ],
)

_BORST_COMPONENTS = (
    "v_pre_d", "v_abs", "i_sti", "g_exc", "g_inh", "g_h", "g_h_rev",
    "num_exc", "num_inh", "num_leak", "num_i_h", "num_i_h_rev", "num_v",
    "num", "den",
)
_HP_LP_COMPONENTS = (
    "v_pre_d", "v_abs", "i_sti", "v_sti", "v_syn", "v_syn_exc", "v_syn_inh",
    "v_slow", "v_in", "v_hp", "dv_leak", "dv_hp",
)

_BORST_COMPONENT_FROM_SERIES: dict[str, str | None] = {
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
_HP_LP_COMPONENT_FROM_SERIES: dict[str, str | None] = {
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
    ("dt_over_cap", None), ("·(i_sti + ", None),
    ("e_exc·", None), ("g_exc", "g_exc"), (" + ", None),
    ("e_inh·", None), ("g_inh", "g_inh"), (" + ", None),
    ("E_leak·", None), ("g_leak", "g_leak"), (" + ", None),
    ("E_h·", None), ("g_h", "g_h"), (" + ", None),
    ("E_h_rev·", None), ("g_h_rev", "g_h_rev"),
    (")) / (1 + ", None),
    ("dt_over_cap", None), ("·(", None),
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
    ("dt_over_cap", None), ("·(", None),
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
    ("dt_over_cap", None), ("·(i_sti + ", None),
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
    ("dt_over_cap", None), ("·(", None),
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
# Series label → reuse another label's plot color (same row cycle).
_TRACE_COLOR_MATCH: dict[str, str] = {
    "dv_leak+dv_hp": "dv_leak",
    "v_ca": "v_post",
    "ca": "v_post",
}
# Report gt_report_key → plot panel series (train GT kind; never mix).
_GT_PLOT_PANEL: dict[str, str] = {
    "gt_v": "v_post",
    "gt_ca": "ca",
}
# Shared ylim across columns: borst current row; hp_lp HP + dv rows.
_BORST_ROW_SHARED_YLIM = frozenset({2})
_HP_LP_ROW_SHARED_YLIM = frozenset({2, 3})


@dataclass(frozen=True)
class _ComponentLayout:
    model: str
    components: tuple[str, ...]
    plot_panels: list[tuple[str, list[tuple[str, str]]]]
    component_from_series: dict[str, str | None]
    formula_g: list[tuple[str, str | None]]
    formula_i: list[tuple[str, str | None]]
    row_shared_ylim: frozenset[int]

    @property
    def n_component(self) -> int:
        return len(self.components)

    @property
    def i_v_abs(self) -> int:
        return self.components.index("v_abs")

    @property
    def i_v_pre_d(self) -> int:
        return self.components.index("v_pre_d")

    @property
    def n_col(self) -> int:
        return max(len(panel_series) for _, panel_series in self.plot_panels)


def _component_layout(model: str, euler: str, *, filter: str = "v") -> _ComponentLayout:
    """Build plot/component layout. ``filter`` is plot token ``v``|``ca`` (row-0 Ca cols)."""
    use_ca = str(filter) == "ca"
    if model == "borst":
        if euler == "implicit":
            formula_g, formula_i = _BORST_FORMULA_G_IMPLICIT, _BORST_FORMULA_I_IMPLICIT
        else:
            formula_g, formula_i = _BORST_FORMULA_G_EXPLICIT, _BORST_FORMULA_I_EXPLICIT
        panels = list(_BORST_PLOT_PANELS)
        if use_ca:
            panels[0] = _CA_PLOT_ROW0
        return _ComponentLayout(
            model="borst",
            components=_BORST_COMPONENTS,
            plot_panels=panels,
            component_from_series=_BORST_COMPONENT_FROM_SERIES,
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
        return _ComponentLayout(
            model="hp_lp",
            components=_HP_LP_COMPONENTS,
            plot_panels=panels,
            component_from_series=_HP_LP_COMPONENT_FROM_SERIES,
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


def _plot_trace_colors(colors: list[str], layout: _ComponentLayout) -> dict[str, str]:
    """Map trace legend label → subplot color (hex index within its row)."""
    colors_by_label: dict[str, str] = {}
    for _panel_ylabel, panel_series in layout.plot_panels:
        for series_idx, (_series, label) in enumerate(panel_series):
            if label in _BLACK_TRACE_LABELS:
                colors_by_label[label] = "0.0"
            else:
                colors_by_label[label] = colors[series_idx % len(colors)]
    for label, src in _TRACE_COLOR_MATCH.items():
        if label in colors_by_label and src in colors_by_label:
            colors_by_label[label] = colors_by_label[src]
    return colors_by_label


def _g_e_note(label: str, *, e_leak: float, globs: dict[str, Any], params: dict[str, Any] | None = None) -> str | None:
    """Reversal annotation for a conductance subplot (``e_exc=+10 mV`` …)."""
    if label == "g_leak":
        return f"E_leak={e_leak:+g} mV"
    notes = {
        "g_exc": "e_exc",
        "g_inh": "e_inh",
        "g_h": "e_h",
        "g_h_rev": "e_h_rev",
    }
    param = notes.get(label)
    if param is None:
        return None
    src = globs
    if param not in src and params is not None and param in params:
        src = params
    if param not in src:
        return None
    pretty = {"e_h": "E_h", "e_h_rev": "E_h_rev"}.get(param, param)
    return f"{pretty}={src[param]:+g} mV"


def _add_component_formula_row(
    fig,
    colors: list[str],
    formula: list[tuple[str, str | None]],
    layout: _ComponentLayout,
    *,
    y: float,
    fontsize: int = 9,
) -> None:
    """One formula line; ``formula`` entries are ``(label, series)`` (series → color, or None)."""
    tc = _plot_trace_colors(colors, layout)
    x = 0.02
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for label, series in formula:
        color = tc[series] if series else "0.2"
        txt = fig.text(
            x, y, label, transform=fig.transFigure,
            ha="left", va="top", fontsize=fontsize, color=color,
        )
        bbox = txt.get_window_extent(renderer=renderer)
        x = inv.transform((bbox.x1, bbox.y0))[0] + 0.003


def _finish_component_figure(
    fig, title: str, colors: list[str], layout: _ComponentLayout,
) -> None:
    fig.suptitle(title, fontsize=11, y=0.995)
    _add_component_formula_row(fig, colors, layout.formula_g, layout, y=0.975, fontsize=8)
    _add_component_formula_row(fig, colors, layout.formula_i, layout, y=0.955, fontsize=9)
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.88))


def _model_driver(session):
    from neuron.forward import MODEL_DRIVERS

    try:
        return MODEL_DRIVERS[session.model]
    except KeyError as exc:
        raise SystemExit(
            f"cell_dynamics supports {tuple(MODEL_DRIVERS)}; got {session.model!r}"
        ) from exc


def _drive_from_i_sti(
    session, params, i_sti: torch.Tensor, *, pack=None,
) -> torch.Tensor:
    """Apply indexed spot/sbar stimulus amplitudes to pack ``i_sti``."""
    from neuron.forward import inject_a_sti_mid, inject_a_sti_radius

    pack = pack or session.primary_pack
    if i_sti.dim() == 2:
        i_sti = i_sti.unsqueeze(0)
    i_sti = inject_a_sti_radius(i_sti, params, pack)
    return inject_a_sti_mid(i_sti, params, pack)


def _equilibrate(session, params, i_sti_b: torch.Tensor, t_onset: int):
    """Equilibrate to ``t_onset``; returns ``v``."""
    _component_layout(session.model, session.euler)  # validate early
    n_b, n_t, n_node = i_sti_b.shape
    drv = _model_driver(session)
    model = session.model
    if model == "hp_lp":
        v_slow, v = drv.pre_steady(session, params, n_b, i_sti=i_sti_b)
        for t in range(1, min(t_onset, n_t)):
            v_slow, v = drv.step(
                v_slow, v, params, i_sti_b[:, t - 1], session,
                delta_ms=train.step_delta_ms(session, t, t_onset),
            )
    else:
        u, u_rev, v = drv.pre_steady(session, params, n_b, i_sti=i_sti_b)
        for t in range(1, min(t_onset, n_t)):
            u, u_rev, v = drv.step(
                u, u_rev, v, params, i_sti_b[:, t - 1], session,
                delta_ms=train.step_delta_ms(session, t, t_onset),
            )
    return v


def _component_nodes_borst(
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
    b: int = 0,
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
        nodes_t = torch.as_tensor(nodes, device=v_pre.device, dtype=torch.long)
        sig_active = (
            sig_t[b, nodes_t] if sig_t.dim() > 1 else sig_t[nodes_t]
        )
        packed = torch.stack(
            (
                v_pre[b, nodes_t],
                g_exc[b, nodes_t],
                g_inh[b, nodes_t],
                g_h[b, nodes_t],
                g_h_rev[b, nodes_t],
                sig_active,
                e_leak[nodes_t],
                v_post[b, nodes_t],
            ),
            dim=0,
        ).detach().cpu().numpy()
        v_pre_np = packed[0]
        ref = (
            v_onset[b, nodes] if np.ndim(v_onset) == 2 else v_onset[nodes]
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


def _component_nodes_hp_lp(
    v_pre,
    v_post,
    component_t: dict[str, torch.Tensor],
    nodes: np.ndarray,
    v_onset: np.ndarray,
    *,
    b: int = 0,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Slice hp_lp ODE component tensors at ``nodes``."""
    nodes = np.asarray(nodes, dtype=np.int64)
    with torch.no_grad():
        nodes_t = torch.as_tensor(nodes, device=v_pre.device, dtype=torch.long)
        v_pre_np = v_pre[b, nodes_t].detach().cpu().numpy()
        v_abs = v_post[b, nodes_t].detach().cpu().numpy()
        ref = (
            v_onset[b, nodes] if np.ndim(v_onset) == 2 else v_onset[nodes]
        )
        component = {
            "v_pre_d": v_pre_np - ref,
            "v_abs": v_abs,
        }
        for component_tok in (
            "i_sti", "v_sti", "v_syn", "v_syn_exc", "v_syn_inh", "v_slow", "v_in", "v_hp",
            "dv_leak", "dv_hp",
        ):
            trace = component_t[component_tok]
            component[component_tok] = (
                trace[b, nodes_t] if trace.dim() > 1 else trace[nodes_t]
            ).detach().cpu().numpy()
        v_post_minus_pre_active = v_abs - v_pre_np
    return component, v_post_minus_pre_active


def _log(msg: str) -> None:
    print(msg, flush=True)


def _component_matrix(component: dict[str, np.ndarray], components: tuple[str, ...]) -> np.ndarray:
    """Stack components to ``(n_node, n_component)`` for vectorized sums."""
    return np.column_stack([component[component_tok] for component_tok in components])


def _std_from_sum_and_sum_sq(sum_: float, sum_sq: float, n_node: int) -> float:
    """Match ``figure.panel.std_from_traces`` (population std)."""
    if n_node <= 1:
        return 0.0
    mean = sum_ / n_node
    var = sum_sq / n_node - mean * mean
    if var <= 0.0:
        return 0.0
    return float(np.sqrt(var))


def _step_std(
    sums: dict[str, float], sum_sqs: dict[str, float], n_node: int, layout: _ComponentLayout,
) -> dict[str, float]:
    return {
        series: (
            0.0 if component is None else _std_from_sum_and_sum_sq(
                sums[component], sum_sqs[component], n_node,
            )
        )
        for series, component in layout.component_from_series.items()
    }


def _step_from_sums(
    *,
    t: int,
    t_rel: int,
    ti: int,
    v_post_val: float,
    sums: dict[str, float],
    sum_sqs: dict[str, float],
    v_post_minus_pre_sum: float,
    n_node: int,
    layout: _ComponentLayout,
    g_leak: float = 0.0,
    dt_over_cap: float = 0.0,
) -> dict[str, Any]:
    """One step dict from per-component sums over ``n_node`` nodes."""
    if n_node <= 0:
        raise ValueError("empty node set for mean component")
    base = {
        "t": t,
        "t_rel": t_rel,
        "ti": ti,
        "v_post": float(v_post_val),
        "v_pre_d": sums["v_pre_d"] / n_node,
        "v_post_minus_pre": v_post_minus_pre_sum / n_node,
        "i_sti": sums["i_sti"] / n_node,
        "std": _step_std(sums, sum_sqs, n_node, layout),
        "n_node": n_node,
    }
    if layout.model == "borst":
        num = sums["num"] / n_node
        den = sums["den"] / n_node
        base.update({
            "g_exc_nS": sums["g_exc"] / n_node,
            "g_inh_nS": sums["g_inh"] / n_node,
            "g_leak_nS": float(g_leak),
            "dt_over_cap": float(dt_over_cap),
            "g_h_nS": sums["g_h"] / n_node,
            "g_h_rev_nS": sums["g_h_rev"] / n_node,
            "num_exc": sums["num_exc"] / n_node,
            "num_inh": sums["num_inh"] / n_node,
            "num_leak": sums["num_leak"] / n_node,
            "num_i_h": sums["num_i_h"] / n_node,
            "num_i_h_rev": sums["num_i_h_rev"] / n_node,
            "num_v": sums["num_v"] / n_node,
            "num": num,
            "den": den,
            "num_over_den": sums["v_abs"] / n_node,
        })
        return base
    base.update({
        "v_syn": sums["v_syn"] / n_node,
        "v_syn_exc": sums["v_syn_exc"] / n_node,
        "v_syn_inh": -sums["v_syn_inh"] / n_node,
        "v_sti": sums["v_sti"] / n_node,
        "v_slow": sums["v_slow"] / n_node,
        "v_in": sums["v_in"] / n_node,
        "v_hp": sums["v_hp"] / n_node,
        "dv_leak": sums["dv_leak"] / n_node,
        "dv_hp": sums["dv_hp"] / n_node,
        "dv_leak_plus_hp": (sums["dv_leak"] + sums["dv_hp"]) / n_node,
    })
    return base


@dataclass
class _ComponentB:
    """One i_sti b row for the shared component forward."""

    nodes: np.ndarray
    node_t0s: np.ndarray
    n_t_aligned: int
    cell_from_node: dict[int, str]
    nodes_by_cell: dict[str, np.ndarray]


@dataclass
class _ComponentSums:
    sums: list[dict[str, np.ndarray]]
    sum_sqs: list[dict[str, np.ndarray]]
    n_node: list[dict[str, np.ndarray]]
    v_post_minus_pre_sums: list[dict[str, np.ndarray]]
    layout: _ComponentLayout
    v_ca_sums: list[dict[str, np.ndarray]] | None = None
    ca_sums: list[dict[str, np.ndarray]] | None = None
    ca_pre_sums: list[dict[str, np.ndarray]] | None = None


def _forward_component(
    session,
    params,
    i_sti: torch.Tensor,
    bs: list[_ComponentB],
    cells: list[str],
    *,
    t_start: int | None = None,
    t_stop: int | None = None,
    stimulus_pack=None,
    stimulus_bs: list[int] | None = None,
) -> _ComponentSums:
    """Step from t=0 (same loop as ``forward_v``); sum component.

    Shared by bar/spot average and bar/spot hex.
    ``v_onset`` matches ``forward_v`` (``v`` at ``t_onset - 1``). Aligned index
    ``t = t_global - node_t0s``. v_post is mean absolute ``v_abs``; STD uses sum /
    sum_sqs like ``std_from_traces``. When ``filter=ca``, also track ``v_ca`` /
    ``ca`` via ``v_ca_from_v`` + ``filter_ca`` (same as ``forward_ca``).

    If ``t_start``/``t_stop`` are set (from ``--ms-shown`` via ``t_from_ms``), only
    sum inside that inclusive aligned window; cheap steps outside it;
    break after every node has passed ``t_stop``.
    """
    if not bs:
        raise SystemExit("component forward requires at least one b")
    if (t_start is None) ^ (t_stop is None):
        raise SystemExit("t_start and t_stop must both be set or both omitted")
    if t_start is not None and t_start > t_stop:
        raise SystemExit(f"t_start={t_start} > t_stop={t_stop}")
    n_b, n_t, n_node = i_sti.shape
    if n_b != len(bs):
        raise SystemExit(f"i_sti n_b={n_b} != len(bs)={len(bs)}")

    layout = _component_layout(session.model, session.euler)
    if stimulus_bs is None:
        drive = _drive_from_i_sti(
            session, params, i_sti, pack=stimulus_pack,
        )
    else:
        if stimulus_pack is None:
            raise SystemExit("stimulus_bs requires stimulus_pack")
        full_i_sti = session.pack_i_sti(stimulus_pack)
        full_drive = _drive_from_i_sti(
            session, params, full_i_sti, pack=stimulus_pack,
        )
        drive = full_drive[stimulus_bs]
    onset_pack = stimulus_pack or session.primary_pack
    t_onset = train.pack_t_onset(onset_pack)

    t_last: int | None = None
    if t_stop is not None:
        t_last = max(int(plan.node_t0s.max()) + int(t_stop) for plan in bs)

    # Same ref as forward_v: v at t_onset-1, then restart so pre is stepped and summed.
    v_onset = _equilibrate(session, params, drive, t_onset).detach().cpu().numpy().copy()
    n_component = layout.n_component
    drv = _model_driver(session)
    model = session.model
    if model == "hp_lp":
        v_slow, v = drv.pre_steady(session, params, n_b, i_sti=drive)
        u = u_rev = None
    else:
        u, u_rev, v = drv.pre_steady(session, params, n_b, i_sti=drive)
        v_slow = None

    use_ca = session_filter_figure_token(session) == "ca"
    ca = train.v_ca_from_v(v, params, session) if use_ca else None
    tau_ca = (
        torch.clamp(params["tau_ca"], min=float(session.delta_ms)) if use_ca else None
    )

    b_sums: list[dict[str, np.ndarray]] = []
    b_sum_sqs: list[dict[str, np.ndarray]] = []
    b_n_node: list[dict[str, np.ndarray]] = []
    b_v_post_minus_pre_sums: list[dict[str, np.ndarray]] = []
    b_v_ca_sums: list[dict[str, np.ndarray]] | None = [] if use_ca else None
    b_ca_sums: list[dict[str, np.ndarray]] | None = [] if use_ca else None
    b_ca_pre_sums: list[dict[str, np.ndarray]] | None = [] if use_ca else None
    for plan in bs:
        n_t_aligned = plan.n_t_aligned
        b_sums.append({cell: np.zeros((n_t_aligned, n_component), dtype=float) for cell in cells})
        b_sum_sqs.append({cell: np.zeros((n_t_aligned, n_component), dtype=float) for cell in cells})
        b_n_node.append({cell: np.zeros(n_t_aligned, dtype=np.int64) for cell in cells})
        b_v_post_minus_pre_sums.append({cell: np.zeros(n_t_aligned, dtype=float) for cell in cells})
        if use_ca:
            b_v_ca_sums.append({cell: np.zeros(n_t_aligned, dtype=float) for cell in cells})
            b_ca_sums.append({cell: np.zeros(n_t_aligned, dtype=float) for cell in cells})
            b_ca_pre_sums.append({cell: np.zeros(n_t_aligned, dtype=float) for cell in cells})

    cell_idxs_from_node_id = [_cell_idx_from_node_id(plan, cells) for plan in bs]

    for t_global in range(1, n_t):
        if t_last is not None and t_global > t_last:
            break
        sig_t = drive[:, t_global - 1]
        actives: list[tuple[np.ndarray, np.ndarray] | None] = []
        need_component = False
        for plan in bs:
            nodes = plan.nodes
            node_aligned_t = t_global - plan.node_t0s
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
                    nodes[in_aligned_t], node_aligned_t[in_aligned_t].astype(np.int64),
                ))
            else:
                actives.append(None)

        step_dt = train.step_delta_ms(session, t_global, t_onset)
        if not need_component:
            if model == "hp_lp":
                v_slow, v = drv.step(
                    v_slow, v, params, sig_t, session,
                    delta_ms=step_dt,
                )
            else:
                u, u_rev, v = drv.step(
                    u, u_rev, v, params, sig_t, session,
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
            if layout.model == "borst":
                u, u_rev, v, (g_exc, g_inh, g_h, g_h_rev) = drv.step(
                    u, u_rev, v, params, sig_t, session,
                    delta_ms=step_dt, return_component=True,
                )
            else:
                v_slow, v, component_t = drv.step(
                    v_slow, v, params, sig_t, session,
                    delta_ms=step_dt, return_component=True,
                )
            v_ca = None
            if ca is not None:
                v_ca = train.v_ca_from_v(v, params, session)
                ca = filter_ca(
                    ca_pre, v_ca, delta_ms=step_dt, tau_ca=tau_ca,
                )

        for b, plan in enumerate(bs):
            active_pack = actives[b]
            if active_pack is None:
                continue
            active_node, active_t = active_pack
            if layout.model == "borst":
                component, v_post_minus_pre_active = _component_nodes_borst(
                    v_pre, v, g_exc, g_inh, g_h, g_h_rev, sig_t,
                    params["e_leak"], active_node, v_onset, b=b,
                    delta_ms=step_dt, cap=session.cap, g_leak=session.g_leak,
                    e_exc=session.e_exc, e_inh=session.e_inh, e_h=session.e_h,
                    euler=session.euler,
                )
            else:
                component, v_post_minus_pre_active = _component_nodes_hp_lp(
                    v_pre, v, component_t, active_node, v_onset, b=b,
                )
            component_mat = _component_matrix(component, layout.components)
            cell_idx_from_node_id = cell_idxs_from_node_id[b]
            active_cell_idxs = cell_idx_from_node_id[active_node]
            v_ca_active = ca_post_active = ca_pre_active = None
            if ca is not None:
                active_node_t = torch.as_tensor(
                    active_node, device=ca.device, dtype=torch.long,
                )
                v_ca_active = v_ca[b, active_node_t].detach().cpu().numpy()
                ca_post_active = ca[b, active_node_t].detach().cpu().numpy()
                ca_pre_active = ca_pre[b, active_node_t].detach().cpu().numpy()
            for cell_idx, cell in enumerate(cells):
                mask = active_cell_idxs == cell_idx
                if not np.any(mask):
                    continue
                ts = active_t[mask]
                chunk = component_mat[mask]
                np.add.at(b_sums[b][cell], ts, chunk)
                np.add.at(b_sum_sqs[b][cell], ts, chunk * chunk)
                np.add.at(
                    b_v_post_minus_pre_sums[b][cell], ts, v_post_minus_pre_active[mask],
                )
                np.add.at(b_n_node[b][cell], ts, 1)
                if v_ca_active is not None:
                    np.add.at(b_v_ca_sums[b][cell], ts, v_ca_active[mask])
                    np.add.at(b_ca_sums[b][cell], ts, ca_post_active[mask])
                    np.add.at(b_ca_pre_sums[b][cell], ts, ca_pre_active[mask])

    return _ComponentSums(
        sums=b_sums,
        sum_sqs=b_sum_sqs,
        n_node=b_n_node,
        v_post_minus_pre_sums=b_v_post_minus_pre_sums,
        layout=layout,
        v_ca_sums=b_v_ca_sums,
        ca_sums=b_ca_sums,
        ca_pre_sums=b_ca_pre_sums,
    )


def _v_post_from_sums(
    sums: np.ndarray, n_node: np.ndarray, layout: _ComponentLayout,
) -> np.ndarray:
    """Mean absolute v_post from summed ``v_abs``."""
    n_node_div = np.maximum(n_node, 1)
    v_post = sums[:, layout.i_v_abs] / n_node_div
    v_post[n_node == 0] = 0.0
    return v_post


def _v_post_d_from_sums(
    sums: np.ndarray, v_post_minus_pre_sums: np.ndarray, n_node: np.ndarray,
    layout: _ComponentLayout,
) -> np.ndarray:
    """Mean ``v_post_d`` = v_post − v_onset = v_pre_d + v_post_minus_pre."""
    n_node_div = np.maximum(n_node, 1)
    v_post_d = sums[:, layout.i_v_pre_d] / n_node_div + v_post_minus_pre_sums / n_node_div
    v_post_d[n_node == 0] = 0.0
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
    contrast: str,
    spec: str | None,
    mode: str,
    before_t: int,
    nodes: np.ndarray,
    params,
    session,
    sums: np.ndarray,
    sum_sqs: np.ndarray,
    n_node: np.ndarray,
    v_post_minus_pre_sums: np.ndarray,
    v_post: np.ndarray,
    time_window: TimeWindow,
    ti_mode: str,
    component_layout: _ComponentLayout,
    extra: dict[str, Any] | None = None,
    v_ca_sums: np.ndarray | None = None,
    ca_sums: np.ndarray | None = None,
    ca_pre_sums: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build one report dict from a single b×cell sums row."""
    # Peak on |v_post_d| (= |v_post − v_onset| = |v_pre_d + v_post_minus_pre|).
    v_post_d = _v_post_d_from_sums(sums, v_post_minus_pre_sums, n_node, component_layout)
    n_t = int(v_post_d.size)
    dt = float(session.delta_ms)
    dt_over_cap = dt / float(session.cap)
    if time_window.kind == "ms":
        t_lo = train.t_from_ms(time_window.start, delta_ms=dt)
        t_hi = train.t_from_ms(time_window.stop, delta_ms=dt)
        if t_lo < 0 or t_hi >= n_t or t_lo > t_hi:
            raise SystemExit(
                f"--ms-shown {time_window.start:g},{time_window.stop:g} "
                f"(t={t_lo}:{t_hi}) out of range for sums length {n_t}"
            )
        trace_slice = v_post_d[t_lo:t_hi + 1]
        peak_t = t_lo + int(np.argmax(np.abs(trace_slice))) if trace_slice.size else t_lo
    else:
        peak_t = _v_post_d_peak_t_rel(v_post_d, before_t)
        t_lo = max(0, peak_t + int(time_window.start))
        t_hi = min(n_t - 1, peak_t + int(time_window.stop))
    steps: list[dict[str, Any]] = []
    peak_step: dict[str, Any] | None = None
    components = component_layout.components
    for t in range(t_lo, t_hi + 1):
        n_node_t = int(n_node[t])
        if n_node_t == 0:
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
            sums={
                component_tok: float(sums[t][component_idx])
                for component_idx, component_tok in enumerate(components)
            },
            sum_sqs={
                component_tok: float(sum_sqs[t][component_idx])
                for component_idx, component_tok in enumerate(components)
            },
            v_post_minus_pre_sum=float(v_post_minus_pre_sums[t]),
            n_node=n_node_t,
            layout=component_layout,
            g_leak=float(session.g_leak),
            dt_over_cap=dt_over_cap,
        )
        if v_ca_sums is not None:
            step["v_ca"] = float(v_ca_sums[t] / n_node_t)
        if ca_sums is not None and ca_pre_sums is not None:
            ca = float(ca_sums[t] / n_node_t)
            ca_pre = float(ca_pre_sums[t] / n_node_t)
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
        "model": component_layout.model,
        "cell": cell,
        "n_node": int(nodes.size),
        "task": task,
        "contrast": contrast,
        "spec": spec,
        "filter": session_filter_figure_token(session),
        "before_t": before_t,
        "time_window_kind": time_window.kind,
        "time_window": [time_window.start, time_window.stop],
        "t_window": [t_lo, t_hi],
        "v_post_d_peak_t": peak_t,
        "v_post_d_peak": float(v_post_d[peak_t]),
        "v_post_d_sign": _sign(float(v_post_d[peak_t])),
        "v_post_d_onset_t": onset,
        "params": _node_params(params, session, int(nodes[0])),
        "globals": _globals(session),
        "steps": steps,
        "peak_step": peak_step,
        "peak_drive": _dominant_drive_from_step(
            peak_step, model=component_layout.model,
        ),
        "v_post": v_post.tolist(),
    }
    if v_ca_sums is not None:
        n_node_div = np.maximum(n_node, 1)
        v_ca_trace = v_ca_sums / n_node_div
        v_ca_trace[n_node == 0] = 0.0
        report["v_ca"] = v_ca_trace.tolist()
    if ca_sums is not None:
        n_node_div = np.maximum(n_node, 1)
        ca_trace = ca_sums / n_node_div
        ca_trace[n_node == 0] = 0.0
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
    v_post_d = np.asarray(v_post_d, dtype=float)
    if before_t is not None and 0 < before_t < v_post_d.size:
        stop = v_post_d.size
        if horizon is not None:
            stop = min(stop, before_t + int(horizon))
        post = v_post_d[before_t:stop]
        if post.size == 0:
            return int(before_t)
        return int(before_t + int(np.argmax(np.abs(post))))
    stop = v_post_d.size if horizon is None else min(v_post_d.size, int(horizon))
    return int(np.argmax(np.abs(v_post_d[:stop])))


def _sign(v: float, *, eps: float = 1e-3) -> str:
    if v > eps:
        return "+"
    if v < -eps:
        return "-"
    return "0"


def _node_params(params, session, node: int) -> dict[str, float]:
    connectome = session.connectome
    for param in ("a_gt", "bias_gt"):
        if param not in params:
            raise SystemExit(f"params missing {param}")
    nodes = torch.tensor([node], dtype=torch.long)
    a_gt = float(node_vals_from_param(params, "a_gt", nodes, connectome)[0])
    bias_gt = float(node_vals_from_param(params, "bias_gt", nodes, connectome)[0])
    if session.model == "hp_lp":
        return {
            "a_in": float(params["a_in"][node]),
            "a_out": float(params["a_out"][node]),
            "v_th": float(params["v_th"][node]),
            "a_gt": a_gt,
            "bias_gt": bias_gt,
            "e_leak": float(params["e_leak"][node]),
            "tau_lp_ms": float(params["tau_lp"][node]),
            "tau_hp_rise_ms": float(params["tau_hp_rise"][node]),
            "tau_hp_fall_ms": float(params["tau_hp_fall"][node]),
            "a_h": float(params["a_h"][node]),
        }
    e_leak = float(params["e_leak"][node])
    return {
        "a_in": float(params["a_in"][node]),
        "a_out": float(params["a_out"][node]),
        "a_gt": a_gt,
        "bias_gt": bias_gt,
        "v_th": float(params["v_th"][node]),
        "h_g_max": float(session.h_g_max),
        "a_h": float(params["a_h"][node]),
        "a_h_rev": float(params["a_h_rev"][node]),
        "e_leak": e_leak,
        "e_h_rev": float(train.e_h_rev(e_leak, session.e_h)),
    }


def _globals(session):
    pack = session.primary_pack
    t_onset = train.pack_t_onset(pack)
    dt = float(session.delta_ms)
    if session.model == "hp_lp":
        return {
            "delta_ms": dt,
            "v_clamp": float(session.v_clamp),
            "g_leak_nS": float(session.g_leak),
            "euler": str(session.euler),
            "t_onset": t_onset,
        }
    return {
        "e_exc": float(session.e_exc),
        "e_inh": float(session.e_inh),
        "e_h": float(session.e_h),
        "g_leak_nS": float(session.g_leak),
        "dt_over_cap": dt / float(session.cap),
        "delta_ms": dt,
        "euler": str(session.euler),
        "t_onset": t_onset,
    }


def cell_from_node(nodes_by_cell: dict[str, np.ndarray]) -> dict[int, str]:
    return {
        int(node): cell
        for cell, cell_nodes in nodes_by_cell.items()
        for node in np.asarray(cell_nodes, dtype=np.int64).ravel()
    }


def _cell_idx_from_node_id(plan: _ComponentB, cells: list[str]) -> np.ndarray:
    """Dense ``node_cell_idx[node]`` in ``cells`` (-1 if absent)."""
    cell_idx = dict(zip(cells, range(len(cells))))
    if plan.nodes.size == 0:
        return np.empty(0, dtype=np.int32)
    node_cell_idx = np.full(int(plan.nodes.max()) + 1, -1, dtype=np.int32)
    for node_id, cname in plan.cell_from_node.items():
        i = cell_idx.get(cname)
        if i is not None:
            node_cell_idx[int(node_id)] = i
    return node_cell_idx


def _merge_component_sums(
    component_sums: _ComponentSums,
    component_bs: list[_ComponentB],
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
    """Sum per-cell sums rows across run bs (spot multi-sti mean)."""
    n_component = component_sums.layout.n_component
    sums = {cell: np.zeros((n_t_aligned, n_component), dtype=float) for cell in cells}
    sum_sqs = {cell: np.zeros((n_t_aligned, n_component), dtype=float) for cell in cells}
    n_node = {cell: np.zeros(n_t_aligned, dtype=np.int64) for cell in cells}
    v_post_minus_pre_sums = {cell: np.zeros(n_t_aligned, dtype=float) for cell in cells}
    nodes_ref = {cell: np.zeros(0, dtype=np.int64) for cell in cells}
    v_ca_sums = (
        {cell: np.zeros(n_t_aligned, dtype=float) for cell in cells}
        if component_sums.v_ca_sums is not None else None
    )
    ca_sums = (
        {cell: np.zeros(n_t_aligned, dtype=float) for cell in cells}
        if component_sums.ca_sums is not None else None
    )
    ca_pre_sums = (
        {cell: np.zeros(n_t_aligned, dtype=float) for cell in cells}
        if component_sums.ca_pre_sums is not None else None
    )
    for b, plan in enumerate(component_bs):
        for cell in cells:
            if cell not in plan.nodes_by_cell:
                continue
            us = plan.nodes_by_cell[cell]
            if us.size == 0:
                continue
            if nodes_ref[cell].size == 0:
                nodes_ref[cell] = us
            sums[cell] += component_sums.sums[b][cell]
            sum_sqs[cell] += component_sums.sum_sqs[b][cell]
            n_node[cell] += component_sums.n_node[b][cell]
            v_post_minus_pre_sums[cell] += component_sums.v_post_minus_pre_sums[b][cell]
            if v_ca_sums is not None:
                v_ca_sums[cell] += component_sums.v_ca_sums[b][cell]
                ca_sums[cell] += component_sums.ca_sums[b][cell]
                ca_pre_sums[cell] += component_sums.ca_pre_sums[b][cell]
    return (
        sums, sum_sqs, n_node, v_post_minus_pre_sums, nodes_ref,
        v_ca_sums, ca_sums, ca_pre_sums,
    )


def _build_component_b(
    nodes_by_cell: dict[str, np.ndarray],
    *,
    t0_bn_row: np.ndarray,
    n_t_aligned: int,
) -> _ComponentB:
    """Build one run b; ``node_t0s[i] = t0_bn_row[nodes[i]]``."""
    nodes = np.unique(np.concatenate([ids for ids in nodes_by_cell.values()]))
    return _ComponentB(
        nodes=nodes,
        node_t0s=np.asarray(t0_bn_row[nodes], dtype=np.int64),
        n_t_aligned=int(n_t_aligned),
        cell_from_node=cell_from_node(nodes_by_cell),
        nodes_by_cell=nodes_by_cell,
    )


# ---------------------------------------------------------------------------
# Average bar components (cost-radius)
# ---------------------------------------------------------------------------


def _bar_meta(session, task: str, contrast: str):
    """One-shot ``(specs, grids)`` for a static/moving-bar task×contrast."""
    if task == "mbar":
        specs = bar_specs_from_task(session, task, contrast)
    elif task == "sbar":
        opts = dict((session.train_opts or {}).get("sbar_sti_opts") or {})
        specs = gruntman_sbar_specs(
            contrasts=(contrast,),
            bar_directions=opts["bar_directions"],
            shift_radius=int(opts.get("shift_mid", opts.get("shift_radius", 0))),
        )
    else:
        raise SystemExit(f"unsupported bar task {task!r}")
    pack = session.packs[task][contrast]
    if task == "sbar":
        return specs, None
    grids = mbar_session_t0_grids(
        session, specs, pack.cost_radius, int(session.n_t),
        t_onset=train.pack_t_onset(pack),
        delta_ms=float(session.delta_ms),
    )
    return specs, grids


def _bar_specs_requested(
    session,
    task: str,
    contrast: str,
    cells: list[str],
    requested: list[str] | None,
    *,
    specs=None,
    grids=None,
) -> list[str]:
    """Spec list for average-mode bar without a component forward readout."""
    if specs is None or grids is None:
        specs, grids = _bar_meta(session, task, contrast)
    spec_defs = list(specs)
    specs = [spec.token for spec in spec_defs]
    if task == "mbar":
        specs_by_cell = mbar_specs_by_cell(
            session, task, contrast, grids.side,
        )
    else:
        specs_by_cell = {
            cell: [
                spec.token for spec in spec_defs
                if sbar_direction_active(cell, spec.direction)
            ]
            for cell in cells
        }
    spec_tokens: list[str] = []
    for cell in cells:
        for token in specs_by_cell.get(cell, specs):
            if token in specs and token not in spec_tokens:
                spec_tokens.append(token)
    available = spec_tokens or list(specs)
    try:
        if requested is not None:
            contrast_names = tuple(
                str(name) for name in ((session.train_opts or {}).get("i_sti") or {})
            )
            requested_here = [
                token for token in requested
                if f"_{contrast}_" in token
                or not any(f"_{name}_" in token for name in contrast_names)
            ]
            if not requested_here:
                return []
            return filter_requested_specs(available, requested_here)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return available


def _resolve_bar_spec_i_sti(
    session,
    task: str,
    contrast: str,
    spec_tokens: list[str],
    *,
    specs=None,
    grids=None,
):
    """Validate specs; return ``(pack, specs, grids, spec_bs, i_sti, t0_bn)``."""
    if task not in ("mbar", "sbar"):
        raise SystemExit(f"unsupported task {task!r}")
    if not spec_tokens:
        raise SystemExit("bar component forward requires at least one spec")
    pack = session.packs[task][contrast]
    if specs is None or grids is None:
        specs, grids = _bar_meta(session, task, contrast)
    spec_b = {
        spec.token: spec_idx for spec_idx, spec in enumerate(specs)
    }
    missing = [token for token in spec_tokens if token not in spec_b]
    if missing:
        raise SystemExit(
            f"spec(s) {missing} not in {[spec.token for spec in specs]}"
        )
    spec_bs = [spec_b[token] for token in spec_tokens]
    if task == "mbar":
        t0_bn = np.asarray(grids.t0_bn)
    else:
        n_node = int(pack.i_sti.shape[-1])
        t0_bn = np.zeros((len(specs), n_node), dtype=np.int64)
    return pack, specs, grids, spec_bs, pack.i_sti[spec_bs], t0_bn


def _analyze_component_forward(
    session,
    *,
    params,
    cells: list[str],
    task: str,
    contrast: str,
    i_sti: torch.Tensor,
    component_bs: list[_ComponentB],
    before_t: list[int],
    b_specs: list[str | None],
    time_window: TimeWindow,
    mode: str,
    ti_mode: str,
    merge_bs: bool = False,
    extra: dict[str, Any] | None = None,
    extra_from_cell=None,
    n_node_from_cell=None,
    stimulus_pack=None,
    stimulus_bs: list[int] | None = None,
):
    """Shared spot/bar: ``_forward_component`` → finalize reports.

    * ``merge_bs=False`` (bar): ``reports[spec][cell]``; ``b_specs`` are str.
    * ``merge_bs=True`` (spot): sum across bs → ``reports[cell]``.
    """
    if not component_bs:
        raise SystemExit("component forward requires at least one b")
    if len(before_t) != len(component_bs) or len(b_specs) != len(component_bs):
        raise SystemExit("before_t/b_specs length must match component_bs")

    dt = float(session.delta_ms)
    component_sums = _forward_component(
        session, params, i_sti, component_bs, cells,
        t_start=time_window.forward_t_start(delta_ms=dt),
        t_stop=time_window.forward_t_stop(delta_ms=dt),
        stimulus_pack=stimulus_pack,
        stimulus_bs=stimulus_bs,
    )
    component_layout = component_sums.layout

    def _one_report(
        *,
        cell: str,
        spec: str | None,
        before: int,
        nodes: np.ndarray,
        sums: np.ndarray,
        sum_sqs: np.ndarray,
        n_node: np.ndarray,
        v_post_minus_pre_sums: np.ndarray,
        v_ca_sums: np.ndarray | None = None,
        ca_sums: np.ndarray | None = None,
        ca_pre_sums: np.ndarray | None = None,
    ) -> dict[str, Any]:
        if nodes.size == 0:
            raise SystemExit(f"no nodes for cell {cell!r}")
        v_post = _v_post_from_sums(sums, n_node, component_layout)
        v_post_d = _v_post_d_from_sums(
            sums, v_post_minus_pre_sums, n_node, component_layout,
        )
        cell_extra = dict(extra) if extra else {}
        if extra_from_cell is not None:
            cell_extra.update(extra_from_cell(cell, v_post, v_post_d) or {})
        report = _finalize_component_report(
            cell=cell,
            task=task,
            contrast=contrast,
            spec=spec,
            mode=mode,
            before_t=before,
            nodes=nodes,
            params=params,
            session=session,
            sums=sums,
            sum_sqs=sum_sqs,
            n_node=n_node,
            v_post_minus_pre_sums=v_post_minus_pre_sums,
            v_post=v_post,
            time_window=time_window,
            ti_mode=ti_mode,
            component_layout=component_layout,
            extra=cell_extra or None,
            v_ca_sums=v_ca_sums,
            ca_sums=ca_sums,
            ca_pre_sums=ca_pre_sums,
        )
        if n_node_from_cell is not None:
            report["n_node"] = int(n_node_from_cell(cell))
        return report

    if merge_bs:
        n_t_aligned = component_bs[0].n_t_aligned
        (
            sums, sum_sqs, n_node, v_post_minus_pre_sums, nodes_ref,
            v_ca_sums, ca_sums, ca_pre_sums,
        ) = _merge_component_sums(
            component_sums, component_bs, cells, n_t_aligned,
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
                n_node=n_node[cell],
                v_post_minus_pre_sums=v_post_minus_pre_sums[cell],
                v_ca_sums=None if v_ca_sums is None else v_ca_sums[cell],
                ca_sums=None if ca_sums is None else ca_sums[cell],
                ca_pre_sums=None if ca_pre_sums is None else ca_pre_sums[cell],
            )
            for cell in cells
        }

    reports_by_spec: dict[str, dict[str, dict[str, Any]]] = {}
    for b, spec in enumerate(b_specs):
        if spec is None:
            raise SystemExit("non-merge component forward requires b_specs as str")
        reports_by_spec[spec] = {}
        for cell in cells:
            reports_by_spec[spec][cell] = _one_report(
                cell=cell,
                spec=spec,
                before=int(before_t[b]),
                nodes=component_bs[b].nodes_by_cell[cell],
                sums=component_sums.sums[b][cell],
                sum_sqs=component_sums.sum_sqs[b][cell],
                n_node=component_sums.n_node[b][cell],
                v_post_minus_pre_sums=component_sums.v_post_minus_pre_sums[b][cell],
                v_ca_sums=(
                    None if component_sums.v_ca_sums is None else component_sums.v_ca_sums[b][cell]
                ),
                ca_sums=(
                    None if component_sums.ca_sums is None else component_sums.ca_sums[b][cell]
                ),
                ca_pre_sums=(
                    None if component_sums.ca_pre_sums is None
                    else component_sums.ca_pre_sums[b][cell]
                ),
            )
    return reports_by_spec


def _analyze_bar_forward(
    session,
    *,
    params,
    cells: list[str],
    task: str,
    contrast: str,
    spec_tokens: list[str],
    time_window: TimeWindow,
    nodes_from_b,
    mode: str,
    specs=None,
    grids=None,
    extra: dict[str, Any] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Bar: resolve i_sti/specs → ``_analyze_component_forward`` (no merge)."""
    pack, specs, grids, spec_bs, i_sti, t0_bn = _resolve_bar_spec_i_sti(
        session, task, contrast, spec_tokens, specs=specs, grids=grids,
    )
    before_by_b: list[int] = []
    component_bs: list[_ComponentB] = []
    for spec_b, spec in zip(spec_bs, spec_tokens):
        spec_nodes_by_cell = nodes_from_b(spec_b, spec, pack=pack, t0_bn=t0_bn)
        if task == "mbar":
            before = int(grids.before_t[spec])
            after = int(grids.after_t[spec])
            n_t_aligned = before + after + 1
        else:
            before = int(train.pack_t_onset(pack))
            n_t_aligned = int(i_sti.shape[1])
        before_by_b.append(before)
        component_bs.append(
            _build_component_b(
                spec_nodes_by_cell,
                t0_bn_row=t0_bn[spec_b],
                n_t_aligned=n_t_aligned,
            ),
        )
    return _analyze_component_forward(
        session,
        params=params,
        cells=cells,
        task=task,
        contrast=contrast,
        i_sti=i_sti,
        component_bs=component_bs,
        before_t=before_by_b,
        b_specs=list(spec_tokens),
        time_window=time_window,
        mode=mode,
        ti_mode="t_rel" if task == "mbar" else "abs_minus_before",
        merge_bs=False,
        extra=extra,
        stimulus_pack=pack,
        stimulus_bs=spec_bs,
    )


def analyze_bar_average(
    session,
    *,
    params,
    cells: list[str],
    task: str,
    contrast: str,
    spec_tokens: list[str],
    time_window: TimeWindow,
    specs=None,
    grids=None,
    mids_req: list[float] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """One v forward over bs over all requested specs; mean component per cell.

    Returns ``reports[spec][cell]``. v_post + component share ``_forward_component``.
    """
    pack = session.packs[task][contrast]
    if task == "mbar":
        cols_holder: list = []

        def nodes_from_b(b, spec, *, pack, t0_bn):
            connectome = session.connectome
            if not cols_holder:
                cols_holder.append(cost_sti_hexes(connectome, cost_radius=pack.cost_radius))
            hexes = cols_holder[0]
            nodes_by_cell: dict[str, np.ndarray] = {}
            for cell in cells:
                try:
                    nodes = nodes_from_hexes(connectome, cell, hexes)
                except ValueError as exc:
                    raise SystemExit(str(exc)) from exc
                nodes = nodes[t0_bn[b, nodes] >= 0]
                if nodes.size == 0:
                    raise SystemExit(f"no valid {cell} nodes in cost_radius for bar")
                nodes_by_cell[cell] = nodes
            return nodes_by_cell
    else:
        spec_b_from_token = {
            token: idx for idx, token in enumerate(spec_tokens)
        }
        entry_bs = np.asarray(pack.entry_bs)
        entry_nodes = np.asarray(pack.entry_nodes)
        entry_t0s = np.asarray(pack.cost_t0s) if pack.cost_t0s is not None else None
        entry_mids: np.ndarray | None = None
        if mids_req is not None:
            entry_part_keys = list(pack.entry_part_keys or ())
            if len(entry_part_keys) != len(entry_bs):
                raise SystemExit(
                    f"mid={mids_req} requires sbar pack with entry_part_keys"
                )
            entry_mids = np.full(entry_bs.shape, np.nan, dtype=float)
            for i, key in enumerate(entry_part_keys):
                try:
                    entry_mids[i] = float(key.rsplit("_mid", 1)[1])
                except (IndexError, ValueError):
                    continue
            available_mids = sorted(set(entry_mids[np.isfinite(entry_mids)]))
            for mid in mids_req:
                if not any(
                    np.isclose(float(mid), available, atol=1e-9, rtol=0.0)
                    for available in available_mids
                ):
                    raise SystemExit(
                        f"mid={mid} not available; choose from {available_mids}"
                    )

        def nodes_from_b(b, spec, *, pack, t0_bn):
            mask = entry_bs == b
            if entry_t0s is not None:
                mask &= entry_t0s >= 0
            if mids_req is not None:
                if entry_mids is None:
                    raise SystemExit(f"mid={mids_req} requires sbar pack with entry_part_keys")
                mask &= np.any(
                    np.isclose(
                        entry_mids[:, None], np.asarray(mids_req)[None, :],
                        atol=1e-9, rtol=0.0,
                    ),
                    axis=1,
                )
            if not np.any(mask):
                raise SystemExit(f"no cost entries for spec {spec!r}" +
                    (f" mid={mids_req}" if mids_req else ""))
            node_subset = np.unique(entry_nodes[mask])
            connectome = session.connectome
            cell_idx = np.asarray(connectome.node_cells)[node_subset]
            nodes_by_cell: dict[str, np.ndarray] = {}
            for cell in cells:
                nodes = node_subset[np.asarray(cell_idx == connectome.cells.index(cell))]
                if nodes.size == 0:
                    raise SystemExit(f"no valid {cell} nodes in cost_radius for bar")
                nodes_by_cell[cell] = nodes
            return nodes_by_cell

    return _analyze_bar_forward(
        session,
        params=params,
        cells=cells,
        task=task,
        contrast=contrast,
        spec_tokens=spec_tokens,
        time_window=time_window,
        nodes_from_b=nodes_from_b,
        mode="mid" if mids_req is not None and len(mids_req) == 1 else "average",
        specs=specs,
        grids=grids,
        extra=(
            {"mid": float(mids_req[0])}
            if mids_req is not None and len(mids_req) == 1
            else None
        ),
    )


# ---------------------------------------------------------------------------
# Average spot components (hex-lattice radius)
# ---------------------------------------------------------------------------


def _spot_session_readout(session_one, cells: list[str]):
    """Session-scoped spot cost readout (all radii) for component forward."""
    pack = session_one.primary_pack
    connectome = session_one.connectome
    opts = dict((session_one.train_opts or {}).get(f"{pack.task}_sti_opts") or {})
    spot = resolve_spot(connectome, sti_opts=opts)
    (
        bs, nodes, radii, type_idx, _sti_u, _sti_v, _du, _dv, _center_entry_mask,
    ) = build_spot_center_readout(
        connectome,
        spot_sti_bs(spot),
        pack_spot_cost_radii(pack),
        pack.cost_radius,
    )
    cell_idx: dict[str, int] = {}
    for cell in cells:
        if cell not in connectome.cells:
            raise SystemExit(f"unknown cell {cell!r}")
        cell_idx[cell] = connectome.cells.index(cell)
    return pack, bs, nodes, radii, type_idx, cell_idx


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
    gt_report_key: str,
) -> dict[str, Any]:
    """Affine GT on readout axis: ``a_gt * gt + bias``; ``gt_report_key`` is ``gt_v`` or ``gt_ca``."""
    if gt_report_key not in _GT_PLOT_PANEL:
        raise SystemExit(f"unknown gt_report_key {gt_report_key!r}; expected gt_v|gt_ca")
    extra: dict[str, Any] = {"gt_peak": None, gt_report_key: None, "radius": radius}
    if cell not in gt_on:
        return extra
    gt = np.asarray(gt_on[cell], dtype=float)
    if radius < 0 or radius >= gt.shape[0]:
        return extra
    gt_row = gt[radius]
    gt_aff = float(a_gt) * gt_row + float(bias_gt)
    peak_probe = _v_post_d_peak_t_rel(v_post_d, t_onset)
    if 0 <= peak_probe < gt_aff.shape[0]:
        extra["gt_peak"] = float(gt_aff[peak_probe])
    mask = np.isfinite(v_post) & np.isfinite(v_post_d)
    if not np.any(mask):
        return extra
    extra[gt_report_key] = gt_aff.tolist()
    return extra


def _spot_extra_from_cell(
    session_one, params, pack, *, radius: int, t_onset: int, train_filter,
):
    """Build ``extra_from_cell`` with train GT named ``gt_v`` / ``gt_ca``."""
    contrast = contrast_from_pack(pack)
    train_filter = str(train_filter)
    gt_report_key = f"gt_{filter_figure_token(train_filter)}"
    gt_on = resolve_spot_gts(
        {contrast: session_one}, filter=train_filter,
    ).get(contrast) or {}
    opts = session_one.train_opts or {}
    from_onset = train.val_from_enabled(opts, "bias_gt")
    lo = param_from_entry("bias_gt", "lo", _config.NEURON_SCHEMA['params'])
    hi = param_from_entry("bias_gt", "hi", _config.NEURON_SCHEMA['params'])
    cells = [str(cell) for cell in session_one.connectome.cells]

    def extra_from_cell(
        cell: str, v_post: np.ndarray, v_post_d: np.ndarray,
    ) -> dict[str, Any]:
        if from_onset and 0 <= int(t_onset) < len(v_post):
            cell_idx = cells.index(str(cell))
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
        a_gt, bias_gt = train.gt_affine_from_cell(
            params, cell, session_one.connectome, session=session_one,
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
            gt_report_key=gt_report_key,
        )

    return extra_from_cell


def analyze_spot_average(
    session_one,
    *,
    params,
    cells: list[str],
    task: str,
    contrast: str,
    time_window: TimeWindow,
    radius: int = 0,
    train_filter="none",
) -> dict[str, dict[str, Any]]:
    """One v forward over bs over spot sti rows; mean at hex-lattice ``radius``.

    ``time_window`` is absolute aligned ms for spot (``0`` = trial start). Pre
    only: ``TimeWindow("ms", 0, ms_pre)``. Do not pass negative ``start`` for
    "before onset" — that confuses sti length with the analyze window.
    ``train_filter`` is the run's train ``filter`` (names GT ``gt_v``/``gt_ca``).
    """
    if task != "spot":
        raise SystemExit(f"unsupported task {task!r}")
    radius = int(radius)
    pack, bs, nodes, radii, type_idx, cell_idx = _spot_session_readout(
        session_one, cells,
    )
    radius_entry_mask = np.asarray(radii, dtype=np.int64) == int(radius)
    t_onset = train.pack_t_onset(pack)

    i_sti = pack.i_sti if pack.i_sti.dim() == 3 else pack.i_sti.unsqueeze(0)
    n_sti_b, n_t, n_node = i_sti.shape
    t0_abs = np.zeros(n_node, dtype=np.int64)
    n_t_aligned = time_window.aligned_n_t(n_t, delta_ms=float(session_one.delta_ms))

    component_bs: list[_ComponentB] = []
    i_sti_rows: list[int] = []
    for sti_b in range(n_sti_b):
        entry_mask = radius_entry_mask & (bs == sti_b)
        if not np.any(entry_mask):
            continue
        nodes_by_cell: dict[str, np.ndarray] = {}
        for cell in cells:
            cell_entry_mask = entry_mask & (type_idx == cell_idx[cell])
            if np.any(cell_entry_mask):
                nodes_by_cell[cell] = np.unique(nodes[cell_entry_mask])
        if not nodes_by_cell:
            continue
        component_bs.append(
            _build_component_b(nodes_by_cell, t0_bn_row=t0_abs, n_t_aligned=n_t_aligned),
        )
        i_sti_rows.append(sti_b)

    if not component_bs:
        raise SystemExit(
            f"no spot nodes at radius={radius} for requested cells in spot readout"
        )

    n_b = len(component_bs)
    return _analyze_component_forward(
        session_one,
        params=params,
        cells=cells,
        task=task,
        contrast=contrast,
        i_sti=i_sti[i_sti_rows],
        component_bs=component_bs,
        before_t=[t_onset] * n_b,
        b_specs=[None] * n_b,
        time_window=time_window,
        mode="average",
        ti_mode="abs_minus_before",
        merge_bs=True,
        extra={"radius": radius},
        extra_from_cell=_spot_extra_from_cell(
            session_one, params, pack, radius=radius, t_onset=t_onset,
            train_filter=train_filter,
        ),
        n_node_from_cell=lambda cell: int(
            np.sum(radius_entry_mask & (type_idx == cell_idx[cell]))
        ),
    )


# ---------------------------------------------------------------------------
# Hex mode (single hex; same run as average)
# ---------------------------------------------------------------------------


def _hex_nodes(session, cell: str, *, at_x: float, at_y: float, cost_radius):
    connectome = session.connectome
    hexes = sti_hexes_at_xy(
        cost_sti_hexes(connectome, cost_radius=cost_radius),
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
    nodes = connectome.nodes_at_uv(int(hex.u), int(hex.v), cell).tolist()
    if not nodes:
        raise SystemExit(f"no {cell} node at hex ({at_x},{at_y})")
    return hex, nodes


def _resolve_hex_node(
    session,
    cell: str,
    *,
    at_x: float,
    at_y: float,
    cost_radius,
    node: int | None,
):
    """Resolve ``(hex, node_id)`` for hex mode; ``--node`` if multiple at hex."""
    hex, nodes = _hex_nodes(
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
    contrast: str,
    at_x: float,
    at_y: float,
    node: int | None,
    time_window: TimeWindow,
    train_filter="none",
) -> dict[str, dict[str, Any]]:
    """One v forward over bs at one hex; sti-on (radius 0) rows for that node only."""
    if task != "spot":
        raise SystemExit(f"unsupported task {task!r}")
    pack, bs, nodes, radii, type_idx, cell_idx = _spot_session_readout(
        session_one, [cell],
    )
    radius_entry_mask = np.asarray(radii, dtype=np.int64) == 0
    hex, node = _resolve_hex_node(
        session_one, cell, at_x=at_x, at_y=at_y,
        cost_radius=pack.cost_radius, node=node,
    )
    t_onset = train.pack_t_onset(pack)

    i_sti = pack.i_sti if pack.i_sti.dim() == 3 else pack.i_sti.unsqueeze(0)
    n_sti_b, n_t, n_node = i_sti.shape
    t0_abs = np.zeros(n_node, dtype=np.int64)
    n_t_aligned = time_window.aligned_n_t(n_t, delta_ms=float(session_one.delta_ms))
    nodes_by_cell = {cell: np.asarray([node], dtype=np.int64)}
    type_cell = cell_idx[cell]

    component_bs: list[_ComponentB] = []
    i_sti_rows: list[int] = []
    for sti_b in range(n_sti_b):
        entry_mask = (
            radius_entry_mask
            & (bs == sti_b)
            & (type_idx == type_cell)
            & (nodes == node)
        )
        if not np.any(entry_mask):
            continue
        component_bs.append(
            _build_component_b(nodes_by_cell, t0_bn_row=t0_abs, n_t_aligned=n_t_aligned),
        )
        i_sti_rows.append(sti_b)

    if not component_bs:
        raise SystemExit(
            f"no sti-on spot rows for {cell} node {node} at hex ({at_x},{at_y})"
        )

    n_b = len(component_bs)
    return _analyze_component_forward(
        session_one,
        params=params,
        cells=[cell],
        task=task,
        contrast=contrast,
        i_sti=i_sti[i_sti_rows],
        component_bs=component_bs,
        before_t=[t_onset] * n_b,
        b_specs=[None] * n_b,
        time_window=time_window,
        mode="hex",
        ti_mode="abs_minus_before",
        merge_bs=True,
        extra={
            "node": node,
            "hex": {"x": at_x, "y": at_y},
            "uv": {"u": int(hex.u), "v": int(hex.v)},
        },
        extra_from_cell=_spot_extra_from_cell(
            session_one, params, pack, radius=0, t_onset=t_onset,
            train_filter=train_filter,
        ),
        n_node_from_cell=lambda _c: 1,
    )


def analyze_spread_average(
    session_one,
    *,
    params,
    cells: list[str],
    task: str,
    contrast: str,
    time_window: TimeWindow,
) -> dict[str, dict[str, Any]]:
    """One v forward; mean components over spread cost ``entry_nodes``."""
    if task != "spread":
        raise SystemExit(f"unsupported task {task!r}")
    pack = session_one.primary_pack
    connectome = session_one.connectome
    entry_bs = np.asarray(pack.entry_bs, dtype=np.int64)
    entry_nodes = np.asarray(pack.entry_nodes, dtype=np.int64)
    if entry_nodes.size == 0:
        raise SystemExit("spread pack has no cost entry_nodes")
    entry_cell_idxs = np.asarray(connectome.node_cells)[entry_nodes]
    cell_idx: dict[str, int] = {}
    for cell in cells:
        if cell not in connectome.cells:
            raise SystemExit(f"unknown cell {cell!r}")
        cell_idx[cell] = connectome.cells.index(cell)

    t_onset = train.pack_t_onset(pack)
    i_sti = pack.i_sti if pack.i_sti.dim() == 3 else pack.i_sti.unsqueeze(0)
    n_sti_b, n_t, n_node = i_sti.shape
    t0_abs = np.zeros(n_node, dtype=np.int64)
    n_t_aligned = time_window.aligned_n_t(n_t, delta_ms=float(session_one.delta_ms))

    component_bs: list[_ComponentB] = []
    i_sti_rows: list[int] = []
    for sti_b in range(n_sti_b):
        entry_mask = entry_bs == sti_b
        if not np.any(entry_mask):
            continue
        nodes_by_cell: dict[str, np.ndarray] = {}
        for cell in cells:
            cell_entry_mask = entry_mask & (entry_cell_idxs == cell_idx[cell])
            if np.any(cell_entry_mask):
                nodes_by_cell[cell] = np.unique(entry_nodes[cell_entry_mask])
        if not nodes_by_cell:
            continue
        component_bs.append(
            _build_component_b(nodes_by_cell, t0_bn_row=t0_abs, n_t_aligned=n_t_aligned),
        )
        i_sti_rows.append(sti_b)

    if not component_bs:
        raise SystemExit(
            f"no spread cost nodes for requested cells {cells!r}"
        )

    n_b = len(component_bs)
    return _analyze_component_forward(
        session_one,
        params=params,
        cells=cells,
        task=task,
        contrast=contrast,
        i_sti=i_sti[i_sti_rows],
        component_bs=component_bs,
        before_t=[t_onset] * n_b,
        b_specs=[None] * n_b,
        time_window=time_window,
        mode="average",
        ti_mode="abs_minus_before",
        merge_bs=True,
        n_node_from_cell=lambda cell: int(
            np.sum(entry_cell_idxs == cell_idx[cell])
        ),
    )


def analyze_spread_hex(
    session_one,
    *,
    params,
    cell: str,
    task: str,
    contrast: str,
    at_x: float,
    at_y: float,
    node: int | None,
    time_window: TimeWindow,
) -> dict[str, dict[str, Any]]:
    """One v forward at one hex under uniform spread ``i_sti``."""
    if task != "spread":
        raise SystemExit(f"unsupported task {task!r}")
    pack = session_one.primary_pack
    hex, node = _resolve_hex_node(
        session_one, cell, at_x=at_x, at_y=at_y,
        cost_radius=None, node=node,
    )
    t_onset = train.pack_t_onset(pack)
    i_sti = pack.i_sti if pack.i_sti.dim() == 3 else pack.i_sti.unsqueeze(0)
    n_sti_b, n_t, n_node = i_sti.shape
    t0_abs = np.zeros(n_node, dtype=np.int64)
    n_t_aligned = time_window.aligned_n_t(n_t, delta_ms=float(session_one.delta_ms))
    nodes_by_cell = {cell: np.asarray([node], dtype=np.int64)}

    component_bs: list[_ComponentB] = [
        _build_component_b(nodes_by_cell, t0_bn_row=t0_abs, n_t_aligned=n_t_aligned)
        for _ in range(n_sti_b)
    ]
    return _analyze_component_forward(
        session_one,
        params=params,
        cells=[cell],
        task=task,
        contrast=contrast,
        i_sti=i_sti,
        component_bs=component_bs,
        before_t=[t_onset] * n_sti_b,
        b_specs=[None] * n_sti_b,
        time_window=time_window,
        mode="hex",
        ti_mode="abs_minus_before",
        merge_bs=True,
        extra={
            "node": node,
            "hex": {"x": at_x, "y": at_y},
            "uv": {"u": int(hex.u), "v": int(hex.v)},
        },
        n_node_from_cell=lambda _c: 1,
    )


def analyze_bar_hex(
    session,
    *,
    params,
    cell: str,
    task: str,
    contrast: str,
    spec_tokens: list[str],
    at_x: float,
    at_y: float,
    node: int | None,
    time_window: TimeWindow,
    specs=None,
    grids=None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """One v forward over bs over specs at one hex; returns ``reports[spec][cell]``."""
    pack = session.packs[task][contrast]
    hex, node = _resolve_hex_node(
        session, cell, at_x=at_x, at_y=at_y,
        cost_radius=pack.cost_radius, node=node,
    )
    nodes_by_cell = {cell: np.asarray([node], dtype=np.int64)}

    def nodes_from_b(b, spec, *, pack, t0_bn):
        if int(t0_bn[b, node]) < 0:
            raise SystemExit(f"no t0 for node {node} on spec {spec!r}")
        return nodes_by_cell

    return _analyze_bar_forward(
        session,
        params=params,
        cells=[cell],
        task=task,
        contrast=contrast,
        spec_tokens=spec_tokens,
        time_window=time_window,
        nodes_from_b=nodes_from_b,
        mode="hex",
        specs=specs,
        grids=grids,
        extra={
            "node": node,
            "hex": {"x": at_x, "y": at_y},
            "uv": {"u": int(hex.u), "v": int(hex.v)},
        },
    )


def analyze_bar_slice(
    session,
    *,
    params,
    cells: list[str],
    task: str,
    contrast: str,
    spec_tokens: list[str],
    at_x: float | None,
    at_y: float | None,
    time_window: TimeWindow,
    specs=None,
    grids=None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Average static-bar components over one x- or y-coordinate slice."""
    if task != "sbar":
        raise SystemExit("one-coordinate slice mode is supported only for sbar")
    if (at_x is None) == (at_y is None):
        raise SystemExit("sbar slice mode needs exactly one of x or y")

    pack = session.packs[task][contrast]
    hexes = sti_hexes_at_xy(
        cost_sti_hexes(session.connectome, cost_radius=pack.cost_radius),
        at_x=at_x,
        at_y=at_y,
    )
    if not hexes:
        axis, value = ("x", at_x) if at_x is not None else ("y", at_y)
        raise SystemExit(
            f"no hexes at {axis}={value!r} within cost_radius={pack.cost_radius}"
        )

    nodes_by_cell: dict[str, np.ndarray] = {}
    for cell in cells:
        try:
            nodes = nodes_from_hexes(session.connectome, cell, hexes)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if nodes.size == 0:
            raise SystemExit(f"no {cell} nodes on requested sbar coordinate slice")
        nodes_by_cell[cell] = nodes

    def nodes_from_b(_b, _spec, *, pack, t0_bn):
        return nodes_by_cell

    return _analyze_bar_forward(
        session,
        params=params,
        cells=cells,
        task=task,
        contrast=contrast,
        spec_tokens=spec_tokens,
        time_window=time_window,
        nodes_from_b=nodes_from_b,
        mode="slice",
        specs=specs,
        grids=grids,
        extra={"slice": {"x": at_x, "y": at_y}},
    )



# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _figure_filename(report: dict[str, Any], *, file_suffix: str = "", html: bool = False) -> str:
    filter_token = str(report.get("filter") or "v")
    parts = [report["cell"], report["task"], report["contrast"], filter_token, report.get("mode", "average")]
    if report.get("spec"):
        parts.append(str(report["spec"]))
    if report.get("mode") == "hex":
        hex = report["hex"]
        parts.append(f"x{hex['x']}_y{hex['y']}")
    elif report.get("mode") == "slice":
        coordinate = report["slice"]
        axis = "x" if coordinate["x"] is not None else "y"
        parts.append(f"{axis}{coordinate[axis]}")
    elif report.get("mode") == "mid":
        parts.append(f"mid{report['mid']}")
    radius = report.get("radius")
    if radius is not None and int(radius) != 0:
        parts.append(f"radius{int(radius)}")
    return "_".join(parts) + f"{file_suffix}{'.html' if html else '.png'}"


def _compare_figure_filename(
    reports: list[dict[str, Any]], *, file_suffix: str = "", html: bool = False,
) -> str:
    first_report = reports[0]
    filter_token = str(first_report.get("filter") or "v")
    if first_report.get("mode") == "slice":
        coordinate = first_report["slice"]
        axis = "x" if coordinate["x"] is not None else "y"
        values = "_".join(
            str(one_report["slice"][axis]) for one_report in reports
        )
        comparison = f"{first_report['spec']}_{axis}{values}"
    elif first_report.get("mode") == "mid":
        values = "_".join(str(one_report["mid"]) for one_report in reports)
        comparison = f"{first_report['spec']}_mid{values}"
    else:
        comparison = "_".join(str(one_report["spec"]) for one_report in reports)
    return (
        f"{first_report['cell']}_{first_report['task']}_{first_report['contrast']}_{filter_token}_compare_{comparison}"
        f"{file_suffix}{'.html' if html else '.png'}"
    )


def _report_comparison_label(report: dict[str, Any]) -> str:
    if report.get("mode") == "slice":
        coordinate = report["slice"]
        axis = "x" if coordinate["x"] is not None else "y"
        return f"{axis}={coordinate[axis]}"
    if report.get("mode") == "mid":
        return f"mid={report['mid']}"
    return str(report.get("spec"))


def _component_figure(title: str, layout: _ComponentLayout):
    """Shared grid figure: rows = plot panels, cols = traces within a row."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from figure.panel import save_figure

    n_row, n_col = len(layout.plot_panels), layout.n_col
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


def _hide_unused_axes(axes, layout: _ComponentLayout) -> None:
    n_row, n_col = axes.shape
    for row, (_panel_ylabel, panel_series) in enumerate(layout.plot_panels):
        for col in range(len(panel_series), n_col):
            axes[row, col].set_visible(False)


def _style_component_ax(
    ax, ylabel: str, *, legend_fontsize: int, legend_ncol: int, show_legend: bool = True,
) -> None:
    ax.set_ylabel(ylabel, fontsize=8)
    if show_legend:
        ax.legend(loc="best", fontsize=legend_fontsize, ncol=legend_ncol)
    ax.tick_params(labelsize=7)
    ax.grid(True, axis="y", alpha=0.3)


def _shared_row_ylim(
    traces: list[np.ndarray],
    *,
    floor_zero: bool = False,
    margin_frac: float = 0.06,
) -> tuple[float, float]:
    """Tight row ylim from data min/max + small relative pad (not symmetric)."""
    trace_vals = []
    for trace in traces:
        val = np.asarray(trace, dtype=float).ravel()
        val = val[np.isfinite(val)]
        if val.size:
            trace_vals.append(val)
    if not trace_vals:
        return -1.0, 1.0
    ylo = float(np.min(concatenated := np.concatenate(trace_vals)))
    yhi = float(np.max(concatenated))
    if floor_zero and ylo >= 0.0:
        ylo = 0.0
    span = yhi - ylo
    pad = max(span * margin_frac, abs(yhi) * 0.02, 1e-3) if span > 0.0 else max(abs(yhi) * 0.05, 1e-3)
    return ylo - pad, yhi + pad


def _set_shared_row_ylim(
    axes,
    traces_by_row: dict[int, list[np.ndarray]],
    layout: _ComponentLayout,
) -> None:
    """One tight data-driven ylim per row in ``layout.row_shared_ylim``."""
    for row, traces in traces_by_row.items():
        if not traces:
            continue
        _, panel_series = layout.plot_panels[row]
        ylo, yhi = _shared_row_ylim(traces)
        for col in range(len(panel_series)):
            axes[row, col].set_ylim(ylo, yhi)


def _save_component_figure(
    fig, axes, *, xlabel: str, out_path, save_figure, layout: _ComponentLayout,
) -> None:
    _hide_unused_axes(axes, layout)
    last_row = axes.shape[0] - 1
    for col in range(axes.shape[1]):
        ax = axes[last_row, col]
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
    """Shared grid PNG for one or more reports (compare = linestyle per report)."""
    if not reports:
        raise SystemExit("no reports to plot")
    for rep in reports:
        if not rep.get("steps"):
            raise SystemExit(f"no steps to plot for {rep.get('cell')} spec={rep.get('spec')!r}")

    models = {str(one_report.get("model", "borst")) for one_report in reports}
    if len(models) != 1:
        raise SystemExit(f"compare requires one model; got {sorted(models)}")
    kinds = {str(one_report.get("time_window_kind", "t_rel")) for one_report in reports}
    if len(kinds) != 1:
        raise SystemExit(f"compare requires one time_window_kind; got {sorted(kinds)}")
    eulers = {
        str((one_report.get("globals") or {}).get("euler", "implicit"))
        for one_report in reports
    }
    if len(eulers) != 1:
        raise SystemExit(f"compare requires one euler; got {sorted(eulers)}")
    filters = {str(one_report.get("filter", "v")) for one_report in reports}
    if len(filters) != 1:
        raise SystemExit(f"compare requires one filter; got {sorted(filters)}")
    layout = _component_layout(models.pop(), eulers.pop(), filter=filters.pop())

    fig, axes, save_figure = _component_figure(title, layout)
    import matplotlib.pyplot as plt

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    linestyles = ("-", "--", "-.", ":")
    multi_report = len(reports) > 1
    e_leak = float(reports[0].get("params", {}).get("e_leak", 0.0))
    globs = reports[0].get("globals") or {}
    params0 = reports[0].get("params") or {}
    delta_ms = float(globs.get("delta_ms", train.MODEL['delta_ms']))
    traces_by_row: dict[int, list[np.ndarray]] = {
        row: [] for row in layout.row_shared_ylim
    }
    tc = _plot_trace_colors(colors, layout)

    for row, (panel_ylabel, panel_series) in enumerate(layout.plot_panels):
        for col, (series, label) in enumerate(panel_series):
            ax = axes[row, col]
            color = tc[label]
            show_legend = multi_report and row == 0 and col == 0
            drew_gt = False
            for report_idx, rep in enumerate(reports):
                ls = linestyles[report_idx % len(linestyles)] if multi_report else "-"
                ts = np.asarray([step["t"] for step in rep["steps"]], dtype=int)
                xs = ts.astype(float) * delta_ms
                y = np.asarray([step[series] for step in rep["steps"]], dtype=float)
                std = np.asarray(
                    [
                        float(step.get("std", {}).get(series, 0.0))
                        for step in rep["steps"]
                    ],
                    dtype=float,
                )
                if row in layout.row_shared_ylim:
                    traces_by_row[row].append(y)
                    if np.any(std):
                        traces_by_row[row].append(y + std)
                        traces_by_row[row].append(y - std)
                plot_std_band(
                    ax, xs, y, std,
                    color=color,
                    alpha=0.3,
                    label="_nolegend_" if multi_report else r"$\pm$STD",
                )
                gt_report_key = next(
                    (
                        mapped_gt_report_key
                        for mapped_gt_report_key, mapped_series in _GT_PLOT_PANEL.items()
                        if mapped_series == series and rep.get(mapped_gt_report_key) is not None
                    ),
                    None,
                )
                model_label = (
                    _report_comparison_label(rep) if show_legend
                    else (series if gt_report_key is not None else "_nolegend_")
                )
                ax.plot(
                    xs, y,
                    label=model_label,
                    color=color,
                    linestyle=ls,
                    linewidth=1.4,
                )
                if gt_report_key is not None:
                    gt = np.asarray(rep[gt_report_key], dtype=float)
                    t_onset = int(rep.get("before_t") or 0)
                    t = ts - t_onset
                    valid = (t >= 0) & (t < gt.shape[0])
                    if np.any(valid):
                        y_gt = gt[t[valid]]
                        xs_gt = xs[valid]
                        if row in layout.row_shared_ylim:
                            traces_by_row[row].append(y_gt)
                        ax.plot(
                            xs_gt, y_gt,
                            color="k",
                            linestyle="--",
                            linewidth=1.2,
                            label="gt" if (show_legend or not multi_report) else "_nolegend_",
                        )
                        drew_gt = True
            e_note = _g_e_note(label, e_leak=e_leak, globs=globs, params=params0)
            if e_note is not None:
                ax.set_title(e_note, fontsize=8)
            _style_component_ax(
                ax, _trace_ylabel(panel_ylabel, label),
                legend_fontsize=6 if multi_report else 7,
                legend_ncol=1,
                show_legend=show_legend or drew_gt,
            )
    _set_shared_row_ylim(axes, traces_by_row, layout)
    _finish_component_figure(fig, title, colors, layout)
    _save_component_figure(
        fig, axes, xlabel="t (ms)",
        out_path=out_path, save_figure=save_figure, layout=layout,
    )


def plot_report(report: dict[str, Any], out_path: str) -> None:
    """Write one multi-panel PNG: component traces vs ``t`` in ms."""
    title = (
        f"{report['cell']}  {report['task']}  {report['contrast']}"
        + (f"  {report['spec']}" if report.get("spec") else "")
        + f"  mode={report.get('mode')}  n={report.get('n_node')}"
    )
    if report.get("mode") == "hex":
        title += f"  hex=({report['hex']['x']},{report['hex']['y']})"
    _plot_component_reports([report], out_path, title=title)


def plot_reports_compare(reports: list[dict[str, Any]], out_path: str) -> None:
    """One grid PNG: reports share color per subplot, differ by linestyle."""
    if not reports:
        raise SystemExit("no reports to compare")
    first_report = reports[0]
    labels_csv = ",".join(_report_comparison_label(report) for report in reports)
    title = (
        f"{first_report['cell']}  {first_report['task']}  {first_report['contrast']}  "
        f"{first_report.get('spec')}  reports=[{labels_csv}]"
        f"  mode={first_report.get('mode')}  n={first_report.get('n_node')}"
    )
    _plot_component_reports(reports, out_path, title=title)


def _emit_report(
    report: dict[str, Any],
    *,
    run_dir: str,
    do_print: bool,
    do_figure: bool,
    file_suffix: str = "",
    html: bool = False,
) -> None:
    if do_print:
        print("")
        # Full per-t table only when not plotting (--plot false).
        _print_report(report, print_steps=not do_figure)
    if do_figure:
        path = os.path.join(
            run_dir,
            "cell_dynamics",
            _figure_filename(report, file_suffix=file_suffix, html=html),
        )
        plot_report(report, path)


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def _print_report(report: dict[str, Any], *, print_steps: bool = True) -> None:
    mode = report.get("mode", "?")
    model = report.get("model", "borst")
    kind = report.get("time_window_kind", "t_rel")
    x_tok = "t_rel" if kind == "t_rel" else "t"
    use_ca = report.get("filter") == "ca"
    trace_hdr = (
        "ca  ca_pre  ca_post_minus_pre"
        if use_ca else "v_post  v_pre  v_post_minus_pre"
    )
    hdr = (
        f"cell={report['cell']} model={model} mode={mode} "
        f"n_node={report.get('n_node', '?')}"
    )
    if mode == "hex":
        hdr += (
            f" node=#{report['node']} hex=({report['hex']['x']},{report['hex']['y']}) "
            f"uv=({report['uv']['u']},{report['uv']['v']})"
        )
    print(hdr)
    print(f"task={report['task']} contrast={report['contrast']} spec={report.get('spec')}")
    tw = report.get("time_window")
    print(
        f"v_post_d_peak={report['v_post_d_peak']:+.4f} mV "
        f"v_post_d_sign={report['v_post_d_sign']}  "
        f"v_post_d_peak_t={report['v_post_d_peak_t']}  "
        f"before_t={report.get('before_t')}  "
        f"peak_drive={report.get('peak_drive')}  "
        f"{kind}={tw[0]}:{tw[1]}"
    )
    print("trained:", "  ".join(f"{k}={v:.6g}" for k, v in report["params"].items()))

    def _trace_cols(step: dict[str, Any]) -> str:
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
                f"\n{x_tok}  n  {trace_hdr}  i_sti "
                "v_syn  v_syn_exc -v_syn_inh  dv_leak  dv_hp"
            )
            for step in report["steps"]:
                print(
                    f"{step[x_tok]:4d} {step.get('n_node', 1):3d} {_trace_cols(step)} "
                    f"{step['i_sti']:+6.3f} {step['v_syn']:+6.3f} {step['v_syn_exc']:+7.3f} "
                    f"{step['v_syn_inh']:+7.3f} {step['dv_leak']:+8.4f} {step['dv_hp']:+7.4f}"
                )
        peak_step = report.get("peak_step")
        if peak_step is not None:
            print(f"\nHP/LP terms at peak {x_tok}={peak_step[x_tok]}:")
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
            f"\n{x_tok}  n  {trace_hdr}  i_sti "
            "g_inh  g_h_rev  g_exc  num_inh  num_exc"
        )
        for step in report["steps"]:
            print(
                f"{step[x_tok]:4d} {step.get('n_node', 1):3d} {_trace_cols(step)} "
                f"{step['i_sti']:5.1f} {step['g_inh_nS']:.4f} {step['g_h_rev_nS']:.4f} "
                f"{step['g_exc_nS']:.4f} {step['num_inh']:+8.2f} {step['num_exc']:+8.2f}"
            )

    peak_step = report.get("peak_step")
    if peak_step is not None:
        num = float(peak_step["num"])
        dt_over_cap = float(peak_step.get("dt_over_cap", 0.0))
        print(f"\nNumerator at peak {x_tok}={peak_step[x_tok]} (num={num:.2f}):")
        for name, val in [
            ("num_v", peak_step["num_v"]),
            ("dt_over_cap*i_sti", dt_over_cap * peak_step["i_sti"]),
            ("dt_over_cap*i_exc", dt_over_cap * peak_step["num_exc"]),
            ("dt_over_cap*i_inh", dt_over_cap * peak_step["num_inh"]),
            ("dt_over_cap*i_leak", dt_over_cap * peak_step["num_leak"]),
            ("dt_over_cap*i_h", dt_over_cap * peak_step["num_i_h"]),
            ("dt_over_cap*i_h_rev", dt_over_cap * peak_step["num_i_h_rev"]),
        ]:
            pct = 100.0 * val / num if num else 0.0
            print(f"  {name:20s} {val:+9.2f} ({pct:.0f}%)")
    if "cost" in report and "best_cost" in report:
        print(f"best_cost={report['best_cost']:.4f}  cost={report['cost']:.4f}")


def _print_sign_compare(
    spot_reports: dict[tuple[str, str], dict[str, Any]],
    bar_reports: dict[tuple[str, str], dict[str, Any]],
) -> None:
    keys = sorted(set(spot_reports) & set(bar_reports))
    if not keys:
        return
    print("\n======== SPOT vs BAR sign (cost-radius averages) ========")
    print(
        f"{'cell':6s} {'contrast':8s} {'spot_post_d':>11s} {'spot_sign':>8s} {'spot_drv':>8s} "
        f"{'bar_post_d':>10s} {'bar_sign':>8s} {'bar_drv':>8s}  note"
    )
    for cell, contrast in keys:
        spot_report = spot_reports[(cell, contrast)]
        bar_report = bar_reports[(cell, contrast)]
        flip = (
            spot_report["v_post_d_sign"] != bar_report["v_post_d_sign"]
            and "0" not in (
                spot_report["v_post_d_sign"], bar_report["v_post_d_sign"],
            )
        )
        note = "FLIP" if flip else "same"
        if flip and spot_report.get("peak_drive") and bar_report.get("peak_drive"):
            if spot_report["peak_drive"] != bar_report["peak_drive"]:
                note += f" (drive {spot_report['peak_drive']}→{bar_report['peak_drive']})"
            else:
                note += f" (same drive={spot_report['peak_drive']}; see num terms)"
        print(
            f"{cell:6s} {contrast:8s} {spot_report['v_post_d_peak']:+11.4f} "
            f"{spot_report['v_post_d_sign']:>8s} "
            f"{str(spot_report.get('peak_drive')):>8s} "
            f"{bar_report['v_post_d_peak']:+10.4f} {bar_report['v_post_d_sign']:>8s} "
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




@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(hydra_config) -> None:
    from config import active_config, resolve_config

    resolve_config(hydra_config)
    cli = resolve_shared_cli()
    radius = int(_config.ANALYZE_CELL_DYNAMICS["radius"])
    node = _config.ANALYZE_CELL_DYNAMICS.get("node")
    if node is not None:
        node = int(node)
    do_figure = bool(_config.ANALYZE_CELL_DYNAMICS["figure"])
    do_json = bool(_config.ANALYZE_CELL_DYNAMICS["json"])
    html = bool(_config.FIGURE_PLOT.get("html", False))
    session_kwargs = session_kwargs_from_cli(hydra_config)
    euler = session_kwargs["euler"]
    filter = session_kwargs["filter"]
    t_rel_start = _config.ANALYZE_CELL_DYNAMICS.get("t_rel_start")
    t_rel_stop = _config.ANALYZE_CELL_DYNAMICS.get("t_rel_stop")
    ms_shown = _config.FIGURE_PLOT.get("ms_shown")
    if t_rel_start is not None or t_rel_stop is not None:
        if t_rel_start is None or t_rel_stop is None:
            raise SystemExit("t_rel_start and t_rel_stop must be set together")
        if ms_shown is not None:
            raise SystemExit("t_rel_start/t_rel_stop and ms_shown are mutually exclusive")
        t_lo, t_hi = int(t_rel_start), int(t_rel_stop)
        if t_lo > t_hi:
            raise SystemExit(f"t_rel_start={t_lo} > t_rel_stop={t_hi}")
        time_window = TimeWindow(kind="t_rel", start=t_lo, stop=t_hi)
        ms_range = None
        use_ms = False
    elif ms_shown is not None:
        try:
            ms_range = plot.parse_ms_shown(ms_shown, flag="ms_shown")
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        use_ms = True
    else:
        ms_range = None
        use_ms = True

    if radius != 0 and "spot" not in cli.tasks:
        raise SystemExit("radius requires a spot task")

    hex_mode = False
    slice_mode = False
    if cli.xs is not None and cli.ys is not None:
        if len(cli.xs) != 1 or len(cli.ys) != 1:
            raise SystemExit(
                "hex mode needs exactly one x and y; "
                "omit both for cost-radius averages"
            )
        hex_mode = True
        if len(cli.cells) != 1:
            raise SystemExit("hex mode supports one cell")
    elif cli.xs is not None or cli.ys is not None:
        if any(task != "sbar" for task in cli.tasks):
            raise SystemExit(
                "one-coordinate slice mode supports sbar only; "
                "pass both x and y for spot/mbar hex mode"
            )
        if node is not None:
            raise SystemExit("node requires both x and y for exact-hex mode")
        slice_mode = True

    if hex_mode and radius != 0:
        raise SystemExit(
            "radius is average-only; omit x/y, or omit radius for hex mode"
        )
    if cli.mids_req is not None:
        if any(task != "sbar" for task in cli.tasks):
            raise SystemExit("mid selection supports sbar only")
        if hex_mode or slice_mode:
            raise SystemExit("mid and x/y selections are mutually exclusive")

    param_vals = dict(active_config().get("param_vals") or {})

    for run_idx, run_arg in enumerate(_config.ANALYZE_RUNS):
        run_dir = plot.resolve_run_dir(run_arg)
        _log(f"load_best {run_dir} ...")
        session, z, best_cost = plot.load_best(run_dir)
        train_opts = plot.load_train_opts(run_dir) or {}
        train_filter = str(train_opts.get("filter", "none"))
        session, z, ms_changed = plot.override_session(
            run_dir=run_dir,
            session=session,
            z=z,
            **session_kwargs,
        )
        file_suffix = (
            plot.ms_filename_suffix(**ms_changed)
            + plot.euler_filename_suffix(euler)
            + plot.param_filename_suffix(param_vals=param_vals)
        )
        if use_ms:
            lo, hi = (
                (0.0, (int(session.n_t) - 1) * float(session.delta_ms))
                if ms_range is None else ms_range
            )
            time_window = TimeWindow(kind="ms", start=lo, stop=hi)
        schema = train.schema_copy(session.schema)
        z = torch.tensor(
            np.asarray(z, dtype=np.float64),
            dtype=session.sim_dtype,
            device=session.device,
        )
        z, schema = train.override_params(
            z, schema, session, param_vals=param_vals,
        )
        session = session.with_schema(schema)
        params = train.params_from_z(z, session)
        calc_cost = bool(
            ms_changed or euler is not None or filter is not None or param_vals
        )
        cost = (
            float(train.calc_cost(z, session).item()) if calc_cost
            else float(best_cost)
        )

        spot_session_cache: dict[tuple[str, str], object] = {}
        bar_session_cache: dict[tuple[str, str], object] = {}
        bar_meta_cache: dict[tuple[str, str], tuple] = {}
        spot_by_cell: dict[tuple[str, str], dict[str, Any]] = {}
        bar_by_cell: dict[tuple[str, str], dict[str, Any]] = {}
        reports: list[dict[str, Any]] = []

        if not do_json:
            _log(f"== RUN {run_idx}: {run_dir} ==")
            _log(
                f"best_cost={best_cost:.6g}  cost={cost:.6g}  "
                f"mode={'hex' if hex_mode else 'average'}  "
                f"radius={radius}  "
                f"{time_window.kind}={time_window.start}:{time_window.stop}"
            )

        for task in cli.tasks:
            for contrast in cli.contrasts:
                if task == "spot":
                    cache_key = (task, contrast)
                    if cache_key not in spot_session_cache:
                        spot_session_cache[cache_key] = plot.session_from_task(
                            session, task, contrast,
                        )
                    session_one = spot_session_cache[cache_key]
                    if hex_mode:
                        at_x = cli.xs[0]
                        at_y = cli.ys[0]
                        _log(
                            f"component forward {task} {contrast} "
                            f"(spot hex=({at_x},{at_y})) ..."
                        )
                        spot_reports = analyze_spot_hex(
                            session_one,
                            params=params,
                            cell=cli.cells[0],
                            task=task,
                            contrast=contrast,
                            at_x=float(at_x),
                            at_y=float(at_y),
                            node=node,
                            time_window=time_window,
                            train_filter=train_filter,
                        )
                    else:
                        _log(
                            f"component forward {task} {contrast} "
                            f"(spot radius={radius}) ..."
                        )
                        spot_reports = analyze_spot_average(
                            session_one,
                            params=params,
                            cells=cli.cells,
                            task=task,
                            contrast=contrast,
                            time_window=time_window,
                            radius=radius,
                            train_filter=train_filter,
                        )
                    for cell, rep in spot_reports.items():
                        rep["best_cost"] = best_cost
                        rep["cost"] = cost
                        spot_by_cell[(cell, contrast)] = rep
                        reports.append(rep)
                        _emit_report(
                            rep,
                            run_dir=run_dir,
                            do_print=not do_json,
                            do_figure=do_figure,
                            file_suffix=file_suffix,
                            html=html,
                        )
                elif task == "spread":
                    cache_key = (task, contrast)
                    if cache_key not in spot_session_cache:
                        spot_session_cache[cache_key] = plot.session_from_task(
                            session, task, contrast,
                        )
                    session_one = spot_session_cache[cache_key]
                    if hex_mode:
                        at_x = cli.xs[0]
                        at_y = cli.ys[0]
                        _log(
                            f"component forward {task} {contrast} "
                            f"(spread hex=({at_x},{at_y})) ..."
                        )
                        spread_reports = analyze_spread_hex(
                            session_one,
                            params=params,
                            cell=cli.cells[0],
                            task=task,
                            contrast=contrast,
                            at_x=float(at_x),
                            at_y=float(at_y),
                            node=node,
                            time_window=time_window,
                        )
                    else:
                        _log(
                            f"component forward {task} {contrast} "
                            f"(spread average) ..."
                        )
                        spread_reports = analyze_spread_average(
                            session_one,
                            params=params,
                            cells=cli.cells,
                            task=task,
                            contrast=contrast,
                            time_window=time_window,
                        )
                    for cell, rep in spread_reports.items():
                        rep["best_cost"] = best_cost
                        rep["cost"] = cost
                        spot_by_cell[(cell, contrast)] = rep
                        reports.append(rep)
                        _emit_report(
                            rep,
                            run_dir=run_dir,
                            do_print=not do_json,
                            do_figure=do_figure,
                            file_suffix=file_suffix,
                            html=html,
                        )
                elif task in ("mbar", "sbar"):
                    at_x = cli.xs[0] if hex_mode and cli.xs is not None else None
                    at_y = cli.ys[0] if hex_mode and cli.ys is not None else None
                    cache_key = (task, contrast)
                    if cache_key not in bar_session_cache:
                        bar_session_cache[cache_key] = plot.session_from_task(
                            session, task, contrast,
                        )
                    session_bar = bar_session_cache[cache_key]
                    if cache_key not in bar_meta_cache:
                        bar_meta_cache[cache_key] = _bar_meta(
                            session_bar, task, contrast,
                        )
                    specs, grids = bar_meta_cache[cache_key]
                    cells_bar = [cli.cells[0]] if hex_mode else cli.cells
                    specs_ordered = _bar_specs_requested(
                        session_bar, task, contrast, cells_bar, cli.specs_req,
                        specs=specs, grids=grids,
                    )
                    if not specs_ordered:
                        continue
                    multi_report = do_figure and len(specs_ordered) > 1
                    if hex_mode:
                        _log(
                            f"component forward {task} {contrast} "
                            f"specs={specs_ordered} "
                            f"hex=({at_x},{at_y}) (no component forward) ..."
                        )
                        reports_by_spec = analyze_bar_hex(
                            session_bar,
                            params=params,
                            cell=cells_bar[0],
                            task=task,
                            contrast=contrast,
                            spec_tokens=specs_ordered,
                            at_x=float(at_x),
                            at_y=float(at_y),
                            node=node,
                            time_window=time_window,
                            specs=specs,
                            grids=grids,
                        )
                    elif slice_mode:
                        axis = "x" if cli.xs is not None else "y"
                        coordinate_values = cli.xs if cli.xs is not None else cli.ys
                        slice_reports: dict[
                            str, dict[str, list[dict[str, Any]]]
                        ] = {
                            spec: {cell: [] for cell in cells_bar}
                            for spec in specs_ordered
                        }
                        for value in coordinate_values:
                            slice_x = float(value) if axis == "x" else None
                            slice_y = float(value) if axis == "y" else None
                            _log(
                                f"component forward {task} {contrast} "
                                f"specs={specs_ordered} {axis}={value} slice ..."
                            )
                            one_coordinate = analyze_bar_slice(
                                session_bar,
                                params=params,
                                cells=cells_bar,
                                task=task,
                                contrast=contrast,
                                spec_tokens=specs_ordered,
                                at_x=slice_x,
                                at_y=slice_y,
                                time_window=time_window,
                                specs=specs,
                                grids=grids,
                            )
                            for spec in specs_ordered:
                                for cell, rep in one_coordinate[spec].items():
                                    rep["best_cost"] = best_cost
                                    rep["cost"] = cost
                                    bar_by_cell[(cell, contrast)] = rep
                                    reports.append(rep)
                                    slice_reports[spec][cell].append(rep)
                                    if not do_json:
                                        print("")
                                        _print_report(rep, print_steps=not do_figure)
                        if do_figure:
                            for spec in specs_ordered:
                                for cell in cells_bar:
                                    coordinate_reports = slice_reports[spec][cell]
                                    path = os.path.join(
                                        run_dir,
                                        "cell_dynamics",
                                        _compare_figure_filename(
                                            coordinate_reports,
                                            file_suffix=file_suffix,
                                            html=html,
                                        ),
                                    )
                                    plot_reports_compare(coordinate_reports, path)
                        continue
                    elif task == "sbar" and cli.mids_req is not None:
                        mid_reports: dict[
                            str, dict[str, list[dict[str, Any]]]
                        ] = {
                            spec: {cell: [] for cell in cells_bar}
                            for spec in specs_ordered
                        }
                        for mid in cli.mids_req:
                            _log(
                                f"component forward {task} {contrast} "
                                f"specs={specs_ordered} mid={mid} ..."
                            )
                            one_mid = analyze_bar_average(
                                session_bar,
                                params=params,
                                cells=cells_bar,
                                task=task,
                                contrast=contrast,
                                spec_tokens=specs_ordered,
                                time_window=time_window,
                                specs=specs,
                                grids=grids,
                                mids_req=[mid],
                            )
                            for spec in specs_ordered:
                                for cell, rep in one_mid[spec].items():
                                    rep["best_cost"] = best_cost
                                    rep["cost"] = cost
                                    bar_by_cell[(cell, contrast)] = rep
                                    reports.append(rep)
                                    mid_reports[spec][cell].append(rep)
                                    if not do_json:
                                        print("")
                                        _print_report(rep, print_steps=not do_figure)
                        if do_figure:
                            for spec in specs_ordered:
                                for cell in cells_bar:
                                    reports_for_mid = mid_reports[spec][cell]
                                    path = os.path.join(
                                        run_dir,
                                        "cell_dynamics",
                                        _compare_figure_filename(
                                            reports_for_mid,
                                            file_suffix=file_suffix,
                                            html=html,
                                        ),
                                    )
                                    plot_reports_compare(reports_for_mid, path)
                        continue
                    else:
                        _log(
                            f"component forward {task} {contrast} "
                            f"specs={specs_ordered} "
                            f"(no component forward) ..."
                        )
                        reports_by_spec = analyze_bar_average(
                            session_bar,
                            params=params,
                            cells=cells_bar,
                            task=task,
                            contrast=contrast,
                            spec_tokens=specs_ordered,
                            time_window=time_window,
                            specs=specs,
                            grids=grids,
                        )
                    reports_by_cell: dict[str, list[dict[str, Any]]] = {
                        cell: [] for cell in cells_bar
                    }
                    for spec in specs_ordered:
                        for cell, rep in reports_by_spec[spec].items():
                            rep["best_cost"] = best_cost
                            rep["cost"] = cost
                            bar_by_cell[(cell, contrast)] = rep
                            reports.append(rep)
                            if multi_report:
                                reports_by_cell[cell].append(rep)
                            _emit_report(
                                rep,
                                run_dir=run_dir,
                                do_print=not do_json,
                                do_figure=do_figure and not multi_report,
                                file_suffix=file_suffix,
                                html=html,
                            )
                    if multi_report:
                        for cell in cells_bar:
                            reports = reports_by_cell[cell]
                            path = os.path.join(
                                run_dir,
                                "cell_dynamics",
                                _compare_figure_filename(
                                    reports, file_suffix=file_suffix, html=html,
                                ),
                            )
                            plot_reports_compare(reports, path)
                else:
                    raise SystemExit(f"unsupported task {task!r}")

        if not do_json and spot_by_cell and bar_by_cell:
            _print_sign_compare(spot_by_cell, bar_by_cell)

        if do_json:
            print(json.dumps(
                {
                    "run": run_dir,
                    "best_cost": best_cost,
                    "cost": cost,
                    "reports": reports,
                },
                indent=2,
            ))


if __name__ == "__main__":
    main()
