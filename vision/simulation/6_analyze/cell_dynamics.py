"""Borst / hp_lp v component, or ``--trace-only`` response curves."""

from __future__ import annotations

DEFAULT_RUN_NAME = """
0801_183307-run-nofsteps-0
""".strip()
DEFAULT_RUN_PATH = "hp_lp/" + DEFAULT_RUN_NAME

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
import figure.plot_run as plot_trained
from figure import moving_bar as moving_bar_plot
from figure import spot as spot_plot
from figure.readout import contrast_for_task
from figure.spot import CENTER_BIN, pack_spot_cost_radii, resolve_spot_data_cubes
from figure.util import parse_axis_slice_list, plot_sem_band, slice_xy_label
from import_bootstrap import parse_bool, parse_comma_list
from network.construction import col2gt
from task.moving_bar.data import (
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
from task.spot.data import spot_center_bin_layout
from task.spot.input import (
    spot_from_opts,
    spot_stimulus_batches,
)
from training.driver import parse_task_list

__doc__ = """Borst / hp_lp v component, or ``--trace-only`` response curves (v).

Shared CLI
----------
``CELL,...`` ``--run`` ``--task`` ``--spec`` ``--x`` ``--y``. Pass comma lists in
one process (do not re-invoke once per cell/spec).

Component mode (default)
---------------------
Per ``--run``: one ``load_best``; one batched v component walk per distinct task.
``--t-rel START:STOP`` window vs peak (default ``-10:10``);
``--t START:STOP`` absolute window (mutually exclusive).

* Omit ``--x`` / ``--y``: cost-extent **average**.
* Exactly one ``--x`` and one ``--y``: **hex** (moving_bar only; one cell).
* Multiple x/y: rejected.

``--plot true|false``: PNGs under ``{run}/cell_dynamics/`` (default true).
``--syn-strength SRC:TAR=VALUE ...`` overrides ``syn_strength_cell`` before walks.

``--trace-only``
----------------
Full plot forward; print curve summaries (no CSV, no component walk).

* one spot forward per distinct ``spot_*``; one bar forward per ``moving_bar_*``
* spot: center-bin impulse + RF (same scaling as spot plots)
* bar: ``t_first_sti``-aligned; summaries use post-onset
  (``idx >= cost_window_start_idx``)
* optional ``--x`` / ``--y`` overlays (multi ok); spot needs both or neither
* ``--values`` / ``--pre-ms`` / ``--response-ms`` / ``--t-max-ms`` only here

Examples
--------
  ../.venv/bin/python analyze/cell_dynamics.py \\
    Mi4,Mi9 --run /abs/path/to/run \\
    --task spot_bright,moving_bar_bright --spec right_bright_w1

  ../.venv/bin/python analyze/cell_dynamics.py \\
    L3 --run /abs/path/to/run --task moving_bar_bright \\
    --spec right_bright_w1 --x -2 --y -1

  ../.venv/bin/python analyze/cell_dynamics.py \\
    T4a --run borst/27252028-... --task moving_bar_bright \\
    --spec left_bright_w4,right_bright_w4 --syn-strength Mi4:T4a=2.0 Mi9:T4a=1.0 \\
    --t-rel -5:15

  ../.venv/bin/python analyze/cell_dynamics.py \\
    Mi4,Mi9 --run /abs/path/to/run --trace-only \\
    --task spot_bright,moving_bar_bright

  ../.venv/bin/python analyze/cell_dynamics.py \\
    L4 --run /abs/path/to/run --trace-only --task spot_dark --x 2 --y 1
"""



def _parse_t_range(token: str, *, flag: str) -> tuple[int, int]:
    """Parse ``START:STOP`` (colon; one token)."""
    parts = str(token).split(":")
    if len(parts) != 2 or parts[0] == "" or parts[1] == "":
        raise SystemExit(f"{flag} must be START:STOP")
    start, stop = int(parts[0]), int(parts[1])
    if start > stop:
        raise SystemExit(f"{flag} START={start} > STOP={stop}")
    return start, stop


@dataclass(frozen=True)
class TimeWindow:
    """``t_rel``: offsets vs |v_post_d| peak; ``t``: absolute aligned indices."""

    kind: str  # "t_rel" | "t"
    start: int
    stop: int

    def __post_init__(self) -> None:
        if self.kind not in ("t_rel", "t"):
            raise ValueError(f"TimeWindow.kind must be t_rel|t; got {self.kind!r}")

    @property
    def walk_t_start(self) -> int | None:
        return self.start if self.kind == "t" else None

    @property
    def walk_t_stop(self) -> int | None:
        return self.stop if self.kind == "t" else None


@dataclass(frozen=True)
class SharedCli:
    """Parsed shared CLI for component / ``--trace-only``."""

    cells: list[str]
    tasks: list[str]
    specs_req: list[str] | None
    x_list: list | None
    y_list: list | None
    slice_label: str | None


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
    tasks = parse_task_list(args.task)
    if not tasks:
        raise SystemExit("--task is required")
    specs_req = parse_comma_list(args.spec) if args.spec is not None else None
    for t in tasks:
        if t not in training.SPOT_TASKS and t not in training.MOVING_BAR_TASKS:
            raise SystemExit(
                f"unsupported task {t!r}; expected spot_* or moving_bar_* "
                f"(after TASK_ALIASES expansion)"
            )
    x_list = parse_axis_slice_list(args.x)
    y_list = parse_axis_slice_list(args.y)
    if (x_list is None) ^ (y_list is None):
        if any(t in training.SPOT_TASKS for t in tasks):
            raise SystemExit("spot tasks require both --x and --y or neither")
    slice_label = None
    if x_list is not None and y_list is not None and len(x_list) == 1 and len(y_list) == 1:
        slice_label = slice_xy_label(x_list[0], y_list[0])
    return SharedCli(
        cells=cells,
        tasks=tasks,
        specs_req=specs_req,
        x_list=x_list,
        y_list=y_list,
        slice_label=slice_label,
    )


# ---------------------------------------------------------------------------
# --trace-only: response curves (no component walk)
# ---------------------------------------------------------------------------


@dataclass
class CurveSummary:
    n: int
    max: float
    min: float
    final: float
    peak_idx: int
    trough_idx: int
    sign_changes: int


def _summarize(arr: np.ndarray, *, idx_offset: int = 0) -> CurveSummary:
    signs = np.sign(arr)
    sign_changes = int(np.sum(signs[1:] * signs[:-1] < 0))
    return CurveSummary(
        n=int(arr.size),
        max=float(np.max(arr)),
        min=float(np.min(arr)),
        final=float(arr[-1]),
        peak_idx=int(np.argmax(arr)) + idx_offset,
        trough_idx=int(np.argmin(arr)) + idx_offset,
        sign_changes=sign_changes,
    )


def _shape_label(s: CurveSummary) -> str:
    amp = max(abs(s.max), abs(s.min), 1e-9)
    pos = s.max > 0.05 * amp
    neg = s.min < -0.05 * amp
    if pos and neg:
        order = "+ then -" if s.peak_idx < s.trough_idx else "- then +"
        return f"biphasic ({order})"
    if pos:
        return "monophasic positive"
    if neg:
        return "monophasic negative"
    return "flat/near-zero"


def _float_curve(arr) -> np.ndarray:
    return np.asarray(arr, dtype=float)


def extract_spot_bundle(session, z, *, task: str, x_list, y_list):
    """One spot forward via ``plot_trained.spot_bundle_fns``.

    Returns ``(session_one, bundle, data_cubes)`` where ``data_cubes`` is
    ``{contrast: {cell: (9, T)}}``.
    """
    one = plot_trained.session_for_task(session, task)
    make_bundle, _, _ = plot_trained.spot_bundle_fns(one)
    bundle = make_bundle(
        one,
        z,
        at_x_list=x_list,
        at_y_list=y_list,
        save_trace_csv_dir=None,
    )
    data_cubes = spot_plot.resolve_spot_data_cubes(
        {contrast_for_task(one.primary_readout.name): one},
    )
    return one, bundle, data_cubes


def extract_moving_bar_bundle(session, z, *, task: str, x_list, y_list):
    """One moving-bar forward; all cells + all specs live on the returned bundle."""
    one = plot_trained.session_for_task(session, task)
    return moving_bar_plot.moving_bar_trace_bundle(
        one,
        z,
        task,
        at_x_list=x_list,
        at_y_list=y_list,
        save_trace_csv_dir=None,
    )


def extract_spot_cell_curves(bundle, data_cubes, *, cell: str, slice_label: str | None = None):
    center = spot_plot.CENTER_BIN
    cell_on = next((c for c in bundle.cells if c["name"] == cell), None)
    if cell_on is None:
        avail = sorted(c["name"] for c in bundle.cells)
        raise SystemExit(f"cell {cell!r} not in spot bundle; available: {avail}")
    contrast = contrast_for_task(bundle.session.primary_readout.name)
    data_on = (data_cubes or {}).get(contrast) or {}
    if cell not in data_on:
        raise SystemExit(
            f"cell {cell!r} not in spot data cubes[{contrast!r}]; "
            f"keys={sorted(data_on)}"
        )

    if bundle.response_start is None:
        raise SystemExit(
            "spot bundle missing response_start (t_onset); "
            "cannot scale spot time courses"
        )
    if bundle.pulse_end is None:
        raise SystemExit(
            "spot bundle missing pulse_end; "
            "cannot scale spot RF peak inside pulse window"
        )
    sc_kw = dict(response_start=bundle.response_start, pulse_end=bundle.pulse_end)
    imp_total, rf_total = spot_plot.scale_curve(
        cell_on["cube"], center, **sc_kw,
    )
    imp_data, rf_data = spot_plot.scale_curve(
        data_on[cell], center, **sc_kw,
    )

    out: dict[str, np.ndarray] = {
        "time_total_model": _float_curve(imp_total),
        "time_total_data": _float_curve(imp_data),
        "rf_total_model": _float_curve(rf_total),
        "rf_total_data": _float_curve(rf_data),
    }

    if slice_label and bundle.slice_overlay and slice_label in bundle.slice_overlay:
        cubes = bundle.slice_overlay[slice_label]
        if cell in cubes:
            imp_slice, rf_slice = spot_plot.scale_curve(
                cubes[cell], center, **sc_kw,
            )
            out[f"time_slice_model:{slice_label}"] = _float_curve(imp_slice)
            out[f"rf_slice_model:{slice_label}"] = _float_curve(rf_slice)
    return out


def extract_moving_bar_cell_curves(bundle, *, cell: str, spec: str) -> dict[str, np.ndarray]:
    """Per-spec traces matching one ``bar_all_ca`` panel (slices + total)."""
    key = (cell, spec)
    ca_mean = bundle.traces.ca_mean
    if key not in ca_mean:
        avail_cells = sorted({c for c, _ in ca_mean})
        avail_specs = sorted(s for c, s in ca_mean if c == cell)
        if cell not in avail_cells:
            raise SystemExit(f"cell {cell!r} not in bar bundle; available cells: {avail_cells}")
        raise SystemExit(f"spec {spec!r} not found for cell {cell!r}; available: {avail_specs}")

    out: dict[str, np.ndarray] = {}
    labels = list(bundle.slice_overlay.keys()) if bundle.slice_overlay else []
    for label in labels:
        wt = bundle.slice_overlay[label]
        if key in wt.ca_mean:
            out[label] = _float_curve(wt.ca_mean[key])
    out["total"] = _float_curve(ca_mean[key])
    return out


def specs_for_cell(bundle, cell: str, requested: list[str] | None) -> list[str]:
    ca_mean = bundle.traces.ca_mean
    avail = sorted(s for c, s in ca_mean if c == cell)
    if not avail:
        cells = sorted({c for c, _ in ca_mean})
        raise SystemExit(f"cell {cell!r} not in bar bundle; available cells: {cells}")
    try:
        return filter_requested_specs(avail, requested)
    except ValueError as exc:
        raise SystemExit(f"{exc}; cell={cell!r}") from exc


def _print_trace_curve(
    name: str,
    arr: np.ndarray,
    *,
    before_t: int | None,
    print_values: bool,
    head_t: int | None,
    head_window: str | None,
):
    """Summarize + (optionally) list values for a trace window."""
    if before_t is not None and 0 < before_t < arr.size:
        start = before_t
        window = f"post_onset[idx>={before_t}]"
    else:
        start = 0
        window = "full"

    if head_t is not None:
        end = min(arr.size, start + head_t)
        use = arr[start:end]
        idx_offset = start
        if head_window:
            window = f"{window} {head_window}"
        else:
            window = f"{window} head[{start}:{end}]"
    else:
        use = arr[start:]
        idx_offset = start
    s = _summarize(use, idx_offset=idx_offset)
    shape = _shape_label(s)
    print(
        f"{name} ({window}): n={s.n} max={s.max:.6g} min={s.min:.6g} "
        f"final={s.final:.6g} peak_idx={s.peak_idx} "
        f"trough_idx={s.trough_idx} sign_changes={s.sign_changes}  "
        f"shape={shape}"
    )
    if print_values:
        print(f"  values: {np.round(use, 4).tolist()}")


def _print_trace_block(
    *,
    run_i: int,
    run_dir: str,
    best_i: int,
    best_cost: float,
    cell: str,
    task: str,
    session,
    bundle,
    curves: dict[str, np.ndarray],
    x_list,
    y_list,
    print_values: bool,
    head_t: int | None,
    head_window: str | None,
    spec: str | None = None,
):
    before_t = None
    if spec is not None and bundle.traces.before_t is not None:
        before_t = bundle.traces.before_t.get(spec)
    head = (
        f"best_i={best_i}  best_cost={best_cost:.6g}  cell={cell}  "
        f"task={task}  trace_kind=v"
    )
    if spec is not None:
        head += f"  spec={spec}"
    print("")
    print(f"== RUN {run_i}: {run_dir} ==")
    print(head)
    print(f"n_t={bundle.n_t}  delta_ms={getattr(session, 'delta_ms', 'NA')}")
    if x_list is not None or y_list is not None:
        print(f"slice_x={x_list}  slice_y={y_list}")
    if before_t is not None:
        print(f"cost_window_start_idx={before_t}")
    if spec is not None:
        slice_names = [k for k in curves if k != "total"]
        order = slice_names + ["total"]
        print(f"traces ({len(order)}): {', '.join(order)}")
        for name in order:
            _print_trace_curve(
                name,
                curves[name],
                before_t=before_t,
                print_values=print_values,
                head_t=head_t,
                head_window=head_window,
            )
    else:
        for key, arr in curves.items():
            if head_t is not None and not key.startswith("time_"):
                continue
            _print_trace_curve(
                key,
                arr,
                before_t=None,
                print_values=print_values,
                head_t=head_t,
                head_window=head_window,
            )


def _maybe_override_spot_timing(
    *,
    run_dir: str,
    session,
    pre_ms: float | None,
    response_ms: float | None,
):
    """Optionally re-open the session with overridden spot timing."""
    if pre_ms is None:
        return session, None, None

    opts = plot_trained.load_train_opts(run_dir)
    if opts is None:
        raise SystemExit(f"missing train opts under {run_dir}")
    if response_ms is None:
        response_ms = float(training.RESPONSE_MS)

    for key in ("spot_bright_stimulus_opts", "spot_dark_stimulus_opts"):
        so = opts.get(key)
        if so is not None:
            so["pre_ms"] = float(pre_ms)
            so["response_ms"] = float(response_ms)
            so.pop("t_onset", None)
            so.pop("n_t", None)

    new_session = training.open_session_from_opts(opts, model=opts.get("model"))
    dt = float((opts.get("spot_bright_stimulus_opts") or {}).get("delta_ms", 10.0))
    new_t_onset = training.ms_to_t(pre_ms, delta_ms=dt)
    return new_session, dt, new_t_onset


def _run_trace_only(args: argparse.Namespace, cli: SharedCli) -> None:
    import training.driver as train_mod

    for run_i, run_arg in enumerate(args.run):
        run_dir = plot_trained.resolve_run_dir(run_arg)
        session0, _z0, best_i, best_cost = plot_trained.load_best(run_dir)

        session, z = session0, _z0
        dt_for_head = None
        head_t = None
        head_window = None
        if args.pre_ms is not None:
            session, dt_for_head, _new_t_onset = _maybe_override_spot_timing(
                run_dir=run_dir,
                session=session0,
                pre_ms=args.pre_ms,
                response_ms=args.response_ms,
            )
            named, cell_names, pair_names = train_mod.load_best_param_named(run_dir)
            remapped = training.remap_named_node_values(
                named,
                cell_names,
                pair_names,
                list(session.schema),
                session.backend,
            )
            schema = training.attach_param_carry(list(session.schema), remapped)
            session = session.with_schema(schema)
            z = training.node_values_to_z(
                remapped,
                schema,
                dtype=session.sim_dtype,
                device=session.device,
            )

        if args.t_max_ms is not None:
            if dt_for_head is None:
                opts = plot_trained.load_train_opts(run_dir)
                if opts is None:
                    raise SystemExit(f"missing train opts under {run_dir}")
                dt_for_head = float(
                    (opts.get("spot_bright_stimulus_opts") or {}).get("delta_ms", 10.0)
                )
            head_t = training.ms_to_t(args.t_max_ms, delta_ms=dt_for_head)
            head_window = f"head[t<{args.t_max_ms:g}ms]"

        spot_cache: dict[str, tuple] = {}
        bar_cache: dict[str, object] = {}

        for task in cli.tasks:
            if task in training.SPOT_TASKS:
                if task not in spot_cache:
                    spot_cache[task] = extract_spot_bundle(
                        session,
                        z,
                        task=task,
                        x_list=cli.x_list,
                        y_list=cli.y_list,
                    )
                _one, bundle, data_cubes = spot_cache[task]
                for cell in cli.cells:
                    curves = extract_spot_cell_curves(
                        bundle, data_cubes, cell=cell, slice_label=cli.slice_label,
                    )
                    _print_trace_block(
                        run_i=run_i,
                        run_dir=run_dir,
                        best_i=best_i,
                        best_cost=best_cost,
                        cell=cell,
                        task=task,
                        session=session,
                        bundle=bundle,
                        curves=curves,
                        x_list=cli.x_list,
                        y_list=cli.y_list,
                        print_values=args.values,
                        head_t=head_t,
                        head_window=head_window,
                    )
            else:
                if task not in bar_cache:
                    bar_cache[task] = extract_moving_bar_bundle(
                        session,
                        z,
                        task=task,
                        x_list=cli.x_list,
                        y_list=cli.y_list,
                    )
                bundle = bar_cache[task]
                for cell in cli.cells:
                    for spec in specs_for_cell(bundle, cell, cli.specs_req):
                        curves = extract_moving_bar_cell_curves(
                            bundle, cell=cell, spec=spec,
                        )
                        _print_trace_block(
                            run_i=run_i,
                            run_dir=run_dir,
                            best_i=best_i,
                            best_cost=best_cost,
                            cell=cell,
                            task=task,
                            session=session,
                            bundle=bundle,
                            curves=curves,
                            x_list=cli.x_list,
                            y_list=cli.y_list,
                            print_values=args.values,
                            head_t=head_t,
                            head_window=head_window,
                            spec=spec,
                        )

# Component-step fields plotted vs time (key, ylabel/legend).
_BORST_PLOT_PANELS: list[tuple[str, list[tuple[str, str]]]] = [
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
            ("v_post_mV", "v_post"),
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
            ("X", "X"),
            ("a", "a"),
            ("X_minus_a", "X-a"),
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
    "num_exc", "num_inh", "num_leak", "num_ihon", "num_ihoff", "num_cdt",
    "num", "den",
)
_HP_LP_COMPONENT_KEYS = (
    "v_pre_d", "v_abs", "i_sti", "v_sti", "v_in", "v_in_exc", "v_in_inh",
    "a", "X", "X_minus_a", "dv_leak", "dv_hp",
)

_BORST_PLOT_KEY_COMPONENT: dict[str, str | None] = {
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
    "i_sti": "i_sti",
    "num_over_den": "v_abs",
    "num": "num",
    "den": "den",
}
_HP_LP_PLOT_KEY_COMPONENT: dict[str, str | None] = {
    "v_post_mV": "v_abs",
    "v_in": "v_in",
    "v_in_exc": "v_in_exc",
    "v_in_inh": "v_in_inh",
    "v_sti": "v_sti",
    "a": "a",
    "X": "X",
    "X_minus_a": "X_minus_a",
    "dv_leak": "dv_leak",
    "dv_hp": "dv_hp",
    "dv_leak_plus_hp": None,
}

_BORST_FORMULA_G_TOKENS: list[tuple[str, str | None]] = [
    ("v_post", "v_post"),
    (" = (", None),
    ("cdt", "cdt"), ("·v_pre + ", None),
    ("E_exc·", None), ("g_exc", "g_exc"), (" + ", None),
    ("E_inh·", None), ("g_inh", "g_inh"), (" + ", None),
    ("E_leak·", None), ("g_leak", "g_leak"), (" + ", None),
    ("E_h_on·", None), ("g_h_on", "g_h_on"), (" + ", None),
    ("E_h_off·", None), ("g_h_off", "g_h_off"), (" + ", None),
    ("i_sti", "i_sti"),
    (") / (", None),
    ("cdt", "cdt"), (" + ", None),
    ("g_exc", "g_exc"), (" + ", None),
    ("g_inh", "g_inh"), (" + ", None),
    ("g_h_on", "g_h_on"), (" + ", None),
    ("g_h_off", "g_h_off"), (" + ", None),
    ("g_leak", "g_leak"),
    (")", None),
]

_BORST_FORMULA_TOKENS: list[tuple[str, str | None]] = [
    ("v_post", "v_post"),
    (" = (", None),
    ("i_cdt", "i_cdt"), (" + ", None),
    ("i_exc", "i_exc"), (" + ", None),
    ("i_inh", "i_inh"), (" + ", None),
    ("i_leak", "i_leak"), (" + ", None),
    ("i_h_on", "i_h_on"), (" + ", None),
    ("i_h_off", "i_h_off"), (" + ", None),
    ("i_sti", "i_sti"),
    (") / (", None),
    ("cdt", "cdt"), (" + ", None),
    ("g_exc", "g_exc"), (" + ", None),
    ("g_inh", "g_inh"), (" + ", None),
    ("g_h_on", "g_h_on"), (" + ", None),
    ("g_h_off", "g_h_off"), (" + ", None),
    ("g_leak", "g_leak"),
    (")", None),
]

_HP_LP_FORMULA_G_TOKENS: list[tuple[str, str | None]] = [
    ("v_post", "v_post"),
    (" = v_pre + (dt/τ_lp)[-(v_pre−v_rest) + G·(v_rest + ", None),
    ("v_in", "v_in"), (" + ", None),
    ("i_sti/g_in", "v_sti"), (" − ", None),
    ("a", "a"), (")]", None),
]

_HP_LP_FORMULA_TOKENS: list[tuple[str, str | None]] = [
    ("v_post", "v_post"), (" = v_pre + ", None),
    ("dv_leak", "dv_leak"), (" + ", None),
    ("dv_hp", "dv_hp"),
]

_BLACK_TRACE_LABELS = frozenset({"num", "den", "X", "a", "X-a", "dv_hp"})
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


def _component_spec(model: str) -> _ComponentSpec:
    if model == "borst":
        return _ComponentSpec(
            model="borst",
            keys=_BORST_COMPONENT_KEYS,
            plot_panels=_BORST_PLOT_PANELS,
            plot_key_component=_BORST_PLOT_KEY_COMPONENT,
            formula_g=_BORST_FORMULA_G_TOKENS,
            formula_i=_BORST_FORMULA_TOKENS,
            row_shared_ylim=_BORST_ROW_SHARED_YLIM,
        )
    if model == "hp_lp":
        return _ComponentSpec(
            model="hp_lp",
            keys=_HP_LP_COMPONENT_KEYS,
            plot_panels=_HP_LP_PLOT_PANELS,
            plot_key_component=_HP_LP_PLOT_KEY_COMPONENT,
            formula_g=_HP_LP_FORMULA_G_TOKENS,
            formula_i=_HP_LP_FORMULA_TOKENS,
            row_shared_ylim=_HP_LP_ROW_SHARED_YLIM,
        )
    raise SystemExit(f"cell_dynamics supports borst|hp_lp; got {model!r}")


def _trace_ylabel(group_ylabel: str, label: str) -> str:
    if label == "num/den":
        return "num/den (mV)"
    if label == "num":
        return "num (pA)"
    if label == "den":
        return "den (nS)"
    if "(" in group_ylabel:
        node = " " + group_ylabel[group_ylabel.index("("):]
        return f"{label}{node}"
    return label


def _component_axes_grid(spec: _ComponentSpec):
    """Return ``(n_rows, n_hexes)`` grid: one row per plot panel group."""
    return len(spec.plot_panels), spec.n_plot_cols


def _plot_colors():
    import matplotlib.pyplot as plt

    return plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _plot_trace_colors(colors: list[str], spec: _ComponentSpec) -> dict[str, str]:
    """Map trace legend label → subplot color (hex index within its row)."""
    out: dict[str, str] = {}
    for _group_ylabel, series in spec.plot_panels:
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


def _finish_component_figure_layout(
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
    """Model ``prepare_i_sti`` on a ``(B, T, N)`` pack ``i_sti``."""
    pack = session.primary_readout
    return _model_driver(session).prepare_i_sti(session, p, i_sti, pack)


def _equilibrate(session, p, i_sti_batch: torch.Tensor, t_onset: int):
    """Equilibrate to ``t_onset``; returns ``(v, state)`` where state is model-specific."""
    _component_spec(session.model)  # validate early
    B, T, _N = i_sti_batch.shape
    drv = _model_driver(session)
    state, v = drv.init_state(session, p, B)
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
            "num": (
                terms["num_exc"] + terms["num_inh"] + terms["num_leak"]
                + terms["num_ihon"] + terms["num_ihoff"] + terms["num_cdt"] + terms["i_sti"]
            ),
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
            "i_sti", "v_sti", "v_in", "v_in_exc", "v_in_inh", "a", "X", "X_minus_a",
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
    cdt: float = 0.0,
) -> dict[str, Any]:
    """One step dict from per-key sums over ``n`` nodes."""
    if n <= 0:
        raise ValueError("empty node set for mean component")
    base = {
        "t": t,
        "t_rel": t_rel,
        "ti": ti,
        "v_post_mV": float(v_post_val),
        "v_pre_d_mV": acc["v_pre_d"] / n,
        "v_post_minus_pre_mV": v_post_minus_pre_sum / n,
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
            "cdt_nS": float(cdt),
            "g_Ih_on_nS": acc["g_Ih_on"] / n,
            "g_Ih_off_nS": acc["g_Ih_off"] / n,
            "num_exc": acc["num_exc"] / n,
            "num_inh": acc["num_inh"] / n,
            "num_leak": acc["num_leak"] / n,
            "num_ihon": acc["num_ihon"] / n,
            "num_ihoff": acc["num_ihoff"] / n,
            "num_cdt": acc["num_cdt"] / n,
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
        "a": acc["a"] / n,
        "X": acc["X"] / n,
        "X_minus_a": acc["X_minus_a"] / n,
        "dv_leak": acc["dv_leak"] / n,
        "dv_hp": acc["dv_hp"] / n,
        "dv_leak_plus_hp": (acc["dv_leak"] + acc["dv_hp"]) / n,
    })
    return base


@dataclass
class _ComponentWalkBatch:
    """One i_sti batch row for the shared post-t_onset v walk."""

    all_nodes: np.ndarray
    t0_u: np.ndarray
    win_len: int
    node_to_cell: dict[int, str]
    nodes_by_cell: dict[str, np.ndarray]


@dataclass
class _ComponentWalkAccum:
    sums: list[dict[str, np.ndarray]]
    sumsq: list[dict[str, np.ndarray]]
    counts: list[dict[str, np.ndarray]]
    v_post_minus_pre_sums: list[dict[str, np.ndarray]]
    spec: _ComponentSpec


def _walk_component(
    session,
    p,
    i_sti: torch.Tensor,
    batches: list[_ComponentWalkBatch],
    cells: list[str],
    *,
    t_start: int | None = None,
    t_stop: int | None = None,
) -> _ComponentWalkAccum:
    """Equilibrate once; walk post-t_onset; accumulate component + absolute v_post.

    Shared by bar average, spot average, and bar hex.
    Aligned index ``t = t_global - t0_u``. v_post is mean absolute ``v_abs``;
    SEM uses sum / sumsq like ``sem_from_traces``.

    If ``t_start``/``t_stop`` are set (``--t``), only accumulate inside that
    inclusive absolute window; cheap steps before it; break after every node
    has passed ``t_stop``.
    """
    if not batches:
        raise SystemExit("component walk requires at least one batch")
    if (t_start is None) ^ (t_stop is None):
        raise SystemExit("t_start and t_stop must both be set or both omitted")
    if t_start is not None and t_start > t_stop:
        raise SystemExit(f"t_start={t_start} > t_stop={t_stop}")
    B, T, _N = i_sti.shape
    if B != len(batches):
        raise SystemExit(f"i_sti B={B} != len(batches)={len(batches)}")

    spec = _component_spec(session.model)
    drive = _prepare_drive(session, p, i_sti)
    t_onset = int(session.primary_readout.i_sti.shape[1] - session.primary_readout.data.shape[1])
    trace_len = T - t_onset

    t_last: int | None = None
    if t_stop is not None:
        t_last = max(int(plan.t0_u.max()) + int(t_stop) for plan in batches)

    v, state = _equilibrate(session, p, drive, t_onset)
    v_onset = v.detach().cpu().numpy().copy()
    backend = session.backend
    n_keys = spec.n_keys

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
    drv = _model_driver(session)

    for ti in range(trace_len):
        t_global = t_onset + ti
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
                    E_LEAK_REST=session.E_LEAK_REST,
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

    return _ComponentWalkAccum(
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
    if time_window.kind == "t":
        t_lo, t_hi = int(time_window.start), int(time_window.stop)
        if t_lo < 0 or t_hi >= n or t_lo > t_hi:
            raise SystemExit(
                f"--t {t_lo}:{t_hi} out of range for accum length {n}"
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
            cdt=float(training.membrane_cdt(session.capac, session.delta_ms)),
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
        "v_post_d_peak_mV": float(v_post_d[peak_t]),
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


def _node_params(p, session, node: int) -> dict[str, float]:
    backend = session.backend
    if session.model == "hp_lp":
        return {
            "in_gain": float(p["in_gain"][node]),
            "out_gain": float(p["out_gain"][node]),
            "v_rest_mV": float(p["v_rest"][node]),
            "tau_lp_ms": float(p["tau_lp"][node]),
            "tau_hp_ms": float(p["tau_hp"][node]),
            "hp_gain": float(p["hp_gain"][node]),
        }
    from neuron.schema import borst_ih_off_kwargs

    ih_off = (session.train_opts or {})["ih_off"]
    gmax_off, *_rest = borst_ih_off_kwargs(p, ih_off)
    return {
        "in_gain": float(p["in_gain"][node]),
        "out_gain": float(p["out_gain"][node]),
        "v_th_mV": float(p["v_th"][node]),
        "Ih_gmax": float(p["Ih_gmax"][node]),
        "Ih_gmax_off": float(gmax_off[node]),
        "e_leak_mV": float(backend.e_leak[node]),
    }


def _globals(session):
    pack = session.primary_readout
    t_onset = int(pack.i_sti.shape[1] - pack.data.shape[1])
    if session.model == "hp_lp":
        return {
            "delta_ms": float(session.delta_ms),
            "state_clamp": float(session.STATE_CLAMP),
            "g_in_nS": float(session.g_in),
            "t_onset": t_onset,
        }
    return {
        "E_exc": float(session.E_exc),
        "E_inh": float(session.E_inh),
        "E_Ih": float(session.E_Ih),
        "E_IH_OFF": float(training.e_ih_off(session.E_LEAK_REST, session.E_Ih)),
        "g_leak_nS": float(session.g_leak),
        "cdt": float(training.membrane_cdt(session.capac, session.delta_ms)),
        "delta_ms": float(session.delta_ms),
        "t_onset": t_onset,
    }


def _node_to_cell_map(nodes_by_cell: dict[str, np.ndarray]) -> dict[int, str]:
    u2c: dict[int, str] = {}
    for cell, us in nodes_by_cell.items():
        for u in np.asarray(us, dtype=np.int64).ravel():
            u2c[int(u)] = cell
    return u2c


def _node_cell_lookup(plan: _ComponentWalkBatch, cells: list[str]) -> np.ndarray:
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


def _merge_walk_accum(
    accum: _ComponentWalkAccum,
    walk_batches: list[_ComponentWalkBatch],
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
    n_keys = accum.spec.n_keys
    sums = {c: np.zeros((win_len, n_keys), dtype=float) for c in cells}
    sumsq = {c: np.zeros((win_len, n_keys), dtype=float) for c in cells}
    counts = {c: np.zeros(win_len, dtype=np.int64) for c in cells}
    v_post_minus_pre_sums = {c: np.zeros(win_len, dtype=float) for c in cells}
    nodes_ref = {c: np.zeros(0, dtype=np.int64) for c in cells}
    for b, plan in enumerate(walk_batches):
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


def _make_walk_batch(
    nodes_by_cell: dict[str, np.ndarray],
    *,
    t0_bn_row: np.ndarray,
    win_len: int,
) -> _ComponentWalkBatch:
    """Build one walk batch; ``t0_u[i] = t0_bn_row[all_nodes[i]]``."""
    all_nodes = np.unique(np.concatenate([us for us in nodes_by_cell.values()]))
    return _ComponentWalkBatch(
        all_nodes=all_nodes,
        t0_u=np.asarray(t0_bn_row[all_nodes], dtype=np.int64),
        win_len=int(win_len),
        node_to_cell=_node_to_cell_map(nodes_by_cell),
        nodes_by_cell=nodes_by_cell,
    )


# ---------------------------------------------------------------------------
# Average bar components (cost-extent)
# ---------------------------------------------------------------------------


def _parse_syn_strength(
    tokens: list[str] | None,
    session,
) -> dict[tuple[int, int], float]:
    """Parse ``--syn-strength SRC:TAR=VALUE ...`` (``PAIR_SEP`` between types)."""
    if not tokens:
        return {}
    if session.backend.network is None:
        raise SystemExit("--syn-strength requires a network backend")
    names = list(session.backend.network.cell_names)
    name_to_i = {n: i for i, n in enumerate(names)}
    sep = training.PAIR_SEP
    out: dict[tuple[int, int], float] = {}
    for tok in tokens:
        if "=" not in tok:
            raise SystemExit(
                f"--syn-strength expected SRC{sep}TAR=VALUE, got {tok!r}"
            )
        left, val_s = tok.split("=", 1)
        if sep not in left:
            raise SystemExit(
                f"--syn-strength expected SRC{sep}TAR=VALUE, got {tok!r}"
            )
        src_name, tar_name = left.split(sep, 1)
        if not src_name or not tar_name or sep in tar_name:
            raise SystemExit(
                f"--syn-strength expected SRC{sep}TAR=VALUE, got {tok!r}"
            )
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


def _apply_syn_strength_cell(
    z: torch.Tensor,
    schema: list,
    session,
    edits: dict[tuple[int, int], float],
) -> torch.Tensor:
    """Return a copy of ``z`` with ``syn_strength_cell`` overrides applied."""
    if not edits:
        return z
    names = list(session.backend.network.cell_names)
    keys = session.backend.conn.pair_keys
    key_to_i = {k: i for i, k in enumerate(keys)}
    named = training.z_to_node_values(z, schema)
    if "syn_strength_cell" not in named:
        raise SystemExit("schema missing syn_strength_cell segment")
    arr = np.array(named["syn_strength_cell"], dtype=np.float64, copy=True)
    for (src_i, tar_i), val in edits.items():
        pair = (src_i, tar_i)
        if pair not in key_to_i:
            raise SystemExit(
                f"no type pair {names[src_i]!r} -> {names[tar_i]!r} in connectome"
            )
        pair_i = key_to_i[pair]
        arr[pair_i] = val
        _log(f"syn_strength_cell {names[src_i]} -> {names[tar_i]} = {val:g}")
    named["syn_strength_cell"] = arr
    return training.node_values_to_z(named, schema, dtype=z.dtype, device=z.device)


def _bar_meta(session, task: str):
    """One-shot ``(specs, grids)`` for a moving-bar task."""
    specs = bar_specs_for_session(session, task)
    pack = session.pack_for(task)
    grids = moving_bar_session_t0_grids(
        session, specs, pack.cost_extent, int(session.n_t),
        t_onset=int(pack.i_sti.shape[1] - pack.data.shape[1]),
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
        raise SystemExit("bar component walk requires at least one spec")
    pack = session.pack_for(task)
    if specs is None or grids is None:
        specs, grids = _bar_meta(session, task)
    name_to_bi = {s.name: i for i, s in enumerate(specs)}
    missing = [s for s in spec_names if s not in name_to_bi]
    if missing:
        raise SystemExit(f"spec(s) {missing} not in {[s.name for s in specs]}")
    bis = [name_to_bi[s] for s in spec_names]
    return pack, specs, grids, bis, pack.i_sti[bis], np.asarray(grids.t0_bn)


def _analyze_component_walk(
    session,
    *,
    p,
    cells: list[str],
    task: str,
    i_sti: torch.Tensor,
    walk_batches: list[_ComponentWalkBatch],
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
    """Shared spot/bar: ``_walk_component`` → finalize reports.

    * ``merge_batches=False`` (bar): ``reports[spec][cell]``; ``batch_specs`` are str.
    * ``merge_batches=True`` (spot): sum across batches → ``reports[cell]``.
    """
    if not walk_batches:
        raise SystemExit("component walk requires at least one batch")
    if len(before_t) != len(walk_batches) or len(batch_specs) != len(walk_batches):
        raise SystemExit("before_t/batch_specs length must match walk_batches")

    accum = _walk_component(
        session, p, i_sti, walk_batches, cells,
        t_start=time_window.walk_t_start, t_stop=time_window.walk_t_stop,
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
        win_len = walk_batches[0].win_len
        sums, sumsq, counts, v_post_minus_pre_sums, nodes_ref = _merge_walk_accum(
            accum, walk_batches, cells, win_len,
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
            raise SystemExit("non-merge component walk requires batch_specs as str")
        out[spec] = {}
        for cell in cells:
            out[spec][cell] = _one_report(
                cell=cell,
                spec=spec,
                before=int(before_t[b]),
                nodes=walk_batches[b].nodes_by_cell[cell],
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
    task: str,
    spec_names: list[str],
    time_window: TimeWindow,
    nodes_for_bi,
    mode: str,
    specs=None,
    grids=None,
    extra: dict[str, Any] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Bar prep: resolve i_sti/specs → ``_analyze_component_walk`` (no merge)."""
    pack, specs, grids, bis, i_sti, t0_bn = _resolve_bar_spec_i_sti(
        session, task, spec_names, specs=specs, grids=grids,
    )
    before_b: list[int] = []
    walk_batches: list[_ComponentWalkBatch] = []
    for bi, spec in zip(bis, spec_names):
        usets = nodes_for_bi(bi, spec, pack=pack, t0_bn=t0_bn)
        before = int(grids.before_t[spec])
        after = int(grids.after_t[spec])
        before_b.append(before)
        walk_batches.append(
            _make_walk_batch(usets, t0_bn_row=t0_bn[bi], win_len=before + after + 1),
        )
    return _analyze_component_walk(
        session,
        p=p,
        cells=cells,
        task=task,
        i_sti=i_sti,
        walk_batches=walk_batches,
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
    """One batched v walk over all requested specs; mean component per cell.

    Returns ``reports[spec][cell]``. v_post + component share ``_walk_component``.
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

    return _analyze_bar_walk(
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
# Average spot components (center-bin / stim-on-hex)
# ---------------------------------------------------------------------------


def _spot_session_layout(session_one, cells: list[str]):
    """Session-scoped center-bin layout for spot component walks."""
    pack = session_one.primary_readout
    C = session_one.backend.network
    if C is None:
        raise SystemExit("spot average requires a network backend")
    opts = dict((session_one.train_opts or {}).get(f"{pack.name}_stimulus_opts") or {})
    spot = spot_from_opts(C, stimulus_opts=opts)
    (
        batch_idx, node_idx, _radius, type_idx, _stim_u, _stim_v, _du, _dv, center_row,
    ) = spot_center_bin_layout(
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
    return pack, batch_idx, node_idx, type_idx, center_row, type_i


def analyze_spot_average(
    session_one,
    *,
    p,
    cells: list[str],
    task: str,
    time_window: TimeWindow,
) -> dict[str, dict[str, Any]]:
    """One batched v walk over spot stimulus rows; mean center-bin component."""
    if task not in training.SPOT_TASKS:
        raise SystemExit(f"unsupported task {task!r}")
    pack, batch_idx, node_idx, type_idx, center_row, type_i = _spot_session_layout(
        session_one, cells,
    )
    t_onset = int(pack.i_sti.shape[1] - pack.data.shape[1])

    i_sti = pack.i_sti if pack.i_sti.dim() == 3 else pack.i_sti.unsqueeze(0)
    B_all, T, _N = i_sti.shape
    t0_abs = np.zeros(_N, dtype=np.int64)
    win_len = int(time_window.stop) + 1 if time_window.kind == "t" else T

    walk_batches: list[_ComponentWalkBatch] = []
    i_sti_rows: list[int] = []
    for b in range(B_all):
        row_mask = center_row & (batch_idx == b)
        if not np.any(row_mask):
            continue
        usets: dict[str, np.ndarray] = {}
        for cell in cells:
            m = row_mask & (type_idx == type_i[cell])
            if np.any(m):
                usets[cell] = np.unique(node_idx[m])
        if not usets:
            continue
        walk_batches.append(
            _make_walk_batch(usets, t0_bn_row=t0_abs, win_len=win_len),
        )
        i_sti_rows.append(b)

    if not walk_batches:
        raise SystemExit("no center-bin nodes for requested cells in spot layout")

    data_by_contrast = resolve_spot_data_cubes(
        {contrast_for_task(pack.name): session_one},
    )
    contrast = contrast_for_task(pack.name)
    data_on = data_by_contrast.get(contrast) or {}

    def extra_for_cell(
        cell: str, v_post: np.ndarray, v_post_d: np.ndarray,
    ) -> dict[str, Any]:
        del v_post  # peak time from |v_post_d|; absolute series unused here
        extra: dict[str, Any] = {"data_peak_mV": None}
        if cell in data_on:
            peak_probe = _v_post_d_peak_t_rel(v_post_d, t_onset)
            data_cube = np.asarray(data_on[cell], dtype=float)
            if peak_probe < data_cube.shape[1]:
                extra["data_peak_mV"] = float(data_cube[CENTER_BIN, peak_probe])
        return extra

    def n_nodes_for_cell(cell: str) -> int:
        # Total center readouts across layout (matches prior semantics).
        return int(np.sum(center_row & (type_idx == type_i[cell])))

    n_b = len(walk_batches)
    return _analyze_component_walk(
        session_one,
        p=p,
        cells=cells,
        task=task,
        i_sti=i_sti[i_sti_rows],
        walk_batches=walk_batches,
        before_t=[t_onset] * n_b,
        batch_specs=[None] * n_b,
        time_window=time_window,
        mode="average",
        ti_mode="abs_minus_before",
        merge_batches=True,
        extra_for_cell=extra_for_cell,
        n_nodes_for_cell=n_nodes_for_cell,
    )


# ---------------------------------------------------------------------------
# Hex-mode bar (single hex; same walk as average)
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
    if not cols:
        raise SystemExit(f"no hex at x={at_x!r} y={at_y!r} within cost_extent={cost_extent}")
    if len(cols) > 1:
        raise SystemExit(f"multiple hexes at x={at_x!r} y={at_y!r}; pick a unique hex")
    col = cols[0]
    if cell not in C.cell_names:
        raise SystemExit(f"unknown cell {cell!r}")
    nodes = col2gt(C, int(col.u), int(col.v), cell).tolist()
    if not nodes:
        raise SystemExit(f"no {cell} node at hex ({at_x},{at_y})")
    return col, nodes


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
    """One batched v walk over specs at one hex; returns ``reports[spec][cell]``."""
    pack = session.pack_for(task)
    col, nodes = _nodes_at_hex(
        session, cell, at_x=at_x, at_y=at_y, cost_extent=pack.cost_extent,
    )
    if node is None:
        if len(nodes) > 1:
            raise SystemExit(f"multiple {cell} at ({at_x},{at_y}): {nodes}; pass --node")
        node = nodes[0]
    elif node not in nodes:
        raise SystemExit(f"node {node} not in {nodes}")
    node_arr = np.asarray([node], dtype=np.int64)
    usets = {cell: node_arr}

    def nodes_for_bi(bi, spec, *, pack, t0_bn):
        if int(t0_bn[bi, node]) < 0:
            raise SystemExit(f"no t0 for node {node} on spec {spec!r}")
        return usets

    return _analyze_bar_walk(
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
            "node": int(node),
            "hex": {"x": at_x, "y": at_y},
            "uv": {"u": int(col.u), "v": int(col.v)},
        },
    )



# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_filename(report: dict[str, Any]) -> str:
    parts = [report["cell"], report["task"], report.get("mode", "average")]
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
    return f"{r0['cell']}_{r0['task']}_overlay_{specs}_v.png"


def _component_figure(title: str, spec: _ComponentSpec):
    """Shared grid figure: rows = panel groups, cols = traces within a row."""
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
    for ri, (_group_ylabel, series) in enumerate(spec.plot_panels):
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
    spec = _component_spec(models.pop())

    fig, axes, save_figure = _component_figure(title, spec)
    colors = _plot_colors()
    linestyles = ("-", "--", "-.", ":")
    overlay = len(reports) > 1
    e_leak_mV = float(reports[0].get("params", {}).get("e_leak_mV", 0.0))
    globs = reports[0].get("globals") or {}
    delta_ms = float(globs.get("delta_ms", training.DELTA_MS))
    row_curves: dict[int, list[np.ndarray]] = {ri: [] for ri in spec.row_shared_ylim}
    tc = _plot_trace_colors(colors, spec)

    for ri, (group_ylabel, series) in enumerate(spec.plot_panels):
        for ci, (key, label) in enumerate(series):
            ax = axes[ri, ci]
            color = tc[label]
            show_legend = overlay and ri == 0 and ci == 0
            for si, rep in enumerate(reports):
                ls = linestyles[si % len(linestyles)] if overlay else "-"
                xs = np.asarray([s["t"] for s in rep["steps"]], dtype=float) * delta_ms
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
                ax.plot(
                    xs, y,
                    label=str(rep["spec"]) if show_legend else "_nolegend_",
                    color=color,
                    linestyle=ls,
                    linewidth=1.4,
                )
            e_note = _g_e_note(label, e_leak_mV=e_leak_mV, globs=globs)
            if e_note is not None:
                ax.set_title(e_note, fontsize=8)
            _style_component_ax(
                ax, _trace_ylabel(group_ylabel, label),
                legend_fontsize=6 if overlay else 7,
                legend_ncol=1,
                show_legend=show_legend,
            )
    _apply_shared_row_ylim(axes, row_curves, spec)
    _finish_component_figure_layout(fig, title, colors, spec)
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
    spec_list = ",".join(str(r["spec"]) for r in reports)
    title = (
        f"{r0['cell']}  {r0['task']}  overlay=[{spec_list}]"
        f"  mode={r0.get('mode')}  n={r0.get('n_nodes')}"
    )
    _plot_component_reports(reports, out_path, title=title)


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
        f"v_post_d_peak={report['v_post_d_peak_mV']:+.4f} mV "
        f"v_post_d_polarity={report['v_post_d_polarity']}  "
        f"v_post_d_peak_t={report['v_post_d_peak_t']}  "
        f"before_t={report.get('before_t')}  "
        f"peak_drive={report.get('peak_drive')}  "
        f"{kind}={tw[0]}:{tw[1]}"
    )
    print("trained:", "  ".join(f"{k}={v:.6g}" for k, v in report["params"].items()))

    if model == "hp_lp":
        print(
            f"\n{x_key}  n  v_post  v_pre_d  v_post_minus_pre  i_sti "
            "v_in  v_in_exc -v_in_inh  dv_leak  dv_hp"
        )
        for s in report["steps"]:
            print(
                f"{s[x_key]:4d} {s.get('n_nodes', 1):3d} {s['v_post_mV']:+8.4f} "
                f"{s['v_pre_d_mV']:+8.4f} {s['v_post_minus_pre_mV']:+8.4f} "
                f"{s['i_sti']:+6.3f} {s['v_in']:+6.3f} {s['v_in_exc']:+7.3f} "
                f"{s['v_in_inh']:+7.3f} {s['dv_leak']:+8.4f} {s['dv_hp']:+7.4f}"
            )
        ps = report.get("peak_step")
        if ps is not None:
            print(f"\nHP/LP terms at peak {x_key}={ps[x_key]}:")
            for name, val in [
                ("v_in", ps["v_in"]), ("v_in_exc", ps["v_in_exc"]), ("-v_in_inh", ps["v_in_inh"]),
                ("a", ps["a"]), ("X", ps["X"]), ("X-a", ps["X_minus_a"]),
                ("dv_leak", ps["dv_leak"]), ("dv_hp", ps["dv_hp"]),
            ]:
                print(f"  {name:8s} {val:+9.4f}")
        return

    print(
        f"\n{x_key}  n  v_post  v_pre_d  v_post_minus_pre  i_sti "
        "g_inh  g_Ih_off  g_exc  num_inh  num_exc"
    )
    for s in report["steps"]:
        print(
            f"{s[x_key]:4d} {s.get('n_nodes', 1):3d} {s['v_post_mV']:+8.4f} "
            f"{s['v_pre_d_mV']:+8.4f} {s['v_post_minus_pre_mV']:+8.4f} "
            f"{s['i_sti']:5.1f} {s['g_inh_nS']:.4f} {s['g_Ih_off_nS']:.4f} "
            f"{s['g_exc_nS']:.4f} {s['num_inh']:+8.2f} {s['num_exc']:+8.2f}"
        )

    ps = report.get("peak_step")
    if ps is not None:
        num = (
            ps["num_exc"] + ps["num_inh"] + ps["num_leak"]
            + ps["num_ihon"] + ps["num_ihoff"] + ps["num_cdt"] + ps["i_sti"]
        )
        print(f"\nNumerator at peak {x_key}={ps[x_key]} (num={num:.2f}):")
        for name, val in [
            ("i_cdt", ps["num_cdt"]), ("i_exc", ps["num_exc"]), ("i_inh", ps["num_inh"]),
            ("i_leak", ps["num_leak"]), ("i_h_on", ps["num_ihon"]), ("i_h_off", ps["num_ihoff"]),
            ("i_sti", ps["i_sti"]),
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
        sps, bps = s.get("peak_step"), b.get("peak_step")
        if not (sps and bps):
            continue
        model = s.get("model", "borst")
        if model == "hp_lp":
            print(
                f"       spot@peak: v_in_exc={sps['v_in_exc']:+.4f} -v_in_inh={sps['v_in_inh']:+.4f} "
                f"dv_hp={sps['dv_hp']:+.4f} dv_leak={sps['dv_leak']:+.4f} "
                f"v_pre_d={sps['v_pre_d_mV']:+.3f}"
            )
            print(
                f"       bar @peak: v_in_exc={bps['v_in_exc']:+.4f} -v_in_inh={bps['v_in_inh']:+.4f} "
                f"dv_hp={bps['dv_hp']:+.4f} dv_leak={bps['dv_leak']:+.4f} "
                f"v_pre_d={bps['v_pre_d_mV']:+.3f}"
            )
        else:
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
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_shared_cli(ap, default_run=DEFAULT_RUN_PATH)
    ap.add_argument(
        "--trace-only",
        action="store_true",
        help="print response-curve summaries (full forward; no component walk)",
    )
    ap.add_argument(
        "--values",
        action="store_true",
        help="with --trace-only: print full analysis-window trace arrays",
    )
    ap.add_argument(
        "--pre-ms",
        type=float,
        default=None,
        help="with --trace-only: override spot pre-stimulus baseline in ms",
    )
    ap.add_argument(
        "--response-ms",
        type=float,
        default=None,
        help="with --trace-only: override spot post-onset response window in ms "
        f"(default: {training.RESPONSE_MS:g})",
    )
    ap.add_argument(
        "--t-max-ms",
        type=float,
        default=None,
        help="with --trace-only --values: only print first N ms of each trace",
    )
    ap.add_argument("--node", type=int, default=None, help="hex-mode node index")
    t_group = ap.add_mutually_exclusive_group()
    t_group.add_argument(
        "--t-rel",
        default=None,
        metavar="START:STOP",
        help=(
            "window relative to |v_post_d| peak (default: "
            f"{training.T_REL_START}:{training.T_REL_STOP})"
        ),
    )
    t_group.add_argument(
        "--t",
        default=None,
        metavar="START:STOP",
        help="absolute aligned t window (spot: abs; bar: vs t0); exclusive with --t-rel",
    )
    ap.add_argument(
        "--plot",
        type=parse_bool,
        default=True,
        metavar="true|false",
        help="save component time-series PNGs under {run}/cell_dynamics/ (default: true)",
    )
    ap.add_argument(
        "--syn-strength",
        nargs="+",
        default=None,
        metavar="SRC:TAR=VALUE",
        help="override syn_strength_cell; space-separated SRC:TAR=VALUE tokens",
    )
    ap.add_argument("--json", action="store_true", help="print JSON to stdout")
    args = ap.parse_args()
    if not args.run:
        args.run = [DEFAULT_RUN_PATH]
    cli = parse_shared_cli(args)

    if args.trace_only:
        if args.node is not None:
            raise SystemExit("--node is component/hex only; omit with --trace-only")
        if args.t_rel is not None or args.t is not None:
            raise SystemExit("--t-rel/--t are component only; omit with --trace-only")
        if args.syn_strength is not None:
            raise SystemExit("--syn-strength is component only; omit with --trace-only")
        if args.json:
            raise SystemExit("--json is component only; omit with --trace-only")
        if (cli.x_list is None) ^ (cli.y_list is None):
            raise SystemExit("pass both --x and --y, or neither")
        _run_trace_only(args, cli)
        return

    if args.values or args.pre_ms is not None or args.response_ms is not None or args.t_max_ms is not None:
        raise SystemExit(
            "--values/--pre-ms/--response-ms/--t-max-ms require --trace-only"
        )

    hex_mode = False
    if cli.x_list is not None and cli.y_list is not None:
        if len(cli.x_list) != 1 or len(cli.y_list) != 1:
            raise SystemExit(
                "hex mode needs exactly one --x and one --y; "
                "omit both for cost-extent averages"
            )
        hex_mode = True
        if any(t in training.SPOT_TASKS for t in cli.tasks):
            raise SystemExit("hex mode is moving_bar-only; omit --x/--y for spot")
        if len(cli.cells) != 1:
            raise SystemExit("hex mode supports one cell")
    elif cli.x_list is not None or cli.y_list is not None:
        raise SystemExit("pass both --x and --y for hex mode, or neither for averages")

    if args.t is not None:
        t_lo, t_hi = _parse_t_range(args.t, flag="--t")
        time_window = TimeWindow(kind="t", start=t_lo, stop=t_hi)
    else:
        raw = (
            args.t_rel
            if args.t_rel is not None
            else f"{training.T_REL_START}:{training.T_REL_STOP}"
        )
        t_lo, t_hi = _parse_t_range(raw, flag="--t-rel")
        time_window = TimeWindow(kind="t_rel", start=t_lo, stop=t_hi)

    for run_i, run_arg in enumerate(args.run):
        run_dir = plot_trained.resolve_run_dir(run_arg)
        _log(f"load_best {run_dir} ...")
        session, z, best_i, best_cost = plot_trained.load_best(run_dir)
        schema = list(session.schema)
        syn_strength_edits = _parse_syn_strength(args.syn_strength, session)
        z_t = torch.tensor(np.asarray(z, dtype=np.float64), dtype=torch.float64, device=session.device)
        z_t = _apply_syn_strength_cell(z_t, schema, session, syn_strength_edits)
        p = training.assign_params(z_t, schema, session.backend)

        spot_session_cache: dict[str, object] = {}
        bar_meta_cache: dict[str, tuple] = {}
        spot_by_cell: dict[str, dict[str, Any]] = {}
        bar_by_cell: dict[str, dict[str, Any]] = {}
        all_reports: list[dict[str, Any]] = []

        if not args.json:
            _log(f"== RUN {run_i}: {run_dir} ==")
            _log(
                f"best_i={best_i}  best_cost={best_cost:.6g}  "
                f"mode={'hex' if hex_mode else 'average'}  "
                f"{time_window.kind}={time_window.start}:{time_window.stop}"
            )

        for task in cli.tasks:
            if task in training.SPOT_TASKS:
                if task not in spot_session_cache:
                    spot_session_cache[task] = plot_trained.session_for_task(
                        session, task,
                    )
                session_one = spot_session_cache[task]
                _log(f"component walk {task} (spot; batched) ...")
                reports = analyze_spot_average(
                    session_one,
                    p=p,
                    cells=cli.cells,
                    task=task,
                    time_window=time_window,
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
                        f"component walk {task} specs={specs_ordered} "
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
                        f"component walk {task} specs={specs_ordered} "
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
