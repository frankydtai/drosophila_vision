#!/usr/bin/env python
"""
Analyze one cell's spot curve without saving CSV.

This script reconstructs the same curves used by spot plots:
- time curve: center-bin impulse (length = maxtime)
- RF curve: spatial profile at time of peak response (length = 45 after rebin+blur)

Examples:
  temporal_filtering/.venv/bin/python temporal_filtering/SimulationCode/test/analyze_cell_curve.py \\
    --run /abs/path/to/run --cell L4 --target spot_dark --trace-kind vm --x 2 --y 1

  temporal_filtering/.venv/bin/python temporal_filtering/SimulationCode/test/analyze_cell_curve.py \\
    --run /abs/path/to/old --run /abs/path/to/new --cell L4 --target spot_dark --trace-kind vm --x 2 --y 1
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch


def _add_simulation_code_to_syspath():
    # This file lives at ``temporal_filtering/SimulationCode/test``.
    # Add ``temporal_filtering/SimulationCode`` so imports like ``plot_trained`` work.
    sim_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, sim_dir)


@dataclass
class CurveSummary:
    n: int
    max: float
    min: float
    final: float
    peak_idx: int
    trough_idx: int
    sign_changes: int


def _summarize(arr: np.ndarray) -> CurveSummary:
    arr = np.asarray(arr, dtype=float)
    signs = np.sign(arr)
    sign_changes = int(np.sum(signs[1:] * signs[:-1] < 0))
    return CurveSummary(
        n=int(arr.size),
        max=float(np.max(arr)),
        min=float(np.min(arr)),
        final=float(arr[-1]),
        peak_idx=int(np.argmax(arr)),
        trough_idx=int(np.argmin(arr)),
        sign_changes=sign_changes,
    )


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


def _extract_spot_bundle(session, z, *, target: str, trace_kind: str, x: list[int] | None, y: list[int] | None):
    import plot_trained
    from plot import spot as spot_plot

    one = plot_trained._session_for_target(session, target)
    bundle = spot_plot._prepare_network_spot_bundle(
        one,
        z,
        all_cells=True,
        group_list=None,
        at_x_list=x,
        at_y_list=y,
        trace_kind=trace_kind,
        save_trace_csv_dir=None,
    )
    ref_on, _ = spot_plot._resolve_spot_ref_cubes(one, None, None, None)
    return bundle, ref_on


def _extract_moving_bar_bundle(session, z, *, target: str, trace_kind: str, x: list[int] | None, y: list[int] | None):
    """Rebuild moving-bar traces (model vs data) matching plot_trained."""
    from plot import moving_bar as moving_bar_plot

    bundle = moving_bar_plot.moving_bar_trace_bundle(
        session,
        z,
        target,
        at_x_list=x,
        at_y_list=y,
        trace_kind=trace_kind,
        save_trace_csv_dir=None,
    )
    return bundle


def _extract_cell_curves(bundle, ref_on, *, cell: str, slice_label: str | None = None):
    from plot import spot as spot_plot

    center = spot_plot.CENTER_BIN
    cell_on = next(c for c in bundle.cells_on if c["name"] == cell)

    imp_total, rf_total = spot_plot._scale_curve(cell_on["cube"], center)
    imp_ref, rf_ref = spot_plot._scale_curve(ref_on[cell], center)

    imp_slice = rf_slice = None
    if slice_label and bundle.slice_overlay and slice_label in bundle.slice_overlay:
        cubes = bundle.slice_overlay[slice_label]
        if cell in cubes:
            imp_slice, rf_slice = spot_plot._scale_curve(cubes[cell], center)

    out = {
        "time_total_model": np.asarray(imp_total, dtype=float),
        "time_total_ref": np.asarray(imp_ref, dtype=float),
        "rf_total_model": np.asarray(rf_total, dtype=float),
        "rf_total_ref": np.asarray(rf_ref, dtype=float),
        "time_slice_model": None if imp_slice is None else np.asarray(imp_slice, dtype=float),
        "rf_slice_model": None if rf_slice is None else np.asarray(rf_slice, dtype=float),
    }
    return out


def _extract_moving_bar_cell_curves(bundle, *, cell: str):
    """Aggregate moving-bar traces for one cell across specs.

    We average over all stimulus specs (keys in ``bundle.traces.model_mean``)
    that match the requested cell. This roughly matches the per-cell time
    traces in moving-bar plots, but without per-direction breakdown.
    """
    traces = bundle.traces
    model_mean = traces.model_mean
    data_mean = traces.model_sem  # we do not have raw data traces; use SEM only when present

    # Keys are (cell_type, spec_name); select those for this cell.
    model_traces = [v for (ctype, _), v in model_mean.items() if ctype == cell]
    if not model_traces:
        raise SystemExit(f'no moving_bar traces found for cell {cell!r}')
    # All windows for this cell have the same length by construction.
    model_arr = np.vstack([np.asarray(v, dtype=float) for v in model_traces])
    imp_total = model_arr.mean(axis=0)

    # moving_bar_plot compares against experimental data when available;
    # here we only summarize model traces, so we leave "ref" empty.
    out = {
        "time_total_model": np.asarray(imp_total, dtype=float),
        "time_total_ref": None,
        "rf_total_model": None,
        "rf_total_ref": None,
        "time_slice_model": None,
        "rf_slice_model": None,
    }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run",
        action="append",
        required=True,
        help="absolute run directory (repeatable for comparison)",
    )
    ap.add_argument("--cell", required=True, help="cell type name, e.g. L4, T4a, Mi1")
    ap.add_argument(
        "--target",
        default="spot_dark",
        help="plot target to analyze (spot_* or moving_bar_*)",
    )
    ap.add_argument(
        "--trace-kind",
        default="vm",
        choices=("vm", "model"),
        help="curve kind: vm (Vm-Vm_ref) or model (raw model output)",
    )
    ap.add_argument("--x", type=int, default=None, help="slice x (requires --y)")
    ap.add_argument("--y", type=int, default=None, help="slice y (requires --x)")
    args = ap.parse_args()

    _add_simulation_code_to_syspath()

    x_list = y_list = None
    slice_label = None
    if (args.x is None) ^ (args.y is None):
        raise SystemExit("must pass both --x and --y or neither")
    if args.x is not None and args.y is not None:
        x_list, y_list = [int(args.x)], [int(args.y)]
        slice_label = f"({int(args.x)},{int(args.y)})"

    for i, run_dir in enumerate(args.run):
        run_dir = os.path.abspath(run_dir)
        if not os.path.isdir(run_dir):
            raise SystemExit(f"run dir not found: {run_dir}")

        session, z, best_i, best_cost = _load_best(run_dir)
        if args.target.startswith("spot_"):
            bundle, ref_on = _extract_spot_bundle(
                session,
                z,
                target=args.target,
                trace_kind=args.trace_kind,
                x=x_list,
                y=y_list,
            )
            curves = _extract_cell_curves(bundle, ref_on, cell=args.cell, slice_label=slice_label)
        elif args.target.startswith("moving_bar_"):
            if x_list is None and y_list is None:
                # For moving_bar, at_x/at_y control which cost columns we align to.
                # Require explicit slice to match the moving_bar plots.
                raise SystemExit("moving_bar targets require --x and --y to select stimulus columns")
            bundle = _extract_moving_bar_bundle(
                session,
                z,
                target=args.target,
                trace_kind=args.trace_kind,
                x=x_list,
                y=y_list,
            )
            curves = _extract_moving_bar_cell_curves(bundle, cell=args.cell)
        else:
            raise SystemExit(f"unsupported target {args.target!r}; expected spot_* or moving_bar_*")

        print("")
        print(f"== RUN {i}: {run_dir} ==")
        print(
            f"best_i={best_i}  best_cost={best_cost:.6g}  cell={args.cell}  target={args.target}  trace_kind={args.trace_kind}"
        )
        print(f"maxtime={bundle.maxtime}  deltat_ms={getattr(session, 'deltat_ms', 'NA')}")

        for key in (
            "time_total_model",
            "time_total_ref",
            "time_slice_model",
            "rf_total_model",
            "rf_total_ref",
            "rf_slice_model",
        ):
            arr = curves.get(key)
            if arr is None:
                continue
            s = _summarize(arr)
            print(
                f"{key}: n={s.n} max={s.max:.6g} min={s.min:.6g} final={s.final:.6g} peak_idx={s.peak_idx} trough_idx={s.trough_idx} sign_changes={s.sign_changes}"
            )


if __name__ == "__main__":
    main()

