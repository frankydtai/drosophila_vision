"""
Analyze one cell's time curve without saving CSV.

Spot: center-bin impulse + RF (same as spot plots).
Moving bar: t_first_sti-aligned window for one spec; with ``--x`` / ``--y`` slice
lists, emits the same overlay traces as ``model_all_bar`` (per-(x,y) + total).

Examples:
  temporal_filtering/.venv/bin/python temporal_filtering/SimulationCode/test/analyze_cell_curve.py \\
    --run /abs/path/to/run --cell L4 --target spot_dark --trace-kind vm --x 2 --y 1

  temporal_filtering/.venv/bin/python temporal_filtering/SimulationCode/test/analyze_cell_curve.py \\
    --run /abs/path/to/run --cell Mi4 --target moving_bar_bright --spec right_bright_w1 \\
    --trace-kind model --x 1,1,0 --y 0.5,-2
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch


def _add_simulation_code_to_syspath():
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


def _shape_label(arr: np.ndarray, *, before_steps: int | None = None) -> str:
    s = _summarize(arr)
    pos = s.max > 0.05 * max(abs(s.max), abs(s.min), 1e-9)
    neg = s.min < -0.05 * max(abs(s.max), abs(s.min), 1e-9)
    if pos and neg:
        order = "+ then -" if s.peak_idx < s.trough_idx else "- then +"
        return f"biphasic ({order})"
    if pos:
        return "monophasic positive"
    if neg:
        return "monophasic negative"
    return "flat/near-zero"


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


def _extract_spot_bundle(session, z, *, target: str, trace_kind: str, x_list, y_list):
    import plot_trained
    from plot import spot as spot_plot

    one = plot_trained._session_for_target(session, target)
    bundle = spot_plot._prepare_network_spot_bundle(
        one,
        z,
        all_cells=True,
        at_x_list=x_list,
        at_y_list=y_list,
        trace_kind=trace_kind,
        save_trace_csv_dir=None,
    )
    ref_on, _ = spot_plot._resolve_spot_ref_cubes(one, None, None, None)
    return bundle, ref_on


def _extract_moving_bar_bundle(session, z, *, target: str, trace_kind: str, x_list, y_list):
    import plot_trained
    from plot import moving_bar as moving_bar_plot

    one = plot_trained._session_for_target(session, target)
    return moving_bar_plot.moving_bar_trace_bundle(
        one,
        z,
        target,
        at_x_list=x_list,
        at_y_list=y_list,
        trace_kind=trace_kind,
        save_trace_csv_dir=None,
    )


def _extract_spot_cell_curves(bundle, ref_on, *, cell: str, slice_label: str | None = None):
    from plot import spot as spot_plot

    center = spot_plot.CENTER_BIN
    cell_on = next(c for c in bundle.cells_on if c["name"] == cell)

    imp_total, rf_total = spot_plot._scale_curve(cell_on["cube"], center)
    imp_ref, rf_ref = spot_plot._scale_curve(ref_on[cell], center)

    out: dict[str, np.ndarray] = {
        "time_total_model": np.asarray(imp_total, dtype=float),
        "time_total_ref": np.asarray(imp_ref, dtype=float),
        "rf_total_model": np.asarray(rf_total, dtype=float),
        "rf_total_ref": np.asarray(rf_ref, dtype=float),
    }

    if slice_label and bundle.slice_overlay and slice_label in bundle.slice_overlay:
        cubes = bundle.slice_overlay[slice_label]
        if cell in cubes:
            imp_slice, rf_slice = spot_plot._scale_curve(cubes[cell], center)
            out[f"time_slice_model:{slice_label}"] = np.asarray(imp_slice, dtype=float)
            out[f"rf_slice_model:{slice_label}"] = np.asarray(rf_slice, dtype=float)
    return out


def _moving_bar_slice_labels(bundle) -> list[str]:
    if bundle.slice_overlay is None:
        return []
    return list(bundle.slice_overlay.keys())


def _extract_moving_bar_cell_curves(bundle, *, cell: str, spec: str) -> dict[str, np.ndarray]:
    """Per-spec traces matching one ``model_all_bar`` panel (slices + total)."""
    key = (cell, spec)
    model_mean = bundle.traces.model_mean
    if key not in model_mean:
        avail = sorted(s for c, s in model_mean if c == cell)
        raise SystemExit(f"spec {spec!r} not found for cell {cell!r}; available: {avail}")

    out: dict[str, np.ndarray] = {}
    for label in _moving_bar_slice_labels(bundle):
        wt = bundle.slice_overlay[label]
        if key in wt.model_mean:
            out[label] = np.asarray(wt.model_mean[key], dtype=float)
    out["total"] = np.asarray(model_mean[key], dtype=float)
    return out


def _print_curve(name: str, arr: np.ndarray, *, before_steps: int | None, print_values: bool):
    s = _summarize(arr)
    shape = _shape_label(arr, before_steps=before_steps)
    print(
        f"{name}: n={s.n} max={s.max:.6g} min={s.min:.6g} final={s.final:.6g} "
        f"peak_idx={s.peak_idx} trough_idx={s.trough_idx} sign_changes={s.sign_changes}  shape={shape}"
    )
    if print_values:
        print(f"  values: {np.round(arr, 4).tolist()}")


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
        "--spec",
        default=None,
        help="moving_bar stimulus spec, e.g. right_bright_w1 (required for moving_bar_*)",
    )
    ap.add_argument(
        "--trace-kind",
        default="vm",
        choices=("vm", "model"),
        help="curve kind: vm (Vm-Vm_ref) or model (scaled conductance output)",
    )
    ap.add_argument(
        "--x",
        default=None,
        metavar="X,...",
        help="comma-separated x slice(s); with --y, one trace per (x,y) pair",
    )
    ap.add_argument(
        "--y",
        default=None,
        metavar="Y,...",
        help="comma-separated y slice(s); with --x, one trace per (x,y) pair",
    )
    ap.add_argument(
        "--values",
        nargs="?",
        const=True,
        default=False,
        type=lambda v: v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes"),
        metavar="BOOL",
        help="print full trace arrays (default false)",
    )
    args = ap.parse_args()

    _add_simulation_code_to_syspath()
    from plot.utils import parse_axis_slice_list, slice_xy_label

    x_list = parse_axis_slice_list(args.x)
    y_list = parse_axis_slice_list(args.y)
    if (x_list is None) ^ (y_list is None):
        if args.target.startswith("spot_"):
            raise SystemExit("spot targets require both --x and --y or neither")
    slice_label = None
    if x_list is not None and y_list is not None and len(x_list) == 1 and len(y_list) == 1:
        slice_label = slice_xy_label(x_list[0], y_list[0])

    for i, run_dir in enumerate(args.run):
        run_dir = os.path.abspath(run_dir)
        if not os.path.isdir(run_dir):
            raise SystemExit(f"run dir not found: {run_dir}")

        session, z, best_i, best_cost = _load_best(run_dir)
        before_steps = None

        if args.target.startswith("spot_"):
            bundle, ref_on = _extract_spot_bundle(
                session,
                z,
                target=args.target,
                trace_kind=args.trace_kind,
                x_list=x_list,
                y_list=y_list,
            )
            curves = _extract_spot_cell_curves(
                bundle, ref_on, cell=args.cell, slice_label=slice_label,
            )
        elif args.target.startswith("moving_bar_"):
            if args.spec is None:
                raise SystemExit("moving_bar targets require --spec (e.g. right_bright_w1)")
            bundle = _extract_moving_bar_bundle(
                session,
                z,
                target=args.target,
                trace_kind=args.trace_kind,
                x_list=x_list,
                y_list=y_list,
            )
            curves = _extract_moving_bar_cell_curves(bundle, cell=args.cell, spec=args.spec)
            before_steps = bundle.traces.before_steps.get(args.spec)
        else:
            raise SystemExit(f"unsupported target {args.target!r}; expected spot_* or moving_bar_*")

        print("")
        print(f"== RUN {i}: {run_dir} ==")
        hdr = (
            f"best_i={best_i}  best_cost={best_cost:.6g}  cell={args.cell}  "
            f"target={args.target}  trace_kind={args.trace_kind}"
        )
        if args.spec is not None:
            hdr += f"  spec={args.spec}"
        print(hdr)
        print(f"maxtime={bundle.maxtime}  deltat_ms={getattr(session, 'deltat_ms', 'NA')}")
        if x_list is not None or y_list is not None:
            print(f"slice_x={x_list}  slice_y={y_list}")
        if before_steps is not None:
            print(f"cost_window_start_idx={before_steps}")

        if args.target.startswith("moving_bar_"):
            slice_names = [k for k in curves if k != "total"]
            order = slice_names + ["total"]
            print(f"traces ({len(order)}): {', '.join(order)}")
            for name in order:
                _print_curve(name, curves[name], before_steps=before_steps, print_values=args.values)
        else:
            for key, arr in curves.items():
                _print_curve(key, arr, before_steps=before_steps, print_values=args.values)


if __name__ == "__main__":
    main()
