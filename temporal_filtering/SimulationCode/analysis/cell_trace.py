"""
Analyze cell time curves (spot + moving bar) without saving CSV.

Speed / agent contract
----------------------
``load_best`` + each target's forward are expensive. This script does, per
``--run``:

  * one ``load_best``
  * one spot forward per distinct ``spot_*`` target
  * one moving-bar forward per distinct ``moving_bar_*`` target

All ``--cell`` / ``--spec`` values are read from those cached bundles.
**Do not** re-invoke this CLI once per cell or once per spec. Pass comma
lists in a single process.

Spot
----
Center-bin impulse + RF (same scaling as spot plots) via
``plot_trained.spot_bundle_fns`` → ``network_spot_trace_bundle`` /
``borst_spot_trace_bundle``.

Moving bar
----------
``t_first_sti``-aligned window; optional ``--x`` / ``--y`` overlays match
``model_all_bar``. Summary stats and shape labels for bar traces use the
**post-onset** segment (``idx >= cost_window_start_idx``); pre-stim extrema
are ignored.

Examples
--------
  # from SimulationCode/ (module form; do not run as a loose file path)
  cd temporal_filtering/SimulationCode
  ../.venv/bin/python -m analysis.cell_trace \\
    --run /abs/path/to/run --cell Mi4,Mi9 \\
    --target spot_bright,moving_bar_bright --trace-kind vm

  ../.venv/bin/python -m analysis.cell_trace \\
    --run /abs/path/to/run --cell L4 --target spot_dark \\
    --trace-kind vm --x 2 --y 1

  ../.venv/bin/python -m analysis.cell_trace \\
    --run /abs/path/to/run --cell Mi4,Mi9 \\
    --target moving_bar_bright --spec right_bright_w1,left_bright_w1 \\
    --trace-kind vm
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

import FiveCol_MedSim_Pytorch as fc
import plot_trained
from plot import moving_bar as moving_bar_plot
from plot import spot as spot_plot
from plot.utils import parse_axis_slice_list, slice_xy_label
from train import parse_comma_list, parse_target_list


@dataclass(frozen=True)
class SharedCli:
    """Parsed shared CLI for ``cell_trace`` / ``cell_dynamics``."""

    cells: list[str]
    targets: list[str]
    specs_req: list[str] | None
    x_list: list | None
    y_list: list | None
    slice_label: str | None


def add_shared_cli(ap: argparse.ArgumentParser) -> None:
    """Register ``--run/--cell/--target/--spec/--trace-kind/--x/--y`` (shared)."""
    ap.add_argument(
        "--run",
        action="append",
        required=True,
        help="run directory (absolute, or relative to PARAMETER_DIR via plot_trained.resolve_run_dir)",
    )
    ap.add_argument(
        "--cell",
        required=True,
        metavar="CELL,...",
        help="comma-separated cell types, e.g. Mi4,Mi9 (analyzed from shared bundles)",
    )
    ap.add_argument(
        "--target",
        default="spot_dark",
        metavar="TARGET,...",
        help="comma-separated targets (spot_* / moving_bar_* or TARGET_ALIASES); "
        "one forward per distinct target per run",
    )
    ap.add_argument(
        "--spec",
        default=None,
        metavar="SPEC,...",
        help="comma-separated moving_bar stimulus specs; omit = all specs on the "
        "bundle for each cell (still one bar forward)",
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
        help="optional comma-separated x slice(s); omit unless overlay needed",
    )
    ap.add_argument(
        "--y",
        default=None,
        metavar="Y,...",
        help="optional comma-separated y slice(s); omit unless overlay needed",
    )


def parse_shared_cli(args: argparse.Namespace) -> SharedCli:
    cells = parse_comma_list(args.cell)
    if not cells:
        raise SystemExit("--cell is required")
    targets = parse_target_list(args.target)
    if not targets:
        raise SystemExit("--target is required")
    specs_req = parse_comma_list(args.spec) if args.spec is not None else None
    for t in targets:
        if t not in fc.SPOT_TARGETS and t not in fc.MOVING_BAR_TARGETS:
            raise SystemExit(
                f"unsupported target {t!r}; expected spot_* or moving_bar_* "
                f"(after TARGET_ALIASES expansion)"
            )
    x_list = parse_axis_slice_list(args.x)
    y_list = parse_axis_slice_list(args.y)
    if (x_list is None) ^ (y_list is None):
        if any(t in fc.SPOT_TARGETS for t in targets):
            raise SystemExit("spot targets require both --x and --y or neither")
    slice_label = None
    if x_list is not None and y_list is not None and len(x_list) == 1 and len(y_list) == 1:
        slice_label = slice_xy_label(x_list[0], y_list[0])
    return SharedCli(
        cells=cells,
        targets=targets,
        specs_req=specs_req,
        x_list=x_list,
        y_list=y_list,
        slice_label=slice_label,
    )


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
    arr = np.asarray(arr, dtype=float)
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


def _shape_label(arr: np.ndarray) -> str:
    s = _summarize(arr)
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


def extract_spot_bundle(session, z, *, target: str, trace_kind: str, x_list, y_list):
    """One spot forward via ``plot_trained.spot_bundle_fns``.

    Returns ``(session_one, bundle, ref_on)``.
    """
    one = plot_trained.session_for_target(session, target)
    make_bundle, _, _ = plot_trained.spot_bundle_fns(one)
    bundle = make_bundle(
        one,
        z,
        at_x_list=x_list,
        at_y_list=y_list,
        trace_kind=trace_kind,
        save_trace_csv_dir=None,
    )
    ref_on, _ = spot_plot.resolve_spot_ref_cubes(one, None, None, None)
    return one, bundle, ref_on


def extract_moving_bar_bundle(session, z, *, target: str, trace_kind: str, x_list, y_list):
    """One moving-bar forward; all cells + all specs live on the returned bundle."""
    one = plot_trained.session_for_target(session, target)
    return moving_bar_plot.moving_bar_trace_bundle(
        one,
        z,
        target,
        at_x_list=x_list,
        at_y_list=y_list,
        trace_kind=trace_kind,
        save_trace_csv_dir=None,
    )


def extract_spot_cell_curves(bundle, ref_on, *, cell: str, slice_label: str | None = None):
    center = spot_plot.CENTER_BIN
    cell_on = next((c for c in bundle.cells_on if c["name"] == cell), None)
    if cell_on is None:
        avail = sorted(c["name"] for c in bundle.cells_on)
        raise SystemExit(f"cell {cell!r} not in spot bundle; available: {avail}")
    if cell not in ref_on:
        raise SystemExit(f"cell {cell!r} not in spot ref cubes; keys={sorted(ref_on)}")

    imp_total, rf_total = spot_plot.scale_curve(cell_on["cube"], center)
    imp_ref, rf_ref = spot_plot.scale_curve(ref_on[cell], center)

    out: dict[str, np.ndarray] = {
        "time_total_model": np.asarray(imp_total, dtype=float),
        "time_total_ref": np.asarray(imp_ref, dtype=float),
        "rf_total_model": np.asarray(rf_total, dtype=float),
        "rf_total_ref": np.asarray(rf_ref, dtype=float),
    }

    if slice_label and bundle.slice_overlay and slice_label in bundle.slice_overlay:
        cubes = bundle.slice_overlay[slice_label]
        if cell in cubes:
            imp_slice, rf_slice = spot_plot.scale_curve(cubes[cell], center)
            out[f"time_slice_model:{slice_label}"] = np.asarray(imp_slice, dtype=float)
            out[f"rf_slice_model:{slice_label}"] = np.asarray(rf_slice, dtype=float)
    return out


def _moving_bar_slice_labels(bundle) -> list[str]:
    if bundle.slice_overlay is None:
        return []
    return list(bundle.slice_overlay.keys())


def extract_moving_bar_cell_curves(bundle, *, cell: str, spec: str) -> dict[str, np.ndarray]:
    """Per-spec traces matching one ``model_all_bar`` panel (slices + total)."""
    key = (cell, spec)
    model_mean = bundle.traces.model_mean
    if key not in model_mean:
        avail_cells = sorted({c for c, _ in model_mean})
        avail_specs = sorted(s for c, s in model_mean if c == cell)
        if cell not in avail_cells:
            raise SystemExit(f"cell {cell!r} not in bar bundle; available cells: {avail_cells}")
        raise SystemExit(f"spec {spec!r} not found for cell {cell!r}; available: {avail_specs}")

    out: dict[str, np.ndarray] = {}
    for label in _moving_bar_slice_labels(bundle):
        wt = bundle.slice_overlay[label]
        if key in wt.model_mean:
            out[label] = np.asarray(wt.model_mean[key], dtype=float)
    out["total"] = np.asarray(model_mean[key], dtype=float)
    return out


def specs_for_cell(bundle, cell: str, requested: list[str] | None) -> list[str]:
    model_mean = bundle.traces.model_mean
    avail = sorted(s for c, s in model_mean if c == cell)
    if not avail:
        cells = sorted({c for c, _ in model_mean})
        raise SystemExit(f"cell {cell!r} not in bar bundle; available cells: {cells}")
    if requested is None:
        return avail
    missing = [s for s in requested if s not in avail]
    if missing:
        raise SystemExit(
            f"spec(s) {missing} not found for cell {cell!r}; available: {avail}"
        )
    return list(requested)


def _print_curve(
    name: str,
    arr: np.ndarray,
    *,
    before_steps: int | None,
    print_values: bool,
):
    """Summarize + shape on post-onset when ``before_steps`` is set, else full trace."""
    arr = np.asarray(arr, dtype=float)
    if before_steps is not None and 0 < before_steps < arr.size:
        use = arr[before_steps:]
        idx_offset = before_steps
        window = f"post_onset[idx>={before_steps}]"
    else:
        use = arr
        idx_offset = 0
        window = "full"
    s = _summarize(use, idx_offset=idx_offset)
    shape = _shape_label(use)
    print(
        f"{name} ({window}): n={s.n} max={s.max:.6g} min={s.min:.6g} "
        f"final={s.final:.6g} peak_idx={s.peak_idx} "
        f"trough_idx={s.trough_idx} sign_changes={s.sign_changes}  "
        f"shape={shape}"
    )
    if print_values:
        print(f"  values: {np.round(use, 4).tolist()}")


def _print_spot_block(
    *,
    run_i: int,
    run_dir: str,
    best_i: int,
    best_cost: float,
    cell: str,
    target: str,
    trace_kind: str,
    session,
    bundle,
    curves: dict[str, np.ndarray],
    x_list,
    y_list,
    print_values: bool,
):
    print("")
    print(f"== RUN {run_i}: {run_dir} ==")
    print(
        f"best_i={best_i}  best_cost={best_cost:.6g}  cell={cell}  "
        f"target={target}  trace_kind={trace_kind}"
    )
    print(f"maxtime={bundle.maxtime}  deltat_ms={getattr(session, 'deltat_ms', 'NA')}")
    if x_list is not None or y_list is not None:
        print(f"slice_x={x_list}  slice_y={y_list}")
    for key, arr in curves.items():
        _print_curve(key, arr, before_steps=None, print_values=print_values)


def _print_bar_block(
    *,
    run_i: int,
    run_dir: str,
    best_i: int,
    best_cost: float,
    cell: str,
    target: str,
    trace_kind: str,
    spec: str,
    session,
    bundle,
    curves: dict[str, np.ndarray],
    x_list,
    y_list,
    print_values: bool,
):
    before_steps = None
    if bundle.traces.before_steps is not None:
        before_steps = bundle.traces.before_steps.get(spec)
    print("")
    print(f"== RUN {run_i}: {run_dir} ==")
    print(
        f"best_i={best_i}  best_cost={best_cost:.6g}  cell={cell}  "
        f"target={target}  trace_kind={trace_kind}  spec={spec}"
    )
    print(f"maxtime={bundle.maxtime}  deltat_ms={getattr(session, 'deltat_ms', 'NA')}")
    if x_list is not None or y_list is not None:
        print(f"slice_x={x_list}  slice_y={y_list}")
    if before_steps is not None:
        print(f"cost_window_start_idx={before_steps}")
    slice_names = [k for k in curves if k != "total"]
    order = slice_names + ["total"]
    print(f"traces ({len(order)}): {', '.join(order)}")
    for name in order:
        _print_curve(
            name, curves[name], before_steps=before_steps, print_values=print_values,
        )


def main():
    if __package__ is None:
        raise SystemExit(
            "run as a module from SimulationCode/: "
            "../.venv/bin/python -m analysis.cell_trace ..."
        )

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_shared_cli(ap)
    ap.add_argument(
        "--values",
        action="store_true",
        help="print full analysis-window trace arrays",
    )
    args = ap.parse_args()
    cli = parse_shared_cli(args)

    for run_i, run_arg in enumerate(args.run):
        run_dir = plot_trained.resolve_run_dir(run_arg)
        session, z, best_i, best_cost = plot_trained.load_best(run_dir)

        spot_cache: dict[str, tuple] = {}
        bar_cache: dict[str, object] = {}

        for target in cli.targets:
            if target in fc.SPOT_TARGETS:
                if target not in spot_cache:
                    spot_cache[target] = extract_spot_bundle(
                        session,
                        z,
                        target=target,
                        trace_kind=args.trace_kind,
                        x_list=cli.x_list,
                        y_list=cli.y_list,
                    )
                _one, bundle, ref_on = spot_cache[target]
                for cell in cli.cells:
                    curves = extract_spot_cell_curves(
                        bundle, ref_on, cell=cell, slice_label=cli.slice_label,
                    )
                    _print_spot_block(
                        run_i=run_i,
                        run_dir=run_dir,
                        best_i=best_i,
                        best_cost=best_cost,
                        cell=cell,
                        target=target,
                        trace_kind=args.trace_kind,
                        session=session,
                        bundle=bundle,
                        curves=curves,
                        x_list=cli.x_list,
                        y_list=cli.y_list,
                        print_values=args.values,
                    )
            else:
                if target not in bar_cache:
                    bar_cache[target] = extract_moving_bar_bundle(
                        session,
                        z,
                        target=target,
                        trace_kind=args.trace_kind,
                        x_list=cli.x_list,
                        y_list=cli.y_list,
                    )
                bundle = bar_cache[target]
                for cell in cli.cells:
                    for spec in specs_for_cell(bundle, cell, cli.specs_req):
                        curves = extract_moving_bar_cell_curves(
                            bundle, cell=cell, spec=spec,
                        )
                        _print_bar_block(
                            run_i=run_i,
                            run_dir=run_dir,
                            best_i=best_i,
                            best_cost=best_cost,
                            cell=cell,
                            target=target,
                            trace_kind=args.trace_kind,
                            spec=spec,
                            session=session,
                            bundle=bundle,
                            curves=curves,
                            x_list=cli.x_list,
                            y_list=cli.y_list,
                            print_values=args.values,
                        )


if __name__ == "__main__":
    main()
