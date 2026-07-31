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
``network_spot_trace_bundle``.

Moving bar
----------
``t_first_sti``-aligned window; optional ``--x`` / ``--y`` overlays match
``bar_all_ca``. Summary stats and shape labels for bar traces use the
**post-onset** segment (``idx >= cost_window_start_idx``); pre-stim extrema
are ignored.

Examples
--------
  # from SimulationCode/ (module form; do not run as a loose file path)
  cd temporal_filtering/SimulationCode
  ../.venv/bin/python -m analyze.cell_trace \\
    --run /abs/path/to/run --cell Mi4,Mi9 \\
    --target spot_bright,moving_bar_bright --trace-kind v

  ../.venv/bin/python -m analyze.cell_trace \\
    --run /abs/path/to/run --cell L4 --target spot_dark \\
    --trace-kind v --x 2 --y 1

  ../.venv/bin/python -m analyze.cell_trace \\
    --run /abs/path/to/run --cell Mi4,Mi9 \\
    --target moving_bar_bright --spec right_bright_w1,left_bright_w1 \\
    --trace-kind v
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import numpy as np

import import_bootstrap  # noqa: F401
import training
import figure.plot_run as plot_trained
from figure import moving_bar as moving_bar_plot
from figure import spot as spot_plot
from figure.util import parse_axis_slice_list, slice_xy_label
from connectome_io import parse_comma_list
from training.driver import parse_target_list
from task.moving_bar.data import filter_requested_specs
from training.config import run_data_dir


DEFAULT_POST_ONSET_MS = 1500.0


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
    """Register ``--run/--cell/--target/--spec/--x/--y`` (shared)."""
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
        if t not in training.SPOT_TARGETS and t not in training.MOVING_BAR_TARGETS:
            raise SystemExit(
                f"unsupported target {t!r}; expected spot_* or moving_bar_* "
                f"(after TARGET_ALIASES expansion)"
            )
    x_list = parse_axis_slice_list(args.x)
    y_list = parse_axis_slice_list(args.y)
    if (x_list is None) ^ (y_list is None):
        if any(t in training.SPOT_TARGETS for t in targets):
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


def extract_spot_bundle(session, z, *, target: str, x_list, y_list):
    """One spot forward via ``plot_trained.spot_bundle_fns``.

    Returns ``(session_one, bundle, data_cubes)`` where ``data_cubes`` is
    ``{contrast: {cell: (9, T)}}``.
    """
    one = plot_trained.session_for_target(session, target)
    make_bundle, _, _ = plot_trained.spot_bundle_fns(one)
    bundle = make_bundle(
        one,
        z,
        at_x_list=x_list,
        at_y_list=y_list,
        save_trace_csv_dir=None,
    )
    from figure.readout import contrast_for_target
    data_cubes = spot_plot.resolve_spot_data_cubes(
        {contrast_for_target(one.primary_pack.name): one},
    )
    return one, bundle, data_cubes


def extract_moving_bar_bundle(session, z, *, target: str, x_list, y_list):
    """One moving-bar forward; all cells + all specs live on the returned bundle."""
    one = plot_trained.session_for_target(session, target)
    return moving_bar_plot.moving_bar_trace_bundle(
        one,
        z,
        target,
        at_x_list=x_list,
        at_y_list=y_list,
        save_trace_csv_dir=None,
    )


def extract_spot_cell_curves(bundle, data_cubes, *, cell: str, slice_label: str | None = None):
    from figure.readout import contrast_for_target

    center = spot_plot.CENTER_BIN
    cell_on = next((c for c in bundle.cells if c["name"] == cell), None)
    if cell_on is None:
        avail = sorted(c["name"] for c in bundle.cells)
        raise SystemExit(f"cell {cell!r} not in spot bundle; available: {avail}")
    contrast = contrast_for_target(bundle.session.primary_pack.name)
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


def _moving_bar_slice_labels(bundle) -> list[str]:
    if bundle.slice_overlay is None:
        return []
    return list(bundle.slice_overlay.keys())


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
    for label in _moving_bar_slice_labels(bundle):
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


def _print_curve(
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


def _print_result_block(
    *,
    run_i: int,
    run_dir: str,
    best_i: int,
    best_cost: float,
    cell: str,
    target: str,
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
        f"target={target}  trace_kind=v"
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
            _print_curve(
                name,
                curves[name],
                before_t=before_t,
                print_values=print_values,
                head_t=head_t,
                head_window=head_window,
            )
    else:
        for key, arr in curves.items():
            # RF arrays are spatial profiles, not time courses.
            if head_t is not None and not key.startswith("time_"):
                continue
            _print_curve(
                key,
                arr,
                before_t=None,
                print_values=print_values,
                head_t=head_t,
                head_window=head_window,
            )


def _ms_to_t(ms: float, delta_ms: float) -> int:
    """Map ms to simulation indices (floor division)."""
    if delta_ms <= 0:
        raise ValueError(f"invalid delta_ms={delta_ms}")
    return int(float(ms) / float(delta_ms))


def _load_train_opts(run_dir: str) -> dict:
    opts_path = os.path.join(run_data_dir(os.path.abspath(run_dir)), training.TRAIN_OPTS_FILE)
    with open(opts_path) as f:
        return json.load(f)


def _maybe_override_spot_timing(
    *,
    run_dir: str,
    session,
    pre_ms: float | None,
    response_ms: float | None,
):
    """Optionally re-open the session with overridden spot timing.

    This must re-open the session (not just mutate ``session.train_opts``),
    because the precomputed stimulus tensors (e.g. ``pack.signal``) depend on
    ``pre_ms`` / ``response_ms``.
    """
    if pre_ms is None:
        return session, None, None

    opts = _load_train_opts(run_dir)
    if response_ms is None:
        response_ms = float(DEFAULT_POST_ONSET_MS)

    for key in ("spot_bright_stimulus_opts", "spot_dark_stimulus_opts"):
        so = opts.get(key)
        if so is not None:
            so["pre_ms"] = float(pre_ms)
            so["response_ms"] = float(response_ms)
            so.pop("t_onset", None)
            so.pop("n_t", None)

    # Re-open the session with updated stimulus opts.
    new_session = training.open_session_from_opts(opts, model=opts.get("model"))
    dt = float((opts.get("spot_bright_stimulus_opts") or {}).get("delta_ms", 10.0))
    new_t_onset = _ms_to_t(pre_ms, dt)
    return new_session, dt, new_t_onset


def main():
    if __package__ is None:
        raise SystemExit(
            "run as a module from SimulationCode/: "
            "../.venv/bin/python -m analyze.cell_trace ..."
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
    ap.add_argument(
        "--pre-ms",
        type=float,
        default=None,
        help="override spot pre-stimulus baseline in ms (re-opens session; affects spot_*)",
    )
    ap.add_argument(
        "--response-ms",
        type=float,
        default=None,
        help="override spot post-onset response window in ms "
        "(default: 1500)",
    )
    ap.add_argument(
        "--t-max-ms",
        type=float,
        default=None,
        help="when --values is set, only print first N ms of each trace",
    )
    args = ap.parse_args()
    cli = parse_shared_cli(args)

    for run_i, run_arg in enumerate(args.run):
        run_dir = plot_trained.resolve_run_dir(run_arg)
        import training.driver as train_mod

        # Base load: get the stored best parameter + its cost for labeling.
        session0, _z0, best_i, best_cost = plot_trained.load_best(run_dir)

        dt_for_head = None
        head_t = None
        head_window = None

        # If requested, re-open session with overridden spot timing, and re-map z.
        session, z = session0, _z0
        dt_for_head = None
        if args.pre_ms is not None:
            session, dt_for_head, _new_t_onset = _maybe_override_spot_timing(
                run_dir=run_dir,
                session=session0,
                pre_ms=args.pre_ms,
                response_ms=args.response_ms,
            )
            # Re-load + re-map the best parameters for the new session schema.
            named, type_names, pair_names = train_mod.load_best_param_named(run_dir)
            remapped = training.remap_named_unit_values(
                named,
                type_names,
                pair_names,
                list(session.schema),
                session.backend,
            )
            schema = training.attach_param_carry(list(session.schema), remapped)
            session = session.with_schema(schema)
            z = training.unit_values_to_z(
                remapped,
                schema,
                dtype=session.sim_dtype,
                device=session.device,
            )

        if args.t_max_ms is not None:
            if dt_for_head is None:
                opts = _load_train_opts(run_dir)
                dt_for_head = float(
                    (opts.get("spot_bright_stimulus_opts") or {}).get("delta_ms", 10.0)
                )
            head_t = _ms_to_t(args.t_max_ms, dt_for_head)
            head_window = f"head[t<{args.t_max_ms:g}ms]"

        spot_cache: dict[str, tuple] = {}
        bar_cache: dict[str, object] = {}

        for target in cli.targets:
            if target in training.SPOT_TARGETS:
                if target not in spot_cache:
                    spot_cache[target] = extract_spot_bundle(
                        session,
                        z,
                        target=target,
                        x_list=cli.x_list,
                        y_list=cli.y_list,
                    )
                _one, bundle, data_cubes = spot_cache[target]
                for cell in cli.cells:
                    curves = extract_spot_cell_curves(
                        bundle, data_cubes, cell=cell, slice_label=cli.slice_label,
                    )
                    _print_result_block(
                        run_i=run_i,
                        run_dir=run_dir,
                        best_i=best_i,
                        best_cost=best_cost,
                        cell=cell,
                        target=target,
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
                if target not in bar_cache:
                    bar_cache[target] = extract_moving_bar_bundle(
                        session,
                        z,
                        target=target,
                        x_list=cli.x_list,
                        y_list=cli.y_list,
                    )
                bundle = bar_cache[target]
                for cell in cli.cells:
                    for spec in specs_for_cell(bundle, cell, cli.specs_req):
                        curves = extract_moving_bar_cell_curves(
                            bundle, cell=cell, spec=spec,
                        )
                        _print_result_block(
                            run_i=run_i,
                            run_dir=run_dir,
                            best_i=best_i,
                            best_cost=best_cost,
                            cell=cell,
                            target=target,
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


if __name__ == "__main__":
    main()
