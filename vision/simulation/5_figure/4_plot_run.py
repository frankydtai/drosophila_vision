"""Simulation + plotting for the FiveCol medulla model."""
import argparse
import json
import os
import numpy as np
import torch

import import_bootstrap  # noqa: F401
import training
from task.moving_bar.input import sti_hexes
from task.spot.input import spot_from_opts
from figure import moving_bar as moving_bar_plot
from figure import spot as spot_plot
from figure.util import (
    add_ms_shown_argument,
    parse_axis_slices,
    parse_align_xy,
    parse_ms_shown_range,
    plot_cost,
    network_hex_count,
)
from training.config import PARAMETER_DIR, run_data_dir
from training.implement import resolve_run_dir

TRAIN_OPTS_FILE = training.TRAIN_OPTS_FILE
KNOWN_MODELS = training.KNOWN_MODELS
DEFAULT_RUN_NAME = """
28602804-run-nofsteps-200-tau-hp-init.L1,L2,L4,L5-200-ms-pre-1000-ms-pulse-100-ms-response-500-model-borst
""".strip()
DEFAULT_RUN_PATH = 'borst/' + DEFAULT_RUN_NAME


def _plot_device_label():
    dev = training.active_device()
    if dev == 'cuda' and torch.cuda.is_available():
        import torch as _torch
        return f'cuda ({_torch.cuda.get_device_name(0)})'
    return dev


def spot_bundle_fns(session):
    if session.backend.network is None:
        raise ValueError("spot_bundle_fns requires session.backend.network")
    return (
        spot_plot.network_spot_trace_bundle,
        spot_plot.plot_network_spot_gt,
        spot_plot.plot_network_spot_all,
    )


def _session_trace_kind(session):
    """Plot/trace kind is always ``v``."""
    return "v"


def _network_spot_tag(session, tname):
    """Subtitle suffix for network spot plots (exact spot/shift counts)."""
    if session.backend.network is None:
        return ''
    opts = (session.train_opts or {}).get(f'{tname}_stimulus_opts') or {}
    spot = spot_from_opts(session.backend.network, stimulus_opts=opts)
    n_spots = len(spot.centers)
    n_shifts = len(spot.shifts)
    n_columns = network_hex_count(session.backend.network)
    return (
        f'  [avg over {n_spots} spots x {n_shifts} shifts = {n_spots * n_shifts}]\n'
        f'({n_columns} hexes in network)'
    )


def load_train_opts(outdir):
    opts_path = os.path.join(run_data_dir(os.path.abspath(outdir)), TRAIN_OPTS_FILE)
    if not os.path.isfile(opts_path):
        return None
    with open(opts_path) as f:
        return json.load(f)


def load_session(outdir, model=None):
    return training.open_session_from_outdir(outdir, model)


def session_for_task(base_session, tname):
    """Single-task session sharing backend/schema with a multi-task run."""
    if base_session.backend.network is None:
        raise ValueError("session_for_task requires base_session.backend.network")
    opts = dict(base_session.train_opts or {})
    opts['tasks'] = [tname]
    opts['packs'] = None
    opts['network'] = base_session.backend.network
    return training.open_session({**opts, 'backend': 'network'}, base_session.model,
                           schema=list(base_session.schema))


def resolve_model(outdir, override=None):
    if override is not None:
        model = override
    else:
        opts = load_train_opts(outdir)
        if not opts or 'model' not in opts:
            raise SystemExit(
                f'cannot determine model for {outdir!r}; '
                f'expected {TRAIN_OPTS_FILE} with "model" in {KNOWN_MODELS}'
            )
        model = opts['model']
    if model not in KNOWN_MODELS:
        raise SystemExit(
            f'invalid model {model!r} in {outdir!r}; '
            f'expected one of {KNOWN_MODELS}'
        )
    return model


def select_best(params, session, *, final_costs=None, best_i=None, verbose=True):
    """Pick the best parameter row; recompute costs only when not supplied."""
    params = np.atleast_2d(params)
    valid_mask = np.any(params != 0, axis=1)
    valid = params[valid_mask]
    if len(valid) == 0:
        raise SystemExit('no trained parameter sets found (file all zeros)')
    valid_idx = np.where(valid_mask)[0]

    if best_i is not None:
        if best_i not in valid_idx:
            raise ValueError(f'best_i={best_i} is not a valid trained run index')
        loc = int(np.where(valid_idx == best_i)[0][0])
        if final_costs is not None:
            best_cost = float(final_costs[best_i])
        else:
            z = torch.tensor(valid[loc], dtype=session.sim_dtype, device=session.device)
            best_cost = training.calc_cost(z, session).item()
        if verbose:
            print(f'{len(valid)} trained set(s); selected #{best_i} (cost={best_cost:.4f})')
        return valid[loc], best_cost, best_i

    if final_costs is not None:
        costs_arr = np.asarray(final_costs, dtype=np.float64)
        if costs_arr.shape[0] != params.shape[0]:
            raise ValueError(
                f'final_costs length {costs_arr.shape[0]} != params runs {params.shape[0]}',
            )
        valid_costs = costs_arr[valid_mask]
        best = int(np.argmin(valid_costs))
        run_i = int(valid_idx[best])
        if verbose:
            print(f'{len(valid)} trained set(s); costs min={valid_costs.min():.4f} '
                  f'max={valid_costs.max():.4f}; selected #{run_i} (from saved final costs)')
        return valid[best], float(costs_arr[run_i]), run_i

    costs_out = []
    for row in valid:
        z = torch.tensor(row, dtype=session.sim_dtype, device=session.device)
        costs_out.append(training.calc_cost(z, session).item())
    costs_out = np.array(costs_out)
    best = int(np.argmin(costs_out))
    run_i = int(valid_idx[best])
    if verbose:
        print(f'{len(valid)} trained set(s); costs min={costs_out.min():.4f} '
              f'max={costs_out.max():.4f}; selected #{run_i}')
    return valid[best], costs_out[best], run_i


def load_best(outdir, *, model=None, verbose=False):
    """Load session and best ``z`` from a train run (``best_param.npz`` + costs)."""
    import training.implement as train_mod

    outdir = os.path.abspath(outdir)
    if not os.path.isdir(outdir):
        raise SystemExit(f'run dir not found: {outdir}')
    model = resolve_model(outdir, override=model)
    session = load_session(outdir, model=model)
    named, cell_names, pair_names = train_mod.load_best_param_named(outdir)
    remapped = training.remap_named_node_values(
        named, cell_names, pair_names, list(session.schema), session.backend,
    )
    schema = training.attach_param_carry(list(session.schema), remapped)
    session = session.with_schema(schema)
    z = training.node_values_to_z(
        remapped, schema, dtype=session.sim_dtype, device=session.device,
    )
    best_i = train_mod.load_best_i(outdir)
    best_cost = None
    final_costs, _, _, _ = train_mod.load_stored_costs(outdir)
    if final_costs is not None:
        idx = best_i if best_i is not None else int(np.argmin(final_costs))
        if idx < len(final_costs):
            best_cost = float(final_costs[idx])
            best_i = idx
    if best_cost is None:
        best_cost = training.calc_cost(z, session).item()
        best_i = best_i if best_i is not None else 0
    elif best_i is None:
        best_i = 0
    if verbose:
        print(f'loaded best_param.npz (cost={best_cost:.4f}, best_i={best_i})')
    return session, z, int(best_i), float(best_cost)


def maybe_override_stimulus_timing(
    *,
    run_dir,
    session,
    z,
    ms_pre=None,
    ms_response=None,
    ms_pulse=None,
    delta_ms=None,
):
    """Re-open session when any timing override is set; remap best ``z``.

    Unset flags keep values from the run's train opts. ``ms_pre`` / ``delta_ms``
    also update moving_bar stimulus opts; ``ms_response`` / ``ms_pulse`` are
    spot-only.
    """
    if (
        ms_pre is None
        and ms_response is None
        and ms_pulse is None
        and delta_ms is None
    ):
        return session, z

    if delta_ms is not None and float(delta_ms) <= 0:
        raise SystemExit("--delta-ms must be > 0")

    import training.implement as train_mod

    opts = load_train_opts(run_dir)
    if opts is None:
        raise SystemExit(f"missing train opts under {run_dir}")

    for key in ("spot_bright_stimulus_opts", "spot_dark_stimulus_opts"):
        so = opts.get(key)
        if so is None:
            continue
        if ms_pre is not None:
            so["ms_pre"] = float(ms_pre)
        if ms_response is not None:
            so["ms_response"] = float(ms_response)
        if ms_pulse is not None:
            so["ms_pulse"] = float(ms_pulse)
        if delta_ms is not None:
            so["delta_ms"] = float(delta_ms)
        so.pop("t_onset", None)
        so.pop("n_t", None)

    if ms_pre is not None or delta_ms is not None:
        for key in (
            "moving_bar_bright_stimulus_opts",
            "moving_bar_dark_stimulus_opts",
        ):
            so = opts.get(key)
            if so is None:
                continue
            if ms_pre is not None:
                so["ms_pre"] = float(ms_pre)
            if delta_ms is not None:
                so["delta_ms"] = float(delta_ms)
            so.pop("t_onset", None)
            so.pop("n_t", None)

    session = training.open_session_from_opts(opts, model=opts.get("model"))
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
    return session, z


def _format_ms_filename_token(value):
    v = float(value)
    if v == int(v):
        return str(int(v))
    return ("%g" % v)


def stimulus_timing_filename_suffix(
    *,
    ms_pre=None,
    ms_pulse=None,
    ms_response=None,
    delta_ms=None,
):
    """PNG stem suffix for non-``None`` timing overrides (plot / analyze).

    Order: pre, pulse, response, delta. Example::

        _pulse_200_response_2000

    Empty string when every override is unset (keep run train opts).
    """
    parts = []
    for name, val in (
        ("ms_pre", ms_pre),
        ("ms_pulse", ms_pulse),
        ("ms_response", ms_response),
        ("delta", delta_ms),
    ):
        if val is not None:
            parts.append(f"{name}_{_format_ms_filename_token(val)}")
    if not parts:
        return ""
    return "_" + "_".join(parts)


def _cost_parts_for_plot(session, z):
    """Unweighted per-part costs at ``z`` for panel titles."""
    with torch.no_grad():
        parts = training.calc_cost_parts(z, session)
    return {k: float(v.item()) for k, v in parts.items()}


def _plot_path(outdir, stem, file_suffix="", *, html=False):
    from figure.util import plot_file_ext

    return os.path.join(outdir, f"{stem}{file_suffix}{plot_file_ext(html=html)}")


def _plot_spot_tasks(session, z, outdir, spot_tasks, suffix, model_all,
                       gt_cubes=None,
                       at_x=None, at_y=None, show_pre=True,
                       file_suffix="", html=False, ms_shown=None):
    """Plot spot task(s); contrasts combined in one figure when both are trained."""
    spot_set = set(spot_tasks)
    make_bundle, plot_gt, plot_all = spot_bundle_fns(session)
    ref_t = 'spot_bright' if 'spot_bright' in spot_set else spot_tasks[0]
    net_tag = _network_spot_tag(session, ref_t)
    kind = _session_trace_kind(session)
    cost_parts = _cost_parts_for_plot(session, z)
    plot_kw = dict(gt_cubes=gt_cubes, cost_parts=cost_parts)
    bundle_kw = dict(
        at_xs=at_x, at_ys=at_y,
        show_pre=show_pre,
        ms_shown=ms_shown,
    )
    if spot_set == set(training.SPOT_TASKS):
        bundles = {
            'bright': make_bundle(
                session_for_task(session, 'spot_bright'), z, **bundle_kw,
            ),
            'dark': make_bundle(
                session_for_task(session, 'spot_dark'), z, **bundle_kw,
            ),
        }
        mvd = _plot_path(outdir, f'spot_gt_{kind}', file_suffix, html=html)
        plot_gt(
            mvd, bundles=bundles,
            title=f'Spot {kind}-gt ({suffix}){net_tag}',
            **plot_kw,
        )
        allc = None
        if model_all:
            allc = _plot_path(outdir, f'spot_all_{kind}', file_suffix, html=html)
            plot_all(
                allc, bundles=bundles,
                title=f'Spot {kind}-all ({suffix}){net_tag}',
                **plot_kw,
            )
        return mvd, allc
    for tname in spot_tasks:
        _plot_one_task(
            session_for_task(session, tname), z, outdir, tname, suffix, model_all,
            gt_cubes=gt_cubes,
            at_x=at_x, at_y=at_y,
            show_pre=show_pre,
            cost_parts=cost_parts,
            file_suffix=file_suffix,
            html=html,
            ms_shown=ms_shown,
        )


def _plot_bar_readouts(session, z, outdir, bar_readouts, suffix, model_all, *,
                      plot_right_only=True, at_x=None, at_y=None,
                      align_at_x=None, align_at_y=None,
                      show_pre=True, file_suffix="", html=False, ms_shown=None):
    """Plot moving-bar task(s); bright left | dark right when both are trained."""
    kind = _session_trace_kind(session)
    cost_parts = _cost_parts_for_plot(session, z)
    bundle_kw = dict(
        at_xs=at_x, at_ys=at_y,
        align_at_x=align_at_x, align_at_y=align_at_y,
        show_pre=show_pre,
        ms_shown=ms_shown,
    )
    bar_set = set(bar_readouts)
    if bar_set == set(training.MOVING_BAR_TASKS):
        s_bright = session_for_task(session, 'moving_bar_bright')
        s_dark = session_for_task(session, 'moving_bar_dark')
        b_bright = moving_bar_plot.moving_bar_trace_bundle(
            s_bright, z, 'moving_bar_bright', **bundle_kw,
        )
        b_dark = moving_bar_plot.moving_bar_trace_bundle(
            s_dark, z, 'moving_bar_dark', **bundle_kw,
        )
        mvd = _plot_path(outdir, f'bar_gt_{kind}', file_suffix, html=html)
        moving_bar_plot.plot_moving_bar_data(
            mvd, bundle=b_bright, bundle_2=b_dark,
            title=f'Moving-bar {kind}-gt ({suffix})',
            cost_parts=cost_parts,
        )
        allc = None
        if model_all:
            allc = _plot_path(outdir, f'bar_all_{kind}', file_suffix, html=html)
            moving_bar_plot.plot_moving_bar_all(
                allc, bundle=b_bright, bundle_2=b_dark,
                title=f'Moving-bar {kind}-all ({suffix})',
                right_only=plot_right_only,
                cost_parts=cost_parts,
            )
        return mvd, allc
    for tname in bar_readouts:
        one = session_for_task(session, tname)
        b = moving_bar_plot.moving_bar_trace_bundle(one, z, tname, **bundle_kw)
        mvd = _plot_path(outdir, f'bar_gt_{kind}', file_suffix, html=html)
        moving_bar_plot.plot_moving_bar_data(
            mvd, bundle=b, title=f'{tname} {kind}-gt ({suffix})',
            cost_parts=cost_parts,
        )
        allc = None
        if model_all:
            allc = _plot_path(outdir, f'bar_all_{kind}', file_suffix, html=html)
            moving_bar_plot.plot_moving_bar_all(
                allc, bundle=b, title=f'{tname} {kind}-all ({suffix})',
                right_only=plot_right_only,
                cost_parts=cost_parts,
            )
        return mvd, allc


def _plot_one_task(session, z, outdir, tname, suffix, model_all,
                     gt_cubes=None,
                     at_x=None, at_y=None, show_pre=True,
                     cost_parts=None, file_suffix="", html=False, ms_shown=None):
    if tname not in training.SPOT_TASKS:
        raise ValueError(f'unknown plot task {tname!r}')
    kind = _session_trace_kind(session)
    mvd = _plot_path(outdir, f'spot_gt_{kind}', file_suffix, html=html)
    allc = _plot_path(outdir, f'spot_all_{kind}', file_suffix, html=html)
    make_bundle, plot_gt, plot_all = spot_bundle_fns(session)
    net_tag = _network_spot_tag(session, tname)
    if cost_parts is None:
        cost_parts = _cost_parts_for_plot(session, z)
    plot_kw = dict(gt_cubes=gt_cubes, cost_parts=cost_parts)
    b = make_bundle(
        session, z,
        at_xs=at_x, at_ys=at_y,
        show_pre=show_pre,
        ms_shown=ms_shown,
    )
    from figure.readout import contrast_for_task
    bundles = {contrast_for_task(tname): b}
    plot_gt(
        mvd, bundles=bundles, title=f'{tname} {kind}-gt ({suffix}){net_tag}', **plot_kw,
    )
    if model_all:
        plot_all(allc, bundles=bundles, title=f'{tname} {kind}-all ({suffix}){net_tag}', **plot_kw)
    return mvd, allc


def plot_param_set(params, outdir, model=None, model_all=True,
                   context_dir=None,
                   plot_tasks=None, session=None, *,
                   final_costs=None, cost_curve=None, costs_by_part=None, best_i=None,
                   save_artifacts=True,
                   gt_cubes=None,
                   plot_right_only=True, at_x=None, at_y=None,
                   align_at_x=None, align_at_y=None,
                   show_pre=True, file_suffix="", html=False, ms_shown=None):
    from figure.util import plot_file_ext

    os.makedirs(outdir, exist_ok=True)
    data_dir = run_data_dir(os.path.abspath(outdir))
    os.makedirs(data_dir, exist_ok=True)
    ctx = context_dir or outdir
    if model is None and session is not None:
        model = session.model
    if model is None:
        raise ValueError('model or session required')
    if session is None:
        session = load_session(ctx, model)

    params = np.atleast_2d(params)
    if final_costs is None or cost_curve is None or costs_by_part is None:
        loaded_final, loaded_curve, loaded_by_part, _ = _load_plot_costs(outdir)
        if final_costs is None:
            final_costs = loaded_final
        if cost_curve is None:
            cost_curve = loaded_curve
        if costs_by_part is None:
            costs_by_part = loaded_by_part

    if best_i is None:
        import training.implement as train_mod
        best_i = train_mod.load_best_i(ctx)

    print(f'plot device={_plot_device_label()}')
    best, best_cost, best_i = select_best(
        params, session, final_costs=final_costs, best_i=best_i,
    )
    z = torch.tensor(best, dtype=session.sim_dtype, device=session.device)

    suffix = f'trained, cost {best_cost:.2f}% of gt power'
    tasks = list(session.tasks)
    if plot_tasks is not None:
        tasks = [t for t in tasks if t in plot_tasks]

    spot_tasks = [t for t in tasks if t in training.SPOT_TASKS]
    bar_readouts = [t for t in tasks if t in training.MOVING_BAR_TASKS]
    other_readouts = [
        t for t in tasks
        if t not in training.SPOT_TASKS and t not in training.MOVING_BAR_TASKS
    ]
    if (at_x is not None or at_y is not None) and not bar_readouts and not spot_tasks:
        raise SystemExit('--x/--y require a moving_bar or spot task in this run')
    if (align_at_x is not None or align_at_y is not None):
        if align_at_x is None or align_at_y is None:
            raise SystemExit('--align-xy requires X,Y')
        if at_x is None and at_y is None:
            raise SystemExit('--align-xy requires --x and/or --y')
        if not bar_readouts:
            raise SystemExit('--align-xy applies to moving_bar slice plots only')

    if cost_curve is not None and len(cost_curve) > 0:
        plot_cost(
            cost_curve,
            os.path.join(outdir, f'cost_curve{plot_file_ext(html=html)}'),
            costs_by_part=costs_by_part,
            part_order=list(training.session_cost_part_keys(session.tasks, session=session)),
        )
    if spot_tasks:
        _plot_spot_tasks(
            session, z, outdir, spot_tasks, suffix, model_all,
            gt_cubes=gt_cubes,
            at_x=at_x, at_y=at_y,
            show_pre=show_pre,
            file_suffix=file_suffix,
            html=html,
            ms_shown=ms_shown,
        )
    if bar_readouts:
        _plot_bar_readouts(
            session, z, outdir, bar_readouts, suffix, model_all,
            plot_right_only=plot_right_only,
            at_x=at_x, at_y=at_y,
            align_at_x=align_at_x, align_at_y=align_at_y,
            show_pre=show_pre,
            file_suffix=file_suffix,
            html=html,
            ms_shown=ms_shown,
        )
    for tname in other_readouts:
        one = session_for_task(session, tname)
        _plot_one_task(
            one, z, outdir, tname, suffix, model_all,
            gt_cubes=gt_cubes,
            show_pre=show_pre,
            file_suffix=file_suffix,
            html=html,
            ms_shown=ms_shown,
        )

    if save_artifacts:
        import training.implement as train_mod
        os.makedirs(train_mod.data_dir(outdir), exist_ok=True)
        z_best = torch.tensor(best, dtype=session.sim_dtype, device=session.device)
        train_mod.save_best_param_named(outdir, z_best, session)
        train_mod.write_best_i(outdir, best_i)
    print(f'plots saved to {outdir}')
    return best, best_cost


def _load_plot_costs(outdir):
    """Load per-run and step costs saved by ``training.implement.save_training_outputs``."""
    import training.implement as train_mod
    return train_mod.load_stored_costs(outdir)


def add_plot_arguments(parser):
    """Register plot-only CLI flags shared by ``run.py`` and ``figure.plot_run``.

    Timing overrides (``--ms-pre`` / …) are registered separately via
    :func:`add_plot_timing_arguments` so ``run.py`` does not double-register
    flags already on the train CLI.
    """
    from import_bootstrap import parse_bool

    parser.add_argument(
        '--html',
        action='store_true',
        help='save interactive plotly HTML (hover x/y) instead of PNG',
    )
    parser.add_argument(
        '--plot-right-only',
        nargs='?',
        const=True,
        default=True,
        type=parse_bool,
        metavar='BOOL',
        help='bar_all_{ca|v}: right-direction specs only (default true); '
             'pass false for all directions',
    )
    parser.add_argument(
        '--show-pre',
        nargs='?',
        const=True,
        default=True,
        type=parse_bool,
        metavar='BOOL',
        help='model pre-t_onset: dashed red when true (default); omit when false. '
             'Gray gt never draws pre.',
    )
    parser.add_argument(
        '--x',
        default=None,
        metavar='X,...',
        help='{bar,spot}_all_{ca|v}: comma-separated x slices; with --y, one trace per (x,y) pair',
    )
    parser.add_argument(
        '--y',
        default=None,
        metavar='Y,...',
        help='{bar,spot}_all_{ca|v}: comma-separated y slices; with --x, one trace per (x,y) pair',
    )
    parser.add_argument(
        '--align-xy',
        default=None,
        metavar='X,Y',
        help='moving_bar slice plots: align --x/--y traces to ref hex hex (x,y); total unchanged',
    )
    add_ms_shown_argument(parser)


def add_plot_timing_arguments(parser):
    """Hang train stimulus-timing CLI onto plot / analyze (defaults: keep run)."""
    import training.implement as train_mod

    train_mod.add_stimulus_timing_arguments(
        parser,
        default_ms_pre=None,
        default_ms_response=None,
        default_ms_pulse=None,
        default_delta_ms=None,
    )


def plot_kwargs_from_args(args):
    """Map a parsed CLI namespace to :func:`plot_param_set` plot kwargs."""
    align_xy = parse_align_xy(args.align_xy)
    align_at_x, align_at_y = align_xy if align_xy is not None else (None, None)
    ms_shown = None
    if getattr(args, 'ms_shown', None) is not None:
        ms_shown = parse_ms_shown_range(args.ms_shown)
    return dict(
        plot_right_only=args.plot_right_only,
        show_pre=args.show_pre,
        at_x=parse_axis_slices(args.x),
        at_y=parse_axis_slices(args.y),
        align_at_x=align_at_x,
        align_at_y=align_at_y,
        html=bool(getattr(args, 'html', False)),
        ms_shown=ms_shown,
    )


def main():
    import training.implement as train_mod

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        'run_path',
        nargs='?',
        default=DEFAULT_RUN_PATH,
        help='run folder under PARAMETER_DIR or absolute path (default: %(default)s)',
    )
    add_plot_arguments(ap)
    add_plot_timing_arguments(ap)
    args = ap.parse_args()
    try:
        plot_kw = plot_kwargs_from_args(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    timing_kw = train_mod.stimulus_timing_kwargs_from_args(args)
    file_suffix = stimulus_timing_filename_suffix(**timing_kw)

    outdir = resolve_run_dir(args.run_path)
    session, z, _stored_best_i, best_cost = load_best(outdir, verbose=True)
    session, z = maybe_override_stimulus_timing(
        run_dir=outdir, session=session, z=z, **timing_kw,
    )
    model = resolve_model(outdir)
    z_np = z.detach().cpu().numpy() if torch.is_tensor(z) else np.asarray(z)
    print(f'outdir={outdir}')
    print(f'params={train_mod.best_param_path(outdir)}')
    print(f'model={model} ({z_np.shape[-1]} params)')
    plot_param_set(
        np.atleast_2d(z_np),
        outdir,
        model=model,
        session=session,
        best_i=0,
        final_costs=np.array([best_cost]),
        file_suffix=file_suffix,
        **plot_kw,
    )


if __name__ == '__main__':
    main()
