"""Simulation + plotting for the FiveCol medulla model."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

import import_bootstrap  # noqa: F401
import training
from task.moving_bar.input import sti_columns
from task.spot.input import spot_from_opts
from figure import moving_bar as moving_bar_plot
from figure import spot as spot_plot
from figure.util import parse_axis_slice_list, parse_align_xy, plot_cost, network_column_count
from training.config import PARAMETER_DIR, run_data_dir
from training.driver import resolve_run_dir

TRAIN_OPTS_FILE = training.TRAIN_OPTS_FILE
KNOWN_MODELS = training.KNOWN_MODELS
DEFAULT_RUN_NAME = """
27252028-train-nofsteps-1000-lrs-0.1-shift-extent-1-cost-extent-9
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
        spot_plot.plot_network_spot_data,
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
    n_columns = network_column_count(session.backend.network)
    return (
        f'  [avg over {n_spots} spots x {n_shifts} shifts = {n_spots * n_shifts}]\n'
        f'({n_columns} columns in network)'
    )


def load_train_opts(outdir):
    opts_path = os.path.join(run_data_dir(os.path.abspath(outdir)), TRAIN_OPTS_FILE)
    if not os.path.isfile(opts_path):
        return None
    with open(opts_path) as f:
        return json.load(f)


def load_session(outdir, model=None):
    return training.open_session_from_outdir(outdir, model)


def session_for_target(base_session, tname):
    """Single-target session sharing backend/schema with a multi-target run."""
    if base_session.backend.network is None:
        raise ValueError("session_for_target requires base_session.backend.network")
    opts = dict(base_session.train_opts or {})
    opts['target_list'] = [tname]
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


def find_training_params(outdir):
    """Locate ``data/training*.npy`` params (exclude ``*_costs*`` sidecars)."""
    import training.driver as train_mod

    data = Path(train_mod.data_dir(outdir))
    candidates = sorted(
        p for p in data.glob('training*.npy')
        if '_costs' not in p.name
    )
    if len(candidates) != 1:
        raise SystemExit(
            f'expected exactly one training*.npy (non-costs) in {str(data)!r}, '
            f'found {len(candidates)}',
        )
    fname = candidates[0].name
    params_path = train_mod.params_path(outdir, fname)
    if not os.path.isfile(params_path):
        raise SystemExit(f'missing training params: {params_path!r}')
    return params_path, fname


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
    import training.driver as train_mod

    outdir = os.path.abspath(outdir)
    if not os.path.isdir(outdir):
        raise SystemExit(f'run dir not found: {outdir}')
    model = resolve_model(outdir, override=model)
    session = load_session(outdir, model=model)
    named, type_names, pair_names = train_mod.load_best_param_named(outdir)
    remapped = training.remap_named_unit_values(
        named, type_names, pair_names, list(session.schema), session.backend,
    )
    schema = training.attach_param_carry(list(session.schema), remapped)
    session = session.with_schema(schema)
    z = training.unit_values_to_z(
        remapped, schema, dtype=session.sim_dtype, device=session.device,
    )
    best_i = train_mod.load_best_i(outdir)
    best_cost = None
    try:
        params_path, fname = find_training_params(outdir)
        params = np.atleast_2d(np.load(params_path))
        final_costs, _, _, _ = train_mod.load_stored_costs(outdir, fname, params.shape[0])
        if final_costs is not None and best_i is not None and best_i < len(final_costs):
            best_cost = float(final_costs[best_i])
    except SystemExit:
        pass
    if best_cost is None:
        best_cost = training.calc_cost(z, session).item()
        best_i = best_i if best_i is not None else 0
    elif best_i is None:
        best_i = 0
    if verbose:
        print(f'loaded best_param.npz (cost={best_cost:.4f}, best_i={best_i})')
    return session, z, int(best_i), float(best_cost)


def _plot_spot_targets(session, z, outdir, spot_targets, suffix, model_all,
                       data_cubes=None,
                       at_x=None, at_y=None, save_trace_csv_dir=None, show_pre=True):
    """Plot spot target(s); contrasts combined in one figure when both are trained."""
    spot_set = set(spot_targets)
    make_bundle, plot_data, plot_all = spot_bundle_fns(session)
    ref_t = 'spot_bright' if 'spot_bright' in spot_set else spot_targets[0]
    net_tag = _network_spot_tag(session, ref_t)
    kind = _session_trace_kind(session)
    plot_kw = dict(data_cubes=data_cubes)
    bundle_kw = dict(
        at_x_list=at_x, at_y_list=at_y,
        save_trace_csv_dir=save_trace_csv_dir,
        show_pre=show_pre,
    )
    if spot_set == set(training.SPOT_TARGETS):
        bundles = {
            'bright': make_bundle(
                session_for_target(session, 'spot_bright'), z, **bundle_kw,
            ),
            'dark': make_bundle(
                session_for_target(session, 'spot_dark'), z, **bundle_kw,
            ),
        }
        mvd = os.path.join(outdir, f'spot_trained_{kind}.png')
        plot_data(
            mvd, bundles=bundles,
            title=f'Spot {kind}-data ({suffix}){net_tag}',
            **plot_kw,
        )
        allc = None
        if model_all:
            allc = os.path.join(outdir, f'spot_all_{kind}.png')
            plot_all(
                allc, bundles=bundles,
                title=f'Spot {kind}-all ({suffix}){net_tag}',
                **plot_kw,
            )
        return mvd, allc
    for tname in spot_targets:
        _plot_one_target(
            session_for_target(session, tname), z, outdir, tname, suffix, model_all,
            data_cubes=data_cubes,
            at_x=at_x, at_y=at_y,
            save_trace_csv_dir=save_trace_csv_dir, show_pre=show_pre,
        )


def _plot_bar_targets(session, z, outdir, bar_targets, suffix, model_all, *,
                      plot_right_only=True, at_x=None, at_y=None,
                      align_at_x=None, align_at_y=None,
                      save_trace_csv_dir=None, show_pre=True):
    """Plot moving-bar target(s); bright left | dark right when both are trained."""
    kind = _session_trace_kind(session)
    bundle_kw = dict(
        at_x_list=at_x, at_y_list=at_y,
        align_at_x=align_at_x, align_at_y=align_at_y,
        save_trace_csv_dir=save_trace_csv_dir,
        show_pre=show_pre,
    )
    bar_set = set(bar_targets)
    if bar_set == set(training.MOVING_BAR_TARGETS):
        s_bright = session_for_target(session, 'moving_bar_bright')
        s_dark = session_for_target(session, 'moving_bar_dark')
        b_bright = moving_bar_plot.moving_bar_trace_bundle(
            s_bright, z, 'moving_bar_bright', **bundle_kw,
        )
        b_dark = moving_bar_plot.moving_bar_trace_bundle(
            s_dark, z, 'moving_bar_dark', **bundle_kw,
        )
        mvd = os.path.join(outdir, f'bar_trained_{kind}.png')
        moving_bar_plot.plot_moving_bar_data(
            mvd, bundle=b_bright, bundle_2=b_dark,
            title=f'Moving-bar {kind}-data ({suffix})',
        )
        allc = None
        if model_all:
            allc = os.path.join(outdir, f'bar_all_{kind}.png')
            moving_bar_plot.plot_moving_bar_all(
                allc, bundle=b_bright, bundle_2=b_dark,
                title=f'Moving-bar {kind}-all ({suffix})',
                right_only=plot_right_only,
            )
        return mvd, allc
    for tname in bar_targets:
        one = session_for_target(session, tname)
        b = moving_bar_plot.moving_bar_trace_bundle(one, z, tname, **bundle_kw)
        mvd = os.path.join(outdir, f'bar_trained_{kind}.png')
        moving_bar_plot.plot_moving_bar_data(
            mvd, bundle=b, title=f'{tname} {kind}-data ({suffix})',
        )
        allc = None
        if model_all:
            allc = os.path.join(outdir, f'bar_all_{kind}.png')
            moving_bar_plot.plot_moving_bar_all(
                allc, bundle=b, title=f'{tname} {kind}-all ({suffix})',
                right_only=plot_right_only,
            )
        return mvd, allc


def _plot_one_target(session, z, outdir, tname, suffix, model_all,
                     data_cubes=None,
                     at_x=None, at_y=None, save_trace_csv_dir=None, show_pre=True):
    if tname not in training.SPOT_TARGETS:
        raise ValueError(f'unknown plot target {tname!r}')
    kind = _session_trace_kind(session)
    mvd = os.path.join(outdir, f'spot_trained_{kind}.png')
    allc = os.path.join(outdir, f'spot_all_{kind}.png')
    make_bundle, plot_data, plot_all = spot_bundle_fns(session)
    net_tag = _network_spot_tag(session, tname)
    plot_kw = dict(data_cubes=data_cubes)
    b = make_bundle(
        session, z,
        at_x_list=at_x, at_y_list=at_y,
        save_trace_csv_dir=save_trace_csv_dir, show_pre=show_pre,
    )
    from figure.readout import contrast_for_target
    bundles = {contrast_for_target(tname): b}
    plot_data(
        mvd, bundles=bundles, title=f'{tname} {kind}-data ({suffix}){net_tag}', **plot_kw,
    )
    if model_all:
        plot_all(allc, bundles=bundles, title=f'{tname} {kind}-all ({suffix}){net_tag}', **plot_kw)
    return mvd, allc


def plot_param_set(params, outdir, model=None, model_all=True,
                   context_dir=None,
                   plot_targets=None, session=None, *,
                   final_costs=None, cost_curve=None, costs_by_target=None, best_i=None,
                   save_artifacts=True, artifact_fname=None,
                   data_cubes=None,
                   plot_right_only=True, at_x=None, at_y=None,
                   align_at_x=None, align_at_y=None,
                   save_csv=False, show_pre=True):
    os.makedirs(outdir, exist_ok=True)
    data_dir = run_data_dir(os.path.abspath(outdir))
    os.makedirs(data_dir, exist_ok=True)
    save_trace_csv_dir = data_dir if save_csv else None
    ctx = context_dir or outdir
    if model is None and session is not None:
        model = session.model
    if model is None:
        raise ValueError('model or session required')
    if session is None:
        session = load_session(ctx, model)

    params = np.atleast_2d(params)
    if final_costs is None and artifact_fname is not None:
        final_costs, loaded_curve, loaded_by_target, _ = _load_plot_costs(
            outdir, artifact_fname, params.shape[0],
        )
        if cost_curve is None:
            cost_curve = loaded_curve
        if costs_by_target is None:
            costs_by_target = loaded_by_target

    if best_i is None:
        import training.driver as train_mod
        best_i = train_mod.load_best_i(ctx)

    print(f'plot device={_plot_device_label()}')
    best, best_cost, best_i = select_best(
        params, session, final_costs=final_costs, best_i=best_i,
    )
    z = torch.tensor(best, dtype=session.sim_dtype, device=session.device)

    suffix = f'trained, cost {best_cost:.2f}% of data power'
    target_list = list(session.target_list)
    if plot_targets is not None:
        target_list = [t for t in target_list if t in plot_targets]

    spot_targets = [t for t in target_list if t in training.SPOT_TARGETS]
    bar_targets = [t for t in target_list if t in training.MOVING_BAR_TARGETS]
    other_targets = [
        t for t in target_list
        if t not in training.SPOT_TARGETS and t not in training.MOVING_BAR_TARGETS
    ]
    if (at_x is not None or at_y is not None) and not bar_targets and not spot_targets:
        raise SystemExit('--x/--y require a moving_bar or spot target in this run')
    if (align_at_x is not None or align_at_y is not None):
        if align_at_x is None or align_at_y is None:
            raise SystemExit('--align-xy requires X,Y')
        if at_x is None and at_y is None:
            raise SystemExit('--align-xy requires --x and/or --y')
        if not bar_targets:
            raise SystemExit('--align-xy applies to moving_bar slice plots only')

    if cost_curve is not None and len(cost_curve) > 0:
        plot_cost(
            cost_curve, os.path.join(outdir, 'cost_curve.png'),
            costs_by_target=costs_by_target,
            target_order=list(training.session_cost_part_keys(session.target_list)),
        )
    if spot_targets:
        _plot_spot_targets(
            session, z, outdir, spot_targets, suffix, model_all,
            data_cubes=data_cubes,
            at_x=at_x, at_y=at_y,
            save_trace_csv_dir=save_trace_csv_dir, show_pre=show_pre,
        )
    if bar_targets:
        _plot_bar_targets(
            session, z, outdir, bar_targets, suffix, model_all,
            plot_right_only=plot_right_only,
            at_x=at_x, at_y=at_y,
            align_at_x=align_at_x, align_at_y=align_at_y,
            save_trace_csv_dir=save_trace_csv_dir, show_pre=show_pre,
        )
    for tname in other_targets:
        one = session_for_target(session, tname)
        _plot_one_target(
            one, z, outdir, tname, suffix, model_all,
            data_cubes=data_cubes,
            save_trace_csv_dir=save_trace_csv_dir, show_pre=show_pre,
        )

    if save_artifacts:
        import training.driver as train_mod
        os.makedirs(train_mod.data_dir(outdir), exist_ok=True)
        z_best = torch.tensor(best, dtype=session.sim_dtype, device=session.device)
        train_mod.save_best_param_named(outdir, z_best, session)
        train_mod.write_best_i(outdir, best_i)
    print(f'plots saved to {outdir}')
    return best, best_cost


def _load_plot_costs(outdir, fname, n_runs):
    """Load per-run and step costs saved by ``training.driver.save_training_outputs``."""
    import training.driver as train_mod
    return train_mod.load_stored_costs(outdir, fname, n_runs)


def add_plot_arguments(parser):
    """Register plot-only CLI flags shared by ``training.driver`` and ``figure.plot_run``."""
    from training.driver import parse_bool

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
             'Gray data never draws pre.',
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
        help='moving_bar slice plots: align --x/--y traces to ref column hex (x,y); total unchanged',
    )


def plot_kwargs_from_args(args):
    """Map a parsed CLI namespace to :func:`plot_param_set` plot kwargs."""
    align_xy = parse_align_xy(args.align_xy)
    align_at_x, align_at_y = align_xy if align_xy is not None else (None, None)
    return dict(
        plot_right_only=args.plot_right_only,
        show_pre=args.show_pre,
        at_x=parse_axis_slice_list(args.x),
        at_y=parse_axis_slice_list(args.y),
        align_at_x=align_at_x,
        align_at_y=align_at_y,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        'run_path',
        nargs='?',
        default=DEFAULT_RUN_PATH,
        help='run folder under PARAMETER_DIR or absolute path (default: %(default)s)',
    )
    ap.add_argument(
        '--best-i',
        type=int,
        default=None,
        help='force parameter row index (default: data/best_i.txt, else infer from costs)',
    )
    ap.add_argument(
        '--save-csv',
        action='store_true',
        help='save trace CSV files to the run data directory',
    )
    add_plot_arguments(ap)
    args = ap.parse_args()
    try:
        plot_kw = plot_kwargs_from_args(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    outdir = resolve_run_dir(args.run_path)
    params_path, artifact_fname = find_training_params(outdir)
    params = np.load(params_path)
    model = resolve_model(outdir)
    print(f'outdir={outdir}')
    print(f'params={params_path}')
    print(f'model={model} ({params.shape[-1]} params per set)')
    plot_param_set(
        params, outdir, model=model,
        artifact_fname=artifact_fname,
        best_i=args.best_i,
        save_csv=args.save_csv,
        **plot_kw,
    )


if __name__ == '__main__':
    main()
