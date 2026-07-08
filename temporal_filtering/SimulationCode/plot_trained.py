#!/usr/bin/env python
"""Simulation + plotting for the FiveCol medulla model."""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

import FiveCol_MedSim_Pytorch as fc
from plot import moving_bar as moving_bar_plot
from plot import spot as spot_plot
from plot.utils import parse_axis_slice_list, plot_cost
from training_config import PARAMETER_DIR, run_data_dir

TRAIN_OPTS_FILE = fc.TRAIN_OPTS_FILE
KNOWN_MODELS = ('conductance', 'adaptive')
RUN_NAME_MAX = 255


def _plot_device_label():
    dev = fc.active_device()
    if dev == 'cuda' and torch.cuda.is_available():
        import torch as _torch
        return f'cuda ({_torch.cuda.get_device_name(0)})'
    return dev


def _slug(text):
    """Filesystem-safe token for a CLI flag value."""
    return re.sub(r'[^\w.,-]+', '-', str(text)).strip('-')


def _argv_cli_tokens(argv):
    """Drop the script path; yield long-option tokens from *argv*."""
    if argv and argv[0].endswith('.py'):
        argv = argv[1:]
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in ('-h', '--help'):
            i += 1
            continue
        if not tok.startswith('--'):
            i += 1
            continue
        key, sep, val = tok[2:].partition('=')
        if sep:
            yield _slug(key), _slug(val)
            i += 1
        elif i + 1 < len(argv) and not argv[i + 1].startswith('-'):
            yield _slug(key), _slug(argv[i + 1])
            i += 2
        else:
            yield _slug(key), None
            i += 1


def command_run_name(script_stem, argv=None):
    """Build a run folder name from flags on the command line (``sys.argv``)."""
    if argv is None:
        argv = sys.argv[1:]
    prefix = os.environ.get('SLURM_JOB_ID') or time.strftime('%m%d_%H%M%S')
    parts = [prefix, script_stem]
    for key, val in _argv_cli_tokens(argv):
        parts.append(key)
        if val is not None:
            parts.append(val)
    name = '-'.join(parts)
    if len(name) <= RUN_NAME_MAX:
        return name
    return name[:RUN_NAME_MAX].rstrip('-')


def run_dir(model, root=None, parent=None, name=None):
    if parent is None:
        root = str(PARAMETER_DIR) if root is None else root
        parent = os.path.join(root, model)
    if name is None:
        job_id = os.environ.get('SLURM_JOB_ID')
        name = f'run_{job_id}' if job_id else time.strftime('run_%m%d_%H%M%S')
    return os.path.join(parent, name)


def _spot_plot_fn(session):
    if session.backend.network is not None:
        return spot_plot.plot_network_spot
    return spot_plot.plot_borst_spot


def _network_spot_tag(session, tname):
    """Subtitle suffix for network spot plots (shift count from sidecar)."""
    if session.backend.network is None:
        return ''
    opts = (session.train_opts or {}).get(f'{tname}_stimulus_opts') or {}
    shift_extent = int(opts.get('shift_extent', 0))
    shifttag = f"{1 + 3 * shift_extent * (shift_extent + 1)} shifts"
    return f'  [avg over spots x {shifttag} x radii]'


def load_train_opts(outdir):
    opts_path = os.path.join(run_data_dir(os.path.abspath(outdir)), TRAIN_OPTS_FILE)
    if not os.path.isfile(opts_path):
        return None
    with open(opts_path) as f:
        return json.load(f)


def load_session(outdir, model=None, param_modes=None):
    return fc.open_session_from_outdir(
        outdir, model,
        param_modes=param_modes,
    )


def _session_for_target(base_session, tname):
    """Single-target session sharing backend/schema with a multi-target run."""
    opts = dict(base_session.train_opts or {})
    opts['target_list'] = [tname]
    opts['packs'] = None
    if base_session.backend.network is not None:
        opts['network'] = base_session.backend.network
        return fc.open_session({**opts, 'backend': 'network'}, base_session.model,
                               schema=list(base_session.schema))
    return fc.open_session({**opts, 'backend': 'borst'}, base_session.model,
                           schema=list(base_session.schema))


def resolve_model(outdir, override=None):
    if override is not None:
        model = override
    else:
        opts = load_train_opts(outdir)
        if not opts or 'model' not in opts:
            raise SystemExit(
                f'cannot determine model for {outdir!r}; '
                f'expected {TRAIN_OPTS_FILE} with "model": conductance or adaptive'
            )
        model = opts['model']
    if model not in KNOWN_MODELS:
        raise SystemExit(
            f'invalid model {model!r} in {outdir!r}; '
            f'expected conductance or adaptive'
        )
    return model


def resolve_run_dir(path):
    """Resolve a run folder under ``PARAMETER_DIR`` or an absolute path."""
    p = Path(path).expanduser()
    outdir = p.resolve() if p.is_absolute() else (PARAMETER_DIR / p).resolve()
    if not outdir.is_dir():
        raise SystemExit(f'run folder not found: {path!r} -> {outdir}')
    return str(outdir)


def find_training_params(outdir):
    """``training*_table.csv`` stem → ``data/<stem>.npy`` (train.py artifact layout)."""
    import train as train_mod

    tables = sorted(Path(outdir).glob('training*_table.csv'))
    if len(tables) != 1:
        raise SystemExit(
            f'expected exactly one training*_table.csv in {outdir!r}, found {len(tables)}',
        )
    fname = tables[0].name.replace('_table.csv', '') + '.npy'
    params_path = train_mod.params_path(outdir, fname)
    if not os.path.isfile(params_path):
        raise SystemExit(f'missing training params: {params_path!r}')
    return params_path, fname


def select_best(params, session, *, final_costs=None, best_i=None):
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
            best_cost = fc.calc_cost(z, session).item()
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
        print(f'{len(valid)} trained set(s); costs min={valid_costs.min():.4f} '
              f'max={valid_costs.max():.4f}; selected #{run_i} (from saved final costs)')
        return valid[best], float(costs_arr[run_i]), run_i

    costs_out = []
    for row in valid:
        z = torch.tensor(row, dtype=session.sim_dtype, device=session.device)
        costs_out.append(fc.calc_cost(z, session).item())
    costs_out = np.array(costs_out)
    best = int(np.argmin(costs_out))
    run_i = int(valid_idx[best])
    print(f'{len(valid)} trained set(s); costs min={costs_out.min():.4f} '
          f'max={costs_out.max():.4f}; selected #{run_i}')
    return valid[best], costs_out[best], run_i


def _plot_spot_targets(session, z, outdir, spot_targets, suffix, model_all,
                       ref_cubes=None, ref_cubes_off=None, mvd_group_list=None,
                       at_x=None, at_y=None):
    """Plot spot target(s); on+off combined in one figure when both are trained."""
    spot_set = set(spot_targets)
    plot_fn = _spot_plot_fn(session)
    ref_t = 'spot_bright' if 'spot_bright' in spot_set else spot_targets[0]
    net_tag = _network_spot_tag(session, ref_t)
    plot_kw = dict(
        ref_cubes=ref_cubes, ref_cubes_off=ref_cubes_off,
        group_list=mvd_group_list,
    )
    slice_kw = dict(at_x_list=at_x, at_y_list=at_y)
    if spot_set == set(fc.SPOT_TARGETS):
        s_on = _session_for_target(session, 'spot_bright')
        s_off = _session_for_target(session, 'spot_dark')
        mvd = os.path.join(outdir, 'model_data_spot.png')
        plot_fn(
            s_on, z, mvd, session_off=s_off,
            title=f'Spot model-data ({suffix}){net_tag}',
            **plot_kw,
        )
        allc = None
        if model_all:
            allc = os.path.join(outdir, 'model_all_spot.png')
            plot_fn(
                s_on, z, allc, session_off=s_off, all_cells=True,
                title=f'Spot model-all ({suffix}){net_tag}',
                **plot_kw, **slice_kw,
            )
        return mvd, allc
    for tname in spot_targets:
        _plot_one_target(
            _session_for_target(session, tname), z, outdir, tname, suffix, model_all,
            ref_cubes=ref_cubes, ref_cubes_off=ref_cubes_off,
            mvd_group_list=mvd_group_list,
            at_x=at_x, at_y=at_y,
        )


def _plot_bar_targets(session, z, outdir, bar_targets, suffix, model_all, *,
                      plot_right_only=True, at_x=None, at_y=None):
    """Plot moving-bar target(s); bright left | dark right when both are trained."""
    slice_kw = dict(at_x_list=at_x, at_y_list=at_y)
    bar_set = set(bar_targets)
    if bar_set == set(fc.MOVING_BAR_TARGETS):
        s_bright = _session_for_target(session, 'moving_bar_bright')
        s_dark = _session_for_target(session, 'moving_bar_dark')
        bundle_b = moving_bar_plot.moving_bar_trace_bundle(
            s_bright, z, 'moving_bar_bright', **slice_kw,
        )
        bundle_d = moving_bar_plot.moving_bar_trace_bundle(
            s_dark, z, 'moving_bar_dark', **slice_kw,
        )
        mvd = os.path.join(outdir, 'model_data_bar.png')
        moving_bar_plot.plot_moving_bar_data(
            s_bright, z, mvd, 'moving_bar_bright', session_off=s_dark,
            title=f'Moving-bar model-data ({suffix})',
            bundle=bundle_b, bundle_off=bundle_d,
        )
        allc = None
        if model_all:
            allc = os.path.join(outdir, 'model_all_bar.png')
            moving_bar_plot.plot_moving_bar_all(
                s_bright, z, allc, 'moving_bar_bright', session_off=s_dark,
                title=f'Moving-bar model-all ({suffix})',
                right_only=plot_right_only,
                bundle=bundle_b, bundle_off=bundle_d,
            )
        return mvd, allc
    for tname in bar_targets:
        one = _session_for_target(session, tname)
        bundle = moving_bar_plot.moving_bar_trace_bundle(one, z, tname, **slice_kw)
        mvd = os.path.join(outdir, 'model_data_bar.png')
        moving_bar_plot.plot_moving_bar_data(
            one, z, mvd, tname, title=f'{tname} model-data ({suffix})', bundle=bundle,
        )
        allc = None
        if model_all:
            allc = os.path.join(outdir, 'model_all_bar.png')
            moving_bar_plot.plot_moving_bar_all(
                one, z, allc, tname, title=f'{tname} model-all ({suffix})',
                right_only=plot_right_only, bundle=bundle,
            )
        return mvd, allc


def _plot_one_target(session, z, outdir, tname, suffix, model_all,
                     ref_cubes=None, ref_cubes_off=None, mvd_group_list=None,
                     at_x=None, at_y=None):
    if tname not in fc.SPOT_TARGETS:
        raise ValueError(f'unknown plot target {tname!r}')
    mvd = os.path.join(outdir, 'model_data_spot.png')
    allc = os.path.join(outdir, 'model_all_spot.png')
    plot_fn = _spot_plot_fn(session)
    net_tag = _network_spot_tag(session, tname)
    plot_kw = dict(
        ref_cubes=ref_cubes, ref_cubes_off=ref_cubes_off,
        group_list=mvd_group_list,
    )
    plot_fn(session, z, mvd, title=f'{tname} model-data ({suffix}){net_tag}', **plot_kw)
    if model_all:
        plot_fn(session, z, allc, all_cells=True,
                title=f'{tname} model-all ({suffix}){net_tag}',
                at_x_list=at_x, at_y_list=at_y, **plot_kw)
    return mvd, allc


def plot_param_set(params, outdir, model=None, model_all=True,
                   context_dir=None, param_modes=None,
                   plot_targets=None, session=None, *,
                   final_costs=None, cost_curve=None, costs_by_target=None, best_i=None,
                   save_artifacts=True, artifact_fname=None,
                   ref_cubes=None, ref_cubes_off=None, mvd_group_list=None,
                   plot_right_only=True, at_x=None, at_y=None,
                   plot_vm=False):
    os.makedirs(outdir, exist_ok=True)
    ctx = context_dir or outdir
    if model is None and session is not None:
        model = session.model
    if model is None:
        raise ValueError('model or session required')
    if session is None:
        session = load_session(ctx, model, param_modes=param_modes)

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
        import train as train_mod
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

    spot_targets = [t for t in target_list if t in fc.SPOT_TARGETS]
    bar_targets = [t for t in target_list if t in fc.MOVING_BAR_TARGETS]
    other_targets = [
        t for t in target_list
        if t not in fc.SPOT_TARGETS and t not in fc.MOVING_BAR_TARGETS
    ]
    if (at_x is not None or at_y is not None) and not bar_targets and not spot_targets:
        raise SystemExit('--x/--y require a moving_bar or spot target in this run')

    if plot_vm:
        if session.model != 'conductance':
            raise SystemExit('--Vm requires model conductance')
        if other_targets:
            raise SystemExit(f'--Vm does not support plot targets: {other_targets}')
        if bar_targets:
            bar_set = set(bar_targets)
            if bar_set == set(fc.MOVING_BAR_TARGETS):
                s_bright = _session_for_target(session, 'moving_bar_bright')
                s_dark = _session_for_target(session, 'moving_bar_dark')
                bundle_b = moving_bar_plot.moving_bar_trace_bundle(
                    s_bright, z, 'moving_bar_bright',
                    at_x_list=at_x, at_y_list=at_y,
                    trace_kind='vm',
                )
                bundle_d = moving_bar_plot.moving_bar_trace_bundle(
                    s_dark, z, 'moving_bar_dark',
                    at_x_list=at_x, at_y_list=at_y,
                    trace_kind='vm',
                )
                allc = os.path.join(outdir, 'model_all_bar_vm.png')
                moving_bar_plot.plot_moving_bar_all(
                    s_bright, z, allc, 'moving_bar_bright', session_off=s_dark,
                    title=f'Moving-bar Vm-all ({suffix})',
                    right_only=plot_right_only,
                    trace_kind='vm',
                    bundle=bundle_b, bundle_off=bundle_d,
                )
            else:
                tname = bar_targets[0]
                one = _session_for_target(session, tname)
                allc = os.path.join(outdir, 'model_all_bar_vm.png')
                moving_bar_plot.plot_moving_bar_all(
                    one, z, allc, tname,
                    title=f'{tname} Vm-all ({suffix})',
                    right_only=plot_right_only,
                    at_x_list=at_x, at_y_list=at_y,
                    trace_kind='vm',
                )
        if spot_targets:
            spot_set = set(spot_targets)
            plot_fn = _spot_plot_fn(session)
            ref_t = 'spot_bright' if 'spot_bright' in spot_set else spot_targets[0]
            net_tag = _network_spot_tag(session, ref_t)
            plot_kw = dict(
                ref_cubes=ref_cubes, ref_cubes_off=ref_cubes_off,
                group_list=mvd_group_list,
            )
            if spot_set == set(fc.SPOT_TARGETS):
                s_on = _session_for_target(session, 'spot_bright')
                s_off = _session_for_target(session, 'spot_dark')
                allc = os.path.join(outdir, 'model_all_spot_vm.png')
                plot_fn(
                    s_on, z, allc, session_off=s_off, all_cells=True,
                    title=f'Spot Vm-all ({suffix}){net_tag}',
                    trace_kind='vm',
                    at_x_list=at_x, at_y_list=at_y,
                    **plot_kw,
                )
            else:
                tname = spot_targets[0]
                one = _session_for_target(session, tname)
                allc = os.path.join(outdir, 'model_all_spot_vm.png')
                plot_fn(
                    one, z, allc, all_cells=True,
                    title=f'{tname} Vm-all ({suffix}){net_tag}',
                    trace_kind='vm',
                    at_x_list=at_x, at_y_list=at_y,
                    **plot_kw,
                )
        print(f'plots saved to {outdir}')
        return best, best_cost

    if cost_curve is not None and len(cost_curve) > 0:
        plot_cost(
            cost_curve, os.path.join(outdir, 'cost_curve.png'),
            costs_by_target=costs_by_target,
            target_order=list(fc.session_cost_part_keys(session.target_list)),
        )
    if spot_targets:
        _plot_spot_targets(
            session, z, outdir, spot_targets, suffix, model_all,
            ref_cubes=ref_cubes, ref_cubes_off=ref_cubes_off,
            mvd_group_list=mvd_group_list,
            at_x=at_x, at_y=at_y,
        )
    if bar_targets:
        _plot_bar_targets(
            session, z, outdir, bar_targets, suffix, model_all,
            plot_right_only=plot_right_only,
            at_x=at_x, at_y=at_y,
        )
    for tname in other_targets:
        one = _session_for_target(session, tname)
        _plot_one_target(
            one, z, outdir, tname, suffix, model_all,
            ref_cubes=ref_cubes, ref_cubes_off=ref_cubes_off,
            mvd_group_list=mvd_group_list,
        )

    if save_artifacts:
        import train as train_mod
        os.makedirs(train_mod.data_dir(outdir), exist_ok=True)
        np.save(train_mod.best_param_path(outdir), best)
        train_mod.write_best_i(outdir, best_i)
    print(f'plots saved to {outdir}')
    return best, best_cost


def _load_plot_costs(outdir, fname, n_runs):
    """Load per-run and step costs saved by ``train.save_training_outputs``."""
    import train as train_mod
    return train_mod.load_stored_costs(outdir, fname, n_runs)


def add_plot_arguments(parser):
    """Register plot-only CLI flags shared by train.py and plot_trained.py."""
    from train import parse_bool

    parser.add_argument(
        '--Vm',
        nargs='?',
        const=True,
        default=False,
        type=parse_bool,
        metavar='BOOL',
        help='only write model_all_bar_vm.png / model_all_spot_vm.png (Vm−Vm_ref); '
             'skip other plots',
    )
    parser.add_argument(
        '--plot-right-only',
        nargs='?',
        const=True,
        default=True,
        type=parse_bool,
        metavar='BOOL',
        help='model_all_bar: right-direction specs only (default true); '
             'pass false for all directions',
    )
    parser.add_argument(
        '--x',
        default=None,
        metavar='X,...',
        help='model_all_{bar,spot}{,_vm}: comma-separated x slices; with --y, one trace per (x,y) pair',
    )
    parser.add_argument(
        '--y',
        default=None,
        metavar='Y,...',
        help='model_all_{bar,spot}{,_vm}: comma-separated y slices; with --x, one trace per (x,y) pair',
    )


def plot_kwargs_from_args(args):
    """Map a parsed CLI namespace to :func:`plot_param_set` plot kwargs."""
    return dict(
        plot_vm=args.Vm,
        plot_right_only=args.plot_right_only,
        at_x=parse_axis_slice_list(args.x),
        at_y=parse_axis_slice_list(args.y),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run_path', help='run folder under PARAMETER_DIR or absolute path')
    ap.add_argument(
        '--best-i',
        type=int,
        default=None,
        help='force parameter row index (default: data/best_i.txt, else infer from costs)',
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
        **plot_kw,
    )


if __name__ == '__main__':
    main()
