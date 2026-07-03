#!/usr/bin/env python
"""Simulation + plotting for the FiveCol medulla model."""
import json
import os
import sys
import time

import numpy as np
import torch

import FiveCol_MedSim_Pytorch as fc
from plot import moving_bar as moving_bar_plot
from plot import tile as tile_plot
from training_config import PARAMETER_DIR

CELL_LIST = tile_plot.CELL_LIST
CENTER_COL = tile_plot.CENTER_COL
CTYPE = tile_plot.CTYPE
CENTER_NEURON_OFFSET = tile_plot.CENTER_NEURON_OFFSET

TRAIN_OPTS_FILE = fc.TRAIN_OPTS_FILE
MODEL_TYPE_FILE = 'model_type.txt'
KNOWN_MODEL_TYPES = ('conductance', 'adaptive')


def _plot_device_label():
    dev = fc.active_device()
    if dev == 'cuda' and torch.cuda.is_available():
        import torch as _torch
        return f'cuda ({_torch.cuda.get_device_name(0)})'
    return dev


default_ref_cubes = tile_plot.default_ref_cubes
reference_cube = tile_plot.reference_cube
mvd_groups = tile_plot.mvd_groups


def run_dir(model_type, root=None, parent=None):
    if parent is None:
        root = str(PARAMETER_DIR) if root is None else root
        parent = os.path.join(root, model_type)
    job_id = os.environ.get('SLURM_JOB_ID')
    name = f'run_{job_id}' if job_id else time.strftime('run_%Y%m%d_%H%M%S')
    outdir = os.path.join(parent, name)
    os.makedirs(outdir, exist_ok=True)
    return outdir


plot_cost = tile_plot.plot_cost
plot_model_data = tile_plot.plot_model_data
plot_model_all = tile_plot.plot_model_all
plot_model_data_network = tile_plot.plot_model_data_network
plot_model_data_moving_bar = moving_bar_plot.plot_model_data_moving_bar
plot_model_all_moving_bar = moving_bar_plot.plot_model_all_moving_bar


def load_train_opts(outdir):
    opts_path = os.path.join(os.path.abspath(outdir), TRAIN_OPTS_FILE)
    if not os.path.isfile(opts_path):
        return None
    with open(opts_path) as f:
        return json.load(f)


def load_session(outdir, model_type, param_modes=None, param_fixes=None, per_type=None):
    return fc.open_session_from_outdir(
        outdir, model_type,
        param_modes=param_modes, param_fixes=param_fixes, per_type=per_type,
    )


def _session_for_target(base_session, tname):
    """Single-target session sharing backend/schema with a multi-target run."""
    opts = dict(base_session.train_opts or {})
    opts['target_list'] = [tname]
    opts['packs'] = None
    if base_session.backend.network is not None:
        opts['network'] = base_session.backend.network
        return fc.open_session({**opts, 'backend': 'network'}, base_session.model_type,
                               schema=list(base_session.schema))
    return fc.open_session({**opts, 'backend': 'borst'}, base_session.model_type,
                           schema=list(base_session.schema))


def _model_type_from_sidecar(params_path):
    side = os.path.join(os.path.dirname(os.path.abspath(params_path)), MODEL_TYPE_FILE)
    if os.path.exists(side):
        with open(side) as f:
            return f.read().strip()
    return None


def _model_type_from_path(params_path):
    parts = os.path.abspath(params_path).split(os.sep)
    for mt in KNOWN_MODEL_TYPES:
        if mt in parts:
            return mt
    return None


def resolve_model_type(params_path, override=None):
    model_type = (override
                  or _model_type_from_sidecar(params_path)
                  or _model_type_from_path(params_path))
    if model_type not in KNOWN_MODEL_TYPES:
        raise SystemExit(
            'cannot determine model_type for '
            f'{params_path!r}; pass it explicitly, e.g.\n'
            '  python plot_trained.py params.npy outdir <conductance|adaptive>'
        )
    return model_type


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
            z = torch.tensor(valid[loc], dtype=torch.float64, device=session.device)
            best_cost = fc.calc_cost(z, session).item()
        print(f'{len(valid)} trained set(s); selected #{best_i} (cost={best_cost:.4f})')
        return valid[loc], best_cost

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
        return valid[best], float(costs_arr[run_i])

    costs_out = []
    for row in valid:
        z = torch.tensor(row, dtype=torch.float64, device=session.device)
        costs_out.append(fc.calc_cost(z, session).item())
    costs_out = np.array(costs_out)
    best = int(np.argmin(costs_out))
    print(f'{len(valid)} trained set(s); costs min={costs_out.min():.4f} '
          f'max={costs_out.max():.4f}; selected #{best}')
    return valid[best], costs_out[best]


def _plot_one_target(session, z, outdir, tname, suffix, model_all):
    if tname == 'moving_bar':
        mvd = os.path.join(outdir, 'model_data_bar.png')
        allc = os.path.join(outdir, 'model_all_bar.png')
        plot_model_data_moving_bar(session, z, mvd, title=f'Moving-bar model-data ({suffix})')
        if model_all:
            plot_model_all_moving_bar(session, z, allc, title=f'Moving-bar model-all ({suffix})')
        return mvd, allc
    mvd = os.path.join(outdir, 'model_data_tile.png')
    allc = os.path.join(outdir, 'model_all_tile.png')
    if session.backend.network is not None:
        plot_model_data_network(session, z, mvd, title=f'Network model-data ({suffix})')
        if model_all:
            plot_model_data_network(session, z, allc, title=f'Network model-all ({suffix})')
    else:
        plot_model_data(session, z, mvd, title=f'Model data ({suffix})')
        if model_all:
            plot_model_all(session, z, allc, title=f'Model-all 65 cells ({suffix})')
    return mvd, allc


def plot_param_set(params, outdir, model_type=None, model_all=True,
                   context_dir=None, param_modes=None, param_fixes=None,
                   plot_targets=None, session=None, *,
                   final_costs=None, cost_curve=None, best_i=None,
                   save_artifacts=True, artifact_fname=None):
    os.makedirs(outdir, exist_ok=True)
    ctx = context_dir or outdir
    if model_type is None and session is not None:
        model_type = session.model_type
    if model_type is None:
        raise ValueError('model_type or session required')
    if session is None:
        session = load_session(ctx, model_type, param_modes=param_modes, param_fixes=param_fixes)

    params = np.atleast_2d(params)
    if final_costs is None and artifact_fname is not None:
        final_costs, loaded_curve = _load_plot_costs(outdir, artifact_fname, params.shape[0])
        if cost_curve is None:
            cost_curve = loaded_curve

    print(f'plot device={_plot_device_label()}')
    best, best_cost = select_best(
        params, session, final_costs=final_costs, best_i=best_i,
    )
    z = torch.tensor(best, dtype=torch.float64, device=session.device)

    if cost_curve is not None and len(cost_curve) > 0:
        plot_cost(cost_curve, os.path.join(outdir, 'cost_curve.png'))

    suffix = f'trained, cost {best_cost:.2f}% of data power'
    target_list = list(session.target_list)
    if plot_targets is not None:
        target_list = [t for t in target_list if t in plot_targets]

    for tname in target_list:
        one = _session_for_target(session, tname)
        _plot_one_target(one, z, outdir, tname, suffix, model_all)

    if save_artifacts:
        np.save(os.path.join(outdir, 'best_param.npy'), best)
    print(f'plots saved to {outdir}')
    return best, best_cost


def _load_plot_costs(outdir, fname, n_runs):
    """Load per-run and step costs saved by ``run.save_training_outputs``."""
    import run as run_mod
    return run_mod.load_stored_costs(outdir, fname, n_runs)


def main():
    params_path = sys.argv[1] if len(sys.argv) > 1 else str(PARAMETER_DIR / 'training_with_Ih.npy')
    outdir = sys.argv[2] if len(sys.argv) > 2 else str(PARAMETER_DIR / 'gpu_test')
    override = sys.argv[3] if len(sys.argv) > 3 else None

    params = np.load(params_path)
    model_type = resolve_model_type(params_path, override)
    print(f'model_type={model_type} ({params.shape[-1]} params per set)')
    plot_param_set(
        params, outdir, model_type=model_type,
        artifact_fname=os.path.basename(params_path),
    )


if __name__ == '__main__':
    main()
