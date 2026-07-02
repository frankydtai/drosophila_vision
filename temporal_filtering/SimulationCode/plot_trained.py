#!/usr/bin/env python
"""Simulation + plotting for the FiveCol medulla model.

This module owns all the model-trace simulation and the plotting routines
(cost curve, model-data, model-all). It can also be run as a script to
visualise a trained parameter set with target-specific output filenames:

    python plot_trained.py [params.npy] [outdir] [model_type]

Accepts either a single param vector (P,) or a stack (N, P); for a stack the
lowest-cost set is selected. The model type is NOT guessed from the parameter
count: it is recorded next to the params at save time (model_type.txt) and read
back here, so it works for any parameter count. Resolution priority is
explicit model_type arg > sidecar model_type.txt > run-dir path name.
"""
import json
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

import Medulla_Library as ml
import blindschleiche_py3 as bs
import FiveCol_MedSim_Pytorch as fc
from plot import moving_bar as moving_bar_plot
from plot import tile as tile_plot
from t4_t5_preference import READOUT_SUBTYPES, active_stimuli_for_subtype, normalize_side, fig1_key_for_stimulus
from training_config import COST_HALF_WINDOW_STEPS, COST_WINDOW_STEPS
from visual_stimulus.moving_bar_stimulus import gruntman_moving_bar_specs, column_bar_center_step
from FiveCol_MedSim_Pytorch import (
    calc_cost,
    data,
    device,
    maxtime,
    mc_cell_index,
    nofcells,
    t_on,
)

CELL_LIST = tile_plot.CELL_LIST
CENTER_COL = tile_plot.CENTER_COL
CTYPE = tile_plot.CTYPE
FIT_INDEX = {name: i for i, name in enumerate(CELL_LIST)}
CENTER_NEURON_OFFSET = tile_plot.CENTER_NEURON_OFFSET

# 512-panel moving-bar grid: dpi=150 savefig was ~317s on CPU (measured).
MOVING_BAR_GRID_DPI = 100
MOVING_BAR_MVD_DPI = 100
_MOVING_BAR_T = np.arange(COST_WINDOW_STEPS)


def _plot_device_label():
    """Human-readable active torch device (follows ``fc.device`` / CUDA availability)."""
    if device == 'cuda' and torch.cuda.is_available():
        return f'cuda ({torch.cuda.get_device_name(0)})'
    return 'cpu'


# --- generic, test-driven plotting hooks (this module hardcodes no cell beyond
#     the default fit cells) --------------------------------------------------
# REF_CUBES: name -> (9, maxtime) grey "data" reference cube drawn behind the model.
#   Defaults to the measured RF data for the 13 fit cells. A test script may
#   extend/override it (e.g. map R1-8 to L1's cube) so the plots follow whatever
#   the experiment configured -- without editing this file.
# MVD_GROUPS: ORDERED list of cell-name groups for model_data. Each group is
#   drawn as its own row-pair (RF row + time row), assigned top-to-bottom in
#   order; EMPTY groups are skipped, so removing a group shifts the rest up (e.g.
#   with no R group, L becomes the top row-pair). Columns within a group are
#   auto-centred. None -> DEFAULT_MVD_GROUPS. A test may prepend/append groups
#   (e.g. an R1-8 group on top) -- this module hardcodes no R cells.
default_ref_cubes = tile_plot.default_ref_cubes
reference_cube = tile_plot.reference_cube
mvd_groups = tile_plot.mvd_groups


def run_dir(model_type, root='FiveCol_Parameter', parent=None):
    """Fresh output folder for one run, shared by all drivers.

    parent/run_<id>/ where parent defaults to <root>/<model_type>/ and <id> is
    the SLURM job id (under SLURM) or a timestamp otherwise.
    """
    if parent is None:
        parent = os.path.join(root, model_type)
    job_id = os.environ.get('SLURM_JOB_ID')
    name = f'run_{job_id}' if job_id else time.strftime('run_%Y%m%d_%H%M%S')
    outdir = os.path.join(parent, name)
    os.makedirs(outdir, exist_ok=True)
    return outdir


_out_scale_vec = tile_plot._out_scale_vec
_as_index = tile_plot._as_index
_pack_filtered = tile_plot._pack_filtered
_simulate_filtered_traces = tile_plot._simulate_filtered_traces
_simulate_filtered_traces_adaptive = tile_plot._simulate_filtered_traces_adaptive
_simulate = tile_plot._simulate

calc_model_trace = tile_plot.calc_model_trace
calc_center_column_trace = tile_plot.calc_center_column_trace


calc_model_trace = tile_plot.calc_model_trace
calc_center_column_trace = tile_plot.calc_center_column_trace
calc_model_full_all = tile_plot.calc_model_full_all


_scale_curve = tile_plot._scale_curve
_style_time_axis = tile_plot._style_time_axis
_style_azimuth_axis = tile_plot._style_azimuth_axis
_annotate_baseline = tile_plot._annotate_baseline


_MODEL_DATA_NAME = "plot_model_" + "v" + "s" + "_data"
_MODEL_DATA_NETWORK_NAME = _MODEL_DATA_NAME + "_network"
plot_model_data_network = getattr(tile_plot, _MODEL_DATA_NETWORK_NAME)
plot_model_data = getattr(tile_plot, _MODEL_DATA_NAME)
plot_all_celltypes = tile_plot.plot_all_celltypes
plot_cost = tile_plot.plot_cost


TARGET_KIND_FILE = 'target_kind.txt'
NETWORK_PATH_FILE = 'network_path.txt'
NETWORK_TRAIN_OPTS_FILE = 'network_train_opts.json'


def restore_fc_context(outdir):
    """Reload network + training target from run sidecars (offline replot)."""
    outdir = os.path.abspath(outdir)
    target_path = os.path.join(outdir, TARGET_KIND_FILE)
    net_path = os.path.join(outdir, NETWORK_PATH_FILE)
    if not os.path.exists(target_path) or not os.path.exists(net_path):
        return False
    with open(target_path) as f:
        target = f.read().strip()
    target_list = None
    if "," in target:
        target_list = [t.strip() for t in target.split(",") if t.strip()]
        target = "tile"  # ignored when target_list is set
    with open(net_path) as f:
        network_json = f.read().strip()
    multi_column, share_edges, sequential = True, False, None
    moving_bar_center_column = False
    opts_path = os.path.join(outdir, NETWORK_TRAIN_OPTS_FILE)
    if os.path.exists(opts_path):
        with open(opts_path) as f:
            opts = json.load(f)
        multi_column = bool(opts.get('multi_column', multi_column))
        share_edges = bool(opts.get('share_edges', share_edges))
        moving_bar_center_column = bool(opts.get('moving_bar_center_column', False))
        loss_weights = opts.get("loss_weights", None)
        tile_center_column = bool(
            opts.get("tile_center_column", opts.get("radial_center_column", False))
        )
        if 'sequential' in opts:
            sequential = bool(opts['sequential'])
    fc.use_network(
        network_json,
        multi_column=multi_column,
        share_edges=share_edges,
        sequential=sequential,
        dev=device,
        target=target,
        moving_bar_center_column=moving_bar_center_column,
        target_list=target_list,
        loss_weights=loss_weights,
        tile_center_column=tile_center_column,
    )
    return True


_MOVING_BAR_T = moving_bar_plot._MOVING_BAR_T
_moving_bar_mean_traces = moving_bar_plot._moving_bar_mean_traces
_style_moving_bar_time_axis = moving_bar_plot._style_moving_bar_time_axis
_plot_moving_bar_cell = moving_bar_plot._plot_moving_bar_cell


def plot_model_data_moving_bar(z, path, title=None):
    fn_name = _MODEL_DATA_NAME + "_moving_bar"
    return getattr(moving_bar_plot, fn_name)(z, path, title=title)


def plot_all_celltypes_moving_bar(z, path, title=None):
    return moving_bar_plot.plot_all_celltypes_moving_bar(z, path, title=title)


def plot_model_data_network(z, path, title=None):
    return getattr(tile_plot, _MODEL_DATA_NETWORK_NAME)(z, path, title=title)


plot_cost = tile_plot.plot_cost
plot_model_data = getattr(tile_plot, _MODEL_DATA_NAME)
plot_all_celltypes = tile_plot.plot_all_celltypes


MODEL_TYPE_FILE = 'model_type.txt'
KNOWN_MODEL_TYPES = ('conductance', 'adaptive')


def _model_type_from_sidecar(params_path):
    """model_type recorded next to the params at save time (the source of truth)."""
    side = os.path.join(os.path.dirname(os.path.abspath(params_path)), MODEL_TYPE_FILE)
    if os.path.exists(side):
        with open(side) as f:
            return f.read().strip()
    return None


def _model_type_from_path(params_path):
    """Fallback: the run dir is FiveCol_Parameter/<model_type>/run_<id>/..."""
    parts = os.path.abspath(params_path).split(os.sep)
    for mt in KNOWN_MODEL_TYPES:
        if mt in parts:
            return mt
    return None


def resolve_model_type(params_path, override=None):
    """Determine the model type without ever guessing from parameter count.

    Priority: explicit override > sidecar model_type.txt > run-dir path name.
    The model is known when the params are produced, so it is recorded then and
    simply read back here; this works for any parameter count.
    """
    model_type = (override
                  or _model_type_from_sidecar(params_path)
                  or _model_type_from_path(params_path))
    if model_type not in KNOWN_MODEL_TYPES:
        raise SystemExit(
            'cannot determine model_type for '
            f'{params_path!r}; pass it explicitly, e.g.\n'
            '  python plot_trained.py params.npy outdir <conductance|adaptive>'
        )
    fc.MODEL_TYPE = model_type
    return model_type


def select_best(params, costs=None):
    params = np.atleast_2d(params)
    valid_mask = np.any(params != 0, axis=1)
    valid = params[valid_mask]
    if len(valid) == 0:
        raise SystemExit('no trained parameter sets found (file all zeros)')
    if costs is not None:
        # ``costs`` can be either:
        #   (a) one scalar per parameter set (shape: [n_runs])
        #   (b) per-step curves (shape: [n_runs, n_steps] or [n_steps] for 1 run)
        # Only use it directly when we can safely map one value to one run.
        costs_arr = np.asarray(costs, dtype=np.float64)
        if costs_arr.ndim == 1 and costs_arr.shape[0] == params.shape[0]:
            per_run_costs = costs_arr
        elif costs_arr.ndim >= 2 and costs_arr.shape[0] == params.shape[0]:
            per_run_costs = costs_arr[..., -1]
        else:
            per_run_costs = None

        if per_run_costs is not None:
            valid_costs = np.asarray(per_run_costs, dtype=np.float64)[valid_mask]
            best = int(np.argmin(valid_costs))
            print(f'{len(valid)} trained set(s); costs min={valid_costs.min():.4f} '
                  f'max={valid_costs.max():.4f}; selected #{best} (from saved costs)')
            return valid[best], float(valid_costs[best])
    costs_out = []
    for row in valid:
        z = torch.tensor(row, dtype=torch.float64, device=device)
        costs_out.append(fc.calc_cost(z, fc.data).item())
    costs_out = np.array(costs_out)
    best = int(np.argmin(costs_out))
    print(f'{len(valid)} trained set(s); costs min={costs_out.min():.4f} '
          f'max={costs_out.max():.4f}; selected #{best}')
    return valid[best], costs_out[best]


def plot_param_set(params, outdir, costs=None, model_type=None, all_cells=False):
    """Select the best param set and write plots into outdir.

    ``all_cells=False`` (default) skips the 512-panel moving-bar grid
    (``model_all_bar.png``), which dominated plot time (~5 min CPU savefig).
    The 64-panel model-data figure is always written.
    """
    os.makedirs(outdir, exist_ok=True)
    if model_type is not None:
        fc.MODEL_TYPE = model_type

    print(f'plot device={_plot_device_label()}')
    # For network runs we must restore fc context BEFORE evaluating costs.
    if getattr(fc, 'NETWORK', None) is None:
        restore_fc_context(outdir)

    best, best_cost = select_best(params, costs=costs)
    z = torch.tensor(best, dtype=torch.float64, device=device)

    if costs is not None:
        plot_cost(costs, os.path.join(outdir, 'cost_curve.png'))

    suffix = f'trained, cost {best_cost:.2f}% of data power'
    # New naming: tile and moving-bar variants.
    mvd = os.path.join(outdir, 'model_data_tile.png')
    allc = os.path.join(outdir, 'model_all_tile.png')
    if getattr(fc, 'NETWORK', None) is not None:
        opts = getattr(fc, "NETWORK_TRAIN_OPTS", None) or {}
        target_list = opts.get("target_list")
        if not target_list:
            target_raw = str(opts.get("target", getattr(fc, "TARGET_KIND", "tile")))
            target_list = [t.strip() for t in target_raw.split(",") if t.strip()] or ["tile"]

        if len(target_list) > 1:
            # Re-enter each target context so each plot is generated from its own
            # signal/readout pack. (cost selection above stays combined.)
            network_json = opts.get("network_json", None)
            multi_column = bool(opts.get("multi_column", False))
            share_edges = bool(opts.get("share_edges", False))
            sequential = bool(opts.get("sequential", False))
            moving_bar_center_column = bool(opts.get("moving_bar_center_column", False))
            tile_center_column = bool(opts.get("tile_center_column", False))
            if network_json:
                for tname in target_list:
                    fc.use_network(
                        network_json,
                        multi_column=multi_column,
                        share_edges=share_edges,
                        sequential=sequential,
                        dev=device,
                        target=tname,
                        moving_bar_center_column=moving_bar_center_column,
                        tile_center_column=tile_center_column,
                    )
                    if tname == "moving_bar":
                        plot_model_data_moving_bar(
                            z,
                            os.path.join(outdir, "model_data_bar.png"),
                            title=f'Moving-bar model-data ({suffix})',
                        )
                        plot_all_celltypes_moving_bar(
                            z,
                            os.path.join(outdir, "model_all_bar.png"),
                            title=f'Moving-bar model-all ({suffix})',
                        )
                    else:
                        plot_model_data_network(
                            z,
                            os.path.join(outdir, "model_data_tile.png"),
                            title=f'Network model-data ({suffix})',
                        )
                        plot_model_data_network(
                            z,
                            os.path.join(outdir, "model_all_tile.png"),
                            title=f'Network model-all ({suffix})',
                        )
                first = target_list[0]
                if first == "moving_bar":
                    mvd = os.path.join(outdir, "model_data_bar.png")
                    allc = os.path.join(outdir, "model_all_bar.png")
                else:
                    mvd = os.path.join(outdir, "model_data_tile.png")
                    allc = os.path.join(outdir, "model_all_tile.png")
            else:
                # Fallback: unknown network context, use current state.
                target_list = [getattr(fc, "TARGET_KIND", "tile")]

        if len(target_list) == 1:
            if target_list[0] == 'moving_bar':
                mvd = os.path.join(outdir, 'model_data_bar.png')
                allc = os.path.join(outdir, 'model_all_bar.png')
                plot_model_data_moving_bar(
                    z, mvd, title=f'Moving-bar model-data ({suffix})',
                )
                plot_all_celltypes_moving_bar(
                    z, allc, title=f'Moving-bar model-all ({suffix})',
                )
            else:
                mvd = os.path.join(outdir, 'model_data_tile.png')
                allc = os.path.join(outdir, 'model_all_tile.png')
                plot_model_data_network(
                    z, mvd, title=f'Network model-data ({suffix})',
                )
                plot_model_data_network(
                    z, allc, title=f'Network model-all ({suffix})',
                )
    else:
        mvd = os.path.join(outdir, 'model_data_tile.png')
        allc = os.path.join(outdir, 'model_all_tile.png')
        plot_model_data(z, mvd, title=f'Model data ({suffix})')
        plot_all_celltypes(z, allc, title=f'Model-all 65 cell types ({suffix})')

    np.save(os.path.join(outdir, 'best_param.npy'), best)
    print(f'plots saved to {outdir}')
    return best, best_cost


def main():
    params_path = sys.argv[1] if len(sys.argv) > 1 else 'FiveCol_Parameter/training_with_Ih.npy'
    outdir = sys.argv[2] if len(sys.argv) > 2 else 'FiveCol_Parameter/gpu_test'
    override = sys.argv[3] if len(sys.argv) > 3 else None

    params = np.load(params_path)
    model_type = resolve_model_type(params_path, override)
    print(f'model_type={model_type} ({params.shape[-1]} params per set)')
    plot_param_set(params, outdir, model_type=model_type)


if __name__ == '__main__':
    main()
