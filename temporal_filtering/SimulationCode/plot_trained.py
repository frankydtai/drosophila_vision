#!/usr/bin/env python
"""Simulation + plotting for the FiveCol medulla model.

This module owns all the model-trace simulation and the plotting routines
(cost curve, model-vs-data, all-cell-types). It can also be run as a script to
visualise a trained parameter set in model_vs_data.png format:

    python plot_trained.py [params.npy] [outdir] [model_type]

Accepts either a single param vector (P,) or a stack (N, P); for a stack the
lowest-cost set is selected. The model type is NOT guessed from the parameter
count: it is recorded next to the params at save time (model_type.txt) and read
back here, so it works for any parameter count. Resolution priority is
explicit model_type arg > sidecar model_type.txt > run-dir path name.
"""
import os
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import matplotlib.pyplot as plt
import numpy as np
import torch

import Medulla_Library as ml
import blindschleiche_py3 as bs
import FiveCol_MedSim_Pytorch as fc
from FiveCol_MedSim_Pytorch import (
    calc_cost,
    data,
    data_amp,
    device,
    mc_cell_index,
    nofcells,
)

CELL_LIST = np.array(
    ['L1', 'L2', 'L3', 'L4', 'L5', 'Mi1', 'Tm3', 'Mi4', 'Mi9', 'Tm1', 'Tm2', 'Tm4', 'Tm9']
)
CENTER_COL = 2  # center of the 5 columns used in the cost function
CTYPE = np.load('Circuits/ctype.npy', allow_pickle=True)
FIT_INDEX = {name: i for i, name in enumerate(CELL_LIST)}
CENTER_NEURON_OFFSET = CENTER_COL * nofcells

# --- generic, test-driven plotting hooks (this module hardcodes no cell beyond
#     the default fit cells) --------------------------------------------------
# REF_CUBES: name -> (9,200) grey "data" reference cube drawn behind the model.
#   Defaults to the measured RF data for the 13 fit cells. A test script may
#   extend/override it (e.g. map R1-8 to L1's cube) so the plots follow whatever
#   the experiment configured -- without editing this file.
# MVD_GROUPS: ORDERED list of cell-name groups for model_vs_data. Each group is
#   drawn as its own row-pair (RF row + time row), assigned top-to-bottom in
#   order; EMPTY groups are skipped, so removing a group shifts the rest up (e.g.
#   with no R group, L becomes the top row-pair). Columns within a group are
#   auto-centred. None -> DEFAULT_MVD_GROUPS. A test may prepend/append groups
#   (e.g. an R1-8 group on top) -- this module hardcodes no R cells.
REF_CUBES = None
MVD_GROUPS = None

DEFAULT_MVD_GROUPS = [
    np.array(['L1', 'L2', 'L3', 'L4', 'L5']),               # lamina
    np.array(['Mi1', 'Mi4', 'Mi9']),                        # Mi
    np.array(['Tm1', 'Tm2', 'Tm3', 'Tm4', 'Tm9']),          # Tm
]


def default_ref_cubes():
    """Grey reference cube per fit-cell name, from the measured RF data."""
    ref = ml.read_RecF_data() * data_amp                 # (13, 9, 200)
    return {name: ref[i] for i, name in enumerate(CELL_LIST)}


def reference_cube(name):
    """(9,200) grey reference for a cell name, or None if none is registered."""
    global REF_CUBES
    if REF_CUBES is None:
        REF_CUBES = default_ref_cubes()
    return REF_CUBES.get(str(name))


def mvd_groups():
    """Present (non-empty) groups for model_vs_data, in display order."""
    groups = MVD_GROUPS if MVD_GROUPS is not None else DEFAULT_MVD_GROUPS
    return [np.asarray(g) for g in groups if len(g) > 0]


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


def _out_scale_vec(z, neuron_index, schema):
    """out_scale to apply to a trace of the cells in neuron_index (cell-type order).
    Returns 1.0 (absent), a 0-dim scalar (global), or an (n,1) per-cell tensor."""
    os_ = fc.assign_params(z, schema).get('out_scale', None)
    if os_ is None:
        return 1.0
    if os_.dim() == 0:
        return os_
    idx = (neuron_index % nofcells).to(os_.device)
    return os_[idx].reshape(-1, 1)


def _as_index(neuron_index, device):
    if not torch.is_tensor(neuron_index):
        return torch.tensor(neuron_index, dtype=torch.long, device=device)
    return neuron_index.to(device)


def _pack_filtered(stacked, z, neuron_index, schema):
    """Repack the core forward's (150, n) center-window response into the plot's
    (n, 200) trace: pre-stimulus zeroed, optional out_scale applied, then shifted
    one step to match the data convention. All model dynamics live in the core
    (fc._run_conductance / fc._run_adaptive); this is presentation-only reshaping."""
    n = stacked.shape[1]
    trace = torch.zeros(n, 200, dtype=torch.float64, device=stacked.device)
    trace[:, 50:200] = stacked.transpose(0, 1)
    trace = trace * _out_scale_vec(z, neuron_index, schema)
    trace[:, 0:50] = 0
    trace[:, 0:199] = trace[:, 1:200]
    return trace


@torch.no_grad()
def _simulate_filtered_traces(z, neuron_index, return_ref=False):
    neuron_index = _as_index(neuron_index, z.device)
    p = fc.assign_params(z, fc.CONDUCTANCE_SCHEMA)
    stacked, ref = fc._run_conductance(p, neuron_index=neuron_index, return_ref=True)
    trace = _pack_filtered(stacked, z, neuron_index, fc.CONDUCTANCE_SCHEMA)
    if return_ref:
        return trace, ref
    return trace


@torch.no_grad()
def _simulate_filtered_traces_adaptive(z, neuron_index, return_ref=False):
    neuron_index = _as_index(neuron_index, z.device)
    p = fc.assign_params_adaptive(z)
    stacked, ref = fc._run_adaptive(p, neuron_index=neuron_index, return_ref=True)
    trace = _pack_filtered(stacked, z, neuron_index, fc.ADAPTIVE_SCHEMA)
    if return_ref:
        return trace, ref
    return trace


def _simulate(z, neuron_index, model_type=None, return_ref=False):
    if model_type is None:
        model_type = fc.MODEL_TYPE
    if model_type == 'adaptive':
        return _simulate_filtered_traces_adaptive(z, neuron_index, return_ref=return_ref)
    return _simulate_filtered_traces(z, neuron_index, return_ref=return_ref)


def calc_model_trace(z, model_type=None):
    return _simulate(z, mc_cell_index, model_type)


def calc_center_column_trace(z, model_type=None):
    center_index = torch.arange(
        CENTER_NEURON_OFFSET,
        CENTER_NEURON_OFFSET + nofcells,
        dtype=torch.long,
        device=z.device,
    )
    return _simulate(z, center_index, model_type)


def calc_model_full_all(z, model_type=None, return_ref=False):
    """All cell types across 5 columns -> (65, 9, 200) spatio-temporal cube.

    With return_ref=True also returns the per-cell resting baseline (the value
    each trace is measured relative to) as a (65, 9) array; NaN where no column
    was simulated.
    """
    model_full = np.zeros((nofcells, 9, 200))
    ref_full = np.full((nofcells, 9), np.nan)
    for col in range(5):
        col_index = torch.arange(col * nofcells, (col + 1) * nofcells, dtype=torch.long, device=z.device)
        if return_ref:
            trace, ref = _simulate(z, col_index, model_type, return_ref=True)
            model_full[:, col + 2] = trace.cpu().numpy()
            ref_full[:, col + 2] = ref.cpu().numpy()
        else:
            model_full[:, col + 2] = _simulate(z, col_index, model_type).cpu().numpy()
    if return_ref:
        return model_full, ref_full
    return model_full


def _scale_curve(xt, center, sem_xt=None):
    """Impulse response (time) and amplitude-scaled azimuth RF for one cube.

    If ``sem_xt`` (same shape as ``xt``) is given, also return the center-row time
    SEM aligned to the impulse response (for a +/-SEM band on the model trace).
    """
    imp = xt[center]
    maxt = int(np.argmax(np.abs(imp)))
    rf = bs.blurr(bs.rebin(xt[:, maxt], 45), 5)
    amp = float(np.max(np.abs(imp)))
    rf = rf / (np.max(np.abs(rf)) + 1e-12) * amp
    if sem_xt is not None:
        return imp, np.roll(rf, -2), sem_xt[center]
    return imp, np.roll(rf, -2)


# ---- connectome multi-column cube (averaged over tiles x shifts x ring) ------

@torch.no_grad()
def _multicol_cube(z):
    """Build a (n_fit, 9, 200) model cube + SEM by averaging the batched forward.

    Runs the 7-shift (B) connectome forward, then for each fit cell type bins every
    readout cell by its ring radius. Following the single-column azimuth convention
    the ring radius is truncated with int() for DISPLAY (so sqrt(3) -> bin 1, same
    as col +/-1); training itself keeps sqrt(3) and 2 as distinct rings. Each bin is
    averaged over all tiles x shifts x ring members; sem = std/sqrt(n).
    Returns (names, cube, sem) for the present fit types.
    """
    p = fc.assign_params(z, fc.CONDUCTANCE_SCHEMA)
    model_full = fc._run_conductance_full(p, fc.signal)        # (B, 150, N)
    b_idx, u_idx = fc.READOUT
    sel = model_full[b_idx, :, u_idx].cpu().numpy()            # (n_cost, 150)
    radius = fc.MC_COST_RADIUS.cpu().numpy()                   # (n_cost,)
    type_idx = fc.CONNECTOME.node_type[u_idx].cpu().numpy()    # (n_cost,)
    type_names = list(fc.CONNECTOME.type_names)

    names = [ft for ft in CELL_LIST if ft in type_names]
    cube = np.zeros((len(names), 9, 200))
    sem = np.zeros((len(names), 9, 200))
    center = 4
    for ti, ft in enumerate(names):
        ft_global = type_names.index(ft)
        for off in range(5):                                  # azimuth offset 0..4
            mask = (type_idx == ft_global) & (np.floor(radius).astype(int) == off)
            if not mask.any():
                continue
            traces = sel[mask]                                # (k, 150)
            m = traces.mean(axis=0)
            s = traces.std(axis=0) / np.sqrt(traces.shape[0])
            for bin_j in {center + off, center - off}:        # mirror to both sides
                if 0 <= bin_j < 9:
                    cube[ti, bin_j, 50:200] = m
                    sem[ti, bin_j, 50:200] = s
    return names, cube, sem


def _nice_ylim(*curves, margin=1.25, step=5.0, floor=5.0, min_pad=3.0):
    vals = [np.asarray(c).ravel() for c in curves if c is not None]
    if not vals:
        return -floor, floor
    peak = float(np.max(np.abs(np.concatenate(vals))))
    ymax = max(peak * margin, peak + min_pad, floor)
    ymax = float(np.ceil(ymax / step) * step)
    return -ymax, ymax


def _style_time_axis(ax, show_xlabel):
    ax.set_xlim(0, 200)
    ax.set_xticks([0, 100, 200])
    ax.set_xticklabels(['0', '1', '2'], fontsize=6)
    if show_xlabel:
        ax.set_xlabel('time [s]', fontsize=7)


def _style_azimuth_axis(ax, show_xlabel):
    ax.set_xlim(0, 40)
    ax.set_xticks([0, 20, 40])
    ax.set_xticklabels(['-20', '0', '20'], fontsize=6)
    if show_xlabel:
        ax.set_xlabel('azimuth [$^\\circ$]', fontsize=7)


def _annotate_baseline(ax, baseline):
    """Relabel the y=0 line with the actual resting value (the trace baseline)."""
    if baseline is None or not np.isfinite(baseline):
        return
    ylo, yhi = ax.get_ylim()
    ax.set_yticks([ylo, 0.0, yhi])
    ax.set_yticklabels([f'{ylo:+.0f}', f'{baseline:.1f}', f'{yhi:+.0f}'], fontsize=6)
    ax.axhline(0.0, color='0.4', linewidth=0.6, linestyle=':', zorder=0)


def _plot_cell_pair_axes(ax_rf, ax_time, model_xt, ref_xt, title, show_legend=False,
                         show_xlabels=False, show_ylabel=False, baseline=None):
    """Match Borst_Fig4-6: azimuth RF on top, time response below."""
    center = CENTER_COL + 2
    imp_model, rf_model = _scale_curve(model_xt, center)
    if ref_xt is not None:
        imp_data, rf_data = _scale_curve(ref_xt, center)
    else:
        imp_data, rf_data = None, None
    curves = [c for c in (imp_model, imp_data, rf_model, rf_data) if c is not None]
    ylo, yhi = _nice_ylim(*curves)

    ax_rf.plot(rf_data, color='gray', linewidth=1.5, label='data') if rf_data is not None else None
    ax_rf.plot(rf_model, color='red', linewidth=1.5, label='model')
    ax_rf.set_title(title, fontsize=8, pad=2)
    ax_rf.set_ylim(ylo, yhi)
    _style_azimuth_axis(ax_rf, show_xlabels)
    if show_ylabel:
        ax_rf.set_ylabel('mV', fontsize=7)
    ax_rf.tick_params(labelsize=6)
    _annotate_baseline(ax_rf, baseline)
    if show_legend:
        ax_rf.legend(loc='upper right', fontsize=6, frameon=False)

    ax_time.plot(imp_data, color='gray', linewidth=1.5) if imp_data is not None else None
    ax_time.plot(imp_model, color='red', linewidth=1.5)
    ax_time.set_ylim(ylo, yhi)
    _style_time_axis(ax_time, show_xlabels)
    if show_ylabel:
        ax_time.set_ylabel('mV', fontsize=7)
    ax_time.tick_params(labelsize=6)
    _annotate_baseline(ax_time, baseline)


def _plot_cell_pair_sem(ax_rf, ax_time, model_xt, sem_xt, ref_xt, title,
                        show_legend=False, show_xlabels=False, show_ylabel=False):
    """Like _plot_cell_pair_axes but draws a pink +/-SEM band on the model trace.

    Used for connectome multi-column plots, where each trace is a mean over many
    tiles x shifts x ring members and the SEM (~0.06 mV) is small relative to the
    +/-20 mV data; the y-limit is set from model+SEM so the band is visible.
    """
    center = 4
    imp_model, rf_model, imp_sem = _scale_curve(model_xt, center, sem_xt)
    if ref_xt is not None:
        imp_data, rf_data = _scale_curve(ref_xt, center)
    else:
        imp_data, rf_data = None, None
    curves = [c for c in (imp_model, imp_model + imp_sem, imp_model - imp_sem,
                          rf_model, imp_data, rf_data) if c is not None]
    ylo, yhi = _nice_ylim(*curves)

    if rf_data is not None:
        ax_rf.plot(rf_data, color='gray', linewidth=1.5, label='data')
    ax_rf.plot(rf_model, color='red', linewidth=1.5, label='model')
    ax_rf.set_title(title, fontsize=8, pad=2)
    ax_rf.set_ylim(ylo, yhi)
    _style_azimuth_axis(ax_rf, show_xlabels)
    if show_ylabel:
        ax_rf.set_ylabel('mV', fontsize=7)
    ax_rf.tick_params(labelsize=6)
    if show_legend:
        ax_rf.legend(loc='upper right', fontsize=6, frameon=False)

    t = np.arange(200)
    if imp_data is not None:
        ax_time.plot(imp_data, color='gray', linewidth=1.5)
    ax_time.fill_between(t, imp_model - imp_sem, imp_model + imp_sem,
                         color='pink', alpha=0.8, linewidth=0, label='$\\pm$SEM')
    ax_time.plot(imp_model, color='red', linewidth=1.5)
    ax_time.set_ylim(ylo, yhi)
    _style_time_axis(ax_time, show_xlabels)
    if show_ylabel:
        ax_time.set_ylabel('mV', fontsize=7)
    ax_time.tick_params(labelsize=6)


def plot_model_vs_data_connectome(z, path, title=None):
    """Connectome model-vs-data: each fit type's ring-averaged trace + SEM band."""
    names, cube, sem = _multicol_cube(z)
    ncols = 5
    nrows = 2 * ((len(names) + ncols - 1) // ncols)
    fig = plt.figure(figsize=(3.0 * ncols, 2.5 * nrows))
    gs = fig.add_gridspec(nrows, ncols, hspace=0.55, wspace=0.55,
                          top=0.93, bottom=0.06, left=0.07, right=0.98)
    legend_done = False
    for i, name in enumerate(names):
        blk, col = divmod(i, ncols)
        ax_rf = fig.add_subplot(gs[2 * blk, col])
        ax_time = fig.add_subplot(gs[2 * blk + 1, col])
        _plot_cell_pair_sem(
            ax_rf, ax_time, cube[i], sem[i], reference_cube(name), name,
            show_legend=not legend_done, show_xlabels=True, show_ylabel=(col == 0),
        )
        legend_done = True
    if title is None:
        title = 'Connectome model vs data'
    n_tiles = fc.CONNECTOME.meta.get('n_centers', '?') if fc.CONNECTOME else '?'
    fig.suptitle(title + '  [avg over tiles x 7 shifts x ring]', fontsize=12)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cost(costs, path):
    plt.figure(figsize=(8, 4))
    plt.plot(costs, color='steelblue', linewidth=2)
    plt.xlabel('step')
    plt.ylabel('cost [% data power]')
    plt.title(f'Training cost ({len(costs)} steps)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_model_vs_data(z, path, n_steps=None, title=None):
    model_full, ref_full = calc_model_full_all(z, return_ref=True)

    groups = mvd_groups()
    ncols = 13
    nrows = 2 * len(groups)
    fig = plt.figure(figsize=(16, 2.5 * nrows))
    gs = fig.add_gridspec(nrows, ncols, hspace=0.5, wspace=0.55,
                          top=0.95, bottom=0.05, left=0.06, right=0.98)

    legend_done = False
    for gi, names in enumerate(groups):
        rf_row = 2 * gi                              # each group gets its own row-pair
        start = (ncols - len(names)) // 2            # auto-centre columns
        for j, name in enumerate(names):
            col = start + j
            ctype_i = int(np.where(CTYPE == name)[0][0])
            ax_rf = fig.add_subplot(gs[rf_row, col])
            ax_time = fig.add_subplot(gs[rf_row + 1, col])
            _plot_cell_pair_axes(
                ax_rf, ax_time, model_full[ctype_i], reference_cube(name), name,
                show_legend=not legend_done,
                show_xlabels=True,
                show_ylabel=(j == 0),                # leftmost cell of each group
                baseline=ref_full[ctype_i, CENTER_COL + 2],
            )
            legend_done = True

    if title is None:
        title = f'Model vs data after {n_steps} steps (center column)'
    fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_all_celltypes(z, path, n_steps=None, title=None):
    """All 65 cell types: azimuth RF top, time bottom (Borst_Fig4-6 layout)."""
    model_full, ref_full = calc_model_full_all(z, return_ref=True)

    ncols = 13
    fig = plt.figure(figsize=(26, 32))
    gs = fig.add_gridspec(10, ncols, hspace=0.65, wspace=0.45,
                          top=0.97, bottom=0.03, left=0.04, right=0.99)

    for i in range(nofcells):
        row, col = divmod(i, ncols)
        name = str(CTYPE[i])
        ref_xt = reference_cube(name)
        ax_rf = fig.add_subplot(gs[row * 2, col])
        ax_time = fig.add_subplot(gs[row * 2 + 1, col])
        _plot_cell_pair_axes(
            ax_rf, ax_time, model_full[i], ref_xt, name,
            show_legend=(i == 0),
            show_xlabels=(row == 4),
            show_ylabel=(col == 0),
            baseline=ref_full[i, CENTER_COL + 2],
        )

    if title is None:
        title = f'All {nofcells} cell types after {n_steps} steps'
    fig.suptitle(title, fontsize=14)
    fig.savefig(path, dpi=150)
    plt.close(fig)


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


def select_best(params):
    params = np.atleast_2d(params)
    # drop rows not yet filled by training (all zeros)
    valid = params[np.any(params != 0, axis=1)]
    if len(valid) == 0:
        raise SystemExit('no trained parameter sets found (file all zeros)')
    costs = []
    for row in valid:
        z = torch.tensor(row, dtype=torch.float64)
        costs.append(fc.calc_cost(z, fc.data).item())
    costs = np.array(costs)
    best = int(np.argmin(costs))
    print(f'{len(valid)} trained set(s); costs min={costs.min():.4f} '
          f'max={costs.max():.4f}; selected #{best}')
    return valid[best], costs[best]


def plot_param_set(params, outdir, costs=None, model_type=None):
    """Select the best param set and write all plots into outdir.

    Shared by run.py (after training) and main() (standalone). The model type
    must already be known: either set in fc.MODEL_TYPE by the caller, or passed
    via model_type. It is never guessed from the parameter count.
    """
    os.makedirs(outdir, exist_ok=True)
    if model_type is not None:
        fc.MODEL_TYPE = model_type

    best, best_cost = select_best(params)
    z = torch.tensor(best, dtype=torch.float64, device=device)

    if costs is not None:
        plot_cost(costs, os.path.join(outdir, 'cost_curve.png'))

    suffix = f'trained, cost {best_cost:.2f}% of data power'
    mvd = os.path.join(outdir, 'model_vs_data.png')
    allc = os.path.join(outdir, 'model_all_cells.png')
    if getattr(fc, 'CONNECTOME', None) is not None:
        # connectome multi-column: ring-averaged cube + SEM (Borst layout doesn't
        # apply -- no 5 fixed columns / 65 named types).
        plot_model_vs_data_connectome(z, mvd, title=f'Connectome model vs data ({suffix})')
    else:
        plot_model_vs_data(z, mvd, title=f'Model vs data ({suffix})')
        plot_all_celltypes(z, allc, title=f'All 65 cell types ({suffix})')
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
