#!/usr/bin/env python
"""Tile plotting (Borst + network tile target) split from plot_trained.

This module contains:
- Borst 5-column plotting (classic "model-data" + "model-all")
- Network tile-target plotting (ring-averaged cube + SEM band)

plot_trained.py should only orchestrate which variant to call.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

import Medulla_Library as ml
import blindschleiche_py3 as bs
import FiveCol_MedSim_Pytorch as fc
from plot.utils import nice_ylim as _nice_ylim

CELL_LIST = np.array(
    ['L1', 'L2', 'L3', 'L4', 'L5', 'Mi1', 'Tm3', 'Mi4', 'Mi9', 'Tm1', 'Tm2', 'Tm4', 'Tm9']
)

CENTER_COL = ml.CENTER_COL
CTYPE = np.load('Circuits/ctype.npy', allow_pickle=True)
CENTER_NEURON_OFFSET = ml.column_start(CENTER_COL)

REF_CUBES = None
MVD_GROUPS = None

DEFAULT_MVD_GROUPS = [
    np.array(['L1', 'L2', 'L3', 'L4', 'L5']),
    np.array(['Mi1', 'Mi4', 'Mi9']),
    np.array(['Tm1', 'Tm2', 'Tm3', 'Tm4', 'Tm9']),
]


def default_ref_cubes():
    ref = ml.read_RecF_data() * ml.DATA_AMP
    return {name: ref[i] for i, name in enumerate(CELL_LIST)}


def reference_cube(name):
    global REF_CUBES
    if REF_CUBES is None:
        REF_CUBES = default_ref_cubes()
    return REF_CUBES.get(str(name))


def mvd_groups():
    groups = MVD_GROUPS if MVD_GROUPS is not None else DEFAULT_MVD_GROUPS
    return [np.asarray(g) for g in groups if len(g) > 0]


def _scale_curve(xt, center, sem_xt=None):
    imp = xt[center]
    maxt = int(np.argmax(np.abs(imp)))
    rf = bs.blurr(bs.rebin(xt[:, maxt], 45), 5)
    amp = float(np.max(np.abs(imp)))
    rf = rf / (np.max(np.abs(rf)) + 1e-12) * amp
    if sem_xt is not None:
        return imp, np.roll(rf, -2), sem_xt[center]
    return imp, np.roll(rf, -2)


def _style_time_axis(ax, show_xlabel):
    t_end = fc.maxtime * fc.deltat / 1000.0
    t_mid = t_end / 2.0
    ax.set_xlim(0, fc.maxtime)
    ax.set_xticks([0, fc.maxtime // 2, fc.maxtime])
    ax.set_xticklabels(['0', f'{t_mid:g}', f'{t_end:g}'], fontsize=6)
    if show_xlabel:
        ax.set_xlabel('time [s]', fontsize=7)


def _style_azimuth_axis(ax, show_xlabel):
    ax.set_xlim(0, 40)
    ax.set_xticks([0, 20, 40])
    ax.set_xticklabels(['-20', '0', '20'], fontsize=6)
    if show_xlabel:
        ax.set_xlabel('azimuth [$^\\circ$]', fontsize=7)


def _annotate_baseline(ax, baseline):
    if baseline is None or not np.isfinite(baseline):
        return
    ylo, yhi = ax.get_ylim()
    ax.set_yticks([ylo, 0.0, yhi])
    ax.set_yticklabels([f'{ylo:+.0f}', f'{baseline:.1f}', f'{yhi:+.0f}'], fontsize=6)
    ax.axhline(0.0, color='0.4', linewidth=0.6, linestyle=':', zorder=0)


def plot_cell_pair_axes(
    ax_rf,
    ax_time,
    model_xt,
    ref_xt,
    title,
    show_legend=False,
    show_xlabels=False,
    show_ylabel=False,
    baseline=None,
):
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


def plot_cell_pair_sem(
    ax_rf,
    ax_time,
    model_xt,
    sem_xt,
    ref_xt,
    title,
    show_legend=False,
    show_xlabels=False,
    show_ylabel=False,
):
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

    t = np.arange(fc.maxtime)
    if imp_data is not None:
        ax_time.plot(imp_data, color='gray', linewidth=1.5)
    ax_time.fill_between(
        t, imp_model - imp_sem, imp_model + imp_sem,
        color='pink', alpha=0.8, linewidth=0, label='$\\pm$SEM',
    )
    ax_time.plot(imp_model, color='red', linewidth=1.5)
    ax_time.set_ylim(ylo, yhi)
    _style_time_axis(ax_time, show_xlabels)
    if show_ylabel:
        ax_time.set_ylabel('mV', fontsize=7)
    ax_time.tick_params(labelsize=6)


def _out_scale_vec(z, neuron_index, schema):
    os_ = fc.assign_params(z, schema).get('out_scale', None)
    if os_ is None:
        return 1.0
    if os_.dim() == 0:
        return os_
    idx = (neuron_index % fc.nofcells).to(os_.device)
    return os_[idx].reshape(-1, 1)


def _as_index(neuron_index, device):
    if not torch.is_tensor(neuron_index):
        return torch.tensor(neuron_index, dtype=torch.long, device=device)
    return neuron_index.to(device)


def _pack_filtered(stacked, z, neuron_index, schema):
    n = stacked.shape[1]
    trace = torch.zeros(n, fc.maxtime, dtype=torch.float64, device=stacked.device)
    trace[:, fc.t_on:fc.maxtime] = stacked.transpose(0, 1)
    trace = trace * _out_scale_vec(z, neuron_index, schema)
    trace[:, 0:fc.t_on] = 0
    trace[:, 0:fc.maxtime - 1] = trace[:, 1:fc.maxtime]
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
    model_type = fc.MODEL_TYPE if model_type is None else model_type
    if model_type == 'adaptive':
        return _simulate_filtered_traces_adaptive(z, neuron_index, return_ref=return_ref)
    return _simulate_filtered_traces(z, neuron_index, return_ref=return_ref)


def calc_model_trace(z, model_type=None):
    return _simulate(z, fc.mc_cell_index, model_type)


def calc_center_column_trace(z, model_type=None):
    center_index = torch.arange(
        CENTER_NEURON_OFFSET,
        CENTER_NEURON_OFFSET + fc.nofcells,
        dtype=torch.long,
        device=z.device,
    )
    return _simulate(z, center_index, model_type)


def calc_model_full_all(z, model_type=None, return_ref=False):
    model_full = np.zeros((fc.nofcells, 9, fc.maxtime))
    ref_full = np.full((fc.nofcells, 9), np.nan)
    for col in range(5):
        col_index = torch.arange(
            col * fc.nofcells,
            (col + 1) * fc.nofcells,
            dtype=torch.long,
            device=z.device,
        )
        if return_ref:
            trace, ref = _simulate(z, col_index, model_type, return_ref=True)
            model_full[:, col + 2] = trace.cpu().numpy()
            ref_full[:, col + 2] = ref.cpu().numpy()
        else:
            model_full[:, col + 2] = _simulate(z, col_index, model_type).cpu().numpy()
    if return_ref:
        return model_full, ref_full
    return model_full


@torch.no_grad()
def multicol_cube(z):
    p = fc.assign_params(z, fc.CONDUCTANCE_SCHEMA)
    model_full = fc._run_conductance_full(p, fc.signal)
    b_idx, u_idx = fc.READOUT
    sel = model_full[b_idx, :, u_idx].cpu().numpy()
    radius = fc.MC_COST_RADIUS.cpu().numpy()
    type_idx = fc.NETWORK.node_type[u_idx].cpu().numpy()
    type_names = list(fc.NETWORK.type_names)

    names = [ft for ft in CELL_LIST if ft in type_names]
    cube = np.zeros((len(names), 9, fc.maxtime))
    sem = np.zeros((len(names), 9, fc.maxtime))
    center = 4
    for ti, ft in enumerate(names):
        ft_global = type_names.index(ft)
        for off in range(5):
            mask = (type_idx == ft_global) & (np.floor(radius).astype(int) == off)
            if not mask.any():
                continue
            traces = sel[mask]
            m = traces.mean(axis=0)
            s = traces.std(axis=0) / np.sqrt(traces.shape[0])
            for bin_j in {center + off, center - off}:
                if 0 <= bin_j < 9:
                    cube[ti, bin_j, fc.t_on:fc.maxtime] = m
                    sem[ti, bin_j, fc.t_on:fc.maxtime] = s
    return names, cube, sem


def plot_model_data_network(z, path, title=None):
    names, cube, sem = multicol_cube(z)
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
        plot_cell_pair_sem(
            ax_rf, ax_time, cube[i], sem[i], reference_cube(name), name,
            show_legend=not legend_done, show_xlabels=True, show_ylabel=(col == 0),
        )
        legend_done = True
    if title is None:
        title = 'Network model-data'
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


def plot_model_data(z, path, n_steps=None, title=None):
    model_full, ref_full = calc_model_full_all(z, return_ref=True)

    groups = mvd_groups()
    ncols = 13
    nrows = 2 * len(groups)
    fig = plt.figure(figsize=(16, 2.5 * nrows))
    gs = fig.add_gridspec(nrows, ncols, hspace=0.5, wspace=0.55,
                          top=0.95, bottom=0.05, left=0.06, right=0.98)

    legend_done = False
    for gi, names in enumerate(groups):
        rf_row = 2 * gi
        start = (ncols - len(names)) // 2
        for j, name in enumerate(names):
            col = start + j
            ctype_i = int(np.where(CTYPE == name)[0][0])
            ax_rf = fig.add_subplot(gs[rf_row, col])
            ax_time = fig.add_subplot(gs[rf_row + 1, col])
            plot_cell_pair_axes(
                ax_rf, ax_time, model_full[ctype_i], reference_cube(name), name,
                show_legend=not legend_done,
                show_xlabels=True,
                show_ylabel=(j == 0),
                baseline=ref_full[ctype_i, CENTER_COL + 2],
            )
            legend_done = True

    if title is None:
        title = f'Model-data after {n_steps} steps (center column)'
    fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_model_all(z, path, n_steps=None, title=None):
    model_full, ref_full = calc_model_full_all(z, return_ref=True)

    ncols = 13
    fig = plt.figure(figsize=(26, 32))
    gs = fig.add_gridspec(10, ncols, hspace=0.65, wspace=0.45,
                          top=0.97, bottom=0.03, left=0.04, right=0.99)

    for i in range(fc.nofcells):
        row, col = divmod(i, ncols)
        name = str(CTYPE[i])
        ref_xt = reference_cube(name)
        ax_rf = fig.add_subplot(gs[row * 2, col])
        ax_time = fig.add_subplot(gs[row * 2 + 1, col])
        plot_cell_pair_axes(
            ax_rf, ax_time, model_full[i], ref_xt, name,
            show_legend=(i == 0),
            show_xlabels=(row == 4),
            show_ylabel=(col == 0),
            baseline=ref_full[i, CENTER_COL + 2],
        )

    if title is None:
        title = f'Model-all {fc.nofcells} cells after {n_steps} steps'
    fig.suptitle(title, fontsize=14)
    fig.savefig(path, dpi=150)
    plt.close(fig)

