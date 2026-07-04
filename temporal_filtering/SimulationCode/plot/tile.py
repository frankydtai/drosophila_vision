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

DEFAULT_MVD_GROUPS = [
    np.array(['L1', 'L2', 'L3', 'L4', 'L5']),
    np.array(['Mi1', 'Mi4', 'Mi9']),
    np.array(['Tm1', 'Tm2', 'Tm3', 'Tm4', 'Tm9']),
]


def default_ref_cubes(dark=False):
    data = ml.read_RecF_data_dark() if dark else ml.read_RecF_data()
    ref = data * ml.DATA_AMP
    return {name: ref[i] for i, name in enumerate(CELL_LIST)}


def reference_cube(name, ref_cubes=None, dark=False):
    if ref_cubes is not None:
        return ref_cubes.get(str(name))
    return default_ref_cubes(dark=dark).get(str(name))


def _session_dark(session):
    return session.primary_pack.name == "tile_dark"


def mvd_groups(groups=None):
    src = DEFAULT_MVD_GROUPS if groups is None else groups
    return [np.asarray(g) for g in src if len(g) > 0]


def _scale_curve(xt, center, sem_xt=None):
    imp = xt[center]
    maxt = int(np.argmax(np.abs(imp)))
    rf = bs.blurr(bs.rebin(xt[:, maxt], 45), 5)
    amp = float(np.max(np.abs(imp)))
    rf = rf / (np.max(np.abs(rf)) + 1e-12) * amp
    if sem_xt is not None:
        return imp, np.roll(rf, -2), sem_xt[center]
    return imp, np.roll(rf, -2)


def _style_time_axis(ax, show_xlabel, maxtime):
    t_end = maxtime * fc.deltat / 1000.0
    t_mid = t_end / 2.0
    ax.set_xlim(0, maxtime)
    ax.set_xticks([0, maxtime // 2, maxtime])
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
    maxtime=ml.IMPULSE_MAXTIME,
    off_model_xt=None,
    off_ref_xt=None,
    off_baseline=None,
):
    center = CENTER_COL + 2
    imp_model, rf_model = _scale_curve(model_xt, center)
    if ref_xt is not None:
        imp_data, rf_data = _scale_curve(ref_xt, center)
    else:
        imp_data, rf_data = None, None
    off_imp_model = off_rf_model = off_imp_data = off_rf_data = None
    if off_model_xt is not None:
        off_imp_model, off_rf_model = _scale_curve(off_model_xt, center)
    if off_ref_xt is not None:
        off_imp_data, off_rf_data = _scale_curve(off_ref_xt, center)
    curves = [c for c in (
        imp_model, imp_data, rf_model, rf_data,
        off_imp_model, off_imp_data, off_rf_model, off_rf_data,
    ) if c is not None]
    ylo, yhi = _nice_ylim(*curves)

    ax_rf.plot(rf_data, color='gray', linewidth=1.5, label='on data') if rf_data is not None else None
    ax_rf.plot(rf_model, color='red', linewidth=1.5, label='on model')
    if off_rf_data is not None:
        ax_rf.plot(off_rf_data, color='gray', linewidth=1.5, linestyle='--', label='off data')
    if off_rf_model is not None:
        ax_rf.plot(off_rf_model, color='red', linewidth=1.5, linestyle='--', label='off model')
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
    if off_imp_data is not None:
        ax_time.plot(off_imp_data, color='gray', linewidth=1.5, linestyle='--')
    if off_imp_model is not None:
        ax_time.plot(off_imp_model, color='red', linewidth=1.5, linestyle='--')
    ax_time.set_ylim(ylo, yhi)
    _style_time_axis(ax_time, show_xlabels, maxtime)
    if show_ylabel:
        ax_time.set_ylabel('mV', fontsize=7)
    ax_time.tick_params(labelsize=6)
    _annotate_baseline(ax_time, off_baseline if off_baseline is not None else baseline)


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
    maxtime=ml.IMPULSE_MAXTIME,
    off_model_xt=None,
    off_ref_xt=None,
):
    center = 4
    imp_model, rf_model, imp_sem = _scale_curve(model_xt, center, sem_xt)
    if ref_xt is not None:
        imp_data, rf_data = _scale_curve(ref_xt, center)
    else:
        imp_data, rf_data = None, None
    off_imp_model = off_rf_model = off_imp_data = off_rf_data = None
    if off_model_xt is not None:
        off_imp_model, off_rf_model = _scale_curve(off_model_xt, center)
    if off_ref_xt is not None:
        off_imp_data, off_rf_data = _scale_curve(off_ref_xt, center)
    curves = [c for c in (
        imp_model, imp_model + imp_sem, imp_model - imp_sem,
        rf_model, imp_data, rf_data,
        off_imp_model, off_rf_model, off_imp_data, off_rf_data,
    ) if c is not None]
    ylo, yhi = _nice_ylim(*curves)

    if rf_data is not None:
        ax_rf.plot(rf_data, color='gray', linewidth=1.5, label='on data')
    ax_rf.plot(rf_model, color='red', linewidth=1.5, label='on model')
    if off_rf_data is not None:
        ax_rf.plot(off_rf_data, color='gray', linewidth=1.5, linestyle='--', label='off data')
    if off_rf_model is not None:
        ax_rf.plot(off_rf_model, color='red', linewidth=1.5, linestyle='--', label='off model')
    ax_rf.set_title(title, fontsize=8, pad=2)
    ax_rf.set_ylim(ylo, yhi)
    _style_azimuth_axis(ax_rf, show_xlabels)
    if show_ylabel:
        ax_rf.set_ylabel('mV', fontsize=7)
    ax_rf.tick_params(labelsize=6)
    if show_legend:
        ax_rf.legend(loc='upper right', fontsize=6, frameon=False)

    t = np.arange(maxtime)
    if imp_data is not None:
        ax_time.plot(imp_data, color='gray', linewidth=1.5)
    ax_time.fill_between(
        t, imp_model - imp_sem, imp_model + imp_sem,
        color='pink', alpha=0.8, linewidth=0, label='$\\pm$SEM',
    )
    ax_time.plot(imp_model, color='red', linewidth=1.5)
    if off_imp_data is not None:
        ax_time.plot(off_imp_data, color='gray', linewidth=1.5, linestyle='--')
    if off_imp_model is not None:
        ax_time.plot(off_imp_model, color='red', linewidth=1.5, linestyle='--')
    ax_time.set_ylim(ylo, yhi)
    _style_time_axis(ax_time, show_xlabels, maxtime)
    if show_ylabel:
        ax_time.set_ylabel('mV', fontsize=7)
    ax_time.tick_params(labelsize=6)


def _out_scale_vec(z, neuron_index, schema, backend):
    os_ = fc.assign_params(z, schema, backend).get('out_scale', None)
    if os_ is None:
        return 1.0
    if os_.dim() == 0:
        return os_
    idx = (neuron_index % backend.n_types).to(os_.device)
    return os_[idx].reshape(-1, 1)


def _as_index(neuron_index, device):
    if not torch.is_tensor(neuron_index):
        return torch.tensor(neuron_index, dtype=torch.long, device=device)
    return neuron_index.to(device)


def _pack_filtered(stacked, z, neuron_index, schema, session):
    n = stacked.shape[1]
    mt = session.maxtime
    trace = torch.zeros(n, mt, dtype=torch.float64, device=stacked.device)
    trace[:, fc.t_on:mt] = stacked.transpose(0, 1)
    trace = trace * _out_scale_vec(z, neuron_index, schema, session.backend)
    trace[:, 0:fc.t_on] = 0
    trace[:, 0:mt - 1] = trace[:, 1:mt]
    return trace


@torch.no_grad()
def _simulate_filtered_traces(session, z, neuron_index, return_ref=False):
    neuron_index = _as_index(neuron_index, z.device)
    schema = list(session.schema)
    p = fc.assign_params(z, schema, session.backend)
    stacked, ref = fc._run_conductance(session, p, neuron_index=neuron_index, return_ref=True)
    trace = _pack_filtered(stacked, z, neuron_index, schema, session)
    if return_ref:
        return trace, ref
    return trace


@torch.no_grad()
def _simulate_filtered_traces_adaptive(session, z, neuron_index, return_ref=False):
    neuron_index = _as_index(neuron_index, z.device)
    schema = list(session.schema)
    p = fc.assign_params_adaptive(z, schema, session.backend)
    stacked, ref = fc._run_adaptive(p, session, neuron_index=neuron_index, return_ref=True)
    trace = _pack_filtered(stacked, z, neuron_index, schema, session)
    if return_ref:
        return trace, ref
    return trace


def _simulate(session, z, neuron_index, return_ref=False):
    if session.model_type == 'adaptive':
        return _simulate_filtered_traces_adaptive(session, z, neuron_index, return_ref=return_ref)
    return _simulate_filtered_traces(session, z, neuron_index, return_ref=return_ref)


def calc_model_full_all(session, z, return_ref=False):
    n_types = session.backend.n_types
    mt = session.maxtime
    model_full = np.zeros((n_types, 9, mt))
    ref_full = np.full((n_types, 9), np.nan)
    for col in range(5):
        col_index = torch.arange(
            col * n_types,
            (col + 1) * n_types,
            dtype=torch.long,
            device=z.device,
        )
        if return_ref:
            trace, ref = _simulate(session, z, col_index, return_ref=True)
            model_full[:, col + 2] = trace.cpu().numpy()
            ref_full[:, col + 2] = ref.cpu().numpy()
        else:
            model_full[:, col + 2] = _simulate(session, z, col_index).cpu().numpy()
    if return_ref:
        return model_full, ref_full
    return model_full


@torch.no_grad()
def multicol_cube(session, z, all_cells=False):
    pack = session.primary_pack
    schema = list(session.schema)
    p = fc.assign_params(z, schema, session.backend)
    model_full = fc._run_conductance_full(session, p, pack.signal)
    sel = fc._readout_model_traces_pack(model_full, pack).cpu().numpy()
    if pack.cost_radius is not None:
        radius = pack.cost_radius.cpu().numpy()
    else:
        radius = np.zeros(pack.cost_weight.shape[0], dtype=np.float64)
    C = session.backend.network
    type_idx = C.node_type[pack.readout_unit].cpu().numpy()
    type_names = list(C.type_names)

    if all_cells:
        names = [str(n) for n in type_names]
    else:
        names = [ft for ft in CELL_LIST if ft in type_names]
    mt = session.maxtime
    cube = np.zeros((len(names), 9, mt))
    sem = np.zeros((len(names), 9, mt))
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
                    cube[ti, bin_j, fc.t_on:mt] = m
                    sem[ti, bin_j, fc.t_on:mt] = s
    return names, cube, sem


def plot_network_tile(session_on, z, path, *, session_off=None, all_cells=False,
                      title, ref_cubes=None, ref_cubes_off=None):
    """Network tile figure; optional on+off overlay (off dashed)."""
    names, cube_on, sem_on = multicol_cube(session_on, z, all_cells=all_cells)
    cube_off = None
    if session_off is not None:
        _, cube_off, _ = multicol_cube(session_off, z, all_cells=all_cells)
    dual = session_off is not None
    if ref_cubes is not None:
        ref_on = ref_cubes
    elif dual:
        ref_on = default_ref_cubes(dark=False)
    else:
        ref_on = default_ref_cubes(dark=_session_dark(session_on))
    if ref_cubes_off is not None:
        ref_off = ref_cubes_off
    elif dual:
        ref_off = default_ref_cubes(dark=True)
    else:
        ref_off = None

    ncols = 5 if not all_cells else 8
    nrows = 2 * ((len(names) + ncols - 1) // ncols)
    fig_w = 3.0 if not all_cells else 2.2
    fig = plt.figure(figsize=(fig_w * ncols, 2.5 * nrows))
    gs = fig.add_gridspec(nrows, ncols, hspace=0.55, wspace=0.55,
                          top=0.93, bottom=0.06, left=0.07, right=0.98)
    legend_done = False
    for i, name in enumerate(names):
        blk, col = divmod(i, ncols)
        ax_rf = fig.add_subplot(gs[2 * blk, col])
        ax_time = fig.add_subplot(gs[2 * blk + 1, col])
        off_kw = {}
        if dual:
            off_kw = {
                'off_model_xt': cube_off[i],
                'off_ref_xt': ref_off.get(name) if ref_off else None,
            }
        plot_cell_pair_sem(
            ax_rf, ax_time, cube_on[i], sem_on[i], ref_on.get(name), name,
            show_legend=not legend_done, show_xlabels=True, show_ylabel=(col == 0),
            maxtime=session_on.maxtime,
            **off_kw,
        )
        legend_done = True
    fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_borst_tile(session_on, z, path, *, session_off=None, all_cells=False,
                    title, ref_cubes=None, ref_cubes_off=None, mvd_group_list=None):
    """Borst tile figure; optional on+off overlay (off dashed)."""
    model_on, ref_on = calc_model_full_all(session_on, z, return_ref=True)
    model_off = ref_off = None
    if session_off is not None:
        model_off, ref_off = calc_model_full_all(session_off, z, return_ref=True)
    dual = session_off is not None
    if ref_cubes is not None:
        ref_on_cubes = ref_cubes
    elif dual:
        ref_on_cubes = default_ref_cubes(dark=False)
    else:
        ref_on_cubes = default_ref_cubes(dark=_session_dark(session_on))
    if ref_cubes_off is not None:
        ref_off_cubes = ref_cubes_off
    elif dual:
        ref_off_cubes = default_ref_cubes(dark=True)
    else:
        ref_off_cubes = None
    maxtime = session_on.maxtime

    def _ref(name):
        return ref_on_cubes.get(name)

    def _off_kw(row_i, name):
        if not dual:
            return {}
        return {
            'off_model_xt': model_off[row_i],
            'off_ref_xt': ref_off_cubes.get(name),
            'off_baseline': ref_off[row_i, CENTER_COL + 2],
        }

    ncols = 13
    if all_cells:
        n_types = session_on.backend.n_types
        fig = plt.figure(figsize=(26, 32))
        gs = fig.add_gridspec(10, ncols, hspace=0.65, wspace=0.45,
                              top=0.97, bottom=0.03, left=0.04, right=0.99)
        for i in range(n_types):
            row, col = divmod(i, ncols)
            name = str(CTYPE[i])
            ax_rf = fig.add_subplot(gs[row * 2, col])
            ax_time = fig.add_subplot(gs[row * 2 + 1, col])
            plot_cell_pair_axes(
                ax_rf, ax_time, model_on[i], _ref(name), name,
                show_legend=(i == 0),
                show_xlabels=(row == 4),
                show_ylabel=(col == 0),
                baseline=ref_on[i, CENTER_COL + 2],
                maxtime=maxtime,
                **_off_kw(i, name),
            )
    else:
        groups = mvd_groups(mvd_group_list)
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
                    ax_rf, ax_time, model_on[ctype_i], _ref(name), name,
                    show_legend=not legend_done,
                    show_xlabels=True,
                    show_ylabel=(j == 0),
                    baseline=ref_on[ctype_i, CENTER_COL + 2],
                    maxtime=maxtime,
                    **_off_kw(ctype_i, name),
                )
                legend_done = True

    fig.suptitle(title, fontsize=12 if not all_cells else 14)
    fig.savefig(path, dpi=150)
    plt.close(fig)
