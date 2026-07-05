#!/usr/bin/env python
"""Tile plotting (Borst + network tile target).

Shared ``plot_cell_pair`` / ``_plot_tile_figure``; backend-specific cube prep only.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

import Medulla_Library as ml
import blindschleiche_py3 as bs
import FiveCol_MedSim_Pytorch as fc
from plot.readout import (
    DEFAULT_MVD_GROUPS,
    borst_ref_cubes,
    tile_model_data_groups,
    tile_model_data_names,
    tile_ref_cubes,
)
from plot.utils import (
    DATA_COLOR,
    MODEL_COLOR,
    TRACE_LW,
    annotate_baseline,
    baselines_for_types,
    center_column_only,
    plot_timecourse,
    save_figure,
    sem_from_traces,
    ylim_for_traces,
)

CENTER_COL = ml.CENTER_COL
CENTER_BIN = CENTER_COL + 2
CTYPE = ml.ctype
CENTER_NEURON_OFFSET = ml.column_start(CENTER_COL)


def _baseline_from_ref_grid(ref_grid, row_i):
    """Resting potential at stimulus onset for one cell type (center column)."""
    return ref_grid[row_i, CENTER_BIN]


def _resolve_tile_ref_cubes(session_on, session_off=None, ref_cubes=None, ref_cubes_off=None):
    dual = session_off is not None
    if ref_cubes is not None:
        ref_on = ref_cubes
    elif dual:
        ref_on = tile_ref_cubes(session_on, 'tile_bright', dark=False)
    else:
        ref_on = tile_ref_cubes(
            session_on, session_on.primary_pack.name, dark=_session_dark(session_on),
        )
    if ref_cubes_off is not None:
        ref_off = ref_cubes_off
    elif dual:
        ref_off = tile_ref_cubes(session_off, 'tile_dark', dark=True)
    else:
        ref_off = None
    return ref_on, ref_off


def _session_dark(session):
    return session.primary_pack.name == "tile_dark"


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


def plot_cell_pair(
    ax_rf,
    ax_time,
    model_xt,
    ref_xt,
    title,
    *,
    sem_xt=None,
    show_legend=False,
    show_xlabels=False,
    show_ylabel=False,
    baseline=None,
    maxtime=ml.IMPULSE_MAXTIME,
    off_model_xt=None,
    off_ref_xt=None,
    off_baseline=None,
):
    center = CENTER_BIN
    if sem_xt is not None:
        imp_model, rf_model, imp_sem = _scale_curve(model_xt, center, sem_xt)
    else:
        imp_model, rf_model = _scale_curve(model_xt, center)
        imp_sem = None
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
    if imp_sem is not None:
        curves.extend([imp_model + imp_sem, imp_model - imp_sem])
    ylo, yhi = ylim_for_traces(imp_model, extra=curves[1:])

    if rf_data is not None:
        ax_rf.plot(rf_data, color=DATA_COLOR, linewidth=TRACE_LW, label='on data')
    ax_rf.plot(rf_model, color=MODEL_COLOR, linewidth=TRACE_LW, label='on model')
    if off_rf_data is not None:
        ax_rf.plot(off_rf_data, color=DATA_COLOR, linewidth=TRACE_LW, linestyle='--', label='off data')
    if off_rf_model is not None:
        ax_rf.plot(off_rf_model, color=MODEL_COLOR, linewidth=TRACE_LW, linestyle='--', label='off model')
    ax_rf.set_title(title, fontsize=8, pad=2)
    ax_rf.set_ylim(ylo, yhi)
    _style_azimuth_axis(ax_rf, show_xlabels)
    if show_ylabel:
        ax_rf.set_ylabel('mV', fontsize=7)
    ax_rf.tick_params(labelsize=6)
    annotate_baseline(ax_rf, baseline)
    if show_legend:
        ax_rf.legend(loc='upper right', fontsize=6, frameon=False)

    t = np.arange(maxtime)
    plot_timecourse(
        ax_time, t, imp_model,
        data=imp_data,
        sem=imp_sem,
        show_sem=imp_sem is not None,
        off_model=off_imp_model,
        off_data=off_imp_data,
        ylim=(ylo, yhi),
        baseline=off_baseline if off_baseline is not None else baseline,
        show_ylabel=show_ylabel,
        style_xaxis=lambda ax: _style_time_axis(ax, show_xlabels, maxtime),
    )


def _as_index(neuron_index, device):
    if not torch.is_tensor(neuron_index):
        return torch.tensor(neuron_index, dtype=torch.long, device=device)
    return neuron_index.to(device)


@torch.no_grad()
def _simulate(session, z, neuron_index, return_ref=False):
    neuron_index = _as_index(neuron_index, z.device)
    schema = list(session.schema)
    backend = session.backend
    if session.model_type == 'adaptive':
        p = fc.assign_params_adaptive(z, schema, backend)
        stacked, ref = fc._run_adaptive(p, session, neuron_index=neuron_index, return_ref=True)
    else:
        p = fc.assign_params(z, schema, backend)
        stacked, ref = fc._run_conductance(session, p, neuron_index=neuron_index, return_ref=True)
    mt = session.maxtime
    scale = fc.out_scale_for_units(p, neuron_index, backend)
    trace = fc.expand_plot_traces(stacked.transpose(0, 1), scale, mt)
    if return_ref:
        return trace, ref
    return trace


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


def _network_readout_layout(pack, C):
    readout = pack.readout_unit.cpu().numpy()
    if pack.cost_radius is not None:
        radius = pack.cost_radius.cpu().numpy()
    else:
        radius = np.zeros(pack.cost_weight.shape[0], dtype=np.float64)
    type_idx = C.node_type[pack.readout_unit].cpu().numpy()
    type_names = list(C.type_names)
    return readout, radius, type_idx, type_names


def _fill_ring_cube(cube, sem, ti, ft_global, type_idx, radius, plot_traces, center):
    for off in range(5):
        mask = (type_idx == ft_global) & (np.floor(radius).astype(int) == off)
        if not mask.any():
            continue
        traces = plot_traces[mask]
        m = traces.mean(axis=0)
        s = sem_from_traces(traces, center_only=False)
        for bin_j in {center + off, center - off}:
            if 0 <= bin_j < 9:
                cube[ti, bin_j] = m
                sem[ti, bin_j] = s


@torch.no_grad()
def multicol_cube(session, z, all_cells=False, group_list=None):
    pack = session.primary_pack
    schema = list(session.schema)
    p = fc.assign_params(z, schema, session.backend)
    sig = pack.signal if pack.signal.dim() == 3 else pack.signal.unsqueeze(0)
    model_full, vm_ref = fc._run_conductance_full(session, p, sig, return_ref=True)
    sel = fc._readout_model_traces_pack(model_full, pack)
    scale = fc._pack_out_scale(p, pack, session.backend)
    plot_traces = fc.expand_plot_traces(sel, scale, session.maxtime).cpu().numpy()
    vm_ref = vm_ref[0].cpu().numpy()
    readout, radius, type_idx, type_names = _network_readout_layout(pack, session.backend.network)
    type_ids = session.backend.network.node_type.cpu().numpy()
    if all_cells:
        names = [str(n) for n in type_names]
    else:
        names = tile_model_data_names(session, pack.name, group_list)
    mt = session.maxtime
    cube = np.zeros((len(names), 9, mt))
    sem = np.zeros((len(names), 9, mt))
    baselines = baselines_for_types(
        pack, session.backend, vm_ref, names, type_ids, type_names,
    )
    center = CENTER_BIN
    for ti, ft in enumerate(names):
        ft_global = type_names.index(ft)
        _fill_ring_cube(cube, sem, ti, ft_global, type_idx, radius, plot_traces, center)
    return names, cube, sem, baselines


def _prepare_borst_tile(session, z, all_cells, group_list):
    model, ref = calc_model_full_all(session, z, return_ref=True)
    if all_cells:
        names = [str(CTYPE[i]) for i in range(session.backend.n_types)]
        cells = [
            dict(name=names[i], cube=model[i], sem=None, baseline=_baseline_from_ref_grid(ref, i))
            for i in range(len(names))
        ]
        return cells, None
    groups = tile_model_data_groups(session, session.primary_pack.name, group_list)
    cells = []
    group_rows = []
    for names_row in groups:
        row_idx = []
        for name in names_row:
            name = str(name)
            ctype_i = int(np.where(CTYPE == name)[0][0])
            row_idx.append(len(cells))
            cells.append(dict(
                name=name, cube=model[ctype_i], sem=None,
                baseline=_baseline_from_ref_grid(ref, ctype_i),
            ))
        group_rows.append(row_idx)
    return cells, group_rows


def _prepare_network_tile(session, z, all_cells, group_list):
    names, cube, sem, baselines = multicol_cube(session, z, all_cells=all_cells, group_list=group_list)
    center_only = center_column_only(session)
    cells = [
        dict(name=n, cube=cube[i], sem=None if center_only else sem[i], baseline=baselines.get(n))
        for i, n in enumerate(names)
    ]
    return cells, None


def _plot_tile_figure(
    session_on,
    z,
    path,
    *,
    prepare_fn,
    session_off=None,
    all_cells=False,
    title,
    ref_cubes=None,
    ref_cubes_off=None,
    group_list=None,
    ncols,
    figsize_fn,
    gridspec_kw,
    suptitle_fs=12,
):
    cells_on, group_rows = prepare_fn(session_on, z, all_cells, group_list)
    cells_off = None
    if session_off is not None:
        cells_off, _ = prepare_fn(session_off, z, all_cells, group_list)
    dual = session_off is not None
    ref_on, ref_off = _resolve_tile_ref_cubes(
        session_on, session_off, ref_cubes, ref_cubes_off,
    )
    if group_rows is not None:
        nrows = 2 * len(group_rows)
    else:
        nrows = 2 * ((len(cells_on) + ncols - 1) // ncols)
    fig = plt.figure(figsize=figsize_fn(ncols, nrows))
    gs = fig.add_gridspec(nrows, ncols, **gridspec_kw)
    legend_done = False
    maxtime = session_on.maxtime

    def _plot_cell(cell_on, cell_off, ax_rf, ax_time, show_ylabel, show_xlabels):
        nonlocal legend_done
        name = cell_on['name']
        off_kw = {}
        if dual and cell_off is not None:
            off_kw = {
                'off_model_xt': cell_off['cube'],
                'off_ref_xt': ref_off.get(name) if ref_off else None,
                'off_baseline': cell_off.get('baseline'),
            }
        plot_cell_pair(
            ax_rf, ax_time, cell_on['cube'], ref_on.get(name), name,
            sem_xt=cell_on.get('sem'),
            show_legend=not legend_done,
            show_xlabels=show_xlabels,
            show_ylabel=show_ylabel,
            maxtime=maxtime,
            baseline=cell_on.get('baseline'),
            **off_kw,
        )
        legend_done = True

    if group_rows is not None:
        for gi, row_idx in enumerate(group_rows):
            rf_row = 2 * gi
            start = (ncols - len(row_idx)) // 2
            for j, ci in enumerate(row_idx):
                col = start + j
                cell_off = cells_off[ci] if dual else None
                ax_rf = fig.add_subplot(gs[rf_row, col])
                ax_time = fig.add_subplot(gs[rf_row + 1, col])
                _plot_cell(cells_on[ci], cell_off, ax_rf, ax_time, show_ylabel=(j == 0), show_xlabels=True)
    else:
        for i, cell_on in enumerate(cells_on):
            blk, col = divmod(i, ncols)
            cell_off = cells_off[i] if dual else None
            ax_rf = fig.add_subplot(gs[2 * blk, col])
            ax_time = fig.add_subplot(gs[2 * blk + 1, col])
            show_xlabels = True
            if all_cells and prepare_fn is _prepare_borst_tile:
                show_xlabels = (blk == (len(cells_on) + ncols - 1) // ncols - 1)
            _plot_cell(cell_on, cell_off, ax_rf, ax_time, show_ylabel=(col == 0), show_xlabels=show_xlabels)
    fig.suptitle(title, fontsize=suptitle_fs)
    save_figure(fig, path, dpi=150)


def plot_network_tile(session_on, z, path, *, session_off=None, all_cells=False,
                      title, ref_cubes=None, ref_cubes_off=None, group_list=None):
    ncols = 5 if not all_cells else 8
    _plot_tile_figure(
        session_on, z, path,
        prepare_fn=_prepare_network_tile,
        session_off=session_off,
        all_cells=all_cells,
        title=title,
        ref_cubes=ref_cubes,
        ref_cubes_off=ref_cubes_off,
        group_list=group_list,
        ncols=ncols,
        figsize_fn=lambda c, r: (3.0 * c if not all_cells else 2.2 * c, 2.5 * r),
        gridspec_kw=dict(hspace=0.55, wspace=0.55, top=0.93, bottom=0.06, left=0.07, right=0.98),
    )


def plot_borst_tile(session_on, z, path, *, session_off=None, all_cells=False,
                    title, ref_cubes=None, ref_cubes_off=None, group_list=None):
    ncols = 13
    if all_cells:
        gs_kw = dict(hspace=0.65, wspace=0.45, top=0.97, bottom=0.03, left=0.04, right=0.99)
        figsize_fn = lambda c, r: (26, 32)
        suptitle_fs = 14
    else:
        gs_kw = dict(hspace=0.5, wspace=0.55, top=0.95, bottom=0.05, left=0.06, right=0.98)
        figsize_fn = lambda c, r: (16, 2.5 * r)
        suptitle_fs = 12
    _plot_tile_figure(
        session_on, z, path,
        prepare_fn=_prepare_borst_tile,
        session_off=session_off,
        all_cells=all_cells,
        title=title,
        ref_cubes=ref_cubes,
        ref_cubes_off=ref_cubes_off,
        group_list=group_list,
        ncols=ncols,
        figsize_fn=figsize_fn,
        gridspec_kw=gs_kw,
        suptitle_fs=suptitle_fs,
    )


def reference_cube(name, ref_cubes=None, dark=False):
    if ref_cubes is not None:
        return ref_cubes.get(str(name))
    return borst_ref_cubes(dark=dark).get(str(name))
