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
from t4_t5_preference import READOUT_SUBTYPES, active_stimuli_for_subtype, normalize_side, fig1_key_for_stimulus
from training_io import COST_HALF_WINDOW_STEPS, COST_WINDOW_STEPS
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

CELL_LIST = np.array(
    ['L1', 'L2', 'L3', 'L4', 'L5', 'Mi1', 'Tm3', 'Mi4', 'Mi9', 'Tm1', 'Tm2', 'Tm4', 'Tm9']
)
CENTER_COL = ml.CENTER_COL
CTYPE = np.load('Circuits/ctype.npy', allow_pickle=True)
FIT_INDEX = {name: i for i, name in enumerate(CELL_LIST)}
CENTER_NEURON_OFFSET = ml.column_start(CENTER_COL)

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
    ref = ml.read_RecF_data() * ml.DATA_AMP                 # (13, 9, maxtime)
    return {name: ref[i] for i, name in enumerate(CELL_LIST)}


def reference_cube(name):
    """(9, maxtime) grey reference for a cell name, or None if none is registered."""
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
    """Repack the core forward's (maxtime-t_on, n) center-window response into the plot's
    (n, maxtime) trace: pre-stimulus zeroed, optional out_scale applied, then shifted
    one step to match the data convention. All model dynamics live in the core
    (fc._run_conductance / fc._run_adaptive); this is presentation-only reshaping."""
    n = stacked.shape[1]
    trace = torch.zeros(n, maxtime, dtype=torch.float64, device=stacked.device)
    trace[:, t_on:maxtime] = stacked.transpose(0, 1)
    trace = trace * _out_scale_vec(z, neuron_index, schema)
    trace[:, 0:t_on] = 0
    trace[:, 0:maxtime - 1] = trace[:, 1:maxtime]
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
    """All cell types across 5 columns -> (65, 9, maxtime) spatio-temporal cube.

    With return_ref=True also returns the per-cell resting baseline (the value
    each trace is measured relative to) as a (65, 9) array; NaN where no column
    was simulated.
    """
    model_full = np.zeros((nofcells, 9, maxtime))
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


# ---- network multi-column cube (averaged over tiles x shifts x ring) ------

@torch.no_grad()
def _multicol_cube(z):
    """Build a (n_fit, 9, maxtime) model cube + SEM by averaging the batched forward.

    Runs the 7-shift (B) network forward, then for each fit cell type bins every
    readout cell by its ring radius. Following the single-column azimuth convention
    the ring radius is truncated with int() for DISPLAY (so sqrt(3) -> bin 1, same
    as col +/-1); training itself keeps sqrt(3) and 2 as distinct rings. Each bin is
    averaged over all tiles x shifts x ring members; sem = std/sqrt(n).
    Returns (names, cube, sem) for the present fit types.
    """
    p = fc.assign_params(z, fc.CONDUCTANCE_SCHEMA)
    model_full = fc._run_conductance_full(p, fc.signal)        # (B, maxtime-t_on, N)
    b_idx, u_idx = fc.READOUT
    sel = model_full[b_idx, :, u_idx].cpu().numpy()            # (n_cost, maxtime-t_on)
    radius = fc.MC_COST_RADIUS.cpu().numpy()                   # (n_cost,)
    type_idx = fc.NETWORK.node_type[u_idx].cpu().numpy()    # (n_cost,)
    type_names = list(fc.NETWORK.type_names)

    names = [ft for ft in CELL_LIST if ft in type_names]
    cube = np.zeros((len(names), 9, maxtime))
    sem = np.zeros((len(names), 9, maxtime))
    center = 4
    for ti, ft in enumerate(names):
        ft_global = type_names.index(ft)
        for off in range(5):                                  # azimuth offset 0..4
            mask = (type_idx == ft_global) & (np.floor(radius).astype(int) == off)
            if not mask.any():
                continue
            traces = sel[mask]                                # (k, maxtime-t_on)
            m = traces.mean(axis=0)
            s = traces.std(axis=0) / np.sqrt(traces.shape[0])
            for bin_j in {center + off, center - off}:        # mirror to both sides
                if 0 <= bin_j < 9:
                    cube[ti, bin_j, t_on:maxtime] = m
                    sem[ti, bin_j, t_on:maxtime] = s
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

    Used for network multi-column plots, where each trace is a mean over many
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

    t = np.arange(maxtime)
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
    )
    return True


def _moving_bar_center_only():
    """True when moving-bar cost/plot should use the hex centre column only."""
    if getattr(fc, 'MOVING_BAR_CENTER_COLUMN', False):
        return True
    opts = getattr(fc, 'NETWORK_TRAIN_OPTS', None) or {}
    return bool(opts.get('moving_bar_center_column', False))


def _moving_bar_ylim(model_mean, model_sem, data_mean, keys, show_sem=False):
    """Shared y-limits for a batch of moving-bar panels."""
    curves = []
    for key in keys:
        m = model_mean[key]
        curves.append(m)
        d = data_mean.get(key) if data_mean else None
        if d is not None:
            curves.append(d)
        if show_sem and key in model_sem:
            s = model_sem[key]
            if np.any(s):
                curves.extend([m + s, m - s])
    return _nice_ylim(*curves)


def _save_moving_bar_fig(fig, path, dpi, rasterize=True):
    if rasterize:
        for ax in fig.axes:
            ax.set_rasterized(True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _network_uv_np(C):
    """Column (u, v) per unit as int64 numpy arrays (network may store torch or numpy)."""
    u = C.u.detach().cpu().numpy() if torch.is_tensor(C.u) else np.asarray(C.u)
    v = C.v.detach().cpu().numpy() if torch.is_tensor(C.v) else np.asarray(C.v)
    return u.astype(np.int64), v.astype(np.int64)


def _network_type_ids(C):
    node_type = C.node_type
    if torch.is_tensor(node_type):
        node_type = node_type.detach().cpu().numpy()
    return np.asarray(node_type, dtype=np.int64)


def _moving_bar_t0_grid(C, cols, n_batch, t0_map):
    """Per-(stimulus, unit) window start step; -1 where the unit is not on a photo column."""
    n_units = C.n_units
    u_np, v_np = _network_uv_np(C)
    t0_bn = np.full((n_batch, n_units), -1, dtype=np.int64)
    for bi in range(n_batch):
        for c in cols:
            t0 = t0_map.get((bi, int(c.u), int(c.v)))
            if t0 is None:
                continue
            on_col = (u_np == int(c.u)) & (v_np == int(c.v))
            t0_bn[bi, on_col] = t0
    return t0_bn


def _extract_moving_bar_windows(model_full, t0_bn):
    """Batched ``t_center ± 0.45 s`` windows from ``model_full`` (B, T', N)."""
    n_batch, t_len, n_units = model_full.shape
    win = np.arange(COST_WINDOW_STEPS, dtype=np.int64)
    t_rel = t0_bn[:, :, None].astype(np.int64) - int(t_on) + win[None, None, :]
    t_max = t_len - 1
    pre = t_rel < 0
    t_safe = np.clip(t_rel, 0, t_max)
    b_ix = np.arange(n_batch, dtype=np.int64)[:, None, None]
    u_ix = np.arange(n_units, dtype=np.int64)[None, :, None]
    out = model_full[b_ix, t_safe, u_ix].astype(np.float64, copy=False)
    out[pre] = 0.0
    return out


def _aggregate_moving_bar_traces(windows, t0_bn, type_ids, types, spec_names, center_only):
    """Mean/SEM per (cell type, stimulus) from ``windows`` (B, N, W)."""
    model_mean, model_sem = {}, {}
    valid = t0_bn >= 0
    for ti, tname in enumerate(types):
        type_mask = type_ids == ti
        if not type_mask.any():
            continue
        for bi, sname in enumerate(spec_names):
            unit_mask = valid[bi] & type_mask
            if not unit_mask.any():
                continue
            arr = windows[bi, unit_mask]
            key = (tname, sname)
            model_mean[key] = arr.mean(axis=0)
            if center_only or arr.shape[0] == 1:
                model_sem[key] = np.zeros(COST_WINDOW_STEPS, dtype=np.float64)
            else:
                model_sem[key] = arr.std(axis=0) / np.sqrt(arr.shape[0])
    return model_mean, model_sem


def _unit_window_trace(model_bt, u, t0):
    """One unit's model trace on ``t_center ± 0.45 s`` from a single-batch forward."""
    t0_bn = np.full((1, model_bt.shape[1]), -1, dtype=np.int64)
    t0_bn[0, u] = t0
    return _extract_moving_bar_windows(model_bt[None, ...], t0_bn)[0, u]


@torch.no_grad()
def _compute_moving_bar_all_type_traces(z):
    """Model/data traces for every network type × 16 stimuli (shared plot core)."""
    from network.moving_bar_target import load_fig1_trace
    from network.stimulus import build_moving_bar_signals, center_photo_column, photo_columns

    specs = gruntman_moving_bar_specs()
    spec_names = [s.name for s in specs]
    C = fc.NETWORK
    side = normalize_side(C.meta.get('side', 'right'))
    center_only = _moving_bar_center_only()
    center_col = center_photo_column(C)
    cols = [center_col] if center_only else photo_columns(C)

    p = fc.assign_params(z, fc.CONDUCTANCE_SCHEMA)
    model_full = fc._run_conductance_full(p, fc.signal).cpu().numpy()
    field_deg = C.meta.get('field_deg')
    if field_deg is None:
        field_deg = build_moving_bar_signals(
            C, t_on=t_on, deltat_ms=fc.deltat, device=device,
        ).info['field_deg']

    t0_map = {}
    for bi, spec in enumerate(specs):
        for c in cols:
            t_center = column_bar_center_step(
                c.x, c.y, spec, field_deg, t_on=t_on, deltat_ms=fc.deltat,
            )
            t0_map[(bi, int(c.u), int(c.v))] = int(t_center - COST_HALF_WINDOW_STEPS)

    types = list(C.type_names)
    type_ids = _network_type_ids(C)
    t0_bn = _moving_bar_t0_grid(C, cols, len(specs), t0_map)
    windows = _extract_moving_bar_windows(model_full, t0_bn)
    model_mean, model_sem = _aggregate_moving_bar_traces(
        windows, t0_bn, type_ids, types, spec_names, center_only,
    )
    data_mean = {}

    # Grey data: fig1 targets for T4/T5 subtypes (same source as training).
    for subtype in READOUT_SUBTYPES:
        if subtype not in types:
            continue
        for bi, spec in enumerate(specs):
            trace_id = fig1_key_for_stimulus(side, subtype, spec)
            if trace_id is None:
                continue
            data_mean[(subtype, spec.name)] = load_fig1_trace(trace_id)

    return types, spec_names, model_mean, model_sem, data_mean


@torch.no_grad()
def _moving_bar_mean_traces(z):
    """Model/data traces per (subtype, stimulus) for the 8×8 moving-bar grid."""
    side = normalize_side(fc.NETWORK.meta.get('side', 'right'))
    _, _, model_mean, model_sem, data_mean = _compute_moving_bar_all_type_traces(z)
    row_specs = {
        st: [f'{d}_{c}_{w}' for d, c, w in active_stimuli_for_subtype(side, st)]
        for st in READOUT_SUBTYPES
    }
    return row_specs, model_mean, model_sem, data_mean


def _set_moving_bar_xlim(ax):
    ax.set_xlim(0, COST_WINDOW_STEPS)


def _set_moving_bar_xticks(ax):
    mid = COST_HALF_WINDOW_STEPS
    end = COST_WINDOW_STEPS
    ax.set_xticks([0, mid, end])
    ax.set_xticklabels(['-0.45', '0', '0.45'], fontsize=6)


def _moving_bar_right_spec_names(spec_names):
    """Gruntman 16-spec grid: keep only right-moving bar columns."""
    return [s for s in spec_names if s.startswith('right_')]


def _moving_bar_hide_ticks(ax):
    ax.tick_params(
        bottom=False, top=False, left=False, right=False,
        labelbottom=False, labeltop=False, labelleft=False, labelright=False,
    )


def _moving_bar_right_ticks(ax):
    ax.tick_params(
        bottom=False, top=False, left=False, right=True,
        labelbottom=False, labeltop=False, labelleft=False, labelright=True,
        labelsize=6,
    )


def _style_moving_bar_time_axis(ax, show_xlabel=False):
    """Moving-bar time axis with explicit ticks on the window."""
    _set_moving_bar_xlim(ax)
    _set_moving_bar_xticks(ax)
    if show_xlabel:
        ax.set_xlabel('time [s]', fontsize=7)


def _plot_moving_bar_cell(ax, model_trace, sem_trace, data_trace, title,
                        show_ylabel=False, show_sem=True, ylim=None, cell_ticks=True,
                        show_xticks=True):
    curves = [model_trace]
    if data_trace is not None:
        curves.append(data_trace)
    if show_sem:
        curves.extend([model_trace + sem_trace, model_trace - sem_trace])
    if ylim is None:
        ylo, yhi = _nice_ylim(*curves)
    else:
        ylo, yhi = ylim
    if data_trace is not None:
        ax.plot(data_trace, color='gray', linewidth=1.5)
    if show_sem and np.any(sem_trace):
        ax.fill_between(_MOVING_BAR_T, model_trace - sem_trace, model_trace + sem_trace,
                        color='pink', alpha=0.8, linewidth=0)
    ax.plot(model_trace, color='red', linewidth=1.5)
    ax.set_title(title, fontsize=7, pad=2)
    ax.set_ylim(ylo, yhi)
    _set_moving_bar_xlim(ax)
    if show_xticks:
        _set_moving_bar_xticks(ax)
    if show_ylabel:
        ax.set_ylabel('mV', fontsize=7)
    if cell_ticks:
        ax.tick_params(labelsize=6)


def plot_model_vs_data_moving_bar(z, path, title=None):
    """Moving-bar model-vs-data: 8 subtypes × 8 stimuli, time window only."""
    center_only = _moving_bar_center_only()
    row_specs, model_mean, model_sem, data_mean = _moving_bar_mean_traces(z)
    nrows = len(READOUT_SUBTYPES)
    ncols = 8
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.2 * ncols, 1.8 * nrows), sharex=True,
    )
    if nrows == 1:
        axes = np.asarray([axes])
    for ri, subtype in enumerate(READOUT_SUBTYPES):
        for ci, sname in enumerate(row_specs[subtype]):
            ax = axes[ri, ci]
            key = (subtype, sname)
            if key not in model_mean:
                ax.axis('off')
                continue
            _plot_moving_bar_cell(
                ax, model_mean[key], model_sem[key], data_mean[key],
                sname,
                show_ylabel=(ci == 0),
                show_sem=not center_only,
            )
        axes[ri, 0].set_ylabel(subtype, fontsize=8, labelpad=12)
    if title is None:
        title = 'Moving-bar model vs data'
    if center_only:
        from network.stimulus import center_photo_column
        col = center_photo_column(fc.NETWORK)
        scope = f'centre column (u,v)=({col.u},{col.v})'
    else:
        from network.stimulus import photo_columns
        scope = f'avg over {len(photo_columns(fc.NETWORK))} photo columns'
    fig.suptitle(
        title + f'  [{scope}, t_center ± 0.45 s]',
        fontsize=12,
    )
    fig.subplots_adjust(top=0.92, bottom=0.08, hspace=0.45, wspace=0.35)
    _save_moving_bar_fig(fig, path, MOVING_BAR_MVD_DPI)


@torch.no_grad()
def plot_all_celltypes_moving_bar(z, path, title=None):
    """All network types × right-direction stimuli (4 cols); grey data on T4/T5."""
    t0 = time.perf_counter()
    center_only = _moving_bar_center_only()
    types, all_spec_names, model_mean, model_sem, data_mean = _compute_moving_bar_all_type_traces(z)
    spec_names = _moving_bar_right_spec_names(all_spec_names)
    t_traces = time.perf_counter() - t0

    keys = [(t, s) for t in types for s in spec_names if (t, s) in model_mean]
    show_sem = not center_only
    ylim = _moving_bar_ylim(model_mean, model_sem, data_mean, keys, show_sem=show_sem)

    nrows = len(types)
    ncols = len(spec_names)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(1.4 * ncols, 0.85 * nrows), sharex=True, sharey=True,
    )
    if nrows == 1:
        axes = np.asarray([axes])
    if ncols == 1:
        axes = axes[:, None]

    t1 = time.perf_counter()
    for ri, tname in enumerate(types):
        for ci, sname in enumerate(spec_names):
            ax = axes[ri, ci]
            key = (tname, sname)
            if key not in model_mean:
                ax.axis('off')
                continue
            _plot_moving_bar_cell(
                ax, model_mean[key], model_sem[key], data_mean.get(key),
                sname if ri == 0 else sname,
                show_ylabel=(ci == 0),
                show_sem=show_sem and key in model_sem and np.any(model_sem[key]),
                ylim=ylim,
                cell_ticks=False,
                show_xticks=(ri == nrows - 1),
            )
        if ncols:
            axes[ri, 0].set_ylabel(tname, fontsize=6, labelpad=4)
    if title is None:
        title = 'Moving-bar all cell types (right only)'
    if center_only:
        from network.stimulus import center_photo_column
        col = center_photo_column(fc.NETWORK)
        scope = f'centre column (u,v)=({col.u},{col.v})'
    else:
        from network.stimulus import photo_columns
        scope = f'avg over {len(photo_columns(fc.NETWORK))} photo columns'
    fig.suptitle(title + f'  [{scope}, t_center ± 0.45 s]', fontsize=10)
    fig.subplots_adjust(top=0.96, bottom=0.05, hspace=0.55, wspace=0.3)
    t_draw = time.perf_counter() - t1
    t2 = time.perf_counter()
    _save_moving_bar_fig(fig, path, MOVING_BAR_GRID_DPI)
    t_save = time.perf_counter() - t2
    print(f'plot_all_celltypes_moving_bar: traces={t_traces:.1f}s  '
          f'draw={t_draw:.1f}s  savefig={t_save:.1f}s  total={t_traces+t_draw+t_save:.1f}s')


def plot_model_vs_data_network(z, path, title=None):
    """Network model-vs-data: each fit type's ring-averaged trace + SEM band."""
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
        title = 'Network model vs data'
    n_tiles = fc.NETWORK.meta.get('n_centers', '?') if fc.NETWORK else '?'
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


def select_best(params, costs=None):
    params = np.atleast_2d(params)
    valid_mask = np.any(params != 0, axis=1)
    valid = params[valid_mask]
    if len(valid) == 0:
        raise SystemExit('no trained parameter sets found (file all zeros)')
    if costs is not None:
        costs = np.asarray(costs, dtype=np.float64).ravel()
        valid_costs = costs[valid_mask]
        if len(valid_costs) == len(valid):
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
    (``model_all_cells.png``), which dominated plot time (~5 min CPU savefig).
    The 64-panel ``model_vs_data.png`` is always written.
    """
    os.makedirs(outdir, exist_ok=True)
    if model_type is not None:
        fc.MODEL_TYPE = model_type

    print(f'plot device={_plot_device_label()}')
    best, best_cost = select_best(params, costs=costs)
    z = torch.tensor(best, dtype=torch.float64, device=device)

    if costs is not None:
        plot_cost(costs, os.path.join(outdir, 'cost_curve.png'))

    if getattr(fc, 'NETWORK', None) is None:
        restore_fc_context(outdir)

    suffix = f'trained, cost {best_cost:.2f}% of data power'
    mvd = os.path.join(outdir, 'model_vs_data.png')
    allc = os.path.join(outdir, 'model_all_cells.png')
    if getattr(fc, 'NETWORK', None) is not None:
        if getattr(fc, 'TARGET_KIND', None) == 'moving_bar':
            plot_model_vs_data_moving_bar(
                z, mvd, title=f'Moving-bar model vs data ({suffix})',
            )
            plot_all_celltypes_moving_bar(
                z, allc, title=f'Moving-bar all cells ({suffix})',
            )
        else:
            plot_model_vs_data_network(
                z, mvd, title=f'Network model vs data ({suffix})',
            )
            plot_model_vs_data_network(
                z, allc, title=f'Network all cells ({suffix})',
            )
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
