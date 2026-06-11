#!/usr/bin/env python
"""Short local CPU training run with cost and model-vs-data plots."""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

import Medulla_Library as ml
import blindschleiche_py3 as bs
from FiveCol_MedSim_Pytorch import (
    Ca_tau,
    E_leak,
    calc_cost,
    data,
    data_amp,
    deltat,
    device,
    guess_initial_params,
    mc_cell_index,
    nofcells,
    signal,
    update_Vm,
    z_bounds,
)

CELL_LIST = np.array(
    ['L1', 'L2', 'L3', 'L4', 'L5', 'Mi1', 'Tm3', 'Mi4', 'Mi9', 'Tm1', 'Tm2', 'Tm4', 'Tm9']
)
CENTER_COL = 2  # center of the 5 columns used in the cost function
OUTDIR = 'FiveCol_Parameter/local_cpu_test'
CTYPE = np.load('Circuits/ctype.npy', allow_pickle=True)
FIT_INDEX = {name: i for i, name in enumerate(CELL_LIST)}
CENTER_NEURON_OFFSET = CENTER_COL * nofcells


def calc_multi_col_params(param):
    return torch.concatenate((param, param, param, param, param))


def assign_params(z):
    inp_gain = calc_multi_col_params(z[0:65])
    out_gain = calc_multi_col_params(z[65:130])
    interim = torch.zeros(65, dtype=torch.float64, device=z.device)
    interim[8:13] = z[130:135]
    ih_gmax = calc_multi_col_params(interim)
    return inp_gain, out_gain, ih_gmax, z[135], z[136], z[137]


@torch.no_grad()
def _simulate_filtered_traces(z, neuron_index):
    if not torch.is_tensor(neuron_index):
        neuron_index = torch.tensor(neuron_index, dtype=torch.long, device=z.device)
    else:
        neuron_index = neuron_index.to(z.device)

    inp_gain, out_gain, ih_gmax, ih_midv, ih_slope, tau_midv = assign_params(z)
    u = torch.zeros(325, dtype=torch.float64, device=z.device)
    vm = E_leak.clone()
    n = len(neuron_index)

    for t in range(1, 50):
        vm, u = update_Vm(vm, u, inp_gain, out_gain, ih_gmax, ih_midv, ih_slope, tau_midv, signal[t - 1])

    vm_ref = vm[neuron_index].clone()
    trace = torch.zeros(n, 200, dtype=torch.float64, device=z.device)
    filtered = torch.zeros(n, dtype=torch.float64, device=z.device)

    for t in range(50, 200):
        vm, u = update_Vm(vm, u, inp_gain, out_gain, ih_gmax, ih_midv, ih_slope, tau_midv, signal[t - 1])
        filtered = deltat / Ca_tau * (vm[neuron_index] - vm_ref - filtered) + filtered
        trace[:, t] = filtered

    trace[:, 0:50] = 0
    trace[:, 0:199] = trace[:, 1:200]
    return trace


def calc_model_trace(z):
    return _simulate_filtered_traces(z, mc_cell_index)


def calc_center_column_trace(z):
    center_index = torch.arange(
        CENTER_NEURON_OFFSET,
        CENTER_NEURON_OFFSET + nofcells,
        dtype=torch.long,
        device=z.device,
    )
    return _simulate_filtered_traces(z, center_index)


def calc_model_full_all(z):
    """All cell types across 5 columns -> (65, 9, 200) spatio-temporal cube."""
    model_full = np.zeros((nofcells, 9, 200))
    for col in range(5):
        col_index = torch.arange(col * nofcells, (col + 1) * nofcells, dtype=torch.long, device=z.device)
        model_full[:, col + 2] = _simulate_filtered_traces(z, col_index).cpu().numpy()
    return model_full


def _scaled_rf(model_xt, ref_xt, center):
    imp_model = model_xt[center]
    maxt_model = int(np.argmax(np.abs(imp_model)))
    rf_model = bs.blurr(bs.rebin(model_xt[:, maxt_model], 45), 5)
    amp_model = float(np.max(np.abs(imp_model)))
    rf_model = rf_model / (np.max(np.abs(rf_model)) + 1e-12) * amp_model

    rf_data = None
    imp_data = None
    if ref_xt is not None:
        imp_data = ref_xt[center]
        maxt_data = int(np.argmax(np.abs(imp_data)))
        rf_data = bs.blurr(bs.rebin(ref_xt[:, maxt_data], 45), 5)
        amp_data = float(np.max(np.abs(imp_data)))
        rf_data = rf_data / (np.max(np.abs(rf_data)) + 1e-12) * amp_data

    return imp_model, imp_data, np.roll(rf_model, -2), None if rf_data is None else np.roll(rf_data, -2)


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


def _plot_cell_pair_axes(ax_rf, ax_time, model_xt, ref_xt, title, show_legend=False,
                         show_xlabels=False, show_ylabel=False):
    """Match Borst_Fig4-6: azimuth RF on top, time response below."""
    center = CENTER_COL + 2
    imp_model, imp_data, rf_model, rf_data = _scaled_rf(model_xt, ref_xt, center)
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
    if show_legend:
        ax_rf.legend(loc='upper right', fontsize=6, frameon=False)

    ax_time.plot(imp_data, color='gray', linewidth=1.5) if imp_data is not None else None
    ax_time.plot(imp_model, color='red', linewidth=1.5)
    ax_time.set_ylim(ylo, yhi)
    _style_time_axis(ax_time, show_xlabels)
    if show_ylabel:
        ax_time.set_ylabel('mV', fontsize=7)
    ax_time.tick_params(labelsize=6)


def train_with_history(n_steps=100, lr=0.1):
    z = nn.Parameter(guess_initial_params().clone())
    optimizer = torch.optim.Adam([z], lr=lr)
    costs = []

    for _ in range(n_steps):
        optimizer.zero_grad()
        cost = calc_cost(z, data)
        costs.append(cost.item())
        cost.backward()
        optimizer.step()
        with torch.no_grad():
            z.clamp_(z_bounds[:, 0].to(device), z_bounds[:, 1].to(device))

    return z.detach(), costs


def plot_cost(costs, path):
    plt.figure(figsize=(8, 4))
    plt.plot(costs, color='steelblue', linewidth=2)
    plt.xlabel('step')
    plt.ylabel('cost [% data power]')
    plt.title(f'Local CPU training ({len(costs)} steps)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_model_vs_data(z, path, n_steps):
    ref_data = ml.read_RecF_data() * data_amp
    model_full = calc_model_full_all(z)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(4, 13, hspace=0.5, wspace=0.55,
                          top=0.93, bottom=0.06, left=0.06, right=0.98)

    layout = [
        (CELL_LIST[0:5], 0, [4, 5, 6, 7, 8]),
        (CELL_LIST[5:9], 2, [1, 2, 3, 4]),
        (CELL_LIST[9:13], 2, [8, 9, 10, 11]),
    ]
    legend_done = False
    for names, rf_row, cols in layout:
        for i, (name, col) in enumerate(zip(names, cols)):
            fit_i = int(np.where(CELL_LIST == name)[0][0])
            ctype_i = int(np.where(CTYPE == name)[0][0])
            ax_rf = fig.add_subplot(gs[rf_row, col])
            ax_time = fig.add_subplot(gs[rf_row + 1, col])
            _plot_cell_pair_axes(
                ax_rf, ax_time, model_full[ctype_i], ref_data[fit_i], name,
                show_legend=not legend_done,
                show_xlabels=True,
                show_ylabel=(name in ('L1', 'Mi1', 'Tm1')),
            )
            legend_done = True

    fig.suptitle(f'Model vs data after {n_steps} CPU steps (center column)', fontsize=12)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_all_celltypes(z, path, n_steps):
    """All 65 cell types: azimuth RF top, time bottom (Borst_Fig4-6 layout)."""
    ref_data = ml.read_RecF_data() * data_amp
    model_full = calc_model_full_all(z)

    ncols = 13
    fig = plt.figure(figsize=(26, 32))
    gs = fig.add_gridspec(10, ncols, hspace=0.65, wspace=0.45,
                          top=0.97, bottom=0.03, left=0.04, right=0.99)

    for i in range(nofcells):
        row, col = divmod(i, ncols)
        name = str(CTYPE[i])
        ref_xt = ref_data[FIT_INDEX[name]] if name in FIT_INDEX else None
        ax_rf = fig.add_subplot(gs[row * 2, col])
        ax_time = fig.add_subplot(gs[row * 2 + 1, col])
        _plot_cell_pair_axes(
            ax_rf, ax_time, model_full[i], ref_xt, name,
            show_legend=(i == 0),
            show_xlabels=(row == 4),
            show_ylabel=(col == 0),
        )

    fig.suptitle(f'All {nofcells} cell types after {n_steps} steps', fontsize=14)
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    os.makedirs(OUTDIR, exist_ok=True)
    print(f'device={device}')

    z_fit, costs_hist = train_with_history(n_steps=100, lr=0.1)
    final_cost = calc_cost(z_fit, data).item()

    print(f'initial cost = {costs_hist[0]:.4f}')
    print(f'final cost   = {final_cost:.4f}')
    print(f'best cost    = {min(costs_hist):.4f}')

    cost_path = os.path.join(OUTDIR, 'cost_curve.png')
    model_path = os.path.join(OUTDIR, 'model_vs_data.png')
    all_cells_path = os.path.join(OUTDIR, 'model_all_cells.png')
    plot_cost(costs_hist, cost_path)
    plot_model_vs_data(z_fit, model_path, n_steps=100)
    plot_all_celltypes(z_fit, all_cells_path, n_steps=100)

    np.save(os.path.join(OUTDIR, 'params_100steps.npy'), z_fit.cpu().numpy())
    np.save(os.path.join(OUTDIR, 'costs.npy'), np.array(costs_hist))
    print(f'saved: {cost_path}')
    print(f'saved: {model_path}')
    print(f'saved: {all_cells_path}')
