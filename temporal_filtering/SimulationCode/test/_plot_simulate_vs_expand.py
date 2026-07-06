"""Plot tile _simulate vs pad_plot_traces stages (raw / padded / shifted).

Reuses plot.tile._simulate and fc.pad_plot_traces — no re-implementation.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import torch
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import Medulla_Library as ml
import FiveCol_MedSim_Pytorch as fc
from plot import tile as tile_plot
from plot_trained import resolve_model_type, _session_for_target
from training_config import PARAMETER_DIR

DEFAULT_RUN = PARAMETER_DIR / 'conductance' / 'run_20260704_133829'
CELLS = ('L4', 'T4a', 'R1')
OUT_NAME = 'simulate_vs_expand.png'


def _padded_scaled(raw_nt, scale, mt, t_on_step):
    """pad_plot_traces steps 1–2 only: scale, embed, zero pre-stimulus."""
    n, t_len = raw_nt.shape
    trace = torch.zeros(n, mt, dtype=raw_nt.dtype, device=raw_nt.device)
    trace[:, t_on_step:t_on_step + t_len] = scale[:, None] * raw_nt
    trace[:, 0:t_on_step] = 0
    return trace


def _load_session_and_z(rundir):
    import json
    best_path = os.path.join(rundir, 'best_param.npy')
    model_type = resolve_model_type(best_path)
    opts_path = os.path.join(rundir, 'train_opts.json')
    if os.path.isfile(opts_path):
        with open(opts_path) as f:
            opts = json.load(f)
        opts['target_list'] = ['tile_bright']
        session = fc.open_session({**opts, 'backend': opts.get('backend', 'borst')}, model_type)
    else:
        session = fc.open_session(
            fc.make_train_opts(backend='borst', target_list=['tile_bright']),
            model_type,
        )
    z = torch.tensor(np.load(best_path), dtype=torch.float64)
    return session, z


def _center_index(session, cell_name):
    if session.backend.network is not None:
        C = session.backend.network
        ti = list(C.type_names).index(cell_name)
        u_np = C.u.detach().cpu().numpy()
        v_np = C.v.detach().cpu().numpy()
        from network.stimulus import center_sti_column
        col = center_sti_column(C)
        mask = (C.node_type.cpu().numpy() == ti) & (u_np == col.u) & (v_np == col.v)
        if not mask.any():
            raise ValueError(f'no centre unit for {cell_name!r} in network')
        return int(np.where(mask)[0][0])
    return ml.center_unit_index(ml.type_index(cell_name))


@torch.no_grad()
def _traces_for_cell(session, z, cell_name):
    tile_session = _session_for_target(session, 'tile_bright')
    unit = _center_index(tile_session, cell_name)
    idx = torch.tensor([unit], dtype=torch.long, device=z.device)
    schema = list(tile_session.schema)
    p = fc.assign_params(z, schema, tile_session.backend)

    stacked, ref = fc._run_conductance(tile_session, p, neuron_index=idx, return_ref=True)
    raw_nt = stacked.transpose(0, 1)  # (N, T')
    scale = fc.out_scale_for_units(p, idx, tile_session.backend)
    mt = tile_session.maxtime
    t_on_step = fc.t_on

    padded = _padded_scaled(raw_nt, scale, mt, t_on_step)
    expanded = fc.pad_plot_traces(raw_nt, scale, mt, t_on_step=t_on_step)
    simulated, sim_ref = tile_plot._simulate(tile_session, z, idx, return_ref=True)

    cost_trace = torch.zeros(1, mt, dtype=raw_nt.dtype, device=raw_nt.device)
    cost_trace[:, t_on_step:t_on_step + raw_nt.shape[1]] = scale[:, None] * raw_nt

    return dict(
        cell=cell_name,
        vm_ref=float(ref[0]),
        sim_ref=float(sim_ref[0]),
        raw=raw_nt[0].cpu().numpy(),
        padded=padded[0].cpu().numpy(),
        expanded=expanded[0].cpu().numpy(),
        simulated=simulated[0].cpu().numpy(),
        cost=cost_trace[0].cpu().numpy(),
        scale=float(scale[0]),
    )


def _plot_cell(ax, traces, title):
    mt = len(traces['simulated'])
    t = np.arange(mt)
    t_on = fc.t_on
    t_ms = t * fc.deltat / 1000.0

    ax.plot(t_ms, traces['padded'], color='tab:blue', lw=1.2, ls='--',
            label='padded (scale×raw, no shift)')
    ax.plot(t_ms, traces['expanded'], color='tab:red', lw=2.0,
            label='pad_plot_traces')
    ax.plot(t_ms, traces['simulated'], color='tab:orange', lw=1.0, ls=':',
            label='_simulate')
    ax.plot(t_ms, traces['cost'], color='0.45', lw=1.0,
            label='cost path (scale×raw, no shift)')
    ax.axvline(t_on * fc.deltat / 1000.0, color='0.6', lw=0.6, ls=':')
    ax.axhline(0, color='0.3', lw=0.5)
    ax.set_title(
        f"{title}  (out_scale={traces['scale']:.3g}, "
        f"Vm_ref={traces['vm_ref']:.1f})",
        fontsize=9,
    )
    ax.set_xlim(0, mt * fc.deltat / 1000.0)
    ax.tick_params(labelsize=7)
    diff = np.max(np.abs(traces['expanded'] - traces['simulated']))
    ax.text(0.02, 0.95, f'max|expand−simulate|={diff:.2e}', transform=ax.transAxes,
            fontsize=7, va='top')


def main():
    rundir = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_RUN)
    cells = tuple(sys.argv[2].split(',')) if len(sys.argv) > 2 else CELLS
    session, z = _load_session_and_z(rundir)

    fig, axes = plt.subplots(len(cells), 1, figsize=(10, 2.6 * len(cells)), sharex=True)
    if len(cells) == 1:
        axes = [axes]

    results = []
    for ax, name in zip(axes, cells):
        tr = _traces_for_cell(session, z, name)
        results.append(tr)
        _plot_cell(ax, tr, name)
        if ax is axes[0]:
            ax.legend(loc='upper right', fontsize=7, frameon=False)
        if ax is axes[-1]:
            ax.set_xlabel('time [s]')
        ax.set_ylabel('mV', fontsize=8)

    fig.suptitle(
        f'tile _simulate vs pad_plot_traces  ({os.path.basename(rundir)})',
        fontsize=11,
    )
    fig.tight_layout()
    out = os.path.join(HERE, OUT_NAME)
    fig.savefig(out, dpi=130)
    print('saved ->', out)
    for tr in results:
        diff = np.max(np.abs(tr['expanded'] - tr['simulated']))
        shift = np.max(np.abs(tr['expanded'] - tr['padded']))
        print(f"{tr['cell']}: max|expand−simulate|={diff:.2e}  max|expand−padded|={shift:.4g}")


if __name__ == '__main__':
    main()
