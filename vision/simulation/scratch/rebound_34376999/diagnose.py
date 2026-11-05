"""Replay spread bright and replace parameters without changing either run.

Run from the repository root with vision/.venv/bin/python.
All interventions use the repository's forward implementation.
"""
from pathlib import Path
import sys
import json
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))
import import_bootstrap  # noqa: F401,E402
import numpy as np
import torch
import train

OUT = Path(__file__).resolve().parent
torch.set_num_threads(2)
RUNS = ROOT / '0_runs/hp_lp'
CELLS = ['L1', 'L2', 'L3', 'L4', 'L5', 'Mi1', 'Tm3', 'Mi4', 'Mi9', 'T4a']


def load(prefix):
    run = next(RUNS.glob(prefix + '-*'))
    opts = json.loads((run / 'data/train_opts.json').read_text())
    opts.update(tasks=['spread'], contrasts=['bright'], device='cpu')
    session = train.resolve_session(opts)
    raw = dict(np.load(run / 'data/best_param.npz', allow_pickle=True))
    z = train.z_from_node_vals(raw, session.schema, device='cpu')
    return session, train.params_from_z(z, session)


@torch.no_grad()
def main():
    s, p = load('34376999')
    s0, p0 = load('34376877')
    assert s.connectome.cells == s0.connectome.cells
    print('PARAMS', {k: list(v.shape) for k, v in p.items() if torch.is_tensor(v)}, flush=True)
    pack = s.primary_pack
    on = train.pack_t_onset(pack)
    off = on + round(s.train_opts['spread_sti_opts']['ms_sti'] / s.delta_ms)
    masks = {c: s.connectome.node_cells == s.connectome.cells.index(c) for c in CELLS}
    cases = [('rebound_run', p), ('spread_run', p0)]
    keys = ['a_h', 'a_in', 'a_out', 'syn_strength_cell', 'v_th', 'tau_lp',
            'tau_hp_rise', 'tau_hp_fall']
    for key in keys:
        q = dict(p)
        q[key] = p0[key]
        cases.append(('restore_' + key, q))
    for cells in [('L1',), ('L2',), ('L1', 'L2')]:
        for value in ['zero', 'old']:
            q = dict(p)
            q['a_h'] = p['a_h'].clone()
            for c in cells:
                q['a_h'][masks[c]] = 0 if value == 'zero' else p0['a_h'][masks[c]]
            cases.append(('a_h_' + '_'.join(cells) + '_' + value, q))
    for c in ['R1-6', 'L1', 'L2', 'L4', 'L5', 'Mi1', 'Mi9']:
        q = dict(p)
        q['a_out'] = p['a_out'].clone()
        mask = s.connectome.node_cells == s.connectome.cells.index(c)
        q['a_out'][mask] = p0['a_out'][mask]
        cases.append(('restore_a_out_' + c, q))
    for value in [0.6, 0.7, 0.8, 0.85, 0.95]:
        q = dict(p)
        q['a_h'] = p['a_h'].clone()
        q['a_h'][masks['L2']] = value
        cases.append(('L2_a_h_' + str(value), q))
    results, traces = {}, {}
    for name, q in cases:
        start = time.monotonic()
        v = train.forward_pack(s, q, s.pack_i_sti(pack), pack)[0]
        if v.ndim == 3:
            v = v[0]
        means = {c: v[:, mask].mean(-1).cpu().numpy() for c, mask in masks.items()}
        result = {}
        for c, a in means.items():
            baseline = float(a[on - 1])
            peak_idx = off + int(a[off:].argmax())
            result[c] = dict(baseline=baseline, post_peak=float(a[peak_idx]),
                             post_peak_delta=float(a[peak_idx]) - baseline,
                             post_peak_ms=float(peak_idx * s.delta_ms),
                             on_trough=float(a[on:off].min()))
        results[name] = result
        for c, a in means.items():
            traces[name + '__' + c] = a
        print(name, {c: round(result[c]['post_peak_delta'], 2) for c in CELLS},
              'seconds', round(time.monotonic() - start, 2), flush=True)
    (OUT / 'results.json').write_text(json.dumps(results, indent=2))
    np.savez_compressed(OUT / 'traces.npz', **traces)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    shown = [('rebound_run', '34376999'), ('spread_run', '34376877'),
             ('a_h_L2_old', 'restore L2 a_h'), ('restore_a_out_R1-6', 'restore R1-6 a_out'),
             ('a_h_L1_L2_zero', 'L1/L2 a_h = 0')]
    for ax, c in zip(axes.flat, ['L1', 'L2', 'L4', 'L5', 'Mi4', 'T4a']):
        for name, label in shown:
            a = traces[name + '__' + c]
            ax.plot(np.arange(len(a)) * s.delta_ms, a, label=label)
        ax.axvspan(on * s.delta_ms, off * s.delta_ms, color='0.9', zorder=-1)
        ax.set_title(c)
        ax.set_ylabel('mV')
        ax.set_xlabel('time (ms)')
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'ablation.png', dpi=160)

    # Reuse the shared component forward, with all network nodes of each cell
    # to match the input figures (rather than only nodes with cost entries).
    import analyze.cell_dynamics as dynamics
    selected = ['L1', 'L2', 'L4', 'Mi4']
    nodes = {c: torch.where(masks[c])[0].cpu().numpy() for c in selected}
    n_t = int(pack.i_sti.shape[-2])
    plan = dynamics._build_component_b(
        nodes, t0_bn_row=np.zeros(s.connectome.n_node, dtype=np.int64),
        n_t_aligned=n_t)
    reports = dynamics._analyze_component_forward(
        s, params=p, cells=selected, task='spread', contrast='bright',
        i_sti=s.pack_i_sti(pack), component_bs=[plan], before_t=[on],
        b_specs=[None], time_window=dynamics.TimeWindow('ms', 0, (n_t - 1) * s.delta_ms),
        mode='average', ti_mode='abs_minus_before', merge_bs=True,
        stimulus_pack=pack)
    snapshots = {}
    for c, report in reports.items():
        snapshots[c] = {}
        for ms in [322, 480, 486, 510, 550, 600, 630, 700]:
            step = next(st for st in report['steps'] if st['t'] == round(ms / s.delta_ms))
            snapshots[c][str(ms)] = {key: step[key] for key in
                ['v_post', 'v_in', 'v_slow', 'v_hp', 'v_syn_exc', 'v_syn_inh']}
    (OUT / 'components.json').write_text(json.dumps(snapshots, indent=2))
    for c in selected:
        print('COMPONENT', c, snapshots[c]['486'], flush=True)


if __name__ == '__main__':
    main()
