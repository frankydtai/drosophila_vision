"""Read-only replay and controlled learning experiments for run 34335688.

All outputs go beside this script. The source run and production model/config
are never modified. Run with vision/.venv/bin/python from the repository root.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import import_bootstrap  # noqa: E402,F401
import train  # noqa: E402
from neuron import hp_lp  # noqa: E402
from neuron.readout import pack_traces  # noqa: E402
from train.cost import (  # noqa: E402
    _pack_cost_parts_from_v_readout, forward_pack, gt_affine_from_pack,
)
from train.param import schema_params  # noqa: E402

OUT = Path(__file__).resolve().parent


def session():
    run = next((ROOT / '0_runs/hp_lp').glob('34335688-*'))
    opts = json.loads((run / 'data/train_opts.json').read_text())
    opts['device'] = 'cpu'
    s = replace(train.resolve_session(opts), sequential=False)
    raw = dict(np.load(run / 'data/best_param.npz', allow_pickle=True))
    z = train.z_from_node_vals(raw, s.schema, device='cpu')
    return run, s, z


def slots(s):
    return {
        k: {str(s.connectome.cells[node]): start + i for i, node in enumerate(sp['indi'])}
        for k, sp, start, stop in schema_params(s.schema)
        if k in ('v_th', 'a_out', 'a_h', 'a_gt')
    }


def metrics(z, s):
    with torch.no_grad():
        parts = {k: float(v) for k, v in train.calc_cost_parts(z, s).items()}
        p = train.params_from_z(z, s)
        pack = s.packs['spread']['bright']
        v = train.forward_v(s, p, s.pack_i_sti(pack), pack=pack)[0]
        on = train.pack_t_onset(pack)
        cells = {}
        for name in ('L1', 'L5', 'Mi1', 'Mi4', 'Tm3'):
            mask = s.connectome.node_cells == s.connectome.cells.index(name)
            vv = v[:, mask]
            mean = vv.mean(-1)
            # Always report hard-ReLU output, even during smooth training.
            rr = torch.relu(vv - p['v_th'][mask]) * p['a_out'][mask]
            cells[name] = dict(
                v_th=float(p['v_th'][mask][0]), baseline=float(mean[on]),
                bright_peak=float((mean[on:on + 80] - mean[on]).max()),
                bright_trough=float((mean[on:on + 80] - mean[on]).min()),
                output_baseline=float(rr[on].mean()),
                output_peak=float(rr[on:on + 80].mean(-1).max()),
            )
    return dict(total=sum(parts.values()), parts=parts, cells=cells)


def diagnose(run, s, z):
    idx = slots(s)['v_th']
    records = []
    for contrast in ('bright', 'dark'):
        zz = z.clone().requires_grad_()
        pack = s.packs['spread'][contrast]
        params = train.params_from_z(zz, s)
        trace = forward_pack(s, params, s.pack_i_sti(pack), pack)
        readout = pack_traces(trace, pack)
        for phase, ts in [('all', None), ('on', torch.arange(80)),
                          ('post', torch.arange(80, pack.gts.shape[1]))]:
            phase_pack = replace(pack, cost_ts=ts)
            parts = _pack_cost_parts_from_v_readout(
                phase_pack, s, *gt_affine_from_pack(params, phase_pack, s), readout,
            )
            for key, value in parts.items():
                # Original run has nine equally weighted parts.
                value = value / 9
                grad = torch.autograd.grad(value, zz, retain_graph=True)[0]
                record = dict(contrast=contrast, phase=phase, part=key,
                              loss=float(value),
                              gradients={c: float(grad[i]) for c, i in idx.items()})
                records.append(record)
                print('GRAD', json.dumps(record), flush=True)
        del trace, readout, parts, params, zz
    (OUT / 'gradients.json').write_text(json.dumps(records, indent=2))

    adam = dict(np.load(run / 'data/best_adam.npz', allow_pickle=True))
    updates = {}
    for cell in ('L1', 'L5'):
        ci = s.connectome.cells.index(cell)
        m, v, step = adam['m_v_th'][ci], adam['v_v_th'][ci], int(adam['iter'])
        updates[cell] = dict(m=float(m), v=float(v), step=step,
            previous_update=-.1 * float(m / (1 - .9 ** step)) /
                (float(np.sqrt(v / (1 - .999 ** step))) + 1e-8))
    (OUT / 'adam.json').write_text(json.dumps(updates, indent=2))
    print('ADAM', updates, flush=True)

    sweeps = []
    for l1 in [float(z[idx['L1']]), -50., -52., -54., -56., -60., -65.]:
        for l5 in [float(z[idx['L5']]), -55., -60., -65.]:
            zz = z.clone()
            zz[idx['L1']], zz[idx['L5']] = l1, l5
            m = metrics(zz, s)
            m.update(l1=l1, l5=l5)
            sweeps.append(m)
            print('SWEEP', l1, l5, m['total'],
                  {k: round(v['bright_peak'], 3) for k, v in m['cells'].items()}, flush=True)
    (OUT / 'sweep.json').write_text(json.dumps(sweeps, indent=2))


def audit(s, z):
    idx = slots(s)['v_th']
    zz = z.clone().requires_grad_()
    parts = train.calc_cost_parts(zz, s)
    grad = torch.autograd.grad(sum(parts.values()), zz)[0]
    result = {'finite_difference': {}, 'baseline_scope': {}}
    for cell in ('L1', 'L5'):
        vals = []
        with torch.no_grad():
            for sign in (-1, 1):
                trial = z.clone()
                trial[idx[cell]] += sign * .001
                vals.append(float(sum(train.calc_cost_parts(trial, s).values())))
        result['finite_difference'][cell] = dict(
            autograd=float(grad[idx[cell]]), central_difference=(vals[1] - vals[0]) / .002)
    on_packs = {'spread': {
        c: replace(p, cost_ts=torch.arange(80)) for c, p in s.packs['spread'].items()
    }}
    for scope in ('cell', 'entry'):
        opts = copy.deepcopy(s.train_opts)
        opts['val_from']['bias_gt']['scope'] = scope
        ss = replace(s, packs=on_packs, train_opts=opts)
        trial = z.clone().requires_grad_()
        parts = train.calc_cost_parts(trial, ss)
        record = {}
        for key in ('spread_bright_Mi1', 'spread_bright_Mi4', 'spread_bright_Tm3'):
            gg = torch.autograd.grad(parts[key], trial, retain_graph=True)[0]
            record[key] = {'loss': float(parts[key].detach()),
                           'gradient': {c: float(gg[idx[c]]) for c in ('L1', 'L5')}}
        result['baseline_scope'][scope] = record
    (OUT / 'objective_audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


class SmoothTorch:
    """Change only hp_lp synaptic rectification in this diagnostic process."""
    temperature = 0.0
    mask = None

    def __getattr__(self, name):
        return getattr(torch, name)

    def relu(self, x):
        if self.temperature <= 0:
            return torch.relu(x)
        smooth = F.softplus(x / self.temperature) * self.temperature
        return smooth if self.mask is None else torch.where(self.mask, smooth, torch.relu(x))


def experiment(s, saved_z, name, iterations, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if 'entry' in name:
        opts = copy.deepcopy(s.train_opts)
        opts['val_from']['bias_gt']['scope'] = 'entry'
        s = replace(s, train_opts=opts)
    idx = slots(s)['v_th']
    z0 = saved_z.clone()
    if name.startswith('fresh'):
        z0 = train.z_init_from_schema(s.schema).cpu()
    if 'tonic_init' in name:
        # A common leak-relative initialization, not fitted response thresholds.
        for cell in ('L1', 'L5'):
            z0[idx[cell]] = -57.
    zz = torch.nn.Parameter(z0)
    optimizer = torch.optim.Adam([zz], lr=.1)
    if name == 'resume':
        raw = dict(np.load(next((ROOT / '0_runs/hp_lp').glob('34335688-*')) /
                           'data/best_adam.npz', allow_pickle=True))
        ad = train.z_adams_from_node_vals(
            {k[2:]: v for k, v in raw.items() if k.startswith('m_')},
            {k[2:]: v for k, v in raw.items() if k.startswith('v_')},
            s.schema, dtype=s.sim_dtype, device='cpu')
        optimizer.state[zz].update(step=torch.tensor(float(raw['iter'])),
                                  exp_avg=ad[0], exp_avg_sq=ad[1])
    bounds = train.schema_clamps(s.schema)
    proxy = SmoothTorch()
    if 'gates' in name:
        proxy.mask = torch.zeros(s.connectome.n_node, dtype=torch.bool)
        for cell in ('L1', 'L5'):
            proxy.mask |= s.connectome.node_cells == s.connectome.cells.index(cell)
    old_torch = hp_lp.torch
    hp_lp.torch = proxy
    history = []
    best_hard = float('inf')
    best_z = z0.clone()
    start = time.time()
    try:
        for it in range(iterations + 1):
            if 'smooth' in name:
                # Smooth continuation followed by actual hard-ReLU training.
                initial_temperature = 4. if 'smooth4' in name else 2.
                proxy.temperature = (initial_temperature * (1 - it / (.75 * iterations))
                                     if it < .75 * iterations else 0.)
            if it % 20 == 0 or it == iterations:
                temp = proxy.temperature
                proxy.temperature = 0
                record = metrics(zz, s)
                if record['total'] < best_hard:
                    best_hard, best_z = record['total'], zz.detach().clone()
                proxy.temperature = temp
                record.update(iter=it, temperature=temp, elapsed=time.time() - start)
                history.append(record)
                (OUT / f'{name}_seed{seed}_history.json').write_text(json.dumps(history, indent=2))
                print('TRAIN', name, seed, it, 'hard_loss', round(record['total'], 2),
                      'theta', [round(float(zz[idx[c]]), 3) for c in ('L1', 'L5')],
                      'peaks', [round(record['cells'][c]['bright_peak'], 3)
                                for c in ('Mi1', 'Mi4', 'Tm3')],
                      'elapsed', round(record['elapsed'], 1), flush=True)
            if it == iterations:
                break
            optimizer.zero_grad()
            parts = train.calc_cost_parts(zz, s)
            loss = sum(parts.values())
            loss.backward()
            if not torch.isfinite(loss) or not torch.isfinite(zz.grad).all():
                raise RuntimeError(f'non-finite loss or gradient at {it}')
            optimizer.step()
            with torch.no_grad():
                zz.clamp_(bounds[:, 0], bounds[:, 1])
        np.savez(OUT / f'{name}_seed{seed}_params.npz', best_z=best_z.numpy(),
                 final_z=zz.detach().numpy())
    finally:
        hp_lp.torch = old_torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment')
    parser.add_argument('--audit', action='store_true')
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run, s, z = session()
    if args.audit:
        audit(s, z)
    elif args.experiment:
        experiment(s, z, args.experiment, args.iterations, args.seed)
    else:
        diagnose(run, s, z)
