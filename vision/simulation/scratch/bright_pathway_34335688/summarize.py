"""Evaluate learned models with both objectives and block learned signal paths."""
from __future__ import annotations

import copy
import json
from dataclasses import replace

import numpy as np
import torch

import diagnose as d


torch.set_num_threads(1)
_, session, saved_z = d.session()
opts = copy.deepcopy(session.train_opts)
opts['val_from']['bias_gt']['scope'] = 'entry'
entry_session = replace(session, train_opts=opts)
summary = {}
for name, seed in [('saved', 0), ('fresh', 0), ('resume', 0), ('fresh_tonic_init', 0),
                   ('fresh_tonic_init_entry', 0), ('fresh_tonic_init_entry', 1)]:
    key = f'{name}_seed{seed}'
    z = saved_z if name == 'saved' else torch.tensor(
        np.load(d.OUT / f'{key}_params.npz')['final_z'], dtype=session.sim_dtype)
    summary[key] = d.metrics(z, session)
    with torch.no_grad():
        summary[key]['entry_objective'] = float(sum(d.train.calc_cost_parts(z, entry_session).values()))
    print(key, summary[key]['total'], summary[key]['entry_objective'], flush=True)
(d.OUT / 'final_comparison.json').write_text(json.dumps(summary, indent=2))

# Block only stimulus-dependent presynaptic output changes; retain the original
# tonic output and the exact pre-stimulus dynamics. This distinguishes learning
# the L1/L5 route from obtaining a response through an unrelated recurrent path.
z = torch.tensor(np.load(d.OUT / 'fresh_tonic_init_entry_seed0_params.npz')['final_z'],
                 dtype=session.sim_dtype)
params = d.train.params_from_z(z, entry_session)
pack = entry_session.packs['spread']['bright']
on = d.train.pack_t_onset(pack)
with torch.no_grad():
    ref = d.train.forward_v(entry_session, params, entry_session.pack_i_sti(pack), pack=pack)
baseline_relu = torch.relu(ref[:, on] - params['v_th'])
original_torch = d.hp_lp.torch
original_step = d.hp_lp.step
ablations = {}
for blocked in [(), ('L1',), ('L5',), ('L1', 'L5')]:
    class ClampOutput(d.SmoothTorch):
        active = False

        def relu(self, x):
            r = torch.relu(x)
            return torch.where(self.mask, baseline_relu, r) if self.active else r

    proxy = ClampOutput()
    proxy.mask = torch.zeros(session.connectome.n_node, dtype=torch.bool)
    for cell in blocked:
        proxy.mask |= session.connectome.node_cells == session.connectome.cells.index(cell)
    step_t = 0

    def step(*args, **kwargs):
        global step_t
        step_t += 1
        proxy.active = step_t > on
        return original_step(*args, **kwargs)

    d.hp_lp.torch, d.hp_lp.step = proxy, step
    try:
        with torch.no_grad():
            v = d.train.forward_v(entry_session, params, entry_session.pack_i_sti(pack), pack=pack)[0]
            record = {}
            for cell in ('L5', 'Mi1', 'Mi4', 'Tm3'):
                mask = session.connectome.node_cells == session.connectome.cells.index(cell)
                mean = v[:, mask].mean(-1)
                record[cell] = float((mean[on:on + 80] - mean[on]).max())
            ablations['+'.join(blocked) or 'none'] = record
            print('CLAMP', blocked, record, flush=True)
    finally:
        d.hp_lp.torch, d.hp_lp.step = original_torch, original_step
(d.OUT / 'pathway_ablation.json').write_text(json.dumps(ablations, indent=2))
