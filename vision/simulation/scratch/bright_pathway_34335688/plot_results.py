"""Generate static scientific figures from the saved diagnostic experiments."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

import diagnose as d


plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.dpi': 180})
fig, axs = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
sweep = json.loads((d.OUT / 'sweep.json').read_text())
xs = sorted({r['l1'] for r in sweep})
ys = sorted({r['l5'] for r in sweep})
surface = np.array([[next(r['total'] for r in sweep if r['l1'] == x and r['l5'] == y)
                     for x in xs] for y in ys]) / 1000
ax = axs[0, 0]
im = ax.pcolormesh(xs, ys, surface, shading='nearest', cmap='viridis', vmin=300, vmax=650)
ax.plot(-48.16078, -52.00684, '*', color='#ff674f', ms=14)
ax.annotate('Saved model', (-48.16078, -52.00684), xytext=(-62, -53),
            arrowprops={'arrowstyle': '->', 'color': '#ff674f'}, color='#c44228')
ax.set(xlabel='L1 threshold (mV)', ylabel='L5 threshold (mV)',
       title='A  Joint threshold loss surface; all other parameters fixed')
fig.colorbar(im, ax=ax, label='Original loss / 1,000', extend='max')

ax = axs[0, 1]
for name, label, color, style in [
    ('fresh', 'Original initialization', '#777777', '-'),
    ('resume', 'Saved model: 200 extra steps', '#777777', '--'),
    ('fresh_tonic_init', 'Only L1/L5 initialization changed', '#007e87', '-'),
]:
    h = json.loads((d.OUT / f'{name}_seed0_history.json').read_text())
    ax.plot([r['iter'] for r in h], [r['total'] / 1000 for r in h],
            label=label, color=color, ls=style, lw=2)
ax.set(xlabel='Optimization steps in comparison', ylabel='Original loss / 1,000',
       title='B  Same loss, hard ReLU, freely optimized thresholds')
ax.legend(fontsize=9)

torch.set_num_threads(1)
_, s, saved = d.session()
pack = s.packs['spread']['bright']
on = d.train.pack_t_onset(pack)
colors = {'Mi1': '#007e87', 'Mi4': '#db8c18', 'Tm3': '#9c4d9c'}
ax = axs[1, 0]
with torch.no_grad():
    for seed in (0, 1):
        z = torch.tensor(np.load(d.OUT / f'fresh_tonic_init_entry_seed{seed}_params.npz')['final_z'],
                         dtype=s.sim_dtype)
        params = d.train.params_from_z(z, s)
        v = d.train.forward_v(s, params, s.pack_i_sti(pack), pack=pack)[0]
        t = np.arange(v.shape[0] - on) * 2
        for cell, color in colors.items():
            mean = v[:, s.connectome.node_cells == s.connectome.cells.index(cell)].mean(-1)
            ax.plot(t, (mean[on:] - mean[on]).numpy(), color=color,
                    ls='-' if seed == 0 else '--', label=cell if seed == 0 else None)
ax.axvspan(0, 160, color='#f3e5b8', alpha=.5)
ax.axhline(0, lw=.7, color='#888888')
ax.text(.04, .95, 'Bright ON', transform=ax.transAxes, va='top')
ax.set(xlabel='Time after onset (ms)', ylabel='Population mean response (mV)',
       title='C  Corrected baseline + tonic initialization; two seeds')
ax.legend(fontsize=9)

ax = axs[1, 1]
ablation = json.loads((d.OUT / 'pathway_ablation.json').read_text())
labels = ['Learned model', 'Hold L1 output\nat its baseline', 'Hold L5 output\nat its baseline']
xx = np.arange(3)
for i, (cell, color) in enumerate(colors.items()):
    ax.bar(xx + (i - 1) * .24, [ablation[k][cell] for k in ('none', 'L1', 'L5')],
           width=.23, label=cell, color=color)
ax.set_xticks(xx, labels)
ax.set(ylabel='Bright-on peak response (mV)',
       title='D  Causal test: preserve tonic output, block its change')
ax.legend(fontsize=9)
fig.savefig(d.OUT / 'diagnostic_overview.png')
fig.savefig(d.OUT / 'diagnostic_overview.pdf')
print(d.OUT / 'diagnostic_overview.png')
