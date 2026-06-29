"""Experiment: add R1-8 photoreceptors as fit cells that mirror L1.

ONE source of truth here (PHOTORECEPTORS + MIRROR) drives everything; the core
(fc) and plotting (plot_trained) only expose generic knobs -- nothing about R1-8
is written into them.

  * target data : R1-8 cost cells get L1's per-column target -> fc.data / fc.mc_cell_index
  * baseline    : (conductance) R1-8 share L1's depolarised E_leak (-20 mV)
  * Ih          : (conductance) R1-8 get a reversed Ih (E_Ih +50 -> -50 mV)
  * lamina share: R1-6 share ONE value; R7, R8, L1-L5 independent, applied to
                  the model's lamina params (conductance: Ih_gmax;
                  adaptive: adapt_gain + tau_adapt)
  * plots       : grey L1 reference + R1-8 panels -> pt.REF_CUBES / pt.MVD_GROUPS

Model is chosen with --model_type (default conductance), consistent with run.py.
Training + all output (npy, costs, table, cost_curve, model_vs_data,
model_all_cells, best_param, model_type) go through run.run_training, landing in
its own FiveCol_Parameter/<model_type>/run_<id>/ folder.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)                       # fc loads Circuits/ by relative path at import
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import torch

import FiveCol_MedSim_Pytorch as fc
import plot_trained as pt
import run

# ---- CLI (same flag names as run.py for consistency) ----
ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--model_type', default='conductance', choices=['conductance', 'adaptive'])
ap.add_argument('--nofruns', type=int, default=1)
ap.add_argument('--nofsteps', type=int, default=100)
ap.add_argument('--lrs', type=float, nargs='+', default=[0.1])
args = ap.parse_args()
MODEL = args.model_type

# ---- single source of truth: which cells mirror which fit cell ----
CTYPE = np.load(os.path.join(ROOT, 'Circuits', 'ctype.npy'), allow_pickle=True)
PHOTORECEPTORS = list(range(8))      # ctype indices 0..7  == R1..R8
R_NAMES = [str(CTYPE[i]) for i in PHOTORECEPTORS]
MIRROR_FIT_CELL = 'L1'               # R1-8 reuse L1's data, baseline and (some) params
MIRROR_SIGN = -1.0                   # R1-8 target = sign-flipped L1 (upward), matching
                                     # the photoreceptors' natural depolarising response
SHARED_R = [0, 1, 2, 3, 4, 5]        # R1-R6 share one lamina value
INDEP_R = [6, 7]                     # R7, R8 independent

fc.MODEL_TYPE = MODEL
N_FIT = fc.data.shape[1] // fc.nofcols       # fit cells per column (13)
L1_COL = 0                                   # L1 is cell_list index 0

# ---- 1) extend cost cells: R1-8 in every column get L1's column target ----
#         (model-agnostic: just adds rows to fc.data / fc.mc_cell_index)
base_data = fc.data                          # (200, N_FIT*nofcols)
extra_cols, extra_idx = [], []
for col in range(fc.nofcols):
    l1_target = MIRROR_SIGN * base_data[:, col * N_FIT + L1_COL: col * N_FIT + L1_COL + 1]  # (200,1)
    for r in PHOTORECEPTORS:
        extra_idx.append(r + col * fc.nofcells)        # 325-state index of R_r in this column
        extra_cols.append(l1_target)
fc.data = torch.cat([base_data] + extra_cols, dim=1)
fc.mc_cell_index = np.concatenate([np.asarray(fc.mc_cell_index), np.array(extra_idx, dtype=int)])
fc.power = torch.sum(fc.data[50:200] ** 2)
print('cost cells: %d -> %d' % (base_data.shape[1], fc.data.shape[1]))

# ---- 2) lamina parameter sharing: R1-6 tied, R7,R8,L1-5 independent ----
lamina_default = list(range(fc.LAMINA_SLICE.start, fc.LAMINA_SLICE.stop))   # [8,9,10,11,12]
groups = [SHARED_R] + INDEP_R + lamina_default      # [[0..5],6,7,8,9,10,11,12]


def group_lamina(schema, names, grp):
    """Return a copy of schema with seg['cells']=grp for each lamina seg in names."""
    out = [dict(s) for s in schema]
    for s in out:
        if s['name'] in names:
            s['cells'] = grp
    return out


# ---- 3) model-specific config (only what the chosen model actually has) ----
if MODEL == 'conductance':
    # R1-8 share L1's depolarised resting potential
    fc.E_LEAK_DEPOL_CELLS = sorted(set(fc.E_LEAK_DEPOL_CELLS) | set(PHOTORECEPTORS))
    fc.E_leak = fc.build_e_leak()
    print('E_leak depol cells:', fc.E_LEAK_DEPOL_CELLS)

    # R1-8 get a reversed Ih (depolarisation-activated, E_Ih +50 -> -50 mV) so
    # their upward response decays the way L's hyperpolarising response does
    fc.IH_DIR_REVERSE_CELLS = sorted(set(fc.IH_DIR_REVERSE_CELLS) | set(PHOTORECEPTORS))
    fc.Ih_dir = fc.build_ih_dir()
    print('Ih reversed cells:', fc.IH_DIR_REVERSE_CELLS)

    # share the lamina Ih_gmax to R1-8; keep L3,L4 initialised to 0
    schema = group_lamina(fc.CONDUCTANCE_SCHEMA, ['Ih_gmax'], groups)
    for s in schema:
        if s['name'] == 'Ih_gmax':
            s['zero'] = [groups.index(10), groups.index(11)]   # L3, L4 value-indices
    fc.CONDUCTANCE_SCHEMA = schema
    lamina_names = ['Ih_gmax']
else:  # adaptive: no E_leak, no Ih; share BOTH lamina adaptation params with R.
    lamina_names = ['adapt_gain', 'tau_adapt']
    fc.ADAPTIVE_SCHEMA = group_lamina(fc.ADAPTIVE_SCHEMA, lamina_names, groups)

active = fc.active_schema()
for name in lamina_names:
    seg = next(s for s in active if s['name'] == name)
    print('%s groups: %s  trainable values: %d' % (name, groups, fc.seg_count(seg)))

# ---- 4) plotting hooks: grey L1 reference for R1-8 + extra R panels ----
ref = pt.default_ref_cubes()
for name in R_NAMES:
    ref[name] = MIRROR_SIGN * ref[MIRROR_FIT_CELL]
pt.REF_CUBES = ref
# R on top, then the default L / Mi / Tm row-pairs (which shift down by one pair).
pt.MVD_GROUPS = [np.array(R_NAMES)] + pt.DEFAULT_MVD_GROUPS

# ---- 5) train through the standard pipeline (own run_ folder, full outputs) ----
fname, outdir = run.run_training(MODEL, args.nofruns, args.nofsteps, args.lrs)
print('done ->', outdir)
