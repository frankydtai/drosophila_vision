# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 09:53:25 2023

@author: aborst
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import Medulla_Library as ml
import time

import torch
from torch import nn
from tqdm import tqdm

from network.connectivity import DenseConn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#################################################################
# Medulla Library contains:
# ml.read_ConnMs()
# ml.read_RecF_data(): RecF_data (13,45), ImpR_data (13,45)
# plot_ConnM(): Big ConnM + intra + inter M
# stimulus generation -> signal
#################################################################

plt.rcParams['axes.facecolor'] = '#EEEEEE'
plt.rcParams['figure.facecolor'] = 'white'

nofcells  = 65
nofcols   = 5
maxtime   = ml.IMPULSE_MAXTIME
t_on      = ml.T_ON

# important model params

deltat    = 10.0  # in msec
g_leak    = 1.0   # in nS
E_exc     = +10.0 # in mV
E_inh     = -70.0 # in mV
capac     = +40.0 # in pF, results in 50ms membrane time-constant for g_leak = 1.0 nS
trld      = -50.0 # in mV: below trld, no signal is transmitted
cdt       = capac/deltat

Ca_tau    = 50.0  # in msec

# Resting potential. Most cells rest at E_LEAK_REST; a configurable set of
# cell-types (default lamina L1-L3) is depolarised to E_LEAK_DEPOL. The depol
# cell list is a module global so experiments can extend it (e.g. give R1-8 the
# same depolarised baseline) WITHOUT editing the core: just append indices and
# call build_e_leak() again. No cell type is hardcoded into the construction.
E_LEAK_REST = -50.0
E_LEAK_DEPOL = -20.0
E_LEAK_DEPOL_CELLS = list(ml.LAMINA_DEPOL_TYPES)  # L1-L3 cell-type indices within the 65 types

def build_e_leak():
    """(nofcells*nofcols,) resting potential, replicating the depol set per column."""
    el = torch.zeros(nofcells * nofcols, dtype=torch.float64).to(device) + E_LEAK_REST
    for col in range(nofcols):
        for c in E_LEAK_DEPOL_CELLS:
            el[nofcells * col + c] = E_LEAK_DEPOL
    return el

E_leak = build_e_leak()

exc_synweight = 0.001
inh_synweight = 0.001

# ----------- H-Current ----------------------------------------

E_Ih          = +50.0  # in mV
Ih_midv       = -50.0
Ih_slope      = -0.25
tau_midv      = -50.0
Ih_gmax       = +50.0 

Ih_gain       = 1.0   # if set to 0, it will block Ih

# Per-cell Ih direction: +1 = standard hyperpolarisation-activated Ih; -1 = Ih
# REVERSED (activation flips to depolarisation-activated AND the reversal flips
# sign about 0, E_Ih=+50 -> -50 mV), giving a gentle downward sag on an upward
# response. Configurable like E_leak: a module global the experiment extends
# (e.g. for photoreceptors); the core hardcodes no cell type. Built after
# nofcells/nofcols/device exist (see build_ih_dir below).
IH_DIR_REVERSE_CELLS = []   # cell-type indices whose Ih is mirrored

def build_ih_dir():
    """(nofcells*nofcols,) Ih direction (+1 normal, -1 mirrored), replicated per
    column. Driven by IH_DIR_REVERSE_CELLS (cell-type indices); default all +1."""
    d = torch.ones(nofcells * nofcols, dtype=torch.float64).to(device)
    for col in range(nofcols):
        for c in IH_DIR_REVERSE_CELLS:
            d[nofcells * col + c] = -1.0
    return d

Ih_dir = build_ih_dir()

# parameter and cost function definition

nofparams = 2 * nofcells + 8 # 8 params for Ih (5 x gmax + 3)

low_gain = 0.1
high_gain = 100.0

# ---- second neuron model: adaptive temporal filter (flyvis-derived) ----
# 'conductance' = Borst conductance-based + Ih (update_Vm)
# 'adaptive'    = passive point neuron + low-pass adaptive temporal filter
MODEL_TYPE = 'conductance'

gate_lag = 1  # delay (in steps) of the stimulus used for the contrast gate
GATE_PIVOT = 0.5  # fixed contrast-gate pivot (non-trainable); input is normalised to [0,1]
STATE_CLAMP = 1.0e6  # bound on adaptive state vars to keep explicit Euler finite

# --- parameter schema: SINGLE SOURCE OF TRUTH -------------------------------
# nofparams, assign, bounds and guess are all derived from these lists.
# To change the parameterisation (sizes, ranges, which cells), edit ONLY here.
# Each segment is a dict:
#   name  : parameter name (becomes a key in the assigned dict)
#   count : number of per-unit values for this segment (the 'individual' width)
#   kind  : how `count` raw values become a usable parameter:
#           'full'   -> count==nofcells; one value per cell type, replicated to 5 cols (325,)
#           'lamina' -> one value per entry in seg['cells'] (default L1-L5, indices
#                       8:13); an entry may be a list of indices that SHARE one value;
#                       other cells = 'fill'. (325,). 'count' is ignored for this kind.
#           'scalar' -> count==1; a single global 0-dim value (e.g. Ih_midv)
#           'output' -> count==nofcells; per-cell-type value applied to the (150,65) OUTPUT
#                       (not the 325-cell state, e.g. out_scale). returned as (65,).
#   lo,hi : training bounds (clamped each Adam step)
#   init  : random-init mean;  jit: init uniform jitter (+/- jit/2)
#   fill  : value for non-listed cells ('lamina' only)
#   zero  : local indices set to 0 at init in 'individual' mode (e.g. Ih L3,L4)
#   mode  : 'individual' (train all `count` values, default), 'shared' (train ONE value
#           broadcast to all units), or 'fixed' (train NOTHING; held at 'fixed' or 'init').
#   fixed : constant value used when mode=='fixed' (defaults to 'init').
# `mode`/`fixed` are normally NOT set here; they are overridden per run from the
# CLI / SLURM (see run.py --mode / --fix), so the schema stays the canonical default.
LAMINA_SLICE = ml.LAMINA_SLICE  # L1-L5 within the 65 cell types
PARAM_MODES = ('individual', 'shared', 'fixed')

# Lamina/scalar segments expanded to per-cell-type (full) by --per_type.
PER_TYPE_PARAM_NAMES = frozenset({
    'Ih_gmax', 'Ih_midv', 'Ih_slope', 'tau_midv',
    'adapt_gain', 'tau_adapt',
})

ADAPTIVE_SCHEMA = [
    {'name': 'inp_gain',   'count': nofcells, 'kind': 'full',   'lo': low_gain, 'hi': high_gain, 'init': 0.5,   'jit': 0.2,  'fill': 0.0},
    {'name': 'out_gain',   'count': nofcells, 'kind': 'full',   'lo': low_gain, 'hi': high_gain, 'init': 0.5,   'jit': 0.2,  'fill': 0.0},
    {'name': 'tau_m',      'count': nofcells, 'kind': 'full',   'lo': deltat,   'hi': 1000.0,    'init': 50.0,  'jit': 10.0, 'fill': 0.0},
    {'name': 'bias',       'count': nofcells, 'kind': 'full',   'lo': -2.0,     'hi': 2.0,       'init': 0.0,   'jit': 0.1,  'fill': 0.0},
    {'name': 'adapt_gain', 'count': 5,        'kind': 'lamina', 'lo': -2.0,     'hi': 2.0,       'init': 0.0,   'jit': 0.1,  'fill': 0.0},
    {'name': 'tau_adapt',  'count': 5,        'kind': 'lamina', 'lo': deltat,   'hi': 2000.0,    'init': 100.0, 'jit': 20.0, 'fill': deltat},
    {'name': 'out_scale',  'count': nofcells, 'kind': 'output', 'lo': 0.0,      'hi': 1.0e4,     'init': 1.0,   'jit': 0.0},
]

# conductance (Borst + Ih) model parameters, same single-source-of-truth contract.
# Layout matches the historical z vector + a trailing per-cell out_scale (65, init 1).
CONDUCTANCE_SCHEMA = [
    {'name': 'inp_gain',  'count': nofcells, 'kind': 'full',   'lo': low_gain, 'hi': high_gain, 'init': 0.5,      'jit': 0.2,  'fill': 0.0},
    {'name': 'out_gain',  'count': nofcells, 'kind': 'full',   'lo': low_gain, 'hi': high_gain, 'init': 0.5,      'jit': 0.2,  'fill': 0.0},
    {'name': 'Ih_gmax',   'count': 5,        'kind': 'lamina', 'lo': 0.0,      'hi': 100.0,     'init': Ih_gmax,  'jit': 10.0, 'fill': 0.0, 'zero': [2, 3]},
    {'name': 'Ih_midv',   'count': 1,        'kind': 'scalar', 'lo': -70.0,    'hi': -30.0,     'init': Ih_midv,  'jit': 5.0},
    {'name': 'Ih_slope',  'count': 1,        'kind': 'scalar', 'lo': -0.40,    'hi': -0.20,     'init': Ih_slope, 'jit': 0.02},
    {'name': 'tau_midv',  'count': 1,        'kind': 'scalar', 'lo': -70.0,    'hi': -40.0,     'init': tau_midv, 'jit': 5.0},
    {'name': 'out_scale', 'count': nofcells, 'kind': 'output', 'lo': 0.0,      'hi': 1.0e4,     'init': 1.0,      'jit': 0.0},
]


def seg_mode(seg):
    mode = seg.get('mode', 'individual')
    if mode not in PARAM_MODES:
        raise ValueError(f"{seg['name']}: bad mode {mode!r}, expected one of {PARAM_MODES}")
    return mode


def lamina_cells(seg):
    """Target cell-type indices for a 'lamina' segment, one entry per trainable value.

    Each entry is either an int (one cell) or a list of ints (a GROUP of cells
    that SHARE that single trainable value). Defaults to L1-L5 (LAMINA_SLICE) as
    five independent values, reproducing the historical behaviour. Experiments
    override seg['cells'] to remap/group/extend the lamina parameter (e.g. tie
    R1-6 to one value while keeping R7, R8, L1-L5 independent) WITHOUT editing
    the core: the placement and the value count both derive from this list."""
    return seg.get('cells', list(range(LAMINA_SLICE.start, LAMINA_SLICE.stop)))


def seg_count(seg):
    """Number of per-unit values for a segment (its 'individual' width).
    For 'lamina' this is the number of (possibly grouped) target entries."""
    if seg['kind'] == 'lamina':
        return len(lamina_cells(seg))
    return seg['count']


def seg_ntrain(seg):
    """Number of trainable values stored in z for this segment, given its mode."""
    mode = seg_mode(seg)
    if mode == 'fixed':
        return 0
    if mode == 'shared':
        return 1
    return seg_count(seg)                      # individual


def schema_segments(schema):
    """Yield (segment, start, stop) slice ranges into z (widths depend on mode)."""
    start = 0
    for seg in schema:
        stop = start + seg_ntrain(seg)
        yield seg, start, stop
        start = stop


def schema_nparams(schema):
    return sum(seg_ntrain(seg) for seg in schema)


def apply_modes(schema, modes=None, fixes=None):
    """Return a COPY of schema with per-parameter mode / fixed-value overrides.

    modes: {name: 'individual'|'shared'|'fixed'};  fixes: {name: value} (implies fixed).
    Keeps the original schema (the canonical default) untouched.
    """
    modes, fixes = modes or {}, fixes or {}
    out = []
    for seg in schema:
        s = dict(seg)
        if s['name'] in modes:
            s['mode'] = modes[s['name']]
        if s['name'] in fixes:
            s['mode'] = 'fixed'
            s['fixed'] = float(fixes[s['name']])
        out.append(s)
    return out


def expand_schema_per_type(schema, names=None):
    """Rebind lamina/scalar segments to per-cell-type (full) trainable params."""
    names = PER_TYPE_PARAM_NAMES if names is None else frozenset(names)
    out = []
    for seg in schema:
        s = dict(seg)
        if s['name'] not in names or s['kind'] not in ('lamina', 'scalar'):
            out.append(s)
            continue
        old_kind = s['kind']
        s['kind'] = 'full'
        s['count'] = nofcells
        if old_kind == 'lamina':
            cells = lamina_cells(s)
            if 'zero' in s:
                s['zero'] = [
                    (cells[j] if isinstance(cells[j], int) else cells[j][0])
                    for j in s['zero']
                ]
            s.pop('cells', None)
        out.append(s)
    return out


def active_schema():
    return ADAPTIVE_SCHEMA if MODEL_TYPE == 'adaptive' else CONDUCTANCE_SCHEMA


def adaptive_segments():
    return schema_segments(ADAPTIVE_SCHEMA)


nofparams_adaptive = schema_nparams(ADAPTIVE_SCHEMA)


# --- schema builders for an arbitrary type vocabulary (network path) ------
# The literal CONDUCTANCE_SCHEMA / ADAPTIVE_SCHEMA above are the 65-type Borst
# defaults. use_network() rebuilds equivalents for the network's own type
# count + lamina indices so the SAME training machinery works on 32 (or N) types.

def build_conductance_schema(n_types, lamina, ih_zero=(2, 3)):
    zero = [j for j in ih_zero if j < len(lamina)]
    return [
        {'name': 'inp_gain',  'count': n_types, 'kind': 'full',   'lo': low_gain, 'hi': high_gain, 'init': 0.5,     'jit': 0.2,  'fill': 0.0},
        {'name': 'out_gain',  'count': n_types, 'kind': 'full',   'lo': low_gain, 'hi': high_gain, 'init': 0.5,     'jit': 0.2,  'fill': 0.0},
        {'name': 'Ih_gmax',   'count': len(lamina), 'kind': 'lamina', 'cells': lamina, 'lo': 0.0, 'hi': 100.0, 'init': Ih_gmax, 'jit': 10.0, 'fill': 0.0, 'zero': zero},
        {'name': 'Ih_midv',   'count': 1,       'kind': 'scalar', 'lo': -70.0,    'hi': -30.0,     'init': Ih_midv,  'jit': 5.0},
        {'name': 'Ih_slope',  'count': 1,       'kind': 'scalar', 'lo': -0.40,    'hi': -0.20,     'init': Ih_slope, 'jit': 0.02},
        {'name': 'tau_midv',  'count': 1,       'kind': 'scalar', 'lo': -70.0,    'hi': -40.0,     'init': tau_midv, 'jit': 5.0},
        {'name': 'out_scale', 'count': n_types, 'kind': 'output', 'lo': 0.0,      'hi': 1.0e4,     'init': 1.0,      'jit': 0.0},
    ]


def build_adaptive_schema(n_types, lamina):
    return [
        {'name': 'inp_gain',   'count': n_types, 'kind': 'full',   'lo': low_gain, 'hi': high_gain, 'init': 0.5,   'jit': 0.2,  'fill': 0.0},
        {'name': 'out_gain',   'count': n_types, 'kind': 'full',   'lo': low_gain, 'hi': high_gain, 'init': 0.5,   'jit': 0.2,  'fill': 0.0},
        {'name': 'tau_m',      'count': n_types, 'kind': 'full',   'lo': deltat,   'hi': 1000.0,    'init': 50.0,  'jit': 10.0, 'fill': 0.0},
        {'name': 'bias',       'count': n_types, 'kind': 'full',   'lo': -2.0,     'hi': 2.0,       'init': 0.0,   'jit': 0.1,  'fill': 0.0},
        {'name': 'adapt_gain', 'count': len(lamina), 'kind': 'lamina', 'cells': lamina, 'lo': -2.0, 'hi': 2.0,    'init': 0.0,   'jit': 0.1,  'fill': 0.0},
        {'name': 'tau_adapt',  'count': len(lamina), 'kind': 'lamina', 'cells': lamina, 'lo': deltat, 'hi': 2000.0, 'init': 100.0, 'jit': 20.0, 'fill': deltat},
        {'name': 'out_scale',  'count': n_types, 'kind': 'output', 'lo': 0.0,      'hi': 1.0e4,     'init': 1.0,   'jit': 0.0},
    ]


# -------------------------------------------------------------------------
# -------------- reading cell data and connectivity matrices --------------
# -------------------------------------------------------------------------

def init_network():
    
    multi_colM    = np.load('Circuits/multi_colM.npy')
    ctype         = np.load('Circuits/ctype.npy')
    mc_cell_index = np.load('Circuits/mc_cell_index.npy')
    
    multi_colM = ml.apply_borst_connectivity_patches(multi_colM)
    
    # chemical synpases

    M_exc = exc_synweight * multi_colM * (multi_colM > 0)
    M_inh = inh_synweight * multi_colM * (multi_colM < 0) * (-1)
    
    M_exc = torch.tensor(M_exc,dtype=torch.float64).to(device)
    M_inh = torch.tensor(M_inh,dtype=torch.float64).to(device)
    
    # signed connectivity for the adaptive (current-based) neuron model
    M_signed = exc_synweight * multi_colM
    M_signed = torch.tensor(M_signed,dtype=torch.float64).to(device)
    
    n_units = ml.n_state_units()
    pr = ml.photoreceptor_slice()
    signal = torch.zeros((maxtime, n_units), dtype=torch.float64).to(device)
    signal[0:t_on, pr] = ml.SIGNAL_BASELINE
    signal[t_on:maxtime, pr] = ml.SIGNAL_BRIGHT

    mydata = ml.read_RecF_data() * ml.DATA_AMP
    mydata = torch.tensor(mydata, dtype=torch.float64)
    data   = torch.zeros((nofcells, maxtime), dtype=torch.float64).to(device)
    
    for i in range(nofcols):
        
        data[ml.fit_data_slice(i)] = mydata[:, 2 + i]
        
    data = torch.transpose(data,0,1)
        
    power = torch.sum((data[t_on:maxtime])**2)
    
    return M_exc, M_inh, M_signed, ctype, mc_cell_index, data, power, signal

M_exc, M_inh, M_signed, ctype, mc_cell_index, data, power, signal = init_network()

# ---- connectivity backend (CONN) -------------------------------------------
# All synaptic drive goes through CONN so the simulator core is agnostic to the
# backend. The default is DenseConn wrapping the 5-column matrices, which is
# bit-identical to the historical torch.mv(M_*, x) path. use_network() swaps
# in a ScatterConn read from a network.json without touching the math below.
#
# node_type[i] is unit i's cell-TYPE index. For the Borst path it is i % nofcells
# (so a (65,) per-type param broadcasts to the (325,) state exactly like the old
# 5x tiling); for a network it maps each node to its type vocabulary index.
NODE_TYPE = (torch.arange(nofcells * nofcols, device=device) % nofcells).long()
CONN = DenseConn(M_exc, M_inh, M_signed, NODE_TYPE)

# ---- multi-column / network training state ------------------------------
# These stay None for the default Borst path (zero behaviour change). They are
# populated by use_network() for network / multi-column training.
NETWORK = None       # the loaded Network (None => Borst dense path)
READOUT = None          # (cost_batch_idx, cost_unit_idx): cost cell selection
COST_WEIGHT = None      # (n_cost,) per-cost-cell weight (1/ring-size -> equal rings)
COST_T0 = None          # (n_cost,) absolute step for windowed cost (moving-bar)
MC_COST_RADIUS = None   # (n_cost,) ring radius {0,1,sqrt3,2} of each cost entry
TARGET_KIND = None      # "tile" | "moving_bar" | None
MOVING_BAR_CENTER_COLUMN = False
NETWORK_TRAIN_OPTS = None  # last use_network() kwargs for run sidecar / plot restore
PARAM_TRAIN_OPTS = None    # e.g. {'per_type': True} for run sidecar / plot restore
MULTI_COLUMN_SEQ = None  # None=auto (CPU sequential, CUDA batched)
TARGET_PACKS = None  # None or dict name->TargetPack (multi-target training)
TARGET_PACK_WEIGHTS = None  # None or dict name->float (loss weights)


@dataclass(frozen=True)
class TargetPack:
    """One training target: stimulus + readout indices + target traces."""

    name: str
    signal: torch.Tensor  # (B, T, N)
    data: torch.Tensor  # (n_cost, T')
    power: torch.Tensor  # scalar
    cost_weight: torch.Tensor  # (n_cost,)
    readout_batch: torch.Tensor  # (n_cost,)
    readout_unit: torch.Tensor  # (n_cost,)
    cost_t0: Optional[torch.Tensor] = None  # (n_cost,) absolute step for windowed targets

# ------- network calculations  -----------------------------------------------

def calc_multi_col_params(param):

    # Broadcast a per-cell-TYPE parameter (n_types,) to the full state (n_units,)
    # via the backend's node_type. For the Borst path node_type == arange%65, so
    # this reproduces the old 5x concatenation exactly.

    return param.index_select(0, CONN.node_type)

def rectsyn(x,thrld):
    
    result=x-thrld
    result=result*(result>0)
    
    return result

def update_Vm(Vm,u,inp_gain,out_gain,Ih_gmax,Ih_midv,Ih_slope,tau_midv,signal):

    # Per-cell Ih direction reverses the current where Ih_dir==-1: the activation
    # slope flips (depolarisation-activated) AND the reversal flips sign about 0
    # (E_Ih=+50 -> -50 mV), a gentle pull-down toward rest. Ih_dir==+1 -> unchanged.
    slope_eff = Ih_dir * Ih_slope
    E_Ih_eff  = Ih_dir * E_Ih
    Ih_ss   = 1.0/(1.0+torch.exp((Ih_midv-Vm)*slope_eff))
    tau     = 1.5/(torch.exp(-0.1*(Vm-tau_midv))+torch.exp(+0.1*(Vm-tau_midv)))*1000.0 + 100.0
    u       = deltat/tau*(Ih_ss-u)+u
    g_Ih    = u * Ih_gmax * Ih_gain
    
    g_exc, g_inh = CONN.exc_inh_drive(rectsyn(Vm,trld)*out_gain)
    g_exc   = g_exc*inp_gain
    g_inh   = g_inh*inp_gain
    
    Vm = (g_exc*E_exc + g_inh*E_inh + g_leak*E_leak + E_Ih_eff * g_Ih + cdt*Vm + signal)
    Vm = Vm / (g_exc + g_inh + g_Ih + g_leak + cdt)
    
    return Vm, u

# ---------- adaptive temporal-filter neuron model (flyvis-derived) -----------

def _reconstruct_raw(seg, z_slice, z):
    """Build the length-`count` per-unit vector from the trainable z slice + mode.
    individual: the slice itself; shared: the one value broadcast; fixed: a constant.
    Gradients flow into the (1 or count) trainable entries; fixed has none."""
    mode, count = seg_mode(seg), seg_count(seg)
    if mode == 'fixed':
        const = float(seg.get('fixed', seg['init']))
        return torch.full((count,), const, dtype=z.dtype, device=z.device)
    if mode == 'shared':
        return z_slice[0].repeat(count)
    return z_slice                                              # individual


def _expand_segment(seg, raw):
    """Map a length-`count` per-unit vector to a usable parameter, per its 'kind'."""
    kind = seg['kind']
    if kind == 'full':
        return calc_multi_col_params(raw).to(device)            # (325,) state param
    if kind == 'lamina':
        cell = torch.full((nofcells,), float(seg['fill']), dtype=raw.dtype, device=raw.device)
        for i, target in enumerate(lamina_cells(seg)):          # int or list-of-ints (shared group)
            cell[target] = raw[i]
        return calc_multi_col_params(cell).to(device)           # (325,) state param
    if kind == 'scalar':
        return raw[0]                                           # 0-dim global value
    if kind == 'output':
        return raw.to(device)                                  # (nofcells,) output gain
    raise ValueError(f"unknown segment kind: {kind}")


def assign_params(z, schema):
    """Unpack z into a dict of parameter tensors, driven by the given schema + modes."""
    p = {}
    for seg, start, stop in schema_segments(schema):
        p[seg['name']] = _expand_segment(seg, _reconstruct_raw(seg, z[start:stop], z))
    return p


def assign_params_adaptive(z):
    """Adaptive params plus the fixed (non-trainable) contrast-gate pivot."""
    p = assign_params(z, ADAPTIVE_SCHEMA)
    p['gate_pivot'] = GATE_PIVOT
    return p

def update_state_adaptive(activity, v_sustained, v_transient, drive_lp, p, x_t, x_t_delayed):
    
    # passive point neuron: tau_m * da/dt = -a + X, with X = bias + syn + x_t.
    # an adaptive low-pass reference (drive_lp) gates a transient component so
    # that activity = v_sustained + v_transient.
    
    bias  = p['bias']
    tau   = torch.clamp(p['tau_m'], min=deltat)
    tau_r = torch.clamp(p['tau_adapt'], min=deltat)
    ratio = tau / tau_r
    
    # presynaptic output gain (per source), postsynaptic input gain (per target)
    syn     = p['inp_gain'] * CONN.signed_drive(torch.relu(activity) * p['out_gain'])
    X       = bias + syn + x_t
    X_gate  = bias + syn + x_t_delayed
    gate    = (X_gate - p['gate_pivot']) * p['adapt_gain']
    gate_src = torch.where(gate >= 0, drive_lp, 1.0 - drive_lp)
    
    drive_lp    = drive_lp    + deltat / tau_r * (-drive_lp + X)
    v_sustained = v_sustained + deltat / tau   * (-v_sustained + (1.0 - gate * ratio) * X)
    v_transient = v_transient + deltat / tau   * (-v_transient + (-gate * (1.0 - ratio) * gate_src))
    
    # explicit Euler on this recurrent ReLU net can diverge for large gains;
    # clamp persistent states so blow-ups stay finite (large cost) instead of NaN.
    drive_lp    = torch.clamp(drive_lp,    -STATE_CLAMP, STATE_CLAMP)
    v_sustained = torch.clamp(v_sustained, -STATE_CLAMP, STATE_CLAMP)
    v_transient = torch.clamp(v_transient, -STATE_CLAMP, STATE_CLAMP)
    activity    = v_sustained + v_transient
    
    return activity, v_sustained, v_transient, drive_lp

def model_cost(model, data, out_scale=1.0):
    # normalised MSE over the response window (t=t_on..maxtime-1); out_scale is an optional
    # global linear output gain (default 1.0 -> unscaled).
    return torch.sum((out_scale * model - data[t_on:maxtime])**2) / power * 100.0

def _run_conductance(p, neuron_index=None, return_ref=False, sig=None):
    # forward pass -> low-pass filtered response for the chosen neurons, shape
    # (maxtime-t_on, n). neuron_index defaults to the center-column cost cells (mc_cell_index);
    # plotting passes other columns. return_ref also yields the resting baseline.
    # sig overrides the module-level stimulus (single-column (T,N) only here).
    inp_gain, out_gain, Ih_gmax = p['inp_gain'], p['out_gain'], p['Ih_gmax']
    Ih_midv, Ih_slope, tau_midv = p['Ih_midv'], p['Ih_slope'], p['tau_midv']
    if neuron_index is None:
        neuron_index = mc_cell_index
    if sig is None:
        sig = signal

    u  = torch.zeros(CONN.n_units, dtype=torch.float64).to(device)
    Vm = E_leak
    for t in range(1, t_on):
        Vm, u = update_Vm(Vm,u,inp_gain,out_gain,Ih_gmax,Ih_midv,Ih_slope,tau_midv,sig[t-1])
    Vm_ref = 1.0*Vm[neuron_index]  # reference 0
    model = 0; rows = []
    for t in range(t_on, maxtime):
        Vm, u = update_Vm(Vm,u,inp_gain,out_gain,Ih_gmax,Ih_midv,Ih_slope,tau_midv,sig[t-1])
        model = deltat/Ca_tau * (Vm[neuron_index] - Vm_ref - model) + model
        rows.append(model)
    out = torch.stack(rows)
    if return_ref:
        return out, Vm_ref
    return out


def _run_conductance_full(p, sig):
    # Multi-column forward: batched stimulus sig (B, T, N) -> (B, T-t_on, N), the
    # Ca-filtered response of EVERY unit for every stimulus. Same per-step math as
    # _run_conductance; the connectivity backend already handles the (B, N) batch.
    inp_gain, out_gain, Ih_gmax = p['inp_gain'], p['out_gain'], p['Ih_gmax']
    Ih_midv, Ih_slope, tau_midv = p['Ih_midv'], p['Ih_slope'], p['tau_midv']
    B = sig.shape[0]
    t_end = sig.shape[1]
    u  = torch.zeros((B, CONN.n_units), dtype=torch.float64).to(device)
    Vm = E_leak.expand(B, CONN.n_units).clone()
    for t in range(1, min(t_on, t_end)):
        Vm, u = update_Vm(Vm,u,inp_gain,out_gain,Ih_gmax,Ih_midv,Ih_slope,tau_midv,sig[:, t-1])
    Vm_ref = Vm.clone()                              # (B, N)
    model = 0; rows = []
    for t in range(t_on, t_end):
        Vm, u = update_Vm(Vm,u,inp_gain,out_gain,Ih_gmax,Ih_midv,Ih_slope,tau_midv,sig[:, t-1])
        model = deltat/Ca_tau * (Vm - Vm_ref - model) + model
        rows.append(model)
    return torch.stack(rows, dim=1)                  # (B, t_end-t_on, N)


def _window_time_traces(model_full, b_idx, u_idx, t0, win=None):
    """Extract per-readout windows from ``model_full`` (B, T', N).

    ``t0`` is the absolute simulation step of window start. Steps before ``t_on``
    are zero (``model_full`` only exists from ``t_on`` onward).
    """
    win = int(data.shape[1] if win is None else win)
    t_rel = t0[:, None] - t_on + torch.arange(win, dtype=torch.long, device=device)
    t_max = model_full.shape[1] - 1
    pre = t_rel < 0
    t_safe = t_rel.clamp(0, t_max)
    sel = model_full[b_idx[:, None], t_safe, u_idx[:, None]]
    return torch.where(pre, torch.zeros_like(sel), sel)


def _readout_model_traces(model_full, b_idx, u_idx, cost_t0=None):
    """Select model traces for cost cells; windowed when ``cost_t0`` / ``COST_T0`` is set."""
    t0 = COST_T0 if cost_t0 is None else cost_t0
    if t0 is None:
        return model_full[b_idx, :, u_idx]
    return _window_time_traces(model_full, b_idx, u_idx, t0, win=data.shape[1])


def _readout_model_traces_pack(model_full, pack: TargetPack):
    """Like _readout_model_traces but driven by a TargetPack (no globals)."""
    if pack.cost_t0 is None:
        return model_full[pack.readout_batch, :, pack.readout_unit]
    return _window_time_traces(
        model_full, pack.readout_batch, pack.readout_unit, pack.cost_t0,
        win=pack.data.shape[1],
    )


def _pack_cost(z, pack: TargetPack):
    p = assign_params(z, active_schema())
    model_full = _run_conductance_full(p, pack.signal)  # (B, T-t_on, N)
    sel = _readout_model_traces_pack(model_full, pack)
    diff = sel - pack.data
    return torch.sum(pack.cost_weight[:, None] * diff ** 2) / pack.power * 100.0


def _pack_shift_cost(z, pack: TargetPack, shift: int):
    """CPU-sequential cost for one batch index of a TargetPack."""
    p = assign_params(z, active_schema())
    sig_b = pack.signal[shift:shift + 1]  # (1, T, N)
    model_full = _run_conductance_full(p, sig_b)  # (1, T-t_on, N)
    mask = (pack.readout_batch == int(shift))
    if not bool(mask.any()):
        return torch.zeros((), dtype=torch.float64, device=device)
    u_m = pack.readout_unit[mask]
    if pack.cost_t0 is None:
        sel = model_full[0, :, u_m].transpose(0, 1)
    else:
        b_zero = torch.zeros_like(u_m)
        sel = _window_time_traces(
            model_full, b_zero, u_m, pack.cost_t0[mask],
            win=pack.data.shape[1],
        )
    diff = sel - pack.data[mask]
    return torch.sum(pack.cost_weight[mask][:, None] * diff ** 2) / pack.power * 100.0


def multicol_cost(z):
    # Multi-column / network cost: batched forward over all stimuli, then the
    # READOUT (batch, unit) cost cells are compared to the target with
    # per-ring COST_WEIGHT (tile) or per-column windows (moving-bar).
    p = assign_params(z, active_schema())
    model_full = _run_conductance_full(p, signal)     # (B, T-t_on, N)
    b_idx, u_idx = READOUT
    sel = _readout_model_traces(model_full, b_idx, u_idx)
    diff = sel - data
    return torch.sum(COST_WEIGHT[:, None] * diff ** 2) / power * 100.0


def multicol_shift_cost(z, shift):
    # Single-stimulus (one batch index) forward+cost, for CPU-sequential training
    # that backprops one stimulus at a time to bound peak memory. Mathematically a
    # partial sum of multicol_cost over the cost cells belonging to this batch.
    p = assign_params(z, active_schema())
    sig_b = signal[shift:shift+1]                     # (1, T, N)
    model_full = _run_conductance_full(p, sig_b)      # (1, T-t_on, N)
    b_idx, u_idx = READOUT
    mask = (b_idx == shift)
    if not bool(mask.any()):
        return torch.zeros((), dtype=torch.float64, device=device)
    u_m = u_idx[mask]
    if COST_T0 is None:
        sel = model_full[0, :, u_m].transpose(0, 1)
    else:
        b_zero = torch.zeros_like(u_m)
        sel = _window_time_traces(model_full, b_zero, u_m, COST_T0[mask], win=data.shape[1])
    diff = sel - data[mask]
    return torch.sum(COST_WEIGHT[mask][:, None] * diff ** 2) / power * 100.0


def use_network(
    network_json,
    multi_column=True,
    share_edges=False,
    sequential=None,
    dev=None,
    target="tile",
    moving_bar_center_column=False,
    target_list=None,
    loss_weights=None,
    tile_center_column=False,
):
    """Switch the simulator onto a network read from ``network.json``.

    Rebinds the connectivity backend (ScatterConn), the type vocabulary / schema,
    the resting / Ih state vectors, and the stimulus + training target.
    The default Borst 5-column path is unaffected until this is called.

    target: ``tile`` (hex RecF×ImpR) or ``moving_bar`` (fig1_ci + per-column windows).
    moving_bar_center_column: if True, moving-bar cost uses hex centre column only.
    multi_column: True -> 7-shift batched tile target; False -> single stimulus.
    share_edges:  31 disjoint tiles (False) vs 43 edge-sharing (True), full graph.
    sequential:   None -> auto (CPU sequential, CUDA batched).
    """
    global CONN, NETWORK, NODE_TYPE, nofcells, nofcols
    global E_LEAK_DEPOL_CELLS, E_leak, Ih_dir
    global CONDUCTANCE_SCHEMA, ADAPTIVE_SCHEMA, nofparams_adaptive
    global z_bounds, z_bounds_adaptive
    global data, power, signal, mc_cell_index, maxtime
    global READOUT, COST_WEIGHT, COST_T0, MC_COST_RADIUS, TARGET_KIND, MULTI_COLUMN_SEQ
    global NETWORK_TRAIN_OPTS, MOVING_BAR_CENTER_COLUMN
    global TARGET_PACKS, TARGET_PACK_WEIGHTS

    from network.construction import load_network

    dev = dev or device
    C = load_network(network_json, device=dev,
                        exc_synweight=exc_synweight, inh_synweight=inh_synweight)
    NETWORK = C
    CONN = C.conn
    NODE_TYPE = C.node_type
    nofcells = C.n_types
    nofcols = 1

    tn = list(C.type_names)
    lamina = [tn.index(t) for t in ['L1', 'L2', 'L3', 'L4', 'L5'] if t in tn]
    E_LEAK_DEPOL_CELLS = [tn.index(t) for t in ['L1', 'L2', 'L3'] if t in tn]

    CONDUCTANCE_SCHEMA = build_conductance_schema(nofcells, lamina)
    ADAPTIVE_SCHEMA = build_adaptive_schema(nofcells, lamina)
    nofparams_adaptive = schema_nparams(ADAPTIVE_SCHEMA)

    # per-TYPE resting potential broadcast to the n_units state via node_type.
    per_type = torch.full((nofcells,), E_LEAK_REST, dtype=torch.float64, device=dev)
    for c in E_LEAK_DEPOL_CELLS:
        per_type[c] = E_LEAK_DEPOL
    E_leak = calc_multi_col_params(per_type)
    Ih_dir = torch.ones(CONN.n_units, dtype=torch.float64, device=dev)

    z_bounds = calc_z_bounds()
    z_bounds_adaptive = calc_z_bounds_adaptive()

    # Multi-target mode: build a TargetPack per requested target name, then calc_cost()
    # sums them (optionally weighted). We still keep the legacy globals pointed at
    # the FIRST target for compatibility with code paths that assume `signal/data`.
    TARGET_PACKS = None
    TARGET_PACK_WEIGHTS = None
    if target_list is not None:
        if isinstance(target_list, str):
            target_list = [t.strip() for t in str(target_list).split(",") if t.strip()]
        target_list = list(target_list)
        packs: Dict[str, TargetPack] = {}
        for tname in target_list:
            if tname == "moving_bar":
                from network.moving_bar_target import build_moving_bar_target
                Tm = build_moving_bar_target(
                    C, device=dev, t_on=t_on, center_column=moving_bar_center_column,
                )
                packs[tname] = TargetPack(
                    name=tname,
                    signal=Tm.signal,
                    data=Tm.data,
                    power=Tm.power,
                    cost_weight=Tm.cost_weight,
                    readout_batch=Tm.readout_batch,
                    readout_unit=Tm.readout_unit,
                    cost_t0=Tm.cost_t0,
                )
            elif tname == "tile":
                from network.target import build_shifted_target
                Tr = build_shifted_target(
                    C,
                    share_edges=share_edges,
                    single_shift=not multi_column,
                    device=dev,
                    maxtime=maxtime,
                    t_on=t_on,
                    center_column=bool(tile_center_column),
                )
                packs[tname] = TargetPack(
                    name=tname,
                    signal=Tr.signal,
                    data=Tr.data,
                    power=Tr.power,
                    cost_weight=Tr.cost_weight,
                    readout_batch=Tr.readout_batch,
                    readout_unit=Tr.readout_unit,
                    cost_t0=None,
                )
            else:
                raise ValueError(f"unknown target {tname!r} (expected moving_bar|tile)")

        TARGET_PACKS = packs
        TARGET_PACK_WEIGHTS = {str(k): float(v) for k, v in (loss_weights or {}).items()}
        TARGET_KIND = "multi"
        first = packs[target_list[0]]
        signal = first.signal
        data = first.data
        power = first.power
        COST_WEIGHT = first.cost_weight
        COST_T0 = first.cost_t0
        READOUT = (first.readout_batch, first.readout_unit)
        mc_cell_index = first.readout_unit
        MC_COST_RADIUS = torch.zeros(first.cost_weight.shape[0], dtype=torch.float64, device=dev)
        maxtime = int(first.signal.shape[1])
        tag = f"multi-target ({'+'.join(target_list)})"
        target_norm = "multi"
    else:
        if target not in ("tile", "moving_bar"):
            raise ValueError(f"unknown target {target!r} (expected tile|moving_bar)")
        target_norm = "moving_bar" if target == "moving_bar" else "tile"
        if target_norm == "moving_bar":
            from network.moving_bar_target import build_moving_bar_target
            T = build_moving_bar_target(
                C, device=dev, t_on=t_on, center_column=moving_bar_center_column,
            )
            TARGET_KIND = "moving_bar"
            COST_T0 = T.cost_t0
            MC_COST_RADIUS = torch.zeros(T.cost_weight.shape[0], dtype=torch.float64, device=dev)
            maxtime = int(T.maxtime)
            coltag = "centre column" if moving_bar_center_column else f"{T.info['n_cost_columns']} photo columns"
            tag = f"moving-bar (B={T.n_batch} stimuli, {T.info['n_cost']} cost cells, {coltag})"
        else:
            from network.target import build_shifted_target
            T = build_shifted_target(
                C, share_edges=share_edges, single_shift=not multi_column, device=dev,
                maxtime=maxtime, t_on=t_on,
            )
            TARGET_KIND = "tile"
            COST_T0 = None
            MC_COST_RADIUS = T.cost_radius
            maxtime = int(T.signal.shape[1]) if T.signal.ndim == 3 else ml.IMPULSE_MAXTIME
            tag = (
                f"{'multi-column' if multi_column else 'single-column'} "
                f"(B={T.n_batch} stimuli [{T.info['n_centers']} tiles x "
                f"{T.info['n_shifts']} shifts], {T.info['n_cost']} cost cells)"
            )

        signal = T.signal
        data = T.data
        power = T.power
        COST_WEIGHT = T.cost_weight
        READOUT = (T.readout_batch, T.readout_unit)
        mc_cell_index = T.readout_unit          # plotting / out_scale fallback
    MULTI_COLUMN_SEQ = (dev == 'cpu') if sequential is None else bool(sequential)
    MOVING_BAR_CENTER_COLUMN = bool(moving_bar_center_column) if target_norm == "moving_bar" else False
    target_field = ",".join(target_list) if target_list is not None else str(target_norm)
    NETWORK_TRAIN_OPTS = {
        "network_json": str(Path(network_json).resolve()),
        "multi_column": bool(multi_column),
        "share_edges": bool(share_edges),
        "target": target_field,
        "sequential": bool(MULTI_COLUMN_SEQ),
        "moving_bar_center_column": bool(moving_bar_center_column),
        "target_list": list(target_list) if target_list is not None else None,
        "loss_weights": {str(k): float(v) for k, v in (loss_weights or {}).items()} if target_list is not None else None,
        "tile_center_column": bool(tile_center_column),
    }

    seqtag = ", sequential CPU" if MULTI_COLUMN_SEQ else ""
    print(f"network: {network_json}")
    print(f"  {tag}{seqtag}")
    print(f"  n_units={CONN.n_units}, n_types={nofcells}, "
          f"nparams={schema_nparams(active_schema())}, maxtime={maxtime}")
    return C

def simulate_conductance(z):
    return _run_conductance(assign_params(z, CONDUCTANCE_SCHEMA))

def _run_adaptive(p, inp_scalar=None, return_diag=False, neuron_index=None, return_ref=False):
    # forward pass -> low-pass filtered response for the chosen neurons, shape
    # (150, n). neuron_index defaults to the center-column cost cells (mc_cell_index);
    # plotting passes other columns.
    # inp_scalar: override per-cell inp_gain with a uniform value (parameter sweeps).
    # return_diag: also return {max|activity|, clamp-hit %} for stability analysis.
    # return_ref: also return the resting baseline the trace is measured against.
    if 'gate_pivot' not in p:
        p = {**p, 'gate_pivot': GATE_PIVOT}
    if inp_scalar is not None:
        p = {**p, 'inp_gain': torch.full_like(p['bias'], float(inp_scalar))}
    if neuron_index is None:
        neuron_index = mc_cell_index
    bias = p['bias']
    x_signal = signal / ml.SIGNAL_BRIGHT  # normalise input to [0,1] so gate_pivot ~ 0.5 is meaningful

    activity    = bias.clone()
    v_sustained = bias.clone()
    v_transient = torch.zeros_like(bias)
    drive_lp    = bias.clone()

    maxact = 0.0; hits = 0; tot = 0
    act_ref = None; model = 0; rows = []
    for t in range(1, maxtime):
        x_t = x_signal[t-1]
        x_d = x_signal[max(t-1-gate_lag, 0)]
        activity, v_sustained, v_transient, drive_lp = update_state_adaptive(
            activity, v_sustained, v_transient, drive_lp, p, x_t, x_d)
        if return_diag:
            maxact = max(maxact, float(activity.abs().max()))
            hits += int((v_sustained.abs() >= STATE_CLAMP).sum() + (v_transient.abs() >= STATE_CLAMP).sum())
            tot += 2 * v_sustained.numel()
        if t == t_on - 1:
            act_ref = 1.0*activity[neuron_index]  # reference 0
        elif t >= t_on:
            model = deltat/Ca_tau * (activity[neuron_index] - act_ref - model) + model
            rows.append(model)
    model = torch.stack(rows)
    if return_diag:
        return model, {'maxact': maxact, 'clamp_pct': 100.0 * hits / tot}
    if return_ref:
        return model, act_ref
    return model

def simulate_adaptive(z, inp_scalar=None, return_diag=False):
    return _run_adaptive(assign_params_adaptive(z), inp_scalar, return_diag)

#@torch.compile
def calc_cost(z, data, out_scale=1.0):
    # out_scale (the arg) multiplies on top of any 'out_scale' parameter declared in
    # the active schema, so legacy callers (arg) and schema-driven training both work.
    # When multi-target packs are active, sum their costs (weights applied).
    # The 'data' arg is ignored in that mode (kept for API compatibility).
    if TARGET_PACKS is not None:
        weights = TARGET_PACK_WEIGHTS or {}
        total = 0.0
        for name, pack in TARGET_PACKS.items():
            w = float(weights.get(name, 1.0))
            if w == 0.0:
                continue
            if MULTI_COLUMN_SEQ:
                part = 0.0
                for b in range(pack.signal.shape[0]):
                    part = part + _pack_shift_cost(z, pack, b)
            else:
                part = _pack_cost(z, pack)
            total = total + w * part
        return total

    # When a network multi-column target is active (signal is (B, T, N)), route to
    # the batched tile-target cost; the Borst 5-column path is untouched.
    if NETWORK is not None and signal.dim() == 3:
        if MULTI_COLUMN_SEQ:
            total = 0.0
            for b in range(signal.shape[0]):
                total = total + multicol_shift_cost(z, b)
            return total
        return multicol_cost(z)
    schema = active_schema()
    p = assign_params(z, schema)
    if MODEL_TYPE == 'adaptive':
        model = _run_adaptive({**p, 'gate_pivot': GATE_PIVOT})
    else:
        model = _run_conductance(p)
    # out_scale is declared per cell-TYPE (nofcells,), but model columns are the
    # COST CELLS (mc_cell_index, here = the default neuron set used by _run_*).
    # Map each cost cell to its cell type so out_scale aligns no matter how many
    # cost cells there are (lets experiments add cost cells, e.g. R1-8).
    os_param = p.get('out_scale', 1.0)
    if torch.is_tensor(os_param) and os_param.dim() > 0:
        ci = torch.as_tensor(mc_cell_index, device=os_param.device, dtype=torch.long) % nofcells
        os_param = os_param[ci]
    return model_cost(model, data, out_scale * os_param)

def calc_cost_adaptive(z, data, out_scale=1.0):
    return model_cost(simulate_adaptive(z), data, out_scale)
    
def schema_bounds(schema):
    zb = torch.zeros((schema_nparams(schema), 2), dtype=torch.float64)
    for seg, start, stop in schema_segments(schema):
        if stop > start:                       # skip fixed (0 trainable rows)
            zb[start:stop] = torch.tensor([seg['lo'], seg['hi']], dtype=torch.float64)
    return zb

def schema_guess(schema):
    z = np.zeros(schema_nparams(schema))
    for seg, start, stop in schema_segments(schema):
        n = stop - start
        if n == 0:                             # fixed: nothing to initialise
            continue
        z[start:stop] = seg['init'] + (np.random.rand(n) - 0.5) * seg['jit']
        if seg_mode(seg) == 'individual':      # 'zero' only meaningful per-unit
            for j in seg.get('zero', []):      # e.g. Ih_gmax L3,L4 start at 0
                z[start + j] = 0.0
    return torch.tensor(z, dtype=torch.float64).to(device)

def calc_z_bounds():
    return schema_bounds(CONDUCTANCE_SCHEMA)

z_bounds = calc_z_bounds()

def calc_z_bounds_adaptive():
    return schema_bounds(ADAPTIVE_SCHEMA)

z_bounds_adaptive = calc_z_bounds_adaptive()

def guess_initial_params_adaptive():
    return schema_guess(ADAPTIVE_SCHEMA)

def guess_initial_params():
    return schema_guess(CONDUCTANCE_SCHEMA)

def gradient_network(data, z, lr=0.0001, cost_fn=None, n_steps=100, device="cpu", z_bounds=None,
                     cost_log=None):
    
    a = time.time()

    z = nn.Parameter(z.clone().to(device))
    data = data.to(device)
    
    optimizer = torch.optim.Adam([z], lr=lr)

    # Calculate initial cost and move it to the specified device
    
    cost = cost_fn(z, data).item()
    best_cost = cost
    best_z = z.clone().detach()
    
    initial_cost = 1.0*cost

    progress_bar = tqdm(range(n_steps), desc=f'Cost: {cost:.4f}')

    for i in progress_bar:
        
        optimizer.zero_grad()
        
        cost = cost_fn(z, data)  
        
        if cost.item() < best_cost:
            
            best_cost = cost.item()
            best_z = z.clone().detach()
        
        if cost_log is not None:
            cost_log.append(cost.item())
        
        cost.backward()
        optimizer.step()

        with torch.no_grad():
            
            z.clamp_(z_bounds[:, 0].to(device), z_bounds[:, 1].to(device))

        progress_bar.set_description(f'Cost: {cost.item():.4f}')

    cost = cost_fn(z, data)  
    
    if cost.item() < best_cost:
        
        best_cost = cost.item()
        best_z = z.clone().detach()

    print()
    print('Initl cost =', format(initial_cost,'.4f'))
    print('Final cost =', format(cost.item(),'.4f'))
    print('Best  cost =', format(best_cost,'.4f'))
    
    b = time.time()
    
    print('time needed  =',format(b-a,'.2f'),' sec')
    print()

    return best_z

def train_staged(z, data, cost_fn, z_bounds, lrs, nsteps, cost_log=None):
    # run gradient_network once per learning-rate stage, chaining the best params.
    for lr in lrs:
        z = gradient_network(data, z, lr=lr, n_steps=nsteps, device=device,
                             cost_fn=cost_fn, z_bounds=z_bounds, cost_log=cost_log)
    return z

def do_many_runs(nofruns,nofsteps,fname,lrs=(0.1, 0.01, 0.001),outdir='FiveCol_Parameter'):
    
    os.makedirs(outdir, exist_ok=True)
    
    # record the model type next to the params so plotting never has to guess it
    with open(os.path.join(outdir, 'model_type.txt'), 'w') as f:
        f.write(MODEL_TYPE)

    if NETWORK_TRAIN_OPTS is not None:
        import json
        with open(os.path.join(outdir, 'target_kind.txt'), 'w') as f:
            f.write(str(NETWORK_TRAIN_OPTS.get('target', TARGET_KIND or 'tile')))
        with open(os.path.join(outdir, 'network_path.txt'), 'w') as f:
            f.write(NETWORK_TRAIN_OPTS['network_json'])
        with open(os.path.join(outdir, 'network_train_opts.json'), 'w') as f:
            json.dump(NETWORK_TRAIN_OPTS, f)
    if PARAM_TRAIN_OPTS:
        import json
        with open(os.path.join(outdir, 'param_train_opts.json'), 'w') as f:
            json.dump(PARAM_TRAIN_OPTS, f)
    
    schema   = active_schema()
    n_params = schema_nparams(schema)
    guess_fn = lambda: schema_guess(schema)
    bounds   = schema_bounds(schema)
    
    all_params = np.zeros((nofruns,n_params))
    
    costs_name = fname.replace('.npy', '') + '_costs.npy'
    best_final_cost = np.inf
    best_cost_history = None
    
    for i in range(nofruns):
        
        print()
        print('round',i)
        print()
        
        z = guess_fn()

        # record per-step cost across all lr stages (same as local_cpu_test)
        cost_history = []

        z_fit = train_staged(z, data, calc_cost, bounds, lrs, nofsteps, cost_log=cost_history)
        
        all_params[i] = z_fit.detach().cpu().numpy()
        
        np.save(os.path.join(outdir, fname), all_params)

        # keep the per-step cost curve of the best run so far
        final_cost = calc_cost(z_fit, data).item()
        if final_cost < best_final_cost:
            best_final_cost = final_cost
            best_cost_history = np.array(cost_history)
            np.save(os.path.join(outdir, costs_name), best_cost_history)
        
def refine_many_runs(all_params,nofsteps,lr = 0.01):
    
    dirname = 'FiveCol_Parameter/'
    fname   = 'refined_paramsets.npy'
    
    nofruns = all_params.shape[0]
    
    ref_params = np.zeros((nofruns,nofparams))
    
    for i in range(nofruns):
        
        print()
        print('round',i)
        print()
        
        z = all_params[i]
        z = torch.tensor(z).double().requires_grad_()

        z_fit = gradient_network(
            data,
            z,
            lr=lr,
            n_steps=nofsteps,
            device=device,
            cost_fn=calc_cost,
            z_bounds = z_bounds
        )
        
        ref_params[i] = z_fit.detach().cpu().numpy()
        
        np.save(dirname+fname,ref_params)
        
def save_numpy_parameters(z_fit,fname):
    
    dirname = 'FiveCol_Parameter/'
    z = z_fit.detach().cpu().numpy()
    np.save(dirname+fname,z)
    
    
if __name__ == "__main__":
    
    dirname = 'FiveCol_Parameter/with_Ih/'

    fname = '4Ih_paramset_L4isol_NoGaps.npy'
    z = np.load(dirname+fname)
    z = torch.tensor(z, dtype=torch.float64).to(device)
    
    #z = guess_initial_params()
    
    z_fit = gradient_network(
        data,
        z,
        lr=0.001,
        n_steps=10,
        device=device,
        cost_fn=calc_cost,
        z_bounds = z_bounds
    )
    # print(f"final cost: {calc_cost(z_fit,data)} , initial cost {calc_cost(z,data)}")



    



