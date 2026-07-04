# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Created on Wed Jul 26 09:53:25 2023

@author: aborst
"""
import os
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import Medulla_Library as ml
import time

import torch
from torch import nn
from tqdm import tqdm

from network.connectivity import DenseConn


def active_device():
    """Pick CUDA or CPU from current runtime (not frozen at import)."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def __getattr__(name):
    if name == 'device':
        return active_device()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

#################################################################
# Medulla Library contains:
# ml.read_ConnMs()
# ml.read_RecF_data(): RecF_data (13,45), ImpR_data (13,45)
# plot_ConnM(): Big ConnM + intra + inter M
# stimulus generation -> signal
#################################################################

BORST_NOFCELLS = 65
BORST_NOFCOLS = 5
t_on = ml.T_ON
TRAIN_OPTS_FILE = "train_opts.json"
TILE_TARGETS = ("tile_bright", "tile_dark")
MOVING_BAR_TARGETS = ("moving_bar_bright", "moving_bar_dark")
VALID_TARGETS = TILE_TARGETS + MOVING_BAR_TARGETS
TARGET_ALIASES = {
    "tile": TILE_TARGETS,
    "moving_bar": MOVING_BAR_TARGETS,
}
CLI_TARGET_NAMES = VALID_TARGETS + tuple(TARGET_ALIASES.keys())
TARGET_I_FIELDS = {
    "tile_bright": frozenset({"i_baseline", "i_bright"}),
    "tile_dark": frozenset({"i_baseline", "i_dark"}),
    "moving_bar_bright": frozenset({"i_baseline", "i_bright_bar"}),
    "moving_bar_dark": frozenset({"i_baseline", "i_dark_bar"}),
}
I_CLI_BRIGHT_TARGETS = {
    "tile": ("tile_bright",),
    "tile_bright": ("tile_bright",),
    "moving_bar": ("moving_bar_bright",),
    "moving_bar_bright": ("moving_bar_bright",),
}
I_CLI_DARK_TARGETS = {
    "tile": ("tile_dark",),
    "tile_dark": ("tile_dark",),
    "moving_bar": ("moving_bar_dark",),
    "moving_bar_dark": ("moving_bar_dark",),
}
I_CLI_SIDECAR_FIELD = {
    ("i_baseline", "tile_bright"): "i_baseline",
    ("i_baseline", "tile_dark"): "i_baseline",
    ("i_baseline", "moving_bar_bright"): "i_baseline",
    ("i_baseline", "moving_bar_dark"): "i_baseline",
    ("i_bright", "tile_bright"): "i_bright",
    ("i_bright", "moving_bar_bright"): "i_bright_bar",
    ("i_dark", "tile_dark"): "i_dark",
    ("i_dark", "moving_bar_dark"): "i_dark_bar",
}

# important model params

deltat    = 10.0  # in msec
g_leak    = 1.0   # in nS
E_exc     = +10.0 # in mV
E_inh     = -70.0 # in mV
capac     = +40.0 # in pF, results in 50ms membrane time-constant for g_leak = 1.0 nS
trld      = -50.0 # in mV: below trld, no signal is transmitted
cdt       = capac/deltat

Ca_tau    = 50.0  # in msec

E_LEAK_REST = -50.0
E_LEAK_DEPOL = -20.0

def calc_multi_col_params(param, conn):
    # Broadcast a per-cell-TYPE parameter (n_types,) to the full state (n_units,)
    # via the backend's node_type. For the Borst path node_type == arange%65, so
    # this reproduces the old 5x concatenation exactly.
    return param.index_select(0, conn.node_type)


def build_e_leak(conn, n_types, depol_cells=None):
    """(conn.n_units,) resting potential; default depol list from ``ml.LEAK_DEPOL_TYPES``."""
    if depol_cells is None:
        depol_cells = ml.leak_depol_indices()
    per_type = torch.full((n_types,), E_LEAK_REST, dtype=torch.float64, device=conn.node_type.device)
    for c in depol_cells:
        per_type[int(c)] = E_LEAK_DEPOL
    return calc_multi_col_params(per_type, conn)

exc_synweight = 0.001
inh_synweight = 0.001

# ----------- H-Current ----------------------------------------

E_Ih          = +50.0  # in mV, ON-channel reversal
E_IH_OFF      = -150.0  # OFF-channel reversal (2*E_LEAK_REST - E_Ih)
Ih_midv       = -50.0
Ih_slope      = -0.25
tau_midv      = -50.0
Ih_gmax       = +50.0 

Ih_gain       = 1.0   # if set to 0, it will block Ih

IH_OFF_MODES = ('shared', 'split', 'off')
IH_OFF_DEFAULT = 'shared'
IH_OFF_SCALAR_SEGMENTS = frozenset({'Ih_midv_off', 'Ih_slope_off', 'tau_midv_off'})
IH_OFF_GMAX_SEGMENT = 'Ih_gmax_off'

# Per-cell Ih direction: +1 normal; -1 mirrored (reversal flips about 0).
# Default: none reversed. Pass ih_reverse_cells= to borst_backend().
IH_DIR_REVERSE_CELLS: Tuple[int, ...] = ()

def build_ih_dir(conn, ih_reverse_cells=IH_DIR_REVERSE_CELLS):
    """(conn.n_units,) Ih direction (+1 normal, -1 mirrored per cell-type)."""
    d = torch.ones(conn.n_units, dtype=torch.float64, device=conn.node_type.device)
    for c in ih_reverse_cells:
        d[conn.node_type == int(c)] = -1.0
    return d

# parameter and cost function definition

low_gain = 0.1
high_gain = 100.0

# ---- second neuron model: adaptive temporal filter (flyvis-derived) ----
# 'conductance' = Borst conductance-based + Ih (update_Vm)
# 'adaptive'    = passive point neuron + low-pass adaptive temporal filter

gate_lag = 1  # delay (in steps) of the stimulus used for the contrast gate
GATE_PIVOT = 0.5  # fixed contrast-gate pivot (non-trainable); input is normalised to [0,1]
STATE_CLAMP = 1.0e6  # bound on adaptive state vars to keep explicit Euler finite

# --- parameter schema: SINGLE SOURCE OF TRUTH -------------------------------
# Segment lists are built by ``build_conductance_schema`` / ``build_adaptive_schema``
# (via ``default_schema``); assign / bounds / guess all derive from that list.
# Each segment is a dict:
#   name  : parameter name (becomes a key in the assigned dict)
#   count : number of per-unit values for this segment (the 'individual' width)
#   kind  : how `count` raw values become a usable parameter:
#           'full'   -> count==n_types; one value per cell type, replicated to columns
#           'lamina' -> one value per entry in seg['cells'] (default L1-L5 via
#                       LAMINA_SLICE); an entry may be a list of indices that SHARE
#                       one value; other cells = 'fill'. 'count' is ignored for this kind.
#           'scalar' -> count==1; a single global 0-dim value (e.g. Ih_midv)
#           'output' -> count==n_types; per-cell-type value on the readout (e.g. out_scale)
#   lo,hi : training bounds (clamped each Adam step)
#   init  : random-init mean;  jit: init uniform jitter (+/- jit/2)
#   fill  : value for non-listed cells ('lamina' only)
#   zero  : lamina-local indices set to 0 at init (from IH_GMAX_ZERO_TYPES cell names)
#   mode  : 'individual' (train all `count` values, default), 'shared' (train ONE value
#           broadcast to all units), or 'fixed' (train NOTHING; held at 'fixed' or 'init').
#   fixed : constant value used when mode=='fixed' (defaults to 'init').
# `mode`/`fixed` are normally NOT set here; they are overridden per run from the
# CLI / SLURM (see run.py --mode / --fix), so the schema stays the canonical default.
LAMINA_SLICE = ml.LAMINA_SLICE  # L1-L5 within the 65 cell types
IH_GMAX_ZERO_TYPES = ('L3', 'L4')  # Ih_gmax z-init pinned to 0 for these types
PARAM_MODES = ('individual', 'shared', 'fixed')

# Lamina/scalar segments expanded to per-cell-type (full) by --per_type.
PER_TYPE_PARAM_NAMES = frozenset({
    'Ih_gmax', 'Ih_gmax_off', 'Ih_midv', 'Ih_slope', 'tau_midv',
    'adapt_gain', 'tau_adapt',
})

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


def apply_ih_off_mode(schema, mode=IH_OFF_DEFAULT):
    """Adjust conductance Ih schema for ON/OFF coupling (``shared|split|off``).

    ``shared`` / ``off``: drop ``Ih_gmax_off`` and OFF shape scalars from z;
    forward resolves OFF gmax via :func:`conductance_ih_off_kwargs`.
    """
    if mode not in IH_OFF_MODES:
        raise ValueError(f"ih_off {mode!r} not in {IH_OFF_MODES}")
    out = []
    for seg in schema:
        s = dict(seg)
        name = s['name']
        if mode == 'split':
            out.append(s)
            continue
        if name in IH_OFF_SCALAR_SEGMENTS or name == IH_OFF_GMAX_SEGMENT:
            continue
        out.append(s)
    return out


def conductance_schema(model_backend, schema=None, ih_off=IH_OFF_DEFAULT):
    """Conductance parameter schema with ``ih_off`` segment selection applied."""
    base = list(schema) if schema is not None else default_schema('conductance', model_backend)
    return apply_ih_off_mode(base, ih_off)


def conductance_ih_off_kwargs(p, ih_off=IH_OFF_DEFAULT):
    """Resolve OFF-channel Ih kwargs for :func:`update_Vm` from assigned params."""
    midv_off = p['Ih_midv'] if ih_off != 'split' else p['Ih_midv_off']
    slope_off = p['Ih_slope'] if ih_off != 'split' else p['Ih_slope_off']
    tau_off = p['tau_midv'] if ih_off != 'split' else p['tau_midv_off']
    if ih_off == 'split':
        gmax_off = p['Ih_gmax_off']
    elif ih_off == 'shared':
        gmax_off = p['Ih_gmax']
    elif ih_off == 'off':
        gmax_off = p['Ih_gmax'] * 0.0
    else:
        raise ValueError(f"ih_off {ih_off!r} not in {IH_OFF_MODES}")
    return gmax_off, midv_off, slope_off, tau_off


def expand_schema_per_type(schema, n_types=BORST_NOFCELLS, names=None):
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
        s['count'] = n_types
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


def lamina_zero_indices(lamina, zero_types, type_names):
    """Map cell-type names to local indices into a lamina ``cells`` list."""
    names = [str(n) for n in type_names]
    by_type = {}
    for j, entry in enumerate(lamina):
        targets = [entry] if isinstance(entry, int) else entry
        for t in targets:
            by_type[int(t)] = j
    out = []
    for zname in zero_types:
        matches = [i for i, n in enumerate(names) if n == str(zname)]
        if len(matches) != 1:
            raise KeyError(f"cell type {zname!r}: {len(matches)} matches in type_names")
        pos = by_type.get(matches[0])
        if pos is not None:
            out.append(pos)
    return out


def build_conductance_schema(n_types, lamina, ih_zero_types=IH_GMAX_ZERO_TYPES, type_names=None):
    type_names = ml.ctype if type_names is None else type_names
    zero = lamina_zero_indices(lamina, ih_zero_types, type_names)
    return [
        {'name': 'inp_gain',  'count': n_types, 'kind': 'full',   'lo': low_gain, 'hi': high_gain, 'init': 0.5,     'jit': 0.2,  'fill': 0.0},
        {'name': 'out_gain',  'count': n_types, 'kind': 'full',   'lo': low_gain, 'hi': high_gain, 'init': 0.5,     'jit': 0.2,  'fill': 0.0},
        {'name': 'Ih_gmax',     'count': len(lamina), 'kind': 'lamina', 'cells': lamina, 'lo': 0.0, 'hi': 100.0, 'init': Ih_gmax, 'jit': 10.0, 'fill': 0.0, 'zero': zero},
        {'name': 'Ih_gmax_off', 'count': len(lamina), 'kind': 'lamina', 'cells': lamina, 'lo': 0.0, 'hi': 100.0, 'init': Ih_gmax, 'jit': 10.0, 'fill': 0.0, 'zero': zero},
        {'name': 'Ih_midv',     'count': 1,       'kind': 'scalar', 'lo': -70.0,    'hi': -30.0,     'init': Ih_midv,  'jit': 5.0},
        {'name': 'Ih_slope',    'count': 1,       'kind': 'scalar', 'lo': -0.40,    'hi': -0.20,     'init': Ih_slope, 'jit': 0.02},
        {'name': 'tau_midv',    'count': 1,       'kind': 'scalar', 'lo': -70.0,    'hi': -40.0,     'init': tau_midv, 'jit': 5.0},
        {'name': 'Ih_midv_off', 'count': 1,       'kind': 'scalar', 'lo': -70.0,    'hi': -30.0,     'init': Ih_midv,  'jit': 5.0},
        {'name': 'Ih_slope_off','count': 1,       'kind': 'scalar', 'lo': -0.40,    'hi': -0.20,     'init': Ih_slope, 'jit': 0.02},
        {'name': 'tau_midv_off','count': 1,       'kind': 'scalar', 'lo': -70.0,    'hi': -40.0,     'init': tau_midv, 'jit': 5.0},
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


BORST_LAMINA = list(range(LAMINA_SLICE.start, LAMINA_SLICE.stop))


def default_schema(model_type: str, backend: "ModelBackend") -> list:
    """Fresh parameter schema for ``model_type`` on the given backend."""
    if backend.network is not None:
        tn = list(backend.network.type_names)
        lamina = [tn.index(t) for t in ['L1', 'L2', 'L3', 'L4', 'L5'] if t in tn]
        n = backend.n_types
        type_names = tn
    else:
        lamina = BORST_LAMINA
        n = BORST_NOFCELLS
        type_names = ml.ctype
    if model_type == 'adaptive':
        return build_adaptive_schema(n, lamina)
    return build_conductance_schema(n, lamina, type_names=type_names)


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
    cost_radius: Optional[torch.Tensor] = None  # (n_cost,) ring radius for network tile
    center_column: bool = False  # cost restricted to centre column / ring r=0


@dataclass(frozen=True)
class ModelBackend:
    """Connectivity + leak/Ih tensors for one simulation graph."""

    conn: object
    e_leak: torch.Tensor
    ih_dir: torch.Tensor
    n_types: int
    n_cols: int
    network: Optional[object] = None
    ctype: Optional[object] = None
    depol_cells: Tuple[int, ...] = field(default_factory=ml.leak_depol_indices)
    ih_reverse_cells: Tuple[int, ...] = IH_DIR_REVERSE_CELLS

    @property
    def n_units(self) -> int:
        return self.conn.n_units


@dataclass(frozen=True)
class TrainSession:
    """Immutable runtime context for one training / plotting run."""

    backend: ModelBackend
    model_type: str
    schema: tuple
    targets: Dict[str, TargetPack]
    target_list: Tuple[str, ...]
    loss_weights: Dict[str, float]
    sequential: bool
    device: str
    train_opts: Optional[dict] = None

    def with_schema(self, schema) -> "TrainSession":
        return replace(self, schema=tuple(schema))

    @property
    def primary_pack(self) -> TargetPack:
        return self.targets[self.target_list[0]]

    @property
    def maxtime(self) -> int:
        sig = self.primary_pack.signal
        return int(sig.shape[1] if sig.dim() == 3 else sig.shape[0])

    def pack_signal(self, pack: Optional[TargetPack] = None) -> torch.Tensor:
        pack = pack or self.primary_pack
        sig = pack.signal
        if pack.name in TILE_TARGETS and sig.dim() == 3 and int(sig.shape[0]) == 1:
            sig = sig.squeeze(0)
        return sig

    def pack_for(self, name: str) -> TargetPack:
        if name not in self.targets:
            raise KeyError(f"target pack {name!r} not in session")
        return self.targets[name]


@dataclass(frozen=True)
class TrainingResult:
    """Output of :func:`do_many_runs` (in memory; persistence is ``run.save_training_outputs``)."""

    all_params: np.ndarray   # (nofruns, n_params)
    final_costs: np.ndarray  # (nofruns,) weighted total
    best_i: int
    cost_curve: np.ndarray   # per-step weighted total for ``best_i``
    cost_curves_by_target: Dict[str, np.ndarray] = field(default_factory=dict)
    final_costs_by_target: Dict[str, np.ndarray] = field(default_factory=dict)


# -------------------------------------------------------------------------
# -------------- reading cell data and connectivity matrices --------------
# -------------------------------------------------------------------------

def _opt_float(opts, *keys, default=None):
    for key in keys:
        if key in opts:
            return float(opts[key])
    if default is not None:
        return float(default)
    raise KeyError(f"expected one of {keys!r} in stimulus opts")


def _tile_bright_i_from_opts(opts):
    """Read tile_bright PR currents (``i_baseline`` / ``i_bright``)."""
    return (
        _opt_float(opts, "i_baseline", default=ml.I_BASELINE),
        _opt_float(opts, "i_bright", default=ml.I_BRIGHT),
    )


def _tile_dark_i_from_opts(opts):
    """Read tile_dark PR currents (``i_baseline`` / ``i_dark``)."""
    return (
        _opt_float(opts, "i_baseline", default=ml.I_BASELINE),
        _opt_float(opts, "i_dark", default=ml.I_DARK),
    )


def _pack_signal_scale(pack: TargetPack, session: TrainSession) -> float:
    """Peak PR current for adaptive ``sig / scale`` (from per-target sidecar opts)."""
    opts = ((session.train_opts or {}).get(f"{pack.name}_stimulus_opts")) or {}
    if pack.name == "tile_bright":
        peak = _opt_float(opts, "i_bright", default=ml.I_BRIGHT)
    elif pack.name == "tile_dark":
        peak = _opt_float(opts, "i_dark", default=ml.I_DARK)
    elif pack.name == "moving_bar_bright":
        peak = _opt_float(opts, "i_bright_bar", default=ml.I_BRIGHT)
    elif pack.name == "moving_bar_dark":
        peak = _opt_float(opts, "i_dark_bar", default=ml.I_DARK)
    else:
        peak = ml.I_BRIGHT
    peak = float(peak)
    if peak == 0.0:
        return float(ml.I_BRIGHT)
    return peak


def make_tile_bright_stimulus_opts(
    i_baseline=None, i_bright=None, mode="borst", center_column=False,
    multi_shift=False, share_edges=False,
):
    """PR step stimulus for tile_bright: baseline pre-``t_on``, bright from ``t_on``."""
    baseline = float(ml.I_BASELINE if i_baseline is None else i_baseline)
    bright = float(ml.I_BRIGHT if i_bright is None else i_bright)
    return {
        "mode": mode,
        "i_baseline": baseline,
        "i_bright": bright,
        "t_on": int(t_on),
        "maxtime": int(ml.IMPULSE_MAXTIME),
        "deltat_ms": float(deltat),
        "center_column": bool(center_column),
        "multi_shift": bool(multi_shift),
        "share_edges": bool(share_edges),
    }


def make_tile_dark_stimulus_opts(
    i_baseline=None, i_dark=None, mode="borst", center_column=False,
    multi_shift=False, share_edges=False,
):
    """PR step stimulus for tile_dark: baseline pre-``t_on``, i_dark from ``t_on``."""
    baseline = float(ml.I_BASELINE if i_baseline is None else i_baseline)
    dark = float(ml.I_DARK if i_dark is None else i_dark)
    return {
        "mode": mode,
        "i_baseline": baseline,
        "i_dark": dark,
        "t_on": int(t_on),
        "maxtime": int(ml.IMPULSE_MAXTIME),
        "deltat_ms": float(deltat),
        "center_column": bool(center_column),
        "multi_shift": bool(multi_shift),
        "share_edges": bool(share_edges),
    }


def make_moving_bar_bright_stimulus_opts(
    i_baseline=None, i_bright_bar=None, mode="borst", center_column=False,
):
    """PR moving-bar bright target defaults (``maxtime``/``spec_names`` filled at build)."""
    return {
        "mode": mode,
        "i_baseline": float(ml.I_BASELINE if i_baseline is None else i_baseline),
        "i_bright_bar": float(ml.I_BRIGHT if i_bright_bar is None else i_bright_bar),
        "t_on": int(t_on),
        "deltat_ms": float(deltat),
        "center_column": bool(center_column),
    }


def make_moving_bar_dark_stimulus_opts(
    i_baseline=None, i_dark_bar=None, mode="borst", center_column=False,
):
    """PR moving-bar dark target defaults (``maxtime``/``spec_names`` filled at build)."""
    return {
        "mode": mode,
        "i_baseline": float(ml.I_BASELINE if i_baseline is None else i_baseline),
        "i_dark_bar": float(ml.I_DARK if i_dark_bar is None else i_dark_bar),
        "t_on": int(t_on),
        "deltat_ms": float(deltat),
        "center_column": bool(center_column),
    }


def _enrich_moving_bar_stimulus_opts(opts, info, *, center_column):
    """Attach runtime fields from a built moving-bar target; keep canonical ``i_*``."""
    out = dict(opts)
    out["maxtime"] = int(info["maxtime"])
    out["t_on"] = int(info["t_on"])
    out["spec_names"] = list(info["spec_names"])
    out["center_column"] = bool(center_column)
    out["deltat_ms"] = float(deltat)
    if "mode" in info:
        out["mode"] = info["mode"]
    return out


def borst_tile_bright_signal(opts=None):
    """Build Borst tile_bright PR step stimulus ``(T, N_units)``."""
    opts = dict(opts or make_tile_bright_stimulus_opts())
    n_units = ml.n_state_units()
    pr = ml.photoreceptor_slice()
    t0, T = int(opts["t_on"]), int(opts["maxtime"])
    b, step = _tile_bright_i_from_opts(opts)
    sig = torch.zeros((T, n_units), dtype=torch.float64, device=active_device())
    sig[:t0, pr] = b
    sig[t0:T, pr] = step
    return sig


def borst_tile_dark_signal(opts=None):
    """Build Borst tile_dark PR step stimulus ``(T, N_units)``."""
    opts = dict(opts or make_tile_dark_stimulus_opts())
    n_units = ml.n_state_units()
    pr = ml.photoreceptor_slice()
    t0, T = int(opts["t_on"]), int(opts["maxtime"])
    b, step = _tile_dark_i_from_opts(opts)
    sig = torch.zeros((T, n_units), dtype=torch.float64, device=active_device())
    sig[:t0, pr] = b
    sig[t0:T, pr] = step
    return sig


def _borst_tile_cost_row_slice(center_column: bool) -> slice:
    if not center_column:
        return slice(None)
    start = ml.N_FIT_CELLS * ml.CENTER_COL
    return slice(start, start + ml.N_FIT_CELLS)


def _borst_tile_pack_from_data(opts, pack_name, signal_fn, data_fn):
    """Shared Borst tile pack builder for bright/dark targets."""
    opts = dict(opts)
    center_column = bool(opts.get("center_column", False))
    row_sl = _borst_tile_cost_row_slice(center_column)
    u_idx = torch.tensor(
        np.load("Circuits/mc_cell_index.npy")[row_sl],
        dtype=torch.long,
        device=active_device(),
    )
    n = int(u_idx.shape[0])
    sig = signal_fn(opts).unsqueeze(0)
    T = int(sig.shape[1])
    tile_data = torch.tensor(data_fn(T), dtype=torch.float64, device=active_device())
    t_data = tile_data[t_on:T].transpose(0, 1).contiguous()[row_sl]
    tile_power = torch.sum(t_data ** 2)
    return TargetPack(
        name=pack_name,
        signal=sig,
        data=t_data,
        power=tile_power,
        cost_weight=torch.ones(n, dtype=torch.float64, device=active_device()),
        readout_batch=torch.zeros(n, dtype=torch.long, device=active_device()),
        readout_unit=u_idx,
        cost_t0=None,
        center_column=center_column,
    )


def build_borst_tile_bright_pack(opts=None):
    """Borst tile_bright target as a :class:`TargetPack` (batch B=1)."""
    return _borst_tile_pack_from_data(
        opts or make_tile_bright_stimulus_opts(),
        "tile_bright",
        borst_tile_bright_signal,
        ml.borst_tile_impulse_data,
    )


def build_borst_tile_dark_pack(opts=None):
    """Borst tile_dark target as a :class:`TargetPack` (batch B=1)."""
    return _borst_tile_pack_from_data(
        opts or make_tile_dark_stimulus_opts(),
        "tile_dark",
        borst_tile_dark_signal,
        ml.borst_tile_impulse_data_dark,
    )


def resolve_type_indices(type_names, backend: ModelBackend):
    """Map cell-type names to indices in the active vocabulary (Borst or network)."""
    names = [str(n) for n in type_names]
    if backend.network is not None:
        tn = list(backend.network.type_names)
        return [tn.index(n) for n in names if n in tn]
    ctype_arr = backend.ctype
    out = []
    for n in names:
        matches = np.where(ctype_arr == n)[0]
        if len(matches) != 1:
            raise KeyError(f"cell type {n!r} not found uniquely in ctype ({len(matches)} matches)")
        out.append(int(matches[0]))
    return out


def extend_target_pack_mirror_fit(pack, mirror_types, mirror_fit, mirror_sign=-1.0, backend=None):
    """Extend a :class:`TargetPack`: mirror *mirror_fit* targets onto *mirror_types*."""
    if backend is not None and backend.network is not None:
        return _extend_pack_mirror_fit_network(
            pack, mirror_types, mirror_fit, mirror_sign, backend.network,
        )
    return _extend_pack_mirror_fit_borst(pack, mirror_types, mirror_fit, mirror_sign, backend)


def _extend_pack_mirror_fit_borst(pack, mirror_types, mirror_fit, mirror_sign, backend):
    base_u = pack.readout_unit.cpu().numpy()
    mirror_type = ml.fit_type_index(mirror_fit)
    mirror_indices = resolve_type_indices(mirror_types, backend)
    n_cols = backend.n_cols
    n_types = backend.n_types
    extra_units, extra_rows = [], []
    for col in range(n_cols):
        mirror_unit = ml.unit_index(col, mirror_type)
        mirror_row = int(np.where(base_u == mirror_unit)[0][0])
        mirror_target = float(mirror_sign) * pack.data[mirror_row:mirror_row + 1]
        for r in mirror_indices:
            extra_units.append(ml.unit_index(col, int(r)))
            extra_rows.append(mirror_target)
    return _append_mirror_pack_rows(pack, extra_units, extra_rows)


def _extend_pack_mirror_fit_network(pack, mirror_types, mirror_fit, mirror_sign, C):
    from network.tiling import unit_type_names

    names = unit_type_names(C)
    u_arr = pack.readout_unit.cpu().numpy()
    b_arr = pack.readout_batch.cpu().numpy()
    w_arr = pack.cost_weight.cpu().numpy()
    r_arr = (
        pack.cost_radius.cpu().numpy()
        if pack.cost_radius is not None else None
    )
    col_u_all = C.u.detach().cpu().numpy() if hasattr(C.u, "detach") else np.asarray(C.u)
    col_v_all = C.v.detach().cpu().numpy() if hasattr(C.v, "detach") else np.asarray(C.v)
    extra_b, extra_u, extra_rows, extra_w, extra_r = [], [], [], [], []
    for row_i in range(len(u_arr)):
        u = int(u_arr[row_i])
        if str(names[u]) != mirror_fit:
            continue
        b = int(b_arr[row_i])
        col_u, col_v = int(col_u_all[u]), int(col_v_all[u])
        mirror_target = float(mirror_sign) * pack.data[row_i:row_i + 1]
        w = float(w_arr[row_i])
        r = float(r_arr[row_i]) if r_arr is not None else None
        for mtype in mirror_types:
            candidates = np.where(
                (col_u_all == col_u)
                & (col_v_all == col_v)
                & (names == str(mtype))
            )[0]
            for uidx in candidates:
                extra_b.append(b)
                extra_u.append(int(uidx))
                extra_rows.append(mirror_target)
                extra_w.append(w)
                if r is not None:
                    extra_r.append(r)
    return _append_mirror_pack_rows(
        pack, extra_u, extra_rows,
        readout_batch=extra_b, cost_weight=extra_w,
        cost_radius=extra_r if extra_r else None,
    )


def _append_mirror_pack_rows(
    pack, extra_units, extra_rows, readout_batch=None, cost_weight=None, cost_radius=None,
):
    extra_units_t = torch.tensor(extra_units, dtype=torch.long, device=active_device())
    extra_data_t = torch.cat(extra_rows, dim=0)
    n_all = int(pack.readout_unit.shape[0]) + len(extra_units)
    n_extra = len(extra_units)
    if readout_batch is None:
        readout_batch = torch.zeros(n_extra, dtype=torch.long, device=active_device())
    else:
        readout_batch = torch.tensor(readout_batch, dtype=torch.long, device=active_device())
    if cost_weight is None:
        cost_weight = torch.ones(n_all, dtype=torch.float64, device=active_device())
    else:
        base_w = pack.cost_weight
        cost_weight = torch.cat([
            base_w,
            torch.tensor(cost_weight, dtype=torch.float64, device=active_device()),
        ])
    cost_radius_out = pack.cost_radius
    if cost_radius is not None:
        base_r = pack.cost_radius
        extra_r_t = torch.tensor(cost_radius, dtype=torch.float64, device=active_device())
        cost_radius_out = (
            torch.cat([base_r, extra_r_t])
            if base_r is not None else extra_r_t
        )
    all_data = torch.cat([pack.data, extra_data_t], dim=0)
    return TargetPack(
        name=pack.name,
        signal=pack.signal,
        data=all_data,
        power=pack.power + torch.sum(extra_data_t ** 2),
        cost_weight=cost_weight,
        readout_batch=torch.cat([pack.readout_batch, readout_batch]),
        readout_unit=torch.cat([pack.readout_unit, extra_units_t]),
        cost_t0=pack.cost_t0,
        cost_radius=cost_radius_out,
        center_column=pack.center_column,
    )


def _mirror_types_from_spec(spec):
    if "mirror_types" not in spec:
        raise ValueError(f"mirror_fit spec needs mirror_types: {spec!r}")
    return [str(t) for t in spec["mirror_types"]]


def _apply_mirror_fit_spec(pack, spec, backend: ModelBackend):
    return extend_target_pack_mirror_fit(
        pack,
        mirror_types=_mirror_types_from_spec(spec),
        mirror_fit=spec["mirror_fit"],
        mirror_sign=float(spec.get("mirror_sign", -1.0)),
        backend=backend,
    )


def apply_pack_override(pack, override, backend: ModelBackend):
    """Apply one serializable pack override dict (saved in ``train_opts.json``)."""
    if "mirror_fits" in override:
        for spec in override["mirror_fits"]:
            pack = _apply_mirror_fit_spec(pack, spec, backend)
        return pack
    if "mirror_fit" in override:
        return _apply_mirror_fit_spec(pack, override["mirror_fit"], backend)
    raise ValueError(f"unknown pack override {override!r}")


def _borst_moving_bar_pack(T, name, *, center_column=False):
    return TargetPack(
        name=name,
        signal=T.signal,
        data=T.data,
        power=T.power,
        cost_weight=T.cost_weight,
        readout_batch=T.readout_batch,
        readout_unit=T.readout_unit,
        cost_t0=T.cost_t0,
        center_column=bool(center_column),
    )


def _load_borst_matrices(dev: Optional[str] = None):
    # Circuits/ relative paths are intentional — see coding-conventions §10 exception.
    dev = dev or active_device()
    multi_colM = np.load('Circuits/multi_colM.npy')
    ctype_arr = np.load('Circuits/ctype.npy')
    multi_colM = ml.apply_borst_connectivity_patches(multi_colM)
    M_exc = exc_synweight * multi_colM * (multi_colM > 0)
    M_inh = inh_synweight * multi_colM * (multi_colM < 0) * (-1)
    M_exc = torch.tensor(M_exc, dtype=torch.float64, device=dev)
    M_inh = torch.tensor(M_inh, dtype=torch.float64, device=dev)
    M_signed = torch.tensor(exc_synweight * multi_colM, dtype=torch.float64, device=dev)
    return M_exc, M_inh, M_signed, ctype_arr


def borst_backend(
    dev: Optional[str] = None,
    *,
    depol_cells=None,
    ih_reverse_cells=None,
) -> ModelBackend:
    """Default 5-column Borst dense connectivity backend."""
    dev = dev or active_device()
    depol = tuple(ml.leak_depol_indices() if depol_cells is None else depol_cells)
    ih_rev = tuple(ih_reverse_cells if ih_reverse_cells is not None else IH_DIR_REVERSE_CELLS)
    M_exc, M_inh, M_signed, ctype_arr = _load_borst_matrices(dev)
    node_type = (torch.arange(BORST_NOFCELLS * BORST_NOFCOLS, device=dev) % BORST_NOFCELLS).long()
    conn = DenseConn(M_exc, M_inh, M_signed, node_type)
    return ModelBackend(
        conn=conn,
        e_leak=build_e_leak(conn, BORST_NOFCELLS, depol_cells=depol),
        ih_dir=build_ih_dir(conn, ih_reverse_cells=ih_rev),
        n_types=BORST_NOFCELLS,
        n_cols=BORST_NOFCOLS,
        network=None,
        ctype=ctype_arr,
        depol_cells=depol,
        ih_reverse_cells=ih_rev,
    )


def _network_backend_from_connectome(C) -> ModelBackend:
    """Build a :class:`ModelBackend` from an already-loaded connectome graph."""
    tn = list(C.type_names)
    depol = tuple(tn.index(t) for t in ml.LEAK_DEPOL_TYPES if t in tn)
    conn = C.conn
    return ModelBackend(
        conn=conn,
        e_leak=build_e_leak(conn, C.n_types, depol_cells=depol),
        ih_dir=build_ih_dir(conn),
        n_types=C.n_types,
        n_cols=1,
        network=C,
        ctype=None,
        depol_cells=depol,
    )


def load_network_backend(network_json, dev: Optional[str] = None) -> ModelBackend:
    """Load connectome network into a :class:`ModelBackend`."""
    from network.construction import load_network

    dev = dev or active_device()
    C = load_network(network_json, device=dev,
                     exc_synweight=exc_synweight, inh_synweight=inh_synweight)
    backend = _network_backend_from_connectome(C)
    print(f"network: {network_json}")
    print(f"  n_units={backend.n_units}, n_types={backend.n_types}, "
          f"nparams={schema_nparams(default_schema('conductance', backend))}")
    return backend


@dataclass
class _TrainBindCtx:
    """Per-target builder context during :func:`open_session`."""

    model_backend: ModelBackend
    dev: str
    tile_bright_stimulus_opts: Optional[dict] = None
    tile_dark_stimulus_opts: Optional[dict] = None
    moving_bar_bright_stimulus_opts: Optional[dict] = None
    moving_bar_dark_stimulus_opts: Optional[dict] = None


def _build_borst_tile_bright_target(ctx: _TrainBindCtx) -> Tuple[TargetPack, dict]:
    opts = dict(ctx.tile_bright_stimulus_opts or make_tile_bright_stimulus_opts())
    return build_borst_tile_bright_pack(opts), opts


def _build_borst_tile_dark_target(ctx: _TrainBindCtx) -> Tuple[TargetPack, dict]:
    opts = dict(ctx.tile_dark_stimulus_opts or make_tile_dark_stimulus_opts())
    return build_borst_tile_dark_pack(opts), opts


def _build_borst_moving_bar_bright_target(ctx: _TrainBindCtx) -> Tuple[TargetPack, dict]:
    from network.moving_bar_target import build_borst_moving_bar_target

    opts = dict(ctx.moving_bar_bright_stimulus_opts or make_moving_bar_bright_stimulus_opts())
    center_column = bool(opts.get("center_column", False))
    T = build_borst_moving_bar_target(
        device=ctx.dev or active_device(),
        t_on=t_on,
        deltat_ms=deltat,
        center_column=center_column,
        i_baseline=opts["i_baseline"],
        i_bright_bar=opts["i_bright_bar"],
        contrasts=("bright",),
    )
    stim = _enrich_moving_bar_stimulus_opts(opts, T.info, center_column=center_column)
    return _borst_moving_bar_pack(T, "moving_bar_bright", center_column=center_column), stim


def _build_borst_moving_bar_dark_target(ctx: _TrainBindCtx) -> Tuple[TargetPack, dict]:
    from network.moving_bar_target import build_borst_moving_bar_target

    opts = dict(ctx.moving_bar_dark_stimulus_opts or make_moving_bar_dark_stimulus_opts())
    center_column = bool(opts.get("center_column", False))
    T = build_borst_moving_bar_target(
        device=ctx.dev or active_device(),
        t_on=t_on,
        deltat_ms=deltat,
        center_column=center_column,
        i_baseline=opts["i_baseline"],
        i_dark_bar=opts["i_dark_bar"],
        contrasts=("dark",),
    )
    stim = _enrich_moving_bar_stimulus_opts(opts, T.info, center_column=center_column)
    return _borst_moving_bar_pack(T, "moving_bar_dark", center_column=center_column), stim


def _validate_tile_stimulus_opts(opts, target_name):
    if bool(opts.get("center_column", False)) and bool(opts.get("multi_shift", False)):
        raise ValueError(
            f"{target_name}: center_column and multi_shift are incompatible",
        )


def _build_network_tile_bright_target(
    ctx: _TrainBindCtx, C,
) -> Tuple[TargetPack, dict, str]:
    from network.target import build_shifted_target

    dev = ctx.dev or active_device()
    opts = dict(ctx.tile_bright_stimulus_opts or make_tile_bright_stimulus_opts(mode="network"))
    _validate_tile_stimulus_opts(opts, "tile_bright")
    center_column = bool(opts.get("center_column", False))
    multi_shift = bool(opts.get("multi_shift", False))
    share_edges = bool(opts.get("share_edges", False))
    T = build_shifted_target(
        C,
        share_edges=share_edges,
        single_shift=not multi_shift,
        device=dev,
        maxtime=ml.IMPULSE_MAXTIME,
        t_on=t_on,
        center_column=center_column,
        i_baseline=opts["i_baseline"],
        i_bright=opts["i_bright"],
        polarity="bright",
    )
    stim = dict(opts)
    pack = TargetPack(
        name="tile_bright",
        signal=T.signal,
        data=T.data,
        power=T.power,
        cost_weight=T.cost_weight,
        readout_batch=T.readout_batch,
        readout_unit=T.readout_unit,
        cost_t0=None,
        cost_radius=T.cost_radius,
        center_column=center_column,
    )
    coltag = "centre column" if center_column else f"{T.info['n_cost']} cost cells"
    shifttag = "7 shifts" if multi_shift else "1 shift"
    tag = (
        f"tile_bright (B={T.n_batch} stimuli [{T.info['n_centers']} tiles x "
        f"{shifttag}], {coltag})"
    )
    return pack, stim, tag


def _build_network_tile_dark_target(
    ctx: _TrainBindCtx, C,
) -> Tuple[TargetPack, dict, str]:
    from network.target import build_shifted_target

    dev = ctx.dev or active_device()
    opts = dict(ctx.tile_dark_stimulus_opts or make_tile_dark_stimulus_opts(mode="network"))
    _validate_tile_stimulus_opts(opts, "tile_dark")
    center_column = bool(opts.get("center_column", False))
    multi_shift = bool(opts.get("multi_shift", False))
    share_edges = bool(opts.get("share_edges", False))
    T = build_shifted_target(
        C,
        share_edges=share_edges,
        single_shift=not multi_shift,
        device=dev,
        maxtime=ml.IMPULSE_MAXTIME,
        t_on=t_on,
        center_column=center_column,
        i_baseline=opts["i_baseline"],
        i_dark=opts["i_dark"],
        polarity="dark",
    )
    stim = dict(opts)
    pack = TargetPack(
        name="tile_dark",
        signal=T.signal,
        data=T.data,
        power=T.power,
        cost_weight=T.cost_weight,
        readout_batch=T.readout_batch,
        readout_unit=T.readout_unit,
        cost_t0=None,
        cost_radius=T.cost_radius,
        center_column=center_column,
    )
    coltag = "centre column" if center_column else f"{T.info['n_cost']} cost cells"
    shifttag = "7 shifts" if multi_shift else "1 shift"
    tag = (
        f"tile_dark (B={T.n_batch} stimuli [{T.info['n_centers']} tiles x "
        f"{shifttag}], {coltag})"
    )
    return pack, stim, tag


def _build_network_moving_bar_bright_target(
    ctx: _TrainBindCtx, C,
) -> Tuple[TargetPack, dict, str]:
    from network.moving_bar_target import build_moving_bar_target

    dev = ctx.dev or active_device()
    opts = dict(ctx.moving_bar_bright_stimulus_opts or make_moving_bar_bright_stimulus_opts(mode="network"))
    center_column = bool(opts.get("center_column", False))
    T = build_moving_bar_target(
        C,
        device=dev,
        t_on=t_on,
        center_column=center_column,
        i_baseline=opts["i_baseline"],
        i_bright_bar=opts["i_bright_bar"],
        contrasts=("bright",),
    )
    stim = _enrich_moving_bar_stimulus_opts(opts, T.info, center_column=center_column)
    pack = TargetPack(
        name="moving_bar_bright",
        signal=T.signal,
        data=T.data,
        power=T.power,
        cost_weight=T.cost_weight,
        readout_batch=T.readout_batch,
        readout_unit=T.readout_unit,
        cost_t0=T.cost_t0,
        center_column=center_column,
    )
    coltag = (
        "centre column" if center_column
        else f"{T.info['n_cost_columns']} photo columns"
    )
    tag = f"moving-bar bright (B={T.n_batch} stimuli, {T.info['n_cost']} cost cells, {coltag})"
    return pack, stim, tag


def _build_network_moving_bar_dark_target(
    ctx: _TrainBindCtx, C,
) -> Tuple[TargetPack, dict, str]:
    from network.moving_bar_target import build_moving_bar_target

    dev = ctx.dev or active_device()
    opts = dict(ctx.moving_bar_dark_stimulus_opts or make_moving_bar_dark_stimulus_opts(mode="network"))
    center_column = bool(opts.get("center_column", False))
    T = build_moving_bar_target(
        C,
        device=dev,
        t_on=t_on,
        center_column=center_column,
        i_baseline=opts["i_baseline"],
        i_dark_bar=opts["i_dark_bar"],
        contrasts=("dark",),
    )
    stim = _enrich_moving_bar_stimulus_opts(opts, T.info, center_column=center_column)
    pack = TargetPack(
        name="moving_bar_dark",
        signal=T.signal,
        data=T.data,
        power=T.power,
        cost_weight=T.cost_weight,
        readout_batch=T.readout_batch,
        readout_unit=T.readout_unit,
        cost_t0=T.cost_t0,
        center_column=center_column,
    )
    coltag = (
        "centre column" if center_column
        else f"{T.info['n_cost_columns']} photo columns"
    )
    tag = f"moving-bar dark (B={T.n_batch} stimuli, {T.info['n_cost']} cost cells, {coltag})"
    return pack, stim, tag


BORST_TARGET_BUILDERS = {
    "tile_bright": _build_borst_tile_bright_target,
    "tile_dark": _build_borst_tile_dark_target,
    "moving_bar_bright": _build_borst_moving_bar_bright_target,
    "moving_bar_dark": _build_borst_moving_bar_dark_target,
}

NETWORK_TARGET_BUILDERS = {
    "tile_bright": _build_network_tile_bright_target,
    "tile_dark": _build_network_tile_dark_target,
    "moving_bar_bright": _build_network_moving_bar_bright_target,
    "moving_bar_dark": _build_network_moving_bar_dark_target,
}


def expand_target_aliases(names) -> List[str]:
    """Expand CLI shorthands: ``tile`` → bright+dark, ``moving_bar`` → bright+dark."""
    out = []
    for name in names:
        if name in TARGET_ALIASES:
            out.extend(TARGET_ALIASES[name])
        else:
            out.append(name)
    return out


def apply_center_only_to_stimulus_opts(opts, target_name, center_only_targets):
    """Set ``center_column`` on one target's stimulus opts from a per-target list."""
    out = dict(opts or {})
    out["center_column"] = target_name in set(center_only_targets or [])
    return out


def apply_multi_shift_to_stimulus_opts(opts, target_name, multi_shift_targets):
    """Set ``multi_shift`` on tile stimulus opts from a per-target list."""
    if target_name not in TILE_TARGETS:
        return opts
    out = dict(opts or {})
    out["multi_shift"] = target_name in set(multi_shift_targets or [])
    return out


def apply_share_edges_to_stimulus_opts(opts, target_name, share_edges_targets):
    """Set ``share_edges`` on tile stimulus opts from a per-target list."""
    if target_name not in TILE_TARGETS:
        return opts
    out = dict(opts or {})
    out["share_edges"] = target_name in set(share_edges_targets or [])
    return out


def _i_cli_target_names(cli_field, name):
    """Resolve CLI target token for one ``--i_*`` flag."""
    if name not in CLI_TARGET_NAMES:
        raise ValueError(
            f"unknown target {name!r} in --{cli_field} "
            f"(expected {'|'.join(CLI_TARGET_NAMES)})",
        )
    if cli_field == "i_baseline":
        if name in TARGET_ALIASES:
            return TARGET_ALIASES[name]
        return [name]
    if cli_field == "i_bright":
        if name not in I_CLI_BRIGHT_TARGETS:
            raise ValueError(
                f"--i_bright does not accept target {name!r} "
                f"(expected tile|tile_bright|moving_bar|moving_bar_bright)",
            )
        return list(I_CLI_BRIGHT_TARGETS[name])
    if name not in I_CLI_DARK_TARGETS:
        raise ValueError(
            f"--i_dark does not accept target {name!r} "
            f"(expected tile|tile_dark|moving_bar|moving_bar_dark)",
        )
    return list(I_CLI_DARK_TARGETS[name])


def build_i_cli_by_target(kv_by_field):
    """Merge per-flag comma KV dicts into ``{'by_target': {target: {field: val}}}``."""
    by_target = {}
    for cli_field, kv in kv_by_field.items():
        if not kv:
            continue
        for name, val in kv.items():
            for t in _i_cli_target_names(cli_field, name):
                sidecar_field = I_CLI_SIDECAR_FIELD[(cli_field, t)]
                by_target.setdefault(t, {})[sidecar_field] = float(val)
    return {"by_target": by_target} if by_target else None


def apply_i_cli_to_stimulus_opts(opts, target_name, i_cli):
    """Merge per-target CLI ``--i_*`` overrides into stimulus opts."""
    if not i_cli:
        return opts
    overrides = (i_cli.get("by_target") or {}).get(target_name)
    if not overrides:
        return opts
    out = dict(opts or {})
    allowed = TARGET_I_FIELDS[target_name]
    for key, val in overrides.items():
        if key not in allowed:
            raise ValueError(f"{key!r} not valid for target {target_name!r}")
        out[key] = float(val)
    return out


def _finalize_stimulus_opts(opts, target_name, *, center_only_targets, multi_shift_targets,
                            share_edges_targets, i_cli):
    if target_name in TILE_TARGETS:
        mode = (opts or {}).get("mode", "borst")
        if target_name == "tile_bright":
            out = make_tile_bright_stimulus_opts(mode=mode, **{
                k: v for k, v in (opts or {}).items()
                if k in ("i_baseline", "i_bright", "center_column", "multi_shift", "share_edges")
            })
        else:
            out = make_tile_dark_stimulus_opts(mode=mode, **{
                k: v for k, v in (opts or {}).items()
                if k in ("i_baseline", "i_dark", "center_column", "multi_shift", "share_edges")
            })
    elif target_name == "moving_bar_bright":
        out = make_moving_bar_bright_stimulus_opts(
            mode=(opts or {}).get("mode", "borst"),
            **{
                k: v for k, v in (opts or {}).items()
                if k in ("i_baseline", "i_bright_bar", "center_column", "mode")
            },
        )
    elif target_name == "moving_bar_dark":
        out = make_moving_bar_dark_stimulus_opts(
            mode=(opts or {}).get("mode", "borst"),
            **{
                k: v for k, v in (opts or {}).items()
                if k in ("i_baseline", "i_dark_bar", "center_column", "mode")
            },
        )
    else:
        out = dict(opts or {})
    out = apply_center_only_to_stimulus_opts(out, target_name, center_only_targets)
    out = apply_multi_shift_to_stimulus_opts(out, target_name, multi_shift_targets)
    out = apply_share_edges_to_stimulus_opts(out, target_name, share_edges_targets)
    out = apply_i_cli_to_stimulus_opts(out, target_name, i_cli)
    if target_name in TILE_TARGETS:
        _validate_tile_stimulus_opts(out, target_name)
    return out


def expand_loss_weights(weights: Optional[dict]) -> Dict[str, float]:
    """Expand alias keys in ``loss_weights`` to concrete target names."""
    if not weights:
        return {}
    out = {}
    for name, val in weights.items():
        if name in TARGET_ALIASES:
            for t in TARGET_ALIASES[name]:
                out[t] = float(val)
        else:
            out[str(name)] = float(val)
    return out


def _normalize_target_list(target_list) -> List[str]:
    if target_list is None:
        raise ValueError("target_list required")
    if isinstance(target_list, str):
        target_list = [t.strip() for t in target_list.split(",") if t.strip()]
    tl = expand_target_aliases(list(target_list))
    if not tl:
        raise ValueError("target_list must not be empty")
    bad = [t for t in tl if t not in VALID_TARGETS]
    if bad:
        raise ValueError(
            f"unknown target(s) {bad!r} (expected {'|'.join(CLI_TARGET_NAMES)})",
        )
    return tl


def make_train_opts(
    backend="borst",
    target_list=None,
    loss_weights=None,
    pack_overrides=None,
    sequential=None,
    center_only_targets=None,
    multi_shift_targets=None,
    share_edges_targets=None,
    i_cli=None,
    moving_bar_bright_stimulus_opts=None,
    moving_bar_dark_stimulus_opts=None,
    tile_bright_stimulus_opts=None,
    tile_dark_stimulus_opts=None,
    network_json=None,
    network=None,
    per_type=False,
    dev=None,
    packs=None,
    ih_off=IH_OFF_DEFAULT,
):
    """Canonical training opts for :func:`open_session` (Borst or network)."""
    tl = _normalize_target_list(target_list)
    mode = "network" if backend == "network" else "borst"
    bright_opts = _finalize_stimulus_opts(
        tile_bright_stimulus_opts if "tile_bright" in tl else None,
        "tile_bright",
        center_only_targets=center_only_targets,
        multi_shift_targets=multi_shift_targets,
        share_edges_targets=share_edges_targets,
        i_cli=i_cli,
    ) if ("tile_bright" in tl or tile_bright_stimulus_opts is not None) else None
    if bright_opts is not None and "tile_bright" in tl:
        bright_opts["mode"] = mode
    dark_opts = _finalize_stimulus_opts(
        tile_dark_stimulus_opts if "tile_dark" in tl else None,
        "tile_dark",
        center_only_targets=center_only_targets,
        multi_shift_targets=multi_shift_targets,
        share_edges_targets=share_edges_targets,
        i_cli=i_cli,
    ) if ("tile_dark" in tl or tile_dark_stimulus_opts is not None) else None
    if dark_opts is not None and "tile_dark" in tl:
        dark_opts["mode"] = mode
    bright_bar_opts = _finalize_stimulus_opts(
        moving_bar_bright_stimulus_opts if "moving_bar_bright" in tl else None,
        "moving_bar_bright",
        center_only_targets=center_only_targets,
        multi_shift_targets=multi_shift_targets,
        share_edges_targets=share_edges_targets,
        i_cli=i_cli,
    ) if ("moving_bar_bright" in tl or moving_bar_bright_stimulus_opts is not None) else None
    if bright_bar_opts is not None and "moving_bar_bright" in tl:
        bright_bar_opts["mode"] = mode
    dark_bar_opts = _finalize_stimulus_opts(
        moving_bar_dark_stimulus_opts if "moving_bar_dark" in tl else None,
        "moving_bar_dark",
        center_only_targets=center_only_targets,
        multi_shift_targets=multi_shift_targets,
        share_edges_targets=share_edges_targets,
        i_cli=i_cli,
    ) if ("moving_bar_dark" in tl or moving_bar_dark_stimulus_opts is not None) else None
    if dark_bar_opts is not None and "moving_bar_dark" in tl:
        dark_bar_opts["mode"] = mode
    opts = {
        "backend": str(backend),
        "target_list": tl,
        "loss_weights": loss_weights or {},
        "sequential": sequential,
        "moving_bar_bright_stimulus_opts": bright_bar_opts,
        "moving_bar_dark_stimulus_opts": dark_bar_opts,
        "tile_bright_stimulus_opts": bright_opts,
        "tile_dark_stimulus_opts": dark_opts,
    }
    if pack_overrides is not None:
        opts["pack_overrides"] = pack_overrides
    if packs is not None:
        opts["packs"] = packs
    if per_type:
        opts["per_type"] = True
    opts["ih_off"] = str(ih_off)
    if backend == "network":
        opts.update({
            "network": network,
            "network_json": str(network_json) if network_json is not None else None,
            "dev": dev,
        })
    return opts



def _train_opts_for_sidecar(
    opts, backend, target_list,
    resolved_tile_bright, resolved_tile_dark,
    resolved_bar_bright, resolved_bar_dark, sequential_bool,
) -> dict:
    record = {
        "backend": str(backend),
        "target_list": list(target_list),
        "loss_weights": {str(k): float(v) for k, v in (opts.get("loss_weights") or {}).items()},
        "sequential": bool(sequential_bool),
    }
    if backend == "network":
        record.update({
            "network_json": str(opts["network_json"]),
            "tile_bright_stimulus_opts": (
                resolved_tile_bright if resolved_tile_bright is not None
                else opts.get("tile_bright_stimulus_opts")
            ),
            "tile_dark_stimulus_opts": (
                resolved_tile_dark if resolved_tile_dark is not None
                else opts.get("tile_dark_stimulus_opts")
            ),
            "moving_bar_bright_stimulus_opts": (
                resolved_bar_bright if resolved_bar_bright is not None
                else opts.get("moving_bar_bright_stimulus_opts")
            ),
            "moving_bar_dark_stimulus_opts": (
                resolved_bar_dark if resolved_bar_dark is not None
                else opts.get("moving_bar_dark_stimulus_opts")
            ),
        })
    else:
        record["moving_bar_bright_stimulus_opts"] = (
            resolved_bar_bright if resolved_bar_bright is not None
            else opts.get("moving_bar_bright_stimulus_opts")
        )
        record["moving_bar_dark_stimulus_opts"] = (
            resolved_bar_dark if resolved_bar_dark is not None
            else opts.get("moving_bar_dark_stimulus_opts")
        )
        record["tile_bright_stimulus_opts"] = (
            resolved_tile_bright if resolved_tile_bright is not None
            else opts.get("tile_bright_stimulus_opts")
        )
        record["tile_dark_stimulus_opts"] = (
            resolved_tile_dark if resolved_tile_dark is not None
            else opts.get("tile_dark_stimulus_opts")
        )
    overrides = opts.get("pack_overrides")
    if overrides:
        record["pack_overrides"] = overrides
    if opts.get("per_type"):
        record["per_type"] = True
    if "ih_off" in opts:
        record["ih_off"] = str(opts["ih_off"])
    return record


def _make_session(
    model_backend: ModelBackend,
    model_type: str,
    target_list: List[str],
    packs: Dict[str, TargetPack],
    *,
    loss_weights=None,
    sequential=None,
    dev=None,
    train_opts_record=None,
    schema: Optional[list] = None,
) -> TrainSession:
    dev_ref = dev or active_device()
    seq = (dev_ref == "cpu") if sequential is None else bool(sequential)
    if train_opts_record is not None:
        train_opts_record["sequential"] = bool(seq)
    ih_off = IH_OFF_DEFAULT
    if train_opts_record is not None and "ih_off" in train_opts_record:
        ih_off = str(train_opts_record["ih_off"])
    if model_type == 'conductance':
        sch = conductance_schema(model_backend, schema, ih_off)
    elif schema is not None:
        sch = list(schema)
    else:
        sch = default_schema(model_type, model_backend)
    return TrainSession(
        backend=model_backend,
        model_type=model_type,
        schema=tuple(sch),
        targets=dict(packs),
        target_list=tuple(target_list),
        loss_weights={str(k): float(v) for k, v in (loss_weights or {}).items()},
        sequential=bool(seq),
        device=dev_ref,
        train_opts=train_opts_record,
    )


def open_session(
    opts: dict,
    model_type: str,
    *,
    schema: Optional[list] = None,
    model_backend: Optional[ModelBackend] = None,
) -> TrainSession:
    """Build a :class:`TrainSession` from canonical training opts."""
    backend_name = str(opts.get("backend", "borst"))
    target_list = _normalize_target_list(opts.get("target_list"))
    bad = [t for t in target_list if t not in VALID_TARGETS]
    if bad:
        raise ValueError(f"unknown target(s) {bad!r} (expected {'|'.join(CLI_TARGET_NAMES)})")
    dev = opts.get("dev") or active_device()

    if backend_name == "borst":
        model_backend = model_backend or borst_backend(dev)
        ctx = _TrainBindCtx(
            model_backend=model_backend,
            dev=dev,
            tile_bright_stimulus_opts=opts.get("tile_bright_stimulus_opts"),
            tile_dark_stimulus_opts=opts.get("tile_dark_stimulus_opts"),
            moving_bar_bright_stimulus_opts=opts.get("moving_bar_bright_stimulus_opts"),
            moving_bar_dark_stimulus_opts=opts.get("moving_bar_dark_stimulus_opts"),
        )
        prebuilt = opts.get("packs")
        pack_overrides = opts.get("pack_overrides") or {}
        if pack_overrides:
            prebuilt = None
        if prebuilt is not None:
            packs = dict(prebuilt)
            resolved_tile_bright = opts.get("tile_bright_stimulus_opts")
            resolved_tile_dark = opts.get("tile_dark_stimulus_opts")
            resolved_bar_bright = opts.get("moving_bar_bright_stimulus_opts")
            resolved_bar_dark = opts.get("moving_bar_dark_stimulus_opts")
        else:
            packs = {}
            resolved_tile_bright = resolved_tile_dark = None
            resolved_bar_bright = resolved_bar_dark = None
            for tname in target_list:
                pack, stim = BORST_TARGET_BUILDERS[tname](ctx)
                if tname in pack_overrides:
                    pack = apply_pack_override(pack, pack_overrides[tname], model_backend)
                packs[tname] = pack
                if tname == "tile_bright":
                    resolved_tile_bright = stim
                elif tname == "tile_dark":
                    resolved_tile_dark = stim
                elif tname == "moving_bar_bright":
                    resolved_bar_bright = stim
                elif tname == "moving_bar_dark":
                    resolved_bar_dark = stim
        record = _train_opts_for_sidecar(
            opts, "borst", target_list,
            resolved_tile_bright, resolved_tile_dark,
            resolved_bar_bright, resolved_bar_dark, False,
        )
        session = _make_session(
            model_backend, model_type, target_list, packs,
            loss_weights=opts.get("loss_weights"),
            sequential=opts.get("sequential"),
            dev=dev,
            train_opts_record=record,
            schema=schema,
        )
        print(
            f"Borst targets: {'+'.join(target_list)}  "
            f"(packs={list(packs.keys())}, sequential={session.sequential})",
        )
        return session

    if backend_name != "network":
        raise ValueError(f"unknown backend {backend_name!r} (expected borst|network)")

    C = opts.get("network")
    if C is None:
        raise ValueError("open_session(network) requires opts['network']")
    if model_backend is None:
        model_backend = _network_backend_from_connectome(C)
    elif model_backend.network is not C:
        raise ValueError("model_backend.network must be opts['network']")
    ctx = _TrainBindCtx(
        model_backend=model_backend,
        dev=dev,
        tile_bright_stimulus_opts=opts.get("tile_bright_stimulus_opts"),
        tile_dark_stimulus_opts=opts.get("tile_dark_stimulus_opts"),
        moving_bar_bright_stimulus_opts=opts.get("moving_bar_bright_stimulus_opts"),
        moving_bar_dark_stimulus_opts=opts.get("moving_bar_dark_stimulus_opts"),
    )
    packs = {}
    pack_overrides = opts.get("pack_overrides") or {}
    resolved_tile_bright = resolved_tile_dark = None
    resolved_bar_bright = resolved_bar_dark = None
    for tname in target_list:
        pack, stim, _tag = NETWORK_TARGET_BUILDERS[tname](ctx, C)
        if tname in pack_overrides:
            pack = apply_pack_override(pack, pack_overrides[tname], model_backend)
        packs[tname] = pack
        if tname == "tile_bright":
            resolved_tile_bright = stim
        elif tname == "tile_dark":
            resolved_tile_dark = stim
        elif tname == "moving_bar_bright":
            resolved_bar_bright = stim
        elif tname == "moving_bar_dark":
            resolved_bar_dark = stim
    record = _train_opts_for_sidecar(
        opts, "network", target_list,
        resolved_tile_bright, resolved_tile_dark,
        resolved_bar_bright, resolved_bar_dark, False,
    )
    return _make_session(
        model_backend, model_type, target_list, packs,
        loss_weights=opts.get("loss_weights"),
        sequential=opts.get("sequential"),
        dev=dev,
        train_opts_record=record,
        schema=schema,
    )


def open_session_from_opts(opts: dict, model_type: str, **kwargs) -> TrainSession:
    """Restore a session from a saved ``train_opts.json`` dict."""
    opts = dict(opts)
    opts["packs"] = None
    backend = str(opts.get("backend", "borst"))
    if backend == "network":
        nj = opts.get("network_json")
        if not nj:
            raise ValueError("train_opts with backend=network requires network_json")
        if not opts.get("target_list"):
            raise ValueError("train_opts requires target_list")
        mb = load_network_backend(
            nj, dev=opts.get("dev") or active_device(),
        )
        opts["network"] = mb.network
        kwargs.setdefault("model_backend", mb)
    return open_session({**opts, "backend": backend}, model_type, **kwargs)


def open_session_from_outdir(
    outdir: str,
    model_type: str,
    *,
    param_modes=None,
    param_fixes=None,
    per_type: bool = False,
) -> TrainSession:
    """Load ``train_opts.json`` from a run folder and return a ready session."""
    import json
    opts_path = os.path.join(os.path.abspath(outdir), TRAIN_OPTS_FILE)
    if not os.path.isfile(opts_path):
        raise FileNotFoundError(f"missing {opts_path}")
    with open(opts_path) as f:
        opts = json.load(f)
    session = open_session_from_opts(opts, model_type)
    if per_type or opts.get("per_type"):
        session = session.with_schema(
            expand_schema_per_type(list(session.schema), session.backend.n_types)
        )
    if param_modes or param_fixes:
        session = session.with_schema(
            apply_modes(list(session.schema), param_modes, param_fixes)
        )
    return session


# ------- network calculations  -----------------------------------------------

def rectsyn(x,thrld):
    
    result=x-thrld
    result=result*(result>0)
    
    return result

def update_Vm(Vm, u_on, u_off, inp_gain, out_gain, Ih_gmax, Ih_gmax_off,
              Ih_midv, Ih_slope, tau_midv, Ih_midv_off, Ih_slope_off, tau_midv_off,
              signal, backend: ModelBackend):

    # ON Ih (hyperpolarization-activated, E_Ih=+50) + OFF Ih (depolarization-activated,
    # E_IH_OFF=-150).
    e_leak = backend.e_leak
    conn = backend.conn
    slope_on = Ih_slope
    slope_off = -Ih_slope_off
    Ih_ss_on  = 1.0/(1.0+torch.exp((Ih_midv-Vm)*slope_on))
    Ih_ss_off = 1.0/(1.0+torch.exp((Ih_midv_off-Vm)*slope_off))
    tau_on  = 1.5/(torch.exp(-0.1*(Vm-tau_midv))+torch.exp(+0.1*(Vm-tau_midv)))*1000.0 + 100.0
    tau_off = 1.5/(torch.exp(-0.1*(Vm-tau_midv_off))+torch.exp(+0.1*(Vm-tau_midv_off)))*1000.0 + 100.0
    u_on    = deltat/tau_on*(Ih_ss_on-u_on)+u_on
    u_off   = deltat/tau_off*(Ih_ss_off-u_off)+u_off
    g_Ih_on  = u_on * Ih_gmax * Ih_gain
    g_Ih_off = u_off * Ih_gmax_off * Ih_gain
    g_Ih     = g_Ih_on + g_Ih_off

    g_exc, g_inh = conn.exc_inh_drive(rectsyn(Vm,trld)*out_gain)
    g_exc   = g_exc*inp_gain
    g_inh   = g_inh*inp_gain

    Vm = (g_exc*E_exc + g_inh*E_inh + g_leak*e_leak
          + E_Ih * g_Ih_on + E_IH_OFF * g_Ih_off + cdt*Vm + signal)
    Vm = Vm / (g_exc + g_inh + g_Ih + g_leak + cdt)

    return Vm, u_on, u_off

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


def _expand_segment(seg, raw, backend: ModelBackend):
    """Map a length-`count` per-unit vector to a usable parameter, per its 'kind'."""
    kind = seg['kind']
    dev = backend.conn.node_type.device
    n_types = backend.n_types
    if kind == 'full':
        return calc_multi_col_params(raw, backend.conn).to(dev)
    if kind == 'lamina':
        cell = torch.full((n_types,), float(seg['fill']), dtype=raw.dtype, device=raw.device)
        for i, target in enumerate(lamina_cells(seg)):
            cell[target] = raw[i]
        return calc_multi_col_params(cell, backend.conn).to(dev)
    if kind == 'scalar':
        return raw[0]
    if kind == 'output':
        return raw.to(dev)
    raise ValueError(f"unknown segment kind: {kind}")


def assign_params(z, schema, backend: ModelBackend):
    """Unpack z into a dict of parameter tensors, driven by the given schema + modes."""
    p = {}
    for seg, start, stop in schema_segments(schema):
        p[seg['name']] = _expand_segment(seg, _reconstruct_raw(seg, z[start:stop], z), backend)
    return p


def assign_params_adaptive(z, schema, backend: ModelBackend):
    """Adaptive params plus the fixed (non-trainable) contrast-gate pivot."""
    p = assign_params(z, schema, backend)
    p['gate_pivot'] = GATE_PIVOT
    return p

def update_state_adaptive(activity, v_sustained, v_transient, drive_lp, p, x_t, x_t_delayed, backend: ModelBackend):
    
    # passive point neuron: tau_m * da/dt = -a + X, with X = bias + syn + x_t.
    # an adaptive low-pass reference (drive_lp) gates a transient component so
    # that activity = v_sustained + v_transient.
    
    bias  = p['bias']
    tau   = torch.clamp(p['tau_m'], min=deltat)
    tau_r = torch.clamp(p['tau_adapt'], min=deltat)
    ratio = tau / tau_r
    
    # presynaptic output gain (per source), postsynaptic input gain (per target)
    syn     = p['inp_gain'] * backend.conn.signed_drive(torch.relu(activity) * p['out_gain'])
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

def model_cost(model, data, session: TrainSession, scale=1.0, power=None):
    # normalised MSE over the response window (t=t_on..maxtime-1); ``scale`` is an
    # arbitrary linear gain on ``model`` (diagnostics only — training uses schema out_scale).
    if power is None:
        power = session.primary_pack.power
    mt = session.maxtime
    return torch.sum((scale * model - data[t_on:mt])**2) / power * 100.0


def _run_conductance(session: TrainSession, p, neuron_index=None, return_ref=False, sig=None, pack=None):
    backend = session.backend
    mt = session.maxtime
    ih_off = (session.train_opts or {}).get('ih_off', IH_OFF_DEFAULT)
    if neuron_index is None:
        pack = pack or session.primary_pack
        neuron_index = pack.readout_unit
    if sig is None:
        sig = session.pack_signal(pack)
    inp_gain, out_gain = p['inp_gain'], p['out_gain']
    Ih_gmax = p['Ih_gmax']
    Ih_gmax_off, Ih_midv_off, Ih_slope_off, tau_midv_off = conductance_ih_off_kwargs(p, ih_off)
    Ih_midv, Ih_slope, tau_midv = p['Ih_midv'], p['Ih_slope'], p['tau_midv']
    u_on = u_off = torch.zeros(backend.n_units, dtype=torch.float64, device=backend.conn.node_type.device)
    Vm = backend.e_leak
    for t in range(1, t_on):
        Vm, u_on, u_off = update_Vm(
            Vm, u_on, u_off, inp_gain, out_gain, Ih_gmax, Ih_gmax_off,
            Ih_midv, Ih_slope, tau_midv, Ih_midv_off, Ih_slope_off, tau_midv_off,
            sig[t - 1], backend)
    Vm_ref = 1.0 * Vm[neuron_index]
    model = 0
    rows = []
    for t in range(t_on, mt):
        Vm, u_on, u_off = update_Vm(
            Vm, u_on, u_off, inp_gain, out_gain, Ih_gmax, Ih_gmax_off,
            Ih_midv, Ih_slope, tau_midv, Ih_midv_off, Ih_slope_off, tau_midv_off,
            sig[t - 1], backend)
        model = deltat / Ca_tau * (Vm[neuron_index] - Vm_ref - model) + model
        rows.append(model)
    out = torch.stack(rows)
    if return_ref:
        return out, Vm_ref
    return out


def _run_conductance_full(session: TrainSession, p, sig):
    backend = session.backend
    ih_off = (session.train_opts or {}).get('ih_off', IH_OFF_DEFAULT)
    inp_gain, out_gain = p['inp_gain'], p['out_gain']
    Ih_gmax = p['Ih_gmax']
    Ih_gmax_off, Ih_midv_off, Ih_slope_off, tau_midv_off = conductance_ih_off_kwargs(p, ih_off)
    Ih_midv, Ih_slope, tau_midv = p['Ih_midv'], p['Ih_slope'], p['tau_midv']
    B = sig.shape[0]
    t_end = sig.shape[1]
    dev = backend.conn.node_type.device
    u_on = u_off = torch.zeros((B, backend.n_units), dtype=torch.float64, device=dev)
    Vm = backend.e_leak.expand(B, backend.n_units).clone()
    for t in range(1, min(t_on, t_end)):
        Vm, u_on, u_off = update_Vm(
            Vm, u_on, u_off, inp_gain, out_gain, Ih_gmax, Ih_gmax_off,
            Ih_midv, Ih_slope, tau_midv, Ih_midv_off, Ih_slope_off, tau_midv_off,
            sig[:, t - 1], backend)
    Vm_ref = Vm.clone()
    model = 0
    rows = []
    for t in range(t_on, t_end):
        Vm, u_on, u_off = update_Vm(
            Vm, u_on, u_off, inp_gain, out_gain, Ih_gmax, Ih_gmax_off,
            Ih_midv, Ih_slope, tau_midv, Ih_midv_off, Ih_slope_off, tau_midv_off,
            sig[:, t - 1], backend)
        model = deltat / Ca_tau * (Vm - Vm_ref - model) + model
        rows.append(model)
    return torch.stack(rows, dim=1)


def _window_time_traces(model_full, b_idx, u_idx, t0, win=None):
    """Extract per-readout windows from ``model_full`` (B, T', N).

    ``t0`` is the absolute simulation step of window start. Steps before ``t_on``
    are zero (``model_full`` only exists from ``t_on`` onward).
    """
    if win is None:
        raise ValueError("window length win required")
    win = int(win)
    t_rel = t0[:, None] - t_on + torch.arange(win, dtype=torch.long, device=active_device())
    t_max = model_full.shape[1] - 1
    pre = t_rel < 0
    t_safe = t_rel.clamp(0, t_max)
    sel = model_full[b_idx[:, None], t_safe, u_idx[:, None]]
    return torch.where(pre, torch.zeros_like(sel), sel)


def _readout_model_traces_pack(model_full, pack: TargetPack):
    """Select model traces for cost cells; windowed when ``pack.cost_t0`` is set."""
    if pack.cost_t0 is None:
        return model_full[pack.readout_batch, :, pack.readout_unit]
    return _window_time_traces(
        model_full, pack.readout_batch, pack.readout_unit, pack.cost_t0,
        win=pack.data.shape[1],
    )


def _pack_out_scale(p, pack: TargetPack, backend: ModelBackend):
    """Per-cost-row output scale from schema ``out_scale`` (single source of truth)."""
    os_param = p.get('out_scale', 1.0)
    if torch.is_tensor(os_param) and os_param.dim() > 0:
        u = pack.readout_unit
        if backend.network is not None:
            ci = backend.network.node_type[u]
        else:
            ci = u % backend.n_types
        os_param = os_param[ci]
    n = int(pack.readout_unit.shape[0])
    dev = backend.conn.node_type.device
    if torch.is_tensor(os_param):
        if os_param.dim() == 0:
            os_param = os_param.expand(n)
        return os_param
    return torch.full((n,), float(os_param), dtype=torch.float64, device=dev)


def _run_adaptive(p, session: TrainSession, neuron_index=None, return_ref=False, sig=None, pack=None):
    backend = session.backend
    mt = session.maxtime
    if 'gate_pivot' not in p:
        p = {**p, 'gate_pivot': GATE_PIVOT}
    pack = pack or session.primary_pack
    if neuron_index is None:
        neuron_index = pack.readout_unit
    if sig is None:
        sig = session.pack_signal(pack)
    bias = p['bias']
    x_signal = sig / _pack_signal_scale(pack, session)

    activity    = bias.clone()
    v_sustained = bias.clone()
    v_transient = torch.zeros_like(bias)
    drive_lp    = bias.clone()

    act_ref = None
    model = 0
    rows = []
    for t in range(1, mt):
        x_t = x_signal[t - 1]
        x_d = x_signal[max(t - 1 - gate_lag, 0)]
        activity, v_sustained, v_transient, drive_lp = update_state_adaptive(
            activity, v_sustained, v_transient, drive_lp, p, x_t, x_d, backend)
        if t == t_on - 1:
            act_ref = 1.0 * activity[neuron_index]
        elif t >= t_on:
            model = deltat / Ca_tau * (activity[neuron_index] - act_ref - model) + model
            rows.append(model)
    model = torch.stack(rows)
    if return_ref:
        return model, act_ref
    return model


def _conductance_pack_readout(p, pack: TargetPack, session: TrainSession, batch_idx=None):
    """Conductance forward + readout. ``batch_idx``: one stimulus (sequential) or all (batched)."""
    sig = pack.signal if batch_idx is None else pack.signal[batch_idx:batch_idx + 1]
    model_full = _run_conductance_full(session, p, sig)
    if batch_idx is None:
        return _readout_model_traces_pack(model_full, pack)
    mask = pack.readout_batch == int(batch_idx)
    u_m = pack.readout_unit[mask]
    if pack.cost_t0 is None:
        return model_full[0, :, u_m].transpose(0, 1)
    b_zero = torch.zeros_like(u_m)
    return _window_time_traces(
        model_full, b_zero, u_m, pack.cost_t0[mask],
        win=pack.data.shape[1],
    )


def _window_adaptive_traces(model, t0, win):
    """Windowed readout from ``_run_adaptive`` output ``(T', K)``."""
    t_rel = t0[:, None] - t_on + torch.arange(win, dtype=torch.long, device=active_device())
    t_max = model.shape[0] - 1
    pre = t_rel < 0
    t_safe = t_rel.clamp(0, t_max)
    k_idx = torch.arange(model.shape[1], dtype=torch.long, device=active_device())
    sel = model[t_safe, k_idx[:, None]]
    return torch.where(pre, torch.zeros_like(sel), sel)


def _adaptive_pack_readout(p, pack: TargetPack, session: TrainSession, batch_idx=None):
    """Adaptive forward + readout. ``batch_idx``: one stimulus (sequential) or all (batched)."""
    p = {**p, 'gate_pivot': GATE_PIVOT}
    if batch_idx is None:
        sig = session.pack_signal(pack)
        if sig.dim() == 3:
            sig = sig[0]
        u = pack.readout_unit
        t0 = pack.cost_t0
    else:
        sig = pack.signal[batch_idx]
        mask = pack.readout_batch == int(batch_idx)
        u = pack.readout_unit[mask]
        t0 = pack.cost_t0[mask] if pack.cost_t0 is not None else None
    model = _run_adaptive(p, session, neuron_index=u, sig=sig, pack=pack)
    if t0 is None:
        return model.transpose(0, 1)
    return _window_adaptive_traces(model, t0, win=pack.data.shape[1])


# Register new model types here only — batching (``batch_idx``) stays in ``_pack_cost``.
MODEL_PACK_READOUT = {
    'conductance': _conductance_pack_readout,
    'adaptive': _adaptive_pack_readout,
}


def _pack_model_readout(p, pack: TargetPack, session: TrainSession, batch_idx=None):
    try:
        readout = MODEL_PACK_READOUT[session.model_type]
    except KeyError:
        raise ValueError(f"no pack readout for model_type={session.model_type!r}") from None
    return readout(p, pack, session, batch_idx)


def _pack_cost_rows(p, pack: TargetPack, session: TrainSession, batch_idx=None):
    """Forward + MSE for one pack. ``batch_idx=None`` batched; ``int`` one stimulus."""
    scale = _pack_out_scale(p, pack, session.backend)
    if batch_idx is not None:
        mask = pack.readout_batch == int(batch_idx)
        if not bool(mask.any()):
            return None
        scale = scale[mask]
        data = pack.data[mask]
        weight = pack.cost_weight[mask]
    else:
        data = pack.data
        weight = pack.cost_weight
    sel = _pack_model_readout(p, pack, session, batch_idx)
    diff = scale[:, None] * sel - data
    return torch.sum(weight[:, None] * diff ** 2) / pack.power * 100.0


def _pack_cost_part(z, pack: TargetPack, session: TrainSession, batch_idx=None):
    schema = list(session.schema)
    if session.model_type == 'adaptive':
        p = assign_params_adaptive(z, schema, session.backend)
    else:
        p = assign_params(z, schema, session.backend)
    part = _pack_cost_rows(p, pack, session, batch_idx)
    if part is None:
        return torch.zeros((), dtype=torch.float64, device=session.device)
    return part


def _pack_cost(z, pack: TargetPack, session: TrainSession, batch_idx=None):
    return _pack_cost_part(z, pack, session, batch_idx)


def calc_cost_parts(z, session: TrainSession) -> Dict[str, torch.Tensor]:
    """Per-target unweighted cost (each pack's contribution before ``loss_weights``)."""
    parts = {}
    for name, pack in session.targets.items():
        w = float(session.loss_weights.get(name, 1.0))
        if w == 0.0:
            continue
        if session.sequential:
            part = torch.zeros((), dtype=torch.float64, device=session.device)
            for b in range(pack.signal.shape[0]):
                part = part + _pack_cost_part(z, pack, session, batch_idx=b)
        else:
            part = _pack_cost_part(z, pack, session, batch_idx=None)
        parts[name] = part
    return parts


def calc_cost(z, session: TrainSession):
    total = torch.zeros((), dtype=torch.float64, device=session.device)
    for name, part in calc_cost_parts(z, session).items():
        w = float(session.loss_weights.get(name, 1.0))
        total = total + w * part
    return total

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
        if seg_mode(seg) == 'individual':
            for j in seg.get('zero', []):      # lamina-local indices (see IH_GMAX_ZERO_TYPES)
                z[start + j] = 0.0
    return torch.tensor(z, dtype=torch.float64).to(active_device())

def guess_initial_params(session: TrainSession):
    return schema_guess(list(session.schema))


def _float_cost_parts(parts_fn, z, target_order=None):
    if parts_fn is None:
        return None
    raw = parts_fn(z)
    out = {k: float(v.item() if torch.is_tensor(v) else v) for k, v in raw.items()}
    if target_order:
        return {k: out[k] for k in target_order if k in out}
    return out


def _fmt_cost_parts(parts):
    if not parts:
        return ""
    return "  [" + "  ".join(f"{k}={v:.4f}" for k, v in parts.items()) + "]"


def gradient_network(z, lr=0.0001, cost_fn=None, n_steps=100, device="cpu", z_bounds=None,
                     cost_log=None, step_log=None, parts_fn=None, target_order=None):
    
    a = time.time()

    z = nn.Parameter(z.clone().to(device))
    
    optimizer = torch.optim.Adam([z], lr=lr)

    cost = cost_fn(z).item()
    best_cost = cost
    best_z = z.clone().detach()
    
    initial_cost = 1.0*cost
    initial_parts = _float_cost_parts(parts_fn, z, target_order)

    progress_bar = tqdm(range(n_steps), desc=f'Cost: {cost:.4f}')

    for i in progress_bar:
        
        optimizer.zero_grad()
        
        cost = cost_fn(z)  
        
        if cost.item() < best_cost:
            
            best_cost = cost.item()
            best_z = z.clone().detach()
        
        if cost_log is not None:
            cost_log.append(cost.item())
        if step_log is not None:
            step_log(z)
        
        cost.backward()
        optimizer.step()

        with torch.no_grad():
            
            z.clamp_(z_bounds[:, 0].to(device), z_bounds[:, 1].to(device))

        progress_bar.set_description(f'Cost: {cost.item():.4f}')

    cost = cost_fn(z)  
    
    if cost.item() < best_cost:
        
        best_cost = cost.item()
        best_z = z.clone().detach()

    print()
    print('Initl cost =', format(initial_cost,'.4f') + _fmt_cost_parts(initial_parts))
    print('Final cost =', format(cost.item(),'.4f') + _fmt_cost_parts(
        _float_cost_parts(parts_fn, z, target_order),
    ))
    print('Best  cost =', format(best_cost,'.4f') + _fmt_cost_parts(
        _float_cost_parts(parts_fn, best_z, target_order),
    ))
    
    b = time.time()
    
    print('time needed  =',format(b-a,'.2f'),' sec')
    print()

    return best_z

def train_staged(z, cost_fn, z_bounds, lrs, nsteps, cost_log=None, step_log=None,
                 parts_fn=None, target_order=None):
    # run gradient_network once per learning-rate stage, chaining the best params.
    for lr in lrs:
        z = gradient_network(z, lr=lr, n_steps=nsteps, device=active_device(),
                             cost_fn=cost_fn, z_bounds=z_bounds, cost_log=cost_log,
                             step_log=step_log, parts_fn=parts_fn, target_order=target_order)
    return z


def _make_step_logger(session: TrainSession):
    """Build ``(cost_fn, target_history, log_step)`` for aligned per-step logging."""
    target_history = {name: [] for name in session.target_list}

    def cost_fn(z):
        return calc_cost(z, session)

    def log_step(z):
        total = float(calc_cost(z, session).item())
        for name, part in calc_cost_parts(z, session).items():
            target_history[name].append(float(part.item()))
        return total

    return cost_fn, target_history, log_step


def do_many_runs(session: TrainSession, nofruns, nofsteps, lrs=(0.1, 0.01, 0.001)) -> TrainingResult:
    """Run ``nofruns`` independent fits; return arrays (no file I/O)."""
    schema = list(session.schema)
    n_params = schema_nparams(schema)
    bounds = schema_bounds(schema)

    all_params = np.zeros((nofruns, n_params))
    final_costs = np.zeros(nofruns)
    final_costs_by_target = {name: np.zeros(nofruns) for name in session.target_list}
    best_i = 0
    best_cost = np.inf
    cost_curve = np.array([], dtype=np.float64)
    cost_curves_by_target = {}

    for i in range(nofruns):
        print()
        print('round', i)
        print()

        z = schema_guess(schema)
        cost_history = []
        cost_fn, target_history, log_step = _make_step_logger(session)

        def step_log(z):
            cost_history.append(log_step(z))

        parts_fn = lambda z: calc_cost_parts(z, session)
        z_fit = train_staged(
            z, cost_fn, bounds, lrs, nofsteps,
            step_log=step_log,
            parts_fn=parts_fn,
            target_order=list(session.target_list),
        )

        all_params[i] = z_fit.detach().cpu().numpy()
        final_costs[i] = calc_cost(z_fit, session).item()
        for name, part in calc_cost_parts(z_fit, session).items():
            final_costs_by_target[name][i] = float(part.item())
        if final_costs[i] < best_cost:
            best_cost = final_costs[i]
            best_i = i
            cost_curve = np.array(cost_history, dtype=np.float64)
            cost_curves_by_target = {
                name: np.array(curve, dtype=np.float64)
                for name, curve in target_history.items()
            }

    return TrainingResult(
        all_params=all_params,
        final_costs=final_costs,
        best_i=best_i,
        cost_curve=cost_curve,
        cost_curves_by_target=cost_curves_by_target,
        final_costs_by_target=final_costs_by_target,
    )
