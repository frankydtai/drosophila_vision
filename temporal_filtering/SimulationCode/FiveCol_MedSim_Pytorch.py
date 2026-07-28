# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Created on Wed Jul 26 09:53:25 2023

@author: aborst
"""
import os
import sys
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import Medulla_Library as ml
import time

import torch
from torch import nn
from tqdm import tqdm

import network_bootstrap  # noqa: F401 — connectome_io on sys.path
from connectome_io import parse_comma_list
from training_config import (
    DELTAT_MS,
    IMPULSE_MAXTIME,
    SIM_DTYPE_DEFAULT,
    T_ON,
    sim_dtype_from_fp32,
)
from param_defaults import DEFAULT_IH_GMAX_INDI_NAMES, P as PARAM_DEFAULTS

import neuron_model  # noqa: F401 — package side-effects / discovery
from neuron_model import (  # noqa: F401 — re-export for callers using fc.*
    ALL_PARAM_NAMES,
    IH_DIR_REVERSE_CELLS,
    IH_OFF_DEFAULT,
    IH_OFF_GMAX_SEGMENT,
    IH_OFF_MODES,
    IH_OFF_SCALAR_SEGMENTS,
    IH_SHAPE_PARAM_NAMES,
    KNOWN_MODELS,
    MODEL_PACK_READOUTS,
    STATE_CLAMP,
    Ca_tau,
    E_IH_OFF,
    E_Ih,
    E_LEAK_DEPOL,
    E_LEAK_REST,
    E_exc,
    E_inh,
    Ih_gain,
    apply_ih_off_mode,
    build_conductance_schema,
    build_hp_lp_schema,
    capac,
    cdt,
    conductance_ih_off_kwargs,
    conductance_schema,
    default_schema,
    deltat,
    exc_synweight,
    g_leak,
    inh_synweight,
    params_from_z,
    rectsyn,
    run_conductance,
    run_conductance_full,
    run_hp_lp,
    update_state_hp_lp,
    update_Vm,
    vm_budget_from_g,
)

# Underscore aliases kept for existing call sites (plot/, analysis/).
_run_conductance = run_conductance
_run_conductance_full = run_conductance_full
_run_hp_lp = run_hp_lp


def active_device():
    """Pick CUDA or CPU from current runtime (not frozen at import)."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def __getattr__(name):
    if name == 'device':
        return active_device()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

t_on = T_ON
TRAIN_OPTS_FILE = "train_opts.json"
SPOT_TARGETS = ("spot_bright", "spot_dark")
SPOT_POLARITIES = frozenset({"bright", "dark"})
MOVING_BAR_POLARITIES = frozenset({"bright", "dark"})
_SPOT_STEP_KEY = {"bright": "i_bright", "dark": "i_dark"}
MOVING_BAR_TARGETS = ("moving_bar_bright", "moving_bar_dark")
VALID_TARGETS = SPOT_TARGETS + MOVING_BAR_TARGETS
PD_ND_LABELS = ("PD", "ND")
PD_IDX, ND_IDX = 0, 1
MOVING_BAR_COST_PARTS = tuple(
    f"{t}_{lab}" for t in MOVING_BAR_TARGETS for lab in (*PD_ND_LABELS, "DSI")
)
TARGET_ALIASES = {
    "spot": SPOT_TARGETS,
    "moving_bar": MOVING_BAR_TARGETS,
}
CLI_TARGET_NAMES = VALID_TARGETS + tuple(TARGET_ALIASES.keys())
TARGET_I_FIELDS = {
    "spot_bright": frozenset({"i_baseline", "i_bright"}),
    "spot_dark": frozenset({"i_baseline", "i_dark"}),
    "moving_bar_bright": frozenset({"i_baseline", "i_bright_bar"}),
    "moving_bar_dark": frozenset({"i_baseline", "i_dark_bar"}),
}
I_CLI_BRIGHT_TARGETS = {
    "spot": ("spot_bright",),
    "spot_bright": ("spot_bright",),
    "moving_bar": ("moving_bar_bright",),
    "moving_bar_bright": ("moving_bar_bright",),
}
I_CLI_DARK_TARGETS = {
    "spot": ("spot_dark",),
    "spot_dark": ("spot_dark",),
    "moving_bar": ("moving_bar_dark",),
    "moving_bar_dark": ("moving_bar_dark",),
}
I_CLI_SIDECAR_FIELD = {
    ("i_baseline", "spot_bright"): "i_baseline",
    ("i_baseline", "spot_dark"): "i_baseline",
    ("i_baseline", "moving_bar_bright"): "i_baseline",
    ("i_baseline", "moving_bar_dark"): "i_baseline",
    ("i_bright", "spot_bright"): "i_bright",
    ("i_bright", "moving_bar_bright"): "i_bright_bar",
    ("i_dark", "spot_dark"): "i_dark",
    ("i_dark", "moving_bar_dark"): "i_dark_bar",
}

COST_WEIGHT_ALIASES = {
    "spot": SPOT_TARGETS,
    "moving_bar": MOVING_BAR_COST_PARTS,
    "moving_bar_bright": (
        "moving_bar_bright_PD", "moving_bar_bright_ND", "moving_bar_bright_DSI",
    ),
    "moving_bar_dark": (
        "moving_bar_dark_PD", "moving_bar_dark_ND", "moving_bar_dark_DSI",
    ),
    "PD": ("moving_bar_bright_PD", "moving_bar_dark_PD"),
    "ND": ("moving_bar_bright_ND", "moving_bar_dark_ND"),
    "DSI": ("moving_bar_bright_DSI", "moving_bar_dark_DSI"),
}

# Neuron models (conductance / hp_lp): see ``neuron_model/``.
# Schema partition machinery below; numeric defaults in ``param_defaults.P``.

def calc_multi_col_params(param, conn):
    # Broadcast a per-cell-TYPE parameter (n_types,) to the full state (n_units,)
    # via the backend's node_type.
    return param.index_select(0, conn.node_type)


def build_e_leak(conn, n_types, depol_cells=(), *, dtype=SIM_DTYPE_DEFAULT):
    """(conn.n_units,) resting potential; ``depol_cells`` are type indices at E_LEAK_DEPOL."""
    per_type = torch.full((n_types,), E_LEAK_REST, dtype=dtype, device=conn.node_type.device)
    for c in depol_cells:
        per_type[int(c)] = E_LEAK_DEPOL
    return calc_multi_col_params(per_type, conn)


def build_ih_dir(conn, ih_reverse_cells=IH_DIR_REVERSE_CELLS, *, dtype=SIM_DTYPE_DEFAULT):
    """(conn.n_units,) Ih direction (+1 normal, -1 mirrored per cell-type)."""
    d = torch.ones(conn.n_units, dtype=dtype, device=conn.node_type.device)
    for c in ih_reverse_cells:
        d[conn.node_type == int(c)] = -1.0
    return d

# --- parameter schema partitions --------------------------------------------
# Numeric lo/hi/init/jit(/fixed_val): ``param_defaults.P``.
# Model segment lists: ``neuron_model.schema``.
# Each segment:
#   name, kind, count, lo/hi/init/jit[, fixed_val]
#   indi / shared / fixed / frozen : disjoint exhaustive lists of unit indices
#       frozen: not in z; values from seg['carry'] (resume) or init (cold start)
# z layout per segment: len(indi) slots + (1 if shared else 0).
# CLI: per-param ``indi=... shared=... fixed=... frozen=...`` (space-separated); ``all`` = remainder.
# ``--all-param`` batches all segments; ``--ih-shape`` batches the six Ih shape params.
# Named ``best_param.npz``.
PARTITION_BUCKETS = ('indi', 'shared', 'fixed', 'frozen')
PAIR_SEP = ':'


def seg_count(seg):
    """Full per-unit width (n_types or n_pairs)."""
    return int(seg['count'])


def seg_ntrain(seg):
    """Trainable z width: one slot per indi unit + one if shared nonempty."""
    n = len(seg.get('indi', ()))
    if seg.get('shared'):
        n += 1
    return n


def schema_segments(schema):
    """Yield (segment, start, stop) slice ranges into z."""
    start = 0
    for seg in schema:
        stop = start + seg_ntrain(seg)
        yield seg, start, stop
        start = stop


def schema_nparams(schema):
    return sum(seg_ntrain(seg) for seg in schema)


def _fixed_const(seg):
    """Constant for units in the fixed partition (fixed_val if set, else init)."""
    if 'fixed_val' in seg:
        return float(seg['fixed_val'])
    return float(seg['init'])


def parse_partition_text(text):
    """Parse ``indi=A,B fixed=all`` (space-separated buckets) → token lists."""
    text = (text or '').strip()
    buckets = {b: [] for b in PARTITION_BUCKETS}
    if not text:
        return buckets
    parts = []
    buf = []
    for tok in text.split():
        if '=' in tok and tok.split('=', 1)[0] in PARTITION_BUCKETS and buf:
            parts.append(' '.join(buf))
            buf = [tok]
        else:
            buf.append(tok)
    if buf:
        parts.append(' '.join(buf))
    for part in parts:
        if '=' not in part:
            raise ValueError(f"partition chunk {part!r} needs bucket=list")
        key, rest = part.split('=', 1)
        key = key.strip()
        if key not in PARTITION_BUCKETS:
            raise ValueError(f"unknown partition bucket {key!r}")
        items = [x.strip() for x in rest.split(',') if x.strip()]
        buckets[key] = items
    return buckets


def resolve_partition_tokens(buckets, unit_names, *, param_name='param'):
    """Resolve token lists (with at most one ``all``) to index lists covering unit_names."""
    unit_names = [str(u) for u in unit_names]
    name_to_i = {n: i for i, n in enumerate(unit_names)}
    all_idx = set(range(len(unit_names)))
    explicit = {b: [] for b in PARTITION_BUCKETS}
    all_bucket = None
    for b in PARTITION_BUCKETS:
        toks = list(buckets.get(b) or [])
        if 'all' in toks:
            if len(toks) != 1:
                raise ValueError(f"{param_name}: 'all' cannot mix with other names in {b}")
            if all_bucket is not None:
                raise ValueError(f"{param_name}: 'all' in both {all_bucket} and {b}")
            all_bucket = b
            continue
        for t in toks:
            if t not in name_to_i:
                raise ValueError(f"{param_name}: unknown unit {t!r}")
            explicit[b].append(name_to_i[t])
    claimed = []
    for b in PARTITION_BUCKETS:
        claimed.extend(explicit[b])
    if len(claimed) != len(set(claimed)):
        raise ValueError(f"{param_name}: overlapping units across indi/shared/fixed/frozen")
    claimed_set = set(claimed)
    leftover = sorted(all_idx - claimed_set)
    if all_bucket is not None:
        explicit[all_bucket] = leftover
        claimed_set |= set(leftover)
    elif claimed_set != all_idx:
        missing = [unit_names[i] for i in sorted(all_idx - claimed_set)]
        raise ValueError(
            f"{param_name}: units not assigned (use all= in one bucket): {missing[:8]}"
            + ("..." if len(missing) > 8 else "")
        )
    return {b: list(explicit[b]) for b in PARTITION_BUCKETS}


def partition_to_names(part, unit_names):
    """Index partition → name lists for train_opts sidecar."""
    return {b: [str(unit_names[i]) for i in part[b]] for b in PARTITION_BUCKETS}


def apply_partitions(schema, partitions_by_name, unit_names_for_seg):
    """Copy schema with resolved partitions.

    *partitions_by_name*: ``{seg_name: {indi/shared/fixed: [names|indices]}}``
    *unit_names_for_seg*: ``callable(seg) -> list[str]`` or ``{seg_name: list[str]}``.
    """
    out = []
    for seg in schema:
        s = dict(seg)
        name = s['name']
        if name not in partitions_by_name:
            out.append(s)
            continue
        if callable(unit_names_for_seg):
            units = unit_names_for_seg(s)
        else:
            units = unit_names_for_seg[name]
        raw = partitions_by_name[name]
        vals = []
        for b in PARTITION_BUCKETS:
            vals.extend(raw.get(b) or [])
        if vals and all(isinstance(x, int) for x in vals):
            part = {b: list(raw.get(b) or []) for b in PARTITION_BUCKETS}
        else:
            buckets = {b: [str(x) for x in (raw.get(b) or [])] for b in PARTITION_BUCKETS}
            part = resolve_partition_tokens(buckets, units, param_name=name)
        for b in PARTITION_BUCKETS:
            s[b] = part[b]
        out.append(s)
    return out


def schema_partitions_record(schema, unit_names_for_seg):
    """Serialize partitions as name lists for train_opts.json."""
    rec = {}
    for seg in schema:
        if callable(unit_names_for_seg):
            units = unit_names_for_seg(seg)
        else:
            units = unit_names_for_seg[seg['name']]
        rec[seg['name']] = partition_to_names(
            {b: list(seg.get(b) or []) for b in PARTITION_BUCKETS}, units,
        )
    return rec


def type_unit_names(backend: "ModelBackend"):
    if backend.network is None:
        raise ValueError("type_unit_names requires backend.network")
    return [str(n) for n in backend.network.type_names]


def pair_unit_names(backend: "ModelBackend"):
    keys = backend.conn.pair_keys
    names = type_unit_names(backend)
    return [f"{names[s]}{PAIR_SEP}{names[t]}" for s, t in keys]


def unit_names_for_segment(seg, backend: "ModelBackend"):
    if seg['kind'] == 'edge_pair':
        return pair_unit_names(backend)
    return type_unit_names(backend)


def _part_indi_all(n):
    return {'indi': list(range(n)), 'shared': [], 'fixed': [], 'frozen': []}


def _part_shared_all(n):
    return {'indi': [], 'shared': list(range(n)), 'fixed': [], 'frozen': []}


def _part_fixed_all(n):
    return {'indi': [], 'shared': [], 'fixed': list(range(n)), 'frozen': []}


def _part_indi_subset_fixed_rest(n, indi_idx):
    indi_set = set(indi_idx)
    return {
        'indi': list(indi_idx),
        'shared': [],
        'fixed': [i for i in range(n) if i not in indi_set],
        'frozen': [],
    }


def attach_param_carry(schema, named=None):
    """Return schema copy with per-seg ``carry`` arrays (full width) for frozen units."""
    import numpy as np

    out = []
    for seg in schema:
        s = dict(seg)
        frozen = list(seg.get('frozen') or [])
        if not frozen:
            s.pop('carry', None)
            out.append(s)
            continue
        count = seg_count(seg)
        if named is not None and seg['name'] in named:
            carry = np.asarray(named[seg['name']], dtype=np.float64).reshape(-1).copy()
        else:
            carry = np.full(count, float(seg['init']), dtype=np.float64)
        if carry.shape[0] != count:
            raise ValueError(
                f"{seg['name']}: carry length {carry.shape[0]} != count {count}"
            )
        s['carry'] = carry
        out.append(s)
    return out


def _with_part(seg, part):
    s = dict(seg)
    for b in PARTITION_BUCKETS:
        s[b] = list(part[b])
    return s


def z_to_unit_values(z, schema):
    """Full-width per-unit arrays (before column expand) for each segment."""
    import numpy as np
    out = {}
    for seg, start, stop in schema_segments(schema):
        raw = _reconstruct_raw(seg, z[start:stop], z)
        out[seg['name']] = raw.detach().cpu().numpy().astype(np.float64)
    return out


def unit_values_to_z(named, schema, *, dtype=None, device=None):
    """Pack full-width named unit values into trainable z for *schema* partitions."""
    import numpy as np
    n = schema_nparams(schema)
    z = torch.zeros(n, dtype=dtype or SIM_DTYPE_DEFAULT, device=device or active_device())
    for seg, start, stop in schema_segments(schema):
        raw = np.asarray(named[seg['name']], dtype=np.float64).reshape(-1)
        if raw.shape[0] != seg_count(seg):
            raise ValueError(
                f"{seg['name']}: named length {raw.shape[0]} != count {seg_count(seg)}"
            )
        slots = []
        for u in seg.get('indi', ()):
            slots.append(float(raw[u]))
        if seg.get('shared'):
            vals = [float(raw[u]) for u in seg['shared']]
            slots.append(float(np.mean(vals)) if vals else float(seg['init']))
        if slots:
            z[start:stop] = torch.tensor(slots, dtype=z.dtype, device=z.device)
    return z


def remap_named_unit_values(named, src_type_names, src_pair_names, schema, backend):
    """Remap named arrays from a prior run onto *backend* unit order for *schema*."""
    import numpy as np
    src_type_names = [str(n) for n in src_type_names]
    src_pair_names = [str(n) for n in (src_pair_names or [])]
    dst_types = type_unit_names(backend)
    dst_pairs = pair_unit_names(backend) if any(s['kind'] == 'edge_pair' for s in schema) else []
    src_t = {n: i for i, n in enumerate(src_type_names)}
    src_p = {n: i for i, n in enumerate(src_pair_names)}
    out = {}
    for seg in schema:
        name = seg['name']
        count = seg_count(seg)
        fixed_val = _fixed_const(seg)
        arr = np.full(count, fixed_val, dtype=np.float64)
        src = named.get(name)
        if src is None:
            out[name] = arr
            continue
        src = np.asarray(src, dtype=np.float64).reshape(-1)
        if seg['kind'] == 'edge_pair':
            for j, pn in enumerate(dst_pairs):
                if pn in src_p:
                    arr[j] = float(src[src_p[pn]])
                else:
                    arr[j] = float(seg['init'])
        else:
            for j, tn in enumerate(dst_types):
                if tn in src_t and src_t[tn] < src.shape[0]:
                    arr[j] = float(src[src_t[tn]])
                else:
                    arr[j] = float(seg['init']) if 'fixed_val' not in seg else fixed_val
        out[name] = arr
    return out


@dataclass(frozen=True)
class TargetPack:
    """One training target: stimulus + readout indices + target traces.

    Spot ``signal`` / ``data`` time dims: :mod:`training_config`. Moving bar
    uses ``COST_WINDOW`` and per-target ``maxtime``.
    """

    name: str
    signal: torch.Tensor  # (B, T, N)
    data: torch.Tensor  # (n_cost, T')
    power: torch.Tensor  # scalar
    cost_weight: torch.Tensor  # (n_cost,)
    readout_batch: torch.Tensor  # (n_cost,)
    readout_unit: torch.Tensor  # (n_cost,)
    cost_t0: Optional[torch.Tensor] = None  # (n_cost,) absolute step for windowed targets
    cost_radius: Optional[torch.Tensor] = None  # (n_cost,) Euclidean radius for network spot
    readout_stim_u: Optional[torch.Tensor] = None  # (n_cost,) stim anchor u per spot cost row
    readout_stim_v: Optional[torch.Tensor] = None  # (n_cost,) stim anchor v per spot cost row
    cost_extent: Optional[int] = None  # network hex-disc radius for cost readouts
    cost_pd_nd: Optional[torch.Tensor] = None  # (n_cost,) long; 0=PD, 1=ND (moving_bar)
    dsi_pos_rows: Optional[torch.Tensor] = None  # flat cost-row idx (right|up)
    dsi_neg_rows: Optional[torch.Tensor] = None  # flat cost-row idx (left|down)
    dsi_pos_ptr: Optional[torch.Tensor] = None  # (n_dsi+1,) CSR
    dsi_neg_ptr: Optional[torch.Tensor] = None  # (n_dsi+1,) CSR
    dsi_target: Optional[torch.Tensor] = None  # (n_dsi,)
    dsi_weight: Optional[torch.Tensor] = None  # (n_dsi,)
    dsi_power: Optional[torch.Tensor] = None  # scalar


def moving_bar_cost_part_key(target_name: str, part: str) -> str:
    return f"{target_name}_{part}"


def cost_part_keys_for_target(target_name: str) -> Tuple[str, ...]:
    if target_name in MOVING_BAR_TARGETS:
        return tuple(
            moving_bar_cost_part_key(target_name, lab)
            for lab in (*PD_ND_LABELS, "DSI")
        )
    return (target_name,)


def session_cost_part_keys(target_list) -> Tuple[str, ...]:
    keys = []
    for name in target_list:
        keys.extend(cost_part_keys_for_target(name))
    return tuple(keys)


@dataclass(frozen=True)
class ModelBackend:
    """Connectivity + leak/Ih tensors for one simulation graph."""

    conn: object
    e_leak: torch.Tensor
    ih_dir: torch.Tensor
    n_types: int
    n_cols: int
    network: Optional[object] = None
    depol_cells: Tuple[int, ...] = ()
    ih_reverse_cells: Tuple[int, ...] = IH_DIR_REVERSE_CELLS

    @property
    def n_units(self) -> int:
        return self.conn.n_units


@dataclass(frozen=True)
class FusedConductanceForward:
    """Conductance packs with identical signal (T, N); one ``_run_conductance_full`` per group."""

    subpacks: Tuple[TargetPack, ...]
    batch_offsets: Tuple[int, ...]


@dataclass(frozen=True)
class TrainSession:
    """Immutable runtime context for one training / plotting run."""

    backend: ModelBackend
    model: str
    schema: tuple
    targets: Dict[str, TargetPack]
    target_list: Tuple[str, ...]
    cost_weights: Dict[str, float]
    sequential: bool
    device: str
    sim_dtype: torch.dtype = SIM_DTYPE_DEFAULT
    train_opts: Optional[dict] = None
    cost_subpacks: Dict[str, TargetPack] = field(default_factory=dict)
    fused_conductance: Tuple[FusedConductanceForward, ...] = ()

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
        if pack.name in SPOT_TARGETS and sig.dim() == 3 and int(sig.shape[0]) == 1:
            sig = sig.squeeze(0)
        return sig

    def pack_for(self, name: str) -> TargetPack:
        if name not in self.targets:
            raise KeyError(f"target pack {name!r} not in session")
        return self.targets[name]


@dataclass(frozen=True)
class TrainingResult:
    """Output of :func:`do_many_runs` (in memory; persistence is ``train.save_training_outputs``)."""

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


def _spot_i_from_opts(opts, polarity: str):
    """Read spot PR currents (``i_baseline`` / bright or dark step)."""
    if polarity not in SPOT_POLARITIES:
        raise ValueError(f"spot polarity must be 'bright' or 'dark', got {polarity!r}")
    step_key = _SPOT_STEP_KEY[polarity]
    step_default = ml.I_BRIGHT if polarity == "bright" else ml.I_DARK
    return (
        _opt_float(opts, "i_baseline", default=ml.I_BASELINE),
        _opt_float(opts, step_key, default=step_default),
    )


def _pack_signal_scale(pack: TargetPack, session: TrainSession) -> float:
    """Peak PR current for activity-model ``sig / scale`` (from per-target sidecar opts)."""
    opts = ((session.train_opts or {}).get(f"{pack.name}_stimulus_opts")) or {}
    if pack.name == "spot_bright":
        peak = _opt_float(opts, "i_bright", default=ml.I_BRIGHT)
    elif pack.name == "spot_dark":
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


def make_spot_stimulus_opts(
    polarity: str,
    *,
    i_baseline=None,
    i_step=None,
    mode="network",
    shift_extent=None,
    spot_extent=None,
    **extra,
):
    """PR step stimulus opts for ``spot_{polarity}`` (baseline pre-``t_on``, step from ``t_on``)."""
    if polarity not in SPOT_POLARITIES:
        raise ValueError(f"spot polarity must be 'bright' or 'dark', got {polarity!r}")
    from network.spot_target import (
        DEFAULT_FULLY_INSIDE,
        DEFAULT_MULTI_SPOT,
        DEFAULT_SHIFT_EXTENT,
        DEFAULT_SPOT_EXTENT,
    )
    if shift_extent is None:
        shift_extent = extra.get("shift_extent", DEFAULT_SHIFT_EXTENT)

    step_key = _SPOT_STEP_KEY[polarity]
    if i_step is None:
        i_step = extra.get(step_key)
    step_default = ml.I_BRIGHT if polarity == "bright" else ml.I_DARK
    if spot_extent is None:
        spot_extent = extra.get("spot_extent", DEFAULT_SPOT_EXTENT)
    return {
        "mode": mode,
        "i_baseline": float(ml.I_BASELINE if i_baseline is None else i_baseline),
        step_key: float(step_default if i_step is None else i_step),
        "t_on": int(t_on),
        "maxtime": int(IMPULSE_MAXTIME),
        "deltat_ms": float(deltat),
        "shift_extent": int(shift_extent),
        "spot_extent": float(spot_extent),
        "multi_spot": bool(extra.get("multi_spot", DEFAULT_MULTI_SPOT)),
        "fully_inside": bool(extra.get("fully_inside", DEFAULT_FULLY_INSIDE)),
    }


def make_moving_bar_stimulus_opts(
    polarity: str,
    *,
    i_baseline=None,
    i_bar=None,
    multi_bar: bool = True,
    mode="network",
    readout_subtypes=None,
    **extra,
):
    """PR moving-bar stimulus opts for ``moving_bar_{polarity}``."""
    from network.moving_bar_target import resolve_i_baseline

    if polarity not in MOVING_BAR_POLARITIES:
        raise ValueError(f"moving-bar polarity must be 'bright' or 'dark', got {polarity!r}")
    bar_key = "i_bright_bar" if polarity == "bright" else "i_dark_bar"
    if i_bar is None:
        i_bar = extra.get(bar_key)
    bar_default = ml.I_BRIGHT if polarity == "bright" else ml.I_DARK
    out = {
        "mode": mode,
        "i_baseline": resolve_i_baseline(i_baseline),
        bar_key: float(bar_default if i_bar is None else i_bar),
        "t_on": int(t_on),
        "deltat_ms": float(deltat),
        "multi_bar": bool(extra.get("multi_bar", multi_bar)),
    }
    rs = _readout_subtypes_stimulus_list(readout_subtypes)
    if rs is not None:
        out["readout_subtypes"] = rs
    return out


def session_moving_bar_i_baseline(train_opts) -> float:
    """``i_baseline`` from moving-bar stimulus opts on a train session."""
    from network.moving_bar_target import moving_bar_i_baseline_from_opts

    return moving_bar_i_baseline_from_opts(train_opts)


def _readout_subtypes_stimulus_list(readout_subtypes):
    if readout_subtypes is None:
        return None
    return [str(s) for s in readout_subtypes]


def _readout_subtypes_from_opts(opts):
    rs = (opts or {}).get("readout_subtypes")
    if rs is None:
        return None
    return tuple(str(s) for s in rs)


def _enrich_moving_bar_stimulus_opts(opts, info, *, cost_extent):
    """Attach runtime fields from a built moving-bar target; keep canonical ``i_*``."""
    out = dict(opts)
    out["maxtime"] = int(info["maxtime"])
    out["t_on"] = int(info["t_on"])
    out["spec_names"] = list(info["spec_names"])
    if cost_extent is not None:
        out["cost_extent"] = int(cost_extent)
    out["deltat_ms"] = float(deltat)
    if "mode" in info:
        out["mode"] = info["mode"]
    if "present_subtypes" in info:
        out["readout_subtypes"] = list(info["present_subtypes"])
    return out


def resolve_type_indices(type_names, backend: ModelBackend):
    """Map cell-type names to indices in the network vocabulary."""
    if backend.network is None:
        raise ValueError("resolve_type_indices requires backend.network")
    names = [str(n) for n in type_names]
    tn = list(backend.network.type_names)
    return [tn.index(n) for n in names if n in tn]


def extend_target_pack_mirror_fit(pack, mirror_types, mirror_fit, mirror_sign=-1.0, backend=None):
    """Extend a :class:`TargetPack`: mirror *mirror_fit* targets onto *mirror_types*."""
    if backend is None or backend.network is None:
        raise ValueError("extend_target_pack_mirror_fit requires backend.network")
    return _extend_pack_mirror_fit_network(
        pack, mirror_types, mirror_fit, mirror_sign, backend.network,
    )


def _extend_pack_mirror_fit_network(pack, mirror_types, mirror_fit, mirror_sign, C):
    from network.construction import unit_type_names

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
    extra_b, extra_u, extra_rows, extra_w, extra_r, extra_pd_nd = [], [], [], [], [], []
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
                if pack.cost_pd_nd is not None:
                    extra_pd_nd.append(int(pack.cost_pd_nd[row_i].item()))
    return _append_mirror_pack_rows(
        pack, extra_u, extra_rows,
        readout_batch=extra_b, cost_weight=extra_w,
        cost_radius=extra_r if extra_r else None,
        cost_pd_nd=extra_pd_nd if extra_pd_nd else None,
    )


def _append_mirror_pack_rows(
    pack, extra_units, extra_rows, readout_batch=None, cost_weight=None, cost_radius=None,
    cost_pd_nd=None,
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
        w_dtype = pack.cost_weight.dtype
        cost_weight = torch.ones(n_all, dtype=w_dtype, device=active_device())
    else:
        base_w = pack.cost_weight
        w_dtype = base_w.dtype
        cost_weight = torch.cat([
            base_w,
            torch.tensor(cost_weight, dtype=w_dtype, device=active_device()),
        ])
    cost_radius_out = pack.cost_radius
    if cost_radius is not None:
        base_r = pack.cost_radius
        r_dtype = base_r.dtype if base_r is not None else SIM_DTYPE_DEFAULT
        extra_r_t = torch.tensor(cost_radius, dtype=r_dtype, device=active_device())
        cost_radius_out = (
            torch.cat([base_r, extra_r_t])
            if base_r is not None else extra_r_t
        )
    all_data = torch.cat([pack.data, extra_data_t], dim=0)
    cost_pd_nd_out = pack.cost_pd_nd
    if cost_pd_nd is not None:
        extra_pd_t = torch.tensor(cost_pd_nd, dtype=torch.long, device=active_device())
        cost_pd_nd_out = (
            torch.cat([pack.cost_pd_nd, extra_pd_t])
            if pack.cost_pd_nd is not None else extra_pd_t
        )
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
        cost_extent=pack.cost_extent,
        cost_pd_nd=cost_pd_nd_out,
        dsi_pos_rows=pack.dsi_pos_rows,
        dsi_neg_rows=pack.dsi_neg_rows,
        dsi_pos_ptr=pack.dsi_pos_ptr,
        dsi_neg_ptr=pack.dsi_neg_ptr,
        dsi_target=pack.dsi_target,
        dsi_weight=pack.dsi_weight,
        dsi_power=pack.dsi_power,
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


def _network_backend_from_connectome(C, *, sim_dtype=SIM_DTYPE_DEFAULT) -> ModelBackend:
    """Build a :class:`ModelBackend` from an already-loaded connectome graph."""
    tn = list(C.type_names)
    depol = tuple(tn.index(t) for t in ml.LEAK_DEPOL_TYPES if t in tn)
    conn = C.conn
    return ModelBackend(
        conn=conn,
        e_leak=build_e_leak(conn, C.n_types, depol_cells=depol, dtype=sim_dtype),
        ih_dir=build_ih_dir(conn, dtype=sim_dtype),
        n_types=C.n_types,
        n_cols=1,
        network=C,
        depol_cells=depol,
    )


def load_network_backend(network_json, dev: Optional[str] = None, *, sim_dtype=SIM_DTYPE_DEFAULT) -> ModelBackend:
    """Load connectome network into a :class:`ModelBackend`."""
    from network.construction import load_network

    dev = dev or active_device()
    C = load_network(network_json, device=dev,
                     exc_synweight=exc_synweight, inh_synweight=inh_synweight,
                     dtype=sim_dtype)
    backend = _network_backend_from_connectome(C, sim_dtype=sim_dtype)
    print(f"network: {network_json}")
    print(f"  n_units={backend.n_units}, n_types={backend.n_types}, "
          f"n_pairs={backend.conn.n_pairs}, "
          f"nparams={schema_nparams(default_schema('conductance', backend))}")
    return backend


@dataclass
class _TrainBindCtx:
    """Per-target builder context during :func:`open_session`."""

    model_backend: ModelBackend
    dev: str
    sim_dtype: torch.dtype = SIM_DTYPE_DEFAULT
    cost_weights: Optional[Dict[str, float]] = None
    spot_bright_stimulus_opts: Optional[dict] = None
    spot_dark_stimulus_opts: Optional[dict] = None
    moving_bar_bright_stimulus_opts: Optional[dict] = None
    moving_bar_dark_stimulus_opts: Optional[dict] = None


def _moving_bar_waveform_mse_enabled(cost_weights: Optional[dict], pack_name: str) -> bool:
    """True if PD or ND waveform MSE weight is non-zero for ``pack_name``."""
    w = expand_cost_weight_dict(cost_weights or {})
    return any(
        float(w.get(moving_bar_cost_part_key(pack_name, lab), 1.0)) != 0.0
        for lab in PD_ND_LABELS
    )


def _pack_needs_waveform_mse(pack: TargetPack) -> bool:
    """Spot always; moving-bar only when cost-window targets were built."""
    if pack.name in MOVING_BAR_TARGETS:
        return pack.cost_t0 is not None
    return True


def _moving_bar_polarity_opts(ctx: _TrainBindCtx, polarity: str) -> dict:
    if polarity == "bright":
        raw = ctx.moving_bar_bright_stimulus_opts
    elif polarity == "dark":
        raw = ctx.moving_bar_dark_stimulus_opts
    else:
        raise ValueError(f"unknown moving-bar polarity {polarity!r}")
    if raw:
        return dict(raw)
    return make_moving_bar_stimulus_opts(polarity)


def _build_network_moving_bar_target(ctx: _TrainBindCtx, C, *, pack_name: str, polarity: str):
    from network.moving_bar_target import build_moving_bar_target

    dev = ctx.dev or active_device()
    opts = _moving_bar_polarity_opts(ctx, polarity)
    if opts.get("mode") != "network":
        opts = dict(opts)
        opts["mode"] = "network"
    from network.stimulus import normalize_cost_extent

    if "cost_extent" in opts:
        cost_extent = normalize_cost_extent(opts["cost_extent"])
    else:
        network_extent = int(C.meta.get("extent", -1))
        default_extent = -1 if network_extent <= 0 else network_extent - 1
        cost_extent = normalize_cost_extent(default_extent)
    build_kw = dict(
        C=C,
        device=dev,
        sim_dtype=ctx.sim_dtype,
        t_on=t_on,
        cost_extent=cost_extent,
        i_baseline=opts["i_baseline"],
        contrasts=(polarity,),
        readout_subtypes=_readout_subtypes_from_opts(opts),
        multi_bar=bool(opts.get("multi_bar", True)),
        waveform_mse=_moving_bar_waveform_mse_enabled(ctx.cost_weights, pack_name),
    )
    if polarity == "bright":
        build_kw["i_bright_bar"] = opts["i_bright_bar"]
    else:
        build_kw["i_dark_bar"] = opts["i_dark_bar"]
    T = build_moving_bar_target(**build_kw)
    stim = _enrich_moving_bar_stimulus_opts(opts, T.info, cost_extent=cost_extent)
    pack = TargetPack(
        name=pack_name,
        signal=T.signal,
        data=T.data,
        power=T.power,
        cost_weight=T.cost_weight,
        readout_batch=T.readout_batch,
        readout_unit=T.readout_unit,
        cost_t0=T.cost_t0,
        cost_extent=cost_extent,
        cost_pd_nd=T.cost_pd_nd,
        dsi_pos_rows=T.dsi_pos_rows,
        dsi_neg_rows=T.dsi_neg_rows,
        dsi_pos_ptr=T.dsi_pos_ptr,
        dsi_neg_ptr=T.dsi_neg_ptr,
        dsi_target=T.dsi_target,
        dsi_weight=T.dsi_weight,
        dsi_power=T.dsi_power,
    )
    coltag = _cost_extent_column_coltag(cost_extent, T.info["n_cost_columns"])
    tag = (
        f"moving-bar {polarity} (B={T.n_batch} stimuli, "
        f"{T.info['n_cost']} cost cells, {coltag})"
    )
    return pack, stim, tag


def _build_network_moving_bar_bright_target(
    ctx: _TrainBindCtx, C,
) -> Tuple[TargetPack, dict, str]:
    return _build_network_moving_bar_target(
        ctx, C, pack_name="moving_bar_bright", polarity="bright",
    )


def _build_network_moving_bar_dark_target(
    ctx: _TrainBindCtx, C,
) -> Tuple[TargetPack, dict, str]:
    return _build_network_moving_bar_target(
        ctx, C, pack_name="moving_bar_dark", polarity="dark",
    )


def _build_network_spot_target(
    ctx: _TrainBindCtx, C, *, polarity: str,
) -> Tuple[TargetPack, dict, str]:
    from network.spot_target import build_shifted_target, expand_spot_cost_r_w_dict

    if polarity not in SPOT_POLARITIES:
        raise ValueError(f"spot polarity must be 'bright' or 'dark', got {polarity!r}")
    pack_name = f"spot_{polarity}"
    step_key = _SPOT_STEP_KEY[polarity]
    ctx_opts = (
        ctx.spot_bright_stimulus_opts if polarity == "bright" else ctx.spot_dark_stimulus_opts
    )
    opts = dict(ctx_opts or make_spot_stimulus_opts(polarity, mode="network"))
    from network.stimulus import normalize_cost_extent

    cost_extent = normalize_cost_extent(opts.get("cost_extent"))
    from network.spot_target import (
        DEFAULT_FULLY_INSIDE,
        DEFAULT_MULTI_SPOT,
        DEFAULT_SHIFT_EXTENT,
        DEFAULT_SPOT_EXTENT,
    )
    shift_extent = int(opts.get("shift_extent", DEFAULT_SHIFT_EXTENT))

    spot_extent = float(opts.get("spot_extent", DEFAULT_SPOT_EXTENT))
    multi_spot = bool(opts.get("multi_spot", DEFAULT_MULTI_SPOT))
    fully_inside = bool(opts.get("fully_inside", DEFAULT_FULLY_INSIDE))
    T = build_shifted_target(
        C,
        spot_extent=spot_extent,
        multi_spot=multi_spot,
        fully_inside=fully_inside,
        shift_extent=shift_extent,
        device=ctx.dev or active_device(),
        sim_dtype=ctx.sim_dtype,
        maxtime=IMPULSE_MAXTIME,
        t_on=t_on,
        cost_extent=cost_extent,
        spot_cost_radius_weight=expand_spot_cost_r_w_dict(stimulus_opts=opts),
        i_baseline=opts["i_baseline"],
        polarity=polarity,
        **{step_key: opts[step_key]},
    )
    stim = dict(opts)
    pack = TargetPack(
        name=pack_name,
        signal=T.signal,
        data=T.data,
        power=T.power,
        cost_weight=T.cost_weight,
        readout_batch=T.readout_batch,
        readout_unit=T.readout_unit,
        cost_t0=None,
        cost_radius=T.cost_radius,
        readout_stim_u=T.readout_stim_u,
        readout_stim_v=T.readout_stim_v,
        cost_extent=cost_extent,
    )
    coltag = _cost_extent_column_coltag(cost_extent, T.info["n_cost_columns"])
    shifttag = f"{T.info['n_shifts']} shifts"
    tag = (
        f"{pack_name} (B={T.n_batch} stimuli [{T.info['n_centers']} centres simultaneous "
        f"x {shifttag}], {T.info['n_cost']} cost cells, {coltag})"
    )
    return pack, stim, tag


NETWORK_TARGET_BUILDERS = {
    "spot_bright": lambda ctx, C: _build_network_spot_target(ctx, C, polarity="bright"),
    "spot_dark": lambda ctx, C: _build_network_spot_target(ctx, C, polarity="dark"),
    "moving_bar_bright": _build_network_moving_bar_bright_target,
    "moving_bar_dark": _build_network_moving_bar_dark_target,
}


def expand_target_list(names) -> List[str]:
    """Expand ``--target`` ``TARGET_ALIASES`` shorthands."""
    out = []
    for name in names:
        if name in TARGET_ALIASES:
            out.extend(TARGET_ALIASES[name])
        else:
            out.append(name)
    return out


def expand_cost_extent_dict(kv: Optional[dict]) -> Dict[str, int]:
    """Expand ``--cost-extent`` ``TARGET_ALIASES`` keys."""
    if not kv:
        return {}
    out: Dict[str, int] = {}
    for name, val in kv.items():
        if name in TARGET_ALIASES:
            for t in TARGET_ALIASES[name]:
                out[t] = int(val)
        else:
            out[str(name)] = int(val)
    return out


def resolve_cost_extent_by_target(target_list, default, by_target_kv) -> Dict[str, int]:
    """Map each concrete target to its explicitly requested cost extent."""
    expanded = expand_cost_extent_dict(by_target_kv or {})
    bad = [k for k in expanded if k not in VALID_TARGETS]
    if bad:
        raise ValueError(
            f"unknown target(s) in --cost-extent: {bad} "
            f"(expected {'|'.join(CLI_TARGET_NAMES)})",
        )
    out: Dict[str, int] = {}
    for tname in target_list:
        if tname in expanded:
            out[tname] = int(expanded[tname])
        elif default is not None:
            out[tname] = int(default)
    return out


def apply_cost_extent_to_stimulus_opts(opts, target_name, cost_extent_by_target):
    """Set an explicitly resolved ``cost_extent`` on one target's options."""
    out = dict(opts or {})
    if cost_extent_by_target and target_name in cost_extent_by_target:
        out["cost_extent"] = int(cost_extent_by_target[target_name])
    elif "cost_extent" in out:
        if out["cost_extent"] is None:
            out.pop("cost_extent", None)
        else:
            out["cost_extent"] = int(out["cost_extent"])
    return out


def apply_shift_extent_to_stimulus_opts(opts, target_name, shift_extent):
    """Set ``shift_extent`` on spot stimulus opts."""
    if target_name not in SPOT_TARGETS:
        return opts
    out = dict(opts or {})
    out["shift_extent"] = int(shift_extent)
    return out


def apply_spot_extent_to_stimulus_opts(opts, target_name, spot_extent):
    """Set ``spot_extent`` on spot stimulus opts."""
    if target_name not in SPOT_TARGETS:
        return opts
    from network.spot_target import spot_extent_half_steps

    spot_extent_half_steps(spot_extent)
    out = dict(opts or {})
    out["spot_extent"] = float(spot_extent)
    return out


def apply_multi_spot_to_stimulus_opts(opts, target_name, multi_spot):
    """Set ``multi_spot`` on spot stimulus opts."""
    if target_name not in SPOT_TARGETS:
        return opts
    out = dict(opts or {})
    out["multi_spot"] = bool(multi_spot)
    return out


def apply_fully_inside_to_stimulus_opts(opts, target_name, fully_inside):
    """Set ``fully_inside`` on spot stimulus opts."""
    if target_name not in SPOT_TARGETS:
        return opts
    out = dict(opts or {})
    out["fully_inside"] = bool(fully_inside)
    return out


def apply_spot_cost_radius_weight_to_stimulus_opts(opts, target_name, spot_cost_radius_weight):
    """Set ``spot_cost_radius_weight`` on spot stimulus opts (``None`` → default weights)."""
    if target_name not in SPOT_TARGETS or spot_cost_radius_weight is None:
        return opts
    out = dict(opts or {})
    out["spot_cost_radius_weight"] = {
        str(k): float(v) for k, v in spot_cost_radius_weight.items()
    }
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
                f"--i-bright does not accept target {name!r} "
                f"(expected spot|spot_bright|moving_bar|moving_bar_bright)",
            )
        return list(I_CLI_BRIGHT_TARGETS[name])
    if name not in I_CLI_DARK_TARGETS:
        raise ValueError(
            f"--i-dark does not accept target {name!r} "
            f"(expected spot|spot_dark|moving_bar|moving_bar_dark)",
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


_STIMULUS_TRAIN_OPT_SPECS = (
    ("spot_bright", "spot_bright_stimulus_opts"),
    ("spot_dark", "spot_dark_stimulus_opts"),
    ("moving_bar_bright", "moving_bar_bright_stimulus_opts"),
    ("moving_bar_dark", "moving_bar_dark_stimulus_opts"),
)


def _finalize_stimulus_opts(
    opts,
    target_name,
    *,
    session_mode=None,
    cost_extent_by_target,
    shift_extent,
    spot_extent,
    multi_spot,
    fully_inside,
    spot_cost_radius_weight,
    i_cli,
):
    build_mode = session_mode if session_mode is not None else (opts or {}).get("mode", "network")
    if target_name in SPOT_TARGETS:
        polarity = "bright" if target_name == "spot_bright" else "dark"
        step_key = _SPOT_STEP_KEY[polarity]
        out = make_spot_stimulus_opts(polarity, mode=build_mode, **{
            k: v for k, v in (opts or {}).items()
            if k in (
                "i_baseline", step_key, "shift_extent", "spot_extent",
                "multi_spot", "fully_inside",
            )
        })
    elif target_name == "moving_bar_bright":
        out = make_moving_bar_stimulus_opts(
            "bright",
            mode=build_mode,
            **{
                k: v for k, v in (opts or {}).items()
                if k in ("i_baseline", "i_bright_bar", "readout_subtypes", "multi_bar")
            },
        )
    elif target_name == "moving_bar_dark":
        out = make_moving_bar_stimulus_opts(
            "dark",
            mode=build_mode,
            **{
                k: v for k, v in (opts or {}).items()
                if k in ("i_baseline", "i_dark_bar", "readout_subtypes", "multi_bar")
            },
        )
    else:
        out = dict(opts or {})
    out = apply_cost_extent_to_stimulus_opts(out, target_name, cost_extent_by_target)
    out = apply_shift_extent_to_stimulus_opts(out, target_name, shift_extent)
    out = apply_spot_extent_to_stimulus_opts(out, target_name, spot_extent)
    out = apply_multi_spot_to_stimulus_opts(out, target_name, multi_spot)
    out = apply_fully_inside_to_stimulus_opts(out, target_name, fully_inside)
    out = apply_spot_cost_radius_weight_to_stimulus_opts(out, target_name, spot_cost_radius_weight)
    out = apply_i_cli_to_stimulus_opts(out, target_name, i_cli)
    if session_mode is not None:
        out["mode"] = session_mode
    return out


def expand_cost_weight_dict(weights: Optional[dict]) -> Dict[str, float]:
    """Expand ``--cost-weight`` ``COST_WEIGHT_ALIASES`` keys."""
    if not weights:
        return {}
    out: Dict[str, float] = {}
    for name, val in weights.items():
        if name in COST_WEIGHT_ALIASES:
            for t in COST_WEIGHT_ALIASES[name]:
                out[t] = float(val)
        else:
            out[str(name)] = float(val)
    return out


def _normalize_target_list(target_list) -> List[str]:
    if target_list is None:
        raise ValueError("target_list required")
    if isinstance(target_list, str):
        target_list = parse_comma_list(target_list)
    tl = expand_target_list(list(target_list))
    if not tl:
        raise ValueError("target_list must not be empty")
    bad = [t for t in tl if t not in VALID_TARGETS]
    if bad:
        raise ValueError(
            f"unknown target(s) {bad!r} (expected {'|'.join(CLI_TARGET_NAMES)})",
        )
    return tl


def make_train_opts(
    backend="network",
    target_list=None,
    cost_weights=None,
    pack_overrides=None,
    sequential=None,
    cost_extent_by_target=None,
    shift_extent=None,
    spot_extent=None,
    multi_spot=True,
    fully_inside=True,
    spot_cost_radius_weight=None,
    i_cli=None,
    moving_bar_bright_stimulus_opts=None,
    moving_bar_dark_stimulus_opts=None,
    spot_bright_stimulus_opts=None,
    spot_dark_stimulus_opts=None,
    network_json=None,
    network=None,
    param_partitions=None,
    dev=None,
    packs=None,
    ih_off=IH_OFF_DEFAULT,
    fp32=False,
):
    """Canonical training opts for :func:`open_session` (network backend)."""
    from network.spot_target import DEFAULT_SHIFT_EXTENT, DEFAULT_SPOT_EXTENT

    if backend != "network":
        raise ValueError(f"backend must be 'network', got {backend!r}")
    if network is None and network_json is None:
        raise ValueError("make_train_opts requires network or network_json")
    tl = _normalize_target_list(target_list)
    mode = "network"
    if spot_extent is None:
        spot_extent = DEFAULT_SPOT_EXTENT
    if shift_extent is None:
        shift_extent = DEFAULT_SHIFT_EXTENT
    raw_by_name = {
        "spot_bright": spot_bright_stimulus_opts,
        "spot_dark": spot_dark_stimulus_opts,
        "moving_bar_bright": moving_bar_bright_stimulus_opts,
        "moving_bar_dark": moving_bar_dark_stimulus_opts,
    }
    finalize_kw = dict(
        cost_extent_by_target=cost_extent_by_target,
        shift_extent=shift_extent,
        spot_extent=spot_extent,
        multi_spot=multi_spot,
        fully_inside=fully_inside,
        spot_cost_radius_weight=spot_cost_radius_weight,
        i_cli=i_cli,
    )
    stimulus_opts = {}
    for tname, opts_key in _STIMULUS_TRAIN_OPT_SPECS:
        raw = raw_by_name[tname]
        if tname not in tl and raw is None:
            stimulus_opts[opts_key] = None
            continue
        stimulus_opts[opts_key] = _finalize_stimulus_opts(
            raw if tname in tl else None,
            tname,
            session_mode=mode if tname in tl else None,
            **finalize_kw,
        )
    opts = {
        "backend": "network",
        "target_list": tl,
        "cost_weights": expand_cost_weight_dict(cost_weights or {}),
        "sequential": sequential,
        **stimulus_opts,
    }
    if pack_overrides is not None:
        opts["pack_overrides"] = pack_overrides
    if packs is not None:
        opts["packs"] = packs
    if param_partitions is not None:
        opts["param_partitions"] = param_partitions
    opts["ih_off"] = str(ih_off)
    if fp32:
        opts["fp32"] = True
    opts.update({
        "network": network,
        "network_json": str(network_json) if network_json is not None else None,
        "dev": dev,
    })
    return opts



def _train_opts_for_sidecar(
    opts, backend, target_list,
    resolved_spot_bright, resolved_spot_dark,
    resolved_bar_bright, resolved_bar_dark, sequential_bool,
) -> dict:
    record = {
        "backend": str(backend),
        "target_list": list(target_list),
        "cost_weights": {str(k): float(v) for k, v in (opts.get("cost_weights") or {}).items()},
        "sequential": bool(sequential_bool),
    }
    if backend == "network":
        record.update({
            "network_json": str(opts["network_json"]),
            "spot_bright_stimulus_opts": (
                resolved_spot_bright if resolved_spot_bright is not None
                else opts.get("spot_bright_stimulus_opts")
            ),
            "spot_dark_stimulus_opts": (
                resolved_spot_dark if resolved_spot_dark is not None
                else opts.get("spot_dark_stimulus_opts")
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
        record["spot_bright_stimulus_opts"] = (
            resolved_spot_bright if resolved_spot_bright is not None
            else opts.get("spot_bright_stimulus_opts")
        )
        record["spot_dark_stimulus_opts"] = (
            resolved_spot_dark if resolved_spot_dark is not None
            else opts.get("spot_dark_stimulus_opts")
        )
    overrides = opts.get("pack_overrides")
    if overrides:
        record["pack_overrides"] = overrides
    if opts.get("param_partitions"):
        record["param_partitions"] = opts["param_partitions"]
    if "ih_off" in opts:
        record["ih_off"] = str(opts["ih_off"])
    if opts.get("fp32"):
        record["fp32"] = True
    return record


def _schema_from_opts(model, model_backend, schema, train_opts_record):
    if schema is not None:
        return list(schema)
    base = default_schema(model, model_backend)
    if not train_opts_record:
        return base
    parts = train_opts_record.get("param_partitions")
    if parts:
        base = apply_partitions(
            base, parts, lambda seg: unit_names_for_segment(seg, model_backend),
        )
    return base


def _make_session(
    model_backend: ModelBackend,
    model: str,
    target_list: List[str],
    packs: Dict[str, TargetPack],
    *,
    cost_weights=None,
    sequential=None,
    dev=None,
    train_opts_record=None,
    schema: Optional[list] = None,
    sim_dtype=SIM_DTYPE_DEFAULT,
) -> TrainSession:
    dev_ref = dev or active_device()
    seq = False if sequential is None else bool(sequential)
    if train_opts_record is not None:
        train_opts_record["model"] = model
        train_opts_record["sequential"] = bool(seq)
    ih_off = IH_OFF_DEFAULT
    if train_opts_record is not None and "ih_off" in train_opts_record:
        ih_off = str(train_opts_record["ih_off"])
    if model == 'conductance':
        base = _schema_from_opts(model, model_backend, schema, train_opts_record)
        sch = conductance_schema(model_backend, base, ih_off)
    elif schema is not None:
        sch = list(schema)
    else:
        sch = _schema_from_opts(model, model_backend, None, train_opts_record)
    if train_opts_record is not None:
        train_opts_record["param_partitions"] = schema_partitions_record(
            sch, lambda seg: unit_names_for_segment(seg, model_backend),
        )
    sch = attach_param_carry(sch)
    session = TrainSession(
        backend=model_backend,
        model=model,
        schema=tuple(sch),
        targets=dict(packs),
        target_list=tuple(target_list),
        cost_weights=expand_cost_weight_dict(cost_weights),
        sequential=bool(seq),
        device=dev_ref,
        sim_dtype=sim_dtype,
        train_opts=train_opts_record,
    )
    cost_subpacks = _build_cost_subpacks(session)
    fused_conductance = _build_fused_conductance(session, cost_subpacks)
    return replace(session, cost_subpacks=cost_subpacks, fused_conductance=fused_conductance)


def open_session(
    opts: dict,
    model: str,
    *,
    schema: Optional[list] = None,
    model_backend: Optional[ModelBackend] = None,
) -> TrainSession:
    """Build a :class:`TrainSession` from canonical training opts."""
    backend_name = str(opts.get("backend", "network"))
    if backend_name != "network":
        raise ValueError(f"backend must be 'network', got {backend_name!r}")
    target_list = _normalize_target_list(opts.get("target_list"))
    bad = [t for t in target_list if t not in VALID_TARGETS]
    if bad:
        raise ValueError(f"unknown target(s) {bad!r} (expected {'|'.join(CLI_TARGET_NAMES)})")
    dev = opts.get("dev") or active_device()
    sim_dtype = sim_dtype_from_fp32(bool(opts.get("fp32", False)))

    C = opts.get("network")
    if C is None:
        nj = opts.get("network_json")
        if not nj:
            raise ValueError("open_session(network) requires opts['network'] or network_json")
        from network.construction import load_network
        C = load_network(
            nj, device=dev,
            exc_synweight=exc_synweight, inh_synweight=inh_synweight,
            dtype=sim_dtype,
        )
    if model_backend is None:
        model_backend = _network_backend_from_connectome(C, sim_dtype=sim_dtype)
    elif model_backend.network is not C:
        raise ValueError("model_backend.network must be opts['network']")
    ctx = _TrainBindCtx(
        model_backend=model_backend,
        dev=dev,
        sim_dtype=sim_dtype,
        cost_weights=opts.get("cost_weights"),
        spot_bright_stimulus_opts=opts.get("spot_bright_stimulus_opts"),
        spot_dark_stimulus_opts=opts.get("spot_dark_stimulus_opts"),
        moving_bar_bright_stimulus_opts=opts.get("moving_bar_bright_stimulus_opts"),
        moving_bar_dark_stimulus_opts=opts.get("moving_bar_dark_stimulus_opts"),
    )
    packs = {}
    pack_overrides = opts.get("pack_overrides") or {}
    resolved_spot_bright = resolved_spot_dark = None
    resolved_bar_bright = resolved_bar_dark = None
    for tname in target_list:
        pack, stim, _tag = NETWORK_TARGET_BUILDERS[tname](ctx, C)
        if tname in pack_overrides:
            pack = apply_pack_override(pack, pack_overrides[tname], model_backend)
        packs[tname] = pack
        if tname == "spot_bright":
            resolved_spot_bright = stim
        elif tname == "spot_dark":
            resolved_spot_dark = stim
        elif tname == "moving_bar_bright":
            resolved_bar_bright = stim
        elif tname == "moving_bar_dark":
            resolved_bar_dark = stim
    record = _train_opts_for_sidecar(
        opts, "network", target_list,
        resolved_spot_bright, resolved_spot_dark,
        resolved_bar_bright, resolved_bar_dark, False,
    )
    return _make_session(
        model_backend, model, target_list, packs,
        cost_weights=opts.get("cost_weights"),
        sequential=opts.get("sequential"),
        dev=dev,
        train_opts_record=record,
        schema=schema,
        sim_dtype=sim_dtype,
    )


def open_session_from_opts(opts: dict, model: str | None = None, **kwargs) -> TrainSession:
    """Restore a session from a saved ``train_opts.json`` dict."""
    opts = dict(opts)
    if model is None:
        model = opts.get("model")
        if not model:
            raise ValueError("train_opts requires model")
    opts["packs"] = None
    backend = str(opts.get("backend", "network"))
    if backend != "network":
        raise ValueError(f"backend must be 'network', got {backend!r}")
    nj = opts.get("network_json")
    if not nj:
        raise ValueError("train_opts requires network_json")
    if not opts.get("target_list"):
        raise ValueError("train_opts requires target_list")
    sim_dtype = sim_dtype_from_fp32(bool(opts.get("fp32", False)))
    mb = load_network_backend(
        nj, dev=opts.get("dev") or active_device(), sim_dtype=sim_dtype,
    )
    opts["network"] = mb.network
    kwargs.setdefault("model_backend", mb)
    return open_session({**opts, "backend": "network"}, model, **kwargs)


def open_session_from_outdir(
    outdir: str,
    model: str | None = None,
) -> TrainSession:
    """Load ``train_opts.json`` from a run folder and return a ready session."""
    import json
    from training_config import run_data_dir
    opts_path = os.path.join(run_data_dir(os.path.abspath(outdir)), TRAIN_OPTS_FILE)
    if not os.path.isfile(opts_path):
        raise FileNotFoundError(f"missing {opts_path}")
    with open(opts_path) as f:
        opts = json.load(f)
    return open_session_from_opts(opts, model)


# ------- param unpack / cost / readout helpers -------------------------
# Neuron dynamics live in ``neuron_model/`` (re-exported at module top).

def _reconstruct_raw(seg, z_slice, z):
    """Build length-`count` per-unit vector from z slice + partition buckets."""
    count = seg_count(seg)
    const = _fixed_const(seg)
    raw = torch.full((count,), const, dtype=z.dtype, device=z.device)
    carry = seg.get('carry')
    i = 0
    for u in seg.get('indi', ()):
        raw[int(u)] = z_slice[i]
        i += 1
    if seg.get('shared'):
        shared_val = z_slice[i]
        for u in seg['shared']:
            raw[int(u)] = shared_val
    for u in seg.get('frozen', ()):
        if carry is not None:
            raw[int(u)] = torch.tensor(float(carry[int(u)]), dtype=z.dtype, device=z.device)
        else:
            raw[int(u)] = torch.tensor(float(seg['init']), dtype=z.dtype, device=z.device)
    return raw


def _expand_segment(seg, raw, backend: ModelBackend):
    """Map a length-`count` per-unit vector to a usable parameter, per its 'kind'."""
    kind = seg['kind']
    dev = backend.conn.node_type.device
    if kind == 'full':
        return calc_multi_col_params(raw, backend.conn).to(dev)
    if kind == 'output':
        return raw.to(dev)
    if kind == 'edge_pair':
        return raw.to(dev)
    raise ValueError(f"unknown segment kind: {kind}")


def assign_params(z, schema, backend: ModelBackend):
    """Unpack z into a dict of parameter tensors, driven by the given schema partitions."""
    p = {}
    for seg, start, stop in schema_segments(schema):
        p[seg['name']] = _expand_segment(seg, _reconstruct_raw(seg, z[start:stop], z), backend)
    return p


def model_cost(model, data, session: TrainSession, scale=1.0, power=None):
    # normalised MSE over the response window (t=t_on..maxtime-1); ``scale`` is an
    # arbitrary linear gain on ``model`` (diagnostics only — training uses schema out_scale).
    if power is None:
        power = session.primary_pack.power
    mt = session.maxtime
    return torch.sum((scale * model - data[t_on:mt])**2) / power * 100.0


def _window_time_traces(model_full, b_idx, u_idx, t0, win=None):
    """Extract per-readout windows from ``model_full`` ``(B, maxtime, N)``.

    ``t0`` is the absolute simulation step of window start (slot ``k`` uses ``t0 + k``).
    Slots with ``t0 + k < t_on`` are zero (cost / training alignment).
    """
    if win is None:
        raise ValueError("window length win required")
    win = int(win)
    dev = model_full.device
    k = torch.arange(win, dtype=torch.long, device=dev)
    t_idx = t0[:, None].to(device=dev, dtype=torch.long) + k[None, :]
    t_max = model_full.shape[1] - 1
    t_safe = t_idx.clamp(0, t_max)
    sel = model_full[b_idx[:, None], t_safe, u_idx[:, None]]
    pre = t_idx < int(t_on)
    return torch.where(pre, torch.zeros_like(sel), sel)


def _readout_model_traces_pack(model_full, pack: TargetPack):
    """Select model traces for cost cells; windowed when ``pack.cost_t0`` is set."""
    if pack.cost_t0 is None:
        return model_full[pack.readout_batch, t_on:, pack.readout_unit]
    return _window_time_traces(
        model_full, pack.readout_batch, pack.readout_unit, pack.cost_t0,
        win=pack.data.shape[1],
    )


def out_scale_for_units(p, unit_index, backend: ModelBackend, *, sim_dtype=SIM_DTYPE_DEFAULT):
    """Per-unit ``out_scale`` using the same indexing as ``_pack_out_scale``."""
    os_param = p.get('out_scale', 1.0)
    n = int(unit_index.shape[0])
    dev = unit_index.device
    if not torch.is_tensor(os_param) or os_param.dim() == 0:
        val = float(os_param if not torch.is_tensor(os_param) else os_param.item())
        return torch.full((n,), val, dtype=sim_dtype, device=dev)
    if backend.network is not None:
        ci = backend.network.node_type[unit_index]
    else:
        ci = unit_index % backend.n_types
    return os_param[ci]


def _pack_out_scale(p, pack: TargetPack, backend: ModelBackend, session: TrainSession):
    """Per-cost-row output scale from schema ``out_scale`` (single source of truth)."""
    return out_scale_for_units(p, pack.readout_unit, backend, sim_dtype=session.sim_dtype)


# MODEL_PACK_READOUTS imported from neuron_model

def _pack_model_readouts(p, pack: TargetPack, session: TrainSession, batch_idx=None):
    try:
        readout = MODEL_PACK_READOUTS[session.model]
    except KeyError:
        raise ValueError(f"no pack readout for model={session.model!r}") from None
    return readout(p, pack, session, batch_idx)


def _subgroup_power(weight, data):
    power = torch.sum(weight[:, None] * data ** 2)
    if float(power) == 0.0:
        power = torch.tensor(1.0, dtype=data.dtype, device=data.device)
    return power


def _pack_cost_mse(scale, data, weight, sel, power):
    diff = scale[:, None] * sel - data
    return torch.sum(weight[:, None] * diff ** 2) / power * 100.0


def _part_weight(session: TrainSession, part_key: str) -> float:
    return float(session.cost_weights.get(part_key, 1.0))


def _pack_part_key_for_cell(pack: TargetPack, cell_idx: int) -> str:
    if pack.cost_pd_nd is not None:
        label = PD_ND_LABELS[int(pack.cost_pd_nd[cell_idx].item())]
        return moving_bar_cost_part_key(pack.name, label)
    return pack.name


def _pack_has_active_cost(pack: TargetPack, session: TrainSession) -> bool:
    for key in cost_part_keys_for_target(pack.name):
        if _part_weight(session, key) != 0.0:
            return True
    return False


def _pack_has_active_mse(pack: TargetPack, session: TrainSession) -> bool:
    """True if waveform MSE parts (pack name or PD/ND) have non-zero weight."""
    if pack.name in MOVING_BAR_TARGETS:
        return any(
            _part_weight(session, moving_bar_cost_part_key(pack.name, lab)) != 0.0
            for lab in PD_ND_LABELS
        )
    return _part_weight(session, pack.name) != 0.0


def _mse_active_row_mask(pack: TargetPack, session: TrainSession) -> torch.Tensor:
    """Boolean mask over cost rows with non-zero PD/ND (or pack) weight."""
    n = int(pack.readout_batch.shape[0])
    dev = pack.readout_batch.device
    if pack.name in MOVING_BAR_TARGETS:
        if not _pack_has_active_mse(pack, session) or pack.cost_pd_nd is None:
            return torch.zeros(n, dtype=torch.bool, device=dev)
        mask = torch.zeros(n, dtype=torch.bool, device=dev)
        for idx, label in ((PD_IDX, "PD"), (ND_IDX, "ND")):
            if _part_weight(session, moving_bar_cost_part_key(pack.name, label)) != 0.0:
                mask |= pack.cost_pd_nd == int(idx)
        return mask
    on = _part_weight(session, pack.name) != 0.0
    return torch.full((n,), on, dtype=torch.bool, device=dev)


def _dsi_active_row_mask(pack: TargetPack, session: TrainSession) -> torch.Tensor:
    """Boolean mask over cost rows needed by a non-zero DSI weight."""
    n = int(pack.readout_batch.shape[0])
    dev = pack.readout_batch.device
    mask = torch.zeros(n, dtype=torch.bool, device=dev)
    dsi_key = moving_bar_cost_part_key(pack.name, "DSI")
    if (
        pack.dsi_pos_rows is None
        or pack.dsi_pos_rows.numel() == 0
        or _part_weight(session, dsi_key) == 0.0
    ):
        return mask
    mask[pack.dsi_pos_rows] = True
    mask[pack.dsi_neg_rows] = True
    return mask


def _pack_active_batch_indices(pack: TargetPack, session: TrainSession) -> Tuple[int, ...]:
    """Stimulus batch indices with at least one non-zero-weight cost cell."""
    row_mask = _mse_active_row_mask(pack, session) | _dsi_active_row_mask(pack, session)
    if not bool(row_mask.any()):
        return ()
    batches = pack.readout_batch[row_mask].unique(sorted=True)
    return tuple(int(b) for b in batches.tolist())


def _active_row_indices(
    pack: TargetPack,
    session: TrainSession,
    batch_idx: Optional[int] = None,
) -> Optional[torch.Tensor]:
    keep = _mse_active_row_mask(pack, session) | _dsi_active_row_mask(pack, session)
    if batch_idx is not None:
        keep = keep & (pack.readout_batch == int(batch_idx))
    if not bool(keep.any()):
        return None
    return torch.nonzero(keep, as_tuple=False).reshape(-1)


def _slice_pack_rows(pack: TargetPack, row_ix: torch.Tensor) -> TargetPack:
    from t4_t5_dsi import remap_dsi_rows

    fields = {
        "data": pack.data[row_ix],
        "cost_weight": pack.cost_weight[row_ix],
        "readout_batch": pack.readout_batch[row_ix],
        "readout_unit": pack.readout_unit[row_ix],
    }
    if pack.cost_t0 is not None:
        fields["cost_t0"] = pack.cost_t0[row_ix]
    if pack.cost_radius is not None:
        fields["cost_radius"] = pack.cost_radius[row_ix]
    if pack.readout_stim_u is not None:
        fields["readout_stim_u"] = pack.readout_stim_u[row_ix]
    if pack.readout_stim_v is not None:
        fields["readout_stim_v"] = pack.readout_stim_v[row_ix]
    if pack.cost_pd_nd is not None:
        fields["cost_pd_nd"] = pack.cost_pd_nd[row_ix]
    fields.update(remap_dsi_rows(pack, row_ix))
    return replace(pack, **fields)


def _subset_pack_batches(pack: TargetPack, batch_indices: Tuple[int, ...]) -> Optional[TargetPack]:
    from t4_t5_dsi import remap_dsi_rows

    if len(batch_indices) == int(pack.signal.shape[0]):
        return pack
    dev = pack.signal.device
    idx_t = torch.tensor(batch_indices, dtype=torch.long, device=dev)
    rb = pack.readout_batch
    keep = torch.isin(rb, idx_t)
    if not bool(keep.any()):
        return None
    lut_size = int(max(max(batch_indices), int(rb.max()))) + 1
    lut = torch.full((lut_size,), -1, dtype=torch.long, device=dev)
    lut[idx_t] = torch.arange(len(batch_indices), dtype=torch.long, device=dev)
    new_rb = lut[rb[keep]]
    kept_old = torch.nonzero(keep, as_tuple=False).reshape(-1)
    fields = {
        "signal": pack.signal.index_select(0, idx_t),
        "data": pack.data[keep],
        "cost_weight": pack.cost_weight[keep],
        "readout_batch": new_rb,
        "readout_unit": pack.readout_unit[keep],
    }
    if pack.cost_t0 is not None:
        fields["cost_t0"] = pack.cost_t0[keep]
    if pack.cost_radius is not None:
        fields["cost_radius"] = pack.cost_radius[keep]
    if pack.readout_stim_u is not None:
        fields["readout_stim_u"] = pack.readout_stim_u[keep]
    if pack.readout_stim_v is not None:
        fields["readout_stim_v"] = pack.readout_stim_v[keep]
    if pack.cost_pd_nd is not None:
        fields["cost_pd_nd"] = pack.cost_pd_nd[keep]
    fields.update(remap_dsi_rows(pack, kept_old))
    return replace(pack, **fields)


def _pack_for_active_cost(
    pack: TargetPack,
    session: TrainSession,
    *,
    batch_idx: Optional[int] = None,
    batch_indices: Optional[Tuple[int, ...]] = None,
) -> Optional[TargetPack]:
    """Drop zero-weight rows and, when requested, inactive stimulus batches."""
    work = pack
    if batch_indices is not None:
        work = _subset_pack_batches(pack, batch_indices)
        if work is None:
            return None
    rows = _active_row_indices(work, session, batch_idx=batch_idx)
    if rows is None:
        return None
    return _slice_pack_rows(work, rows)


def _build_cost_subpacks(session: TrainSession) -> Dict[str, TargetPack]:
    """Active cost row/batch subsets per target (batched mode only)."""
    if session.sequential:
        return {}
    out: Dict[str, TargetPack] = {}
    for name, pack in session.targets.items():
        if not _pack_has_active_cost(pack, session):
            continue
        active_batches = _pack_active_batch_indices(pack, session)
        if not active_batches:
            continue
        sub = _pack_for_active_cost(pack, session, batch_indices=active_batches)
        if sub is not None:
            out[name] = sub
    return out


def _signal_fuse_key(pack: TargetPack) -> Tuple[int, int, str, torch.dtype]:
    sig = pack.signal
    return (int(sig.shape[1]), int(sig.shape[2]), str(sig.device), sig.dtype)


def _build_fused_conductance(
    session: TrainSession,
    cost_subpacks: Dict[str, TargetPack],
) -> Tuple[FusedConductanceForward, ...]:
    if session.model != "conductance" or session.sequential or not cost_subpacks:
        return ()
    by_key: Dict[Tuple[int, int, str, torch.dtype], List[TargetPack]] = {}
    for pack in cost_subpacks.values():
        by_key.setdefault(_signal_fuse_key(pack), []).append(pack)
    fused: List[FusedConductanceForward] = []
    for packs in by_key.values():
        offsets: List[int] = []
        off = 0
        for pack in packs:
            offsets.append(off)
            off += int(pack.signal.shape[0])
        fused.append(FusedConductanceForward(subpacks=tuple(packs), batch_offsets=tuple(offsets)))
    return tuple(fused)


def _conductance_readout_from_model_full(
    model_full: torch.Tensor,
    pack: TargetPack,
    *,
    batch_offset: int = 0,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    rb = pack.readout_batch if batch_offset == 0 else pack.readout_batch + batch_offset
    dsi_sel = model_full[rb, t_on:, pack.readout_unit]
    if not _pack_needs_waveform_mse(pack):
        return None, dsi_sel
    if pack.cost_t0 is None:
        return dsi_sel, dsi_sel
    mse_sel = _window_time_traces(
        model_full, rb, pack.readout_unit, pack.cost_t0,
        win=pack.data.shape[1],
    )
    return mse_sel, dsi_sel


def _pack_cost_dsi_from_sel(
    pack: TargetPack,
    session: TrainSession,
    scale: torch.Tensor,
    dsi_sel: torch.Tensor,
) -> Optional[torch.Tensor]:
    """DSI cost from full post-stimulus traces; independent of cost windows."""
    from t4_t5_dsi import cost_dsi_from_sel

    key = moving_bar_cost_part_key(pack.name, "DSI")
    if _part_weight(session, key) == 0.0:
        return None
    return cost_dsi_from_sel(pack, scale, dsi_sel)


def _pack_cost_parts_from_sel(
    pack: TargetPack,
    session: TrainSession,
    scale: torch.Tensor,
    sel: Optional[torch.Tensor],
    dsi_sel: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    pd_nd = pack.cost_pd_nd
    if pack.name in MOVING_BAR_TARGETS:
        out: Dict[str, torch.Tensor] = {}
        if pd_nd is not None:
            for pd_nd_idx, label in ((PD_IDX, "PD"), (ND_IDX, "ND")):
                key = moving_bar_cost_part_key(pack.name, label)
                if _part_weight(session, key) == 0.0:
                    continue
                if sel is None:
                    raise ValueError(
                        f"waveform readout required for {key} but pack has no cost window",
                    )
                mask = pd_nd == pd_nd_idx
                if not bool(mask.any()):
                    out[key] = torch.zeros(
                        (), dtype=session.sim_dtype, device=session.device,
                    )
                    continue
                power = _subgroup_power(pack.cost_weight[mask], pack.data[mask])
                out[key] = _pack_cost_mse(
                    scale[mask], pack.data[mask], pack.cost_weight[mask],
                    sel[mask], power,
                )
        dsi_part = _pack_cost_dsi_from_sel(pack, session, scale, dsi_sel)
        if dsi_part is not None:
            out[moving_bar_cost_part_key(pack.name, "DSI")] = dsi_part
        return out
    if sel is None:
        raise ValueError(f"waveform readout required for pack {pack.name!r}")
    return {
        pack.name: _pack_cost_mse(
            scale, pack.data, pack.cost_weight, sel, pack.power,
        ),
    }


def _pack_cost_parts_from_conductance_model(
    p,
    pack: TargetPack,
    session: TrainSession,
    model_full: torch.Tensor,
    *,
    batch_offset: int = 0,
) -> Dict[str, torch.Tensor]:
    scale = _pack_out_scale(p, pack, session.backend, session)
    sel, dsi_sel = _conductance_readout_from_model_full(
        model_full, pack, batch_offset=batch_offset,
    )
    return _pack_cost_parts_from_sel(pack, session, scale, sel, dsi_sel)


def _calc_cost_parts_fused_conductance(
    p,
    session: TrainSession,
) -> Dict[str, torch.Tensor]:
    parts: Dict[str, torch.Tensor] = {}
    for group in session.fused_conductance:
        if len(group.subpacks) == 1:
            sig = group.subpacks[0].signal
        else:
            sig = torch.cat([pack.signal for pack in group.subpacks], dim=0)
        model_full = _run_conductance_full(session, p, sig)
        for pack, off in zip(group.subpacks, group.batch_offsets):
            for key, part in _pack_cost_parts_from_conductance_model(
                p, pack, session, model_full, batch_offset=off,
            ).items():
                if _part_weight(session, key) != 0.0:
                    parts[key] = part
    return parts


def _pack_cost_forward(p, pack: TargetPack, session: TrainSession, batch_idx=None):
    scale = _pack_out_scale(p, pack, session.backend, session)
    pd_nd = pack.cost_pd_nd
    if batch_idx is not None:
        mask = pack.readout_batch == int(batch_idx)
        if not bool(mask.any()):
            return None
        scale = scale[mask]
        data = pack.data[mask]
        weight = pack.cost_weight[mask]
        if pd_nd is not None:
            pd_nd = pd_nd[mask]
    else:
        data = pack.data
        weight = pack.cost_weight
    sel, dsi_sel = _pack_model_readouts(p, pack, session, batch_idx)
    return scale, data, weight, sel, dsi_sel, pd_nd


def _pack_cost_parts_from_params(p, pack: TargetPack, session: TrainSession, batch_idx=None):
    """Unweighted cost parts for one pack (PD/ND split for moving_bar)."""
    fwd = _pack_cost_forward(p, pack, session, batch_idx)
    if fwd is None:
        return {}
    scale, data, weight, sel, dsi_sel, pd_nd = fwd
    return _pack_cost_parts_from_sel(pack, session, scale, sel, dsi_sel)


def _pack_cost_rows(p, pack: TargetPack, session: TrainSession, batch_idx=None):
    """Forward + MSE for one pack (full aggregate; used for diagnostics)."""
    fwd = _pack_cost_forward(p, pack, session, batch_idx)
    if fwd is None:
        return None
    scale, data, weight, sel, dsi_sel, _pd_nd = fwd
    if sel is None:
        return _pack_cost_dsi_from_sel(pack, session, scale, dsi_sel)
    return _pack_cost_mse(scale, data, weight, sel, pack.power)


def _pack_cost_parts_for_pack(z, pack: TargetPack, session: TrainSession, batch_idx=None, p=None):
    if p is None:
        p = params_from_z(z, session)
    return _pack_cost_parts_from_params(p, pack, session, batch_idx)


def _pack_cost_part(z, pack: TargetPack, session: TrainSession, batch_idx=None):
    parts = _pack_cost_parts_for_pack(z, pack, session, batch_idx)
    if not parts:
        return torch.zeros((), dtype=session.sim_dtype, device=session.device)
    return sum(parts.values())


def _pack_cost(z, pack: TargetPack, session: TrainSession, batch_idx=None):
    return _pack_cost_part(z, pack, session, batch_idx)


def calc_cost_parts(z, session: TrainSession) -> Dict[str, torch.Tensor]:
    """Per-part unweighted cost (before ``cost_weights``)."""
    p = params_from_z(z, session)
    if session.fused_conductance:
        return _calc_cost_parts_fused_conductance(p, session)
    parts: Dict[str, torch.Tensor] = {}
    zero = torch.zeros((), dtype=session.sim_dtype, device=session.device)
    if session.cost_subpacks and not session.sequential:
        for _name, pack in session.cost_subpacks.items():
            if not _pack_has_active_cost(pack, session):
                continue
            pack_parts = _pack_cost_parts_from_params(p, pack, session, batch_idx=None)
            for part_key, part in pack_parts.items():
                if _part_weight(session, part_key) == 0.0:
                    continue
                parts[part_key] = part
        return parts
    for _name, pack in session.targets.items():
        if not _pack_has_active_cost(pack, session):
            continue
        active_batches = _pack_active_batch_indices(pack, session)
        if not active_batches:
            continue
        if session.sequential:
            pack_parts: Dict[str, torch.Tensor] = {}
            if _pack_has_active_mse(pack, session):
                for b in active_batches:
                    sub = _pack_for_active_cost(pack, session, batch_idx=b)
                    if sub is None:
                        continue
                    for key, part in _pack_cost_parts_from_params(
                        p, sub, session, batch_idx=b,
                    ).items():
                        pack_parts[key] = pack_parts.get(key, zero) + part
            dsi_key = moving_bar_cost_part_key(pack.name, "DSI")
            if _part_weight(session, dsi_key) != 0.0:
                for group in _dsi_sequential_batch_groups(pack, session):
                    sub_dsi = _pack_for_dsi_batch_group(pack, session, group)
                    if sub_dsi is None:
                        continue
                    dsi_parts = _pack_cost_parts_from_params(
                        p, sub_dsi, session, batch_idx=None,
                    )
                    if dsi_key in dsi_parts:
                        pack_parts[dsi_key] = (
                            pack_parts.get(dsi_key, zero) + dsi_parts[dsi_key]
                        )
        else:
            sub = _pack_for_active_cost(pack, session, batch_indices=active_batches)
            if sub is None:
                continue
            pack_parts = _pack_cost_parts_from_params(p, sub, session, batch_idx=None)
        for part_key, part in pack_parts.items():
            if _part_weight(session, part_key) == 0.0:
                continue
            parts[part_key] = part
    return parts


def _weighted_cost_from_parts(parts: Dict[str, torch.Tensor], session: TrainSession):
    total = torch.zeros((), dtype=session.sim_dtype, device=session.device)
    for name, part in parts.items():
        w = float(session.cost_weights.get(name, 1.0))
        total = total + w * part
    return total


def calc_cost(z, session: TrainSession):
    return _weighted_cost_from_parts(calc_cost_parts(z, session), session)


def _params_from_z(z, session: TrainSession):
    return params_from_z(z, session)


def _pack_spec_names(session: TrainSession, pack: TargetPack) -> Tuple[str, ...]:
    opts = ((session.train_opts or {}).get(f"{pack.name}_stimulus_opts")) or {}
    names = opts.get("spec_names")
    if names:
        return tuple(str(s) for s in names)
    from network.moving_bar_target import bar_specs_for_session

    return tuple(s.name for s in bar_specs_for_session(session, pack.name))


def _dsi_sequential_batch_groups(
    pack: TargetPack, session: TrainSession,
) -> Tuple[Tuple[int, ...], ...]:
    """Active DSI microbatches: each group is one axis×width (typically B=2)."""
    from t4_t5_dsi import dsi_sequential_batch_pairs

    active = set(_pack_active_batch_indices(pack, session))
    groups: list[tuple[int, ...]] = []
    for pair in dsi_sequential_batch_pairs(_pack_spec_names(session, pack)):
        kept = tuple(b for b in pair if b in active)
        if len(kept) < 2:
            continue
        groups.append(kept)
    return tuple(groups)


def _pack_for_dsi_batch_group(
    pack: TargetPack,
    session: TrainSession,
    batch_indices: Tuple[int, ...],
) -> Optional[TargetPack]:
    """Subset to one DSI direction pair; keep parent ``dsi_power`` for additive costs."""
    sub = _pack_for_active_cost(pack, session, batch_indices=batch_indices)
    if sub is None or pack.dsi_power is None:
        return sub
    return replace(sub, dsi_power=pack.dsi_power)


def _iter_cost_microbatches(session: TrainSession):
    """Yield ``(pack, batch_idx, sub_pack)`` for gradient accumulation.

    Sequential moving-bar DSI yields one microbatch per axis×width (B≈2), not all
    directions at once.
    """
    for _name, pack in session.targets.items():
        if not _pack_has_active_cost(pack, session):
            continue
        active_batches = _pack_active_batch_indices(pack, session)
        if not active_batches:
            continue
        if session.sequential:
            if _pack_has_active_mse(pack, session):
                for b in active_batches:
                    sub = _pack_for_active_cost(pack, session, batch_idx=b)
                    if sub is not None:
                        yield pack, b, sub
            dsi_key = moving_bar_cost_part_key(pack.name, "DSI")
            if _part_weight(session, dsi_key) != 0.0:
                for group in _dsi_sequential_batch_groups(pack, session):
                    sub_dsi = _pack_for_dsi_batch_group(pack, session, group)
                    if sub_dsi is not None:
                        yield pack, None, sub_dsi
        else:
            sub = _pack_for_active_cost(pack, session, batch_indices=active_batches)
            if sub is not None:
                yield pack, None, sub


def backward_accum_weighted_cost(z, session: TrainSession):
    """Backward weighted cost one micro-batch at a time (releases graph each step)."""
    parts_sum: Dict[str, float] = {}
    zero = torch.zeros((), dtype=session.sim_dtype, device=session.device)
    for pack, batch_idx, sub in _iter_cost_microbatches(session):
        p = _params_from_z(z, session)
        mb_loss = zero
        has_loss = False
        dsi_only = (
            session.sequential
            and batch_idx is None
            and pack.name in MOVING_BAR_TARGETS
        )
        dsi_key = moving_bar_cost_part_key(pack.name, "DSI")
        for key, part in _pack_cost_parts_from_params(
            p, sub, session, batch_idx=batch_idx,
        ).items():
            if dsi_only and key != dsi_key:
                continue
            if (not dsi_only) and session.sequential and key == dsi_key:
                # single-batch slices have no complete DSI pairs; skip zeros
                continue
            w = _part_weight(session, key)
            if w == 0.0:
                continue
            mb_loss = mb_loss + w * part
            has_loss = True
            parts_sum[key] = parts_sum.get(key, 0.0) + float(part.item())
        if has_loss:
            mb_loss.backward()
    total = sum(_part_weight(session, k) * v for k, v in parts_sum.items())
    return total, parts_sum


def schema_bounds(schema, sim_dtype=SIM_DTYPE_DEFAULT):
    zb = torch.zeros((schema_nparams(schema), 2), dtype=sim_dtype)
    for seg, start, stop in schema_segments(schema):
        if stop > start:                       # skip fixed (0 trainable rows)
            zb[start:stop] = torch.tensor([seg['lo'], seg['hi']], dtype=sim_dtype)
    return zb

def schema_guess(schema, sim_dtype=SIM_DTYPE_DEFAULT):
    z = np.zeros(schema_nparams(schema))
    for seg, start, stop in schema_segments(schema):
        n = stop - start
        if n == 0:
            continue
        z[start:stop] = seg['init'] + (np.random.rand(n) - 0.5) * seg['jit']
    return torch.tensor(z, dtype=sim_dtype).to(active_device())

def guess_initial_params(session: TrainSession):
    return schema_guess(list(session.schema), session.sim_dtype)


def _float_parts_dict(parts: Optional[Dict[str, torch.Tensor]], target_order=None):
    if not parts:
        return None
    out = {k: float(v.item() if torch.is_tensor(v) else v) for k, v in parts.items()}
    if target_order:
        return {k: out[k] for k in target_order if k in out}
    return out


def _fmt_cost_parts(parts):
    if not parts:
        return ""
    return "  [" + "  ".join(f"{k}={v:.4f}" for k, v in parts.items()) + "]"


_TQDM_REFRESH_INTERVAL = 10


def gradient_network(z, lr=0.0001, cost_fn=None, n_steps=100, device="cpu", z_bounds=None,
                     cost_log=None, step_log=None, float_last_parts=None, target_order=None,
                     backward_step=None, eval_cost=None):
    
    a = time.time()

    z = nn.Parameter(z.clone().to(device))
    
    optimizer = torch.optim.Adam([z], lr=lr)

    try:
        if eval_cost is not None:
            cost = eval_cost(z)
        else:
            cost = cost_fn(z).item()
    except RuntimeError as e:
        raise RuntimeError(f'non-finite at init: {e}') from e
    if not np.isfinite(cost):
        raise RuntimeError(f'non-finite cost at init: {cost}')
    best_cost = cost
    best_z = z.clone().detach()
    
    initial_cost = 1.0 * cost
    initial_parts = float_last_parts(target_order) if float_last_parts else None
    best_parts = initial_parts

    progress_bar = tqdm(
        range(n_steps),
        desc=f'Cost: {cost:.4f}' + _fmt_cost_parts(initial_parts),
        miniters=_TQDM_REFRESH_INTERVAL,
        maxinterval=60,
        file=sys.stderr,
    )
    aborted = None

    for i in progress_bar:
        
        optimizer.zero_grad()
        
        try:
            if backward_step is not None:
                cost = backward_step(z)
            else:
                cost_t = cost_fn(z)
                cost = cost_t.item()
                cost_t.backward()
        except RuntimeError as e:
            aborted = f'step {i}: {e}'
            break

        if not np.isfinite(cost):
            aborted = f'step {i}: non-finite cost={cost}'
            break
        if not torch.isfinite(z).all():
            aborted = f'step {i}: non-finite z'
            break
        if z.grad is not None and not torch.isfinite(z.grad).all():
            aborted = f'step {i}: non-finite grad'
            break
        
        if cost < best_cost:
            
            best_cost = cost
            best_z = z.clone().detach()
            if float_last_parts is not None:
                best_parts = float_last_parts(target_order)
        
        if cost_log is not None:
            cost_log.append(cost)
        if step_log is not None:
            step_log(z)
        
        optimizer.step()

        with torch.no_grad():
            
            z.clamp_(z_bounds[:, 0].to(device), z_bounds[:, 1].to(device))

        step_parts = float_last_parts(target_order) if float_last_parts else None
        if (i + 1) % _TQDM_REFRESH_INTERVAL == 0 or i == n_steps - 1:
            progress_bar.set_description(
                f'Cost: {cost:.4f}' + _fmt_cost_parts(step_parts),
                refresh=False,
            )

    if aborted is None:
        try:
            if eval_cost is not None:
                cost = eval_cost(z)
            else:
                cost = cost_fn(z).item()
            final_parts = float_last_parts(target_order) if float_last_parts else None
        except RuntimeError as e:
            aborted = f'final eval: {e}'
            cost = float('nan')
            final_parts = None
        else:
            if np.isfinite(cost) and cost < best_cost:
                best_cost = cost
                best_z = z.clone().detach()
                best_parts = final_parts
    else:
        cost = float('nan')
        final_parts = None

    print()
    if aborted is not None:
        print('ABORT:', aborted)
    print('Initl cost =', format(initial_cost,'.4f') + _fmt_cost_parts(initial_parts))
    print('Final cost =', format(cost,'.4f') + _fmt_cost_parts(final_parts))
    print('Best  cost =', format(best_cost,'.4f') + _fmt_cost_parts(best_parts))
    
    b = time.time()
    
    print('time needed  =',format(b-a,'.2f'),' sec')
    print()

    return best_z

def train_staged(z, cost_fn, z_bounds, lrs, nsteps, cost_log=None, step_log=None,
                 float_last_parts=None, target_order=None,
                 backward_step=None, eval_cost=None):
    # run gradient_network once per learning-rate stage, chaining the best params.
    for lr in lrs:
        z = gradient_network(z, lr=lr, n_steps=nsteps, device=active_device(),
                             cost_fn=cost_fn, z_bounds=z_bounds, cost_log=cost_log,
                             step_log=step_log, float_last_parts=float_last_parts,
                             target_order=target_order,
                             backward_step=backward_step, eval_cost=eval_cost)
    return z


def _make_step_logger(session: TrainSession):
    """Build training step hooks for :func:`gradient_network`.

  Batched mode: ``cost_fn`` runs one forward + backward on the full graph.
  Sequential mode: ``backward_step`` accumulates gradients per micro-batch;
  ``eval_cost`` evaluates cost under ``no_grad`` for logging.
    """
    part_keys = session_cost_part_keys(session.target_list)
    target_history = {name: [] for name in part_keys}
    _last_parts: Optional[Dict[str, float]] = None
    _last_total: Optional[float] = None

    def _set_last(parts, total):
        nonlocal _last_parts, _last_total
        _last_parts = dict(parts)
        _last_total = float(total)

    def cost_fn(z):
        parts = calc_cost_parts(z, session)
        total = _weighted_cost_from_parts(parts, session)
        _set_last({k: float(v.item()) for k, v in parts.items()}, float(total.item()))
        return total

    def eval_cost(z):
        with torch.no_grad():
            parts = calc_cost_parts(z, session)
            total = _weighted_cost_from_parts(parts, session)
        _set_last({k: float(v.item()) for k, v in parts.items()}, float(total.item()))
        return float(total.item())

    def backward_step(z):
        total, parts_sum = backward_accum_weighted_cost(z, session)
        _set_last(parts_sum, total)
        return total

    def log_step(z=None):
        if _last_parts is None or _last_total is None:
            raise RuntimeError("log_step called before cost_fn in the same training step")
        for name in part_keys:
            if name in _last_parts:
                target_history[name].append(float(_last_parts[name]))
            else:
                target_history[name].append(0.0)
        return float(_last_total)

    def float_last_parts(target_order=None):
        if _last_parts is None:
            raise RuntimeError("float_last_parts called before cost_fn")
        return _float_parts_dict(_last_parts, target_order)

    if session.sequential:
        return cost_fn, target_history, log_step, float_last_parts, backward_step, eval_cost
    return cost_fn, target_history, log_step, float_last_parts, None, None


def do_many_runs(session: TrainSession, nofruns, nofsteps, lrs=(0.1, 0.01, 0.001),
                 z_init=None) -> TrainingResult:
    """Run ``nofruns`` independent fits; return arrays (no file I/O).

    When *z_init* is set, every round starts from ``z_init.clone()`` instead of
    ``schema_guess``.
    """
    schema = list(session.schema)
    n_params = schema_nparams(schema)
    bounds = schema_bounds(schema, session.sim_dtype)

    all_params = np.zeros((nofruns, n_params))
    final_costs = np.zeros(nofruns)
    part_keys = session_cost_part_keys(session.target_list)
    final_costs_by_target = {name: np.zeros(nofruns) for name in part_keys}
    best_i = 0
    best_cost = np.inf
    cost_curve = np.array([], dtype=np.float64)
    cost_curves_by_target = {}

    for i in range(nofruns):
        print()
        print('round', i)
        print()

        z = z_init.clone() if z_init is not None else schema_guess(schema, session.sim_dtype)
        cost_history = []
        (cost_fn, target_history, log_step, float_last_parts,
         backward_step, eval_cost) = _make_step_logger(session)

        def step_log(z):
            cost_history.append(log_step(z))

        z_fit = train_staged(
            z, cost_fn, bounds, lrs, nofsteps,
            step_log=step_log,
            float_last_parts=float_last_parts,
            target_order=list(part_keys),
            backward_step=backward_step,
            eval_cost=eval_cost,
        )

        all_params[i] = z_fit.detach().cpu().numpy()
        fit_parts = calc_cost_parts(z_fit, session)
        final_costs[i] = float(_weighted_cost_from_parts(fit_parts, session).item())
        for name, part in fit_parts.items():
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
