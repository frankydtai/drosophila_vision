# -*- coding: utf-8 -*-
"""Parameter schema param_modes: pack/unpack ``z`` <-> physical params.

Segment packing, param_modes (indi/shared/fixed/frozen), non-linear
``z_map`` decoding, and the ``z``-space bounds / initial guess. ``assign_params``
turns a ``z`` vector into the per-parameter tensors consumed by
``neuron`` dynamics; ``params_from_z`` binds it to a session.

Also owns simulation-graph ``ModelBackend`` and device/dtype helpers used when
materializing ``z`` and connectivity tensors.

Model segment lists come from ``neuron.schema``; numeric lo/hi/init/jit
live in ``neuron.param.P``.
"""
from __future__ import annotations

from default_params import (
    NEURON_SCHEMA,
    TRAIN_OPTIMIZATION,
    TRAIN_SESSION,
    VAL_FROM,
)

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from neuron import I_H_DIR_REVERSE_CELLS
from neuron.schema import (
    expand_param_nodes,
    parse_optimizable_tokens,
    optimizable_scalar,
    resolve_mode_edits,
)


def active_device():
    """Pick CUDA or CPU from current runtime (not frozen at import)."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


_FP_DTYPE = {
    16: torch.float16,
    32: torch.float32,
    64: torch.float64,
}


def sim_dtype_from_fp(fp: int) -> torch.dtype:
    """Map ``--fp`` / opts ``fp`` (16|32|64) to simulation tensor dtype."""
    try:
        return _FP_DTYPE[int(fp)]
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"fp must be 16, 32, or 64; got {fp!r}") from e


SIM_DTYPE = sim_dtype_from_fp(TRAIN_SESSION['fp'])


@dataclass(frozen=True)
class ModelBackend:
    """Connectivity + i_h tensors for one simulation graph."""

    conn: object
    i_h_dirs: torch.Tensor
    n_cells: int
    n_hexes: int
    network: Optional[object] = None

    @property
    def n_nodes(self) -> int:
        return self.conn.n_nodes


def calc_multi_col_params(param, conn):
    # Broadcast a per-cell-TYPE parameter (n_cells,) to the full state (n_nodes,)
    # via the backend's node_cells.
    return param.index_select(0, conn.node_cells)


def build_i_h_dirs(conn, i_h_reverse_cells=I_H_DIR_REVERSE_CELLS, *, dtype=SIM_DTYPE):
    """(conn.n_nodes,) i_h direction (+1 normal, -1 mirrored per cell)."""
    i_h_dirs = torch.ones(conn.n_nodes, dtype=dtype, device=conn.node_cells.device)
    for cell_idx in i_h_reverse_cells:
        i_h_dirs[conn.node_cells == int(cell_idx)] = -1.0
    return i_h_dirs


# --- parameter schema param_modes --------------------------------------------
# Numeric lo/hi/init/jit + mode: ``default_params.NEURON_SCHEMA['optimizable']``.
# Model segment lists: ``neuron.schema``.
# Each segment:
#   name, kind, n_nodes, lo/hi/init/jit[, z_map]
#   z_map: ``linear`` (default), ``log`` (z = log(physical)), or ``inv`` (z = 1/physical); lo/hi/init physical
#   indi / shared / fixed / frozen : disjoint exhaustive lists of node indices
#       fixed: not in z; value = effective_init(segment, node_idx)  (init_override or init);
#              --init-from seeds fixed via init_override_from_named → init_override
#       frozen: not in z; values from segment['carry'] (resume) or effective_init (cold)
# z packing per segment: len(indi) slots + (1 if shared else 0).
PARAM_MODES = ('indi', 'shared', 'fixed', 'frozen')
_REMAINDER_PARAM_MODES = ('fixed', 'frozen')
_PARAM_MODE_ALL = '__all__'
PAIR_SEP = ':'


def _segment_z_map(segment):
    z_map = segment.get('z_map', 'linear')
    if z_map not in ('linear', 'log', 'inv'):
        raise ValueError(f"{segment.get('name', 'param')}: unknown z_map {z_map!r}")
    return z_map


def _physical_bounds(segment):
    return float(segment['lo']), float(segment['hi'])


def _decode_z(segment, z_slot):
    """Map z slot -> physical parameter value."""
    z_map = _segment_z_map(segment)
    if z_map == 'linear':
        return z_slot
    lo, hi = _physical_bounds(segment)
    phys = torch.exp(z_slot) if z_map == 'log' else 1.0 / z_slot
    return torch.clamp(phys, min=lo, max=hi)


def _encode_physical(segment, physical):
    """Map physical parameter value -> z slot."""
    z_map = _segment_z_map(segment)
    if z_map == 'linear':
        return float(physical)
    lo, hi = _physical_bounds(segment)
    clipped = float(np.clip(float(physical), lo, hi))
    return float(np.log(clipped)) if z_map == 'log' else 1.0 / clipped


def _z_bounds(segment):
    """Per-slot (lo, hi) in z space."""
    lo, hi = _physical_bounds(segment)
    z_map = _segment_z_map(segment)
    if z_map == 'log':
        return float(np.log(lo)), float(np.log(hi))
    if z_map == 'inv':
        return 1.0 / hi, 1.0 / lo
    return lo, hi


def segment_n_nodes(segment):
    """Full per-node width (n_cells or n_pairs)."""
    return int(segment['n_nodes'])


def segment_n_z(segment):
    """Trainable z width: one slot per indi node + one if shared nonempty."""
    n = len(segment.get('indi', ()))
    if segment.get('shared'):
        n += 1
    return n


def schema_segments(schema):
    """Yield (segment, start, stop) slice ranges into z."""
    start = 0
    for segment in schema:
        stop = start + segment_n_z(segment)
        yield segment, start, stop
        start = stop


def schema_nparams(schema):
    return sum(segment_n_z(segment) for segment in schema)


def effective_init(segment, node_idx):
    """Per-node init: ``init_override[node_idx]`` if present, else ``segment['init']``.

    Fixed-mode nodes always use this (no separate fixed init path).
    """
    io = segment.get('init_override')
    node_i = int(node_idx)
    if io is not None and node_i in io:
        return float(io[node_i])
    return float(segment['init'])


def _mode_edits_from_text(text):
    edits = []
    for tok in (text or '').split():
        key, rest = tok.split('=', 1)
        if key not in PARAM_MODES:
            raise ValueError(f"unknown param_mode {key!r}")
        if rest == 'all':
            edits.append((None, key))
        else:
            edits.append((expand_param_nodes(rest), key))
    return edits


def parse_param_mode_text(text):
    return _mode_edits_from_text(text)


def resolve_param_mode_tokens(mode_tokens, node_names, *, param_name='param'):
    """Resolve token lists (with at most one ``all``) to index lists covering node_names."""
    node_names = [str(node_name) for node_name in node_names]
    i_from_name = {n: i for i, n in enumerate(node_names)}
    all_idx = set(range(len(node_names)))
    explicit = {mode_bucket: [] for mode_bucket in PARAM_MODES}
    all_mode = None
    for mode_bucket in PARAM_MODES:
        toks = list(mode_tokens.get(mode_bucket) or [])
        if 'all' in toks:
            if len(toks) != 1:
                raise ValueError(f"{param_name}: 'all' cannot mix with other names in {mode_bucket}")
            if all_mode is not None:
                raise ValueError(f"{param_name}: 'all' in both {all_mode} and {mode_bucket}")
            all_mode = mode_bucket
            continue
        for tok in toks:
            if tok not in i_from_name:
                raise ValueError(f"{param_name}: unknown node {tok!r}")
            explicit[mode_bucket].append(i_from_name[tok])
    claimed = []
    for mode_bucket in PARAM_MODES:
        claimed.extend(explicit[mode_bucket])
    if len(claimed) != len(set(claimed)):
        raise ValueError(f"{param_name}: overlapping nodes across indi/shared/fixed/frozen")
    claimed_set = set(claimed)
    leftover = sorted(all_idx - claimed_set)
    if all_mode is not None:
        explicit[all_mode] = leftover
        claimed_set |= set(leftover)
    elif claimed_set != all_idx:
        missing = [node_names[node_idx] for node_idx in sorted(all_idx - claimed_set)]
        raise ValueError(
            f"{param_name}: nodes not assigned (use all= in one param_mode): {missing[:8]}"
            + ("..." if len(missing) > 8 else "")
        )
    return {mode_bucket: list(explicit[mode_bucket]) for mode_bucket in PARAM_MODES}


def names_from_param_mode(mode, node_names):
    """Index param_mode -> name lists for train_opts sidecar."""
    return {mode_bucket: [str(node_names[node_idx]) for node_idx in mode[mode_bucket]] for mode_bucket in PARAM_MODES}


def apply_param_modes(schema, param_modes_by_name, node_names_for_segment):
    """Copy schema with resolved param_modes."""
    out = []
    for segment in schema:
        s = dict(segment)
        name = s['name']
        if name not in param_modes_by_name:
            out.append(s)
            continue
        if callable(node_names_for_segment):
            nodes = node_names_for_segment(s)
        else:
            nodes = node_names_for_segment[name]
        raw = param_modes_by_name[name]
        if isinstance(raw, list):
            i_from_name = {str(node_name): node_idx for node_idx, node_name in enumerate(nodes)}
            if s.get('kind') == 'edge':
                validate_syn_strength_edge_param_mode(raw, param_name=name)
            mode = resolve_mode_edits(raw, i_from_name)
            for mode_bucket in PARAM_MODES:
                s[mode_bucket] = list(mode[mode_bucket])
            out.append(s)
            continue
        vals = []
        for mode_bucket in PARAM_MODES:
            vals.extend(raw.get(mode_bucket) or [])
        if s.get('kind') == 'edge' and vals:
            if all(isinstance(x, int) for x in vals):
                n = segment_n_nodes(s)
                nonempty = [mode_bucket for mode_bucket in PARAM_MODES if raw.get(mode_bucket)]
                if len(nonempty) != 1 or nonempty[0] == 'shared':
                    raise ValueError(
                        f"{name}: edge param_modes must be a single "
                        f"indi|fixed|frozen=all param_mode"
                    )
                if set(raw[nonempty[0]]) != set(range(n)):
                    raise ValueError(
                        f"{name}: edge param_modes must cover all {n} edges"
                    )
            else:
                mode_tokens = {mode_bucket: [str(x) for x in (raw.get(mode_bucket) or [])] for mode_bucket in PARAM_MODES}
                if raw.get('init_override'):
                    mode_tokens['init_override'] = raw['init_override']
                validate_syn_strength_edge_param_mode(mode_tokens, param_name=name)
        if not vals:
            mode = {mode_bucket: list(segment.get(mode_bucket) or []) for mode_bucket in PARAM_MODES}
        elif all(isinstance(x, int) for x in vals):
            mode = {mode_bucket: list(raw.get(mode_bucket) or []) for mode_bucket in PARAM_MODES}
        else:
            mode_tokens = {mode_bucket: [str(x) for x in (raw.get(mode_bucket) or [])] for mode_bucket in PARAM_MODES}
            mode = resolve_param_mode_tokens(mode_tokens, nodes, param_name=name)
        for mode_bucket in PARAM_MODES:
            s[mode_bucket] = mode[mode_bucket]
        raw_io = raw.get('init_override')
        if raw_io:
            idx_from_name = {str(node_name): node_idx for node_idx, node_name in enumerate(nodes)}
            io = {}
            all_init = raw_io.get('all')
            for cell_name, val in raw_io.items():
                if cell_name == 'all':
                    continue
                if cell_name not in idx_from_name:
                    raise ValueError(f"{name}: init override unknown node {cell_name!r}")
                io[idx_from_name[cell_name]] = val
            if all_init is not None:
                for node_idx in range(len(nodes)):
                    if node_idx not in io:
                        io[node_idx] = all_init
            s['init_override'] = io
        out.append(s)
    return out


def schema_param_modes_record(schema, node_names_for_segment):
    """Serialize param_modes (+ init_override) as name lists for train_opts.json."""
    rec = {}
    for segment in schema:
        if callable(node_names_for_segment):
            nodes = node_names_for_segment(segment)
        else:
            nodes = node_names_for_segment[segment['name']]
        mode = {mode_bucket: list(segment.get(mode_bucket) or []) for mode_bucket in PARAM_MODES}
        if segment.get('kind') == 'edge':
            n = segment_n_nodes(segment)
            compact = {mode_bucket: [] for mode_bucket in PARAM_MODES}
            for mode_bucket in PARAM_MODES:
                idxs = mode[mode_bucket]
                if not idxs:
                    continue
                if len(idxs) == n and set(idxs) == set(range(n)):
                    compact[mode_bucket] = ['all']
                else:
                    raise ValueError(
                        f"{segment['name']}: edge param_modes must be a single "
                        f"indi|fixed|frozen=all param_mode (got {mode_bucket}={len(idxs)}/{n})"
                    )
            entry = compact
        else:
            entry = names_from_param_mode(mode, nodes)
        io = segment.get('init_override')
        if io:
            entry = dict(entry)
            entry['init_override'] = {
                str(nodes[int(i)]): float(v) for i, v in io.items()
            }
        rec[segment['name']] = entry
    return rec


def cell_node_names(backend: "ModelBackend"):
    if backend.network is None:
        raise ValueError("cell_node_names requires backend.network")
    return [str(n) for n in backend.network.cells]


def pair_node_names(backend: "ModelBackend"):
    keys = backend.conn.pair_keys
    names = cell_node_names(backend)
    return [f"{names[s]}{PAIR_SEP}{names[t]}" for s, t in keys]


def edge_node_names(backend: "ModelBackend"):
    """Opaque per-edge labels for param_mode resolve (``e0`` ... ``e{n-1}``)."""
    n = int(backend.conn.n_edges)
    return [f"e{edge_idx}" for edge_idx in range(n)]


def node_names_for_segment(segment, backend: "ModelBackend"):
    if segment.get('node_names') is not None:
        return [str(n) for n in segment['node_names']]
    kind = segment['kind']
    if kind == 'edge_pair':
        return pair_node_names(backend)
    if kind == 'edge':
        return edge_node_names(backend)
    return cell_node_names(backend)


def validate_syn_strength_edge_param_mode(raw, *, param_name='syn_strength_edge'):
    """Require a single segment-wide fixed or frozen param_mode."""
    if isinstance(raw, list):
        edits = raw
    else:
        if raw.get('init_override'):
            raise ValueError(f"{param_name}: val. overrides are not supported")
        if raw.get('shared'):
            raise ValueError(f"{param_name}: shared= is not supported")
        edits = []
        for mode_bucket in PARAM_MODES:
            toks = list(raw.get(mode_bucket) or [])
            if not toks:
                continue
            if toks != ['all']:
                raise ValueError(
                    f"{param_name}: only segment-wide fixed or frozen "
                    f"(got {mode_bucket}={','.join(toks)})"
                )
            edits.append((None, mode_bucket))
    if len(edits) != 1 or edits[0][0] is not None:
        raise ValueError(
            f"{param_name}: need one segment-wide fixed or frozen param_mode"
        )
    if edits[0][1] not in ('fixed', 'frozen'):
        raise ValueError(
            f"{param_name}: syn_strength_edge must be fixed or frozen "
            f"(got {edits[0][1]!r})"
        )
    return raw


def init_override_from_named(schema, named):
    """Stamp *named* values into ``init_override`` for fixed nodes (not in z)."""
    out = []
    for segment in schema:
        s = dict(segment)
        fixed = list(segment.get('fixed') or [])
        if not fixed or segment['name'] not in named:
            out.append(s)
            continue
        arr = np.asarray(named[segment['name']], dtype=np.float64).reshape(-1)
        io = dict(segment.get('init_override') or {})
        for node_idx in fixed:
            node_i = int(node_idx)
            if 0 <= node_i < arr.shape[0]:
                io[node_i] = float(arr[node_i])
        s['init_override'] = io
        out.append(s)
    return out


def attach_param_carry(schema, named=None):
    """Return schema copy with per-segment ``carry`` arrays (full width) for frozen nodes."""
    out = []
    for segment in schema:
        s = dict(segment)
        frozen = list(segment.get('frozen') or [])
        if not frozen:
            s.pop('carry', None)
            out.append(s)
            continue
        n_nodes = segment_n_nodes(segment)
        if named is not None and segment['name'] in named:
            carry = np.asarray(named[segment['name']], dtype=np.float64).reshape(-1).copy()
        else:
            carry = np.asarray(
                [effective_init(segment, node_idx) for node_idx in range(n_nodes)], dtype=np.float64,
            )
        if carry.shape[0] != n_nodes:
            raise ValueError(
                f"{segment['name']}: carry length {carry.shape[0]} != n_nodes {n_nodes}"
            )
        s['carry'] = carry
        out.append(s)
    return out


def node_values_from_z(z, schema):
    """Full-width per-node arrays (before column expand) for each segment."""
    out = {}
    for segment, start, stop in schema_segments(schema):
        raw = _reconstruct_raw(segment, z[start:stop], z)
        out[segment['name']] = raw.detach().cpu().numpy().astype(np.float64)
    return out


def z_from_node_values(named, schema, *, dtype=None, device=None):
    """Pack full-width named node values into z for *schema* param_modes."""
    n = schema_nparams(schema)
    z = torch.zeros(n, dtype=dtype or SIM_DTYPE, device=device or active_device())
    for segment, start, stop in schema_segments(schema):
        raw = np.asarray(named[segment['name']], dtype=np.float64).reshape(-1)
        if raw.shape[0] != segment_n_nodes(segment):
            raise ValueError(
                f"{segment['name']}: named length {raw.shape[0]} != n_nodes {segment_n_nodes(segment)}"
            )
        slots = []
        for node_idx in segment.get('indi', ()):
            slots.append(_encode_physical(segment, raw[node_idx]))
        if segment.get('shared'):
            vals = [float(raw[node_idx]) for node_idx in segment['shared']]
            shared_mean = float(np.mean(vals)) if vals else float(segment['init'])
            slots.append(_encode_physical(segment, shared_mean))
        if slots:
            z[start:stop] = torch.tensor(slots, dtype=z.dtype, device=z.device)
    return z


def named_moments_from_z(exp_avg, exp_avg_sq, schema):
    """Expand Adam z-space moments to full-width named arrays (no encode/decode).

    ``indi`` slots map 1:1 onto nodes; a ``shared`` slot is broadcast to every
    shared node. Fixed/frozen nodes are 0.
    """
    exp_avg = np.asarray(exp_avg, dtype=np.float64).reshape(-1)
    exp_avg_sq = np.asarray(exp_avg_sq, dtype=np.float64).reshape(-1)
    if exp_avg.shape[0] != schema_nparams(schema) or exp_avg_sq.shape[0] != exp_avg.shape[0]:
        raise ValueError(
            f"moment length {exp_avg.shape[0]}/{exp_avg_sq.shape[0]} "
            f"!= schema nparams {schema_nparams(schema)}"
        )
    named_m = {}
    named_v = {}
    for segment, start, stop in schema_segments(schema):
        n_nodes = segment_n_nodes(segment)
        m_arr = np.zeros(n_nodes, dtype=np.float64)
        v_arr = np.zeros(n_nodes, dtype=np.float64)
        z_slot_idx = 0
        for node_idx in segment.get('indi', ()):
            m_arr[int(node_idx)] = float(exp_avg[start + z_slot_idx])
            v_arr[int(node_idx)] = float(exp_avg_sq[start + z_slot_idx])
            z_slot_idx += 1
        if segment.get('shared'):
            m_s = float(exp_avg[start + z_slot_idx])
            v_s = float(exp_avg_sq[start + z_slot_idx])
            for node_idx in segment['shared']:
                m_arr[int(node_idx)] = m_s
                v_arr[int(node_idx)] = v_s
        named_m[segment['name']] = m_arr
        named_v[segment['name']] = v_arr
    return named_m, named_v


def z_moments_from_named(named_m, named_v, schema, *, dtype=None, device=None):
    """Pack named Adam moments into z-space tensors (no encode; shared = mean)."""
    n = schema_nparams(schema)
    dt = dtype or SIM_DTYPE
    dev = device or active_device()
    exp_avg = torch.zeros(n, dtype=dt, device=dev)
    exp_avg_sq = torch.zeros(n, dtype=dt, device=dev)
    for segment, start, stop in schema_segments(schema):
        m_raw = np.asarray(named_m[segment['name']], dtype=np.float64).reshape(-1)
        v_raw = np.asarray(named_v[segment['name']], dtype=np.float64).reshape(-1)
        if m_raw.shape[0] != segment_n_nodes(segment) or v_raw.shape[0] != segment_n_nodes(segment):
            raise ValueError(
                f"{segment['name']}: moment length "
                f"{m_raw.shape[0]}/{v_raw.shape[0]} != n_nodes {segment_n_nodes(segment)}"
            )
        slots_m = []
        slots_v = []
        for node_idx in segment.get('indi', ()):
            slots_m.append(float(m_raw[node_idx]))
            slots_v.append(float(v_raw[node_idx]))
        if segment.get('shared'):
            idxs = list(segment['shared'])
            slots_m.append(float(np.mean([m_raw[node_idx] for node_idx in idxs])) if idxs else 0.0)
            slots_v.append(float(np.mean([v_raw[node_idx] for node_idx in idxs])) if idxs else 0.0)
        if slots_m:
            exp_avg[start:stop] = torch.tensor(slots_m, dtype=dt, device=dev)
            exp_avg_sq[start:stop] = torch.tensor(slots_v, dtype=dt, device=dev)
    return exp_avg, exp_avg_sq


def _remap_named(named, src_cells, src_pair_names, schema, backend, *, fill):
    """Remap named arrays onto *backend* node order; ``fill(segment, j)`` for gaps."""
    src_cells = [str(n) for n in src_cells]
    src_pair_names = [str(n) for n in (src_pair_names or [])]
    dst_cells = cell_node_names(backend)
    dst_pairs = (
        pair_node_names(backend) if any(s['kind'] == 'edge_pair' for s in schema) else []
    )
    src_t = {n: node_idx for node_idx, n in enumerate(src_cells)}
    src_p = {n: node_idx for node_idx, n in enumerate(src_pair_names)}
    out = {}
    for segment in schema:
        name = segment['name']
        n_nodes = segment_n_nodes(segment)
        arr = np.asarray([fill(segment, node_idx) for node_idx in range(n_nodes)], dtype=np.float64)
        src = named.get(name)
        if src is None:
            out[name] = arr
            continue
        src = np.asarray(src, dtype=np.float64).reshape(-1)
        if segment['kind'] == 'edge_pair':
            for node_idx, pn in enumerate(dst_pairs):
                if pn in src_p and src_p[pn] < src.shape[0]:
                    arr[node_idx] = float(src[src_p[pn]])
        elif segment['kind'] == 'edge':
            if src.shape[0] == n_nodes:
                arr[:] = src
        elif segment.get('node_names') is not None:
            n_copy = min(n_nodes, int(src.shape[0]))
            arr[:n_copy] = src[:n_copy]
        else:
            for node_idx, tn in enumerate(dst_cells):
                if tn in src_t and src_t[tn] < src.shape[0]:
                    arr[node_idx] = float(src[src_t[tn]])
        out[name] = arr
    return out


def remap_named_moments(named, src_cells, src_pair_names, schema, backend):
    """Remap named moment arrays; missing nodes/params fill 0."""
    return _remap_named(
        named, src_cells, src_pair_names, schema, backend,
        fill=lambda segment, _node_idx: 0.0,
    )


def remap_named_node_values(named, src_cells, src_pair_names, schema, backend):
    """Remap named arrays from a prior run onto *backend* node order for *schema*."""
    return _remap_named(
        named, src_cells, src_pair_names, schema, backend, fill=effective_init,
    )


def _reconstruct_raw(segment, z_slice, z):
    """Build length-``n_nodes`` per-node vector from z slice + param_mode lists."""
    n_nodes = segment_n_nodes(segment)
    raw = torch.empty((n_nodes,), dtype=z.dtype, device=z.device)
    carry = segment.get('carry')
    for node_idx in segment.get('fixed', ()):
        raw[int(node_idx)] = effective_init(segment, node_idx)
    z_slot_idx = 0
    for node_idx in segment.get('indi', ()):
        raw[int(node_idx)] = _decode_z(segment, z_slice[z_slot_idx])
        z_slot_idx += 1
    if segment.get('shared'):
        shared_decoded = _decode_z(segment, z_slice[z_slot_idx])
        for node_idx in segment['shared']:
            raw[int(node_idx)] = shared_decoded
    for node_idx in segment.get('frozen', ()):
        if carry is not None:
            raw[int(node_idx)] = float(carry[int(node_idx)])
        else:
            raw[int(node_idx)] = effective_init(segment, node_idx)
    return raw


def _expand_segment(segment, raw, backend: ModelBackend):
    """Map a length-``n_nodes`` per-node vector to a usable parameter, per its 'kind'."""
    kind = segment['kind']
    dev = backend.conn.node_cells.device
    if kind == 'full':
        return calc_multi_col_params(raw, backend.conn).to(dev)
    if kind in ('output', 'edge_pair', 'edge'):
        return raw.to(dev)
    raise ValueError(f"unknown segment kind: {kind}")


def assign_params(z, schema, backend: ModelBackend):
    """Unpack z into a dict of parameter tensors, driven by the given schema param_modes."""
    params = {}
    for segment, start, stop in schema_segments(schema):
        params[segment['name']] = _expand_segment(segment, _reconstruct_raw(segment, z[start:stop], z), backend)
    return params


def bias_gt_from_onset_trace(onset_trace, t_onset, session):
    """Per-cell-type mean of ``onset_trace[:, t_onset, :]``, clamped to ``bias_gt`` default."""
    lo = optimizable_scalar("bias_gt", "lo", NEURON_SCHEMA['optimizable'])
    hi = optimizable_scalar("bias_gt", "hi", NEURON_SCHEMA['optimizable'])
    t0 = int(t_onset)
    if not torch.is_tensor(onset_trace):
        onset_trace = torch.as_tensor(
            onset_trace, dtype=session.sim_dtype, device=session.device,
        )
    x = onset_trace[:, t0, :]
    node_cells = session.backend.conn.node_cells
    n_cells = int(session.backend.n_cells)
    onset_mean = x.mean(dim=0)
    out = onset_mean.new_empty(n_cells)
    for cell_idx in range(n_cells):
        mask = node_cells == cell_idx
        out[cell_idx] = onset_mean[mask].mean() if bool(mask.any()) else onset_mean.new_tensor(float("nan"))
    return torch.clamp(out, min=lo, max=hi)


def apply_val_from(params, session, *, onset_trace=None, t_onset=None):
    """Write enabled ``val_from`` sources into target params (mutates, returns ``params``)."""
    val_from = (session.train_opts or {}).get("val_from") or {}
    for target, entry in val_from.items():
        if not entry.get("enabled") or target not in params:
            continue
        source = entry.get("source")
        if source == "v_onset":
            if onset_trace is None or t_onset is None:
                continue
            params[target] = bias_gt_from_onset_trace(onset_trace, t_onset, session)
        elif source in params:
            params[target] = params[source]
    return params


def params_from_z(z, session):
    """Bind :func:`assign_params` to a session's schema + backend; apply ``val_from``."""
    return apply_val_from(
        assign_params(z, list(session.schema), session.backend), session,
    )


def schema_bounds(schema, sim_dtype=SIM_DTYPE):
    z_bounds = torch.zeros((schema_nparams(schema), 2), dtype=sim_dtype)
    for segment, start, stop in schema_segments(schema):
        if stop > start:
            z_lo, z_hi = _z_bounds(segment)
            z_bounds[start:stop] = torch.tensor([z_lo, z_hi], dtype=sim_dtype)
    return z_bounds


def schema_guess(schema, sim_dtype=SIM_DTYPE):
    z = np.zeros(schema_nparams(schema))
    for segment, start, stop in schema_segments(schema):
        n = stop - start
        if n == 0:
            continue
        z_slot_idx = 0
        for node_idx in segment.get('indi', ()):
            phys = effective_init(segment, node_idx)
            z[start + z_slot_idx] = (
                _encode_physical(segment, phys) + (np.random.random() - 0.5) * segment['jit']
            )
            z_slot_idx += 1
        if segment.get('shared'):
            phys = effective_init(segment, segment['shared'][0])
            z[start + z_slot_idx] = (
                _encode_physical(segment, phys) + (np.random.random() - 0.5) * segment['jit']
            )
    return torch.tensor(z, dtype=sim_dtype).to(active_device())


def guess_initial_params(session):
    return schema_guess(list(session.schema), session.sim_dtype)


def parse_param_cli(tokens):
    init_edits = []
    mode_edits_by_segment_name = {}
    for segment_name, key, nodes, right in parse_optimizable_tokens(tokens or []):
        if key == "val":
            val = float(right)
            if not nodes:
                init_edits.append((segment_name, None, val))
            else:
                for node in expand_param_nodes(nodes):
                    init_edits.append((segment_name, node, val))
        elif key == "mode":
            if right not in PARAM_MODES:
                raise ValueError(f"unknown param_mode {right!r}")
            mode_edits_by_segment_name.setdefault(segment_name, []).append(
                (None if not nodes else expand_param_nodes(nodes), right)
            )
    return init_edits, mode_edits_by_segment_name


def parse_optimizable_param_tokens(tokens):
    init_edits, _ = parse_param_cli(tokens)
    return init_edits


def _param_edit_idxs(name, node, val, segment, backend):
    labels = node_names_for_segment(segment, backend)
    idx_from_label = {str(label): i for i, label in enumerate(labels)}
    if node is None:
        return labels, list(range(len(labels)))
    node = str(node)
    if node not in idx_from_label:
        raise ValueError(f"--param {name}.{node}={val}: unknown node {node!r}")
    return [node], [idx_from_label[node]]


def apply_param_init(schema, backend, init_edits):
    segment_by_name = {s["name"]: dict(s) for s in schema}
    for name, node, val in init_edits:
        segment = segment_by_name.get(name)
        if segment is None:
            avail = sorted(segment_by_name)
            raise ValueError(f"--param {name}: unknown segment (have {avail})")
        labels, idxs = _param_edit_idxs(name, node, val, segment, backend)
        io = dict(segment.get("init_override") or {})
        for label, idx in zip(labels, idxs):
            io[int(idx)] = float(val)
        segment["init_override"] = io
    return [segment_by_name[s["name"]] for s in schema]


def apply_param_overrides(z, schema, session, edits):
    schema = apply_param_init(list(schema), session.backend, edits)
    z = z_from_node_values(
        {s["name"]: np.asarray(
            [effective_init(s, node_idx) for node_idx in range(segment_n_nodes(s))], dtype=np.float64,
        ) for s in schema},
        schema,
        dtype=session.sim_dtype,
        device=session.device,
    )
    return z, schema


def parse_val_from_tokens(tokens):
    out = {}
    for tok in tokens or []:
        target, _, rest = tok.partition("=")
        if not target or not rest:
            raise ValueError(f"--val-from expected TARGET=SOURCE:BOOL, got {tok!r}")
        source, _, enabled_s = rest.partition(":")
        if not source or not enabled_s:
            raise ValueError(f"--val-from expected TARGET=SOURCE:BOOL, got {tok!r}")
        out[target] = {"source": source, "enabled": enabled_s.lower() in ("1", "true", "yes")}
    return out


def resolve_val_from(val_from=None):
    resolved = {k: dict(v) for k, v in VAL_FROM.items()}
    if not val_from:
        return resolved
    if isinstance(val_from, dict):
        for target, entry in val_from.items():
            resolved[target] = dict(entry)
        return resolved
    for target, entry in parse_val_from_tokens(val_from).items():
        resolved[target] = entry
    return resolved


def val_from_enabled(opts, name):
    val_from = (opts or {}).get("val_from") or {}
    entry = val_from.get(name) or {}
    return bool(entry.get("enabled"))


def resolve_param_modes(param_modes, opts):
    val_from = (opts or {}).get("val_from") or {}
    if param_modes:
        for target, entry in val_from.items():
            if not entry.get("enabled"):
                continue
            if target in param_modes:
                raise ValueError(
                    f"--val-from {target} conflicts with --param mode on the same segment"
                )
    out = dict(param_modes or {})
    for target, entry in val_from.items():
        if not entry.get("enabled"):
            continue
        out[target] = [(None, "frozen")]
    return out or None
