# -*- coding: utf-8 -*-
"""Parameter schema train_modes: pack/unpack trainable ``z`` <-> physical params.

Segment packing, train_modes (indi/shared/fixed/frozen), non-linear
``scale`` decoding, and the ``z``-space bounds / initial guess. ``assign_params``
turns a ``z`` vector into the per-parameter tensors consumed by
``neuron`` dynamics; ``params_from_z`` binds it to a session.

Model segment lists come from ``neuron.schema``; numeric lo/hi/init/jit
live in ``neuron.params.P``.
"""
from __future__ import annotations

import numpy as np
import torch

from training.readout_pack import ModelBackend, active_device, SIM_DTYPE
from neuron import IH_DIR_REVERSE_CELLS


def calc_multi_col_params(param, conn):
    # Broadcast a per-cell-TYPE parameter (n_cells,) to the full state (n_nodes,)
    # via the backend's node_cell.
    return param.index_select(0, conn.node_cell)


def build_e_leak(
    conn, n_cells, depol_cells=(), *, e_leak_rest: float, e_leak_depol: float,
    dtype=SIM_DTYPE,
):
    """(conn.n_nodes,) resting potential; ``depol_cells`` are type indices at ``e_leak_depol``."""
    per_type = torch.full((n_cells,), e_leak_rest, dtype=dtype, device=conn.node_cell.device)
    for c in depol_cells:
        per_type[int(c)] = e_leak_depol
    return calc_multi_col_params(per_type, conn)


def build_ih_dir(conn, ih_reverse_cells=IH_DIR_REVERSE_CELLS, *, dtype=SIM_DTYPE):
    """(conn.n_nodes,) Ih direction (+1 normal, -1 mirrored per cell)."""
    d = torch.ones(conn.n_nodes, dtype=dtype, device=conn.node_cell.device)
    for c in ih_reverse_cells:
        d[conn.node_cell == int(c)] = -1.0
    return d


# --- parameter schema train_modes --------------------------------------------
# Numeric lo/hi/init/jit(/fixed_val) + train_mode: ``param_defaults.PARAM_BOXES``.
# Model segment lists: ``neuron.schema``.
# Each segment:
#   name, kind, count, lo/hi/init/jit[, fixed_val][, scale]
#   scale: ``linear`` (default), ``log`` (z = log(physical)), or ``inv`` (z = 1/physical); lo/hi/init physical
#   indi / shared / fixed / frozen : disjoint exhaustive lists of node indices
#       fixed: fixed_value(seg, u) = fixed_val if set else effective_init(seg, u)
#       frozen: not in z; values from seg['carry'] (resume) or effective_init (cold)
# z packing per segment: len(indi) slots + (1 if shared else 0).
TRAIN_MODES = ('indi', 'shared', 'fixed', 'frozen')
PAIR_SEP = ':'


def _seg_scale(seg):
    scale = seg.get('scale', 'linear')
    if scale not in ('linear', 'log', 'inv'):
        raise ValueError(f"{seg.get('name', 'param')}: unknown scale {scale!r}")
    return scale


def _physical_bounds(seg):
    return float(seg['lo']), float(seg['hi'])


def _decode_z(seg, z_val):
    """Map trainable z slot -> physical parameter value."""
    scale = _seg_scale(seg)
    if scale == 'log':
        lo, hi = _physical_bounds(seg)
        return torch.clamp(torch.exp(z_val), min=lo, max=hi)
    if scale == 'inv':
        lo, hi = _physical_bounds(seg)
        return torch.clamp(1.0 / z_val, min=lo, max=hi)
    return z_val


def _encode_physical(seg, physical):
    """Map physical parameter value -> trainable z slot."""
    scale = _seg_scale(seg)
    if scale == 'log':
        lo, hi = _physical_bounds(seg)
        return float(np.log(np.clip(float(physical), lo, hi)))
    if scale == 'inv':
        lo, hi = _physical_bounds(seg)
        return 1.0 / float(np.clip(float(physical), lo, hi))
    return float(physical)


def _z_bounds(seg):
    """Per-slot (lo, hi) in z space."""
    lo, hi = _physical_bounds(seg)
    scale = _seg_scale(seg)
    if scale == 'log':
        return float(np.log(lo)), float(np.log(hi))
    if scale == 'inv':
        return 1.0 / hi, 1.0 / lo
    return lo, hi


def seg_count(seg):
    """Full per-node width (n_cells or n_pairs)."""
    return int(seg['count'])


def seg_ntrain(seg):
    """Trainable z width: one slot per indi node + one if shared nonempty."""
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


def effective_init(seg, u):
    """Per-node init: ``init_override[u]`` if present, else ``seg['init']``."""
    io = seg.get('init_override')
    iu = int(u)
    if io is not None and iu in io:
        return float(io[iu])
    return float(seg['init'])


def fixed_value(seg, u):
    """Value for a fixed-mode node: ``fixed_val`` if set, else ``effective_init``."""
    if 'fixed_val' in seg:
        return float(seg['fixed_val'])
    return effective_init(seg, u)


def _merge_init_override(dst, names_part, val):
    """Write ``NAMES`` / ``all`` into init_override dict from one ``=VAL`` token."""
    for n in names_part.split(','):
        n = n.strip()
        if n:
            dst[n] = val


def parse_train_mode_text(text):
    """Parse ``indi=all init.L1,L2,L4,L5=200 all=10000`` -> token dict."""
    text = (text or '').strip()
    mode_tokens = {b: [] for b in TRAIN_MODES}
    if not text:
        return mode_tokens
    init_override = {}
    for tok in text.split():
        if '=' not in tok:
            raise ValueError(f"train_mode token {tok!r} needs KEY=VALUE")
        key, rest = tok.split('=', 1)
        key = key.strip()
        if key in TRAIN_MODES:
            mode_tokens[key] = [x.strip() for x in rest.split(',') if x.strip()]
            continue
        if key == 'all':
            init_override['all'] = float(rest)
            continue
        if key.startswith('init.'):
            names_part = key[len('init.'):]
            if not names_part:
                raise ValueError(f"init override token {tok!r} needs init.NAMES=VALUE")
            _merge_init_override(init_override, names_part, float(rest))
            continue
        raise ValueError(f"unknown train_mode {key!r}")
    if init_override:
        mode_tokens['init_override'] = init_override
    return mode_tokens


def resolve_train_mode_tokens(mode_tokens, node_names, *, param_name='param'):
    """Resolve token lists (with at most one ``all``) to index lists covering node_names."""
    node_names = [str(u) for u in node_names]
    name_to_i = {n: i for i, n in enumerate(node_names)}
    all_idx = set(range(len(node_names)))
    explicit = {b: [] for b in TRAIN_MODES}
    all_mode = None
    for b in TRAIN_MODES:
        toks = list(mode_tokens.get(b) or [])
        if 'all' in toks:
            if len(toks) != 1:
                raise ValueError(f"{param_name}: 'all' cannot mix with other names in {b}")
            if all_mode is not None:
                raise ValueError(f"{param_name}: 'all' in both {all_mode} and {b}")
            all_mode = b
            continue
        for t in toks:
            if t not in name_to_i:
                raise ValueError(f"{param_name}: unknown node {t!r}")
            explicit[b].append(name_to_i[t])
    claimed = []
    for b in TRAIN_MODES:
        claimed.extend(explicit[b])
    if len(claimed) != len(set(claimed)):
        raise ValueError(f"{param_name}: overlapping nodes across indi/shared/fixed/frozen")
    claimed_set = set(claimed)
    leftover = sorted(all_idx - claimed_set)
    if all_mode is not None:
        explicit[all_mode] = leftover
        claimed_set |= set(leftover)
    elif claimed_set != all_idx:
        missing = [node_names[i] for i in sorted(all_idx - claimed_set)]
        raise ValueError(
            f"{param_name}: nodes not assigned (use all= in one train_mode): {missing[:8]}"
            + ("..." if len(missing) > 8 else "")
        )
    return {b: list(explicit[b]) for b in TRAIN_MODES}


def train_mode_to_names(mode, node_names):
    """Index train_mode -> name lists for train_opts sidecar."""
    return {b: [str(node_names[i]) for i in mode[b]] for b in TRAIN_MODES}


def apply_train_modes(schema, train_modes_by_name, node_names_for_seg):
    """Copy schema with resolved train_modes."""
    out = []
    for seg in schema:
        s = dict(seg)
        name = s['name']
        if name not in train_modes_by_name:
            out.append(s)
            continue
        if callable(node_names_for_seg):
            nodes = node_names_for_seg(s)
        else:
            nodes = node_names_for_seg[name]
        raw = train_modes_by_name[name]
        vals = []
        for b in TRAIN_MODES:
            vals.extend(raw.get(b) or [])
        if s.get('kind') == 'edge' and vals:
            if all(isinstance(x, int) for x in vals):
                n = seg_count(s)
                nonempty = [b for b in TRAIN_MODES if raw.get(b)]
                if len(nonempty) != 1 or nonempty[0] == 'shared':
                    raise ValueError(
                        f"{name}: edge train_modes must be a single "
                        f"indi|fixed|frozen=all train_mode"
                    )
                if set(raw[nonempty[0]]) != set(range(n)):
                    raise ValueError(
                        f"{name}: edge train_modes must cover all {n} edges"
                    )
            else:
                mode_tokens = {b: [str(x) for x in (raw.get(b) or [])] for b in TRAIN_MODES}
                if raw.get('init_override'):
                    mode_tokens['init_override'] = raw['init_override']
                validate_syn_strength_edge_train_mode(mode_tokens, param_name=name)
        if not vals:
            mode = {b: list(seg.get(b) or []) for b in TRAIN_MODES}
        elif all(isinstance(x, int) for x in vals):
            mode = {b: list(raw.get(b) or []) for b in TRAIN_MODES}
        else:
            mode_tokens = {b: [str(x) for x in (raw.get(b) or [])] for b in TRAIN_MODES}
            mode = resolve_train_mode_tokens(mode_tokens, nodes, param_name=name)
        for b in TRAIN_MODES:
            s[b] = mode[b]
        raw_io = raw.get('init_override')
        if raw_io:
            name_to_idx = {str(u): i for i, u in enumerate(nodes)}
            io = {}
            all_val = raw_io.get('all')
            for cell_name, val in raw_io.items():
                if cell_name == 'all':
                    continue
                if cell_name not in name_to_idx:
                    raise ValueError(f"{name}: init override unknown node {cell_name!r}")
                io[name_to_idx[cell_name]] = val
            if all_val is not None:
                for i in range(len(nodes)):
                    if i not in io:
                        io[i] = all_val
            s['init_override'] = io
        out.append(s)
    return out


def schema_train_modes_record(schema, node_names_for_seg):
    """Serialize train_modes as name lists for train_opts.json."""
    rec = {}
    for seg in schema:
        if callable(node_names_for_seg):
            nodes = node_names_for_seg(seg)
        else:
            nodes = node_names_for_seg[seg['name']]
        mode = {b: list(seg.get(b) or []) for b in TRAIN_MODES}
        if seg.get('kind') == 'edge':
            n = seg_count(seg)
            compact = {b: [] for b in TRAIN_MODES}
            for b in TRAIN_MODES:
                idxs = mode[b]
                if not idxs:
                    continue
                if len(idxs) == n and set(idxs) == set(range(n)):
                    compact[b] = ['all']
                else:
                    raise ValueError(
                        f"{seg['name']}: edge train_modes must be a single "
                        f"indi|fixed|frozen=all train_mode (got {b}={len(idxs)}/{n})"
                    )
            rec[seg['name']] = compact
        else:
            rec[seg['name']] = train_mode_to_names(mode, nodes)
    return rec


def cell_node_names(backend: "ModelBackend"):
    if backend.network is None:
        raise ValueError("cell_node_names requires backend.network")
    return [str(n) for n in backend.network.cell_names]


def pair_node_names(backend: "ModelBackend"):
    keys = backend.conn.pair_keys
    names = cell_node_names(backend)
    return [f"{names[s]}{PAIR_SEP}{names[t]}" for s, t in keys]


def edge_node_names(backend: "ModelBackend"):
    """Opaque per-edge labels for train_mode resolve (``e0`` ... ``e{n-1}``)."""
    n = int(backend.conn.n_edges)
    return [f"e{i}" for i in range(n)]


def node_names_for_segment(seg, backend: "ModelBackend"):
    kind = seg['kind']
    if kind == 'edge_pair':
        return pair_node_names(backend)
    if kind == 'edge':
        return edge_node_names(backend)
    return cell_node_names(backend)


def validate_syn_strength_edge_train_mode(mode_tokens, *, param_name='syn_strength_edge'):
    """Require a single ``indi|fixed|frozen=all`` train_mode (no shared / named edges)."""
    if mode_tokens.get('init_override'):
        raise ValueError(f"{param_name}: init. overrides are not supported")
    if mode_tokens.get('shared'):
        raise ValueError(f"{param_name}: shared= is not supported (use indi|fixed|frozen=all)")
    all_mode = None
    for b in ('indi', 'fixed', 'frozen'):
        toks = list(mode_tokens.get(b) or [])
        if not toks:
            continue
        if toks != ['all']:
            raise ValueError(
                f"{param_name}: only indi=all / fixed=all / frozen=all "
                f"(got {b}={','.join(toks)})"
            )
        if all_mode is not None:
            raise ValueError(f"{param_name}: 'all' in both {all_mode} and {b}")
        all_mode = b
    if all_mode is None:
        raise ValueError(f"{param_name}: need one of indi=all / fixed=all / frozen=all")
    return mode_tokens


def _mode_indi_all(n):
    return {'indi': list(range(n)), 'shared': [], 'fixed': [], 'frozen': []}


def _mode_shared_all(n):
    return {'indi': [], 'shared': list(range(n)), 'fixed': [], 'frozen': []}


def _mode_fixed_all(n):
    return {'indi': [], 'shared': [], 'fixed': list(range(n)), 'frozen': []}


def _mode_indi_subset_fixed_rest(n, indi_idx):
    indi_set = set(indi_idx)
    return {
        'indi': list(indi_idx),
        'shared': [],
        'fixed': [i for i in range(n) if i not in indi_set],
        'frozen': [],
    }


def attach_param_carry(schema, named=None):
    """Return schema copy with per-seg ``carry`` arrays (full width) for frozen nodes."""
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
            carry = np.asarray(
                [effective_init(seg, i) for i in range(count)], dtype=np.float64,
            )
        if carry.shape[0] != count:
            raise ValueError(
                f"{seg['name']}: carry length {carry.shape[0]} != count {count}"
            )
        s['carry'] = carry
        out.append(s)
    return out


def _with_train_mode(seg, mode):
    s = dict(seg)
    for b in TRAIN_MODES:
        s[b] = list(mode[b])
    return s


def z_to_node_values(z, schema):
    """Full-width per-node arrays (before column expand) for each segment."""
    out = {}
    for seg, start, stop in schema_segments(schema):
        raw = _reconstruct_raw(seg, z[start:stop], z)
        out[seg['name']] = raw.detach().cpu().numpy().astype(np.float64)
    return out


def node_values_to_z(named, schema, *, dtype=None, device=None):
    """Pack full-width named node values into trainable z for *schema* train_modes."""
    n = schema_nparams(schema)
    z = torch.zeros(n, dtype=dtype or SIM_DTYPE, device=device or active_device())
    for seg, start, stop in schema_segments(schema):
        raw = np.asarray(named[seg['name']], dtype=np.float64).reshape(-1)
        if raw.shape[0] != seg_count(seg):
            raise ValueError(
                f"{seg['name']}: named length {raw.shape[0]} != count {seg_count(seg)}"
            )
        slots = []
        for u in seg.get('indi', ()):
            slots.append(_encode_physical(seg, raw[u]))
        if seg.get('shared'):
            vals = [float(raw[u]) for u in seg['shared']]
            mean_val = float(np.mean(vals)) if vals else float(seg['init'])
            slots.append(_encode_physical(seg, mean_val))
        if slots:
            z[start:stop] = torch.tensor(slots, dtype=z.dtype, device=z.device)
    return z


def remap_named_node_values(named, src_cell_names, src_pair_names, schema, backend):
    """Remap named arrays from a prior run onto *backend* node order for *schema*."""
    src_cell_names = [str(n) for n in src_cell_names]
    src_pair_names = [str(n) for n in (src_pair_names or [])]
    dst_cells = cell_node_names(backend)
    dst_pairs = pair_node_names(backend) if any(s['kind'] == 'edge_pair' for s in schema) else []
    src_t = {n: i for i, n in enumerate(src_cell_names)}
    src_p = {n: i for i, n in enumerate(src_pair_names)}
    out = {}
    for seg in schema:
        name = seg['name']
        count = seg_count(seg)
        arr = np.asarray(
            [fixed_value(seg, j) for j in range(count)], dtype=np.float64,
        )
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
                    arr[j] = fixed_value(seg, j)
        elif seg['kind'] == 'edge':
            if src.shape[0] == count:
                arr[:] = src
            else:
                for j in range(count):
                    arr[j] = fixed_value(seg, j)
        else:
            for j, tn in enumerate(dst_cells):
                if tn in src_t and src_t[tn] < src.shape[0]:
                    arr[j] = float(src[src_t[tn]])
                else:
                    arr[j] = fixed_value(seg, j)
        out[name] = arr
    return out


def _reconstruct_raw(seg, z_slice, z):
    """Build length-`count` per-node vector from z slice + train_mode lists."""
    count = seg_count(seg)
    raw = torch.empty((count,), dtype=z.dtype, device=z.device)
    carry = seg.get('carry')
    for u in seg.get('fixed', ()):
        raw[int(u)] = fixed_value(seg, u)
    i = 0
    for u in seg.get('indi', ()):
        raw[int(u)] = _decode_z(seg, z_slice[i])
        i += 1
    if seg.get('shared'):
        shared_val = _decode_z(seg, z_slice[i])
        for u in seg['shared']:
            raw[int(u)] = shared_val
    for u in seg.get('frozen', ()):
        if carry is not None:
            raw[int(u)] = float(carry[int(u)])
        else:
            raw[int(u)] = effective_init(seg, u)
    return raw


def _expand_segment(seg, raw, backend: ModelBackend):
    """Map a length-`count` per-node vector to a usable parameter, per its 'kind'."""
    kind = seg['kind']
    dev = backend.conn.node_cell.device
    if kind == 'full':
        return calc_multi_col_params(raw, backend.conn).to(dev)
    if kind == 'output':
        return raw.to(dev)
    if kind == 'edge_pair':
        return raw.to(dev)
    if kind == 'edge':
        return raw.to(dev)
    raise ValueError(f"unknown segment kind: {kind}")


def assign_params(z, schema, backend: ModelBackend):
    """Unpack z into a dict of parameter tensors, driven by the given schema train_modes."""
    p = {}
    for seg, start, stop in schema_segments(schema):
        p[seg['name']] = _expand_segment(seg, _reconstruct_raw(seg, z[start:stop], z), backend)
    return p


def params_from_z(z, session):
    """Bind :func:`assign_params` to a session's schema + backend."""
    return assign_params(z, list(session.schema), session.backend)


def schema_bounds(schema, sim_dtype=SIM_DTYPE):
    zb = torch.zeros((schema_nparams(schema), 2), dtype=sim_dtype)
    for seg, start, stop in schema_segments(schema):
        if stop > start:
            zlo, zhi = _z_bounds(seg)
            zb[start:stop] = torch.tensor([zlo, zhi], dtype=sim_dtype)
    return zb


def schema_guess(schema, sim_dtype=SIM_DTYPE):
    z = np.zeros(schema_nparams(schema))
    for seg, start, stop in schema_segments(schema):
        n = stop - start
        if n == 0:
            continue
        i = 0
        for u in seg.get('indi', ()):
            phys = effective_init(seg, u)
            z[start + i] = (
                _encode_physical(seg, phys) + (np.random.random() - 0.5) * seg['jit']
            )
            i += 1
        if seg.get('shared'):
            phys = effective_init(seg, seg['shared'][0])
            z[start + i] = (
                _encode_physical(seg, phys) + (np.random.random() - 0.5) * seg['jit']
            )
    return torch.tensor(z, dtype=sim_dtype).to(active_device())


def guess_initial_params(session):
    return schema_guess(list(session.schema), session.sim_dtype)
