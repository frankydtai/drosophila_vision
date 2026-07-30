# -*- coding: utf-8 -*-
"""Parameter schema partitions: pack/unpack trainable ``z`` <-> physical params.

Segment layout, partition buckets (indi/shared/fixed/frozen), non-linear
``scale`` decoding, and the ``z``-space bounds / initial guess. ``assign_params``
turns a ``z`` vector into the per-parameter tensors consumed by
``neuron_model`` dynamics; ``params_from_z`` binds it to a session.

Model segment lists come from ``neuron_model.schema``; numeric lo/hi/init/jit
live in ``neuron_model.params.P``.
"""
from __future__ import annotations

import numpy as np
import torch

from network.connectivity import SIM_DTYPE_DEFAULT
from neuron_model import E_LEAK_DEPOL, E_LEAK_REST, IH_DIR_REVERSE_CELLS

from training.target_pack import ModelBackend, active_device


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
# Numeric lo/hi/init/jit(/fixed_val): ``neuron_model.params.P``.
# Model segment lists: ``neuron_model.schema``.
# Each segment:
#   name, kind, count, lo/hi/init/jit[, fixed_val][, scale]
#   scale: ``linear`` (default), ``log`` (z = log(physical)), or ``inv`` (z = 1/physical); lo/hi/init physical
#   indi / shared / fixed / frozen : disjoint exhaustive lists of unit indices
#       frozen: not in z; values from seg['carry'] (resume) or init (cold start)
# z layout per segment: len(indi) slots + (1 if shared else 0).
PARTITION_BUCKETS = ('indi', 'shared', 'fixed', 'frozen')
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


def _parse_init_override(text):
    """Parse ``init=L1,L2,L4,L5:200 all:10000`` -> ``{name_str: float}``."""
    out = {}
    for group in text.split():
        if ':' not in group:
            raise ValueError(f"init override group {group!r} needs NAMES:VALUE")
        names_part, val_str = group.rsplit(':', 1)
        val = float(val_str)
        for n in names_part.split(','):
            n = n.strip()
            if n:
                out[n] = val
    return out


def parse_partition_text(text):
    """Parse ``indi=all init=L1,L2,L4,L5:200 all:10000`` -> token dict."""
    text = (text or '').strip()
    buckets = {b: [] for b in PARTITION_BUCKETS}
    if not text:
        return buckets
    _KNOWN_KEYS = set(PARTITION_BUCKETS) | {'init'}
    parts = []
    buf = []
    for tok in text.split():
        if '=' in tok and tok.split('=', 1)[0] in _KNOWN_KEYS and buf:
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
        if key == 'init':
            buckets['init_override'] = _parse_init_override(rest)
            continue
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
    """Index partition -> name lists for train_opts sidecar."""
    return {b: [str(unit_names[i]) for i in part[b]] for b in PARTITION_BUCKETS}


def apply_partitions(schema, partitions_by_name, unit_names_for_seg):
    """Copy schema with resolved partitions."""
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
        if s.get('kind') == 'edge' and vals:
            if all(isinstance(x, int) for x in vals):
                n = seg_count(s)
                nonempty = [b for b in PARTITION_BUCKETS if raw.get(b)]
                if len(nonempty) != 1 or nonempty[0] == 'shared':
                    raise ValueError(
                        f"{name}: edge partitions must be a single "
                        f"indi|fixed|frozen=all bucket"
                    )
                if set(raw[nonempty[0]]) != set(range(n)):
                    raise ValueError(
                        f"{name}: edge partitions must cover all {n} edges"
                    )
            else:
                buckets = {b: [str(x) for x in (raw.get(b) or [])] for b in PARTITION_BUCKETS}
                if raw.get('init_override'):
                    buckets['init_override'] = raw['init_override']
                validate_edge_weight_partition(buckets, param_name=name)
        if not vals:
            part = {b: list(seg.get(b) or []) for b in PARTITION_BUCKETS}
        elif all(isinstance(x, int) for x in vals):
            part = {b: list(raw.get(b) or []) for b in PARTITION_BUCKETS}
        else:
            buckets = {b: [str(x) for x in (raw.get(b) or [])] for b in PARTITION_BUCKETS}
            part = resolve_partition_tokens(buckets, units, param_name=name)
        for b in PARTITION_BUCKETS:
            s[b] = part[b]
        raw_io = raw.get('init_override')
        if raw_io:
            name_to_idx = {str(u): i for i, u in enumerate(units)}
            io = {}
            all_val = raw_io.get('all')
            for cell_name, val in raw_io.items():
                if cell_name == 'all':
                    continue
                if cell_name not in name_to_idx:
                    raise ValueError(f"{name}: init override unknown unit {cell_name!r}")
                io[name_to_idx[cell_name]] = val
            if all_val is not None:
                for i in range(len(units)):
                    if i not in io:
                        io[i] = all_val
            s['init_override'] = io
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
        part = {b: list(seg.get(b) or []) for b in PARTITION_BUCKETS}
        if seg.get('kind') == 'edge':
            n = seg_count(seg)
            compact = {b: [] for b in PARTITION_BUCKETS}
            for b in PARTITION_BUCKETS:
                idxs = part[b]
                if not idxs:
                    continue
                if len(idxs) == n and set(idxs) == set(range(n)):
                    compact[b] = ['all']
                else:
                    raise ValueError(
                        f"{seg['name']}: edge partitions must be a single "
                        f"indi|fixed|frozen=all bucket (got {b}={len(idxs)}/{n})"
                    )
            rec[seg['name']] = compact
        else:
            rec[seg['name']] = partition_to_names(part, units)
    return rec


def type_unit_names(backend: "ModelBackend"):
    if backend.network is None:
        raise ValueError("type_unit_names requires backend.network")
    return [str(n) for n in backend.network.type_names]


def pair_unit_names(backend: "ModelBackend"):
    keys = backend.conn.pair_keys
    names = type_unit_names(backend)
    return [f"{names[s]}{PAIR_SEP}{names[t]}" for s, t in keys]


def edge_unit_names(backend: "ModelBackend"):
    """Opaque per-edge labels for partition resolve (``e0`` ... ``e{n-1}``)."""
    n = int(backend.conn.n_edges)
    return [f"e{i}" for i in range(n)]


def unit_names_for_segment(seg, backend: "ModelBackend"):
    kind = seg['kind']
    if kind == 'edge_pair':
        return pair_unit_names(backend)
    if kind == 'edge':
        return edge_unit_names(backend)
    return type_unit_names(backend)


def validate_edge_weight_partition(buckets, *, param_name='edge_weight'):
    """Require a single ``indi|fixed|frozen=all`` bucket (no shared / named edges)."""
    if buckets.get('init_override'):
        raise ValueError(f"{param_name}: init= overrides are not supported")
    if buckets.get('shared'):
        raise ValueError(f"{param_name}: shared= is not supported (use indi|fixed|frozen=all)")
    all_bucket = None
    for b in ('indi', 'fixed', 'frozen'):
        toks = list(buckets.get(b) or [])
        if not toks:
            continue
        if toks != ['all']:
            raise ValueError(
                f"{param_name}: only indi=all / fixed=all / frozen=all "
                f"(got {b}={','.join(toks)})"
            )
        if all_bucket is not None:
            raise ValueError(f"{param_name}: 'all' in both {all_bucket} and {b}")
        all_bucket = b
    if all_bucket is None:
        raise ValueError(f"{param_name}: need one of indi=all / fixed=all / frozen=all")
    return buckets


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
            io = seg.get('init_override')
            if io:
                carry = np.full(count, float(seg['init']), dtype=np.float64)
                for idx, val in io.items():
                    carry[int(idx)] = val
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
    out = {}
    for seg, start, stop in schema_segments(schema):
        raw = _reconstruct_raw(seg, z[start:stop], z)
        out[seg['name']] = raw.detach().cpu().numpy().astype(np.float64)
    return out


def unit_values_to_z(named, schema, *, dtype=None, device=None):
    """Pack full-width named unit values into trainable z for *schema* partitions."""
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
            slots.append(_encode_physical(seg, raw[u]))
        if seg.get('shared'):
            vals = [float(raw[u]) for u in seg['shared']]
            mean_val = float(np.mean(vals)) if vals else float(seg['init'])
            slots.append(_encode_physical(seg, mean_val))
        if slots:
            z[start:stop] = torch.tensor(slots, dtype=z.dtype, device=z.device)
    return z


def remap_named_unit_values(named, src_type_names, src_pair_names, schema, backend):
    """Remap named arrays from a prior run onto *backend* unit order for *schema*."""
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
        elif seg['kind'] == 'edge':
            if src.shape[0] == count:
                arr[:] = src
            else:
                arr[:] = float(seg['init'])
        else:
            for j, tn in enumerate(dst_types):
                if tn in src_t and src_t[tn] < src.shape[0]:
                    arr[j] = float(src[src_t[tn]])
                else:
                    arr[j] = float(seg['init']) if 'fixed_val' not in seg else fixed_val
        out[name] = arr
    return out


def _reconstruct_raw(seg, z_slice, z):
    """Build length-`count` per-unit vector from z slice + partition buckets."""
    count = seg_count(seg)
    const = _fixed_const(seg)
    raw = torch.full((count,), const, dtype=z.dtype, device=z.device)
    carry = seg.get('carry')
    i = 0
    for u in seg.get('indi', ()):
        raw[int(u)] = _decode_z(seg, z_slice[i])
        i += 1
    if seg.get('shared'):
        shared_val = _decode_z(seg, z_slice[i])
        for u in seg['shared']:
            raw[int(u)] = shared_val
    io = seg.get('init_override')
    for u in seg.get('frozen', ()):
        if carry is not None:
            raw[int(u)] = torch.tensor(float(carry[int(u)]), dtype=z.dtype, device=z.device)
        else:
            v = io.get(int(u), seg['init']) if io else seg['init']
            raw[int(u)] = torch.tensor(float(v), dtype=z.dtype, device=z.device)
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
    if kind == 'edge':
        return raw.to(dev)
    raise ValueError(f"unknown segment kind: {kind}")


def assign_params(z, schema, backend: ModelBackend):
    """Unpack z into a dict of parameter tensors, driven by the given schema partitions."""
    p = {}
    for seg, start, stop in schema_segments(schema):
        p[seg['name']] = _expand_segment(seg, _reconstruct_raw(seg, z[start:stop], z), backend)
    return p


def params_from_z(z, session):
    """Bind :func:`assign_params` to a session's schema + backend."""
    return assign_params(z, list(session.schema), session.backend)


def schema_bounds(schema, sim_dtype=SIM_DTYPE_DEFAULT):
    zb = torch.zeros((schema_nparams(schema), 2), dtype=sim_dtype)
    for seg, start, stop in schema_segments(schema):
        if stop > start:
            zlo, zhi = _z_bounds(seg)
            zb[start:stop] = torch.tensor([zlo, zhi], dtype=sim_dtype)
    return zb


def schema_guess(schema, sim_dtype=SIM_DTYPE_DEFAULT):
    z = np.zeros(schema_nparams(schema))
    for seg, start, stop in schema_segments(schema):
        n = stop - start
        if n == 0:
            continue
        io = seg.get('init_override')
        if io:
            i = 0
            for u in seg.get('indi', ()):
                phys = io.get(int(u), seg['init'])
                z[start + i] = _encode_physical(seg, phys) + (np.random.random() - 0.5) * seg['jit']
                i += 1
            if seg.get('shared'):
                z[start + i] = _encode_physical(seg, seg['init']) + (np.random.random() - 0.5) * seg['jit']
        else:
            z_init = _encode_physical(seg, seg['init'])
            z[start:stop] = z_init + (np.random.rand(n) - 0.5) * seg['jit']
    return torch.tensor(z, dtype=sim_dtype).to(active_device())


def guess_initial_params(session):
    return schema_guess(list(session.schema), session.sim_dtype)
