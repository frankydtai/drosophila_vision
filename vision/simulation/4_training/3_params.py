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
from neuron import I_H_DIR_REVERSE_CELLS
from param_defaults import (
    A_CA_FROM_A_OUT,
    BIAS_GT_FROM_V_ONSET,
    BIAS_GT_FROM_V_ONSET_GRAD,
    PARAM_BOXES,
    V_TH_CA_FROM_V_TH,
)


def calc_multi_col_params(param, conn):
    # Broadcast a per-cell-TYPE parameter (n_cells,) to the full state (n_nodes,)
    # via the backend's node_cell.
    return param.index_select(0, conn.node_cell)


def build_i_h_dir(conn, i_h_reverse_cells=I_H_DIR_REVERSE_CELLS, *, dtype=SIM_DTYPE):
    """(conn.n_nodes,) i_h direction (+1 normal, -1 mirrored per cell)."""
    i_h_dir = torch.ones(conn.n_nodes, dtype=dtype, device=conn.node_cell.device)
    for c in i_h_reverse_cells:
        i_h_dir[conn.node_cell == int(c)] = -1.0
    return i_h_dir


# --- parameter schema train_modes --------------------------------------------
# Numeric lo/hi/init/jit + train_mode: ``param_defaults.PARAM_BOXES``.
# Model segment lists: ``neuron.schema``.
# Each segment:
#   name, kind, count, lo/hi/init/jit[, scale]
#   scale: ``linear`` (default), ``log`` (z = log(physical)), or ``inv`` (z = 1/physical); lo/hi/init physical
#   indi / shared / fixed / frozen : disjoint exhaustive lists of node indices
#       fixed: not in z; value = effective_init(seg, node_idx)  (init_override or init);
#              --init-from seeds fixed via seed_fixed_from_named → init_override
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


def _decode_z(seg, z_slot):
    """Map trainable z slot -> physical parameter value."""
    scale = _seg_scale(seg)
    if scale == 'linear':
        return z_slot
    lo, hi = _physical_bounds(seg)
    phys = torch.exp(z_slot) if scale == 'log' else 1.0 / z_slot
    return torch.clamp(phys, min=lo, max=hi)


def _encode_physical(seg, physical):
    """Map physical parameter value -> trainable z slot."""
    scale = _seg_scale(seg)
    if scale == 'linear':
        return float(physical)
    lo, hi = _physical_bounds(seg)
    clipped = float(np.clip(float(physical), lo, hi))
    return float(np.log(clipped)) if scale == 'log' else 1.0 / clipped


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


def effective_init(seg, node_idx):
    """Per-node init: ``init_override[node_idx]`` if present, else ``seg['init']``.

    Fixed-mode nodes always use this (no separate fixed init path).
    """
    io = seg.get('init_override')
    node_i = int(node_idx)
    if io is not None and node_i in io:
        return float(io[node_i])
    return float(seg['init'])


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
            for n in names_part.split(','):
                n = n.strip()
                if n:
                    init_override[n] = float(rest)
            continue
        raise ValueError(f"unknown train_mode {key!r}")
    if init_override:
        mode_tokens['init_override'] = init_override
    return mode_tokens


def resolve_train_mode_tokens(mode_tokens, node_names, *, param_name='param'):
    """Resolve token lists (with at most one ``all``) to index lists covering node_names."""
    node_names = [str(node_name) for node_name in node_names]
    i_from_name = {n: i for i, n in enumerate(node_names)}
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
            if t not in i_from_name:
                raise ValueError(f"{param_name}: unknown node {t!r}")
            explicit[b].append(i_from_name[t])
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


def names_from_train_mode(mode, node_names):
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
            idx_from_name = {str(node_name): i for i, node_name in enumerate(nodes)}
            io = {}
            all_init = raw_io.get('all')
            for cell_name, val in raw_io.items():
                if cell_name == 'all':
                    continue
                if cell_name not in idx_from_name:
                    raise ValueError(f"{name}: init override unknown node {cell_name!r}")
                io[idx_from_name[cell_name]] = val
            if all_init is not None:
                for i in range(len(nodes)):
                    if i not in io:
                        io[i] = all_init
            s['init_override'] = io
        out.append(s)
    return out


def schema_train_modes_record(schema, node_names_for_seg):
    """Serialize train_modes (+ init_override) as name lists for train_opts.json."""
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
            entry = compact
        else:
            entry = names_from_train_mode(mode, nodes)
        io = seg.get('init_override')
        if io:
            entry = dict(entry)
            entry['init_override'] = {
                str(nodes[int(i)]): float(v) for i, v in io.items()
            }
        rec[seg['name']] = entry
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
    """Opaque per-edge labels for train_mode resolve (``e0`` ... ``e{n-1}``)."""
    n = int(backend.conn.n_edges)
    return [f"e{i}" for i in range(n)]


def node_names_for_segment(seg, backend: "ModelBackend"):
    if seg.get('node_names') is not None:
        return [str(n) for n in seg['node_names']]
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


def seed_fixed_from_named(schema, named):
    """Stamp *named* values into ``init_override`` for fixed nodes (not in z)."""
    out = []
    for seg in schema:
        s = dict(seg)
        fixed = list(seg.get('fixed') or [])
        if not fixed or seg['name'] not in named:
            out.append(s)
            continue
        arr = np.asarray(named[seg['name']], dtype=np.float64).reshape(-1)
        io = dict(seg.get('init_override') or {})
        for node_idx in fixed:
            node_i = int(node_idx)
            if 0 <= node_i < arr.shape[0]:
                io[node_i] = float(arr[node_i])
        s['init_override'] = io
        out.append(s)
    return out


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


def node_values_from_z(z, schema):
    """Full-width per-node arrays (before column expand) for each segment."""
    out = {}
    for seg, start, stop in schema_segments(schema):
        raw = _reconstruct_raw(seg, z[start:stop], z)
        out[seg['name']] = raw.detach().cpu().numpy().astype(np.float64)
    return out


def z_from_node_values(named, schema, *, dtype=None, device=None):
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
        for node_idx in seg.get('indi', ()):
            slots.append(_encode_physical(seg, raw[node_idx]))
        if seg.get('shared'):
            vals = [float(raw[node_idx]) for node_idx in seg['shared']]
            shared_mean = float(np.mean(vals)) if vals else float(seg['init'])
            slots.append(_encode_physical(seg, shared_mean))
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
    for seg, start, stop in schema_segments(schema):
        count = seg_count(seg)
        m_arr = np.zeros(count, dtype=np.float64)
        v_arr = np.zeros(count, dtype=np.float64)
        z_slot_idx = 0
        for node_idx in seg.get('indi', ()):
            m_arr[int(node_idx)] = float(exp_avg[start + z_slot_idx])
            v_arr[int(node_idx)] = float(exp_avg_sq[start + z_slot_idx])
            z_slot_idx += 1
        if seg.get('shared'):
            m_s = float(exp_avg[start + z_slot_idx])
            v_s = float(exp_avg_sq[start + z_slot_idx])
            for node_idx in seg['shared']:
                m_arr[int(node_idx)] = m_s
                v_arr[int(node_idx)] = v_s
        named_m[seg['name']] = m_arr
        named_v[seg['name']] = v_arr
    return named_m, named_v


def z_moments_from_named(named_m, named_v, schema, *, dtype=None, device=None):
    """Pack named Adam moments into z-space tensors (no encode; shared = mean)."""
    n = schema_nparams(schema)
    dt = dtype or SIM_DTYPE
    dev = device or active_device()
    exp_avg = torch.zeros(n, dtype=dt, device=dev)
    exp_avg_sq = torch.zeros(n, dtype=dt, device=dev)
    for seg, start, stop in schema_segments(schema):
        m_raw = np.asarray(named_m[seg['name']], dtype=np.float64).reshape(-1)
        v_raw = np.asarray(named_v[seg['name']], dtype=np.float64).reshape(-1)
        if m_raw.shape[0] != seg_count(seg) or v_raw.shape[0] != seg_count(seg):
            raise ValueError(
                f"{seg['name']}: moment length "
                f"{m_raw.shape[0]}/{v_raw.shape[0]} != count {seg_count(seg)}"
            )
        slots_m = []
        slots_v = []
        for node_idx in seg.get('indi', ()):
            slots_m.append(float(m_raw[node_idx]))
            slots_v.append(float(v_raw[node_idx]))
        if seg.get('shared'):
            idxs = list(seg['shared'])
            slots_m.append(float(np.mean([m_raw[node_idx] for node_idx in idxs])) if idxs else 0.0)
            slots_v.append(float(np.mean([v_raw[node_idx] for node_idx in idxs])) if idxs else 0.0)
        if slots_m:
            exp_avg[start:stop] = torch.tensor(slots_m, dtype=dt, device=dev)
            exp_avg_sq[start:stop] = torch.tensor(slots_v, dtype=dt, device=dev)
    return exp_avg, exp_avg_sq


def _remap_named(named, src_cells, src_pair_names, schema, backend, *, fill):
    """Remap named arrays onto *backend* node order; ``fill(seg, j)`` for gaps."""
    src_cells = [str(n) for n in src_cells]
    src_pair_names = [str(n) for n in (src_pair_names or [])]
    dst_cells = cell_node_names(backend)
    dst_pairs = (
        pair_node_names(backend) if any(s['kind'] == 'edge_pair' for s in schema) else []
    )
    src_t = {n: i for i, n in enumerate(src_cells)}
    src_p = {n: i for i, n in enumerate(src_pair_names)}
    out = {}
    for seg in schema:
        name = seg['name']
        count = seg_count(seg)
        arr = np.asarray([fill(seg, j) for j in range(count)], dtype=np.float64)
        src = named.get(name)
        if src is None:
            out[name] = arr
            continue
        src = np.asarray(src, dtype=np.float64).reshape(-1)
        if seg['kind'] == 'edge_pair':
            for j, pn in enumerate(dst_pairs):
                if pn in src_p and src_p[pn] < src.shape[0]:
                    arr[j] = float(src[src_p[pn]])
        elif seg['kind'] == 'edge':
            if src.shape[0] == count:
                arr[:] = src
        elif seg.get('node_names') is not None:
            n_copy = min(count, int(src.shape[0]))
            arr[:n_copy] = src[:n_copy]
        else:
            for j, tn in enumerate(dst_cells):
                if tn in src_t and src_t[tn] < src.shape[0]:
                    arr[j] = float(src[src_t[tn]])
        out[name] = arr
    return out


def remap_named_moments(named, src_cells, src_pair_names, schema, backend):
    """Remap named moment arrays; missing nodes/params fill 0."""
    return _remap_named(
        named, src_cells, src_pair_names, schema, backend,
        fill=lambda _seg, _j: 0.0,
    )


def remap_named_node_values(named, src_cells, src_pair_names, schema, backend):
    """Remap named arrays from a prior run onto *backend* node order for *schema*."""
    return _remap_named(
        named, src_cells, src_pair_names, schema, backend, fill=effective_init,
    )


def _reconstruct_raw(seg, z_slice, z):
    """Build length-`count` per-node vector from z slice + train_mode lists."""
    count = seg_count(seg)
    raw = torch.empty((count,), dtype=z.dtype, device=z.device)
    carry = seg.get('carry')
    for node_idx in seg.get('fixed', ()):
        raw[int(node_idx)] = effective_init(seg, node_idx)
    z_slot_idx = 0
    for node_idx in seg.get('indi', ()):
        raw[int(node_idx)] = _decode_z(seg, z_slice[z_slot_idx])
        z_slot_idx += 1
    if seg.get('shared'):
        shared_decoded = _decode_z(seg, z_slice[z_slot_idx])
        for node_idx in seg['shared']:
            raw[int(node_idx)] = shared_decoded
    for node_idx in seg.get('frozen', ()):
        if carry is not None:
            raw[int(node_idx)] = float(carry[int(node_idx)])
        else:
            raw[int(node_idx)] = effective_init(seg, node_idx)
    return raw


def _expand_segment(seg, raw, backend: ModelBackend):
    """Map a length-`count` per-node vector to a usable parameter, per its 'kind'."""
    kind = seg['kind']
    dev = backend.conn.node_cell.device
    if kind == 'full':
        return calc_multi_col_params(raw, backend.conn).to(dev)
    if kind in ('output', 'edge_pair', 'edge'):
        return raw.to(dev)
    raise ValueError(f"unknown segment kind: {kind}")


def assign_params(z, schema, backend: ModelBackend):
    """Unpack z into a dict of parameter tensors, driven by the given schema train_modes."""
    params = {}
    for seg, start, stop in schema_segments(schema):
        params[seg['name']] = _expand_segment(seg, _reconstruct_raw(seg, z[start:stop], z), backend)
    return params


def bias_gt_from_onset_trace(onset_trace, t_onset, session):
    """Per-cell-type mean of ``onset_trace[:, t_onset, :]``, clamped to ``bias_gt`` box."""
    opts = session.train_opts or {}
    lo = float(PARAM_BOXES["bias_gt"]["lo"])
    hi = float(PARAM_BOXES["bias_gt"]["hi"])
    t0 = int(t_onset)
    if not torch.is_tensor(onset_trace):
        onset_trace = torch.as_tensor(
            onset_trace, dtype=session.sim_dtype, device=session.device,
        )
    x = onset_trace[:, t0, :]
    if not bool(opts.get("bias_gt_from_v_onset_grad", BIAS_GT_FROM_V_ONSET_GRAD)):
        x = x.detach()
    node_cell = session.backend.conn.node_cell
    n_cells = int(session.backend.n_cells)
    onset_mean = x.mean(dim=0)
    out = onset_mean.new_empty(n_cells)
    for c in range(n_cells):
        mask = node_cell == c
        out[c] = onset_mean[mask].mean() if bool(mask.any()) else onset_mean.new_tensor(float("nan"))
    return torch.clamp(out, min=lo, max=hi)


def materialize_from_opts(params, session, *, onset_trace=None, t_onset=None):
    """Write ``*_from_*`` sources into target params on ``params`` (mutates, returns ``params``).

    * ``v_th_ca_from_v_th`` → ``params['v_th_ca'] = params['v_th']``
    * ``a_ca_from_a_out`` → ``params['a_ca'] = params['a_out']``
    * ``bias_gt_from_v_onset`` → ``params['bias_gt']`` from onset (needs ``onset_trace``)
    """
    opts = session.train_opts or {}
    if bool(opts.get("v_th_ca_from_v_th", V_TH_CA_FROM_V_TH)):
        params["v_th_ca"] = params["v_th"]
    if bool(opts.get("a_ca_from_a_out", A_CA_FROM_A_OUT)):
        params["a_ca"] = params["a_out"]
    if bool(opts.get("bias_gt_from_v_onset", BIAS_GT_FROM_V_ONSET)):
        if onset_trace is None or t_onset is None:
            return params
        params["bias_gt"] = bias_gt_from_onset_trace(onset_trace, t_onset, session)
    return params


def params_from_z(z, session):
    """Bind :func:`assign_params` to a session's schema + backend; apply ``*_from_*``."""
    return materialize_from_opts(
        assign_params(z, list(session.schema), session.backend), session,
    )


def schema_bounds(schema, sim_dtype=SIM_DTYPE):
    z_bounds = torch.zeros((schema_nparams(schema), 2), dtype=sim_dtype)
    for seg, start, stop in schema_segments(schema):
        if stop > start:
            z_lo, z_hi = _z_bounds(seg)
            z_bounds[start:stop] = torch.tensor([z_lo, z_hi], dtype=sim_dtype)
    return z_bounds


def schema_guess(schema, sim_dtype=SIM_DTYPE):
    z = np.zeros(schema_nparams(schema))
    for seg, start, stop in schema_segments(schema):
        n = stop - start
        if n == 0:
            continue
        z_slot_idx = 0
        for node_idx in seg.get('indi', ()):
            phys = effective_init(seg, node_idx)
            z[start + z_slot_idx] = (
                _encode_physical(seg, phys) + (np.random.random() - 0.5) * seg['jit']
            )
            z_slot_idx += 1
        if seg.get('shared'):
            phys = effective_init(seg, seg['shared'][0])
            z[start + z_slot_idx] = (
                _encode_physical(seg, phys) + (np.random.random() - 0.5) * seg['jit']
            )
    return torch.tensor(z, dtype=sim_dtype).to(active_device())


def guess_initial_params(session):
    return schema_guess(list(session.schema), session.sim_dtype)
