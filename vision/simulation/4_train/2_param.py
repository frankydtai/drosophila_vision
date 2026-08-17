# -*- coding: utf-8 -*-
"""Parameter schema param_modes: pack/unpack ``z`` <-> param.

Param packing, param_modes (indi/shared/fixed/frozen), non-linear
``z_mode`` decoding, and the ``z``-space clamps / ``z_init``. ``assign_params``
turns a ``z`` vector into the per-parameter tensors consumed by
``neuron`` dynamics; ``params_from_z`` binds it to a session.

Also owns device/dtype helpers used when materializing ``z`` and connectivity
tensors.

Model param lists come from ``neuron.schema``; numeric lo/hi/init/jit
live in ``const_default.NEURON_SCHEMA['params']``.
"""
from __future__ import annotations

from const_default import (
    NEURON_SCHEMA,
    TRAIN_OPTIMIZATION,
    TRAIN_SESSION,
    VAL_FROM,
)

import numpy as np
import torch

from neuron.schema import (
    PARAM_MODES,
    expand_param_nodes,
    parse_param_tokens,
    param_from_entry,
    resolve_modes,
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


# --- parameter schema param_modes --------------------------------------------
# Numeric lo/hi/init/jit + mode: ``const_default.NEURON_SCHEMA['params']``.
# Model param lists: ``neuron.schema``.
# Each param:
#   param, kind, n_node, lo/hi/init/jit[, z_mode]
#   z_mode: ``linear`` (default), ``log`` (z = log(param)), or ``inv`` (z = 1/param); lo/hi/init are param
#   indi / shared / fixed / frozen : disjoint exhaustive lists of node indices
#       fixed: not in z; value = effective_init(param, node)  (inits or init);
#              --init-from seeds fixed via inits_from_node_vals → inits
#       frozen: not in z; values from param['carry'] (resume) or effective_init (cold)
# z packing per param: len(indi) scalars + (1 if shared else 0).
_REMAINDER_PARAM_MODES = ('fixed', 'frozen')
_PARAM_MODE_ALL = '__all__'
PAIR_SEP = ':'


def _param_z_mode(spec, *, param="param"):
    z_mode = spec.get('z_mode', 'linear')
    if z_mode not in ('linear', 'log', 'inv'):
        raise ValueError(f"{param}: unknown z_mode {z_mode!r}")
    return z_mode


def _node_param_key(spec, node, param_key):
    """``param_key`` / ``param_key+'s'`` lookup (``init``/``inits``, ``lo``/``los``, …)."""
    node = int(node)
    by_node = spec.get(param_key + "s")
    if by_node is not None and node in by_node:
        return float(by_node[node])
    return float(spec[param_key])


def param_from_z(spec, z, *, param="param", node=None):
    """Packed ``z`` scalar -> param value. ``node`` selects per-node ``lo``/``hi``."""
    z_mode = _param_z_mode(spec, param=param)
    if z_mode == 'linear':
        return z
    if node is None:
        lo, hi = float(spec['lo']), float(spec['hi'])
    else:
        lo = _node_param_key(spec, node, "lo")
        hi = _node_param_key(spec, node, "hi")
    val = torch.exp(z) if z_mode == 'log' else 1.0 / z
    return torch.clamp(val, min=lo, max=hi)


def z_from_param(spec, val, *, param="param", node=None):
    """Param value -> packed ``z`` scalar."""
    z_mode = _param_z_mode(spec, param=param)
    if z_mode == 'linear':
        return float(val)
    if node is None:
        lo, hi = float(spec['lo']), float(spec['hi'])
    else:
        lo = _node_param_key(spec, node, "lo")
        hi = _node_param_key(spec, node, "hi")
    clipped = float(np.clip(float(val), lo, hi))
    return float(np.log(clipped)) if z_mode == 'log' else 1.0 / clipped


def _z_clamps(spec, node, *, param="param"):
    """Per-node (lo, hi) in z space."""
    lo = _node_param_key(spec, node, "lo")
    hi = _node_param_key(spec, node, "hi")
    z_mode = _param_z_mode(spec, param=param)
    if z_mode == 'log':
        return float(np.log(lo)), float(np.log(hi))
    if z_mode == 'inv':
        return 1.0 / hi, 1.0 / lo
    return lo, hi

def param_n_node(spec):
    """Full per-node w (n_cell or n_pair)."""
    return int(spec['n_node'])


def param_n_z(spec):
    """Trainable z w: one scalar per indi node + one if shared nonempty."""
    n = len(spec.get('indi', ()))
    if spec.get('shared'):
        n += 1
    return n


def schema_copy(schema):
    """Shallow-copy each schema ``spec`` (ordered ``dict[param, spec]``)."""
    return {param: dict(spec) for param, spec in schema.items()}


def schema_params(schema):
    """Yield ``(param, spec, start, stop)`` slice ranges into z."""
    start = 0
    for param, spec in schema.items():
        stop = start + param_n_z(spec)
        yield param, spec, start, stop
        start = stop


def schema_n_z(schema):
    return sum(param_n_z(spec) for spec in schema.values())


def effective_init(spec, node):
    """Per-node init: ``inits[node]`` if present, else ``spec['init']``."""
    return _node_param_key(spec, node, "init")


def resolve_param_mode_tokens(mode_tokens, *, cli_idx, param='param'):
    """Resolve token lists (with at most one ``all``) to node lists."""
    cli_ids = [str(cli_id) for cli_id in cli_idx]
    node_set = set(range(len(cli_ids)))
    explicit = {mode: [] for mode in PARAM_MODES}
    all_mode = None
    for mode in PARAM_MODES:
        tokens = list(mode_tokens.get(mode) or [])
        if 'all' in tokens:
            if len(tokens) != 1:
                raise ValueError(f"{param}: 'all' cannot mix with other ids in {mode}")
            if all_mode is not None:
                raise ValueError(f"{param}: 'all' in both {all_mode} and {mode}")
            all_mode = mode
            continue
        for token in tokens:
            if token not in cli_idx:
                raise ValueError(f"{param}: unknown id {token!r}")
            explicit[mode].append(cli_idx[token])
    claimed = []
    for mode in PARAM_MODES:
        claimed.extend(explicit[mode])
    if len(claimed) != len(set(claimed)):
        raise ValueError(f"{param}: overlapping nodes across indi/shared/fixed/frozen")
    claimed_set = set(claimed)
    leftover = sorted(node_set - claimed_set)
    if all_mode is not None:
        explicit[all_mode] = leftover
        claimed_set |= set(leftover)
    elif claimed_set != node_set:
        missing = [
            cli_id for cli_id in cli_ids if cli_idx[cli_id] not in claimed_set
        ]
        raise ValueError(
            f"{param}: nodes not assigned (use all= in one param_mode): {missing[:8]}"
            + ("..." if len(missing) > 8 else "")
        )
    return explicit


def cli_from_param_mode(modes, cli_ids):
    """Param_mode node lists → CLI id lists for train_opts sidecar."""
    return {
        mode: [str(cli_ids[cli_idx_val]) for cli_idx_val in modes[mode]]
        for mode in PARAM_MODES
    }


def _param_cli(spec, connectome):
    """``(cli_ids, cli_idx)`` for this param — cell / radius / pair / edge."""
    if spec.get("radii") is not None:
        radii = [str(radius) for radius in spec["radii"]]
        return radii, dict(zip(radii, range(len(radii))))
    kind = spec["kind"]
    if kind == "edge_pair":
        pairs = pairs_from_connectome(connectome)
        return pairs, dict(zip(pairs, range(len(pairs))))
    if kind == "edge":
        edges = edges_from_connectome(connectome)
        return edges, dict(zip(edges, range(len(edges))))
    cells = cells_from_connectome(connectome)
    return cells, dict(zip(cells, range(len(cells))))


def _stamp_param_modes(schema, param_modes, connectome):
    """Stamp CLI / sidecar ``param_modes`` into a schema copy."""
    out = {}
    for param, spec in schema.items():
        out_spec = dict(spec)
        if param not in param_modes:
            out[param] = out_spec
            continue
        cli_ids, cli_idx = _param_cli(out_spec, connectome)
        raw = param_modes[param]
        if isinstance(raw, list):
            if out_spec.get('kind') == 'edge':
                validate_syn_strength_edge_param_mode(raw, param=param)
            modes = resolve_modes(raw, cli_idx=cli_idx)
            for mode in PARAM_MODES:
                out_spec[mode] = list(modes[mode])
            out[param] = out_spec
            continue
        vals = []
        for mode in PARAM_MODES:
            vals.extend(raw.get(mode) or [])
        if out_spec.get('kind') == 'edge' and vals:
            if all(isinstance(x, int) for x in vals):
                n = param_n_node(out_spec)
                nonempty = [mode for mode in PARAM_MODES if raw.get(mode)]
                if len(nonempty) != 1 or nonempty[0] == 'shared':
                    raise ValueError(
                        f"{param}: edge param_modes must be a single "
                        f"indi|fixed|frozen=all param_mode"
                    )
                if set(raw[nonempty[0]]) != set(range(n)):
                    raise ValueError(
                        f"{param}: edge param_modes must cover all {n} edges"
                    )
            else:
                mode_tokens = {
                    mode: [str(x) for x in (raw.get(mode) or [])] for mode in PARAM_MODES
                }
                if raw.get('inits'):
                    mode_tokens['inits'] = raw['inits']
                validate_syn_strength_edge_param_mode(mode_tokens, param=param)
        if not vals:
            modes = {mode: list(spec.get(mode) or []) for mode in PARAM_MODES}
        elif all(isinstance(x, int) for x in vals):
            modes = {mode: list(raw.get(mode) or []) for mode in PARAM_MODES}
        else:
            mode_tokens = {
                mode: [str(x) for x in (raw.get(mode) or [])] for mode in PARAM_MODES
            }
            modes = resolve_param_mode_tokens(
                mode_tokens, cli_idx=cli_idx, param=param,
            )
        for mode in PARAM_MODES:
            out_spec[mode] = modes[mode]
        raw_io = raw.get('inits')
        if raw_io:
            io = {}
            all_init = raw_io.get('all')
            for cli_id, val in raw_io.items():
                if cli_id == 'all':
                    continue
                if cli_id not in cli_idx:
                    raise ValueError(f"{param}: inits unknown id {cli_id!r}")
                io[cli_idx[cli_id]] = val
            if all_init is not None:
                for node in range(len(cli_ids)):
                    if node not in io:
                        io[node] = all_init
            out_spec['inits'] = io
        shared = list(out_spec.get("shared") or ())
        if len(shared) >= 2:
            for key in ("lo", "hi", "jit"):
                plural = key + "s"
                vals = []
                for node in shared:
                    by_node = out_spec.get(plural)
                    if by_node is not None and int(node) in by_node:
                        vals.append(float(by_node[int(node)]))
                    else:
                        vals.append(float(out_spec[key]))
                if len(set(vals)) > 1:
                    raise ValueError(
                        f"{param}: shared nodes must share the same {key}, "
                        f"got {vals} for nodes {shared}"
                    )
        out[param] = out_spec
    return out


def param_modes_from_schema(schema, connectome):
    """Serialize param_modes (+ inits) as CLI id lists for train_opts.json."""
    param_modes = {}
    for param, spec in schema.items():
        cli_ids, _ = _param_cli(spec, connectome)
        modes = {mode: list(spec.get(mode) or []) for mode in PARAM_MODES}
        if spec.get('kind') == 'edge':
            n = param_n_node(spec)
            compact = {mode: [] for mode in PARAM_MODES}
            for mode in PARAM_MODES:
                idxs = modes[mode]
                if not idxs:
                    continue
                if len(idxs) == n and set(idxs) == set(range(n)):
                    compact[mode] = ['all']
                else:
                    raise ValueError(
                        f"{param}: edge param_modes must be a single "
                        f"indi|fixed|frozen=all param_mode (got {mode}={len(idxs)}/{n})"
                    )
            entry = compact
        else:
            entry = cli_from_param_mode(modes, cli_ids)
        io = spec.get('inits')
        if io:
            entry = dict(entry)
            entry['inits'] = {
                str(cli_ids[cli_idx_val]): float(val)
                for cli_idx_val, val in io.items()
            }
        param_modes[param] = entry
    return param_modes


def cells_from_connectome(connectome):
    return [str(cell) for cell in connectome.cells]


def pairs_from_connectome(connectome):
    pairs = connectome.conn.pairs
    cells = cells_from_connectome(connectome)
    return [f"{cells[source]}{PAIR_SEP}{cells[target]}" for source, target in pairs]


def edges_from_connectome(connectome):
    """Opaque per-edge CLI ids for param_mode resolve (``e0`` ... ``e{n-1}``)."""
    n = int(connectome.conn.n_edge)
    return [f"e{edge_idx}" for edge_idx in range(n)]


def validate_syn_strength_edge_param_mode(raw, *, param='syn_strength_edge'):
    """Require a single param-wide fixed or frozen param_mode."""
    if isinstance(raw, list):
        param_modes = raw
    else:
        if raw.get('inits'):
            raise ValueError(f"{param}: per-node inits are not supported")
        if raw.get('shared'):
            raise ValueError(f"{param}: shared= is not supported")
        param_modes = []
        for mode in PARAM_MODES:
            tokens = list(raw.get(mode) or [])
            if not tokens:
                continue
            if tokens != ['all']:
                raise ValueError(
                    f"{param}: only param-wide fixed or frozen "
                    f"(got {mode}={','.join(tokens)})"
                )
            param_modes.append((None, mode))
    if len(param_modes) != 1 or param_modes[0][0] is not None:
        raise ValueError(
            f"{param}: need one param-wide fixed or frozen param_mode"
        )
    if param_modes[0][1] not in ('fixed', 'frozen'):
        raise ValueError(
            f"{param}: syn_strength_edge must be fixed or frozen "
            f"(got {param_modes[0][1]!r})"
        )
    return raw


def inits_from_node_vals(schema, node_vals):
    """Stamp *node_vals* into ``inits`` for fixed nodes (not in z)."""
    out = {}
    for param, spec in schema.items():
        out_spec = dict(spec)
        fixed = list(spec.get('fixed') or [])
        if not fixed or param not in node_vals:
            out[param] = out_spec
            continue
        arr = np.asarray(node_vals[param], dtype=np.float64).reshape(-1)
        io = dict(spec.get('inits') or {})
        for node in fixed:
            node = int(node)
            if 0 <= node < arr.shape[0]:
                io[node] = float(arr[node])
        out_spec['inits'] = io
        out[param] = out_spec
    return out


def schema_with_param_carry(schema, node_vals=None):
    """Return schema copy with per-param ``carry`` arrays (full-w) for frozen nodes."""
    out = {}
    for param, spec in schema.items():
        out_spec = dict(spec)
        frozen = list(spec.get('frozen') or [])
        if not frozen:
            out_spec.pop('carry', None)
            out[param] = out_spec
            continue
        n_node = param_n_node(spec)
        if node_vals is not None and param in node_vals:
            carry = np.asarray(node_vals[param], dtype=np.float64).reshape(-1).copy()
        else:
            carry = np.asarray(
                [effective_init(spec, node) for node in range(n_node)], dtype=np.float64,
            )
        if carry.shape[0] != n_node:
            raise ValueError(
                f"{param}: carry length {carry.shape[0]} != n_node {n_node}"
            )
        out_spec['carry'] = carry
        out[param] = out_spec
    return out


def node_vals_from_z(z, schema):
    """Full-w per-node arrays (before column expand) for each param."""
    out = {}
    for param, spec, start, stop in schema_params(schema):
        raw = _reconstruct_raw(spec, z[start:stop], z)
        out[param] = raw.detach().cpu().numpy().astype(np.float64)
    return out


def z_from_node_vals(node_vals, schema, *, dtype=None, device=None):
    """Pack full-w per-param node vals into z for *schema* param_modes."""
    n = schema_n_z(schema)
    z = torch.zeros(n, dtype=dtype or SIM_DTYPE, device=device or active_device())
    for param, spec, start, stop in schema_params(schema):
        raw = np.asarray(node_vals[param], dtype=np.float64).reshape(-1)
        if raw.shape[0] != param_n_node(spec):
            raise ValueError(
                f"{param}: node_vals length {raw.shape[0]} != n_node {param_n_node(spec)}"
            )
        zs = []
        for node in spec.get('indi', ()):
            zs.append(z_from_param(spec, raw[node], param=param, node=node))
        if spec.get('shared'):
            vals = [float(raw[node]) for node in spec['shared']]
            shared_mean = float(np.mean(vals)) if vals else float(spec['init'])
            zs.append(z_from_param(
                spec, shared_mean, param=param, node=spec['shared'][0],
            ))
        if zs:
            z[start:stop] = torch.tensor(zs, dtype=z.dtype, device=z.device)
    return z


def adams_from_z(exp_avg, exp_avg_sq, schema):
    """Expand z-space adams to full-w per-param arrays (no encode/decode).

    ``indi`` z pack 1:1 onto nodes; a ``shared`` z is broadcast to every
    shared node. Fixed/frozen nodes are 0.
    """
    exp_avg = np.asarray(exp_avg, dtype=np.float64).reshape(-1)
    exp_avg_sq = np.asarray(exp_avg_sq, dtype=np.float64).reshape(-1)
    if exp_avg.shape[0] != schema_n_z(schema) or exp_avg_sq.shape[0] != exp_avg.shape[0]:
        raise ValueError(
            f"adam length {exp_avg.shape[0]}/{exp_avg_sq.shape[0]} "
            f"!= schema n_z {schema_n_z(schema)}"
        )
    adams_m = {}
    adams_v = {}
    for param, spec, start, stop in schema_params(schema):
        n_node = param_n_node(spec)
        m_arr = np.zeros(n_node, dtype=np.float64)
        v_arr = np.zeros(n_node, dtype=np.float64)
        z_idx = 0
        for node in spec.get('indi', ()):
            m_arr[int(node)] = float(exp_avg[start + z_idx])
            v_arr[int(node)] = float(exp_avg_sq[start + z_idx])
            z_idx += 1
        if spec.get('shared'):
            shared_m = float(exp_avg[start + z_idx])
            shared_v = float(exp_avg_sq[start + z_idx])
            for node in spec['shared']:
                m_arr[int(node)] = shared_m
                v_arr[int(node)] = shared_v
        adams_m[param] = m_arr
        adams_v[param] = v_arr
    return adams_m, adams_v


def z_adams_from_node_vals(adams_m, adams_v, schema, *, dtype=None, device=None):
    """Pack per-param adams into z-space tensors (no encode; shared = mean)."""
    n = schema_n_z(schema)
    dt = dtype or SIM_DTYPE
    device = device or active_device()
    exp_avg = torch.zeros(n, dtype=dt, device=device)
    exp_avg_sq = torch.zeros(n, dtype=dt, device=device)
    for param, spec, start, stop in schema_params(schema):
        m_raw = np.asarray(adams_m[param], dtype=np.float64).reshape(-1)
        v_raw = np.asarray(adams_v[param], dtype=np.float64).reshape(-1)
        if m_raw.shape[0] != param_n_node(spec) or v_raw.shape[0] != param_n_node(spec):
            raise ValueError(
                f"{param}: adam length "
                f"{m_raw.shape[0]}/{v_raw.shape[0]} != n_node {param_n_node(spec)}"
            )
        m_pack = []
        v_pack = []
        for node in spec.get('indi', ()):
            m_pack.append(float(m_raw[node]))
            v_pack.append(float(v_raw[node]))
        if spec.get('shared'):
            shared_nodes = list(spec['shared'])
            m_pack.append(
                float(np.mean([m_raw[node] for node in shared_nodes]))
                if shared_nodes else 0.0
            )
            v_pack.append(
                float(np.mean([v_raw[node] for node in shared_nodes]))
                if shared_nodes else 0.0
            )
        if m_pack:
            exp_avg[start:stop] = torch.tensor(m_pack, dtype=dt, device=device)
            exp_avg_sq[start:stop] = torch.tensor(v_pack, dtype=dt, device=device)
    return exp_avg, exp_avg_sq


def _remap_node_vals(node_vals, cells, pairs, schema, connectome, *, fill):
    """Remap per-param arrays onto connectome node order; ``fill(spec, j)`` for gaps."""
    cells = [str(cell) for cell in cells]
    pairs = [str(pair) for pair in (pairs or [])]
    connectome_cells = cells_from_connectome(connectome)
    connectome_pairs = (
        pairs_from_connectome(connectome)
        if any(spec['kind'] == 'edge_pair' for spec in schema.values())
        else []
    )
    cell_idx = dict(zip(cells, range(len(cells))))
    pair_idx = dict(zip(pairs, range(len(pairs))))
    connectome_cell_idx = dict(zip(connectome_cells, range(len(connectome_cells))))
    connectome_pair_idx = dict(zip(connectome_pairs, range(len(connectome_pairs))))
    out = {}
    for param, spec in schema.items():
        n_node = param_n_node(spec)
        arr = np.asarray([fill(spec, node) for node in range(n_node)], dtype=np.float64)
        vals = node_vals.get(param)
        if vals is None:
            out[param] = arr
            continue
        vals = np.asarray(vals, dtype=np.float64).reshape(-1)
        if spec['kind'] == 'edge_pair':
            for pair in connectome_pairs:
                if pair in pair_idx and pair_idx[pair] < vals.shape[0]:
                    arr[connectome_pair_idx[pair]] = float(vals[pair_idx[pair]])
        elif spec['kind'] == 'edge':
            if vals.shape[0] == n_node:
                arr[:] = vals
        elif spec.get("radii") is not None:
            n_copy = min(n_node, int(vals.shape[0]))
            arr[:n_copy] = vals[:n_copy]
        else:
            for cell in connectome_cells:
                if cell in cell_idx and cell_idx[cell] < vals.shape[0]:
                    arr[connectome_cell_idx[cell]] = float(vals[cell_idx[cell]])
        out[param] = arr
    return out


def remap_node_vals_adams(node_vals, cells, pairs, schema, connectome):
    """Remap per-param adam arrays; missing nodes/params fill 0."""
    return _remap_node_vals(
        node_vals, cells, pairs, schema, connectome,
        fill=lambda _spec, _node: 0.0,
    )


def remap_node_vals(node_vals, cells, pairs, schema, connectome):
    """Remap per-param arrays from a prior run onto connectome node order for *schema*."""
    return _remap_node_vals(
        node_vals, cells, pairs, schema, connectome, fill=effective_init,
    )


def _reconstruct_raw(spec, z_slice, z):
    """Build length-``n_node`` per-node vector from z slice + param_mode lists."""
    n_node = param_n_node(spec)
    raw = torch.empty((n_node,), dtype=z.dtype, device=z.device)
    carry = spec.get('carry')
    for node in spec.get('fixed', ()):
        raw[int(node)] = effective_init(spec, node)
    z_idx = 0
    for node in spec.get('indi', ()):
        raw[int(node)] = param_from_z(
            spec, z_slice[z_idx], node=node,
        )
        z_idx += 1
    if spec.get('shared'):
        shared0 = spec['shared'][0]
        shared_decoded = param_from_z(
            spec, z_slice[z_idx], node=shared0,
        )
        for node in spec['shared']:
            raw[int(node)] = shared_decoded
    for node in spec.get('frozen', ()):
        if carry is not None:
            raw[int(node)] = float(carry[int(node)])
        else:
            raw[int(node)] = effective_init(spec, node)
    return raw


def _expand_param(spec, raw, connectome):
    """Map reconstructed raw to a usable parameter tensor, per its 'kind'.

    ``node``: ``(n_cell,)`` → ``(n_node,)`` via ``conn.node_cells``.
    ``output`` / ``edge_pair`` / ``edge``: raw already full-w.
    """
    kind = spec['kind']
    device = connectome.conn.node_cells.device
    if kind == 'node':
        return raw[connectome.conn.node_cells].to(device)
    if kind in ('output', 'edge_pair', 'edge'):
        return raw.to(device)
    raise ValueError(f"unknown param kind: {kind}")


def assign_params(z, schema, connectome):
    """Unpack z into a dict of parameter tensors, driven by the given schema param_modes."""
    params = {}
    for param, spec, start, stop in schema_params(schema):
        params[param] = _expand_param(
            spec, _reconstruct_raw(spec, z[start:stop], z), connectome,
        )
    return params


def bias_gt_from_onset_trace(onset_trace, t_onset, session):
    """Per-cell-type mean of ``onset_trace[:, t_onset, :]``, clamped to ``bias_gt`` default."""
    lo = param_from_entry("bias_gt", "lo", NEURON_SCHEMA['params'])
    hi = param_from_entry("bias_gt", "hi", NEURON_SCHEMA['params'])
    t0 = int(t_onset)
    if not torch.is_tensor(onset_trace):
        onset_trace = torch.as_tensor(
            onset_trace, dtype=session.sim_dtype, device=session.device,
        )
    x = onset_trace[:, t0, :]
    node_cells = session.connectome.conn.node_cells
    n_cell = int(session.connectome.n_cell)
    onset_mean = x.mean(dim=0)
    out = onset_mean.new_empty(n_cell)
    for cell_idx in range(n_cell):
        mask = node_cells == cell_idx
        out[cell_idx] = onset_mean[mask].mean() if bool(mask.any()) else onset_mean.new_tensor(float("nan"))
    return torch.clamp(out, min=lo, max=hi)


def override_val_from(params, session, *, onset_trace=None, t_onset=None):
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
    """Bind :func:`assign_params` to a session's schema + connectome; apply ``val_from``."""
    return override_val_from(
        assign_params(z, schema_copy(session.schema), session.connectome), session,
    )


def schema_clamps(schema, sim_dtype=SIM_DTYPE):
    z_clamps = torch.zeros((schema_n_z(schema), 2), dtype=sim_dtype)
    for param, spec, start, stop in schema_params(schema):
        z_idx = 0
        for node in spec.get('indi', ()):
            z_lo, z_hi = _z_clamps(spec, node, param=param)
            z_clamps[start + z_idx] = torch.tensor([z_lo, z_hi], dtype=sim_dtype)
            z_idx += 1
        if spec.get('shared'):
            z_lo, z_hi = _z_clamps(spec, spec['shared'][0], param=param)
            z_clamps[start + z_idx] = torch.tensor([z_lo, z_hi], dtype=sim_dtype)
    return z_clamps


def z_init_from_schema(schema, sim_dtype=SIM_DTYPE):
    z = np.zeros(schema_n_z(schema))
    for param, spec, start, stop in schema_params(schema):
        n = stop - start
        if n == 0:
            continue
        z_idx = 0
        for node in spec.get('indi', ()):
            init = effective_init(spec, node)
            z[start + z_idx] = (
                z_from_param(spec, init, param=param, node=node)
                + (np.random.random() - 0.5) * _node_param_key(spec, node, "jit")
            )
            z_idx += 1
        if spec.get('shared'):
            shared0 = spec['shared'][0]
            init = effective_init(spec, shared0)
            z[start + z_idx] = (
                z_from_param(spec, init, param=param, node=shared0)
                + (np.random.random() - 0.5) * _node_param_key(spec, shared0, "jit")
            )
    return torch.tensor(z, dtype=sim_dtype).to(active_device())


def parse_param_cli(tokens, *, h_cells=NEURON_SCHEMA['h_cells']):
    """Parse ``--param`` → ``(param_inits, param_vals, param_modes, param_clamps, param_jits)``.

    ``init`` / ``lo`` / ``hi`` / ``jit`` write schema param_keys; ``val`` is
    plot/analyze only; ``mode`` sets param_modes.
    ``param_clamps`` are ``(param, lo|hi, node|None, val)``;
    ``param_jits`` are ``(param, node|None, val)``.
    """
    param_inits = []
    param_vals = []
    param_modes = {}
    param_clamps = []
    param_jits = []
    for param, param_key, nodes, right in parse_param_tokens(tokens or []):
        if param_key == "init":
            init = float(right)
            if not nodes:
                param_inits.append((param, None, init))
            else:
                for node in expand_param_nodes(nodes, h_cells=h_cells):
                    param_inits.append((param, node, init))
        elif param_key in ("lo", "hi"):
            val = float(right)
            if not nodes:
                param_clamps.append((param, param_key, None, val))
            else:
                for node in expand_param_nodes(nodes, h_cells=h_cells):
                    param_clamps.append((param, param_key, node, val))
        elif param_key == "jit":
            val = float(right)
            if not nodes:
                param_jits.append((param, None, val))
            else:
                for node in expand_param_nodes(nodes, h_cells=h_cells):
                    param_jits.append((param, node, val))
        elif param_key == "val":
            val = float(right)
            if not nodes:
                param_vals.append((param, None, val))
            else:
                for node in expand_param_nodes(nodes, h_cells=h_cells):
                    param_vals.append((param, node, val))
        elif param_key == "mode":
            if right not in PARAM_MODES:
                raise ValueError(f"unknown param_mode {right!r}")
            param_modes.setdefault(param, []).append(
                (
                    None if not nodes else expand_param_nodes(nodes, h_cells=h_cells),
                    right,
                )
            )
        else:
            raise ValueError(
                f"--param unknown param_key {param_key!r}; "
                f"expected init, lo, hi, jit, val, or mode"
            )
    return param_inits, param_vals, param_modes, param_clamps, param_jits


def parse_param_init_val_tokens(tokens, *, h_cells=NEURON_SCHEMA['h_cells']):
    """Return ``(param_inits, param_vals, param_clamps, param_jits)`` (modes ignored)."""
    param_inits, param_vals, _, param_clamps, param_jits = parse_param_cli(
        tokens, h_cells=h_cells,
    )
    return param_inits, param_vals, param_clamps, param_jits


def _param_nodes(param, node, val, spec, connectome):
    cli_ids, cli_idx = _param_cli(spec, connectome)
    if node is None:
        return cli_ids, list(range(len(cli_ids)))
    node = str(node)
    if node not in cli_idx:
        raise ValueError(f"--param {param}.{node}={val}: unknown id {node!r}")
    return [node], [cli_idx[node]]


def _stamp_param_key(schema, connectome, rows, *, key):
    """Mutate ``schema``: write ``key`` / ``key+'s'`` from ``(param, node|None, val)``."""
    plural = key + "s"
    for param, node, val in rows:
        spec = schema.get(param)
        if spec is None:
            raise ValueError(f"--param {param}: unknown param (have {sorted(schema)})")
        if node is None:
            spec[key] = float(val)
            spec.pop(plural, None)
        else:
            _, nodes = _param_nodes(param, node, val, spec, connectome)
            io = dict(spec.get(plural) or {})
            for node in nodes:
                io[int(node)] = float(val)
            spec[plural] = io
        shared = list(spec.get("shared") or ())
        if len(shared) >= 2 and key in ("lo", "hi", "jit"):
            vals = [_node_param_key(spec, node, key) for node in shared]
            if len(set(vals)) > 1:
                raise ValueError(
                    f"{param}: shared nodes must share the same {key}, "
                    f"got {vals} for nodes {shared}"
                )


def schema_with_params(
    schema, connectome, *,
    param_modes=None, param_inits=None, param_clamps=None, param_jits=None,
):
    """Write CLI / sidecar ``param_key`` fields onto a schema copy.

    ``param_modes`` rewrites indi/shared/fixed/frozen partitions.
    ``param_inits`` / ``param_clamps`` / ``param_jits`` write
    ``init`` / ``lo``|``hi`` / ``jit``.
    """
    if not (param_modes or param_inits or param_clamps or param_jits):
        return schema
    out = (
        _stamp_param_modes(schema, param_modes, connectome)
        if param_modes else schema_copy(schema)
    )
    if param_inits:
        _stamp_param_key(out, connectome, param_inits, key="init")
    if param_clamps:
        by_key = {}
        for param, key, node, val in param_clamps:
            by_key.setdefault(key, []).append((param, node, val))
        for key, rows in by_key.items():
            _stamp_param_key(out, connectome, rows, key=key)
    if param_jits:
        _stamp_param_key(out, connectome, param_jits, key="jit")
    return out


def override_params(
    z, schema, session, param_vals=None, param_inits=None,
    param_clamps=None, param_jits=None,
):
    """Apply ``--param`` overrides for plot/analyze without retraining.

    ``param_inits`` / ``param_clamps`` / ``param_jits`` write the schema;
    ``param_vals`` patches decoded vals then re-packs ``z``.
    """
    if param_inits or param_clamps or param_jits:
        schema = schema_with_params(
            schema, session.connectome,
            param_inits=param_inits, param_clamps=param_clamps, param_jits=param_jits,
        )
    else:
        schema = schema_copy(schema)
    if not param_vals:
        return z, schema
    node_vals = node_vals_from_z(z, schema)
    for param, node, val in param_vals:
        spec = schema.get(param)
        if spec is None:
            raise ValueError(f"--param {param}: unknown param (have {sorted(schema)})")
        _, nodes = _param_nodes(param, node, val, spec, session.connectome)
        arr = np.asarray(node_vals[param], dtype=np.float64).reshape(-1).copy()
        for node in nodes:
            arr[int(node)] = float(val)
        node_vals[param] = arr
        fixed = set(map(int, spec.get("fixed") or ()))
        if fixed:
            io = dict(spec.get("inits") or {})
            for node in nodes:
                if int(node) in fixed:
                    io[int(node)] = float(val)
            spec["inits"] = io
    schema = schema_with_param_carry(schema, node_vals)
    z = z_from_node_vals(
        node_vals, schema,
        dtype=session.sim_dtype, device=session.device,
    )
    return z, schema


def parse_val_from_tokens(tokens):
    out = {}
    for token in tokens or []:
        target, _, rest = token.partition("=")
        if not target or not rest:
            raise ValueError(f"--val-from expected TARGET=SOURCE:BOOL, got {token!r}")
        source, _, enabled_token = rest.partition(":")
        if not source or not enabled_token:
            raise ValueError(f"--val-from expected TARGET=SOURCE:BOOL, got {token!r}")
        out[target] = {"source": source, "enabled": enabled_token.lower() in ("1", "true", "yes")}
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


def val_from_enabled(opts, param):
    val_from = (opts or {}).get("val_from") or {}
    entry = val_from.get(param) or {}
    return bool(entry.get("enabled"))


def resolve_param_modes(param_modes, opts):
    val_from = (opts or {}).get("val_from") or {}
    if param_modes:
        for target, entry in val_from.items():
            if not entry.get("enabled"):
                continue
            if target in param_modes:
                raise ValueError(
                    f"--val-from {target} conflicts with --param mode on the same param"
                )
    out = dict(param_modes or {})
    for target, entry in val_from.items():
        if not entry.get("enabled"):
            continue
        out[target] = [(None, "frozen")]
    return out or None
