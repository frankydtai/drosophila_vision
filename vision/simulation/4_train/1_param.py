# -*- coding: utf-8 -*-
"""Parameter schema param_modes: pack/unpack ``z`` <-> param.

Param packing, param_modes (indi/shared/fixed/frozen), non-linear
``z_mode`` decoding, and the ``z``-space clamps / ``z_init``. ``assign_params``
turns a ``z`` vector into the per-parameter tensors consumed by
``neuron`` dynamics; ``params_from_z`` binds it to a session.

Also owns device/dtype helpers used when materializing ``z`` and connectivity
tensors.

Model param lists come from ``neuron.schema``; numeric lo/hi/init/jit
live in ``config.NEURON_SCHEMA['params']``.
"""
from __future__ import annotations

from config import (
    NEURON_SCHEMA,
    TRAIN_SESSION,
    VAL_FROM,
)

import numpy as np
import torch

from neuron.schema import (
    PARAM_MODES,
    param_from_entry,
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


PAIR_SEP = ':'


def _param_z_mode(spec, *, param="param"):
    z_mode = spec.get('z_mode', 'linear')
    if z_mode not in ('linear', 'log', 'inv'):
        raise ValueError(f"{param}: unknown z_mode {z_mode!r}")
    return z_mode


def _node_param_key(spec, node, param_key):
    """``param_key`` / ``param_key+'s'`` lookup (``init``/``inits``, ``lo``/``los``, …)."""
    node = int(node)
    if param_key + "s" in spec and node in spec[param_key + "s"]:
        return float(spec[param_key + "s"][node])
    return float(spec[param_key])


def _lo_hi(spec, node=None):
    if node is None:
        return float(spec['lo']), float(spec['hi'])
    return _node_param_key(spec, node, "lo"), _node_param_key(spec, node, "hi")


def param_from_z(spec, z, *, param="param", node=None):
    """Packed ``z`` scalar -> param value. ``node`` selects per-node ``lo``/``hi``."""
    z_mode = _param_z_mode(spec, param=param)
    if z_mode == 'linear':
        return z
    lo, hi = _lo_hi(spec, node)
    return torch.clamp(
        torch.exp(z) if z_mode == 'log' else 1.0 / z,
        min=lo,
        max=hi,
    )


def z_from_param(spec, val, *, param="param", node=None):
    """Param value -> packed ``z`` scalar."""
    z_mode = _param_z_mode(spec, param=param)
    if z_mode == 'linear':
        return float(val)
    lo, hi = _lo_hi(spec, node)
    clipped = float(np.clip(float(val), lo, hi))
    return float(np.log(clipped)) if z_mode == 'log' else 1.0 / clipped


def _z_clamps(spec, node, *, param="param"):
    """Per-node (lo, hi) in z space."""
    lo, hi = _lo_hi(spec, node)
    z_mode = _param_z_mode(spec, param=param)
    if z_mode == 'log':
        return float(np.log(lo)), float(np.log(hi))
    if z_mode == 'inv':
        return 1.0 / hi, 1.0 / lo
    return lo, hi


def param_n_z(spec):
    """Trainable z w: one scalar per indi node + one if shared nonempty."""
    n = len(spec.get('indi', ()))
    if spec.get('shared'):
        n += 1
    return n


def _z_pack_slots(spec):
    """Yield ``(z_idx, nodes)`` for each packed z scalar (indi 1:1, shared broadcast)."""
    z_idx = 0
    for node in spec.get('indi', ()):
        yield z_idx, (int(node),)
        z_idx += 1
    shared = spec.get('shared') or ()
    if shared:
        yield z_idx, tuple(int(node) for node in shared)


def _slot_val(node_vals, nodes):
    if len(nodes) == 1:
        return float(node_vals[nodes[0]])
    return float(np.mean([node_vals[node] for node in nodes]))


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


def _param_names(param, spec, connectome):
    """``(names, node_from)`` for this param — cell / radius / pair / edge."""
    if spec.get("radii") is not None:
        names = [str(radius) for radius in spec["radii"]]
    elif param == "syn_strength_cell":
        names = pairs_from_connectome(connectome)
    elif param == "syn_strength_edge":
        names = edges_from_connectome(connectome)
    else:
        names = cells_from_connectome(connectome)
    return names, dict(zip(names, range(len(names))))


def cells_from_connectome(connectome):
    return [str(cell) for cell in connectome.cells]


def pairs_from_connectome(connectome):
    pairs = connectome.conn.pairs
    cells = cells_from_connectome(connectome)
    return [f"{cells[source]}{PAIR_SEP}{cells[target]}" for source, target in pairs]


def edges_from_connectome(connectome):
    """Per-edge names ``e0`` ... ``e{n-1}``."""
    return [f"e{edge_idx}" for edge_idx in range(int(connectome.conn.n_edge))]


def inits_from_node_vals(schema, node_vals):
    """Stamp *node_vals* into ``inits`` for fixed nodes (not in z)."""
    schema = schema_copy(schema)
    for param, spec in schema.items():
        fixed = list(spec.get('fixed') or [])
        if not fixed or param not in node_vals:
            continue
        node_vals[param] = np.asarray(node_vals[param], dtype=np.float64).reshape(-1)
        inits = dict(spec.get('inits') or {})
        for node in fixed:
            node = int(node)
            if 0 <= node < node_vals[param].shape[0]:
                inits[node] = float(node_vals[param][node])
        spec['inits'] = inits
    return schema


def schema_with_param_carry(schema, node_vals=None):
    """Return schema copy with per-param ``carry`` (full-w) for frozen nodes."""
    schema = schema_copy(schema)
    for param, spec in schema.items():
        frozen = list(spec.get('frozen') or [])
        if not frozen:
            spec.pop('carry', None)
            continue
        n_node = spec['n_node']
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
        spec['carry'] = carry
    return schema


def node_vals_from_z(z, schema):
    """Full-w per-node vals (before column expand) for each param."""
    return {
        param: _vals_from_z_slice(spec, z[start:stop]).detach().cpu().numpy().astype(np.float64)
        for param, spec, start, stop in schema_params(schema)
    }


def z_from_node_vals(node_vals, schema, *, dtype=None, device=None):
    """Pack full-w per-param node vals into z for *schema* param_modes."""
    n = schema_n_z(schema)
    z = torch.zeros(n, dtype=dtype or SIM_DTYPE, device=device or active_device())
    for param, spec, start, stop in schema_params(schema):
        node_vals[param] = np.asarray(node_vals[param], dtype=np.float64).reshape(-1)
        n_node = spec['n_node']
        if node_vals[param].shape[0] != n_node:
            raise ValueError(
                f"{param}: node_vals length {node_vals[param].shape[0]} != n_node {n_node}"
            )
        zs = [
            z_from_param(spec, _slot_val(node_vals[param], nodes), param=param, node=nodes[0])
            for _, nodes in _z_pack_slots(spec)
        ]
        if zs:
            z[start:stop] = torch.tensor(zs, dtype=z.dtype, device=z.device)
    return z


def adams_from_z(exp_avg, exp_avg_sq, schema):
    """Expand z-space adams to full-w per-param adams (no encode/decode).

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
        n_node = spec['n_node']
        adams_m[param] = np.zeros(n_node, dtype=np.float64)
        adams_v[param] = np.zeros(n_node, dtype=np.float64)
        for z_idx, nodes in _z_pack_slots(spec):
            m = float(exp_avg[start + z_idx])
            v = float(exp_avg_sq[start + z_idx])
            for node in nodes:
                adams_m[param][node] = m
                adams_v[param][node] = v
    return adams_m, adams_v


def z_adams_from_node_vals(adams_m, adams_v, schema, *, dtype=None, device=None):
    """Pack per-param adams into z-space tensors (no encode; shared = mean)."""
    n = schema_n_z(schema)
    dt = dtype or SIM_DTYPE
    device = device or active_device()
    exp_avg = torch.zeros(n, dtype=dt, device=device)
    exp_avg_sq = torch.zeros(n, dtype=dt, device=device)
    for param, spec, start, stop in schema_params(schema):
        m = np.asarray(adams_m[param], dtype=np.float64).reshape(-1)
        v = np.asarray(adams_v[param], dtype=np.float64).reshape(-1)
        n_node = spec['n_node']
        if m.shape[0] != n_node or v.shape[0] != n_node:
            raise ValueError(
                f"{param}: adam length "
                f"{m.shape[0]}/{v.shape[0]} != n_node {n_node}"
            )
        m_pack = []
        v_pack = []
        for _, nodes in _z_pack_slots(spec):
            m_pack.append(_slot_val(m, nodes))
            v_pack.append(_slot_val(v, nodes))
        if m_pack:
            exp_avg[start:stop] = torch.tensor(m_pack, dtype=dt, device=device)
            exp_avg_sq[start:stop] = torch.tensor(v_pack, dtype=dt, device=device)
    return exp_avg, exp_avg_sq


def _remap_node_vals(node_vals, cells, pairs, schema, connectome, *, fill):
    """Remap per-param vals onto connectome node order; ``fill(spec, j)`` for gaps."""
    cells = [str(cell) for cell in cells]
    pairs = [str(pair) for pair in (pairs or [])]
    connectome_cells = cells_from_connectome(connectome)
    connectome_pairs = (
        pairs_from_connectome(connectome)
        if "syn_strength_cell" in schema
        else []
    )
    cell_idx = dict(zip(cells, range(len(cells))))
    pair_idx = dict(zip(pairs, range(len(pairs))))
    connectome_cell_idx = dict(zip(connectome_cells, range(len(connectome_cells))))
    connectome_pair_idx = dict(zip(connectome_pairs, range(len(connectome_pairs))))

    node_vals_remapped = {
        param: np.asarray(
            [fill(spec, node) for node in range(spec['n_node'])], dtype=np.float64,
        )
        for param, spec in schema.items()
    }
    for param, spec in schema.items():
        if node_vals.get(param) is None:
            continue
        n_node = spec['n_node']
        if param == "syn_strength_cell":
            for pair in connectome_pairs:
                if pair in pair_idx and pair_idx[pair] < len(node_vals[param]):
                    node_vals_remapped[param][connectome_pair_idx[pair]] = float(
                        np.asarray(node_vals[param], dtype=np.float64).reshape(-1)[
                            pair_idx[pair]
                        ],
                    )
        elif param == "syn_strength_edge":
            if len(np.asarray(node_vals[param], dtype=np.float64).reshape(-1)) == n_node:
                node_vals_remapped[param][:] = np.asarray(
                    node_vals[param], dtype=np.float64,
                ).reshape(-1)
        elif spec.get("radii") is not None:
            n_copy = min(
                n_node,
                len(np.asarray(node_vals[param], dtype=np.float64).reshape(-1)),
            )
            node_vals_remapped[param][:n_copy] = np.asarray(
                node_vals[param], dtype=np.float64,
            ).reshape(-1)[:n_copy]
        else:
            for cell in connectome_cells:
                if cell in cell_idx and cell_idx[cell] < len(node_vals[param]):
                    node_vals_remapped[param][connectome_cell_idx[cell]] = float(
                        np.asarray(node_vals[param], dtype=np.float64).reshape(-1)[
                            cell_idx[cell]
                        ],
                    )
    return node_vals_remapped


def remap_node_vals_adams(node_vals, cells, pairs, schema, connectome):
    """Remap per-param adams; missing nodes/params fill 0."""
    return _remap_node_vals(
        node_vals, cells, pairs, schema, connectome,
        fill=lambda _spec, _node: 0.0,
    )


def remap_node_vals(node_vals, cells, pairs, schema, connectome):
    """Remap per-param vals from a prior run onto connectome node order for *schema*."""
    return _remap_node_vals(
        node_vals, cells, pairs, schema, connectome, fill=effective_init,
    )


def _vals_from_z_slice(spec, z_slice):
    """Build length-``n_node`` per-node vals from z slice + param_mode lists."""
    n_node = spec['n_node']
    node_vals = torch.empty((n_node,), dtype=z_slice.dtype, device=z_slice.device)
    carry = spec.get('carry')
    for node in spec.get('fixed', ()):
        node_vals[int(node)] = effective_init(spec, node)
    for z_idx, nodes in _z_pack_slots(spec):
        val = param_from_z(spec, z_slice[z_idx], node=nodes[0])
        for node in nodes:
            node_vals[node] = val
    for node in spec.get('frozen', ()):
        node_vals[int(node)] = (
            float(carry[int(node)]) if carry is not None else effective_init(spec, node)
        )
    return node_vals


def _expand_param(param, spec, node_vals, connectome):
    """Map packed vals to the tensor consumed by dynamics.

    Per-cell neuron params: ``(n_cell,)`` → ``(n_node,)`` via ``conn.node_cells``.
    ``a_gt`` / ``bias_gt`` stay ``(n_cell,)``; syn / ``a_sti_radius`` stay full-w.
    """
    device = connectome.conn.node_cells.device
    if (
        param in ("a_gt", "bias_gt", "syn_strength_cell", "syn_strength_edge")
        or spec.get("radii") is not None
    ):
        return node_vals.to(device)
    return node_vals[connectome.conn.node_cells].to(device)


def assign_params(z, schema, connectome):
    """Unpack z into a dict of parameter tensors, driven by the given schema param_modes."""
    params = {}
    for param, spec, start, stop in schema_params(schema):
        params[param] = _expand_param(
            param, spec, _vals_from_z_slice(spec, z[start:stop]), connectome,
        )
    return params


def bias_gt_from_onset_trace(onset_trace, t_onset, session):
    """Per-cell-type mean of ``onset_trace[:, t_onset, :]``, clamped to ``bias_gt`` default."""
    lo = param_from_entry("bias_gt", "lo", NEURON_SCHEMA['params'])
    hi = param_from_entry("bias_gt", "hi", NEURON_SCHEMA['params'])
    if not torch.is_tensor(onset_trace):
        onset_trace = torch.as_tensor(
            onset_trace, dtype=session.sim_dtype, device=session.device,
        )
    node_cells = session.connectome.conn.node_cells
    onset_mean = onset_trace[:, int(t_onset), :].mean(dim=0)
    bias_gt = onset_mean.new_empty(int(session.connectome.n_cell))
    for cell_idx in range(int(session.connectome.n_cell)):
        mask = node_cells == cell_idx
        bias_gt[cell_idx] = onset_mean[mask].mean() if bool(mask.any()) else onset_mean.new_tensor(float("nan"))
    return torch.clamp(bias_gt, min=lo, max=hi)


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
        for z_idx, nodes in _z_pack_slots(spec):
            z_clamps[start + z_idx] = torch.tensor(
                _z_clamps(spec, nodes[0], param=param), dtype=sim_dtype,
            )
    return z_clamps


def z_init_from_schema(schema, sim_dtype=SIM_DTYPE):
    z = np.zeros(schema_n_z(schema))
    for param, spec, start, stop in schema_params(schema):
        for z_idx, nodes in _z_pack_slots(spec):
            node = nodes[0]
            z[start + z_idx] = (
                z_from_param(spec, effective_init(spec, node), param=param, node=node)
                + (np.random.random() - 0.5) * _node_param_key(spec, node, "jit")
            )
    return torch.tensor(z, dtype=sim_dtype).to(active_device())


def override_params(z, schema, session, param_vals=None):
    """Patch decoded vals from nested ``param_vals`` then re-pack ``z``.

    ``param_vals`` is ``{param: {name: float}}`` or ``{param: float}`` (all nodes).
    """
    schema = schema_copy(schema)
    if not param_vals:
        return z, schema
    node_vals = node_vals_from_z(z, schema)
    for param, bag in param_vals.items():
        spec = schema.get(param)
        if spec is None:
            raise ValueError(f"param_vals {param}: unknown param (have {sorted(schema)})")
        _, node_from = _param_names(param, spec, session.connectome)
        if isinstance(bag, dict):
            rows = []
            for name, val in bag.items():
                name = str(name)
                if name not in node_from:
                    raise ValueError(
                        f"param_vals {param}.{name}={val}: unknown id {name!r}"
                    )
                rows.append((node_from[name], float(val)))
        else:
            rows = [(node, float(bag)) for node in range(len(node_from))]
        node_vals[param] = np.asarray(node_vals[param], dtype=np.float64).reshape(-1).copy()
        for node, val in rows:
            node_vals[param][int(node)] = val
        fixed = set(map(int, spec.get("fixed") or ()))
        if fixed:
            inits = dict(spec.get("inits") or {})
            for node, val in rows:
                if int(node) in fixed:
                    inits[int(node)] = val
            spec["inits"] = inits
    schema = schema_with_param_carry(schema, node_vals)
    z = z_from_node_vals(
        node_vals, schema,
        dtype=session.sim_dtype, device=session.device,
    )
    return z, schema


def resolve_val_from(val_from=None):
    merged = {param: dict(entry) for param, entry in VAL_FROM.items()}
    if not val_from:
        return merged
    if not isinstance(val_from, dict):
        raise ValueError(f"val_from must be a config dict, got {type(val_from).__name__}")
    merged.update({target: dict(entry) for target, entry in val_from.items()})
    return merged


def val_from_enabled(opts, param):
    entry = ((opts or {}).get("val_from") or {}).get(param) or {}
    return bool(entry.get("enabled"))
