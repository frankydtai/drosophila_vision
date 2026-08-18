# -*- coding: utf-8 -*-
"""Parameter schemas for borst / hp_lp neuron models.

Numeric lo/hi/init/jit, leftover ``mode``, optional per-node ``inits`` /
``los`` / ``his`` / ``jits``, and optional ``modes`` (name lists keyed by
mode token) live in ``config.NEURON_SCHEMA['params']``. Unlisted nodes
take leftover ``mode``. Default ``model`` scalar lives in
``config.NEURON_SCHEMA['model']``.

Built schema is an ordered ``dict[param, spec]`` — the param name is the
dict key; ``spec`` has no self-id field. Per-node bags on ``spec`` are
``int`` node → float (or mode → list of node).
"""
from __future__ import annotations

SYN_MODES = ("per_cell", "per_edge")
PARAM_MODES = ("indi", "shared", "fixed", "frozen")
_PARAM_ENTRY_KEYS = frozenset({
    "lo", "hi", "init", "jit", "mode", "inits", "los", "his", "jits", "modes",
    "z_mode",
})

I_H_SHAPE_PARAMS = (
    "v_mid_h_g", "v_mid_h_tau", "h_slope",
    "v_mid_h_g_rev", "v_mid_h_tau_rev", "h_slope_rev",
)
_HP_LP_ONLY = frozenset({"tau_lp", "tau_hp_rise", "tau_hp_fall"})
_I_H_REV_ONLY = frozenset({
    "a_h_rev", "v_mid_h_g_rev", "v_mid_h_tau_rev", "h_slope_rev",
})
_BORST_ONLY = frozenset({"v_mid_h_g", "v_mid_h_tau", "h_slope"}) | _I_H_REV_ONLY
_CA_ONLY = frozenset({"v_th_ca", "a_ca", "tau_ca"})


def param_from_entry(param, param_key, params):
    """``params[param][param_key]`` from a NEURON_SCHEMA entry."""
    return float(params[param][param_key])


def _axis_names(param, n_node, *, cells=None, radii=None, pairs=None, edges=None):
    n_axis = sum(x is not None for x in (cells, radii, pairs, edges))
    if n_axis > 1:
        raise ValueError(
            f"{param}: pass at most one of cells, radii, pairs, edges"
        )
    if cells is not None:
        names = [str(cell) for cell in cells]
    elif radii is not None:
        names = [str(radius) for radius in radii]
    elif pairs is not None:
        names = [str(pair) for pair in pairs]
    elif edges is not None:
        names = [str(edge) for edge in edges]
    else:
        names = [str(i) for i in range(n_node)]
    if len(names) != n_node:
        raise ValueError(f"{param}: {len(names)} names != n_node {n_node}")
    return names


def _node_vals_from_bag(bag, names, param, key):
    if not bag:
        return {}
    node_from = dict(zip(names, range(len(names))))
    vals = {}
    for name, val in bag.items():
        name = str(name)
        if name not in node_from:
            raise ValueError(f"{param}: {key} unknown id {name!r}")
        vals[node_from[name]] = float(val)
    return vals


def build_param_spec(
    param, n_node, entry, *, cells=None, radii=None, pairs=None, edges=None,
):
    """Build one schema ``spec`` dict (no self-id; caller keys the schema by ``param``).

    Pass exactly one of ``cells`` / ``radii`` / ``pairs`` / ``edges`` (YAML
    vocabulary for node 0..n_node-1). Omit all only when ids are ``str(node)``.
    """
    n_node = int(n_node)
    unknown = set(entry) - _PARAM_ENTRY_KEYS
    if unknown:
        raise ValueError(f"{param}: unknown fields {sorted(unknown)}")
    mode = entry["mode"]
    if mode not in PARAM_MODES:
        raise ValueError(
            f"{param}: unknown mode {mode!r}; expected one of {PARAM_MODES}"
        )
    names = _axis_names(
        param, n_node, cells=cells, radii=radii, pairs=pairs, edges=edges,
    )
    node_from = dict(zip(names, range(len(names))))
    spec = {
        "n_node": n_node,
        "lo": float(entry["lo"]),
        "hi": float(entry["hi"]),
        "jit": float(entry["jit"]),
        "init": float(entry["init"]),
    }
    if "z_mode" in entry:
        spec["z_mode"] = entry["z_mode"]
    inits = _node_vals_from_bag(entry.get("inits"), names, param, "inits")
    los = _node_vals_from_bag(entry.get("los"), names, param, "los")
    his = _node_vals_from_bag(entry.get("his"), names, param, "his")
    jits = _node_vals_from_bag(entry.get("jits"), names, param, "jits")
    if inits:
        spec["inits"] = inits
    if los:
        spec["los"] = los
    if his:
        spec["his"] = his
    if jits:
        spec["jits"] = jits
    for m in PARAM_MODES:
        spec[m] = []
    claimed = set()
    for m, group in (entry.get("modes") or {}).items():
        if m not in PARAM_MODES:
            raise ValueError(
                f"{param}: unknown mode {m!r}; expected one of {PARAM_MODES}"
            )
        for name in group:
            name = str(name)
            if name not in node_from:
                raise ValueError(f"{param}: modes.{m} unknown id {name!r}")
            node = node_from[name]
            if node in claimed:
                raise ValueError(f"{param}: {name} listed in multiple modes")
            claimed.add(node)
            spec[m].append(node)
    spec[mode].extend(sorted(set(range(n_node)) - claimed))
    for m in PARAM_MODES:
        spec[m].sort()
    shared = list(spec.get("shared") or ())
    if len(shared) >= 2:
        for param_key in ("lo", "hi", "jit"):
            plural = param_key + "s"
            vals = []
            for node in shared:
                by_node = spec.get(plural)
                if by_node is not None and int(node) in by_node:
                    vals.append(float(by_node[int(node)]))
                else:
                    vals.append(float(spec[param_key]))
            if len(set(vals)) > 1:
                raise ValueError(
                    f"{param}: shared nodes must share the same {param_key}, "
                    f"got {vals} for nodes {shared}"
                )
    return spec


def _syn_param(syn_mode, n_pair, n_edge, params, *, pairs):
    """One synaptic param: type-pair or per-edge syn_strength → ``(param, spec)``."""
    if syn_mode == "per_edge":
        if n_edge is None:
            raise TypeError("per_edge syn_strength_edge requires n_edge from network ScatterConn")
        n_edge = int(n_edge)
        edges = [f"e{i}" for i in range(n_edge)]
        return "syn_strength_edge", build_param_spec(
            "syn_strength_edge", n_edge, params["syn_strength_edge"],
            edges=edges,
        )
    if n_pair is None:
        raise TypeError("per_cell syn_strength_cell requires n_pair from network ScatterConn")
    n_pair = int(n_pair)
    return "syn_strength_cell", build_param_spec(
        "syn_strength_cell", n_pair, params["syn_strength_cell"],
        pairs=pairs,
    )


def _a_sti_radius_param(params: dict, a_sti_radii):
    """Per-radius ``a_sti_radius`` → ``(param, spec)``."""
    radii = [str(int(radius)) for radius in a_sti_radii]
    n = len(radii)
    if n == 0:
        raise ValueError("a_sti_radius requires non-empty a_sti_radii")
    spec = build_param_spec(
        "a_sti_radius", n, params["a_sti_radius"], radii=radii,
    )
    spec["radii"] = radii
    return "a_sti_radius", spec


def params_from_defaults(
    params,
    *,
    skip,
    n_cell,
    cells,
    syn_mode,
    n_pair,
    n_edge,
    pairs,
    a_sti_radii,
):
    """Build ordered schema ``dict[param, spec]``; ``skip`` omits unused params."""
    mode = syn_mode
    active_syn = (
        "syn_strength_edge" if mode == "per_edge" else "syn_strength_cell"
    )
    cells = [str(cell) for cell in cells]
    out = {}
    for param in params:
        if param in skip:
            continue
        if param in ("syn_strength_cell", "syn_strength_edge"):
            if param != active_syn:
                continue
            p, spec = _syn_param(mode, n_pair, n_edge, params, pairs=pairs)
            out[p] = spec
            continue
        if param == "a_sti_radius":
            if not a_sti_radii:
                continue
            p, spec = _a_sti_radius_param(params, a_sti_radii)
            out[p] = spec
            continue
        out[param] = build_param_spec(
            param, n_cell, params[param], cells=cells,
        )
    return out


def build_borst_schema(
    n_cell,
    cells=None,
    n_pair=None,
    *,
    syn_mode: str,
    params: dict,
    pairs,
    filter: str = "none",
    n_edge=None,
    a_sti_radii=(),
):
    """Borst schema in NEURON_SCHEMA['params'] order (rev i_h params always included)."""
    if cells is None:
        raise TypeError("borst schema requires cells from network")
    skip = set(_HP_LP_ONLY)
    if str(filter) != "ca":
        skip |= _CA_ONLY
    return params_from_defaults(
        params,
        skip=skip,
        n_cell=n_cell,
        cells=list(cells),
        syn_mode=syn_mode,
        n_pair=n_pair,
        n_edge=n_edge,
        pairs=pairs,
        a_sti_radii=a_sti_radii,
    )


def build_hp_lp_schema(
    n_cell,
    cells=None,
    n_pair=None,
    *,
    syn_mode: str,
    params: dict,
    pairs,
    filter: str = "none",
    n_edge=None,
    a_sti_radii=(),
):
    """HP-then-LP schema in NEURON_SCHEMA['params'] order (borst-only keys skipped)."""
    if cells is None:
        raise TypeError("hp_lp schema requires cells from network")
    skip = set(_BORST_ONLY)
    if str(filter) != "ca":
        skip |= _CA_ONLY
    return params_from_defaults(
        params,
        skip=skip,
        n_cell=n_cell,
        cells=list(cells),
        syn_mode=syn_mode,
        n_pair=n_pair,
        n_edge=n_edge,
        pairs=pairs,
        a_sti_radii=a_sti_radii,
    )


def build_schema(
    model: str,
    connectome,
    *,
    syn_mode: str,
    params: dict,
    filter: str = "none",
    a_sti_radii=(),
) -> dict:
    """Fresh parameter schema for ``model`` on the given connectome.

    Returns ordered ``dict[param, spec]``. ``filter``: ``none`` skips
    ``v_th_ca``/``a_ca``/``tau_ca``; ``ca`` keeps them.
    """
    n = connectome.n_cell
    cells = [str(t) for t in connectome.cells]
    pairs = [f"{cells[s]}:{cells[t]}" for s, t in connectome.conn.pairs]
    mode = syn_mode
    n_pair = getattr(connectome.conn, "n_pair", None)
    n_edge = getattr(connectome.conn, "n_edge", None)
    if mode == "per_edge":
        if n_edge is None:
            raise TypeError(f"{model} syn_strength_edge requires network ScatterConn")
    elif n_pair is None:
        raise TypeError(f"{model} syn_strength_cell requires network ScatterConn")
    kwargs = dict(
        syn_mode=mode,
        n_pair=n_pair,
        n_edge=n_edge,
        params=params,
        pairs=pairs,
        filter=filter,
        a_sti_radii=a_sti_radii,
    )
    if model == "hp_lp":
        return build_hp_lp_schema(n, cells=cells, **kwargs)
    if model == "borst":
        return build_borst_schema(n, cells=cells, **kwargs)
    raise ValueError(f"unknown model {model!r}; expected borst or hp_lp")
