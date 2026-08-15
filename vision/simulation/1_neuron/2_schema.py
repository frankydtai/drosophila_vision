# -*- coding: utf-8 -*-
"""Parameter schemas for borst / hp_lp neuron models.

Numeric lo/hi/init/jit and default ``mode`` live in
``const_default.NEURON_SCHEMA['params']`` dict entries. Optional
``exception`` holds space-separated ``param_key[.NODES]=VALUE`` tokens
(same grammar as CLI ``--param a_h.init...`` without the leading param);
base dict first, ``exception`` tokens last.

Built schema is an ordered ``dict[param, spec]`` — the param name is the
dict key; ``spec`` has no self-id field.
"""
from __future__ import annotations

from const_default import (
    NEURON_SCHEMA,
)

from neuron.param import (
    KNOWN_MODELS,
)

SYN_MODES = ("per_cell", "per_edge")
PARAM_MODES = ("indi", "shared", "fixed", "frozen")

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
_OUTPUT_KIND = frozenset({"a_gt", "bias_gt"})
_NUMBER_PARAM_KEYS = ("lo", "hi", "jit", "init")


def parse_param_tokens(tokens, *, param=None):
    """Yield ``(param, param_key, nodes, right)`` e.g. ``a_h``, ``init``, ``L1``, ``0.5``."""
    for tok in tokens:
        left, _, right = tok.partition("=")
        if param is None:
            left_param, _, rest = left.partition(".")
            param_key, _, nodes = rest.partition(".")
            yield left_param, param_key, nodes, right
        else:
            param_key, _, nodes = left.partition(".")
            yield param, param_key, nodes, right


def _comma_nodes(text):
    return [x.strip() for x in text.split(",") if x.strip()]


def expand_param_nodes(nodes):
    if nodes == "h_cells":
        return list(NEURON_SCHEMA["h_cells"])
    return _comma_nodes(nodes)


def split_param_tokens(tokens, *, param=None):
    """Parse tokens → ``(number_pairs, mode_pairs)``.

    ``number_pairs`` maps ``lo``/``hi``/``jit``/``init`` →
    lists of ``(nodes|None, float)``.
    """
    number_pairs = {param_key: [] for param_key in _NUMBER_PARAM_KEYS}
    mode_pairs = []
    for _, param_key, nodes, right in parse_param_tokens(tokens, param=param):
        if param_key in number_pairs:
            number_pairs[param_key].append(
                (None if not nodes else expand_param_nodes(nodes), float(right))
            )
        elif param_key == "mode":
            if right not in PARAM_MODES:
                raise KeyError(right)
            mode_pairs.append((None if not nodes else expand_param_nodes(nodes), right))
        else:
            raise KeyError(param_key)
    return number_pairs, mode_pairs


def resolve_inits(init_pairs, slots):
    """Fold ``(nodes|None, number)`` pairs into scalar + per-node bag ``dict[int, float]``.

    Used for ``init``/``inits`` and the same shape for ``lo``/``los``, ``hi``/``his``,
    ``jit``/``jits``.
    """
    slots = [str(slot) for slot in slots]
    idx_from_slot = {slot: node_idx for node_idx, slot in enumerate(slots)}
    node_init = {}
    for nodes, init in init_pairs:
        if nodes is None:
            node_init = {slot: init for slot in slots}
        else:
            for node in nodes:
                node_init[str(node)] = init
    if not node_init:
        return 0.0, {}
    if len(node_init) == len(slots) and len(set(node_init.values())) == 1:
        return next(iter(node_init.values())), {}
    return 0.0, {int(idx_from_slot[slot]): node_init[slot] for slot in node_init}


def resolve_modes(mode_pairs, slots):
    """Fold ``(nodes|None, mode)`` pairs into ``modes``."""
    slots = [str(slot) for slot in slots]
    idx_from_slot = {slot: node_idx for node_idx, slot in enumerate(slots)}
    node_mode = {}
    for nodes, mode in mode_pairs:
        if nodes is None:
            for slot in slots:
                node_mode[slot] = mode
        else:
            for node in nodes:
                node_mode[str(node)] = mode
    out = {mode: [] for mode in PARAM_MODES}
    for slot, mode in node_mode.items():
        out[mode].append(int(idx_from_slot[slot]))
    for mode in PARAM_MODES:
        out[mode].sort()
    return out


def param_scalar(param, param_key, params):
    entry = params[param]
    return float(entry[param_key])


def init_mode_pairs_from_entry(entry, param):
    """Base entry + ``exception`` → ``(number_pairs, mode_pairs)``."""
    mode = entry["mode"]
    if mode not in PARAM_MODES:
        raise ValueError(
            f"{param}: unknown mode {mode!r}; "
            f"expected one of {PARAM_MODES}"
        )
    number_pairs = {param_key: [(None, float(entry[param_key]))] for param_key in _NUMBER_PARAM_KEYS}
    mode_pairs = [(None, mode)]
    exception = entry.get("exception")
    if exception:
        more_numbers, more_modes = split_param_tokens(
            str(exception).split(), param=param,
        )
        for param_key in _NUMBER_PARAM_KEYS:
            number_pairs[param_key].extend(more_numbers[param_key])
        mode_pairs.extend(more_modes)
    return number_pairs, mode_pairs


def syn_strength(params):
    """Active syn_strength tensor (exactly one of syn_strength_cell / syn_strength_edge)."""
    if "syn_strength_edge" in params:
        return params["syn_strength_edge"]
    return params["syn_strength_cell"]


def build_param_spec(param, n_nodes, kind, entry, n, *, slots=None):
    """Build one schema ``spec`` dict (no self-id; caller keys the schema by ``param``)."""
    number_pairs, mode_pairs = init_mode_pairs_from_entry(entry, param)
    if slots is None:
        slots = [str(node_idx) for node_idx in range(n)]
    else:
        slots = [str(slot) for slot in slots]
    modes = resolve_modes(mode_pairs, slots)
    spec = {
        "n_nodes": n_nodes,
        "kind": kind,
    }
    for param_key in _NUMBER_PARAM_KEYS:
        scalar, bag = resolve_inits(number_pairs[param_key], slots)
        spec[param_key] = float(scalar)
        if bag:
            spec[param_key + "s"] = bag
    for mode in PARAM_MODES:
        spec[mode] = list(modes[mode])
    shared = list(spec.get("shared") or ())
    if len(shared) >= 2:
        for param_key in ("lo", "hi", "jit"):
            plural = param_key + "s"
            vals = []
            for node_idx in shared:
                bag = spec.get(plural)
                if bag is not None and int(node_idx) in bag:
                    vals.append(float(bag[int(node_idx)]))
                else:
                    vals.append(float(spec[param_key]))
            if len(set(vals)) > 1:
                raise ValueError(
                    f"{param}: shared nodes must share the same {param_key}, "
                    f"got {vals} for nodes {shared}"
                )
    return spec


def _syn_param(syn_mode, n_pairs, n_edges, params):
    """One synaptic param: type-pair or per-edge syn_strength → ``(param, spec)``."""
    if syn_mode == "per_edge":
        if n_edges is None:
            raise TypeError("per_edge syn_strength_edge requires n_edges from network ScatterConn")
        n_edges = int(n_edges)
        return "syn_strength_edge", build_param_spec(
            "syn_strength_edge", n_edges, "edge", params["syn_strength_edge"], n_edges,
        )
    if n_pairs is None:
        raise TypeError("per_cell syn_strength_cell requires n_pairs from network ScatterConn")
    n_pairs = int(n_pairs)
    return "syn_strength_cell", build_param_spec(
        "syn_strength_cell", n_pairs, "edge_pair", params["syn_strength_cell"], n_pairs,
    )


def _a_sti_radius_param(params: dict, a_sti_radii):
    """Per-radius ``a_sti_radius`` → ``(param, spec)``."""
    radii = [str(int(r)) for r in a_sti_radii]
    n = len(radii)
    if n == 0:
        raise ValueError("a_sti_radius requires non-empty a_sti_radii")
    spec = build_param_spec(
        "a_sti_radius", n, "output", params["a_sti_radius"], n,
        slots=radii,
    )
    spec["radii"] = radii
    return "a_sti_radius", spec


def params_from_defaults(
    params,
    *,
    skip,
    n_cells,
    cells,
    syn_mode,
    n_pairs,
    n_edges,
    h_cells,
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
            p, spec = _syn_param(mode, n_pairs, n_edges, params)
            out[p] = spec
            continue
        if param == "a_sti_radius":
            if not a_sti_radii:
                continue
            p, spec = _a_sti_radius_param(params, a_sti_radii)
            out[p] = spec
            continue
        kind = "output" if param in _OUTPUT_KIND else "full"
        out[param] = build_param_spec(
            param, n_cells, kind, params[param], n_cells,
            slots=cells,
        )
    return out


def build_borst_schema(
    n_cells,
    cells=None,
    n_pairs=None,
    *,
    syn_mode: str,
    params: dict,
    h_cells,
    filter: str = "none",
    n_edges=None,
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
        n_cells=n_cells,
        cells=list(cells),
        syn_mode=syn_mode,
        n_pairs=n_pairs,
        n_edges=n_edges,
        h_cells=h_cells,
        a_sti_radii=a_sti_radii,
    )


def build_hp_lp_schema(
    n_cells,
    cells=None,
    n_pairs=None,
    *,
    syn_mode: str,
    params: dict,
    h_cells,
    filter: str = "none",
    n_edges=None,
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
        n_cells=n_cells,
        cells=list(cells),
        syn_mode=syn_mode,
        n_pairs=n_pairs,
        n_edges=n_edges,
        h_cells=h_cells,
        a_sti_radii=a_sti_radii,
    )


def build_schema(
    model: str,
    backend,
    *,
    syn_mode: str,
    params: dict,
    h_cells,
    filter: str = "none",
    a_sti_radii=(),
) -> dict:
    """Fresh parameter schema for ``model`` on the given backend.

    Returns ordered ``dict[param, spec]``. ``filter``: ``none`` skips
    ``v_th_ca``/``a_ca``/``tau_ca``; ``ca`` keeps them.
    """
    if model not in KNOWN_MODELS:
        raise ValueError(f"unknown model {model!r}; expected one of {KNOWN_MODELS}")
    n = backend.n_cells
    if backend.network is None:
        raise ValueError("build_schema requires backend.network")
    cells = [str(t) for t in backend.network.cells]
    mode = syn_mode
    n_pairs = getattr(backend.conn, "n_pairs", None)
    n_edges = getattr(backend.conn, "n_edges", None)
    if mode == "per_edge":
        if n_edges is None:
            raise TypeError(f"{model} syn_strength_edge requires network ScatterConn backend")
    elif n_pairs is None:
        raise TypeError(f"{model} syn_strength_cell requires network ScatterConn backend")
    kw = dict(
        syn_mode=mode,
        n_pairs=n_pairs,
        n_edges=n_edges,
        params=params,
        h_cells=h_cells,
        filter=filter,
        a_sti_radii=a_sti_radii,
    )
    if model == "hp_lp":
        return build_hp_lp_schema(n, cells=cells, **kw)
    return build_borst_schema(n, cells=cells, **kw)
