# -*- coding: utf-8 -*-
"""Parameter schemas for borst / hp_lp neuron models.

Numeric lo/hi/init/jit and default ``mode`` live in
``config.NEURON_SCHEMA['params']`` dict entries. Default ``model`` scalar
lives in ``config.NEURON_SCHEMA['model']``. Optional
``exception`` holds space-separated ``param_key[.NODES]=VALUE`` tokens
(same grammar as CLI ``--param a_h.init...`` without the leading param);
base dict first, ``exception`` tokens last.

Built schema is an ordered ``dict[param, spec]`` — the param name is the
dict key; ``spec`` has no self-id field.
"""
from __future__ import annotations

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


def parse_param_tokens(tokens, *, param=None):
    """Yield ``(param, param_key, nodes, right)`` e.g. ``a_h``, ``init``, ``L1``, ``0.5``."""
    for token in tokens:
        left, _, right = token.partition("=")
        if param is None:
            left_param, _, rest = left.partition(".")
            param_key, _, nodes = rest.partition(".")
            yield left_param, param_key, nodes, right
        else:
            param_key, _, nodes = left.partition(".")
            yield param, param_key, nodes, right


def _comma_nodes(token):
    return [x.strip() for x in token.split(",") if x.strip()]


def expand_param_nodes(nodes, *, h_cells):
    if nodes == "h_cells":
        return list(h_cells)
    return _comma_nodes(nodes)


def split_param_tokens(tokens, *, param=None, h_cells):
    """Parse tokens → ``(param_inits, param_clamps, param_jits, param_modes)``.

    Single-param shapes (``param=`` bound): ``(cli_ids|None, init)``,
    ``(lo|hi, cli_ids|None, val)``, ``(cli_ids|None, jit)``,
    ``(cli_ids|None, mode)``.
    """
    param_inits = []
    param_clamps = []
    param_jits = []
    param_modes = []
    for _, param_key, nodes, right in parse_param_tokens(tokens, param=param):
        cli_ids = None if not nodes else expand_param_nodes(nodes, h_cells=h_cells)
        if param_key == "init":
            param_inits.append((cli_ids, float(right)))
        elif param_key in ("lo", "hi"):
            param_clamps.append((param_key, cli_ids, float(right)))
        elif param_key == "jit":
            param_jits.append((cli_ids, float(right)))
        elif param_key == "mode":
            if right not in PARAM_MODES:
                raise KeyError(right)
            param_modes.append((cli_ids, right))
        else:
            raise KeyError(param_key)
    return param_inits, param_clamps, param_jits, param_modes


def _param_key_from_cli(cli, *, cli_idx):
    """Fold ``(cli_ids|None, val)`` → unpack as ``init, inits`` / ``lo, los`` / …"""
    cli_ids = [str(cli_id) for cli_id in cli_idx]
    by_cli = {}
    for cli_group, val in cli:
        if cli_group is None:
            by_cli = {cli_id: val for cli_id in cli_ids}
        else:
            for cli_id in cli_group:
                by_cli[str(cli_id)] = val
    if not by_cli:
        return 0.0, {}
    if len(by_cli) == len(cli_ids) and len(set(by_cli.values())) == 1:
        return next(iter(by_cli.values())), {}
    return 0.0, {int(cli_idx[cli_id]): by_cli[cli_id] for cli_id in by_cli}


def resolve_modes(param_modes, *, cli_idx):
    """Fold ``(cli_ids|None, mode)`` list into ``modes`` (lists of node)."""
    cli_ids = [str(cli_id) for cli_id in cli_idx]
    node_mode = {}
    for cli_group, mode in param_modes:
        if cli_group is None:
            for cli_id in cli_ids:
                node_mode[cli_id] = mode
        else:
            for cli_id in cli_group:
                node_mode[str(cli_id)] = mode
    out = {mode: [] for mode in PARAM_MODES}
    for cli_id, mode in node_mode.items():
        out[mode].append(int(cli_idx[cli_id]))
    for mode in PARAM_MODES:
        out[mode].sort()
    return out


def param_from_entry(param, param_key, params):
    """``params[param][param_key]`` from a NEURON_SCHEMA entry."""
    return float(params[param][param_key])


def _param_cli_from_entry(entry, param, *, h_cells):
    """Base entry + ``exception`` → param_inits / clamps / jits / modes lists."""
    mode = entry["mode"]
    if mode not in PARAM_MODES:
        raise ValueError(
            f"{param}: unknown mode {mode!r}; "
            f"expected one of {PARAM_MODES}"
        )
    param_inits = [(None, float(entry["init"]))]
    param_clamps = [
        ("lo", None, float(entry["lo"])),
        ("hi", None, float(entry["hi"])),
    ]
    param_jits = [(None, float(entry["jit"]))]
    param_modes = [(None, mode)]
    exception = entry.get("exception")
    if exception:
        more_inits, more_clamps, more_jits, more_modes = split_param_tokens(
            str(exception).split(), param=param, h_cells=h_cells,
        )
        param_inits.extend(more_inits)
        param_clamps.extend(more_clamps)
        param_jits.extend(more_jits)
        param_modes.extend(more_modes)
    return param_inits, param_clamps, param_jits, param_modes


def build_param_spec(
    param, n_node, kind, entry, n,
    *, h_cells, cells=None, radii=None, pairs=None, edges=None,
):
    """Build one schema ``spec`` dict (no self-id; caller keys the schema by ``param``).

    Pass exactly one of ``cells`` / ``radii`` / ``pairs`` / ``edges`` (CLI vocabulary
    for node 0..n-1). Omit all only when tokens are ``str(node)``.
    """
    param_inits, param_clamps, param_jits, param_modes = _param_cli_from_entry(
        entry, param, h_cells=h_cells,
    )
    n_cli = sum(x is not None for x in (cells, radii, pairs, edges))
    if n_cli > 1:
        raise ValueError(
            f"{param}: pass at most one of cells, radii, pairs, edges"
        )
    if cells is not None:
        cells = [str(cell) for cell in cells]
        cli_idx = dict(zip(cells, range(len(cells))))
    elif radii is not None:
        radii = [str(radius) for radius in radii]
        cli_idx = dict(zip(radii, range(len(radii))))
    elif pairs is not None:
        pairs = [str(pair) for pair in pairs]
        cli_idx = dict(zip(pairs, range(len(pairs))))
    elif edges is not None:
        edges = [str(edge) for edge in edges]
        cli_idx = dict(zip(edges, range(len(edges))))
    else:
        tokens = list(map(str, range(n)))
        cli_idx = dict(zip(tokens, range(len(tokens))))
    modes = resolve_modes(param_modes, cli_idx=cli_idx)
    init, inits = _param_key_from_cli(param_inits, cli_idx=cli_idx)
    lo, los = _param_key_from_cli(
        [(cli_ids, val) for key, cli_ids, val in param_clamps if key == "lo"],
        cli_idx=cli_idx,
    )
    hi, his = _param_key_from_cli(
        [(cli_ids, val) for key, cli_ids, val in param_clamps if key == "hi"],
        cli_idx=cli_idx,
    )
    jit, jits = _param_key_from_cli(param_jits, cli_idx=cli_idx)
    spec = {
        "n_node": n_node,
        "kind": kind,
        "lo": float(lo),
        "hi": float(hi),
        "jit": float(jit),
        "init": float(init),
    }
    if los:
        spec["los"] = los
    if his:
        spec["his"] = his
    if jits:
        spec["jits"] = jits
    if inits:
        spec["inits"] = inits
    for mode in PARAM_MODES:
        spec[mode] = list(modes[mode])
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


def _syn_param(syn_mode, n_pair, n_edge, params, *, h_cells):
    """One synaptic param: type-pair or per-edge syn_strength → ``(param, spec)``."""
    if syn_mode == "per_edge":
        if n_edge is None:
            raise TypeError("per_edge syn_strength_edge requires n_edge from network ScatterConn")
        n_edge = int(n_edge)
        return "syn_strength_edge", build_param_spec(
            "syn_strength_edge", n_edge, "edge", params["syn_strength_edge"], n_edge,
            h_cells=h_cells,
        )
    if n_pair is None:
        raise TypeError("per_cell syn_strength_cell requires n_pair from network ScatterConn")
    n_pair = int(n_pair)
    return "syn_strength_cell", build_param_spec(
        "syn_strength_cell", n_pair, "edge_pair", params["syn_strength_cell"], n_pair,
        h_cells=h_cells,
    )


def _a_sti_radius_param(params: dict, a_sti_radii, *, h_cells):
    """Per-radius ``a_sti_radius`` → ``(param, spec)``."""
    radii = [str(int(radius)) for radius in a_sti_radii]
    n = len(radii)
    if n == 0:
        raise ValueError("a_sti_radius requires non-empty a_sti_radii")
    spec = build_param_spec(
        "a_sti_radius", n, "output", params["a_sti_radius"], n,
        h_cells=h_cells, radii=radii,
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
            p, spec = _syn_param(mode, n_pair, n_edge, params, h_cells=h_cells)
            out[p] = spec
            continue
        if param == "a_sti_radius":
            if not a_sti_radii:
                continue
            p, spec = _a_sti_radius_param(params, a_sti_radii, h_cells=h_cells)
            out[p] = spec
            continue
        kind = "output" if param in _OUTPUT_KIND else "node"
        out[param] = build_param_spec(
            param, n_cell, kind, params[param], n_cell,
            h_cells=h_cells, cells=cells,
        )
    return out


def build_borst_schema(
    n_cell,
    cells=None,
    n_pair=None,
    *,
    syn_mode: str,
    params: dict,
    h_cells,
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
        h_cells=h_cells,
        a_sti_radii=a_sti_radii,
    )


def build_hp_lp_schema(
    n_cell,
    cells=None,
    n_pair=None,
    *,
    syn_mode: str,
    params: dict,
    h_cells,
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
        h_cells=h_cells,
        a_sti_radii=a_sti_radii,
    )


def build_schema(
    model: str,
    connectome,
    *,
    syn_mode: str,
    params: dict,
    h_cells,
    filter: str = "none",
    a_sti_radii=(),
) -> dict:
    """Fresh parameter schema for ``model`` on the given connectome.

    Returns ordered ``dict[param, spec]``. ``filter``: ``none`` skips
    ``v_th_ca``/``a_ca``/``tau_ca``; ``ca`` keeps them.
    """
    n = connectome.n_cell
    cells = [str(t) for t in connectome.cells]
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
        h_cells=h_cells,
        filter=filter,
        a_sti_radii=a_sti_radii,
    )
    if model == "hp_lp":
        return build_hp_lp_schema(n, cells=cells, **kwargs)
    if model == "borst":
        return build_borst_schema(n, cells=cells, **kwargs)
    raise ValueError(f"unknown model {model!r}; expected borst or hp_lp")
