# -*- coding: utf-8 -*-
"""Parameter schemas for borst / hp_lp neuron models.

Numeric lo/hi/val/jit and default ``mode`` live in
``default_params.NEURON_SCHEMA['optimizable']`` dict entries. Optional
``overrides`` holds space-separated ``KEY[.NODES]=VALUE`` tokens (same
grammar as CLI ``--param NAME.KEY...``); base dict first, overrides last.
"""
from __future__ import annotations

from default_params import (
    NEURON_SCHEMA,
)

from neuron.param import (
    KNOWN_MODELS,
)

SYN_MODES = ("per_cell", "per_edge")
PARAM_MODE_KEYS = ("indi", "shared", "fixed", "frozen")

# Mirror ``default_params.NEURON_SCHEMA['optimizable']`` insertion order (injected; no import).
SEGMENT_NAMES = (
    "a_gt", "bias_gt",
    "syn_strength_cell", "syn_strength_edge",
    "a_in", "a_out", "e_leak", "v_th", "v_th_ca", "a_ca", "tau_ca",
    "tau_lp", "tau_hp",
    "a_h", "v_mid_h_g", "v_mid_h_tau", "h_slope",
    "a_h_rev", "v_mid_h_g_rev", "v_mid_h_tau_rev", "h_slope_rev",
    "a_sti_radius",
)
I_H_SHAPE_SEGMENT_NAMES = (
    "v_mid_h_g", "v_mid_h_tau", "h_slope",
    "v_mid_h_g_rev", "v_mid_h_tau_rev", "h_slope_rev",
)
_HP_LP_ONLY = frozenset({"tau_lp", "tau_hp"})
_I_H_REV_ONLY = frozenset({
    "a_h_rev", "v_mid_h_g_rev", "v_mid_h_tau_rev", "h_slope_rev",
})
_BORST_ONLY = frozenset({"v_mid_h_g", "v_mid_h_tau", "h_slope"}) | _I_H_REV_ONLY
_CA_ONLY = frozenset({"v_th_ca", "a_ca", "tau_ca"})
_OUTPUT_KIND = frozenset({"a_gt", "bias_gt"})
_OPTIMIZABLE_SCALAR_KEYS = frozenset({"lo", "hi", "jit"})


def parse_optimizable_tokens(tokens, *, segment_name=None):
    for tok in tokens:
        left, _, right = tok.partition("=")
        if segment_name is None:
            name, _, rest = left.partition(".")
            key, _, nodes = rest.partition(".")
        else:
            key, _, nodes = left.partition(".")
            name = segment_name
        yield name, key, nodes, right


def _comma_nodes(text):
    return [x.strip() for x in text.split(",") if x.strip()]


def expand_param_nodes(nodes):
    if nodes == "h_cells":
        return list(NEURON_SCHEMA["h_cells"])
    return _comma_nodes(nodes)


def split_optimizable_tokens(tokens, *, segment_name=None):
    meta = {}
    init_edits = []
    mode_edits = []
    for name, key, nodes, right in parse_optimizable_tokens(tokens, segment_name=segment_name):
        if key in _OPTIMIZABLE_SCALAR_KEYS:
            meta[key] = float(right)
        elif key == "val":
            init_edits.append((None if not nodes else expand_param_nodes(nodes), float(right)))
        elif key == "mode":
            if right not in PARAM_MODE_KEYS:
                raise KeyError(right)
            mode_edits.append((None if not nodes else expand_param_nodes(nodes), right))
        else:
            raise KeyError(key)
    return meta, init_edits, mode_edits


def resolve_init_edits(init_edits, i_from_name):
    node_init = {}
    names = [str(name) for name in i_from_name]
    for nodes, init in init_edits:
        if nodes is None:
            node_init = {name: init for name in names}
        else:
            for node in nodes:
                node_init[str(node)] = init
    if not node_init:
        return 0.0, {}
    if len(node_init) == len(names) and len(set(node_init.values())) == 1:
        return next(iter(node_init.values())), {}
    return 0.0, {int(i_from_name[name]): node_init[name] for name in node_init}


def resolve_mode_edits(mode_edits, i_from_name):
    node_mode = {}
    names = [str(name) for name in i_from_name]
    for nodes, bucket in mode_edits:
        if nodes is None:
            for name in names:
                node_mode[name] = bucket
        else:
            for node in nodes:
                node_mode[str(node)] = bucket
    out = {b: [] for b in PARAM_MODE_KEYS}
    for name, bucket in node_mode.items():
        out[bucket].append(int(i_from_name[name]))
    for b in PARAM_MODE_KEYS:
        out[b].sort()
    return out


def optimizable_scalar(segment_name, key, optimizable):
    entry = optimizable[segment_name]
    return float(entry[key])


def edits_from_optimizable(segment_optimizable, name):
    tm = segment_optimizable["mode"]
    if tm not in PARAM_MODE_KEYS:
        raise ValueError(
            f"{name}: unknown mode {tm!r}; "
            f"expected one of {PARAM_MODE_KEYS}"
        )
    init_edits = [(None, float(segment_optimizable["val"]))]
    mode_edits = [(None, tm)]
    overrides = segment_optimizable.get("overrides")
    if overrides:
        _, o_init, o_mode = split_optimizable_tokens(str(overrides).split(), segment_name=name)
        init_edits.extend(o_init)
        mode_edits.extend(o_mode)
    return init_edits, mode_edits


def normalize_syn_mode(syn_mode: str) -> str:
    mode = str(syn_mode)
    if mode not in SYN_MODES:
        raise ValueError(f"syn_mode {mode!r} not in {SYN_MODES}")
    return mode


def syn_strength(params):
    """Active syn_strength tensor (exactly one of syn_strength_cell / syn_strength_edge)."""
    if "syn_strength_edge" in params:
        return params["syn_strength_edge"]
    return params["syn_strength_cell"]


def build_segment(name, count, kind, segment_optimizable, n, *, i_from_name=None):
    init_edits, mode_edits = edits_from_optimizable(segment_optimizable, name)
    i_from = i_from_name if i_from_name else {str(i): i for i in range(n)}
    mode = resolve_mode_edits(mode_edits, i_from)
    init, init_override = resolve_init_edits(init_edits, i_from)
    s = {
        "name": name,
        "count": count,
        "kind": kind,
        "lo": float(segment_optimizable["lo"]),
        "hi": float(segment_optimizable["hi"]),
        "jit": float(segment_optimizable["jit"]),
        "init": init,
    }
    for b in PARAM_MODE_KEYS:
        s[b] = list(mode[b])
    if init_override:
        s["init_override"] = init_override
    return s



def _syn_segment(syn_mode, n_pairs, n_edges, optimizable):
    """One synaptic segment: type-pair or per-edge syn_strength."""
    if syn_mode == "per_edge":
        if n_edges is None:
            raise TypeError("per_edge syn_strength_edge requires n_edges from network ScatterConn")
        n_edges = int(n_edges)
        return build_segment("syn_strength_edge", n_edges, "edge", optimizable["syn_strength_edge"], n_edges)
    if n_pairs is None:
        raise TypeError("per_cell syn_strength_cell requires n_pairs from network ScatterConn")
    n_pairs = int(n_pairs)
    return build_segment("syn_strength_cell", n_pairs, "edge_pair", optimizable["syn_strength_cell"], n_pairs)


def spot_radius_key(radius, *, aliases) -> str:
    """Label for a Euclidean spot radius (alias name, else integer / float text)."""
    r = round(float(radius), 6)
    for name, val in aliases.items():
        if round(float(val), 6) == r:
            return str(name)
    if r == int(r):
        return str(int(r))
    return str(r)


def _a_sti_radius_segment(optimizable: dict, a_sti_radii, radius_key_aliases):
    """Per-radius spot drive ``a_sti_radius`` for non-center radii (``a_sti_radii`` order).

    Center r=0 is baked into ``i_sti`` at 1 (not a param). Slot
    names come from ``radius_key_aliases`` via :func:`spot_radius_key`.
    Default ``mode`` applies; CLI ``--a-sti-radius`` may still override.
    """
    radii = list(a_sti_radii)
    n = len(radii)
    if n == 0:
        raise ValueError("a_sti_radius requires non-empty a_sti_radii")
    names = [spot_radius_key(r, aliases=radius_key_aliases) for r in radii]
    segment = build_segment("a_sti_radius", n, "output", optimizable["a_sti_radius"], n)
    segment["node_names"] = names
    return segment


def segments_from_optimizable(
    optimizable,
    *,
    skip,
    n_cells,
    cells,
    syn_mode,
    n_pairs,
    n_edges,
    h_cells,
    a_sti_radii,
    radius_key_aliases,
):
    """Build segments in ``optimizable`` insertion order; ``skip`` omits unused names."""
    i_from_name = {str(n): i for i, n in enumerate(cells)}
    mode = normalize_syn_mode(syn_mode)
    active_syn = (
        "syn_strength_edge" if mode == "per_edge" else "syn_strength_cell"
    )
    segments = []
    for name in optimizable:
        if name in skip:
            continue
        if name in ("syn_strength_cell", "syn_strength_edge"):
            if name != active_syn:
                continue
            segments.append(_syn_segment(mode, n_pairs, n_edges, optimizable))
            continue
        if name == "a_sti_radius":
            if not a_sti_radii:
                continue
            segments.append(
                _a_sti_radius_segment(
                    optimizable, a_sti_radii, radius_key_aliases or {},
                )
            )
            continue
        kind = "output" if name in _OUTPUT_KIND else "full"
        segments.append(
            build_segment(name, n_cells, kind, optimizable[name], n_cells, i_from_name=i_from_name)
        )
    return segments


def build_borst_schema(
    n_cells,
    cells=None,
    n_pairs=None,
    *,
    syn_mode: str,
    optimizable: dict,
    h_cells,
    filter: str = "none",
    n_edges=None,
    a_sti_radii=(),
    radius_key_aliases=None,
):
    """Borst schema in NEURON_SCHEMA['optimizable'] order (rev i_h segments always included)."""
    if cells is None:
        raise TypeError("borst schema requires cells from network")
    skip = set(_HP_LP_ONLY)
    if str(filter) != "ca":
        skip |= _CA_ONLY
    return segments_from_optimizable(
        optimizable,
        skip=skip,
        n_cells=n_cells,
        cells=list(cells),
        syn_mode=syn_mode,
        n_pairs=n_pairs,
        n_edges=n_edges,
        h_cells=h_cells,
        a_sti_radii=a_sti_radii,
        radius_key_aliases=radius_key_aliases,
    )


def build_hp_lp_schema(
    n_cells,
    cells=None,
    n_pairs=None,
    *,
    syn_mode: str,
    optimizable: dict,
    h_cells,
    filter: str = "none",
    n_edges=None,
    a_sti_radii=(),
    radius_key_aliases=None,
):
    """HP-then-membrane-LP schema in NEURON_SCHEMA['optimizable'] order (borst-only keys skipped)."""
    if cells is None:
        raise TypeError("hp_lp schema requires cells from network")
    skip = set(_BORST_ONLY)
    if str(filter) != "ca":
        skip |= _CA_ONLY
    return segments_from_optimizable(
        optimizable,
        skip=skip,
        n_cells=n_cells,
        cells=list(cells),
        syn_mode=syn_mode,
        n_pairs=n_pairs,
        n_edges=n_edges,
        h_cells=h_cells,
        a_sti_radii=a_sti_radii,
        radius_key_aliases=radius_key_aliases,
    )


def build_schema(
    model: str,
    backend,
    *,
    syn_mode: str,
    optimizable: dict,
    h_cells,
    filter: str = "none",
    a_sti_radii=(),
    radius_key_aliases=None,
) -> list:
    """Fresh parameter schema for ``model`` on the given backend.

    ``filter``: ``none`` skips ``v_th_ca``/``a_ca``/``tau_ca``; ``ca`` keeps them.
    Rev i_h (borst): train/--param mode/init and ``--val-from`` (not a separate enum).
    ``a_sti_radii`` + ``radius_key_aliases`` label ``a_sti_radius`` slots
    (injected from train).
    """
    if model not in KNOWN_MODELS:
        raise ValueError(f"unknown model {model!r}; expected one of {KNOWN_MODELS}")
    n = backend.n_cells
    if backend.network is None:
        raise ValueError("build_schema requires backend.network")
    cells = [str(t) for t in backend.network.cells]
    mode = normalize_syn_mode(syn_mode)
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
        optimizable=optimizable,
        h_cells=h_cells,
        filter=filter,
        a_sti_radii=a_sti_radii,
        radius_key_aliases=radius_key_aliases,
    )
    if model == "hp_lp":
        return build_hp_lp_schema(n, cells=cells, **kw)
    return build_borst_schema(n, cells=cells, **kw)
