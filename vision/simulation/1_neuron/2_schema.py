# -*- coding: utf-8 -*-
"""Parameter schemas for borst / hp_lp neuron models.

Numeric lo/hi/val/jit and default ``mode`` live in
``default_params.NEURON_SCHEMA['optimizable']`` dict entries. Optional
``param`` holds space-separated ``KEY[.NODES]=VALUE`` tokens (same grammar
as CLI ``--param NAME.KEY...`` without the segment); base dict first,
``param`` tokens last.
"""
from __future__ import annotations

from default_params import (
    NEURON_SCHEMA,
)

from neuron.param import (
    KNOWN_MODELS,
)

SYN_MODES = ("per_cell", "per_edge")
PARAM_MODES = ("indi", "shared", "fixed", "frozen")

# Mirror ``default_params.NEURON_SCHEMA['optimizable']`` insertion order (injected; no import).
SEGMENT_NAMES = (
    "a_gt", "bias_gt",
    "syn_strength_cell", "syn_strength_edge",
    "a_in", "a_out", "e_leak", "v_th", "v_th_ca", "a_ca", "tau_ca",
    "tau_lp", "tau_hp_rise", "tau_hp_fall",
    "a_h", "v_mid_h_g", "v_mid_h_tau", "h_slope",
    "a_h_rev", "v_mid_h_g_rev", "v_mid_h_tau_rev", "h_slope_rev",
    "a_sti_radius",
)
I_H_SHAPE_SEGMENT_NAMES = (
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
_OPTIMIZABLE_SCALAR_KEYS = frozenset({"lo", "hi", "jit"})


def parse_optimizable_tokens(tokens, *, segment=None):
    for tok in tokens:
        left, _, right = tok.partition("=")
        if segment is None:
            segment_tok, _, rest = left.partition(".")
            key, _, nodes = rest.partition(".")
        else:
            key, _, nodes = left.partition(".")
            segment_tok = segment
        yield segment_tok, key, nodes, right


def _comma_nodes(text):
    return [x.strip() for x in text.split(",") if x.strip()]


def expand_param_nodes(nodes):
    if nodes == "h_cells":
        return list(NEURON_SCHEMA["h_cells"])
    return _comma_nodes(nodes)


def split_optimizable_tokens(tokens, *, segment=None):
    meta = {}
    val_pairs = []
    mode_pairs = []
    for segment_tok, key, nodes, right in parse_optimizable_tokens(tokens, segment=segment):
        if key in _OPTIMIZABLE_SCALAR_KEYS:
            meta[key] = float(right)
        elif key == "val":
            val_pairs.append((None if not nodes else expand_param_nodes(nodes), float(right)))
        elif key == "mode":
            if right not in PARAM_MODES:
                raise KeyError(right)
            mode_pairs.append((None if not nodes else expand_param_nodes(nodes), right))
        else:
            raise KeyError(key)
    return meta, val_pairs, mode_pairs


def resolve_inits(val_pairs, slots):
    """Fold ``(nodes|None, val)`` pairs into scalar ``init`` + per-node ``inits``."""
    slots = [str(slot) for slot in slots]
    idx_from = {slot: i for i, slot in enumerate(slots)}
    node_init = {}
    for nodes, init in val_pairs:
        if nodes is None:
            node_init = {slot: init for slot in slots}
        else:
            for node in nodes:
                node_init[str(node)] = init
    if not node_init:
        return 0.0, {}
    if len(node_init) == len(slots) and len(set(node_init.values())) == 1:
        return next(iter(node_init.values())), {}
    return 0.0, {int(idx_from[slot]): node_init[slot] for slot in node_init}


def resolve_modes(mode_pairs, slots):
    """Fold ``(nodes|None, mode)`` pairs into a ``modes`` bag."""
    slots = [str(slot) for slot in slots]
    idx_from = {slot: i for i, slot in enumerate(slots)}
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
        out[mode].append(int(idx_from[slot]))
    for mode in PARAM_MODES:
        out[mode].sort()
    return out


def optimizable_scalar(segment, key, optimizable):
    entry = optimizable[segment]
    return float(entry[key])


def val_mode_pairs_from_optimizable(segment_optimizable, segment):
    mode = segment_optimizable["mode"]
    if mode not in PARAM_MODES:
        raise ValueError(
            f"{segment}: unknown mode {mode!r}; "
            f"expected one of {PARAM_MODES}"
        )
    val_pairs = [(None, float(segment_optimizable["val"]))]
    mode_pairs = [(None, mode)]
    param = segment_optimizable.get("param")
    if param:
        _, more_vals, more_modes = split_optimizable_tokens(
            str(param).split(), segment=segment,
        )
        val_pairs.extend(more_vals)
        mode_pairs.extend(more_modes)
    return val_pairs, mode_pairs


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


def build_segment(segment, n_nodes, kind, segment_optimizable, n, *, slots=None):
    val_pairs, mode_pairs = val_mode_pairs_from_optimizable(
        segment_optimizable, segment,
    )
    if slots is None:
        slots = [str(node_idx) for node_idx in range(n)]
    else:
        slots = [str(slot) for slot in slots]
    modes = resolve_modes(mode_pairs, slots)
    init, inits = resolve_inits(val_pairs, slots)
    s = {
        "segment": segment,
        "n_nodes": n_nodes,
        "kind": kind,
        "lo": float(segment_optimizable["lo"]),
        "hi": float(segment_optimizable["hi"]),
        "jit": float(segment_optimizable["jit"]),
        "init": init,
    }
    for mode in PARAM_MODES:
        s[mode] = list(modes[mode])
    if inits:
        s["inits"] = inits
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


def _a_sti_radius_segment(optimizable: dict, a_sti_radii):
    """Per-radius spot drive ``a_sti_radius`` for non-center radii (``a_sti_radii`` order).

    Center r=0 is baked into ``i_sti`` at 1 (not a param). Slot keys are
    ``str(int(radius))``. Default ``mode`` applies; CLI ``--a-sti-radius`` may
    still change it.
    """
    radii = list(a_sti_radii)
    n = len(radii)
    if n == 0:
        raise ValueError("a_sti_radius requires non-empty a_sti_radii")
    radius_keys = [str(int(r)) for r in radii]
    segment = build_segment(
        "a_sti_radius", n, "output", optimizable["a_sti_radius"], n,
        slots=radius_keys,
    )
    segment["radius_keys"] = radius_keys
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
):
    """Build segments in ``optimizable`` insertion order; ``skip`` omits unused segments."""
    mode = normalize_syn_mode(syn_mode)
    active_syn = (
        "syn_strength_edge" if mode == "per_edge" else "syn_strength_cell"
    )
    cells = [str(cell) for cell in cells]
    segments = []
    for segment in optimizable:
        if segment in skip:
            continue
        if segment in ("syn_strength_cell", "syn_strength_edge"):
            if segment != active_syn:
                continue
            segments.append(_syn_segment(mode, n_pairs, n_edges, optimizable))
            continue
        if segment == "a_sti_radius":
            if not a_sti_radii:
                continue
            segments.append(_a_sti_radius_segment(optimizable, a_sti_radii))
            continue
        kind = "output" if segment in _OUTPUT_KIND else "full"
        segments.append(
            build_segment(
                segment, n_cells, kind, optimizable[segment], n_cells,
                slots=cells,
            )
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
) -> list:
    """Fresh parameter schema for ``model`` on the given backend.

    ``filter``: ``none`` skips ``v_th_ca``/``a_ca``/``tau_ca``; ``ca`` keeps them.
    Rev i_h (borst): train/--param mode/init and ``--val-from`` (not a separate enum).
    ``a_sti_radii`` labels ``a_sti_radius`` slots (injected from train).
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
    )
    if model == "hp_lp":
        return build_hp_lp_schema(n, cells=cells, **kw)
    return build_borst_schema(n, cells=cells, **kw)
