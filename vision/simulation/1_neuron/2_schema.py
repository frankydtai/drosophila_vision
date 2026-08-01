# -*- coding: utf-8 -*-
"""Parameter schemas for borst / hp_lp neuron models.

Numeric lo/hi/init live in ``training.defaults.PARAM_BOXES`` and are passed
in as ``param_boxes``. This module only builds structure + partitions.
"""
from __future__ import annotations

from neuron.params import (
    IH_OFF_MODES,
    KNOWN_MODELS,
)

SYN_MODES = ("per_cell", "per_edge")

ALL_PARAM_NAMES = (
    "in_gain", "out_gain", "out_scale", "syn_strength_cell", "syn_strength_edge", "v_th",
    "Ih_gmax", "Ih_gmax_off",
    "Ih_midv", "Ih_slope", "tau_midv",
    "Ih_midv_off", "Ih_slope_off", "tau_midv_off",
    "tau_lp", "v_rest", "tau_hp", "hp_gain",
)
IH_SHAPE_PARAM_NAMES = (
    "Ih_midv", "Ih_slope", "tau_midv",
    "Ih_midv_off", "Ih_slope_off", "tau_midv_off",
)


def normalize_syn_mode(syn_mode: str) -> str:
    mode = str(syn_mode)
    if mode not in SYN_MODES:
        raise ValueError(f"syn_mode {mode!r} not in {SYN_MODES}")
    return mode


def syn_strength(p):
    """Active syn_strength tensor (exactly one of syn_strength_cell / syn_strength_edge)."""
    if "syn_strength_edge" in p:
        return p["syn_strength_edge"]
    return p["syn_strength_cell"]


def _part_indi_all(n):
    return {"indi": list(range(n)), "shared": [], "fixed": [], "frozen": []}


def _part_shared_all(n):
    return {"indi": [], "shared": list(range(n)), "fixed": [], "frozen": []}


def _part_fixed_all(n):
    return {"indi": [], "shared": [], "fixed": list(range(n)), "frozen": []}


def _part_indi_subset_fixed_rest(n, indi_idx):
    indi = sorted({int(i) for i in indi_idx})
    fixed = [i for i in range(n) if i not in set(indi)]
    return {"indi": indi, "shared": [], "fixed": fixed, "frozen": []}


def _with_part(seg, part):
    s = dict(seg)
    for b in ("indi", "shared", "fixed", "frozen"):
        s[b] = list(part[b])
    return s


def borst_ih_off_kwargs(p, ih_off: str):
    """Resolve OFF-channel Ih kwargs for ``update_v`` from assigned params."""
    midv_off = p["Ih_midv"] if ih_off != "on" else p["Ih_midv_off"]
    slope_off = p["Ih_slope"] if ih_off != "on" else p["Ih_slope_off"]
    tau_off = p["tau_midv"] if ih_off != "on" else p["tau_midv_off"]
    if ih_off == "on":
        gmax_off = p["Ih_gmax_off"]
    elif ih_off == "mirrored":
        gmax_off = p["Ih_gmax"]
    elif ih_off == "off":
        gmax_off = p["Ih_gmax"] * 0.0
    else:
        raise ValueError(f"ih_off {ih_off!r} not in {IH_OFF_MODES}")
    return gmax_off, midv_off, slope_off, tau_off


def _syn_segment(syn_mode, n_pairs, n_edges, param_boxes):
    """One synaptic segment: type-pair or per-edge syn_strength."""
    D = param_boxes
    mode = normalize_syn_mode(syn_mode)
    if mode == "per_edge":
        if n_edges is None:
            raise TypeError("per_edge syn_strength_edge requires n_edges from network ScatterConn")
        n_edges = int(n_edges)
        return _with_part(
            {"name": "syn_strength_edge", "count": n_edges, "kind": "edge", **D["syn_strength_edge"]},
            _part_indi_all(n_edges),
        )
    if n_pairs is None:
        raise TypeError("per_cell syn_strength_cell requires n_pairs from network ScatterConn")
    n_pairs = int(n_pairs)
    return _with_part(
        {"name": "syn_strength_cell", "count": n_pairs, "kind": "edge_pair", **D["syn_strength_cell"]},
        _part_indi_all(n_pairs),
    )


def build_borst_schema(
    n_cells,
    cell_names=None,
    n_pairs=None,
    *,
    syn_mode: str,
    param_boxes: dict,
    ih_gmax_indi_names,
    ih_off: str,
    n_edges=None,
):
    """Borst schema; OFF Ih segments only when ``ih_off == 'on'``."""
    if ih_off not in IH_OFF_MODES:
        raise ValueError(f"ih_off {ih_off!r} not in {IH_OFF_MODES}")
    if cell_names is None:
        raise TypeError("borst schema requires cell_names from network")
    cell_names = list(cell_names)
    name_to_i = {str(n): i for i, n in enumerate(cell_names)}
    ih_gmax = [name_to_i[n] for n in ih_gmax_indi_names]
    D = param_boxes
    indi_all = _part_indi_all(n_cells)
    fixed_all = _part_fixed_all(n_cells)
    shared_all = _part_shared_all(n_cells)
    ih_gmax_part = _part_indi_subset_fixed_rest(n_cells, ih_gmax)
    segs = [
        _with_part({"name": "in_gain", "count": n_cells, "kind": "full", **D["in_gain"]}, fixed_all),
        _with_part({"name": "out_gain", "count": n_cells, "kind": "full", **D["out_gain"]}, indi_all),
        _syn_segment(syn_mode, n_pairs, n_edges, D),
        _with_part({"name": "v_th", "count": n_cells, "kind": "full", **D["v_th"]}, fixed_all),
        _with_part({"name": "out_scale", "count": n_cells, "kind": "output", **D["out_scale"]}, indi_all),
        _with_part({"name": "Ih_gmax", "count": n_cells, "kind": "full", **D["Ih_gmax"]}, ih_gmax_part),
    ]
    if ih_off == "on":
        segs.append(
            _with_part({"name": "Ih_gmax_off", "count": n_cells, "kind": "full", **D["Ih_gmax_off"]}, ih_gmax_part),
        )
    segs.extend([
        _with_part({"name": "Ih_midv", "count": n_cells, "kind": "full", **D["Ih_midv"]}, shared_all),
        _with_part({"name": "Ih_slope", "count": n_cells, "kind": "full", **D["Ih_slope"]}, shared_all),
        _with_part({"name": "tau_midv", "count": n_cells, "kind": "full", **D["tau_midv"]}, shared_all),
    ])
    if ih_off == "on":
        segs.extend([
            _with_part({"name": "Ih_midv_off", "count": n_cells, "kind": "full", **D["Ih_midv_off"]}, shared_all),
            _with_part({"name": "Ih_slope_off", "count": n_cells, "kind": "full", **D["Ih_slope_off"]}, shared_all),
            _with_part({"name": "tau_midv_off", "count": n_cells, "kind": "full", **D["tau_midv_off"]}, shared_all),
        ])
    return segs


def build_hp_lp_schema(
    n_cells,
    cell_names=None,
    n_pairs=None,
    *,
    syn_mode: str,
    param_boxes: dict,
    ih_gmax_indi_names,
    n_edges=None,
):
    """HP-then-membrane-LP: τ_HP on slow average a, τ_lp on V, drive G(X−a)."""
    if cell_names is None:
        raise TypeError("hp_lp schema requires cell_names from network")
    cell_names = list(cell_names)
    name_to_i = {str(n): i for i, n in enumerate(cell_names)}
    hp_gain_indi = [name_to_i[n] for n in ih_gmax_indi_names]
    D = param_boxes
    indi_all = _part_indi_all(n_cells)
    fixed_all = _part_fixed_all(n_cells)
    hp_gain_part = _part_indi_subset_fixed_rest(n_cells, hp_gain_indi)
    return [
        _with_part({"name": "in_gain", "count": n_cells, "kind": "full", **D["in_gain"]}, fixed_all),
        _with_part({"name": "out_gain", "count": n_cells, "kind": "full", **D["out_gain"]}, indi_all),
        _syn_segment(syn_mode, n_pairs, n_edges, D),
        _with_part({"name": "out_scale", "count": n_cells, "kind": "output", **D["out_scale"]}, indi_all),
        _with_part({"name": "tau_lp", "count": n_cells, "kind": "full", **D["tau_lp"]}, indi_all),
        _with_part({"name": "tau_hp", "count": n_cells, "kind": "full", **D["tau_hp"]}, indi_all),
        _with_part({"name": "v_rest", "count": n_cells, "kind": "full", **D["v_rest"]}, indi_all),
        _with_part({"name": "hp_gain", "count": n_cells, "kind": "full", **D["hp_gain"]}, hp_gain_part),
    ]


def default_schema(
    model: str,
    backend,
    *,
    syn_mode: str,
    param_boxes: dict,
    ih_gmax_indi_names,
    ih_off: str = "on",
) -> list:
    """Fresh parameter schema for ``model`` on the given backend.

    ``ih_off`` is used only for borst (OFF Ih segments when ``\"on\"``).
    """
    if model not in KNOWN_MODELS:
        raise ValueError(f"unknown model {model!r}; expected one of {KNOWN_MODELS}")
    n = backend.n_cells
    if backend.network is None:
        raise ValueError("default_schema requires backend.network")
    cell_names = [str(t) for t in backend.network.cell_names]
    mode = normalize_syn_mode(syn_mode)
    n_pairs = getattr(backend.conn, "n_pairs", None)
    n_edges = getattr(backend.conn, "n_edges", None)
    if mode == "per_edge":
        if n_edges is None:
            raise TypeError(f"{model} syn_strength_edge requires network ScatterConn backend")
        kw = dict(
            syn_mode=mode, n_edges=n_edges, n_pairs=n_pairs,
            param_boxes=param_boxes, ih_gmax_indi_names=ih_gmax_indi_names,
        )
    else:
        if n_pairs is None:
            raise TypeError(f"{model} syn_strength_cell requires network ScatterConn backend")
        kw = dict(
            syn_mode=mode, n_pairs=n_pairs, n_edges=n_edges,
            param_boxes=param_boxes, ih_gmax_indi_names=ih_gmax_indi_names,
        )
    if model == "hp_lp":
        return build_hp_lp_schema(n, cell_names=cell_names, **kw)
    return build_borst_schema(n, cell_names=cell_names, ih_off=ih_off, **kw)
