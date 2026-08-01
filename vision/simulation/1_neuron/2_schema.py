# -*- coding: utf-8 -*-
"""Parameter schemas for borst / hp_lp neuron models.

Numeric lo/hi/init/jit(/fixed_val) and default ``train_mode`` live in
``param_defaults.PARAM_BOXES`` and are passed in as ``param_boxes``.
This module builds segment structure and resolves train_modes.
"""
from __future__ import annotations

from neuron.params import (
    IH_OFF_MODES,
    KNOWN_MODELS,
)

SYN_MODES = ("per_cell", "per_edge")
TRAIN_MODE_KEYS = ("indi", "shared", "fixed", "frozen")

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


def _mode_indi_all(n):
    return {"indi": list(range(n)), "shared": [], "fixed": [], "frozen": []}


def _mode_shared_all(n):
    return {"indi": [], "shared": list(range(n)), "fixed": [], "frozen": []}


def _mode_fixed_all(n):
    return {"indi": [], "shared": [], "fixed": list(range(n)), "frozen": []}


def _mode_indi_subset_fixed_rest(n, indi_idx):
    indi = sorted({int(i) for i in indi_idx})
    fixed = [i for i in range(n) if i not in set(indi)]
    return {"indi": indi, "shared": [], "fixed": fixed, "frozen": []}


def _box_numeric(box):
    """PARAM_BOXES entry without ``train_mode`` (segment numeric fields only)."""
    return {k: v for k, v in box.items() if k != "train_mode"}


def _mode_from_box(box, n, *, name_to_i=None, indi_names=()):
    """Resolve ``box['train_mode']`` to indi/shared/fixed/frozen index lists."""
    tm = box["train_mode"]
    if tm == "fixed":
        return _mode_fixed_all(n)
    if tm == "indi":
        return _mode_indi_all(n)
    if tm == "shared":
        return _mode_shared_all(n)
    if tm == "indi_named":
        if name_to_i is None:
            raise TypeError("indi_named train_mode requires cell name_to_i")
        idxs = [name_to_i[str(n)] for n in indi_names]
        return _mode_indi_subset_fixed_rest(n, idxs)
    raise ValueError(
        f"unknown train_mode {tm!r}; expected indi|shared|fixed|indi_named"
    )


def _with_train_mode(seg, mode):
    s = dict(seg)
    for b in TRAIN_MODE_KEYS:
        s[b] = list(mode[b])
    return s


def _seg(name, count, kind, box, n, *, name_to_i=None, indi_names=()):
    return _with_train_mode(
        {"name": name, "count": count, "kind": kind, **_box_numeric(box)},
        _mode_from_box(box, n, name_to_i=name_to_i, indi_names=indi_names),
    )


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
        return _seg("syn_strength_edge", n_edges, "edge", D["syn_strength_edge"], n_edges)
    if n_pairs is None:
        raise TypeError("per_cell syn_strength_cell requires n_pairs from network ScatterConn")
    n_pairs = int(n_pairs)
    return _seg("syn_strength_cell", n_pairs, "edge_pair", D["syn_strength_cell"], n_pairs)


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
    D = param_boxes
    named_kw = dict(name_to_i=name_to_i, indi_names=ih_gmax_indi_names)
    segs = [
        _seg("in_gain", n_cells, "full", D["in_gain"], n_cells),
        _seg("out_gain", n_cells, "full", D["out_gain"], n_cells),
        _syn_segment(syn_mode, n_pairs, n_edges, D),
        _seg("v_th", n_cells, "full", D["v_th"], n_cells),
        _seg("out_scale", n_cells, "output", D["out_scale"], n_cells),
        _seg("Ih_gmax", n_cells, "full", D["Ih_gmax"], n_cells, **named_kw),
    ]
    if ih_off == "on":
        segs.append(
            _seg("Ih_gmax_off", n_cells, "full", D["Ih_gmax_off"], n_cells, **named_kw),
        )
    segs.extend([
        _seg("Ih_midv", n_cells, "full", D["Ih_midv"], n_cells),
        _seg("Ih_slope", n_cells, "full", D["Ih_slope"], n_cells),
        _seg("tau_midv", n_cells, "full", D["tau_midv"], n_cells),
    ])
    if ih_off == "on":
        segs.extend([
            _seg("Ih_midv_off", n_cells, "full", D["Ih_midv_off"], n_cells),
            _seg("Ih_slope_off", n_cells, "full", D["Ih_slope_off"], n_cells),
            _seg("tau_midv_off", n_cells, "full", D["tau_midv_off"], n_cells),
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
    D = param_boxes
    named_kw = dict(name_to_i=name_to_i, indi_names=ih_gmax_indi_names)
    return [
        _seg("in_gain", n_cells, "full", D["in_gain"], n_cells),
        _seg("out_gain", n_cells, "full", D["out_gain"], n_cells),
        _syn_segment(syn_mode, n_pairs, n_edges, D),
        _seg("out_scale", n_cells, "output", D["out_scale"], n_cells),
        _seg("tau_lp", n_cells, "full", D["tau_lp"], n_cells),
        _seg("tau_hp", n_cells, "full", D["tau_hp"], n_cells),
        _seg("v_rest", n_cells, "full", D["v_rest"], n_cells),
        _seg("hp_gain", n_cells, "full", D["hp_gain"], n_cells, **named_kw),
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
