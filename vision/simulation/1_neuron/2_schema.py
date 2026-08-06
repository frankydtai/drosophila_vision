# -*- coding: utf-8 -*-
"""Parameter schemas for borst / hp_lp neuron models.

Numeric lo/hi/init/jit and default ``train_mode`` live in
``param_defaults.PARAM_BOXES`` and are passed in as ``param_boxes``.
This module builds segment structure and resolves train_modes.
Fixed nodes always use ``effective_init`` (init / init_override); no ``fixed_val``.
"""
from __future__ import annotations

from neuron.params import (
    I_H_OFF_MODES,
    KNOWN_MODELS,
)

SYN_MODES = ("per_cell", "per_edge")
TRAIN_MODE_KEYS = ("indi", "shared", "fixed", "frozen")

ALL_PARAM_NAMES = (
    "a_in", "a_out", "a_gt", "bias_gt",
    "syn_strength_cell", "syn_strength_edge", "v_th",
    "h_g_max", "h_g_max_off",
    "h_v_mid", "h_slope", "tau_v_mid",
    "h_v_mid_off", "h_slope_off", "tau_v_mid_off",
    "tau_lp", "v_rest", "tau_hp", "a_slow", "a_sti_r",
)
I_H_SHAPE_PARAM_NAMES = (
    "h_v_mid", "h_slope", "tau_v_mid",
    "h_v_mid_off", "h_slope_off", "tau_v_mid_off",
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
    mode = _mode_from_box(box, n, name_to_i=name_to_i, indi_names=indi_names)
    s = _with_train_mode(
        {"name": name, "count": count, "kind": kind, **_box_numeric(box)},
        mode,
    )
    # indi_named: fixed remainder off (0); indi keep box init via effective_init.
    if box.get("train_mode") == "indi_named" and mode["fixed"]:
        s["init_override"] = {int(i): 0.0 for i in mode["fixed"]}
    return s


def borst_i_h_off_kwargs(p, i_h_off: str):
    """Resolve OFF-channel i_h kwargs for ``update_v`` from assigned params."""
    v_mid_off = p["h_v_mid"] if i_h_off != "on" else p["h_v_mid_off"]
    slope_off = p["h_slope"] if i_h_off != "on" else p["h_slope_off"]
    tau_off = p["tau_v_mid"] if i_h_off != "on" else p["tau_v_mid_off"]
    if i_h_off == "on":
        gmax_off = p["h_g_max_off"]
    elif i_h_off == "mirrored":
        gmax_off = p["h_g_max"]
    elif i_h_off == "off":
        gmax_off = p["h_g_max"] * 0.0
    else:
        raise ValueError(f"i_h_off {i_h_off!r} not in {I_H_OFF_MODES}")
    return gmax_off, v_mid_off, slope_off, tau_off


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


def _a_sti_r_segment(param_boxes: dict, sti_r_names):
    """Per-radius spot drive scale; r=0 fixed at 1, other radii indi.

    Box ``train_mode`` is ignored (CLI ``--a-sti-r`` may still override).
    """
    names = [str(n) for n in sti_r_names]
    n = len(names)
    if n == 0:
        raise ValueError("a_sti_r requires non-empty sti_r_names")
    mode = _mode_indi_subset_fixed_rest(n, list(range(1, n)))
    seg = _with_train_mode(
        {
            "name": "a_sti_r",
            "count": n,
            "kind": "output",
            **_box_numeric(param_boxes["a_sti_r"]),
        },
        mode,
    )
    seg["node_names"] = names
    seg["init_override"] = {0: 1.0}
    return seg


def build_borst_schema(
    n_cells,
    cell_names=None,
    n_pairs=None,
    *,
    syn_mode: str,
    param_boxes: dict,
    h_cells,
    i_h_off: str,
    n_edges=None,
    sti_r_names=(),
):
    """Borst schema; OFF i_h segments only when ``i_h_off == 'on'``."""
    if i_h_off not in I_H_OFF_MODES:
        raise ValueError(f"i_h_off {i_h_off!r} not in {I_H_OFF_MODES}")
    if cell_names is None:
        raise TypeError("borst schema requires cell_names from network")
    cell_names = list(cell_names)
    name_to_i = {str(n): i for i, n in enumerate(cell_names)}
    D = param_boxes
    named_kw = dict(name_to_i=name_to_i, indi_names=h_cells)
    segs = [
        _seg("a_in", n_cells, "full", D["a_in"], n_cells),
        _seg("a_out", n_cells, "full", D["a_out"], n_cells),
        _syn_segment(syn_mode, n_pairs, n_edges, D),
        _seg("v_th", n_cells, "full", D["v_th"], n_cells),
        _seg("a_gt", n_cells, "output", D["a_gt"], n_cells),
        _seg("bias_gt", n_cells, "output", D["bias_gt"], n_cells),
        _seg("h_g_max", n_cells, "full", D["h_g_max"], n_cells, **named_kw),
    ]
    if i_h_off == "on":
        segs.append(
            _seg("h_g_max_off", n_cells, "full", D["h_g_max_off"], n_cells, **named_kw),
        )
    segs.extend([
        _seg("h_v_mid", n_cells, "full", D["h_v_mid"], n_cells),
        _seg("h_slope", n_cells, "full", D["h_slope"], n_cells),
        _seg("tau_v_mid", n_cells, "full", D["tau_v_mid"], n_cells),
    ])
    if i_h_off == "on":
        segs.extend([
            _seg("h_v_mid_off", n_cells, "full", D["h_v_mid_off"], n_cells),
            _seg("h_slope_off", n_cells, "full", D["h_slope_off"], n_cells),
            _seg("tau_v_mid_off", n_cells, "full", D["tau_v_mid_off"], n_cells),
        ])
    segs.append(_a_sti_r_segment(D, sti_r_names))
    return segs


def build_hp_lp_schema(
    n_cells,
    cell_names=None,
    n_pairs=None,
    *,
    syn_mode: str,
    param_boxes: dict,
    h_cells,
    n_edges=None,
    sti_r_names=(),
):
    """HP-then-membrane-LP: τ_HP on v_slow, τ_lp on V, drive v_tot−a_slow v_slow."""
    if cell_names is None:
        raise TypeError("hp_lp schema requires cell_names from network")
    cell_names = list(cell_names)
    name_to_i = {str(n): i for i, n in enumerate(cell_names)}
    D = param_boxes
    named_kw = dict(name_to_i=name_to_i, indi_names=h_cells)
    return [
        _seg("a_in", n_cells, "full", D["a_in"], n_cells),
        _seg("a_out", n_cells, "full", D["a_out"], n_cells),
        _syn_segment(syn_mode, n_pairs, n_edges, D),
        _seg("v_th", n_cells, "full", D["v_th"], n_cells),
        _seg("a_gt", n_cells, "output", D["a_gt"], n_cells),
        _seg("bias_gt", n_cells, "output", D["bias_gt"], n_cells),
        _seg("tau_lp", n_cells, "full", D["tau_lp"], n_cells),
        _seg("tau_hp", n_cells, "full", D["tau_hp"], n_cells),
        _seg("v_rest", n_cells, "full", D["v_rest"], n_cells),
        _seg("a_slow", n_cells, "full", D["a_slow"], n_cells, **named_kw),
        _a_sti_r_segment(D, sti_r_names),
    ]


def default_schema(
    model: str,
    backend,
    *,
    syn_mode: str,
    param_boxes: dict,
    h_cells,
    i_h_off: str = "on",
    sti_r_names=(),
) -> list:
    """Fresh parameter schema for ``model`` on the given backend.

    ``i_h_off`` is used only for borst (OFF i_h segments when ``\"on\"``).
    ``sti_r_names`` labels ``a_sti_r`` slots (injected from training).
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
            param_boxes=param_boxes, h_cells=h_cells,
            sti_r_names=sti_r_names,
        )
    else:
        if n_pairs is None:
            raise TypeError(f"{model} syn_strength_cell requires network ScatterConn backend")
        kw = dict(
            syn_mode=mode, n_pairs=n_pairs, n_edges=n_edges,
            param_boxes=param_boxes, h_cells=h_cells,
            sti_r_names=sti_r_names,
        )
    if model == "hp_lp":
        return build_hp_lp_schema(n, cell_names=cell_names, **kw)
    return build_borst_schema(n, cell_names=cell_names, i_h_off=i_h_off, **kw)
