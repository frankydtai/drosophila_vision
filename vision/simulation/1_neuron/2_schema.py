# -*- coding: utf-8 -*-
"""Parameter schemas for borst / hp_lp neuron models.

Numeric lo/hi/init/jit and default ``train_mode`` live in
``param_defaults.PARAM_BOXES`` and are passed in as ``param_boxes``.
This module builds segment structure and resolves train_modes.
Fixed nodes always use ``effective_init`` (init / init_override); no ``fixed_val``.
"""
from __future__ import annotations

from neuron.params import (
    I_H_REV_MODES,
    KNOWN_MODELS,
)

SYN_MODES = ("per_cell", "per_edge")
TRAIN_MODE_KEYS = ("indi", "shared", "fixed", "frozen")

# Mirror ``param_defaults.PARAM_BOXES`` insertion order (injected; no import).
ALL_PARAM_NAMES = (
    "a_gt", "bias_gt",
    "syn_strength_cell", "syn_strength_edge",
    "a_in", "a_out", "e_leak", "v_th", "v_th_ca", "a_ca", "tau_ca",
    "tau_lp", "tau_hp",
    "a_h", "v_mid_h_g", "v_mid_h_tau", "h_slope",
    "a_h_rev", "v_mid_h_g_rev", "v_mid_h_tau_rev", "h_slope_rev",
    "a_sti_radius",
)
I_H_SHAPE_PARAM_NAMES = (
    "v_mid_h_g", "v_mid_h_tau", "h_slope",
    "v_mid_h_g_rev", "v_mid_h_tau_rev", "h_slope_rev",
)
_HP_LP_ONLY = frozenset({"tau_lp", "tau_hp"})
_I_H_REV_ONLY = frozenset({
    "a_h_rev", "v_mid_h_g_rev", "v_mid_h_tau_rev", "h_slope_rev",
})
_BORST_ONLY = frozenset({"v_mid_h_g", "v_mid_h_tau", "h_slope"}) | _I_H_REV_ONLY
_OUTPUT_KIND = frozenset({"a_gt", "bias_gt"})
_NAMED_H = frozenset({"a_h", "a_h_rev"})


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


def _mode_all(n, key):
    return {k: (list(range(n)) if k == key else []) for k in TRAIN_MODE_KEYS}


def _mode_from_box(box, n, *, i_from_name=None, indi_names=()):
    """Resolve ``box['train_mode']`` to indi/shared/fixed/frozen index lists."""
    tm = box["train_mode"]
    if tm in ("fixed", "indi", "shared"):
        return _mode_all(n, tm)
    if tm == "indi_named":
        if i_from_name is None:
            raise TypeError("indi_named train_mode requires cell i_from_name")
        indi = sorted({int(i_from_name[str(name)]) for name in indi_names})
        fixed = [i for i in range(n) if i not in set(indi)]
        return {"indi": indi, "shared": [], "fixed": fixed, "frozen": []}
    raise ValueError(
        f"unknown train_mode {tm!r}; expected indi|shared|fixed|indi_named"
    )


def _seg(name, count, kind, box, n, *, i_from_name=None, indi_names=()):
    mode = _mode_from_box(box, n, i_from_name=i_from_name, indi_names=indi_names)
    s = {
        "name": name,
        "count": count,
        "kind": kind,
        **{k: v for k, v in box.items() if k != "train_mode"},
    }
    for b in TRAIN_MODE_KEYS:
        s[b] = list(mode[b])
    # indi_named: fixed remainder off (0); indi keep box init via effective_init.
    if box.get("train_mode") == "indi_named" and mode["fixed"]:
        s["init_override"] = {int(i): 0.0 for i in mode["fixed"]}
    return s


def borst_i_h_rev_kwargs(params, i_h_rev: str):
    """Resolve rev-channel i_h kwargs for ``update_v`` from assigned params."""
    if i_h_rev == "on":
        return params["a_h_rev"], params["v_mid_h_g_rev"], params["h_slope_rev"], params["v_mid_h_tau_rev"]
    if i_h_rev == "mirrored":
        a_h_rev = params["a_h"]
    elif i_h_rev == "off":
        a_h_rev = params["a_h"] * 0.0
    else:
        raise ValueError(f"i_h_rev {i_h_rev!r} not in {I_H_REV_MODES}")
    return a_h_rev, params["v_mid_h_g"], params["h_slope"], params["v_mid_h_tau"]


def _syn_segment(syn_mode, n_pairs, n_edges, param_boxes):
    """One synaptic segment: type-pair or per-edge syn_strength."""
    if syn_mode == "per_edge":
        if n_edges is None:
            raise TypeError("per_edge syn_strength_edge requires n_edges from network ScatterConn")
        n_edges = int(n_edges)
        return _seg("syn_strength_edge", n_edges, "edge", param_boxes["syn_strength_edge"], n_edges)
    if n_pairs is None:
        raise TypeError("per_cell syn_strength_cell requires n_pairs from network ScatterConn")
    n_pairs = int(n_pairs)
    return _seg("syn_strength_cell", n_pairs, "edge_pair", param_boxes["syn_strength_cell"], n_pairs)


def spot_radius_key(radius, *, aliases) -> str:
    """Label for a Euclidean spot radius (alias name, else integer / float text)."""
    r = round(float(radius), 6)
    for name, val in aliases.items():
        if round(float(val), 6) == r:
            return str(name)
    if r == int(r):
        return str(int(r))
    return str(r)


def _a_sti_radius_segment(param_boxes: dict, sti_radii, radius_key_aliases):
    """Per-radius spot drive scale for non-center radii (``sti_radii`` order).

    Center r=0 is baked into ``i_sti`` at scale 1 (not a param). Slot
    names come from ``radius_key_aliases`` via :func:`spot_radius_key`.
    Box ``train_mode`` applies; CLI ``--a-sti-radius`` may still override.
    """
    radii = list(sti_radii)
    n = len(radii)
    if n == 0:
        raise ValueError("a_sti_radius requires non-empty sti_radii")
    names = [spot_radius_key(r, aliases=radius_key_aliases) for r in radii]
    seg = _seg("a_sti_radius", n, "output", param_boxes["a_sti_radius"], n)
    seg["node_names"] = names
    return seg


def _segments_from_boxes(
    param_boxes,
    *,
    skip,
    n_cells,
    cells,
    syn_mode,
    n_pairs,
    n_edges,
    h_cells,
    sti_radii,
    radius_key_aliases,
):
    """Build segments in ``param_boxes`` insertion order; ``skip`` omits unused names."""
    i_from_name = {str(n): i for i, n in enumerate(cells)}
    named_kw = dict(i_from_name=i_from_name, indi_names=h_cells)
    mode = normalize_syn_mode(syn_mode)
    active_syn = (
        "syn_strength_edge" if mode == "per_edge" else "syn_strength_cell"
    )
    segs = []
    for name in param_boxes:
        if name in skip:
            continue
        if name in ("syn_strength_cell", "syn_strength_edge"):
            if name != active_syn:
                continue
            segs.append(_syn_segment(mode, n_pairs, n_edges, param_boxes))
            continue
        if name == "a_sti_radius":
            if not sti_radii:
                continue
            segs.append(
                _a_sti_radius_segment(
                    param_boxes, sti_radii, radius_key_aliases or {},
                )
            )
            continue
        kind = "output" if name in _OUTPUT_KIND else "full"
        kw = named_kw if name in _NAMED_H else {}
        segs.append(
            _seg(name, n_cells, kind, param_boxes[name], n_cells, **kw)
        )
    return segs


def build_borst_schema(
    n_cells,
    cells=None,
    n_pairs=None,
    *,
    syn_mode: str,
    param_boxes: dict,
    h_cells,
    i_h_rev: str,
    n_edges=None,
    sti_radii=(),
    radius_key_aliases=None,
):
    """Borst schema in PARAM_BOXES order; rev i_h only when ``i_h_rev == 'on'``."""
    if i_h_rev not in I_H_REV_MODES:
        raise ValueError(f"i_h_rev {i_h_rev!r} not in {I_H_REV_MODES}")
    if cells is None:
        raise TypeError("borst schema requires cells from network")
    skip = set(_HP_LP_ONLY)
    if i_h_rev != "on":
        skip |= _I_H_REV_ONLY
    return _segments_from_boxes(
        param_boxes,
        skip=skip,
        n_cells=n_cells,
        cells=list(cells),
        syn_mode=syn_mode,
        n_pairs=n_pairs,
        n_edges=n_edges,
        h_cells=h_cells,
        sti_radii=sti_radii,
        radius_key_aliases=radius_key_aliases,
    )


def build_hp_lp_schema(
    n_cells,
    cells=None,
    n_pairs=None,
    *,
    syn_mode: str,
    param_boxes: dict,
    h_cells,
    n_edges=None,
    sti_radii=(),
    radius_key_aliases=None,
):
    """HP-then-membrane-LP schema in PARAM_BOXES order (borst-only keys skipped)."""
    if cells is None:
        raise TypeError("hp_lp schema requires cells from network")
    return _segments_from_boxes(
        param_boxes,
        skip=_BORST_ONLY,
        n_cells=n_cells,
        cells=list(cells),
        syn_mode=syn_mode,
        n_pairs=n_pairs,
        n_edges=n_edges,
        h_cells=h_cells,
        sti_radii=sti_radii,
        radius_key_aliases=radius_key_aliases,
    )


def default_schema(
    model: str,
    backend,
    *,
    syn_mode: str,
    param_boxes: dict,
    h_cells,
    i_h_rev: str = "on",
    sti_radii=(),
    radius_key_aliases=None,
) -> list:
    """Fresh parameter schema for ``model`` on the given backend.

    ``i_h_rev`` is used only for borst (rev i_h segments when ``\"on\"``).
    ``sti_radii`` + ``radius_key_aliases`` label ``a_sti_radius`` slots
    (injected from training).
    """
    if model not in KNOWN_MODELS:
        raise ValueError(f"unknown model {model!r}; expected one of {KNOWN_MODELS}")
    n = backend.n_cells
    if backend.network is None:
        raise ValueError("default_schema requires backend.network")
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
        param_boxes=param_boxes,
        h_cells=h_cells,
        sti_radii=sti_radii,
        radius_key_aliases=radius_key_aliases,
    )
    if model == "hp_lp":
        return build_hp_lp_schema(n, cells=cells, **kw)
    return build_borst_schema(n, cells=cells, i_h_rev=i_h_rev, **kw)
