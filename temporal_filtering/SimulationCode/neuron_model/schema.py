# -*- coding: utf-8 -*-
"""Parameter schemas for conductance / hp_lp neuron models."""
from __future__ import annotations

from param_defaults import DEFAULT_IH_GMAX_INDI_NAMES, P as PARAM_DEFAULTS

from neuron_model.constants import (
    IH_OFF_DEFAULT,
    IH_OFF_GMAX_SEGMENT,
    IH_OFF_MODES,
    IH_OFF_SCALAR_SEGMENTS,
    KNOWN_MODELS,
)

SYN_MODES = ("type_pair", "per_edge")
SYN_MODE_DEFAULT = "type_pair"

ALL_PARAM_NAMES = (
    "in_gain", "out_gain", "out_scale", "syn_strength", "edge_weight", "v_th",
    "Ih_gmax", "Ih_gmax_off",
    "Ih_midv", "Ih_slope", "tau_midv",
    "Ih_midv_off", "Ih_slope_off", "tau_midv_off",
    "tau_lp", "bias", "tau_hp", "hp_gain",
)
IH_SHAPE_PARAM_NAMES = (
    "Ih_midv", "Ih_slope", "tau_midv",
    "Ih_midv_off", "Ih_slope_off", "tau_midv_off",
)


def normalize_syn_mode(syn_mode=SYN_MODE_DEFAULT) -> str:
    mode = str(syn_mode)
    if mode not in SYN_MODES:
        raise ValueError(f"syn_mode {mode!r} not in {SYN_MODES}")
    return mode


def synaptic_scale(p):
    """Edge scaling tensor from assigned params (exactly one of syn_strength / edge_weight)."""
    if "edge_weight" in p:
        return p["edge_weight"]
    return p["syn_strength"]


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


def apply_ih_off_mode(schema, mode=IH_OFF_DEFAULT):
    """Adjust conductance Ih schema for ON/OFF coupling (``on|off|mirrored``)."""
    if mode not in IH_OFF_MODES:
        raise ValueError(f"ih_off {mode!r} not in {IH_OFF_MODES}")
    out = []
    for seg in schema:
        s = dict(seg)
        name = s["name"]
        if mode == "on":
            out.append(s)
            continue
        if name in IH_OFF_SCALAR_SEGMENTS or name == IH_OFF_GMAX_SEGMENT:
            continue
        out.append(s)
    return out


def conductance_ih_off_kwargs(p, ih_off=IH_OFF_DEFAULT):
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


def _syn_segment(syn_mode, n_pairs, n_edges):
    """One synaptic segment: type-pair α or per-edge magnitude."""
    D = PARAM_DEFAULTS
    mode = normalize_syn_mode(syn_mode)
    if mode == "per_edge":
        if n_edges is None:
            raise TypeError("per_edge edge_weight requires n_edges from network ScatterConn")
        n_edges = int(n_edges)
        return _with_part(
            {"name": "edge_weight", "count": n_edges, "kind": "edge", **D["edge_weight"]},
            _part_indi_all(n_edges),
        )
    if n_pairs is None:
        raise TypeError("type_pair syn_strength requires n_pairs from network ScatterConn")
    n_pairs = int(n_pairs)
    return _with_part(
        {"name": "syn_strength", "count": n_pairs, "kind": "edge_pair", **D["syn_strength"]},
        _part_indi_all(n_pairs),
    )


def build_conductance_schema(
    n_types, type_names=None, n_pairs=None, *, syn_mode=SYN_MODE_DEFAULT, n_edges=None,
):
    if type_names is None:
        raise TypeError("conductance schema requires type_names from network")
    type_names = list(type_names)
    name_to_i = {str(n): i for i, n in enumerate(type_names)}
    ih_gmax = [name_to_i[n] for n in DEFAULT_IH_GMAX_INDI_NAMES]
    D = PARAM_DEFAULTS
    indi_all = _part_indi_all(n_types)
    fixed_all = _part_fixed_all(n_types)
    shared_all = _part_shared_all(n_types)
    ih_gmax_part = _part_indi_subset_fixed_rest(n_types, ih_gmax)
    return [
        _with_part({"name": "in_gain", "count": n_types, "kind": "full", **D["in_gain"]}, fixed_all),
        _with_part({"name": "out_gain", "count": n_types, "kind": "full", **D["out_gain"]}, indi_all),
        _syn_segment(syn_mode, n_pairs, n_edges),
        _with_part({"name": "v_th", "count": n_types, "kind": "full", **D["v_th"]}, fixed_all),
        _with_part({"name": "out_scale", "count": n_types, "kind": "output", **D["out_scale"]}, indi_all),
        _with_part({"name": "Ih_gmax", "count": n_types, "kind": "full", **D["Ih_gmax"]}, ih_gmax_part),
        _with_part({"name": "Ih_gmax_off", "count": n_types, "kind": "full", **D["Ih_gmax_off"]}, ih_gmax_part),
        _with_part({"name": "Ih_midv", "count": n_types, "kind": "full", **D["Ih_midv"]}, shared_all),
        _with_part({"name": "Ih_slope", "count": n_types, "kind": "full", **D["Ih_slope"]}, shared_all),
        _with_part({"name": "tau_midv", "count": n_types, "kind": "full", **D["tau_midv"]}, shared_all),
        _with_part({"name": "Ih_midv_off", "count": n_types, "kind": "full", **D["Ih_midv_off"]}, shared_all),
        _with_part({"name": "Ih_slope_off", "count": n_types, "kind": "full", **D["Ih_slope_off"]}, shared_all),
        _with_part({"name": "tau_midv_off", "count": n_types, "kind": "full", **D["tau_midv_off"]}, shared_all),
    ]


def build_hp_lp_schema(
    n_types, type_names=None, n_pairs=None, *, syn_mode=SYN_MODE_DEFAULT, n_edges=None,
):
    """HP-then-membrane-LP: τ_HP on slow average a, τ_lp on V, drive G(X−a)."""
    if type_names is None:
        raise TypeError("hp_lp schema requires type_names from network")
    type_names = list(type_names)
    name_to_i = {str(n): i for i, n in enumerate(type_names)}
    hp_gain_indi = [name_to_i[n] for n in DEFAULT_IH_GMAX_INDI_NAMES]
    D = PARAM_DEFAULTS
    indi_all = _part_indi_all(n_types)
    fixed_all = _part_fixed_all(n_types)
    hp_gain_part = _part_indi_subset_fixed_rest(n_types, hp_gain_indi)
    return [
        _with_part({"name": "in_gain", "count": n_types, "kind": "full", **D["in_gain"]}, fixed_all),
        _with_part({"name": "out_gain", "count": n_types, "kind": "full", **D["out_gain"]}, indi_all),
        _syn_segment(syn_mode, n_pairs, n_edges),
        _with_part({"name": "out_scale", "count": n_types, "kind": "output", **D["out_scale"]}, indi_all),
        _with_part({"name": "tau_lp", "count": n_types, "kind": "full", **D["tau_lp"]}, indi_all),
        _with_part({"name": "tau_hp", "count": n_types, "kind": "full", **D["tau_hp"]}, indi_all),
        _with_part({"name": "bias", "count": n_types, "kind": "full", **D["bias"]}, indi_all),
        _with_part({"name": "hp_gain", "count": n_types, "kind": "full", **D["hp_gain"]}, indi_all),
    ]


def default_schema(model: str, backend, *, syn_mode=SYN_MODE_DEFAULT) -> list:
    """Fresh parameter schema for ``model`` on the given backend."""
    if model not in KNOWN_MODELS:
        raise ValueError(f"unknown model {model!r}; expected one of {KNOWN_MODELS}")
    n = backend.n_types
    import FiveCol_MedSim_Pytorch as fc

    type_names = fc.type_unit_names(backend)
    mode = normalize_syn_mode(syn_mode)
    n_pairs = getattr(backend.conn, "n_pairs", None)
    n_edges = getattr(backend.conn, "n_edges", None)
    if mode == "per_edge":
        if n_edges is None:
            raise TypeError(f"{model} edge_weight requires network ScatterConn backend")
        kw = dict(syn_mode=mode, n_edges=n_edges, n_pairs=n_pairs)
    else:
        if n_pairs is None:
            raise TypeError(f"{model} syn_strength requires network ScatterConn backend")
        kw = dict(syn_mode=mode, n_pairs=n_pairs, n_edges=n_edges)
    if model == "hp_lp":
        return build_hp_lp_schema(n, type_names=type_names, **kw)
    return build_conductance_schema(n, type_names=type_names, **kw)


def conductance_schema(model_backend, schema=None, ih_off=IH_OFF_DEFAULT, *, syn_mode=None):
    """Conductance parameter schema with ``ih_off`` segment selection applied."""
    if schema is not None:
        base = list(schema)
    elif syn_mode is not None:
        base = default_schema("conductance", model_backend, syn_mode=syn_mode)
    else:
        base = default_schema("conductance", model_backend)
    return apply_ih_off_mode(base, ih_off)
