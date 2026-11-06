# -*- coding: utf-8 -*-
"""Differentiable single-compartment Hodgkin-Huxley dynamics.

The numerical interface uses ``ms``, ``mV``, ``pA``, ``pF`` and ``nS``.
Membrane parameters are per node and ionotropic parameters are per directed
edge.  Optional V6 gene mapping supplies conductances before the dynamics.

``pre_steady`` / ``step`` form the model-driver API used by
``neuron.forward``. ``initial_state`` is also public so the kernel can be
tested independently of a training session.
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch


MEMBRANE_COLUMNS = ("gNa", "gK", "gLeak", "eNa", "eK", "eLeak")
SYNAPSE_COLUMNS = ("gS", "e_syn", "v_th", "k_minus", "delta")

# Ripple mapping_v6.ACTIVE_NO_IH_ION_LINK_SPECS and transmitter.ACTIVE_GS_LINK_SPECS.
ION_GENE_LINKS = tuple(
    (parameter, gene)
    for parameter, genes in (
        ("gNa", ("para", "tipE", "Teh1", "Teh2", "Teh3", "Teh4")),
        ("gK", ("Sh", "Shal", "Shab", "Shaw", "eag", "sei", "Elk")),
        ("gLeak", ("Ork1", "Irk1", "Irk2", "Irk3", "ClC-a", "pHCl-1")),
    )
    for gene in genes
)
SYNAPSE_GENE_LINKS = tuple(
    (gene, side, transmitters)
    for genes, side, transmitters in (
        (("brp", "Rim", "Liprin-alpha", "Syt1", "Syt7", "Sytbeta", "nSyb",
          "Syx1A", "Snap25", "Csp", "cpx", "shi", "Syn", "Syp", "Vap33"),
         "pre", ("ACH", "GABA", "Histamine", "GLUT")),
        (("ChAT", "VAChT"), "pre", ("ACH",)),
        (("nAChRalpha1", "nAChRalpha2", "nAChRbeta1"), "post", ("ACH",)),
        (("Gad1", "VGAT"), "pre", ("GABA",)),
        (("Rdl", "Lcch3", "Grd"), "post", ("GABA",)),
        (("Hdc",), "pre", ("Histamine",)),
        (("ort", "HisCl1"), "post", ("Histamine",)),
        (("VGlut",), "pre", ("GLUT",)),
        (("GluClalpha",), "post", ("GLUT",)),
    )
    for gene in genes
)
GENE_PARAM_NAMES = tuple(
    f"gene_{parameter}_{gene}" for parameter, gene in ION_GENE_LINKS
) + tuple(f"gene_gS_{gene}" for gene, _side, _transmitters in SYNAPSE_GENE_LINKS)


@dataclass(frozen=True)
class PointHHContext:
    params: Mapping[str, torch.Tensor]
    type_rows: tuple[dict, ...]
    edge_rows: tuple[dict, ...]
    provenance: dict
    gene_design: Mapping[str, torch.Tensor] | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise_nt(value: object) -> str:
    token = str(value).strip()
    upper = token.upper()
    aliases = {
        "ACETYLCHOLINE": "ACH",
        "CHOLINERGIC": "ACH",
        "GLUTAMATE": "GLUT",
        "GLUTAMATERGIC": "GLUT",
        "HISTAMINE": "Histamine",
        "HISTAMINERGIC": "Histamine",
        "GABAERGIC": "GABA",
    }
    return aliases.get(upper, upper)


def _ion_rule_key(cell: str, available: set[str]) -> tuple[str, str]:
    if cell in available:
        return cell, "exact"
    # Preserve the historical T4/T5 curated-bundle behavior.
    if cell.startswith("T4") and "T4a" in available:
        return "T4a", "direction_proxy"
    if cell == "L2" and "L3" in available:
        return "L3", "historical_L2_proxy"
    raise KeyError(f"curated Point-HH ion table has no rule for {cell!r}")


def _read_ion_rules(path: Path) -> pd.DataFrame:
    rules = pd.read_csv(path)
    required = {"neuron_type", *MEMBRANE_COLUMNS}
    missing = required - set(rules.columns)
    if missing:
        raise ValueError(f"{path}: missing ion columns {sorted(missing)}")
    if rules["neuron_type"].astype(str).duplicated().any():
        raise ValueError(f"{path}: duplicate neuron_type rows")
    return rules.set_index(rules["neuron_type"].astype(str).str.strip())


def _read_synapse_rules(path: Path) -> list[dict]:
    frame = pd.read_csv(path, skipinitialspace=True)
    required = {"pre_type", "post_type", "nt_type", *SYNAPSE_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing synapse columns {sorted(missing)}")
    rows = []
    seen = set()
    for source_row, row in frame.iterrows():
        pre = str(row["pre_type"]).strip()
        post = str(row["post_type"]).strip()
        nt = _normalise_nt(row["nt_type"])
        key = (pre, post, nt)
        if key in seen:
            raise ValueError(f"{path}: duplicate synapse rule {key}")
        seen.add(key)
        values = {
            column: float(row[column])
            for column in SYNAPSE_COLUMNS
        }
        if values["gS"] < 0 or values["delta"] <= 0 or values["k_minus"] <= 0:
            raise ValueError(f"{path}: invalid synapse rule {key}")
        rows.append(
            {
                "pre_type": pre,
                "post_type": post,
                "nt_type": nt,
                "source_row": int(source_row) + 2,
                **values,
            }
        )
    return rows


def _network_connection_rows(connectome, path: Path) -> pd.DataFrame:
    source = connectome.conn.source_idxs.detach().cpu().numpy()
    target = connectome.conn.target_idxs.detach().cpu().numpy()
    required_pairs = pd.DataFrame(
        {
            "edge_index": np.arange(len(source), dtype=np.int64),
            "pre_root_id": np.asarray(connectome.node_ids, dtype=np.int64)[source],
            "post_root_id": np.asarray(connectome.node_ids, dtype=np.int64)[target],
        }
    )
    columns = [
        "pre_root_id", "post_root_id", "syn_count", "nt_type",
    ]
    connections = pd.read_csv(path, usecols=columns)
    selected = connections.merge(
        required_pairs, on=["pre_root_id", "post_root_id"], how="inner",
        validate="many_to_one",
    )
    if selected.empty and len(required_pairs):
        raise ValueError(f"{path}: none of the network edges were found")

    count_by_nt = (
        selected.groupby(["edge_index", "nt_type"], sort=False)
        .size()
        .reset_index(name="n")
        .sort_values("n")
        .groupby("edge_index", sort=False)
        .tail(1)
        .set_index("edge_index")["nt_type"]
    )
    syn_count = selected.groupby("edge_index", sort=False)["syn_count"].sum()
    result = required_pairs.set_index("edge_index")
    result["raw_nt_type"] = count_by_nt
    result["source_n_syn"] = syn_count
    missing = result.index[result["raw_nt_type"].isna()].tolist()
    if missing:
        raise ValueError(
            f"{path}: missing local connection rows for {len(missing)} network edges"
        )
    expected = connectome.edge_n_syn.detach().cpu().numpy()
    actual = result["source_n_syn"].to_numpy(dtype=np.float64)
    if not np.allclose(actual, expected, rtol=0, atol=1e-9):
        bad = np.flatnonzero(~np.isclose(actual, expected, rtol=0, atol=1e-9))
        raise ValueError(
            f"{path}: n_syn differs from network.json on {len(bad)} edges; "
            f"first edge={int(bad[0])}, table={actual[bad[0]]}, "
            f"network={expected[bad[0]]}"
        )
    return result.reset_index()


def _resolve_synapse_rule(
    rules: list[dict], pre: str, post: str, nt: str,
) -> tuple[dict | None, str]:
    for rule in rules:
        if (
            rule["pre_type"] == pre
            and rule["post_type"] == post
            and rule["nt_type"] == nt
        ):
            return rule, "exact"
    for rule in rules:
        if rule["pre_type"] == pre and rule["nt_type"] == nt:
            return rule, "same_pre_transmitter_fallback"
    return None, "unresolved_zero"


def build_point_hh_context(
    connectome,
    *,
    ion_rules_path,
    synapse_rules_path,
    connections_path,
    area_um2: float = 100.0,
    capacitance_uF_cm2: float = 1.0,
) -> PointHHContext:
    """Build fixed node/edge tensors in the exact loaded network order."""
    ion_path = Path(ion_rules_path).expanduser().resolve()
    synapse_path = Path(synapse_rules_path).expanduser().resolve()
    connection_path = Path(connections_path).expanduser().resolve()
    for path in (ion_path, synapse_path, connection_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    area_um2 = float(area_um2)
    capacitance_uF_cm2 = float(capacitance_uF_cm2)
    if area_um2 <= 0 or capacitance_uF_cm2 <= 0:
        raise ValueError("Point-HH area and capacitance must be positive")

    ion_rules = _read_ion_rules(ion_path)
    available = set(map(str, ion_rules.index))
    type_rows = []
    cell_values = {column: [] for column in MEMBRANE_COLUMNS}
    cell_values["cap"] = []
    for cell in connectome.cells:
        key, assignment = _ion_rule_key(str(cell), available)
        source = ion_rules.loc[key]
        type_row = {
            "neuron_type": str(cell),
            "ion_rule_key": key,
            "assignment": assignment,
            "area_um2": area_um2,
            "capacitance_uF_cm2": capacitance_uF_cm2,
            "cap_pF": capacitance_uF_cm2 * area_um2 * 0.01,
        }
        for column in MEMBRANE_COLUMNS:
            density = float(source[column])
            value = density * area_um2 * 10.0 if column.startswith("g") else density
            cell_values[column].append(value)
            type_row[column] = value
        cell_values["cap"].append(type_row["cap_pF"])
        type_rows.append(type_row)

    node_cells = connectome.node_cells
    params = {
        name: torch.as_tensor(values, dtype=connectome.edge_n_syn.dtype,
                              device=connectome.device).index_select(0, node_cells)
        for name, values in cell_values.items()
    }

    rules = _read_synapse_rules(synapse_path)
    connection_rows = _network_connection_rows(connectome, connection_path)
    edge_n_syn = connectome.edge_n_syn.detach().cpu().numpy()
    network_sign = torch.sign(connectome.conn.w_signed).detach().cpu().numpy()
    edge_values = {
        "gS": [], "e_syn": [], "v_th_syn": [],
        "delta_syn": [], "k_minus_syn": [],
    }
    edge_rows = []
    for edge_index, connection in connection_rows.iterrows():
        pre = connectome.edge_source_cells[edge_index]
        post = connectome.edge_target_cells[edge_index]
        raw_nt = _normalise_nt(connection["raw_nt_type"])
        model_nt = "Histamine" if pre == "R1-6" else raw_nt
        rule, resolution = _resolve_synapse_rule(rules, pre, post, model_nt)
        if rule is None:
            values = {
                "gS": 0.0,
                "e_syn": 0.0,
                "v_th": -40.0,
                "delta": 1.0,
                "k_minus": 0.1,
                "source_row": None,
            }
        else:
            values = rule
        # Curated gS is uS per contact.  The kernel consumes nS per aggregated
        # network edge, so multiply by 1000 and by the original contact count.
        g_s_nS = float(values["gS"]) * 1000.0 * float(edge_n_syn[edge_index])
        edge_values["gS"].append(g_s_nS)
        edge_values["e_syn"].append(float(values["e_syn"]))
        edge_values["v_th_syn"].append(float(values["v_th"]))
        edge_values["delta_syn"].append(float(values["delta"]))
        edge_values["k_minus_syn"].append(float(values["k_minus"]))
        model_sign = 0 if g_s_nS == 0 else (1 if float(values["e_syn"]) > -40 else -1)
        edge_rows.append(
            {
                "edge_index": int(edge_index),
                "src": int(connection["pre_root_id"]),
                "tar": int(connection["post_root_id"]),
                "pre_type": pre,
                "post_type": post,
                "n_syn": float(edge_n_syn[edge_index]),
                "raw_nt_type": raw_nt,
                "model_nt_type": model_nt,
                "nt_overridden": bool(raw_nt != model_nt),
                "resolution": resolution,
                "rule_source_row": values["source_row"],
                "network_sign": int(network_sign[edge_index]),
                "model_sign": int(model_sign),
                "sign_differs": bool(model_sign != 0 and model_sign != network_sign[edge_index]),
                "gS_nS": g_s_nS,
                "e_syn_mV": float(values["e_syn"]),
                "v_th_syn_mV": float(values["v_th"]),
                "delta_syn_mV": float(values["delta"]),
                "k_minus_syn_per_ms": float(values["k_minus"]),
            }
        )
    params.update(
        {
            name: torch.as_tensor(values, dtype=connectome.edge_n_syn.dtype,
                                  device=connectome.device)
            for name, values in edge_values.items()
        }
    )
    resolution_counts: dict[str, int] = {}
    for row in edge_rows:
        resolution_counts[row["resolution"]] = (
            resolution_counts.get(row["resolution"], 0) + 1
        )
    provenance = {
        "mapping_version": "point_hh_curated_v1",
        "units": {"time": "ms", "voltage": "mV", "current": "pA",
                  "capacitance": "pF", "conductance": "nS"},
        "area_um2": area_um2,
        "capacitance_uF_cm2": capacitance_uF_cm2,
        "contact_scaling": "gS_nS = curated_gS_uS * 1000 * network_n_syn",
        "ion_fallbacks": {"L2": "L3", "T4b/T4c/T4d": "T4a"},
        "unresolved_synapse_policy": "gS=0",
        "resolution_counts": resolution_counts,
        "n_edge": int(connectome.conn.n_edge),
        "sources": {
            "ion_rules": {"path": str(ion_path), "sha256": _sha256(ion_path)},
            "synapse_rules": {
                "path": str(synapse_path), "sha256": _sha256(synapse_path),
            },
            "connections": {
                "path": str(connection_path), "sha256": _sha256(connection_path),
            },
            "network": str(connectome.source_json),
        },
    }
    return PointHHContext(
        params=params,
        type_rows=tuple(type_rows),
        edge_rows=tuple(edge_rows),
        provenance=provenance,
    )


def save_point_hh_context(context: PointHHContext, output_dir) -> None:
    """Persist auditable fixed parameters beside a no-training run."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "point_hh_context.json").open("w") as handle:
        json.dump(context.provenance, handle, indent=2)
        handle.write("\n")
    for name, rows in (
        ("point_hh_type_params.csv", context.type_rows),
        ("point_hh_edge_params.csv", context.edge_rows),
    ):
        if not rows:
            continue
        with (output / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)



STATE_KEYS = ("v", "m", "h", "n", "s")
_MEMBRANE_PARAM_NAMES = ("cap", "gNa", "gK", "gLeak", "eNa", "eK", "eLeak")
_SYNAPSE_PARAM_NAMES = (
    "gS", "e_syn", "v_th_syn", "delta_syn", "k_minus_syn",
)
_PREPARED_KEY = "_point_hh_prepared"


def stable_vtrap(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Return ``x / (exp(x / scale) - 1)`` with its analytic zero limit.

    The regular branch receives a safe nonzero argument.  This is important in
    PyTorch because an unselected ``where`` branch containing ``0 / 0`` can
    still contaminate gradients.
    """
    scale_t = x.new_tensor(float(scale))
    z = x / scale_t
    small = torch.abs(z) < 1.0e-7
    safe_z = torch.where(small, torch.ones_like(z), z)
    regular = x / torch.expm1(safe_z)
    series = scale_t * (1.0 - z / 2.0 + z.square() / 12.0)
    return torch.where(small, series, regular)


def gate_rates(v: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Stable classic-HH ``alpha/beta`` rates in inverse milliseconds."""
    alpha_m = 0.1 * stable_vtrap(-(v + 40.0), 10.0)
    beta_m = 4.0 * torch.exp(-(v + 65.0) / 18.0)
    alpha_h = 0.07 * torch.exp(-(v + 65.0) / 20.0)
    beta_h = 1.0 / (torch.exp(-(v + 35.0) / 10.0) + 1.0)
    alpha_n = 0.01 * stable_vtrap(-(v + 55.0), 10.0)
    beta_n = 0.125 * torch.exp(-(v + 65.0) / 80.0)
    return alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n


def steady_gates(v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``m_inf, h_inf, n_inf`` at voltage ``v``."""
    am, bm, ah, bh, an, bn = gate_rates(v)
    return am / (am + bm), ah / (ah + bh), an / (an + bn)


def _rush_larsen(
    value: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    dt_ms: float,
) -> torch.Tensor:
    rate = alpha + beta
    inf = alpha / rate
    updated = inf + (value - inf) * torch.exp(-float(dt_ms) * rate)
    return torch.clamp(updated, 0.0, 1.0)


def _as_param(
    params: Mapping[str, torch.Tensor],
    name: str,
    reference: torch.Tensor,
    length: int,
) -> torch.Tensor:
    if name not in params:
        raise KeyError(f"point_hh requires parameter {name!r}")
    if params.get(_PREPARED_KEY, False):
        return params[name]
    value = torch.as_tensor(params[name], dtype=reference.dtype, device=reference.device)
    if value.numel() == 1:
        return value.reshape(1).expand(int(length))
    value = value.reshape(-1)
    if value.numel() != int(length):
        raise ValueError(
            f"point_hh {name} length {value.numel()} != expected {int(length)}"
        )
    return value


def _edge_indices(
    session,
    device,
    params: Mapping[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if params is not None and params.get(_PREPARED_KEY, False):
        return params["_source_idx"], params["_target_idx"]
    conn = session.connectome.conn
    source = torch.as_tensor(conn.source_idxs, dtype=torch.long, device=device)
    target = torch.as_tensor(conn.target_idxs, dtype=torch.long, device=device)
    if source.shape != target.shape:
        raise ValueError("point_hh source/target edge arrays must have equal shape")
    return source, target


def _edge_params(
    params: Mapping[str, torch.Tensor],
    reference: torch.Tensor,
    n_edge: int,
) -> tuple[torch.Tensor, ...] | None:
    if params.get(_PREPARED_KEY, False):
        if "gS" not in params:
            return None
        return tuple(params[name] for name in _SYNAPSE_PARAM_NAMES)
    present = [
        name in params
        for name in _SYNAPSE_PARAM_NAMES
    ]
    if not any(present):
        return None
    if not all(present):
        missing = [
            name
            for name, exists in zip(
                _SYNAPSE_PARAM_NAMES,
                present,
            )
            if not exists
        ]
        raise KeyError(f"incomplete point_hh synapse parameters: missing {missing}")
    values = tuple(
        _as_param(params, name, reference, n_edge)
        for name in _SYNAPSE_PARAM_NAMES
    )
    if bool((values[0] < 0).any()):
        raise ValueError("point_hh gS must be non-negative")
    if bool((values[3] <= 0).any()):
        raise ValueError("point_hh delta_syn must be positive")
    if bool((values[4] <= 0).any()):
        raise ValueError("point_hh k_minus_syn must be positive")
    return values


def prepare_params(
    params: Mapping[str, torch.Tensor],
    reference: torch.Tensor,
    session,
) -> dict[str, torch.Tensor]:
    """Materialize and validate fixed-size tensors once per full forward."""
    if params.get(_PREPARED_KEY, False):
        return params
    n_node = int(session.connectome.n_node)
    prepared = {
        name: _as_param(params, name, reference, n_node)
        for name in _MEMBRANE_PARAM_NAMES
    }
    if bool((prepared["cap"] <= 0).any()):
        raise ValueError("point_hh cap must be positive")
    if bool(
        (prepared["gNa"] < 0).any()
        or (prepared["gK"] < 0).any()
        or (prepared["gLeak"] < 0).any()
    ):
        raise ValueError("point_hh membrane conductances must be non-negative")
    source, target = _edge_indices(session, reference.device)
    syn_params = _edge_params(params, reference, int(source.numel()))
    if syn_params is not None:
        prepared.update(zip(_SYNAPSE_PARAM_NAMES, syn_params))
    prepared["_source_idx"] = source
    prepared["_target_idx"] = target
    prepared[_PREPARED_KEY] = True
    return prepared


def release_steady(
    pre_v: torch.Tensor,
    v_th_syn: torch.Tensor,
    delta_syn: torch.Tensor,
) -> torch.Tensor:
    """Steady ionotropic open probability for each batch and edge."""
    return torch.sigmoid((pre_v - v_th_syn) / delta_syn)


def _release_step(
    s: torch.Tensor,
    pre_v: torch.Tensor,
    v_th_syn: torch.Tensor,
    delta_syn: torch.Tensor,
    k_minus_syn: torch.Tensor,
    dt_ms: float,
) -> torch.Tensor:
    s_inf = release_steady(pre_v, v_th_syn, delta_syn)
    tau_ms = (1.0 - s_inf) / k_minus_syn
    updated = s_inf + (s - s_inf) * torch.exp(
        -float(dt_ms) / torch.clamp(tau_ms, min=torch.finfo(s.dtype).tiny)
    )
    return torch.clamp(updated, 0.0, 1.0)


def _synaptic_terms(
    v: torch.Tensor,
    s: torch.Tensor,
    params: Mapping[str, torch.Tensor],
    session,
    *,
    return_current: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return node ``sum(g)``, ``sum(gE)`` and optional inward synaptic current."""
    source, target = _edge_indices(session, v.device, params)
    n_edge = int(source.numel())
    syn_params = _edge_params(params, v, n_edge)
    if syn_params is None or n_edge == 0:
        zeros = torch.zeros_like(v)
        if s.shape[-1] != n_edge:
            raise ValueError(f"point_hh s has {s.shape[-1]} edges, expected {n_edge}")
        return zeros, zeros, zeros if return_current else None
    g_s, e_syn, _v_th, _delta, _k_minus = syn_params
    if s.shape != v.shape[:-1] + (n_edge,):
        raise ValueError(
            f"point_hh s shape {tuple(s.shape)} != {tuple(v.shape[:-1] + (n_edge,))}"
        )
    edge_g = s * g_s
    sum_g = torch.zeros_like(v)
    sum_g_e = torch.zeros_like(v)
    target_b = target.expand(edge_g.shape)
    sum_g.scatter_add_(-1, target_b, edge_g)
    sum_g_e.scatter_add_(-1, target_b, edge_g * e_syn)
    if not return_current:
        return sum_g, sum_g_e, None
    post_v = v.index_select(-1, target)
    edge_i_in = edge_g * (e_syn - post_v)
    i_syn = torch.zeros_like(v)
    i_syn.scatter_add_(-1, target_b, edge_i_in)
    return sum_g, sum_g_e, i_syn


def initial_state(
    v: torch.Tensor,
    *,
    n_edge: int = 0,
    s: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Construct a state from voltage, using steady membrane gates."""
    m, h, n = steady_gates(v)
    if s is None:
        s = v.new_zeros(v.shape[:-1] + (int(n_edge),))
    return {"v": v, "m": m, "h": h, "n": n, "s": s}


def detach_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Detach every Point-HH state tensor."""
    return {name: state[name].detach() for name in STATE_KEYS}


def step(
    state: Mapping[str, torch.Tensor],
    params: Mapping[str, torch.Tensor],
    i_sti: torch.Tensor,
    session,
    *,
    delta_ms: float,
    return_component: bool = False,
):
    """Advance one Point-HH model step.

    Gates and release use exact constant-voltage updates.  Voltage uses a
    backward-Euler conductance update with the post-gate/post-release
    conductances frozen over the model step.
    """
    dt = float(delta_ms)
    if dt <= 0.0:
        raise ValueError(f"point_hh delta_ms must be positive; got {dt}")
    v = state["v"]
    params = prepare_params(params, v, session)
    m, h, n, s = (state[name] for name in ("m", "h", "n", "s"))
    n_node = int(v.shape[-1])
    i_sti = torch.as_tensor(i_sti, dtype=v.dtype, device=v.device)
    if i_sti.shape != v.shape:
        i_sti = torch.broadcast_to(i_sti, v.shape)

    cap = _as_param(params, "cap", v, n_node)
    g_na = _as_param(params, "gNa", v, n_node)
    g_k = _as_param(params, "gK", v, n_node)
    g_leak = _as_param(params, "gLeak", v, n_node)
    e_na = _as_param(params, "eNa", v, n_node)
    e_k = _as_param(params, "eK", v, n_node)
    e_leak = _as_param(params, "eLeak", v, n_node)

    am, bm, ah, bh, an, bn = gate_rates(v)
    m_new = _rush_larsen(m, am, bm, dt)
    h_new = _rush_larsen(h, ah, bh, dt)
    n_new = _rush_larsen(n, an, bn, dt)

    source, _target = _edge_indices(session, v.device, params)
    n_edge = int(source.numel())
    syn_params = _edge_params(params, v, n_edge)
    if syn_params is None or n_edge == 0:
        if s.shape[-1] != n_edge:
            raise ValueError(f"point_hh s has {s.shape[-1]} edges, expected {n_edge}")
        s_new = s
    else:
        _g_s, _e_syn, v_th, delta, k_minus = syn_params
        pre_v = v.index_select(-1, source)
        s_new = _release_step(s, pre_v, v_th, delta, k_minus, dt)

    g_na_open = g_na * m_new.pow(3) * h_new
    g_k_open = g_k * n_new.pow(4)
    syn_g, syn_g_e, _ = _synaptic_terms(
        v, s_new, params, session, return_current=False,
    )
    sum_g = g_na_open + g_k_open + g_leak + syn_g
    sum_g_e = (
        g_na_open * e_na + g_k_open * e_k + g_leak * e_leak + syn_g_e
    )
    dt_over_cap = dt / cap
    v_new = (v + dt_over_cap * (i_sti + sum_g_e)) / (
        1.0 + dt_over_cap * sum_g
    )
    v_min = float(getattr(session, "point_hh_v_min", -120.0))
    v_max = float(getattr(session, "point_hh_v_max", 80.0))
    v_new = torch.clamp(v_new, v_min, v_max)
    new_state = {"v": v_new, "m": m_new, "h": h_new, "n": n_new, "s": s_new}
    if not return_component:
        return new_state

    i_na = g_na_open * (v_new - e_na)
    i_k = g_k_open * (v_new - e_k)
    i_leak = g_leak * (v_new - e_leak)
    _syn_g, _syn_g_e, i_syn = _synaptic_terms(v_new, s_new, params, session)
    component = {
        "i_na": i_na,
        "i_k": i_k,
        "i_leak": i_leak,
        "i_syn": i_syn,
        "i_sti": i_sti,
        "g_na": g_na_open,
        "g_k": g_k_open,
        "g_leak": g_leak.expand_as(v_new),
        "s": s_new,
        "current_residual": (
            cap * (v_new - v) / dt
            - (i_sti - i_na - i_k - i_leak + i_syn)
        ),
    }
    return new_state, component


def _dc_state_from_v(
    v: torch.Tensor,
    params: Mapping[str, torch.Tensor],
    i_sti: torch.Tensor,
    session,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    n_node = int(v.shape[-1])
    cap_dummy = params.get("cap")
    del cap_dummy  # Capacitance does not enter a DC balance.
    g_na = _as_param(params, "gNa", v, n_node)
    g_k = _as_param(params, "gK", v, n_node)
    g_leak = _as_param(params, "gLeak", v, n_node)
    e_na = _as_param(params, "eNa", v, n_node)
    e_k = _as_param(params, "eK", v, n_node)
    e_leak = _as_param(params, "eLeak", v, n_node)
    m, h, n = steady_gates(v)
    source, _target = _edge_indices(session, v.device, params)
    n_edge = int(source.numel())
    syn_params = _edge_params(params, v, n_edge)
    if syn_params is None or n_edge == 0:
        s = v.new_zeros(v.shape[:-1] + (n_edge,))
    else:
        _g_s, _e_syn, v_th, delta, _k_minus = syn_params
        s = release_steady(v.index_select(-1, source), v_th, delta)
    state = {"v": v, "m": m, "h": h, "n": n, "s": s}
    g_na_open = g_na * m.pow(3) * h
    g_k_open = g_k * n.pow(4)
    syn_g, syn_g_e, _ = _synaptic_terms(
        v, s, params, session, return_current=False,
    )
    sum_g = g_na_open + g_k_open + g_leak + syn_g
    if bool((sum_g <= 0).any()):
        raise ValueError("point_hh DC balance requires positive total conductance")
    sum_g_e = (
        g_na_open * e_na + g_k_open * e_k + g_leak * e_leak + syn_g_e
    )
    return (i_sti + sum_g_e) / sum_g, state


def pre_steady(session, params, n_b, i_sti=None):
    """Build the background-balanced Point-HH state at output sample zero."""
    if i_sti is None:
        raise TypeError("point_hh pre_steady requires i_sti")
    if i_sti.dim() == 2:
        i_sti = i_sti.unsqueeze(1)
    i0 = i_sti[:, 0, :]
    params = prepare_params(params, i0, session)
    n_node = int(session.connectome.n_node)
    if i0.shape != (int(n_b), n_node):
        raise ValueError(
            f"point_hh background current shape {tuple(i0.shape)} "
            f"!= {(int(n_b), n_node)}"
        )
    e_leak = _as_param(params, "eLeak", i0, n_node)
    v = e_leak.expand(int(n_b), n_node).clone()
    mode = str(session.pre_steady)
    if mode not in ("probe", "solve"):
        raise ValueError(f"point_hh pre_steady must be probe|solve; got {mode!r}")
    n_iter = 1 if mode == "probe" else int(session.pre_steady_n_iter)
    damp = 1.0 if mode == "probe" else float(session.pre_steady_damp)
    v_min = float(getattr(session, "point_hh_v_min", -120.0))
    v_max = float(getattr(session, "point_hh_v_max", 80.0))
    state = None
    for _ in range(n_iter):
        target, state = _dc_state_from_v(v, params, i0, session)
        v = torch.clamp(v + damp * (target - v), v_min, v_max)
    _target, state = _dc_state_from_v(v, params, i0, session)
    state["v"] = v
    return state


def _gene_options(opts):
    point_hh = opts.get("point_hh") or {}
    mode = point_hh.get("gene_mode", "off")
    if isinstance(mode, bool):
        mode = "on" if mode else "off"
    mode = str(mode).lower()
    if mode not in ("on", "off"):
        raise ValueError("point_hh.gene_mode must be on|off")
    if mode == "off":
        return None
    gene = point_hh.get("gene")
    if not isinstance(gene, dict) or gene.get("version") != "v6":
        raise ValueError("point_hh gene mode requires gene.version=v6")
    return gene


def prepare_training_schema(schema, connectome, opts):
    """Replace the four direct G parameters with globally shared gene links."""
    gene = _gene_options(opts)
    if gene is None:
        return schema
    from neuron.schema import build_param_spec

    entries = gene.get("params") or {}
    if set(entries) != set(GENE_PARAM_NAMES):
        raise ValueError("point_hh V6 requires the 19 ion and 30 synapse gene entries")
    schema = {
        param: spec for param, spec in schema.items()
        if param not in ("gNa", "gK", "gLeak", "gS")
    }
    for param in GENE_PARAM_NAMES:
        spec = schema.get(param)
        if spec is None:
            spec = build_param_spec(
                param, connectome.n_cell, entries[param], cells=connectome.cells,
            )
        if spec.get("shared") != list(range(connectome.n_cell)):
            raise ValueError(f"{param}: V6 requires one shared coefficient across all types")
        schema[param] = spec
    return schema


def _gene_features(path, genes, training_types):
    """Ripple V6 log1p features, fitted on distinct training source profiles."""
    frame = pd.read_csv(path)
    names = frame.iloc[:, 0].astype(str).str.strip()
    if names.duplicated().any():
        raise ValueError(f"{path}: duplicate gene rows")
    raw = frame.iloc[:, 1:].copy()
    raw.index = names
    raw.columns = [str(column).strip() for column in raw.columns]
    if raw.columns.duplicated().any():
        raise ValueError(f"{path}: duplicate expression type columns")
    missing = set(genes) - set(raw.index)
    if missing:
        raise ValueError(f"{path}: missing gene rows {sorted(missing)}")
    raw = raw.loc[list(genes)].apply(pd.to_numeric, errors="coerce").T
    if (raw.to_numpy() < 0).any():
        raise ValueError(f"{path}: negative gene expression")
    logged = np.log1p(raw)

    profile_sources = {}
    for source in logged.index:
        canonical = source.replace("R1.6", "R1-6").replace("R1_6", "R1-6")
        targets = ("R7", "R8") if canonical == "R7.8" else (canonical,)
        for target in targets:
            if target in profile_sources:
                raise ValueError(f"{path}: duplicate expression profile for {target}")
            profile_sources[target] = source
    sources = []
    for cell in training_types:
        cell = str(cell).strip()
        canonical = cell.replace("R1.6", "R1-6").replace("R1_6", "R1-6")
        source = cell if cell in logged.index else profile_sources.get(canonical)
        if source is None:
            raise ValueError(f"{path}: no training expression profile for {cell}")
        if source not in sources:
            sources.append(source)
    if not sources:
        raise ValueError("gene training_types must contain at least one profile")

    centers, scales, methods = [], [], []
    for gene in genes:
        values = logged.loc[sources, gene].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            raise ValueError(f"{path}: no finite training expression for {gene}")
        center = float(np.median(values))
        mad = 1.4826 * float(np.median(np.abs(values - center)))
        q25, q75 = np.quantile(values, [0.25, 0.75])
        candidates = (
            (mad, "mad"), ((q75 - q25) / 1.349, "iqr"),
            (float(np.std(values)), "standard_deviation"),
        )
        scale, method = next(
            ((float(value), label) for value, label in candidates if value > 1e-8),
            (1.0, "unit_constant_training_feature"),
        )
        centers.append(center)
        scales.append(scale)
        methods.append(method)
    scaled = (logged - np.asarray(centers)) / np.asarray(scales)
    features = pd.DataFrame(
        [scaled.loc[source].to_numpy(dtype=float) for source in profile_sources.values()],
        index=list(profile_sources), columns=list(genes),
    )
    return features, {
        "genes": list(genes), "training_source_profiles": sources,
        "centers": centers, "scales": scales, "scale_methods": methods,
    }


def _build_gene_context(context, connectome, gene):
    path = Path(gene["expression"]).expanduser().resolve()
    genes = tuple(dict.fromkeys(
        [name for _parameter, name in ION_GENE_LINKS]
        + [name for name, _side, _transmitters in SYNAPSE_GENE_LINKS]
    ))
    training_types = gene.get("training_types")
    if training_types is None:
        training_types = connectome.cells
    features, scaler = _gene_features(path, genes, training_types)
    cells = [str(cell) for cell in connectome.cells]
    missing = set(cells) - set(features.index)
    if missing:
        raise ValueError(f"{path}: missing expression types {sorted(missing)}")
    selected = features.loc[cells]
    if not np.isfinite(selected.to_numpy()).all():
        raise ValueError(f"{path}: non-finite gene features for network types")
    parameters = ("gNa", "gK", "gLeak")
    incidence = np.zeros((len(ION_GENE_LINKS), len(parameters)))
    for index, (parameter, _name) in enumerate(ION_GENE_LINKS):
        incidence[index, parameters.index(parameter)] = 1.0
    edge_features = np.empty((connectome.conn.n_edge, len(SYNAPSE_GENE_LINKS)))
    rows = context.edge_rows
    for index, (name, side, transmitters) in enumerate(SYNAPSE_GENE_LINKS):
        types = [row[f"{side}_type"] for row in rows]
        values = features.loc[types, name].to_numpy(dtype=float)
        mask = np.asarray([row["model_nt_type"] in transmitters for row in rows])
        edge_features[:, index] = values * mask
    folds = np.asarray([
        gene["max_fold_changes"][parameter] for parameter in (*parameters, "gS")
    ], dtype=float)
    if not np.isfinite(folds).all() or (folds <= 1.0).any():
        raise ValueError("gene max_fold_changes must be finite and greater than 1")
    reference = context.params["gNa"]
    design = {
        name: torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
        for name, value in {
            "ion_features": selected.loc[:, [name for _p, name in ION_GENE_LINKS]].to_numpy(),
            "ion_to_parameter": incidence,
            "edge_features": edge_features,
            "log_folds": np.log(folds),
        }.items()
    }
    provenance = dict(context.provenance)
    provenance["gene_mapping"] = {
        "ion_version": "v6", "synapse_version": "transmitter_v1",
        "coefficient_scope": "shared_across_all_neuron_types_and_edges",
        "coefficient_names": list(GENE_PARAM_NAMES),
        "baseline": "curated_conductances_with_contact_counts_already_applied",
        "expression": {"path": str(path), "sha256": _sha256(path)},
        "max_fold_changes": dict(gene["max_fold_changes"]),
        "feature_scaler": scaler,
    }
    return replace(context, provenance=provenance, gene_design=design)


def build_context(connectome, opts):
    """Load/cache this model's curated parameters without changing its rules."""
    point_hh = opts.get("point_hh")
    if not isinstance(point_hh, dict):
        raise ValueError("point_hh model requires train_opts['point_hh']")
    gene = _gene_options(opts)
    cache_key = json.dumps(point_hh, sort_keys=True)
    context = getattr(connectome, "_point_hh_context", None)
    if (
        context is None
        or getattr(connectome, "_point_hh_context_key", None) != cache_key
    ):
        context = build_point_hh_context(
            connectome,
            ion_rules_path=point_hh.get("ion_rules"),
            synapse_rules_path=point_hh.get("synapse_rules"),
            connections_path=point_hh.get("connections"),
            area_um2=point_hh.get("area_um2", 100.0),
            capacitance_uF_cm2=point_hh.get("capacitance_uF_cm2", 1.0),
        )
        if gene is not None:
            context = _build_gene_context(context, connectome, gene)
        connectome._point_hh_context = context
        connectome._point_hh_context_key = cache_key
    return context


def prepare_training_params(params, session):
    """Apply V6 genes or direct G values; contact counts enter gS only once."""
    context = getattr(session, "model_context", None)
    if context is None:
        raise ValueError("point_hh session has no fixed parameter context")
    decoded = params
    params = dict(context.params)
    params.update(decoded)
    if _gene_options(session.train_opts or {}) is not None:
        design = context.gene_design
        if design is None:
            raise ValueError("point_hh session has no V6 gene design")
        coefficients = torch.stack([decoded[name][0] for name in GENE_PARAM_NAMES])
        n_ion = len(ION_GENE_LINKS)
        signal = (
            design["ion_features"] * coefficients[:n_ion]
        ) @ design["ion_to_parameter"]
        folds = torch.exp(torch.tanh(signal) * design["log_folds"][:3])
        for index, parameter in enumerate(("gNa", "gK", "gLeak")):
            params[parameter] = context.params[parameter] * folds[:, index].index_select(
                0, session.connectome.node_cells,
            )
        signal = design["edge_features"] @ coefficients[n_ion:]
        params["gS"] = context.params["gS"] * torch.exp(
            design["log_folds"][3] * torch.tanh(signal)
        )
        return params
    if "gS" in decoded:
        params["gS"] = (
            decoded["gS"] * session.connectome.edge_n_syn
            * (context.params["gS"] > 0).to(decoded["gS"].dtype)
        )
    return params


save_context = save_point_hh_context


__all__ = [
    "PointHHContext",
    "build_point_hh_context",
    "save_point_hh_context",
    "build_context",
    "prepare_training_schema",
    "prepare_training_params",
    "save_context",
    "STATE_KEYS",
    "detach_state",
    "gate_rates",
    "initial_state",
    "prepare_params",
    "pre_steady",
    "release_steady",
    "stable_vtrap",
    "steady_gates",
    "step",
]
