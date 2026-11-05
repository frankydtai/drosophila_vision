# -*- coding: utf-8 -*-
"""Fixed curated parameters for the Point-HH network.

The curated tables remain the source of truth.  This module only converts their
density/unit conventions to the neuron kernel's pF/nS convention and aligns
them to the loaded network's node/edge order.
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch


MEMBRANE_COLUMNS = ("gNa", "gK", "gLeak", "eNa", "eK", "eLeak")
SYNAPSE_COLUMNS = ("gS", "e_syn", "v_th", "k_minus", "delta")


@dataclass(frozen=True)
class PointHHContext:
    params: Mapping[str, torch.Tensor]
    type_rows: tuple[dict, ...]
    edge_rows: tuple[dict, ...]
    provenance: dict


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


__all__ = [
    "PointHHContext",
    "build_point_hh_context",
    "save_point_hh_context",
]
