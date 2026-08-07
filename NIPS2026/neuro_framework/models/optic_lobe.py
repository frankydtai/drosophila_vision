"""
Optic Lobe HH Network
=====================
Single-compartment Hodgkin-Huxley network for Drosophila optic lobe columns,
with per-neuron-type ion channel parameters and per-edge ionotropic synapse
parameters loaded from CSV rule files.

The model supports graded-potential neurons (typical of the Drosophila visual
system) and is compatible with ``TorchTrainer``.

Parameter files
---------------
``ion_channel_rules.csv``
    neuron_type, gNa, gK, gLeak, eLeak, eNa, eK

``synapse_rules.csv``
    pre_type, post_type, nt_type, gS, e_syn, v_th, k_minus, delta

Units (Jaxley-compatible)
-------------------------
  V : mV,  t : ms,  g : S/cm²  (convert to mS/cm² × 1e3 internally),
  C_m : μF/cm²,  I : μA/cm²
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)

__all__ = ["OpticLobeHHNetwork"]

_PKG_ROOT = Path(__file__).parents[2]

# ===================================================================
# Cell-type → ion channel parameter mapping
# ===================================================================

_CELL_TYPE_ALIAS = {
    "R1-R6": "R1-6",
    "R7_unclear": "R7", "R7p": "R7", "R7y": "R7",
    "R8_unclear": "R8", "R8p": "R8", "R8y": "R8",
    "T4b": "T4a", "T4c": "T4a", "T4d": "T4a",
    "T5a": "T4a", "T5b": "T4a", "T5c": "T4a", "T5d": "T4a",
}

_FALLBACK_TYPE = {
    "L2": "L1", "L4": "L3",
    "T1": "Mi1",
    "C2": "L5", "C3": "L5",
    "Tm1": "Tm3", "Tm2": "Tm3", "Tm4": "Tm3",
    "Tm9": "Tm3", "Tm20": "Tm3",
}

# Default HH params for genuinely unknown types (conservative non-spiking)
_DEFAULT_ION = {
    "gNa": 0.002, "gK": 0.001, "gLeak": 0.001,
    "eLeak": -55.0, "eNa": 50.0, "eK": -80.0,
}

# Default synapse params by NT class
_DEFAULT_SYN_BY_NT = {
    "acetylcholine": {"gS": 3e-4, "e_syn": 0.0,   "v_th": -45.0, "k_minus": 0.12, "delta": 3.5},
    "gaba":          {"gS": 5e-3, "e_syn": -72.0,  "v_th": -55.0, "k_minus": 0.03, "delta": 3.2},
    "glutamate":     {"gS": 5e-4, "e_syn": -70.0,  "v_th": -45.0, "k_minus": 0.08, "delta": 4.0},
    "histamine":     {"gS": 4e-4, "e_syn": -72.0,  "v_th": -60.0, "k_minus": 0.10, "delta": 3.0},
}

_NT_ALIAS = {
    "ach": "acetylcholine", "acetylcholine": "acetylcholine",
    "glut": "glutamate", "glutamate": "glutamate",
    "gaba": "gaba",
    "histamine": "histamine",
}


def _resolve_cell_type(ct: str) -> str:
    """Map a data cell_type string to the canonical rules key."""
    if ct in _CELL_TYPE_ALIAS:
        return _CELL_TYPE_ALIAS[ct]
    return ct


def _load_ion_rules(path: Union[str, Path]) -> Dict[str, Dict[str, float]]:
    df = pd.read_csv(path)
    rules = {}
    for _, row in df.iterrows():
        rules[row["neuron_type"]] = {
            "gNa": float(row["gNa"]),
            "gK": float(row["gK"]),
            "gLeak": float(row["gLeak"]),
            "eLeak": float(row["eLeak"]),
            "eNa": float(row["eNa"]),
            "eK": float(row["eK"]),
        }
    return rules


def _load_syn_rules(path: Union[str, Path]) -> Dict[Tuple[str, str], Dict[str, float]]:
    df = pd.read_csv(path)
    rules = {}
    for _, row in df.iterrows():
        key = (str(row["pre_type"]).strip(), str(row["post_type"]).strip())
        rules[key] = {
            "gS": float(row["gS"]),
            "e_syn": float(row["e_syn"]),
            "v_th": float(row["v_th"]),
            "k_minus": float(row["k_minus"]),
            "delta": float(row["delta"]),
        }
    return rules


def _get_ion_params(
    cell_type: str,
    ion_rules: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    ct = _resolve_cell_type(cell_type)
    if ct in ion_rules:
        return ion_rules[ct]
    if ct in _FALLBACK_TYPE:
        fb = _FALLBACK_TYPE[ct]
        if fb in ion_rules:
            return ion_rules[fb]
    return _DEFAULT_ION


def _get_syn_params(
    pre_type: str,
    post_type: str,
    pre_nt: str,
    syn_rules: Dict[Tuple[str, str], Dict[str, float]],
) -> Dict[str, float]:
    pre_canon = _resolve_cell_type(pre_type)
    post_canon = _resolve_cell_type(post_type)
    key = (pre_canon, post_canon)
    if key in syn_rules:
        return syn_rules[key]
    nt = pre_nt.lower().strip() if isinstance(pre_nt, str) else "acetylcholine"
    nt = _NT_ALIAS.get(nt, nt)
    if nt in _DEFAULT_SYN_BY_NT:
        return _DEFAULT_SYN_BY_NT[nt]
    return _DEFAULT_SYN_BY_NT["acetylcholine"]


# ===================================================================
# HH gating helpers (numerically stable)
# ===================================================================

def _safe_x_over_expm1(x: Tensor) -> Tensor:
    return torch.where(torch.abs(x) < 1e-6, torch.ones_like(x),
                       x / (1.0 - torch.exp(-x)))

def _alpha_m(V: Tensor) -> Tensor:
    return _safe_x_over_expm1((V + 40.0) / 10.0)

def _beta_m(V: Tensor) -> Tensor:
    return 4.0 * torch.exp(-(V + 65.0) / 18.0)

def _alpha_h(V: Tensor) -> Tensor:
    return 0.07 * torch.exp(-(V + 65.0) / 20.0)

def _beta_h(V: Tensor) -> Tensor:
    return 1.0 / (1.0 + torch.exp(-(V + 35.0) / 10.0))

def _alpha_n(V: Tensor) -> Tensor:
    return 0.1 * _safe_x_over_expm1((V + 55.0) / 10.0)

def _beta_n(V: Tensor) -> Tensor:
    return 0.125 * torch.exp(-(V + 65.0) / 80.0)

def _x_inf(alpha_fn, beta_fn, V: Tensor) -> Tensor:
    a = alpha_fn(V)
    return a / (a + beta_fn(V))


# ===================================================================
# OpticLobeHHNetwork
# ===================================================================

class OpticLobeHHNetwork(nn.Module):
    """
    Single-compartment HH network for Drosophila optic lobe with:

    - Per-neuron-type ion channel parameters (from ``ion_channel_rules.csv``)
    - Per-edge ionotropic synapse with biophysical parameters (from
      ``synapse_rules.csv``), with NT-type defaults for unmapped connections

    Compatible with ``TorchTrainer`` — returns ``(B, T, N)`` voltage traces.

    Parameters
    ----------
    n_neurons : int
    pre_idx, post_idx : LongTensor (E,)
    neuron_g_Na, neuron_g_K, neuron_g_L : Tensor (N,)
        Ion channel conductances in S/cm².
    neuron_E_Na, neuron_E_K, neuron_E_L : Tensor (N,)
        Reversal potentials in mV.
    syn_gS, syn_e_syn, syn_v_th, syn_k_minus, syn_delta : Tensor (E,)
        Per-edge ionotropic synapse parameters.
    input_node_mask : BoolTensor (N,)
        Which neurons receive external stimulus (photoreceptors).
    dt : float
    """

    def __init__(
        self,
        n_neurons: int,
        pre_idx: Tensor,
        post_idx: Tensor,
        neuron_g_Na: Tensor,
        neuron_g_K: Tensor,
        neuron_g_L: Tensor,
        neuron_E_Na: Tensor,
        neuron_E_K: Tensor,
        neuron_E_L: Tensor,
        syn_gS: Tensor,
        syn_e_syn: Tensor,
        syn_v_th: Tensor,
        syn_k_minus: Tensor,
        syn_delta: Tensor,
        syn_count: Optional[Tensor] = None,
        input_node_mask: Optional[Tensor] = None,
        cell_types: Optional[List[str]] = None,
        dt: float = 0.025,
        C_m: float = 1.0,
        learn_ion: bool = True,
        learn_syn: bool = True,
    ):
        super().__init__()
        self.n_neurons = n_neurons
        n_edges = len(pre_idx)
        self.n_edges = n_edges
        self.dt = dt

        self.register_buffer("pre_idx", pre_idx.long())
        self.register_buffer("post_idx", post_idx.long())

        if syn_count is not None:
            self.register_buffer("syn_count", syn_count.float())
        else:
            self.register_buffer("syn_count", torch.ones(n_edges))

        if input_node_mask is not None:
            self.register_buffer("input_mask", input_node_mask.bool())
        else:
            self.register_buffer("input_mask", torch.ones(n_neurons, dtype=torch.bool))

        self.cell_types = cell_types or [f"N{i}" for i in range(n_neurons)]

        # -- Per-neuron ion channel params (trainable in log-space) --
        # Conductances stored in S/cm², internally converted to mS/cm² (* 1e3)
        self.log_g_Na = nn.Parameter(torch.log(neuron_g_Na.float().clamp(min=1e-8)),
                                     requires_grad=learn_ion)
        self.log_g_K = nn.Parameter(torch.log(neuron_g_K.float().clamp(min=1e-8)),
                                    requires_grad=learn_ion)
        self.log_g_L = nn.Parameter(torch.log(neuron_g_L.float().clamp(min=1e-8)),
                                    requires_grad=learn_ion)
        self.E_Na = nn.Parameter(neuron_E_Na.float(), requires_grad=learn_ion)
        self.E_K = nn.Parameter(neuron_E_K.float(), requires_grad=learn_ion)
        self.E_L = nn.Parameter(neuron_E_L.float(), requires_grad=learn_ion)
        self.log_C_m = nn.Parameter(torch.tensor(math.log(C_m)), requires_grad=learn_ion)

        # -- Per-edge synapse params --
        self.log_syn_gS = nn.Parameter(torch.log(syn_gS.float().clamp(min=1e-10)),
                                       requires_grad=learn_syn)
        self.syn_e_syn = nn.Parameter(syn_e_syn.float(), requires_grad=learn_syn)
        self.syn_v_th = nn.Parameter(syn_v_th.float(), requires_grad=learn_syn)
        self.syn_k_minus = nn.Parameter(syn_k_minus.float().clamp(min=1e-6),
                                        requires_grad=learn_syn)
        self.syn_delta = nn.Parameter(syn_delta.float().clamp(min=0.1),
                                      requires_grad=learn_syn)

    # -- properties -------------------------------------------------------
    @property
    def g_Na(self) -> Tensor:
        return torch.exp(self.log_g_Na) * 1e3  # S/cm² → mS/cm²

    @property
    def g_K(self) -> Tensor:
        return torch.exp(self.log_g_K) * 1e3

    @property
    def g_L(self) -> Tensor:
        return torch.exp(self.log_g_L) * 1e3

    @property
    def C_m(self) -> Tensor:
        return torch.exp(self.log_C_m)

    @property
    def syn_gS(self) -> Tensor:
        return torch.exp(self.log_syn_gS) * 1e3  # S/cm² → mS/cm²

    # -- factory ----------------------------------------------------------
    @classmethod
    def from_rules(
        cls,
        ion_rules_path: Union[str, Path],
        syn_rules_path: Union[str, Path],
        data_dir: Optional[Union[str, Path]] = None,
        cell_types: Optional[List[str]] = None,
        min_syn_count: int = 1,
        dt: float = 0.025,
        **kwargs,
    ) -> "OpticLobeHHNetwork":
        """Build from optic lobe connectome + parameter rule CSVs.

        Parameters
        ----------
        ion_rules_path : path to ``ion_channel_rules.csv``
        syn_rules_path : path to ``synapse_rules.csv``
        data_dir : directory containing the malecns feather files
            (default: ``Jaxley_notebook/.../tutorial/``)
        cell_types : optional filter on cell types to include
        min_syn_count : minimum synapse count per edge
        """
        if data_dir is None:
            data_dir = _PKG_ROOT / "Jaxley_notebook" / "jaxley_tutorial-sjcabs" / "tutorial"
        data_dir = Path(data_dir)

        # Load connectome
        meta = pd.read_feather(data_dir / "malecns_09_optic_lobe_hex_08_meta.feather")
        edges_df = pd.read_feather(data_dir / "malecns_09_optic_lobe_hex_08_simple_edgelist.feather")

        # Build ID mappings
        id_to_ct = dict(zip(meta["malecns_09_id"].astype(str), meta["cell_type"]))
        id_to_nt = dict(zip(meta["malecns_09_id"].astype(str),
                            meta["neurotransmitter_predicted"]))
        edges_df["pre"] = edges_df["pre"].astype(str)
        edges_df["post"] = edges_df["post"].astype(str)
        edges_df["pre_type"] = edges_df["pre"].map(id_to_ct)
        edges_df["post_type"] = edges_df["post"].map(id_to_ct)
        edges_df["pre_nt"] = edges_df["pre"].map(id_to_nt)

        # Filter by cell_types if specified
        if cell_types is not None:
            ct_set = set(cell_types)
            mask = meta["cell_type"].isin(ct_set)
            meta = meta[mask].reset_index(drop=True)
            valid_ids = set(meta["malecns_09_id"].astype(str))
            edges_df = edges_df[
                edges_df["pre"].isin(valid_ids) & edges_df["post"].isin(valid_ids)
            ].reset_index(drop=True)

        # Filter by syn count
        edges_df = edges_df[edges_df["count"] >= min_syn_count].reset_index(drop=True)

        # Build node index
        all_ids = sorted(meta["malecns_09_id"].astype(str).unique())
        id_to_idx = {nid: i for i, nid in enumerate(all_ids)}
        n_neurons = len(all_ids)

        # Build edge indices
        pre_ids = edges_df["pre"].values
        post_ids = edges_df["post"].values
        valid_edge_mask = np.array([
            pid in id_to_idx and qid in id_to_idx
            for pid, qid in zip(pre_ids, post_ids)
        ])
        edges_df = edges_df[valid_edge_mask].reset_index(drop=True)

        pre_idx = torch.tensor([id_to_idx[str(p)] for p in edges_df["pre"]], dtype=torch.long)
        post_idx = torch.tensor([id_to_idx[str(p)] for p in edges_df["post"]], dtype=torch.long)
        syn_count_t = torch.tensor(edges_df["count"].values, dtype=torch.float32)

        # Load parameter rules
        ion_rules = _load_ion_rules(ion_rules_path)
        syn_rules = _load_syn_rules(syn_rules_path)

        # Build per-neuron ion channel params
        id_ct_list = [id_to_ct.get(nid, "unknown") for nid in all_ids]
        g_Na_list, g_K_list, g_L_list = [], [], []
        E_Na_list, E_K_list, E_L_list = [], [], []
        for ct in id_ct_list:
            p = _get_ion_params(ct, ion_rules)
            g_Na_list.append(p["gNa"])
            g_K_list.append(p["gK"])
            g_L_list.append(p["gLeak"])
            E_Na_list.append(p["eNa"])
            E_K_list.append(p["eK"])
            E_L_list.append(p["eLeak"])

        # Build per-edge synapse params
        gS_list, e_syn_list, v_th_list, k_minus_list, delta_list = [], [], [], [], []
        for _, row in edges_df.iterrows():
            sp = _get_syn_params(
                row["pre_type"], row["post_type"],
                row.get("pre_nt", "acetylcholine"), syn_rules,
            )
            gS_list.append(sp["gS"])
            e_syn_list.append(sp["e_syn"])
            v_th_list.append(sp["v_th"])
            k_minus_list.append(sp["k_minus"])
            delta_list.append(sp["delta"])

        # Photoreceptor mask
        photo_types = {"R1-R6", "R1-6", "R7", "R7_unclear", "R7p", "R7y",
                       "R8", "R8_unclear", "R8p", "R8y"}
        input_mask = torch.tensor(
            [id_ct_list[i] in photo_types for i in range(n_neurons)],
            dtype=torch.bool,
        )

        logger.info("OpticLobeHHNetwork: %d neurons, %d edges, %d photoreceptors",
                     n_neurons, len(pre_idx), input_mask.sum().item())

        return cls(
            n_neurons=n_neurons,
            pre_idx=pre_idx,
            post_idx=post_idx,
            neuron_g_Na=torch.tensor(g_Na_list),
            neuron_g_K=torch.tensor(g_K_list),
            neuron_g_L=torch.tensor(g_L_list),
            neuron_E_Na=torch.tensor(E_Na_list),
            neuron_E_K=torch.tensor(E_K_list),
            neuron_E_L=torch.tensor(E_L_list),
            syn_gS=torch.tensor(gS_list),
            syn_e_syn=torch.tensor(e_syn_list),
            syn_v_th=torch.tensor(v_th_list),
            syn_k_minus=torch.tensor(k_minus_list),
            syn_delta=torch.tensor(delta_list),
            syn_count=syn_count_t,
            input_node_mask=input_mask,
            cell_types=id_ct_list,
            dt=dt,
            **kwargs,
        )

    @classmethod
    def from_fafb_pathway(
        cls,
        neurons_path: Union[str, Path],
        connections_path: Union[str, Path],
        ion_rules_path: Union[str, Path],
        syn_rules_path: Union[str, Path],
        min_syn_count: int = 1,
        dt: float = 0.025,
        **kwargs,
    ) -> "OpticLobeHHNetwork":
        """Build from pre-filtered FAFB T4/T5 pathway data.

        Parameters
        ----------
        neurons_path : path to ``fafb_t4_pathway_neurons.csv``
            Columns: root_id, type, sort_key
        connections_path : path to ``fafb_t4_pathway_connections.csv``
            Columns: pre_root_id, post_root_id, neuropil, syn_count, nt_type,
            pre_type, post_type
        ion_rules_path : path to ion channel rules CSV
        syn_rules_path : path to synapse rules CSV
        min_syn_count : minimum synapse count per edge
        """
        neurons = pd.read_csv(neurons_path)
        connections = pd.read_csv(connections_path)

        connections = connections[connections["syn_count"] >= min_syn_count].reset_index(drop=True)

        id_to_type = dict(zip(neurons["root_id"], neurons["type"]))
        all_ids = sorted(neurons["root_id"].unique())
        id_to_idx = {int(rid): i for i, rid in enumerate(all_ids)}
        n_neurons = len(all_ids)

        valid = (
            connections["pre_root_id"].isin(id_to_idx)
            & connections["post_root_id"].isin(id_to_idx)
        )
        connections = connections[valid].reset_index(drop=True)

        pre_idx = torch.tensor(
            [id_to_idx[int(r)] for r in connections["pre_root_id"]], dtype=torch.long
        )
        post_idx = torch.tensor(
            [id_to_idx[int(r)] for r in connections["post_root_id"]], dtype=torch.long
        )
        syn_count_t = torch.tensor(connections["syn_count"].values, dtype=torch.float32)

        ion_rules = _load_ion_rules(ion_rules_path)
        syn_rules = _load_syn_rules(syn_rules_path)

        id_ct_list = [id_to_type.get(int(rid), "unknown") for rid in all_ids]
        g_Na_list, g_K_list, g_L_list = [], [], []
        E_Na_list, E_K_list, E_L_list = [], [], []
        for ct in id_ct_list:
            p = _get_ion_params(ct, ion_rules)
            g_Na_list.append(p["gNa"])
            g_K_list.append(p["gK"])
            g_L_list.append(p["gLeak"])
            E_Na_list.append(p["eNa"])
            E_K_list.append(p["eK"])
            E_L_list.append(p["eLeak"])

        gS_list, e_syn_list, v_th_list, k_minus_list, delta_list = [], [], [], [], []
        for _, row in connections.iterrows():
            pre_type = row.get("pre_type", id_to_type.get(int(row["pre_root_id"]), "unknown"))
            post_type = row.get("post_type", id_to_type.get(int(row["post_root_id"]), "unknown"))
            nt = row.get("nt_type", "ACH")
            sp = _get_syn_params(pre_type, post_type, nt, syn_rules)
            gS_list.append(sp["gS"])
            e_syn_list.append(sp["e_syn"])
            v_th_list.append(sp["v_th"])
            k_minus_list.append(sp["k_minus"])
            delta_list.append(sp["delta"])

        photo_types = {"R1-R6", "R1-6", "R7", "R7_unclear", "R7p", "R7y",
                       "R8", "R8_unclear", "R8p", "R8y"}
        input_mask = torch.tensor(
            [id_ct_list[i] in photo_types for i in range(n_neurons)],
            dtype=torch.bool,
        )

        logger.info(
            "OpticLobeHHNetwork (FAFB): %d neurons, %d edges, %d photoreceptors",
            n_neurons, len(pre_idx), input_mask.sum().item(),
        )

        return cls(
            n_neurons=n_neurons,
            pre_idx=pre_idx,
            post_idx=post_idx,
            neuron_g_Na=torch.tensor(g_Na_list),
            neuron_g_K=torch.tensor(g_K_list),
            neuron_g_L=torch.tensor(g_L_list),
            neuron_E_Na=torch.tensor(E_Na_list),
            neuron_E_K=torch.tensor(E_K_list),
            neuron_E_L=torch.tensor(E_L_list),
            syn_gS=torch.tensor(gS_list),
            syn_e_syn=torch.tensor(e_syn_list),
            syn_v_th=torch.tensor(v_th_list),
            syn_k_minus=torch.tensor(k_minus_list),
            syn_delta=torch.tensor(delta_list),
            syn_count=syn_count_t,
            input_node_mask=input_mask,
            cell_types=id_ct_list,
            dt=dt,
            **kwargs,
        )

    # -- state initialisation ---------------------------------------------
    def init_state(self, batch_size: int = 1,
                   device: torch.device = torch.device("cpu")) -> Dict[str, Tensor]:
        N = self.n_neurons
        V0 = self.E_L.unsqueeze(0).expand(batch_size, -1).clone().to(device)
        return {
            "V": V0,
            "m": _x_inf(_alpha_m, _beta_m, V0),
            "h": _x_inf(_alpha_h, _beta_h, V0),
            "n": _x_inf(_alpha_n, _beta_n, V0),
            "s": torch.full((batch_size, self.n_edges), 0.0, device=device),
        }

    # -- single step ------------------------------------------------------
    def _step(
        self,
        state: Dict[str, Tensor],
        I_ext: Tensor,
        dt: float,
    ) -> Dict[str, Tensor]:
        V, m, h, n, s = state["V"], state["m"], state["h"], state["n"], state["s"]
        B, N = V.shape

        # 1. Ionic currents (per-neuron params broadcast over batch)
        I_Na = self.g_Na * m**3 * h * (V - self.E_Na)
        I_K = self.g_K * n**4 * (V - self.E_K)
        I_L = self.g_L * (V - self.E_L)

        # 2. Synaptic currents
        pre_v = V[:, self.pre_idx]    # (B, E)
        post_v = V[:, self.post_idx]  # (B, E)

        # Update synapse state s (exponential Euler)
        s_inf = 1.0 / (1.0 + torch.exp((self.syn_v_th - pre_v) / self.syn_delta))
        tau_s = (1.0 - s_inf) / (self.syn_k_minus + 1e-8)
        slope = -1.0 / (tau_s + 1e-8)
        exp_term = torch.exp(slope * dt)
        s_new = s * exp_term + s_inf * (1.0 - exp_term)

        # Synaptic current: I_syn = gS * w * s * (V_post - e_syn)
        # syn_count normalized so mean=1 to preserve relative strength without
        # blowing up conductances (gS from rules is already per-edge calibrated)
        w = self.syn_count / (self.syn_count.mean() + 1e-8)
        I_syn_edge = self.syn_gS * w * s_new * (post_v - self.syn_e_syn)

        # Scatter synaptic current to post-synaptic neurons (subtract from dV)
        I_syn_total = torch.zeros(B, N, device=V.device)
        idx_expand = self.post_idx.unsqueeze(0).expand(B, -1)
        I_syn_total.scatter_add_(1, idx_expand, I_syn_edge)

        # 3. Voltage update (Euler)
        dV = (1.0 / self.C_m) * (-I_Na - I_K - I_L - I_syn_total + I_ext)
        V_new = V + dt * dV

        # 4. Gating variables (Euler, clamped)
        am, bm = _alpha_m(V), _beta_m(V)
        ah, bh = _alpha_h(V), _beta_h(V)
        an, bn = _alpha_n(V), _beta_n(V)

        return {
            "V": V_new,
            "m": torch.clamp(m + dt * (am * (1 - m) - bm * m), 0, 1),
            "h": torch.clamp(h + dt * (ah * (1 - h) - bh * h), 0, 1),
            "n": torch.clamp(n + dt * (an * (1 - n) - bn * n), 0, 1),
            "s": s_new,
        }

    # -- forward / simulate -----------------------------------------------
    def forward(
        self,
        x: Tensor,
        dt: Optional[float] = None,
        state: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        """
        Integrate network dynamics.

        Parameters
        ----------
        x : (B, T, N_input) or (B, T, N_neurons)
            External current. If ``x.shape[2]`` matches the number of
            photoreceptor nodes, it is mapped to those nodes; otherwise it
            must match ``n_neurons``.
        dt : float, optional
        state : dict, optional

        Returns
        -------
        V_trace : (B, T, N_neurons)  membrane voltage in mV
        """
        dt = dt or self.dt
        B, T = x.shape[0], x.shape[1]
        N = self.n_neurons

        # Map input to full neuron array
        n_input = x.shape[2]
        if n_input == N:
            x_full = x
        else:
            n_photo = self.input_mask.sum().item()
            assert n_input == n_photo, (
                f"Input dim {n_input} != n_neurons {N} and != n_photoreceptors {n_photo}"
            )
            x_full = torch.zeros(B, T, N, device=x.device, dtype=x.dtype)
            photo_indices = torch.where(self.input_mask)[0]
            x_full[:, :, photo_indices] = x

        if state is None:
            state = self.init_state(B, device=x.device)

        traces: List[Tensor] = []
        for t in range(T):
            state = self._step(state, x_full[:, t], dt)
            traces.append(state["V"])

        return torch.stack(traces, dim=1)  # (B, T, N)

    def simulate(self, x: Tensor, **kw) -> Tensor:
        return self.forward(x, **kw)

    # -- utilities --------------------------------------------------------
    def get_photoreceptor_indices(self) -> Tensor:
        return torch.where(self.input_mask)[0]

    def get_indices_by_type(self, cell_type: str, exact: bool = True) -> List[int]:
        if exact:
            return [i for i, ct in enumerate(self.cell_types) if ct == cell_type]
        return [i for i, ct in enumerate(self.cell_types)
                if ct == cell_type or _resolve_cell_type(ct) == cell_type]

    def type_counts(self) -> Dict[str, int]:
        from collections import Counter
        return dict(Counter(self.cell_types))

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        n_photo = self.input_mask.sum().item()
        n_types = len(set(self.cell_types))
        return (
            f"n_neurons={self.n_neurons}, n_edges={self.n_edges}, "
            f"n_photoreceptors={n_photo}, n_cell_types={n_types}, "
            f"trainable={self.n_parameters()}"
        )

    # -- retinotopic coordinate inference ---------------------------------
    @staticmethod
    def compute_retinotopic_coords(
        neurons_path: Union[str, Path],
        connections_path: Union[str, Path],
        min_syn_count: int = 2,
    ) -> np.ndarray:
        """Infer 2D retinotopic coordinates for photoreceptors from connectivity.

        Each L1 neuron defines one columnar unit.  R-type neurons are assigned
        to the column of their strongest L1 target.  Columns are arranged in
        2D via force-directed layout on the column adjacency graph (two columns
        are adjacent if they share downstream medulla/lobula targets).

        Returns
        -------
        coords : ndarray (n_photoreceptors, 2)
            Coordinates ordered as ``get_photoreceptor_indices()`` returns.
        """
        from collections import defaultdict

        neurons = pd.read_csv(neurons_path)
        connections = pd.read_csv(connections_path)
        if min_syn_count > 1:
            connections = connections[connections["syn_count"] >= min_syn_count]
        id_to_type = dict(zip(neurons["root_id"], neurons["type"]))

        all_ids = sorted(neurons["root_id"].unique())
        id_to_idx = {int(rid): i for i, rid in enumerate(all_ids)}

        photo_types = {"R1-R6", "R1-6", "R7", "R7_unclear", "R7p", "R7y",
                       "R8", "R8_unclear", "R8p", "R8y"}
        photo_model_indices = [
            i for i, rid in enumerate(all_ids)
            if id_to_type.get(int(rid), "?") in photo_types
        ]
        photo_root_ids = [int(all_ids[i]) for i in photo_model_indices]
        n_photo = len(photo_root_ids)
        photo_rid_to_pos = {rid: pos for pos, rid in enumerate(photo_root_ids)}

        # --- Step 1: identify columns via R → L1 edges ---
        lamina_types = {"L1", "L3"}
        r_lam_edges = connections[
            connections["pre_type"].str.startswith("R", na=False)
            & connections["post_type"].isin(lamina_types)
        ]
        col_to_r: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        r_to_col_weight: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        for _, row in r_lam_edges.iterrows():
            rid = int(row["pre_root_id"])
            lid = int(row["post_root_id"])
            w = float(row["syn_count"])
            col_to_r[lid][rid] += w
            r_to_col_weight[rid][lid] += w

        col_ids = sorted(col_to_r.keys())
        n_col = len(col_ids)
        col_id_to_ci = {cid: ci for ci, cid in enumerate(col_ids)}

        # Assign each R to primary column (strongest connection)
        r_to_primary_col = {}
        for rid in photo_root_ids:
            if rid in r_to_col_weight:
                best = max(r_to_col_weight[rid].items(), key=lambda x: x[1])
                r_to_primary_col[rid] = col_id_to_ci[best[0]]

        # --- Step 2: column adjacency via shared downstream targets ---
        medulla_types = {"Mi1", "Mi4", "Mi9", "Tm3", "L5"}
        neuron_to_downstream = defaultdict(set)
        for _, row in connections.iterrows():
            pre_t = str(row.get("pre_type", ""))
            post_t = str(row.get("post_type", ""))
            if pre_t in lamina_types and post_t in medulla_types:
                neuron_to_downstream[int(row["pre_root_id"])].add(int(row["post_root_id"]))

        adj = np.zeros((n_col, n_col))
        for ci, cid_a in enumerate(col_ids):
            ds_a = neuron_to_downstream.get(cid_a, set())
            for cj in range(ci + 1, n_col):
                cid_b = col_ids[cj]
                ds_b = neuron_to_downstream.get(cid_b, set())
                shared = len(ds_a & ds_b)
                if shared > 0:
                    adj[ci, cj] = shared
                    adj[cj, ci] = shared

        # --- Step 3: force-directed layout for columns ---
        rng = np.random.RandomState(42)
        pos = rng.randn(n_col, 2).astype(np.float64) * 2.0
        lr = 0.05
        for _ in range(200):
            forces = np.zeros_like(pos)
            for ci in range(n_col):
                for cj in range(ci + 1, n_col):
                    diff = pos[ci] - pos[cj]
                    dist = np.linalg.norm(diff) + 1e-6
                    repulsion = diff / (dist ** 2) * 1.0
                    forces[ci] += repulsion
                    forces[cj] -= repulsion
                    if adj[ci, cj] > 0:
                        attraction = -diff * adj[ci, cj] * 0.1
                        forces[ci] += attraction
                        forces[cj] -= attraction
            pos += lr * forces
            pos -= pos.mean(axis=0)

        # --- Step 4: assign coordinates to photoreceptors ---
        coords = np.zeros((n_photo, 2), dtype=np.float32)
        unassigned = []
        for pos_idx, rid in enumerate(photo_root_ids):
            ci = r_to_primary_col.get(rid)
            if ci is not None:
                coords[pos_idx] = pos[ci].astype(np.float32)
            else:
                unassigned.append(pos_idx)

        if unassigned:
            centroid = coords[~np.isin(np.arange(n_photo), unassigned)].mean(axis=0)
            jitter = rng.randn(len(unassigned), 2).astype(np.float32) * 0.3
            for i, idx in enumerate(unassigned):
                coords[idx] = centroid + jitter[i]

        logger.info(
            "Retinotopic coords: %d photoreceptors, %d columns, %d unassigned",
            n_photo, n_col, len(unassigned),
        )
        return coords
