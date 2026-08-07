"""
FAFB Multi-Compartment HH Network
==================================
Multi-compartment Hodgkin-Huxley network for large-scale FAFB connectome data.

Unlike ``MCNetwork`` (shared HH params), this class supports:
- Per-neuron-type ion channel parameters
- Per-edge ionotropic synapse parameters (from synapse rules)
- Loading from pre-processed FAFB data files

The base path supports a shared simplified morphology, and the optic-lobe
factory can also upgrade the network to a per-neuron morphology solver by
concatenating all neuron morphologies into one sparse axial graph.

Units (Jaxley-compatible)
-------------------------
  V : mV,  t : ms,  g : mS/cm²,  C_m : μF/cm²,
  I : μA/cm²,  R_a : Ω·cm,  area : μm²,  length : μm

Ion channel rules CSV stores conductances in S/cm²; they are
converted to mS/cm² (× 1e3) on loading.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor

from .fafb_param_config import DEFAULT_ION, load_nt_ion_priors, load_type_fallback_rules
from .morphology import MorphologyGraph
from .mc_hh import _solve_implicit, _alpha_m, _beta_m, _alpha_h, _beta_h, _alpha_n, _beta_n, _x_inf, MultiCompartmentHH

logger = logging.getLogger(__name__)

__all__ = ["FAFBMCNetwork"]

_PKG_DATA = Path(__file__).parents[1] / "data"

_NT_ALIAS = {
    "ach": "acetylcholine", "acetylcholine": "acetylcholine",
    "glut": "glutamate", "glutamate": "glutamate",
    "gaba": "gaba", "histamine": "histamine",
}

_DEFAULT_SYN_BY_NT = {
    "acetylcholine": {"gS": 3e-4, "e_syn": 0.0, "v_th": -45.0, "k_minus": 0.10, "delta": 3.5},
    "gaba": {"gS": 5e-4, "e_syn": -72.0, "v_th": -55.0, "k_minus": 0.10, "delta": 3.2},
    "glutamate": {"gS": 5e-4, "e_syn": -70.0, "v_th": -45.0, "k_minus": 0.08, "delta": 4.0},
    "histamine": {"gS": 4e-4, "e_syn": -72.0, "v_th": -60.0, "k_minus": 0.10, "delta": 3.0},
}

_CELL_TYPE_ALIAS, _FALLBACK_TYPE = load_type_fallback_rules()


def _resolve(ct: str) -> str:
    return _CELL_TYPE_ALIAS.get(ct, ct)


def _fallback_type(ct: str) -> Optional[str]:
    c = _resolve(ct)
    return _FALLBACK_TYPE.get(c, _FALLBACK_TYPE.get(ct))


def _candidate_types(ct: str) -> List[str]:
    out = []
    c = _resolve(ct)
    if c:
        out.append(c)
    fb = _fallback_type(ct)
    if fb and fb not in out:
        out.append(fb)
    return out


def _normalize_nt(nt: object) -> str:
    if not isinstance(nt, str):
        return "unknown"
    s = nt.lower().strip()
    return _NT_ALIAS.get(s, s if s else "unknown")


def _get_ion(ct: str, rules: Dict, nt_default: Optional[Dict] = None) -> Dict:
    c = _resolve(ct)
    if c in rules:
        return rules[c]
    fb = _fallback_type(ct)
    if fb and fb in rules:
        return rules[fb]
    if nt_default is not None:
        return nt_default
    return DEFAULT_ION


def _get_syn(
    pre_t: str,
    post_t: str,
    nt: str,
    rules: Dict,
    pre_type_defaults: Optional[Dict[str, Dict]] = None,
) -> Tuple[Dict, str, Optional[str]]:
    pre_candidates = _candidate_types(pre_t)
    post_candidates = _candidate_types(post_t)

    for pre_c in pre_candidates:
        for post_c in post_candidates:
            key = (pre_c, post_c)
            if key in rules:
                source = f"type:{pre_c}->{post_c}"
                fallback_target = None
                if pre_c != _resolve(pre_t) or post_c != _resolve(post_t):
                    fallback_target = f"{pre_c}->{post_c}"
                return dict(rules[key]), source, fallback_target

    if pre_type_defaults is not None:
        for pre_c in pre_candidates:
            if pre_c in pre_type_defaults:
                source = f"pre_type:{pre_c}"
                fallback_target = pre_c if pre_c != _resolve(pre_t) else None
                return dict(pre_type_defaults[pre_c]), source, fallback_target

    nt_n = _NT_ALIAS.get(nt.lower().strip(), nt.lower().strip()) if isinstance(nt, str) else "acetylcholine"
    return dict(_DEFAULT_SYN_BY_NT.get(nt_n, _DEFAULT_SYN_BY_NT["acetylcholine"])), f"nt:{nt_n}", None


def _load_ion_rules(path: Union[str, Path]) -> Dict:
    df = pd.read_csv(path)
    rules = {}
    for _, r in df.iterrows():
        rules[r["neuron_type"]] = {
            "gNa": float(r["gNa"]), "gK": float(r["gK"]),
            "gLeak": float(r.get("gLeak", r.get("gL", 0.001))),
            "eNa": float(r["eNa"]), "eK": float(r["eK"]),
            "eLeak": float(r["eLeak"]),
        }
    return rules


def _load_nt_ion_rules(path: Union[str, Path]) -> Dict:
    rules = load_nt_ion_priors(path)
    return {_normalize_nt(k): dict(v) for k, v in rules.items()}


def _load_root_ion_overrides(path: Union[str, Path]) -> Dict[int, Dict]:
    df = pd.read_csv(path)
    required = {"root_id"}
    if not required.issubset(df.columns):
        raise ValueError(f"Override CSV must contain columns {sorted(required)}")
    overrides: Dict[int, Dict] = {}
    for r in df.itertuples(index=False):
        rid = int(getattr(r, "root_id"))
        row = {}
        for key in ["gNa", "gK", "gLeak", "eNa", "eK", "eLeak"]:
            val = getattr(r, key, None)
            if key in df.columns and pd.notna(val):
                row[key] = float(val)
        overrides[rid] = row
    return overrides


def _infer_root_nt_types(connections: pd.DataFrame, all_ids: np.ndarray) -> Dict[int, str]:
    if "nt_type" not in connections.columns or len(connections) == 0:
        return {int(rid): "unknown" for rid in all_ids}

    tmp = connections[["pre_root_id", "nt_type", "syn_count"]].copy()
    tmp["nt_type"] = tmp["nt_type"].map(_normalize_nt)
    tmp["syn_count"] = tmp["syn_count"].astype(float)
    grouped = (
        tmp.groupby(["pre_root_id", "nt_type"], as_index=False)["syn_count"]
        .sum()
        .sort_values(["pre_root_id", "syn_count"], ascending=[True, False])
    )
    dominant = grouped.drop_duplicates("pre_root_id")
    out = {int(rid): "unknown" for rid in all_ids}
    for r in dominant.itertuples(index=False):
        out[int(getattr(r, "pre_root_id"))] = str(getattr(r, "nt_type"))
    return out


def _load_syn_rules(path: Union[str, Path]) -> Tuple[Dict, Dict[str, Dict]]:
    df = pd.read_csv(path)
    rules = {}
    numeric_cols = ["gS", "e_syn", "v_th", "k_minus", "delta"]
    for _, r in df.iterrows():
        key = (_resolve(str(r["pre_type"]).strip()), _resolve(str(r["post_type"]).strip()))
        rules[key] = {
            "gS": float(r["gS"]), "e_syn": float(r["e_syn"]),
            "v_th": float(r["v_th"]), "k_minus": float(r["k_minus"]),
            "delta": float(r["delta"]),
        }

    pre_type_defaults = {}
    if len(df) > 0:
        tmp = df.copy()
        tmp["pre_type"] = tmp["pre_type"].map(lambda x: _resolve(str(x).strip()))
        grouped = tmp.groupby("pre_type", as_index=False)[numeric_cols].mean()
        for r in grouped.itertuples(index=False):
            pre_type_defaults[str(getattr(r, "pre_type"))] = {
                "gS": float(getattr(r, "gS")),
                "e_syn": float(getattr(r, "e_syn")),
                "v_th": float(getattr(r, "v_th")),
                "k_minus": float(getattr(r, "k_minus")),
                "delta": float(getattr(r, "delta")),
            }
    return rules, pre_type_defaults


class FAFBMCNetwork(nn.Module):
    """Multi-compartment HH network with per-neuron-type parameters.

    Parameters
    ----------
    morph : MorphologyGraph
        Shared morphology for all neurons.
    n_neurons : int
    pre_idx, post_idx : LongTensor (E,)
    neuron_g_Na ... neuron_E_L : Tensor (N,)
        Per-neuron HH params in **S/cm²** (conductances) and mV (potentials).
    syn_gS ... syn_delta : Tensor (E,)
        Per-edge ionotropic synapse params (gS in S/cm²).
    syn_count : FloatTensor (E,), optional
    input_node_mask : BoolTensor (N,)
    cell_types : list[str]
    post_comp_idx : int
        Which compartment receives synaptic input (default 1 = dendrite).
    """

    def __init__(
        self,
        morph: MorphologyGraph,
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
        R_a: float = 150.0,
        post_comp_idx: int = -1,
        learn_ion: bool = True,
        learn_syn: bool = True,
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.n_comp = morph.n_comp
        self.total_comp = morph.n_comp
        n_edges = len(pre_idx)
        self.n_edges = n_edges
        self.dt = dt
        self.ragged_mode = False

        # -- Morphology (shared) --
        self.register_buffer("areas", torch.tensor(morph.areas, dtype=torch.float64))
        G_sparse = MultiCompartmentHH._build_geom_matrix(morph)
        self.register_buffer("G_row", G_sparse[0])
        self.register_buffer("G_col", G_sparse[1])
        self.register_buffer("G_val", G_sparse[2])

        # -- Connectivity --
        self.register_buffer("pre_idx", pre_idx.long())
        self.register_buffer("post_idx", post_idx.long())
        syn_comp = min(morph.n_comp - 1, max(0, post_comp_idx if post_comp_idx >= 0 else morph.n_comp - 1))
        self.register_buffer("pre_comp", torch.zeros(n_edges, dtype=torch.long))
        self.register_buffer("post_comp", torch.full((n_edges,), syn_comp, dtype=torch.long))

        if syn_count is not None:
            self.register_buffer("syn_count", syn_count.float())
        else:
            self.register_buffer("syn_count", torch.ones(n_edges))

        if input_node_mask is not None:
            self.register_buffer("input_mask", input_node_mask.bool())
        else:
            self.register_buffer("input_mask", torch.ones(n_neurons, dtype=torch.bool))

        self.cell_types = cell_types or [f"N{i}" for i in range(n_neurons)]

        # -- Per-neuron HH (S/cm² → mS/cm² at property access) --
        self.log_g_Na = nn.Parameter(torch.log(neuron_g_Na.float().clamp(min=1e-8)), requires_grad=learn_ion)
        self.log_g_K = nn.Parameter(torch.log(neuron_g_K.float().clamp(min=1e-8)), requires_grad=learn_ion)
        self.log_g_L = nn.Parameter(torch.log(neuron_g_L.float().clamp(min=1e-8)), requires_grad=learn_ion)
        self.E_Na = nn.Parameter(neuron_E_Na.float(), requires_grad=learn_ion)
        self.E_K = nn.Parameter(neuron_E_K.float(), requires_grad=learn_ion)
        self.E_L = nn.Parameter(neuron_E_L.float(), requires_grad=learn_ion)
        self.log_C_m = nn.Parameter(torch.tensor(math.log(C_m)), requires_grad=learn_ion)
        self.log_R_a = nn.Parameter(torch.tensor(math.log(R_a)), requires_grad=learn_ion)

        # -- Per-edge synapse params --
        self.log_syn_gS = nn.Parameter(torch.log(syn_gS.float().clamp(min=1e-10)), requires_grad=learn_syn)
        self.syn_e_syn = nn.Parameter(syn_e_syn.float(), requires_grad=learn_syn)
        self.syn_v_th = nn.Parameter(syn_v_th.float(), requires_grad=learn_syn)
        self.syn_k_minus = nn.Parameter(syn_k_minus.float().clamp(min=1e-6), requires_grad=learn_syn)
        self.syn_delta = nn.Parameter(syn_delta.float().clamp(min=0.1), requires_grad=learn_syn)

    # -- properties --
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
    def R_a(self) -> Tensor:
        return torch.exp(self.log_R_a)

    @property
    def syn_gS(self) -> Tensor:
        return torch.exp(self.log_syn_gS) * 1e3

    # -- factory --
    @classmethod
    def from_preprocessed(
        cls,
        data_dir: Union[str, Path],
        swc_path: Union[str, Path],
        ion_rules_path: Union[str, Path],
        syn_rules_path: Union[str, Path],
        nt_ion_rules_path: Optional[Union[str, Path]] = None,
        neuron_ion_overrides_path: Optional[Union[str, Path]] = None,
        ncomp: int = 1,
        min_syn_count: int = 2,
        dt: float = 0.025,
        **kwargs,
    ) -> "FAFBMCNetwork":
        """Build from preprocessed FAFB data files (vectorized, fast).

        Expects ``data_dir`` to contain:
          - ``neurons.csv`` : root_id, type, subsystem, category, side
          - ``connections.csv`` : pre_root_id, post_root_id, neuropil,
            syn_count, nt_type, pre_type, post_type
        """
        import time
        t0 = time.time()
        data_dir = Path(data_dir)
        neurons = pd.read_csv(data_dir / "neurons.csv")
        connections = pd.read_csv(data_dir / "connections.csv")
        logger.info("Loaded CSVs in %.1fs", time.time() - t0)

        connections = connections[connections["syn_count"] >= min_syn_count].reset_index(drop=True)

        all_ids = np.sort(neurons["root_id"].unique())
        id_map = pd.Series(np.arange(len(all_ids)), index=all_ids)
        n_neurons = len(all_ids)

        id_to_type_s = pd.Series(neurons["type"].values, index=neurons["root_id"].values)

        pre_mapped = id_map.reindex(connections["pre_root_id"].values)
        post_mapped = id_map.reindex(connections["post_root_id"].values)
        valid = pre_mapped.notna().values & post_mapped.notna().values
        connections = connections[valid].reset_index(drop=True)
        pre_arr = pre_mapped[valid].values.astype(np.int64)
        post_arr = post_mapped[valid].values.astype(np.int64)
        syn_count_arr = connections["syn_count"].values.astype(np.float32)

        pre_t = torch.from_numpy(pre_arr)
        post_t = torch.from_numpy(post_arr)
        syn_count_t = torch.from_numpy(syn_count_arr)

        logger.info("Mapped %d edges in %.1fs", len(pre_t), time.time() - t0)

        # --- Ion defaults hierarchy ---
        # 1) neurotransmitter defaults
        # 2) neuron type defaults
        # 3) root_id overrides
        ion_rules = _load_ion_rules(ion_rules_path)
        nt_ion_rules = load_nt_ion_priors()
        if nt_ion_rules_path is not None:
            nt_ion_rules.update(_load_nt_ion_rules(nt_ion_rules_path))
        root_ion_overrides = (
            _load_root_ion_overrides(neuron_ion_overrides_path)
            if neuron_ion_overrides_path is not None else {}
        )

        id_ct_arr = np.array([id_to_type_s.get(rid, "unknown") for rid in all_ids])
        id_ct_list = id_ct_arr.tolist()
        root_nt = _infer_root_nt_types(connections, all_ids)
        nt_arr = np.array([root_nt.get(int(rid), "unknown") for rid in all_ids])

        resolved = np.array([_resolve(ct) for ct in id_ct_list])
        unique_resolved = np.unique(resolved)
        ion_lut: Dict[Tuple[str, str], Dict] = {}
        for ct in unique_resolved:
            for nt in np.unique(nt_arr):
                ion_lut[(ct, nt)] = _get_ion(ct, ion_rules, nt_default=nt_ion_rules.get(nt, DEFAULT_ION))

        param_rows = []
        param_source = []
        param_resolved_type = []
        param_fallback_target = []
        for rid, raw_ct, ct, nt in zip(all_ids, id_ct_list, resolved, nt_arr):
            params = dict(ion_lut[(ct, nt)])
            source = f"type:{ct}"
            fallback_target = None
            override = root_ion_overrides.get(int(rid))
            if override:
                params.update(override)
                source = f"root_id:{int(rid)}"
            elif ct not in ion_rules:
                fb = _FALLBACK_TYPE.get(ct, _FALLBACK_TYPE.get(str(raw_ct)))
                if fb and fb in ion_rules:
                    source = f"type_fallback:{fb}"
                    fallback_target = fb
                else:
                    source = f"nt:{nt}"
            param_rows.append(params)
            param_source.append(source)
            param_resolved_type.append(ct)
            param_fallback_target.append(fallback_target)

        g_Na_a = np.array([p["gNa"] for p in param_rows])
        g_K_a  = np.array([p["gK"] for p in param_rows])
        g_L_a  = np.array([p["gLeak"] for p in param_rows])
        E_Na_a = np.array([p["eNa"] for p in param_rows])
        E_K_a  = np.array([p["eK"] for p in param_rows])
        E_L_a  = np.array([p["eLeak"] for p in param_rows])

        # --- Vectorized synapse lookup ---
        syn_rules, syn_pre_type_defaults = _load_syn_rules(syn_rules_path)
        pre_types = connections["pre_type"].values if "pre_type" in connections.columns else \
            np.array([id_to_type_s.get(rid, "unknown") for rid in connections["pre_root_id"].values])
        post_types = connections["post_type"].values if "post_type" in connections.columns else \
            np.array([id_to_type_s.get(rid, "unknown") for rid in connections["post_root_id"].values])
        nt_types = connections["nt_type"].values if "nt_type" in connections.columns else \
            np.full(len(connections), "ACH")

        syn_param_rows = []
        syn_param_source = []
        syn_fallback_target = []
        for pre_t_raw, post_t_raw, nt_raw in zip(pre_types, post_types, nt_types):
            params, source, fallback_target = _get_syn(
                str(pre_t_raw),
                str(post_t_raw),
                str(nt_raw),
                syn_rules,
                pre_type_defaults=syn_pre_type_defaults,
            )
            syn_param_rows.append(params)
            syn_param_source.append(source)
            syn_fallback_target.append(fallback_target)

        gS_a    = np.array([p["gS"] for p in syn_param_rows])
        e_syn_a = np.array([p["e_syn"] for p in syn_param_rows])
        v_th_a  = np.array([p["v_th"] for p in syn_param_rows])
        k_m_a   = np.array([p["k_minus"] for p in syn_param_rows])
        delta_a = np.array([p["delta"] for p in syn_param_rows])

        logger.info("Synapse lookup done in %.1fs", time.time() - t0)

        photo_types = {"R1-R6", "R1-6", "R7", "R8"}
        input_mask = torch.tensor([ct in photo_types for ct in id_ct_list], dtype=torch.bool)

        morph = MorphologyGraph.from_swc(str(swc_path), ncomp=ncomp)

        logger.info("FAFBMCNetwork: %d neurons (%d comp each), %d edges, %d photoreceptors [%.1fs]",
                     n_neurons, morph.n_comp, len(pre_t), input_mask.sum().item(), time.time() - t0)

        net = cls(
            morph=morph, n_neurons=n_neurons,
            pre_idx=pre_t, post_idx=post_t,
            neuron_g_Na=torch.from_numpy(g_Na_a.astype(np.float32)),
            neuron_g_K=torch.from_numpy(g_K_a.astype(np.float32)),
            neuron_g_L=torch.from_numpy(g_L_a.astype(np.float32)),
            neuron_E_Na=torch.from_numpy(E_Na_a.astype(np.float32)),
            neuron_E_K=torch.from_numpy(E_K_a.astype(np.float32)),
            neuron_E_L=torch.from_numpy(E_L_a.astype(np.float32)),
            syn_gS=torch.from_numpy(gS_a.astype(np.float32)),
            syn_e_syn=torch.from_numpy(e_syn_a.astype(np.float32)),
            syn_v_th=torch.from_numpy(v_th_a.astype(np.float32)),
            syn_k_minus=torch.from_numpy(k_m_a.astype(np.float32)),
            syn_delta=torch.from_numpy(delta_a.astype(np.float32)),
            syn_count=syn_count_t,
            input_node_mask=input_mask, cell_types=id_ct_list, dt=dt,
            **kwargs,
        )
        net.root_ids = all_ids.astype(np.int64)
        net.raw_cell_types = id_ct_list
        net.resolved_cell_types = param_resolved_type
        net.neuron_nt_types = nt_arr.tolist()
        net.neuron_param_source = param_source
        net.neuron_fallback_target = param_fallback_target
        net.synapse_param_source = syn_param_source
        net.synapse_fallback_target = syn_fallback_target
        net.root_ion_overrides_count = len(root_ion_overrides)
        logger.info(
            "Ion init hierarchy: %d neurotransmitter defaults, %d type rules, %d root overrides",
            len(nt_ion_rules), len(ion_rules), len(root_ion_overrides),
        )
        logger.info(
            "Synapse init hierarchy: %d exact type rules, %d pre-type defaults, %d NT defaults",
            sum(1 for s in syn_param_source if s.startswith("type:")),
            sum(1 for s in syn_param_source if s.startswith("pre_type:")),
            sum(1 for s in syn_param_source if s.startswith("nt:")),
        )
        return net

    @staticmethod
    def _build_global_morphology_from_packages(
        loader,
        root_ids: np.ndarray,
        ncomp: int = 1,
        show_progress: bool = True,
        progress_every: int = 100,
        fallback_rows=None,
    ) -> Tuple[MorphologyGraph, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Concatenate per-neuron morphologies into one global sparse graph."""
        import time

        from .morphology_pack import _get_rss_mb

        all_areas: List[np.ndarray] = []
        all_volumes: List[np.ndarray] = []
        all_r_in: List[np.ndarray] = []
        all_r_out: List[np.ndarray] = []
        all_radii: List[np.ndarray] = []
        all_lengths: List[np.ndarray] = []
        comp_edges: List[Tuple[int, int]] = []
        branch_parents: List[int] = []
        comp_owner: List[int] = []
        soma_comp_idx: List[int] = []
        target_comp_idx: List[int] = []
        neuron_n_comp = np.zeros(len(root_ids), dtype=np.int32)
        morph_node_counts = np.zeros(len(root_ids), dtype=np.int32)
        fallback_root_ids: List[int] = []

        comp_offset = 0
        branch_offset = 0
        t0 = time.time()

        for i, rid in enumerate(root_ids, start=1):
            try:
                packed = loader.load_neuron(int(rid))
                packed_rows = packed.nodes
                packed_node_count = packed.node_count
            except Exception:
                if fallback_rows is None:
                    raise
                packed_rows = fallback_rows
                packed_node_count = len(fallback_rows)
                fallback_root_ids.append(int(rid))

            morph = MorphologyGraph.from_swc_rows(packed_rows, ncomp=ncomp)
            if morph.n_comp <= 0:
                raise ValueError(f"Invalid morphology with zero compartments for root_id={int(rid)}")

            neuron_idx = i - 1
            morph_node_counts[neuron_idx] = packed_node_count
            neuron_n_comp[neuron_idx] = morph.n_comp
            soma_comp_idx.append(comp_offset)
            target_comp_idx.append(comp_offset + morph.n_comp - 1)
            comp_owner.extend([neuron_idx] * morph.n_comp)

            all_areas.append(np.asarray(morph.areas))
            all_volumes.append(np.asarray(morph.volumes))
            all_r_in.append(np.asarray(morph.resistive_load_in))
            all_r_out.append(np.asarray(morph.resistive_load_out))
            all_radii.append(np.asarray(morph.radii))
            all_lengths.append(np.asarray(morph.lengths))
            comp_edges.extend([(src + comp_offset, dst + comp_offset) for src, dst in morph.comp_edges])
            branch_parents.extend(
                [
                    (parent + branch_offset) if parent >= 0 else -1
                    for parent in morph.branch_parents
                ]
            )

            comp_offset += morph.n_comp
            branch_offset += morph.n_branches

            if show_progress and (i % max(progress_every, 1) == 0 or i == len(root_ids)):
                msg = f"built {i:,}/{len(root_ids):,} neuron morphologies, total_comp={comp_offset:,}"
                rss = _get_rss_mb()
                if rss is not None:
                    msg += f", RSS={rss:.1f} MB"
                msg += f", elapsed={time.time() - t0:.1f}s"
                print(msg)

        morph = MorphologyGraph(
            n_comp=comp_offset,
            areas=np.concatenate(all_areas) if all_areas else np.zeros(0, dtype=np.float64),
            volumes=np.concatenate(all_volumes) if all_volumes else np.zeros(0, dtype=np.float64),
            resistive_load_in=np.concatenate(all_r_in) if all_r_in else np.zeros(0, dtype=np.float64),
            resistive_load_out=np.concatenate(all_r_out) if all_r_out else np.zeros(0, dtype=np.float64),
            radii=np.concatenate(all_radii) if all_radii else np.zeros(0, dtype=np.float64),
            lengths=np.concatenate(all_lengths) if all_lengths else np.zeros(0, dtype=np.float64),
            comp_edges=comp_edges,
            branch_parents=branch_parents,
            n_branches=branch_offset,
        )
        return (
            morph,
            np.asarray(comp_owner, dtype=np.int64),
            neuron_n_comp,
            np.asarray(soma_comp_idx, dtype=np.int64),
            np.asarray(target_comp_idx, dtype=np.int64),
            morph_node_counts,
            np.asarray(fallback_root_ids, dtype=np.int64),
        )


    @classmethod
    def from_preprocessed_with_morphology_packages(
        cls,
        data_dir: Union[str, Path],
        morphology_package_dir: Union[str, Path],
        ion_rules_path: Union[str, Path],
        syn_rules_path: Union[str, Path],
        ncomp: int = 2,
        min_syn_count: int = 2,
        dt: float = 0.025,
        morphology_limit: Optional[int] = None,
        morphology_types: Optional[Iterable[str]] = None,
        show_morphology_progress: bool = True,
        progress_every: int = 100,
        **kwargs,
    ) -> "FAFBMCNetwork":
        """Build network and switch to a per-neuron morphology solver."""
        import time

        from .morphology_pack import PackedMorphologyLoader, estimate_package_memory

        t0 = time.time()
        package_dir = Path(morphology_package_dir)
        loader = PackedMorphologyLoader(package_dir, preload_index=True)
        mem_est = estimate_package_memory(package_dir)
        logger.info(
            "Morphology packages: %d neurons, %d groups, %.1f MB node payload",
            loader.total_neurons,
            loader.total_groups,
            mem_est["estimated_node_payload_mb"],
        )
        if morphology_limit is not None or morphology_types is not None:
            raise ValueError(
                "Per-neuron morphology solver requires the full network morphology set; "
                "do not pass morphology_limit or morphology_types."
            )

        # Reuse the shared-morphology builder for neuron/synapse parameter lookup,
        # then replace the geometry with a concatenated per-neuron sparse graph.
        first_morph = next(
            loader.iter_neurons(
                limit=1,
                show_progress=False,
                report_memory=False,
            )
        )
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".swc", mode="w", delete=False) as tmp:
            tmp.write("# representative simplified morphology\n")
            for row in first_morph.nodes:
                tmp.write(
                    f"{int(row['id'])} {int(row['type'])} "
                    f"{float(row['x'])} {float(row['y'])} {float(row['z'])} "
                    f"{float(row['r'])} {int(row['parent'])}\n"
                )
            rep_swc = tmp.name

        try:
            net = cls.from_preprocessed(
                data_dir=data_dir,
                swc_path=rep_swc,
                ion_rules_path=ion_rules_path,
                syn_rules_path=syn_rules_path,
                ncomp=ncomp,
                min_syn_count=min_syn_count,
                dt=dt,
                **kwargs,
            )
        finally:
            Path(rep_swc).unlink(missing_ok=True)

        global_morph, comp_owner, neuron_n_comp, soma_comp_idx, target_comp_idx, morph_node_counts, fallback_root_ids = (
            cls._build_global_morphology_from_packages(
                loader=loader,
                root_ids=net.root_ids,
                ncomp=ncomp,
                show_progress=show_morphology_progress,
                progress_every=progress_every,
                fallback_rows=first_morph.nodes,
            )
        )
        geom = MultiCompartmentHH._build_geom_matrix(global_morph)
        net.areas = torch.tensor(global_morph.areas, dtype=torch.float64)
        net.G_row = geom[0]
        net.G_col = geom[1]
        net.G_val = geom[2]
        net.register_buffer("comp_owner", torch.from_numpy(comp_owner).long())
        net.register_buffer("neuron_n_comp", torch.from_numpy(neuron_n_comp).long())
        net.register_buffer("soma_comp_idx", torch.from_numpy(soma_comp_idx).long())
        net.register_buffer("target_comp_idx", torch.from_numpy(target_comp_idx).long())
        net.pre_comp = net.soma_comp_idx[net.pre_idx]
        net.post_comp = net.target_comp_idx[net.post_idx]
        net.total_comp = int(global_morph.n_comp)
        net.n_comp = int(neuron_n_comp.max()) if len(neuron_n_comp) else 0
        net.ragged_mode = True

        loaded = int((morph_node_counts > 0).sum())
        if show_morphology_progress:
            try:
                from .morphology_pack import _get_rss_mb

                rss = _get_rss_mb()
            except Exception:
                rss = None
            msg = (
                f"per-neuron solver ready: neurons={net.n_neurons:,}, "
                f"total_comp={net.total_comp:,}, max_comp={net.n_comp:,}"
            )
            if rss is not None:
                msg += f", RSS={rss:.1f} MB"
            print(msg)

        net.morphology_graph_mode = "per_neuron_global_sparse"
        net.morphology_compartment_total = net.total_comp
        net.morphology_compartment_max = net.n_comp
        net.morphology_loader = loader
        net.morphology_package_dir = str(package_dir)
        net.morphology_memory_estimate = mem_est
        net.morphology_node_counts = morph_node_counts
        net.loaded_morphology_count = loaded
        net.morphology_fallback_root_ids = fallback_root_ids
        net.morphology_fallback_count = int(len(fallback_root_ids))
        net.morphology_target_policy = "last_compartment"
        logger.info(
            "Built per-neuron morphology solver for %d neurons, %d total compartments, %d fallbacks [%.1fs]",
            net.n_neurons,
            net.total_comp,
            len(fallback_root_ids),
            time.time() - t0,
        )
        logger.info(
            "Attached morphology package metadata for %d/%d network neurons [%.1fs]",
            loaded,
            net.n_neurons,
            time.time() - t0,
        )
        return net

    # -- state --
    def init_state(self, batch_size: int = 1,
                   device: torch.device = torch.device("cpu")) -> Dict[str, Tensor]:
        if self.ragged_mode:
            M = self.total_comp
            base_e_l = self.E_L[self.comp_owner].to(device).double()
            V0 = base_e_l.unsqueeze(0).expand(batch_size, M).clone()
        else:
            N, C = self.n_neurons, self.n_comp
            V0 = self.E_L.unsqueeze(0).unsqueeze(-1).expand(batch_size, N, C).clone().to(device).double()
        return {
            "V": V0,
            "m": _x_inf(_alpha_m, _beta_m, V0),
            "h": _x_inf(_alpha_h, _beta_h, V0),
            "n": _x_inf(_alpha_n, _beta_n, V0),
            "s": torch.zeros(batch_size, self.n_edges, device=device, dtype=torch.float64),
        }

    # -- single step --
    def _step(self, state: Dict[str, Tensor], I_ext: Tensor, dt: float) -> Dict[str, Tensor]:
        V, m, h, n, s = state["V"], state["m"], state["h"], state["n"], state["s"]

        g_Na = self._g_Na_bc
        g_K = self._g_K_bc
        g_L = self._g_L_bc
        E_Na = self._E_Na_bc
        E_K = self._E_K_bc
        E_L = self._E_L_bc

        g_tot = g_Na * m**3 * h + g_K * n**4 + g_L
        vt = g_tot / self.C_m
        ct = (g_Na * m**3 * h * E_Na + g_K * n**4 * E_K + g_L * E_L + I_ext) / self.C_m

        if self.ragged_mode:
            B, M = V.shape
            if self.n_edges > 0:
                pre_v = V[:, self.pre_comp]
                post_v = V[:, self.post_comp]

                s_inf = 1.0 / (1.0 + torch.exp((self._syn_v_th_d - pre_v) / self._syn_delta_d))
                tau_s = (1.0 - s_inf) / (self._syn_k_minus_d + 1e-8)
                slope = -1.0 / (tau_s + 1e-8)
                exp_slope = torch.exp(slope * dt)
                s_new = s * exp_slope + s_inf * (1.0 - exp_slope)

                I_syn_edge = self._syn_gS_w * s_new * (post_v - self._syn_e_syn_d)

                syn_contrib = torch.zeros(B, M, dtype=torch.float64, device=V.device)
                syn_contrib.scatter_add_(1, self._flat_target_exp[:B], -I_syn_edge / self.C_m)
                ct = ct + syn_contrib
            else:
                s_new = s

            V_new = _solve_implicit(
                self.G_row, self.G_col, self.G_val,
                self.R_a, self.C_m, self.areas,
                V, vt, ct, dt,
            )
        else:
            B, N, C = V.shape
            if self.n_edges > 0:
                pre_v = V[:, self.pre_idx, self.pre_comp]
                post_v = V[:, self.post_idx, self.post_comp]

                s_inf = 1.0 / (1.0 + torch.exp((self._syn_v_th_d - pre_v) / self._syn_delta_d))
                tau_s = (1.0 - s_inf) / (self._syn_k_minus_d + 1e-8)
                slope = -1.0 / (tau_s + 1e-8)
                exp_slope = torch.exp(slope * dt)
                s_new = s * exp_slope + s_inf * (1.0 - exp_slope)

                I_syn_edge = self._syn_gS_w * s_new * (post_v - self._syn_e_syn_d)

                syn_contrib = torch.zeros(B, N * C, dtype=torch.float64, device=V.device)
                syn_contrib.scatter_add_(1, self._flat_target_exp[:B], -I_syn_edge / self.C_m)
                ct = ct + syn_contrib.view(B, N, C)
            else:
                s_new = s

            V_flat = V.reshape(B * N, C)
            vt_flat = vt.reshape(B * N, C)
            ct_flat = ct.reshape(B * N, C)

            V_new = _solve_implicit(
                self.G_row, self.G_col, self.G_val,
                self.R_a, self.C_m, self.areas,
                V_flat, vt_flat, ct_flat, dt,
            ).reshape(B, N, C)

        V_new = torch.clamp(V_new, -120.0, 80.0)

        am, bm = _alpha_m(V), _beta_m(V)
        ah, bh = _alpha_h(V), _beta_h(V)
        an, bn = _alpha_n(V), _beta_n(V)

        return {
            "V": V_new,
            "m": torch.clamp(m + dt * (am * (1 - m) - bm * m), 0, 1),
            "h": torch.clamp(h + dt * (ah * (1 - h) - bh * h), 0, 1),
            "n": torch.clamp(n + dt * (an * (1 - n) - bn * n), 0, 1),
            "s": torch.clamp(s_new, 0, 1),
        }

    def _cache_step_tensors(self, max_batch: int = 4):
        """Pre-compute constant tensors used in _step to avoid reallocation."""
        if self.ragged_mode:
            self._g_Na_bc = self.g_Na[self.comp_owner].unsqueeze(0).double()
            self._g_K_bc = self.g_K[self.comp_owner].unsqueeze(0).double()
            self._g_L_bc = self.g_L[self.comp_owner].unsqueeze(0).double()
            self._E_Na_bc = self.E_Na[self.comp_owner].unsqueeze(0).double()
            self._E_K_bc = self.E_K[self.comp_owner].unsqueeze(0).double()
            self._E_L_bc = self.E_L[self.comp_owner].unsqueeze(0).double()
        else:
            self._g_Na_bc = self.g_Na.unsqueeze(0).unsqueeze(-1).double()
            self._g_K_bc = self.g_K.unsqueeze(0).unsqueeze(-1).double()
            self._g_L_bc = self.g_L.unsqueeze(0).unsqueeze(-1).double()
            self._E_Na_bc = self.E_Na.unsqueeze(0).unsqueeze(-1).double()
            self._E_K_bc = self.E_K.unsqueeze(0).unsqueeze(-1).double()
            self._E_L_bc = self.E_L.unsqueeze(0).unsqueeze(-1).double()

        self._syn_v_th_d = self.syn_v_th.double()
        self._syn_delta_d = self.syn_delta.double().clamp(min=0.1)
        self._syn_k_minus_d = self.syn_k_minus.double()
        self._syn_e_syn_d = self.syn_e_syn.double()

        # Per-edge weight: syn_count weighted by post-neuron in-degree
        # so total syn input per neuron stays bounded
        in_degree = torch.zeros(self.n_neurons, device=self.syn_count.device)
        in_degree.scatter_add_(0, self.post_idx, torch.ones_like(self.post_idx, dtype=torch.float32))
        in_degree.clamp_(min=1.0)
        per_edge_norm = 1.0 / in_degree[self.post_idx]

        w = self.syn_count * per_edge_norm
        w = w / (w.mean() + 1e-8)
        self._syn_gS_w = (self.syn_gS.double() * w.double())

        flat_target = self.post_comp if self.ragged_mode else (self.post_idx * self.n_comp + self.post_comp)
        self._flat_target_exp = flat_target.unsqueeze(0).expand(max_batch, -1)

    # -- forward --
    def forward(self, x: Tensor, dt: Optional[float] = None,
                state: Optional[Dict[str, Tensor]] = None,
                show_progress: bool = True) -> Tensor:
        """Simulate and return soma voltages (B, T, N).

        Parameters
        ----------
        show_progress : bool
            If True, show a tqdm progress bar (works in both terminal and
            Jupyter).  Falls back to simple print if tqdm is unavailable.
        """
        dt = dt or self.dt
        B, T = x.shape[0], x.shape[1]
        N = self.n_neurons

        n_input = x.shape[2]
        if n_input == N:
            pass
        else:
            n_photo = self.input_mask.sum().item()
            assert n_input == n_photo, f"Input {n_input} != N={N} and != n_photo={n_photo}"
            x_mapped = torch.zeros(B, T, N, device=x.device, dtype=x.dtype)
            photo_indices = torch.where(self.input_mask)[0]
            x_mapped[:, :, photo_indices] = x
            x = x_mapped

        if state is None:
            state = self.init_state(B, device=x.device)

        self._cache_step_tensors(max_batch=B)

        traces: List[Tensor] = []

        # Progress bar
        pbar = None
        if show_progress:
            try:
                from tqdm.auto import tqdm
                pbar = tqdm(total=T, desc=f"Sim {N}N/{self.n_edges}E",
                            unit="step")
            except ImportError:
                pass

        for t_i in range(T):
            if self.ragged_mode:
                I_ext = torch.zeros(B, self.total_comp, dtype=torch.float64, device=x.device)
                I_ext[:, self.soma_comp_idx] = x[:, t_i].double()
                soma_v = self.soma_comp_idx
            else:
                I_ext = torch.zeros(B, N, self.n_comp, dtype=torch.float64, device=x.device)
                I_ext[:, :, 0] = x[:, t_i].double()
                soma_v = 0
            state = self._step(state, I_ext, dt)
            if self.ragged_mode:
                traces.append(state["V"][:, soma_v].float())
            else:
                traces.append(state["V"][:, :, soma_v].float())
            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        return torch.stack(traces, dim=1)

    def simulate(self, x: Tensor, **kw) -> Tensor:
        return self.forward(x, **kw)

    # -- utilities --
    def get_photoreceptor_indices(self) -> Tensor:
        return torch.where(self.input_mask)[0]

    def get_indices_by_type(self, cell_type: str) -> List[int]:
        return [i for i, ct in enumerate(self.cell_types) if ct == cell_type]

    def type_counts(self) -> Dict[str, int]:
        from collections import Counter
        return dict(Counter(self.cell_types))

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        n_photo = self.input_mask.sum().item()
        if self.ragged_mode:
            return (
                f"n_neurons={self.n_neurons}, total_comp={self.total_comp}, "
                f"max_comp={self.n_comp}, n_edges={self.n_edges}, "
                f"n_photo={n_photo}, trainable={self.n_parameters()}"
            )
        return (
            f"n_neurons={self.n_neurons}, n_comp={self.n_comp}, "
            f"n_edges={self.n_edges}, n_photo={n_photo}, "
            f"trainable={self.n_parameters()}"
        )
