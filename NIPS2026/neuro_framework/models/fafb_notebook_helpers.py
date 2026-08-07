from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .fafb_mc_network import FAFBMCNetwork


def mode_or_na(series: pd.Series):
    mode = series.mode()
    return mode.iloc[0] if not mode.empty else None


_DIRECT_PARAM_SPECS = {
    "E_L": "E_L",
    "E_Na": "E_Na",
    "E_K": "E_K",
    "syn_v_th": "syn_v_th",
    "syn_e_syn": "syn_e_syn",
    "syn_k_minus": "syn_k_minus",
    "syn_delta": "syn_delta",
}

_LOG_PARAM_SPECS = {
    "g_Na": "log_g_Na",
    "g_K": "log_g_K",
    "g_L": "log_g_L",
    "syn_gS": "log_syn_gS",
}


def _coerce_list(values):
    if values is None:
        return None
    if isinstance(values, (list, tuple, set, np.ndarray, pd.Series)):
        return list(values)
    return [values]


def _ensure_override_bases(net: FAFBMCNetwork) -> None:
    if not hasattr(net, "_override_base_E_L"):
        net._override_base_E_L = net.E_L.detach().clone()
        net._override_base_E_Na = net.E_Na.detach().clone()
        net._override_base_E_K = net.E_K.detach().clone()
        net._override_base_syn_v_th = net.syn_v_th.detach().clone()
        net._override_base_syn_e_syn = net.syn_e_syn.detach().clone()
        net._override_base_syn_k_minus = net.syn_k_minus.detach().clone()
        net._override_base_syn_delta = net.syn_delta.detach().clone()
        net._override_base_log_g_Na = net.log_g_Na.detach().clone()
        net._override_base_log_g_K = net.log_g_K.detach().clone()
        net._override_base_log_g_L = net.log_g_L.detach().clone()
        net._override_base_log_syn_gS = net.log_syn_gS.detach().clone()


def reset_postbuild_parameter_overrides(net: FAFBMCNetwork) -> FAFBMCNetwork:
    _ensure_override_bases(net)
    with torch.no_grad():
        net.E_L.copy_(net._override_base_E_L)
        net.E_Na.copy_(net._override_base_E_Na)
        net.E_K.copy_(net._override_base_E_K)
        net.syn_v_th.copy_(net._override_base_syn_v_th)
        net.syn_e_syn.copy_(net._override_base_syn_e_syn)
        net.syn_k_minus.copy_(net._override_base_syn_k_minus)
        net.syn_delta.copy_(net._override_base_syn_delta)
        net.log_g_Na.copy_(net._override_base_log_g_Na)
        net.log_g_K.copy_(net._override_base_log_g_K)
        net.log_g_L.copy_(net._override_base_log_g_L)
        net.log_syn_gS.copy_(net._override_base_log_syn_gS)
    net.manual_postbuild_parameter_overrides = {"neuron_rules": [], "synapse_rules": []}
    return net


def _make_neuron_mask(
    net: FAFBMCNetwork,
    *,
    cell_types: Optional[Sequence[str]] = None,
    root_ids: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    mask_np = np.ones(net.n_neurons, dtype=bool)
    if cell_types is not None:
        mask_np &= np.isin(np.array(net.cell_types, dtype=object), _coerce_list(cell_types))
    if root_ids is not None:
        mask_np &= np.isin(np.array(net.root_ids), np.array(_coerce_list(root_ids), dtype=np.int64))
    return torch.tensor(mask_np, dtype=torch.bool, device=net.E_L.device)


def _make_synapse_mask(
    net: FAFBMCNetwork,
    *,
    pre_cell_types: Optional[Sequence[str]] = None,
    post_cell_types: Optional[Sequence[str]] = None,
    pre_root_ids: Optional[Sequence[int]] = None,
    post_root_ids: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    pre_idx_np = net.pre_idx.detach().cpu().numpy()
    post_idx_np = net.post_idx.detach().cpu().numpy()
    pre_types_np = np.array(net.cell_types, dtype=object)[pre_idx_np]
    post_types_np = np.array(net.cell_types, dtype=object)[post_idx_np]
    pre_root_ids_np = np.array(net.root_ids)[pre_idx_np]
    post_root_ids_np = np.array(net.root_ids)[post_idx_np]
    mask_np = np.ones(net.n_edges, dtype=bool)
    if pre_cell_types is not None:
        mask_np &= np.isin(pre_types_np, _coerce_list(pre_cell_types))
    if post_cell_types is not None:
        mask_np &= np.isin(post_types_np, _coerce_list(post_cell_types))
    if pre_root_ids is not None:
        mask_np &= np.isin(pre_root_ids_np, np.array(_coerce_list(pre_root_ids), dtype=np.int64))
    if post_root_ids is not None:
        mask_np &= np.isin(post_root_ids_np, np.array(_coerce_list(post_root_ids), dtype=np.int64))
    return torch.tensor(mask_np, dtype=torch.bool, device=net.syn_v_th.device)


def _apply_rule_update_to_direct_param(param: torch.nn.Parameter, mask: torch.Tensor, spec: Dict) -> None:
    if not mask.any():
        return
    current = param[mask]
    if "set" in spec:
        current = torch.full_like(current, float(spec["set"]))
    if "scale" in spec:
        current = current * float(spec["scale"])
    if "shift" in spec:
        current = current + float(spec["shift"])
    param[mask] = current


def _apply_rule_update_to_log_param(log_param: torch.nn.Parameter, mask: torch.Tensor, spec: Dict) -> None:
    if not mask.any():
        return
    current_ms = torch.exp(log_param[mask]) * 1e3
    if "set" in spec:
        current_ms = torch.full_like(current_ms, float(spec["set"]))
    if "scale" in spec:
        current_ms = current_ms * float(spec["scale"])
    if "shift" in spec:
        current_ms = current_ms + float(spec["shift"])
    current_ms = current_ms.clamp(min=1e-8)
    log_param[mask] = torch.log(current_ms / 1e3)


def apply_postbuild_parameter_overrides(
    net: FAFBMCNetwork,
    *,
    neuron_rules: Optional[List[Dict]] = None,
    synapse_rules: Optional[List[Dict]] = None,
    reset_first: bool = True,
) -> FAFBMCNetwork:
    neuron_rules = list(neuron_rules or [])
    synapse_rules = list(synapse_rules or [])
    _ensure_override_bases(net)
    if reset_first:
        reset_postbuild_parameter_overrides(net)

    neuron_rule_summaries = []
    synapse_rule_summaries = []
    neuron_masks = []
    synapse_masks = []

    with torch.no_grad():
        for rule in neuron_rules:
            mask = _make_neuron_mask(
                net,
                cell_types=rule.get("cell_types"),
                root_ids=rule.get("root_ids"),
            )
            neuron_masks.append(mask.detach().cpu().numpy())
            neuron_rule_summaries.append(
                {
                    "cell_types": list(_coerce_list(rule.get("cell_types")) or []),
                    "root_ids": list(_coerce_list(rule.get("root_ids")) or []),
                    "updates": dict(rule.get("updates", {})),
                    "matched_neurons": int(mask.sum().item()),
                }
            )
            updates = dict(rule.get("updates", {}))
            for name, spec in updates.items():
                if name in _DIRECT_PARAM_SPECS:
                    _apply_rule_update_to_direct_param(getattr(net, _DIRECT_PARAM_SPECS[name]), mask, spec)
                elif name in _LOG_PARAM_SPECS:
                    _apply_rule_update_to_log_param(getattr(net, _LOG_PARAM_SPECS[name]), mask, spec)
                else:
                    raise KeyError(f"Unsupported neuron parameter override: {name}")

        for rule in synapse_rules:
            mask = _make_synapse_mask(
                net,
                pre_cell_types=rule.get("pre_cell_types"),
                post_cell_types=rule.get("post_cell_types"),
                pre_root_ids=rule.get("pre_root_ids"),
                post_root_ids=rule.get("post_root_ids"),
            )
            synapse_masks.append(mask.detach().cpu().numpy())
            synapse_rule_summaries.append(
                {
                    "pre_cell_types": list(_coerce_list(rule.get("pre_cell_types")) or []),
                    "post_cell_types": list(_coerce_list(rule.get("post_cell_types")) or []),
                    "pre_root_ids": list(_coerce_list(rule.get("pre_root_ids")) or []),
                    "post_root_ids": list(_coerce_list(rule.get("post_root_ids")) or []),
                    "updates": dict(rule.get("updates", {})),
                    "matched_synapses": int(mask.sum().item()),
                }
            )
            updates = dict(rule.get("updates", {}))
            for name, spec in updates.items():
                if name in _DIRECT_PARAM_SPECS:
                    _apply_rule_update_to_direct_param(getattr(net, _DIRECT_PARAM_SPECS[name]), mask, spec)
                elif name in _LOG_PARAM_SPECS:
                    _apply_rule_update_to_log_param(getattr(net, _LOG_PARAM_SPECS[name]), mask, spec)
                else:
                    raise KeyError(f"Unsupported synapse parameter override: {name}")

    net.manual_postbuild_parameter_overrides = {
        "neuron_rules": neuron_rules,
        "synapse_rules": synapse_rules,
    }
    neuron_union = int(np.logical_or.reduce(neuron_masks).sum()) if neuron_masks else 0
    synapse_union = int(np.logical_or.reduce(synapse_masks).sum()) if synapse_masks else 0
    net.manual_override_summary = {
        "neuron_rule_count": len(neuron_rule_summaries),
        "synapse_rule_count": len(synapse_rule_summaries),
        "unique_neurons_touched": neuron_union,
        "unique_synapses_touched": synapse_union,
        "neuron_rule_summaries": neuron_rule_summaries,
        "synapse_rule_summaries": synapse_rule_summaries,
    }
    return net


def build_pathway_override_rules(
    *,
    eLeak_shift_mV: float = 0.0,
    v_th_shift_mV: float = 0.0,
    target_v_th_shifts_mV: Optional[Dict[str, float]] = None,
    target_gs_gains: Optional[Dict[str, float]] = None,
    pre_cell_types: Optional[Sequence[str]] = None,
    target_root_ids: Optional[Sequence[int]] = None,
) -> Tuple[List[Dict], List[Dict]]:
    pre_cell_types = list(pre_cell_types or ["R1-6", "R1-R6"])
    neuron_rules = []
    synapse_rules = []
    if eLeak_shift_mV != 0.0:
        neuron_rules.append(
            {
                "cell_types": pre_cell_types,
                "root_ids": target_root_ids,
                "updates": {"E_L": {"shift": float(eLeak_shift_mV)}},
            }
        )
    if v_th_shift_mV != 0.0:
        synapse_rules.append(
            {
                "pre_cell_types": pre_cell_types,
                "pre_root_ids": target_root_ids,
                "updates": {"syn_v_th": {"shift": float(v_th_shift_mV)}},
            }
        )
    for post_type, shift in (target_v_th_shifts_mV or {}).items():
        if shift == 0.0:
            continue
        synapse_rules.append(
            {
                "pre_cell_types": pre_cell_types,
                "pre_root_ids": target_root_ids,
                "post_cell_types": [post_type],
                "updates": {"syn_v_th": {"shift": float(shift)}},
            }
        )
    for post_type, gain in (target_gs_gains or {}).items():
        if gain in (0.0, 1.0):
            continue
        synapse_rules.append(
            {
                "pre_cell_types": pre_cell_types,
                "pre_root_ids": target_root_ids,
                "post_cell_types": [post_type],
                "updates": {"syn_gS": {"scale": float(gain)}},
            }
        )
    return neuron_rules, synapse_rules


def build_r16_override_rules(
    *,
    eLeak_shift_mV: float = 0.0,
    v_th_shift_mV: float = 0.0,
    target_v_th_shifts_mV: Optional[Dict[str, float]] = None,
    target_gs_gains: Optional[Dict[str, float]] = None,
    pre_cell_types: Optional[Sequence[str]] = None,
    target_root_ids: Optional[Sequence[int]] = None,
) -> Tuple[List[Dict], List[Dict]]:
    return build_pathway_override_rules(
        eLeak_shift_mV=eLeak_shift_mV,
        v_th_shift_mV=v_th_shift_mV,
        target_v_th_shifts_mV=target_v_th_shifts_mV,
        target_gs_gains=target_gs_gains,
        pre_cell_types=pre_cell_types,
        target_root_ids=target_root_ids,
    )


def apply_manual_r16_overrides(
    net: FAFBMCNetwork,
    eLeak_shift_mV: float = 0.0,
    v_th_shift_mV: float = 0.0,
    target_v_th_shifts_mV: Optional[Dict[str, float]] = None,
    target_gs_gains: Optional[Dict[str, float]] = None,
    target_root_ids: Optional[Sequence[int]] = None,
) -> FAFBMCNetwork:
    neuron_rules, synapse_rules = build_pathway_override_rules(
        eLeak_shift_mV=eLeak_shift_mV,
        v_th_shift_mV=v_th_shift_mV,
        target_v_th_shifts_mV=target_v_th_shifts_mV,
        target_gs_gains=target_gs_gains,
        target_root_ids=target_root_ids,
    )
    return apply_postbuild_parameter_overrides(
        net,
        neuron_rules=neuron_rules,
        synapse_rules=synapse_rules,
        reset_first=True,
    )


def build_optic_lobe_net(
    *,
    data_dir: str,
    morphology_package_dir: str,
    ion_rules_path: str,
    syn_rules_path: str,
    nt_ion_rules_path: Optional[str] = None,
    neuron_ion_overrides_path: Optional[str] = None,
    dt: float = 0.1,
    ncomp: int = 2,
    min_syn_count: int = 3,
    morphology_progress_every: int = 5000,
) -> Tuple[FAFBMCNetwork, float]:
    t0 = time.time()
    net = FAFBMCNetwork.from_preprocessed_with_morphology_packages(
        data_dir=data_dir,
        morphology_package_dir=morphology_package_dir,
        ion_rules_path=ion_rules_path,
        syn_rules_path=syn_rules_path,
        nt_ion_rules_path=nt_ion_rules_path,
        neuron_ion_overrides_path=neuron_ion_overrides_path,
        ncomp=ncomp,
        min_syn_count=min_syn_count,
        dt=dt,
        show_morphology_progress=True,
        progress_every=morphology_progress_every,
    )
    return net, time.time() - t0


def save_cached_net(
    net: FAFBMCNetwork,
    path: str | Path,
    meta: Optional[Dict] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"net": net.cpu(), "meta": meta or {}}
    torch.save(payload, path)


def load_cached_net(path: str | Path) -> Tuple[Optional[FAFBMCNetwork], Optional[Dict]]:
    path = Path(path)
    if not path.exists():
        return None, None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "net" in payload:
        return payload["net"], payload.get("meta", {})
    return payload, None


def load_or_build_cached_net(
    *,
    cache_path: str | Path,
    force_rebuild: bool = False,
    save_cache_after_build: bool = True,
    cache_meta: Optional[Dict] = None,
    **build_kwargs,
) -> Tuple[FAFBMCNetwork, Dict, bool, float]:
    cache_path = Path(cache_path)
    if cache_path.exists() and not force_rebuild:
        net, meta = load_cached_net(cache_path)
        if net is None:
            raise RuntimeError(f"Failed to load cached net from {cache_path}")
        loaded_meta = meta or {}
        desired_meta = cache_meta or {}
        mismatch_keys = [k for k, v in desired_meta.items() if loaded_meta.get(k) != v]
        if not mismatch_keys:
            return net, loaded_meta, True, 0.0
        print(f"Cache metadata mismatch for {cache_path}; rebuilding. Keys: {mismatch_keys}")

    net, build_elapsed = build_optic_lobe_net(**build_kwargs)
    meta = dict(cache_meta or {})
    if save_cache_after_build:
        save_cached_net(net, cache_path, meta=meta)
    return net.cpu(), meta, False, build_elapsed


def net_summary(net: FAFBMCNetwork) -> Dict:
    return {
        "n_neurons": net.n_neurons,
        "n_edges": net.n_edges,
        "n_photo": int(net.input_mask.sum().item()) if hasattr(net, "input_mask") else None,
        "morphology_mode": getattr(net, "morphology_graph_mode", "shared"),
        "total_comp": getattr(net, "total_comp", net.n_comp),
        "max_comp_per_neuron": getattr(net, "n_comp", None),
        "loaded_morphology_count": getattr(net, "loaded_morphology_count", None),
        "morphology_fallback_count": getattr(net, "morphology_fallback_count", None),
        "root_override_count": getattr(net, "root_ion_overrides_count", None),
        "manual_override_summary": getattr(net, "manual_override_summary", None),
    }


def neuron_index_from_root_id(net: FAFBMCNetwork, root_id: int) -> int:
    root_id = int(root_id)
    hits = np.where(net.root_ids == root_id)[0]
    if len(hits) == 0:
        raise KeyError(f"root_id {root_id} not found")
    return int(hits[0])


def neuron_indices_from_type(net: FAFBMCNetwork, cell_type: str, limit: Optional[int] = 20) -> List[int]:
    hits = [i for i, ct in enumerate(net.cell_types) if ct == cell_type]
    return hits if limit is None else hits[:limit]


def type_indices(net: FAFBMCNetwork, cell_type: str) -> np.ndarray:
    return np.where(np.array(net.cell_types, dtype=object) == cell_type)[0]


def neuron_row(net: FAFBMCNetwork, idx: int) -> pd.Series:
    idx = int(idx)
    return pd.Series(
        {
            "neuron_idx": idx,
            "root_id": int(net.root_ids[idx]),
            "cell_type": net.cell_types[idx],
            "resolved_cell_type": net.resolved_cell_types[idx] if hasattr(net, "resolved_cell_types") else None,
            "dominant_nt": net.neuron_nt_types[idx] if hasattr(net, "neuron_nt_types") else None,
            "param_source": net.neuron_param_source[idx] if hasattr(net, "neuron_param_source") else None,
            "fallback_target": net.neuron_fallback_target[idx] if hasattr(net, "neuron_fallback_target") else None,
            "gNa_mS_cm2": float(net.g_Na[idx].detach().cpu().item()),
            "gK_mS_cm2": float(net.g_K[idx].detach().cpu().item()),
            "gLeak_mS_cm2": float(net.g_L[idx].detach().cpu().item()),
            "eNa_mV": float(net.E_Na[idx].detach().cpu().item()),
            "eK_mV": float(net.E_K[idx].detach().cpu().item()),
            "eLeak_mV": float(net.E_L[idx].detach().cpu().item()),
            "n_solver_compartments": int(net.neuron_n_comp[idx].item()) if hasattr(net, "neuron_n_comp") else int(net.n_comp),
            "n_swc_nodes": int(net.morphology_node_counts[idx]) if hasattr(net, "morphology_node_counts") else None,
            "soma_comp_idx": int(net.soma_comp_idx[idx].item()) if hasattr(net, "soma_comp_idx") else None,
            "target_comp_idx": int(net.target_comp_idx[idx].item()) if hasattr(net, "target_comp_idx") else None,
        }
    )


def edge_table_for_neuron(
    net: FAFBMCNetwork,
    idx: int,
    direction: str = "out",
    limit: int = 20,
) -> pd.DataFrame:
    idx = int(idx)
    if direction == "out":
        mask = net.pre_idx == idx
        partner_key = "post_idx"
        partner_type_key = "post_type"
        partner_root_key = "post_root_id"
    elif direction == "in":
        mask = net.post_idx == idx
        partner_key = "pre_idx"
        partner_type_key = "pre_type"
        partner_root_key = "pre_root_id"
    else:
        raise ValueError("direction must be 'in' or 'out'")

    edge_ids = torch.where(mask)[0][:limit]
    rows = []
    for e in edge_ids.tolist():
        pre_i = int(net.pre_idx[e].item())
        post_i = int(net.post_idx[e].item())
        partner_i = int(getattr(net, partner_key)[e].item())
        rows.append(
            {
                "edge_idx": e,
                "pre_idx": pre_i,
                "post_idx": post_i,
                "pre_root_id": int(net.root_ids[pre_i]),
                "post_root_id": int(net.root_ids[post_i]),
                "pre_type": net.cell_types[pre_i],
                "post_type": net.cell_types[post_i],
                partner_root_key: int(net.root_ids[partner_i]),
                partner_type_key: net.cell_types[partner_i],
                "syn_count": float(net.syn_count[e].detach().cpu().item()) if net.syn_count is not None else None,
                "gS_mS_cm2": float(net.syn_gS[e].detach().cpu().item()),
                "e_syn_mV": float(net.syn_e_syn[e].detach().cpu().item()),
                "v_th_mV": float(net.syn_v_th[e].detach().cpu().item()),
                "k_minus": float(net.syn_k_minus[e].detach().cpu().item()),
                "delta_mV": float(net.syn_delta[e].detach().cpu().item()),
                "param_source": net.synapse_param_source[e] if hasattr(net, "synapse_param_source") else None,
                "fallback_target": net.synapse_fallback_target[e] if hasattr(net, "synapse_fallback_target") else None,
            }
        )
    return pd.DataFrame(rows)


def build_per_neuron_table(net: FAFBMCNetwork) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "neuron_idx": np.arange(net.n_neurons),
            "root_id": net.root_ids,
            "cell_type": net.cell_types,
            "resolved_cell_type": getattr(net, "resolved_cell_types", net.cell_types),
            "dominant_nt": getattr(net, "neuron_nt_types", [None] * net.n_neurons),
            "param_source": getattr(net, "neuron_param_source", [None] * net.n_neurons),
            "fallback_target": getattr(net, "neuron_fallback_target", [None] * net.n_neurons),
            "gNa_mS_cm2": net.g_Na.detach().cpu().numpy(),
            "gK_mS_cm2": net.g_K.detach().cpu().numpy(),
            "gLeak_mS_cm2": net.g_L.detach().cpu().numpy(),
            "eNa_mV": net.E_Na.detach().cpu().numpy(),
            "eK_mV": net.E_K.detach().cpu().numpy(),
            "eLeak_mV": net.E_L.detach().cpu().numpy(),
            "n_solver_compartments": net.neuron_n_comp.detach().cpu().numpy() if hasattr(net, "neuron_n_comp") else np.full(net.n_neurons, net.n_comp),
            "n_swc_nodes": getattr(net, "morphology_node_counts", np.full(net.n_neurons, np.nan)),
        }
    )


def run_equilibration(
    net: FAFBMCNetwork,
    t_ms: float = 100.0,
    dt: float = 0.1,
    synapse_scale: float = 1.0,
    trace_types: Optional[Sequence[str]] = None,
    progress_every: int = 0,
) -> Tuple[np.ndarray, pd.DataFrame]:
    trace_types = list(trace_types or ["R1-6", "L5"])
    steps = int(round(t_ms / dt))
    device = torch.device("cpu")
    net = net.to(device)
    net.eval()
    state = net.init_state(batch_size=1, device=device)
    net._cache_step_tensors(max_batch=1)
    original_syn = net._syn_gS_w.clone()
    try:
        net._syn_gS_w = net._syn_gS_w * float(synapse_scale)
        traces = {ct: [] for ct in trace_types}
        time_ms = []
        with torch.no_grad():
            for step in range(steps):
                if getattr(net, "ragged_mode", False):
                    I_ext = torch.zeros(1, net.total_comp, dtype=torch.float64, device=device)
                    state = net._step(state, I_ext, dt)
                    soma_v = state["V"][0, net.soma_comp_idx].detach().cpu().numpy()
                else:
                    I_ext = torch.zeros(1, net.n_neurons, net.n_comp, dtype=torch.float64, device=device)
                    state = net._step(state, I_ext, dt)
                    soma_v = state["V"][0, :, 0].detach().cpu().numpy()

                for ct in trace_types:
                    idx = type_indices(net, ct)
                    traces[ct].append(float(np.mean(soma_v[idx])) if len(idx) else np.nan)
                time_ms.append((step + 1) * dt)
                if progress_every and ((step + 1) % progress_every == 0 or step + 1 == steps):
                    print(f"{step + 1}/{steps} steps done (synapse_scale={synapse_scale})")
        return soma_v, pd.DataFrame({"time_ms": time_ms, **traces})
    finally:
        net._syn_gS_w = original_syn


def build_edge_audit_table(
    net: FAFBMCNetwork,
    baseline_soma_v: np.ndarray,
) -> pd.DataFrame:
    pre_idx = net.pre_idx.detach().cpu().numpy()
    post_idx = net.post_idx.detach().cpu().numpy()
    edge_df = pd.DataFrame(
        {
            "edge_idx": np.arange(net.n_edges),
            "pre_idx": pre_idx,
            "post_idx": post_idx,
            "pre_root_id": net.root_ids[pre_idx],
            "post_root_id": net.root_ids[post_idx],
            "pre_type": np.array(net.cell_types, dtype=object)[pre_idx],
            "post_type": np.array(net.cell_types, dtype=object)[post_idx],
            "pre_baseline_mV": baseline_soma_v[pre_idx],
            "post_baseline_mV": baseline_soma_v[post_idx],
            "syn_count": net.syn_count.detach().cpu().numpy() if net.syn_count is not None else np.ones(net.n_edges),
            "gS_mS_cm2": net.syn_gS.detach().cpu().numpy(),
            "e_syn_mV": net.syn_e_syn.detach().cpu().numpy(),
            "v_th_mV": net.syn_v_th.detach().cpu().numpy(),
            "k_minus": net.syn_k_minus.detach().cpu().numpy(),
            "delta_mV": net.syn_delta.detach().cpu().numpy(),
            "param_source": getattr(net, "synapse_param_source", [None] * net.n_edges),
            "fallback_target": getattr(net, "synapse_fallback_target", [None] * net.n_edges),
        }
    )
    edge_df["s_inf_baseline"] = 1.0 / (
        1.0 + np.exp((edge_df["v_th_mV"] - edge_df["pre_baseline_mV"]) / np.clip(edge_df["delta_mV"], 1e-3, None))
    )
    edge_df["tau_s_baseline_ms"] = (1.0 - edge_df["s_inf_baseline"]) / np.clip(edge_df["k_minus"], 1e-6, None)
    edge_df["baseline_drive_score"] = edge_df["gS_mS_cm2"] * edge_df["syn_count"] * edge_df["s_inf_baseline"]
    return edge_df


def run_flash_sanity(
    net: FAFBMCNetwork,
    *,
    amp: float,
    t_pre: float,
    t_on: float,
    t_off: float,
    stable_seconds: float,
    dt: float,
    progress_every: int,
    stop_on_nan: bool = True,
    r16_synapse_gain: float = 1.0,
) -> Tuple[torch.Tensor, Dict, Dict]:
    import time as _time

    requested_steps = int((t_pre + t_on + t_off) / dt)
    max_steps_by_time = max(1, int(stable_seconds / dt))
    T_steps = min(requested_steps, max_steps_by_time)
    t_on_s = min(int(t_pre / dt), T_steps)
    t_on_e = min(int((t_pre + t_on) / dt), T_steps)

    n_photo = int(net.input_mask.sum().item())
    photo_indices = torch.where(net.input_mask)[0]
    x_photo = torch.zeros(1, T_steps, n_photo)
    if t_on_e > t_on_s:
        x_photo[:, t_on_s:t_on_e, :] = amp

    x_full = torch.zeros(1, T_steps, net.n_neurons)
    x_full[:, :, photo_indices] = x_photo

    print(
        f"Flash sanity: amp={amp}, steps={T_steps}, dt={dt}, simulated_ms={T_steps * dt:.1f}, "
        f"n_photo={n_photo:,}, r16_synapse_gain={r16_synapse_gain}"
    )
    print(f"Network: neurons={net.n_neurons:,}, edges={net.n_edges:,}, total_comp={getattr(net, 'total_comp', net.n_comp):,}")

    traces = []
    t0 = _time.time()
    with torch.no_grad():
        net._cache_step_tensors(max_batch=1)
        original_syn_gS_w = net._syn_gS_w.clone()
        if r16_synapse_gain != 1.0:
            r16_mask = torch.tensor([ct in ("R1-6", "R1-R6") for ct in net.cell_types], device=net.pre_idx.device)
            r16_edge_mask = r16_mask[net.pre_idx]
            net._syn_gS_w[r16_edge_mask] = net._syn_gS_w[r16_edge_mask] * float(r16_synapse_gain)

        state = net.init_state(1, device=x_full.device)
        for t_i in range(T_steps):
            if getattr(net, "ragged_mode", False):
                I_ext = torch.zeros(1, net.total_comp, dtype=torch.float64, device=x_full.device)
                I_ext[:, net.soma_comp_idx] = x_full[:, t_i].double()
                state = net._step(state, I_ext, dt)
                soma_v = state["V"][:, net.soma_comp_idx].float()
            else:
                I_ext = torch.zeros(1, net.n_neurons, net.n_comp, dtype=torch.float64, device=x_full.device)
                I_ext[:, :, 0] = x_full[:, t_i].double()
                state = net._step(state, I_ext, dt)
                soma_v = state["V"][:, :, 0].float()

            traces.append(soma_v.cpu())
            if torch.isnan(soma_v).any():
                print(f"NaN detected at step {t_i + 1}; stopping early.")
                if stop_on_nan:
                    break

            if (t_i + 1) % max(progress_every, 1) == 0 or (t_i + 1) == T_steps:
                print(f"  step {t_i + 1:,}/{T_steps:,}, elapsed={_time.time() - t0:.1f}s")

        net._syn_gS_w = original_syn_gS_w

    V = torch.stack(traces, dim=1)
    summary = {
        "amp": amp,
        "steps_run": V.shape[1],
        "simulated_ms": V.shape[1] * dt,
        "output_shape": tuple(V.shape),
        "v_min": float(V.min().item()),
        "v_max": float(V.max().item()),
        "has_nan": bool(torch.isnan(V).any().item()),
        "elapsed_s": _time.time() - t0,
        "r16_synapse_gain": float(r16_synapse_gain),
    }
    meta = {
        "t_on_s": t_on_s,
        "t_on_e": t_on_e,
        "dt": dt,
        "amp": amp,
        "r16_synapse_gain": r16_synapse_gain,
    }
    return V, summary, meta


def summarize_flash_by_type(
    net: FAFBMCNetwork,
    V: torch.Tensor,
    meta: Dict,
    show_types: Iterable[str],
) -> pd.DataFrame:
    t_on_s = meta["t_on_s"]
    t_on_e = meta["t_on_e"]
    rows = []
    for ct in show_types:
        idx = net.get_indices_by_type(ct)
        if not idx:
            continue
        vm = V[0, :, idx].mean(dim=1)
        base = vm[:t_on_s].mean().item() if t_on_s > 0 else float("nan")
        on = vm[t_on_s:t_on_e].mean().item() if t_on_e > t_on_s else float("nan")
        off = vm[t_on_e:].mean().item() if t_on_e < V.shape[1] else float("nan")
        rows.append(
            {
                "type": ct,
                "n": len(idx),
                "baseline_mV": base,
                "on_mV": on,
                "off_mV": off,
                "delta_on_minus_base": on - base,
                "peak_max_mV": vm.max().item(),
                "peak_min_mV": vm.min().item(),
            }
        )
    return pd.DataFrame(rows)

