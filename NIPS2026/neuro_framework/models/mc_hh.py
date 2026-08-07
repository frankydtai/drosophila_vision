"""
Multi-Compartment Hodgkin-Huxley Models
========================================
Biophysical neuron models loaded from SWC morphology files.

Classes
-------
MultiCompartmentHH
    Standalone single neuron with branched cable equation + HH channels.
    Loaded from an SWC file; uses semi-implicit Euler (implicit axial,
    explicit channels) for unconditional stability.

MCNetwork
    Network of multi-compartment HH neurons connected by synapses.
    Compatible with ``TorchTrainer`` — same ``(B, T, N)`` I/O interface.

Solver strategy
---------------
Small morphologies (< ``_DENSE_THRESHOLD`` compartments): dense
``torch.linalg.solve`` — exact, fast for small systems.

Large morphologies (≥ ``_DENSE_THRESHOLD``): sparse Jacobi iteration —
O(n_edges) per iteration, memory-efficient, differentiable.

Unit conventions (Jaxley-compatible)
------------------------------------
  V : mV,  t : ms,  g : mS/cm²,  C_m : μF/cm²,
  I : μA/cm²,  R_a : Ω·cm,  area : μm²,  length : μm
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from .morphology import MorphologyGraph
from .synapses import (
    BaseSynapse,
    TanhRateSynapse,
    TanhConductanceSynapse,
    IonotropicSynapse,
    NMDASynapse,
    GABAaSynapse,
)

__all__ = ["MultiCompartmentHH", "MCNetwork"]

_DENSE_THRESHOLD = 500


# ===================================================================
# Numerically stable HH gating helpers
# ===================================================================

def _safe_x_over_1_minus_exp_neg_x(x: Tensor) -> Tensor:
    """x / (1 - exp(-x)), with L'Hôpital limit = 1 near x=0."""
    return torch.where(
        torch.abs(x) < 1e-6,
        torch.ones_like(x),
        x / (1.0 - torch.exp(-x)),
    )


def _alpha_m(V: Tensor) -> Tensor:
    return 0.1 * 10.0 * _safe_x_over_1_minus_exp_neg_x((V + 40.0) / 10.0)

def _beta_m(V: Tensor) -> Tensor:
    return 4.0 * torch.exp(-(V + 65.0) / 18.0)

def _alpha_h(V: Tensor) -> Tensor:
    return 0.07 * torch.exp(-(V + 65.0) / 20.0)

def _beta_h(V: Tensor) -> Tensor:
    return 1.0 / (1.0 + torch.exp(-(V + 35.0) / 10.0))

def _alpha_n(V: Tensor) -> Tensor:
    return 0.01 * 10.0 * _safe_x_over_1_minus_exp_neg_x((V + 55.0) / 10.0)

def _beta_n(V: Tensor) -> Tensor:
    return 0.125 * torch.exp(-(V + 65.0) / 80.0)

def _x_inf(alpha_fn, beta_fn, V: Tensor) -> Tensor:
    a = alpha_fn(V)
    return a / (a + beta_fn(V))


# ===================================================================
# Sparse implicit solver
# ===================================================================

def _solve_implicit(
    G_row: Tensor,
    G_col: Tensor,
    G_val: Tensor,
    R_a: Tensor,
    C_m: Tensor,
    areas: Tensor,
    V_flat: Tensor,
    vt_flat: Tensor,
    ct_flat: Tensor,
    dt: float,
    n_jacobi_iter: int = 20,
) -> Tensor:
    """Solve the semi-implicit voltage update for all neurons.

    Builds the system  ``A x = b``  where
    ``A = I + dt·diag(vt) − dt·G``  and  ``b = V + dt·ct``,
    then solves using either dense LU (small C) or sparse Jacobi.

    Parameters
    ----------
    G_row, G_col, G_val : sparse COO description of the *geometric*
        axial conductance (before dividing by R_a).
    R_a, C_m : trainable scalars.
    areas : (C,) compartment surface areas.
    V_flat, vt_flat, ct_flat : (BN, C) state tensors.
    dt : time-step (ms).
    n_jacobi_iter : iterations for the sparse path.

    Returns
    -------
    V_new : (BN, C)
    """
    C = V_flat.shape[1]
    device = V_flat.device
    dtype = V_flat.dtype

    # Axial conductance scaling: g_ax = G_val / R_a
    # Per-compartment normalisation: norm = 1e7 / (C_m * areas)
    g_ax = G_val / R_a
    norm = 1e7 / (C_m * areas)

    # Off-diagonal entries of A:  a_ij = -dt * g_ax[k] * norm[i]
    #   (k is the edge index where row=i, col=j)
    off_vals = -dt * g_ax * norm[G_row]  # (K,)

    # Diagonal correction from coupling:  diag_corr[i] = +sum_k(-off_vals[k])
    diag_corr = torch.zeros(C, dtype=dtype, device=device)
    diag_corr.scatter_add_(0, G_row, -off_vals)

    # Full diagonal:  a_ii = 1 + dt * vt[i] + diag_corr[i]
    a_diag = 1.0 + dt * vt_flat + diag_corr.unsqueeze(0)  # (BN, C)

    rhs = V_flat + dt * ct_flat  # (BN, C)

    if C <= _DENSE_THRESHOLD:
        # ---------- dense path ----------
        A = torch.zeros(C, C, dtype=dtype, device=device)
        A.index_put_((G_row, G_col), off_vals, accumulate=True)
        BN = V_flat.shape[0]
        A_batch = A.unsqueeze(0).expand(BN, -1, -1).clone()
        cidx = torch.arange(C, device=device)
        A_batch[:, cidx, cidx] = a_diag
        return torch.linalg.solve(A_batch, rhs.unsqueeze(-1)).squeeze(-1)

    # ---------- sparse Jacobi path ----------
    off_sparse = torch.sparse_coo_tensor(
        torch.stack([G_row, G_col]), off_vals, size=(C, C),
    ).coalesce()

    d_inv = 1.0 / a_diag  # (BN, C)
    x = rhs * d_inv  # initial guess

    for _ in range(n_jacobi_iter):
        # sparse (C,C) @ dense (C,BN) → (C,BN) → transpose to (BN,C)
        off_x = torch.sparse.mm(off_sparse, x.t()).t()
        x = d_inv * (rhs - off_x)

    return x


# ===================================================================
# MultiCompartmentHH  — single neuron
# ===================================================================

class MultiCompartmentHH(nn.Module):
    """
    Multi-compartment Hodgkin-Huxley neuron loaded from an SWC file.

    Solves the branched cable equation with semi-implicit Euler:
      C_m dV_i/dt = −I_ion,i + (1/A_i) Σ_j G_ij(V_j − V_i) + I_ext,i

    Parameters
    ----------
    morph : MorphologyGraph
        Compartmental morphology built from an SWC file.

    Example
    -------
    >>> model = MultiCompartmentHH.from_swc("neuron.swc", ncomp=1)
    >>> V_trace = model.simulate(I_ext, dt=0.025)
    """

    def __init__(self, morph: MorphologyGraph):
        super().__init__()
        self.n_comp = morph.n_comp
        self.n_branches = morph.n_branches

        self.register_buffer("areas", torch.tensor(morph.areas, dtype=torch.float64))

        G_sparse = self._build_geom_matrix(morph)
        self.register_buffer("G_row", G_sparse[0])
        self.register_buffer("G_col", G_sparse[1])
        self.register_buffer("G_val", G_sparse[2])

        self.log_g_Na = nn.Parameter(torch.tensor(math.log(120.0)))
        self.log_g_K = nn.Parameter(torch.tensor(math.log(36.0)))
        self.log_g_L = nn.Parameter(torch.tensor(math.log(0.3)))
        self.E_Na = nn.Parameter(torch.tensor(50.0))
        self.E_K = nn.Parameter(torch.tensor(-77.0))
        self.E_L = nn.Parameter(torch.tensor(-54.4))
        self.log_C_m = nn.Parameter(torch.tensor(math.log(1.0)))
        self.log_R_a = nn.Parameter(torch.tensor(math.log(100.0)))

    @classmethod
    def from_swc(cls, swc_path: str, ncomp: int = 1, **kw) -> "MultiCompartmentHH":
        return cls(MorphologyGraph.from_swc(swc_path, ncomp=ncomp, **kw))

    # -- properties -------------------------------------------------------
    @property
    def g_Na(self) -> Tensor: return torch.exp(self.log_g_Na)
    @property
    def g_K(self) -> Tensor:  return torch.exp(self.log_g_K)
    @property
    def g_L(self) -> Tensor:  return torch.exp(self.log_g_L)
    @property
    def C_m(self) -> Tensor:  return torch.exp(self.log_C_m)
    @property
    def R_a(self) -> Tensor:  return torch.exp(self.log_R_a)

    # -- geometry ---------------------------------------------------------
    @staticmethod
    def _build_geom_matrix(morph: MorphologyGraph):
        rows, cols, vals = [], [], []
        for src, snk in morph.comp_edges:
            r_load = morph.resistive_load_out[src] + morph.resistive_load_in[snk]
            r_load = max(r_load, 1e-30)
            rows.append(snk)
            cols.append(src)
            vals.append(1.0 / r_load)
        return (
            torch.tensor(rows, dtype=torch.long),
            torch.tensor(cols, dtype=torch.long),
            torch.tensor(vals, dtype=torch.float64),
        )

    # -- state ------------------------------------------------------------
    def init_state(self, batch_size: int = 1, device: torch.device = torch.device("cpu")) -> Dict[str, Tensor]:
        V0 = torch.full((batch_size, self.n_comp), -65.0, dtype=torch.float64, device=device)
        return {
            "V": V0,
            "m": _x_inf(_alpha_m, _beta_m, V0),
            "h": _x_inf(_alpha_h, _beta_h, V0),
            "n": _x_inf(_alpha_n, _beta_n, V0),
        }

    # -- step (semi-implicit) ---------------------------------------------
    def step(self, state: Dict[str, Tensor], I_ext: Tensor, dt: float = 0.025) -> Dict[str, Tensor]:
        V, m, h, n = state["V"], state["m"], state["h"], state["n"]

        g_tot = self.g_Na * m**3 * h + self.g_K * n**4 + self.g_L
        vt = g_tot / self.C_m
        ct = (self.g_Na * m**3 * h * self.E_Na
              + self.g_K * n**4 * self.E_K
              + self.g_L * self.E_L + I_ext) / self.C_m

        V_new = _solve_implicit(
            self.G_row, self.G_col, self.G_val,
            self.R_a, self.C_m, self.areas,
            V, vt, ct, dt,
        )

        am, bm = _alpha_m(V), _beta_m(V)
        ah, bh = _alpha_h(V), _beta_h(V)
        an, bn = _alpha_n(V), _beta_n(V)
        return {
            "V": V_new,
            "m": torch.clamp(m + dt * (am * (1 - m) - bm * m), 0, 1),
            "h": torch.clamp(h + dt * (ah * (1 - h) - bh * h), 0, 1),
            "n": torch.clamp(n + dt * (an * (1 - n) - bn * n), 0, 1),
        }

    # -- simulate ---------------------------------------------------------
    def simulate(self, I_ext: Tensor, dt: float = 0.025,
                 state: Optional[Dict[str, Tensor]] = None,
                 record_comp_indices: Optional[List[int]] = None) -> Tensor:
        if I_ext.dim() == 2:
            I_ext = I_ext.unsqueeze(0)
        B, T, N = I_ext.shape
        assert N == self.n_comp
        if state is None:
            state = self.init_state(B, device=I_ext.device)
        rec = torch.tensor(record_comp_indices or list(range(N)),
                           dtype=torch.long, device=I_ext.device)
        traces = []
        for t in range(T):
            state = self.step(state, I_ext[:, t, :], dt)
            traces.append(state["V"][:, rec])
        return torch.stack(traces, dim=1)

    def forward(self, I_ext: Tensor, dt: float = 0.025, **kw) -> Tensor:
        return self.simulate(I_ext, dt=dt, **kw)

    def extra_repr(self) -> str:
        return f"n_comp={self.n_comp}, n_branches={self.n_branches}"


# ===================================================================
# MCNetwork  — network of multi-compartment neurons
# ===================================================================

_SYNAPSE_REGISTRY = {
    "tanh_rate": TanhRateSynapse,
    "tanh_conductance": TanhConductanceSynapse,
    "ionotropic": IonotropicSynapse,
    "nmda": NMDASynapse,
    "gaba_a": GABAaSynapse,
}


class MCNetwork(nn.Module):
    """
    Network of multi-compartment HH neurons connected by synapses.

    All neurons share the same morphology (loaded from a single SWC file).
    Synapses connect at user-specified compartments (default: soma, index 0).

    The ``forward()`` method returns ``(B, T, N_neurons)`` soma voltages,
    making it a drop-in replacement for ``ConnectomeNetwork`` in the
    ``TorchTrainer`` pipeline.

    Parameters
    ----------
    morph : MorphologyGraph
        Shared compartmental morphology.
    n_neurons : int
        Number of neurons in the network.
    pre_neuron_idx, post_neuron_idx : LongTensor (n_synapses,)
        Pre-/post-synaptic neuron indices.
    syn_count : FloatTensor (n_synapses,), optional
        Synapse counts for weight initialisation.
    nt_sign : FloatTensor (n_neurons,), optional
        Neurotransmitter sign per neuron (+1 excit, −1 inhib).
    pre_comp_idx, post_comp_idx : LongTensor (n_synapses,), optional
        Compartment indices for each synapse endpoint (default: 0 = soma).
    synapse_model : str or BaseSynapse
        ``'simple'`` (weight-only), ``'ionotropic'``, ``'tanh_rate'``,
        ``'tanh_conductance'``, ``'nmda'``, ``'gaba_a'``, or a
        ``BaseSynapse`` instance.
    dt : float
        Default time-step (ms).
    soma_idx : int
        Index of the soma compartment (default: 0).
    """

    def __init__(
        self,
        morph: MorphologyGraph,
        n_neurons: int,
        pre_neuron_idx: Tensor,
        post_neuron_idx: Tensor,
        syn_count: Optional[Tensor] = None,
        nt_sign: Optional[Tensor] = None,
        pre_comp_idx: Optional[Tensor] = None,
        post_comp_idx: Optional[Tensor] = None,
        synapse_model: Union[str, BaseSynapse] = "ionotropic",
        dt: float = 0.025,
        learn_weights: bool = True,
        learn_hh_params: bool = True,
        init_weight_scale: float = 0.01,
        soma_idx: int = 0,
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.n_comp = morph.n_comp
        n_edges = len(pre_neuron_idx)
        self.n_edges = n_edges
        self.dt = dt
        self.soma_idx = soma_idx

        # -- Morphology (shared) ----------------------------------------
        self.register_buffer("areas", torch.tensor(morph.areas, dtype=torch.float64))

        G_sparse = MultiCompartmentHH._build_geom_matrix(morph)
        self.register_buffer("G_row", G_sparse[0])
        self.register_buffer("G_col", G_sparse[1])
        self.register_buffer("G_val", G_sparse[2])

        # -- Connectivity -----------------------------------------------
        self.register_buffer("pre_neuron_idx", pre_neuron_idx.long())
        self.register_buffer("post_neuron_idx", post_neuron_idx.long())
        self.register_buffer(
            "pre_comp_idx",
            pre_comp_idx.long() if pre_comp_idx is not None
            else torch.zeros(n_edges, dtype=torch.long),
        )
        self.register_buffer(
            "post_comp_idx",
            post_comp_idx.long() if post_comp_idx is not None
            else torch.zeros(n_edges, dtype=torch.long),
        )

        if nt_sign is None:
            nt_sign = torch.ones(n_neurons)
        self.register_buffer("nt_sign", nt_sign.float())

        # -- Synaptic weights -------------------------------------------
        if syn_count is None:
            syn_count = torch.ones(n_edges)
        syn_count = syn_count.float()
        max_syn = syn_count.max().clamp(min=1.0)
        w_init = init_weight_scale * syn_count / max_syn
        self.log_weight_abs = nn.Parameter(
            torch.log(w_init.abs().clamp(min=1e-6)),
            requires_grad=learn_weights,
        )

        # -- HH parameters (shared, trainable) --------------------------
        self.log_g_Na = nn.Parameter(torch.tensor(math.log(120.0)), requires_grad=learn_hh_params)
        self.log_g_K = nn.Parameter(torch.tensor(math.log(36.0)), requires_grad=learn_hh_params)
        self.log_g_L = nn.Parameter(torch.tensor(math.log(0.3)), requires_grad=learn_hh_params)
        self.E_Na = nn.Parameter(torch.tensor(50.0), requires_grad=learn_hh_params)
        self.E_K = nn.Parameter(torch.tensor(-77.0), requires_grad=learn_hh_params)
        self.E_L = nn.Parameter(torch.tensor(-54.4), requires_grad=learn_hh_params)
        self.log_C_m = nn.Parameter(torch.tensor(math.log(1.0)), requires_grad=learn_hh_params)
        self.log_R_a = nn.Parameter(torch.tensor(math.log(100.0)), requires_grad=learn_hh_params)

        # -- Synapse model -----------------------------------------------
        self._synapse_model: Optional[BaseSynapse] = self._make_synapse(
            synapse_model, n_edges, learn_weights,
        )

    # -- properties -------------------------------------------------------
    @property
    def g_Na(self) -> Tensor: return torch.exp(self.log_g_Na)
    @property
    def g_K(self) -> Tensor: return torch.exp(self.log_g_K)
    @property
    def g_L(self) -> Tensor: return torch.exp(self.log_g_L)
    @property
    def C_m(self) -> Tensor: return torch.exp(self.log_C_m)
    @property
    def R_a(self) -> Tensor: return torch.exp(self.log_R_a)

    @property
    def weight(self) -> Tensor:
        """Synaptic weight per edge.

        For conductance-based synapse models the weight is always positive
        (excitation vs inhibition is determined by E_syn).  For the
        ``'simple'`` coupling mode ``nt_sign`` is applied.
        """
        w_abs = torch.exp(self.log_weight_abs)
        if self._synapse_model is None:
            return self.nt_sign[self.pre_neuron_idx] * w_abs
        return w_abs

    # -- factories --------------------------------------------------------
    @classmethod
    def from_swc(cls, swc_path: str, n_neurons: int,
                 pre_neuron_idx: Tensor, post_neuron_idx: Tensor,
                 ncomp: int = 1, **kwargs) -> "MCNetwork":
        """Build network from an SWC file and neuron-level connectivity."""
        morph = MorphologyGraph.from_swc(swc_path, ncomp=ncomp)
        return cls(morph=morph, n_neurons=n_neurons,
                   pre_neuron_idx=pre_neuron_idx,
                   post_neuron_idx=post_neuron_idx, **kwargs)

    @classmethod
    def from_loader(cls, loader, swc_path: str, ncomp: int = 4,
                    **kwargs) -> "MCNetwork":
        """Build from a ``ConnectomeLoader`` and shared SWC morphology."""
        nodes, edges = loader.load()
        pre_idx = torch.tensor(edges["pre_idx"].to_numpy(), dtype=torch.long)
        post_idx = torch.tensor(edges["post_idx"].to_numpy(), dtype=torch.long)
        syn_count = torch.tensor(edges["syn_count"].to_numpy(), dtype=torch.float)
        nt_sign = torch.tensor(loader.nt_sign(), dtype=torch.float)
        morph = MorphologyGraph.from_swc(swc_path, ncomp=ncomp)
        return cls(morph=morph, n_neurons=len(nodes),
                   pre_neuron_idx=pre_idx, post_neuron_idx=post_idx,
                   syn_count=syn_count, nt_sign=nt_sign, **kwargs)

    # -- synapse builder --------------------------------------------------
    @staticmethod
    def _make_synapse(spec, n_edges, learn):
        if spec is None or spec == "simple":
            return None
        if isinstance(spec, BaseSynapse):
            return spec
        name = spec.lower()
        if name not in _SYNAPSE_REGISTRY:
            raise ValueError(
                f"Unknown synapse '{name}'. "
                f"Choose from: {list(_SYNAPSE_REGISTRY)} or 'simple'."
            )
        return _SYNAPSE_REGISTRY[name](n_edges, learn_params=learn)

    # -- state initialisation ---------------------------------------------
    def init_state(self, batch_size: int = 1,
                   device: torch.device = torch.device("cpu")) -> Dict[str, Tensor]:
        V0 = torch.full(
            (batch_size, self.n_neurons, self.n_comp),
            -65.0, dtype=torch.float64, device=device,
        )
        return {
            "V": V0,
            "m": _x_inf(_alpha_m, _beta_m, V0),
            "h": _x_inf(_alpha_h, _beta_h, V0),
            "n": _x_inf(_alpha_n, _beta_n, V0),
        }

    def _init_syn_states(self, batch_size, device):
        if self._synapse_model is not None and hasattr(self._synapse_model, "init_states"):
            return self._synapse_model.init_states(batch_size, device)
        return None

    # -- single step ------------------------------------------------------
    def _step(self, state, I_ext, dt, syn_states):
        V, m, h, n = state["V"], state["m"], state["h"], state["n"]
        B, N, C = V.shape

        # 1. Membrane terms
        g_tot = self.g_Na * m**3 * h + self.g_K * n**4 + self.g_L
        vt = g_tot / self.C_m
        ct = (
            self.g_Na * m**3 * h * self.E_Na
            + self.g_K * n**4 * self.E_K
            + self.g_L * self.E_L
            + I_ext
        ) / self.C_m

        # 2. Synaptic current
        if self.n_edges > 0:
            pre_v = V[:, self.pre_neuron_idx, self.pre_comp_idx]
            post_v = V[:, self.post_neuron_idx, self.post_comp_idx]
            w = self.weight.unsqueeze(0).expand(B, -1).double()

            if self._synapse_model is not None:
                if syn_states is not None:
                    syn_states = self._synapse_model.update_states(
                        syn_states, dt, pre_v.float(), post_v.float())
                syn_cur = self._synapse_model.compute_current(
                    syn_states or {}, pre_v.float(), post_v.float(),
                ).double()
                weighted = -w * syn_cur / self.C_m
            else:
                weighted = w * pre_v / self.C_m

            flat_target = self.post_neuron_idx * C + self.post_comp_idx
            syn_contrib = torch.zeros(B, N * C, dtype=torch.float64, device=V.device)
            syn_contrib.scatter_add_(1, flat_target.unsqueeze(0).expand(B, -1), weighted)
            ct = ct + syn_contrib.view(B, N, C)

        # 3. Implicit solve per neuron (memory-efficient)
        V_flat = V.reshape(B * N, C)
        vt_flat = vt.reshape(B * N, C)
        ct_flat = ct.reshape(B * N, C)

        V_new = _solve_implicit(
            self.G_row, self.G_col, self.G_val,
            self.R_a, self.C_m, self.areas,
            V_flat, vt_flat, ct_flat, dt,
        ).reshape(B, N, C)

        # 4. Gating variables (explicit Euler)
        am, bm = _alpha_m(V), _beta_m(V)
        ah, bh = _alpha_h(V), _beta_h(V)
        an, bn = _alpha_n(V), _beta_n(V)

        new_state = {
            "V": V_new,
            "m": torch.clamp(m + dt * (am * (1 - m) - bm * m), 0, 1),
            "h": torch.clamp(h + dt * (ah * (1 - h) - bh * h), 0, 1),
            "n": torch.clamp(n + dt * (an * (1 - n) - bn * n), 0, 1),
        }
        return new_state, syn_states

    # -- forward / simulate -----------------------------------------------
    def forward(self, x, dt=None, state=None,
                return_all_compartments=False):
        """
        Integrate network dynamics over a stimulus sequence.

        Parameters
        ----------
        x : (B, T, N_neurons)  — stimulus current injected into soma, **or**
            (B, T, N_neurons, C) — per-compartment stimulus.  μA/cm².
        dt : float, optional
        state : dict, optional
        return_all_compartments : bool
            If True, also return ``(B, T, N, C)`` full voltage tensor.

        Returns
        -------
        soma_v : (B, T, N_neurons)
        all_v  : (B, T, N, C) — only when *return_all_compartments=True*
        """
        dt = dt or self.dt
        B, T = x.shape[0], x.shape[1]
        N, C = self.n_neurons, self.n_comp
        per_comp = x.dim() == 4
        if not per_comp:
            assert x.shape[2] == N

        if state is None:
            state = self.init_state(B, device=x.device)
        syn_states = self._init_syn_states(B, x.device)

        soma_traces: List[Tensor] = []
        all_traces: Optional[List[Tensor]] = [] if return_all_compartments else None

        for t in range(T):
            if per_comp:
                I_ext_t = x[:, t].double()
            else:
                I_ext_t = torch.zeros(B, N, C, dtype=torch.float64, device=x.device)
                I_ext_t[:, :, self.soma_idx] = x[:, t].double()

            state, syn_states = self._step(state, I_ext_t, dt, syn_states)
            soma_traces.append(state["V"][:, :, self.soma_idx].float())
            if all_traces is not None:
                all_traces.append(state["V"].float())

        soma = torch.stack(soma_traces, dim=1)
        if return_all_compartments:
            return soma, torch.stack(all_traces, dim=1)
        return soma

    def simulate(self, x, dt=None, **kw):
        """Alias for ``forward``."""
        return self.forward(x, dt=dt, **kw)

    # -- utilities --------------------------------------------------------
    def n_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self):
        syn_name = (
            self._synapse_model.__class__.__name__
            if self._synapse_model is not None else "simple"
        )
        solver = "dense" if self.n_comp <= _DENSE_THRESHOLD else "sparse-jacobi"
        return (
            f"n_neurons={self.n_neurons}, n_comp={self.n_comp}, "
            f"n_edges={self.n_edges}, synapse={syn_name}, "
            f"solver={solver}, trainable={self.n_parameters()}"
        )
