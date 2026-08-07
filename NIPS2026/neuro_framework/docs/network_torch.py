"""
PyTorch Connectome-Constrained Network
======================================
Builds a spiking / rate-coded recurrent network from a loaded connectome
(BANC or FAFB) and integrates it forward using one of the dynamics models
defined in `models/dynamics.py`.

Design mirrors the FlyVis `Network` but is self-contained and backend-agnostic
enough to be used without the FlyVis data pipeline.

Quick start
-----------
    from neuro_framework.connectome.loader import ConnectomeLoader
    from neuro_framework.models.network_torch import ConnectomeNetwork

    loader = ConnectomeLoader.from_banc(cell_types=['T4a','T4b','LC4'])
    net = ConnectomeNetwork.from_loader(loader, dynamics='lif')
    # x_t: (batch, T, n_nodes)  external stimulus
    activity = net.simulate(x_t, dt=1.0)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .dynamics import BaseDynamics, build_dynamics

logger = logging.getLogger(__name__)

__all__ = ["ConnectomeNetwork"]


class ConnectomeNetwork(nn.Module):
    """
    Connectome-constrained recurrent neural network (PyTorch).

    Parameters
    ----------
    n_nodes : int
        Total number of neurons.
    pre_idx : LongTensor  (n_edges,)
        Pre-synaptic neuron indices.
    post_idx : LongTensor  (n_edges,)
        Post-synaptic neuron indices.
    syn_count : FloatTensor  (n_edges,)
        Synapse counts (used to initialise weights).
    nt_sign : FloatTensor  (n_nodes,)
        Neurotransmitter sign per *pre* neuron (+1 excit, -1 inhib, 0 unk).
    dynamics : str or BaseDynamics
        Neuron dynamics model (``'voltage'``, ``'lif'``, ``'hh'``).
    dt : float
        Default integration time-step in ms.
    learn_weights : bool
        Whether synapse weights are trainable. Default True.
    learn_time_const : bool
        Whether per-neuron time constants are trainable. Default True.
    init_weight_scale : float
        Scaling factor applied to syn_count at initialisation.
    """

    def __init__(
        self,
        n_nodes: int,
        pre_idx: Tensor,
        post_idx: Tensor,
        syn_count: Tensor,
        nt_sign: Optional[Tensor] = None,
        dynamics: str | BaseDynamics = "voltage",
        dt: float = 1.0,
        learn_weights: bool = True,
        learn_time_const: bool = True,
        init_weight_scale: float = 0.01,
    ):
        super().__init__()
        self.n_nodes   = n_nodes
        self.n_edges   = len(pre_idx)
        self.dt        = dt

        # Register topology buffers (not parameters)
        self.register_buffer("pre_idx",   pre_idx.long())
        self.register_buffer("post_idx",  post_idx.long())
        self.register_buffer("syn_count", syn_count.float())

        if nt_sign is None:
            nt_sign = torch.ones(n_nodes)
        self.register_buffer("nt_sign", nt_sign.float())

        # --- Learnable parameters -----------------------------------------
        # Synaptic weight:  sign * |weight| * syn_count / max_syn
        max_syn = syn_count.max() if syn_count.numel() > 0 else torch.tensor(1.0)
        w_init = init_weight_scale * syn_count / (max_syn + 1e-8)
        self.log_weight_abs = nn.Parameter(
            torch.log(w_init.abs().clamp(min=1e-6)), requires_grad=learn_weights
        )

        # Per-neuron time constant (in ms), parameterised in log-space
        self.log_tau = nn.Parameter(
            torch.zeros(n_nodes), requires_grad=learn_time_const
        )  # exp(0) = 1 ms; override after construction if needed

        # Per-neuron bias (resting potential offset)
        self.bias = nn.Parameter(torch.zeros(n_nodes))

        # Dynamics model
        self.dynamics: BaseDynamics = (
            dynamics if isinstance(dynamics, BaseDynamics) else build_dynamics(dynamics)
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def from_loader(
        cls,
        loader,  # ConnectomeLoader
        dynamics: str | BaseDynamics = "voltage",
        **kwargs,
    ) -> "ConnectomeNetwork":
        """Build a network directly from a ConnectomeLoader instance."""
        nodes, edges = loader.load()
        pre_idx   = torch.tensor(edges["pre_idx"].to_numpy(),  dtype=torch.long)
        post_idx  = torch.tensor(edges["post_idx"].to_numpy(), dtype=torch.long)
        syn_count = torch.tensor(edges["syn_count"].to_numpy(), dtype=torch.float)
        nt_sign   = torch.tensor(loader.nt_sign(), dtype=torch.float)
        return cls(
            n_nodes=len(nodes),
            pre_idx=pre_idx,
            post_idx=post_idx,
            syn_count=syn_count,
            nt_sign=nt_sign,
            dynamics=dynamics,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Parameter accessors
    # ------------------------------------------------------------------
    @property
    def weight(self) -> Tensor:
        """Signed synaptic weight per edge, shape (n_edges,)."""
        sign = self.nt_sign[self.pre_idx]        # sign of pre-synaptic neuron
        return sign * torch.exp(self.log_weight_abs)

    @property
    def time_const(self) -> Tensor:
        """Per-neuron time constant (ms), shape (n_nodes,)."""
        return torch.exp(self.log_tau)

    def _params_dict(self) -> Dict[str, Tensor]:
        return {
            "weight":     self.weight,          # (n_edges,)
            "time_const": self.time_const,      # (n_nodes,)
            "bias":       self.bias,            # (n_nodes,)
        }

    # ------------------------------------------------------------------
    # Target-sum helper  (scatter-add over post-synaptic indices)
    # ------------------------------------------------------------------
    def _target_sum(self, edge_values: Tensor) -> Tensor:
        """
        Aggregate per-edge values onto post-synaptic nodes.

        Parameters
        ----------
        edge_values : Tensor  (batch, n_edges)  or  (n_edges,)

        Returns
        -------
        Tensor  (batch, n_nodes)  or  (n_nodes,)
        """
        if edge_values.dim() == 1:
            out = torch.zeros(self.n_nodes, device=edge_values.device)
            out.scatter_add_(0, self.post_idx, edge_values)
        else:
            B = edge_values.shape[0]
            out = torch.zeros(B, self.n_nodes, device=edge_values.device)
            idx = self.post_idx.unsqueeze(0).expand(B, -1)  # (B, E)
            out.scatter_add_(1, idx, edge_values)
        return out

    # ------------------------------------------------------------------
    # Forward / simulate
    # ------------------------------------------------------------------
    def forward(
        self,
        x: Tensor,
        dt: Optional[float] = None,
        state: Optional[Dict[str, Tensor]] = None,
        return_states: bool = False,
    ) -> Tensor | Tuple[Tensor, List[Dict[str, Tensor]]]:
        """
        Integrate network dynamics over a stimulus sequence.

        Parameters
        ----------
        x : Tensor  (batch, T, n_nodes)
            External stimulus current / input per node per timestep.
            Nodes not receiving direct input should be zero.
        dt : float, optional
            Integration step (ms). Defaults to ``self.dt``.
        state : dict, optional
            Initial state; created from zeros if not provided.
        return_states : bool
            If True, return all intermediate states as a list.

        Returns
        -------
        activity : Tensor  (batch, T, n_nodes)
            Primary state variable over time (voltage / activity / V_m).
        states : list of dict  (only when return_states=True)
        """
        dt    = dt or self.dt
        B, T, N = x.shape
        assert N == self.n_nodes, f"Stimulus has {N} nodes but network has {self.n_nodes}"

        params = self._params_dict()
        if state is None:
            state = self.dynamics.write_initial_state(B, N, dt, device=x.device)

        activities = []
        all_states  = [] if return_states else None

        for t in range(T):
            x_t = x[:, t, :]  # (B, N)
            # Broadcast edge weights over batch
            w   = self.weight.unsqueeze(0).expand(B, -1)  # (B, E)

            def target_sum(ev):
                return self._target_sum(ev)

            # Build per-step params with batch-broadcasted weight
            step_params = dict(params)
            step_params["weight"] = w

            state = self.dynamics.step(state, step_params, x_t, dt, target_sum)

            # Primary observable: first state variable
            primary_key = next(iter(state))
            activities.append(state[primary_key])   # (B, N)
            if return_states:
                all_states.append({k: v.detach() for k, v in state.items()})

        activity = torch.stack(activities, dim=1)   # (B, T, N)
        if return_states:
            return activity, all_states
        return activity

    def simulate(
        self,
        x: Tensor,
        dt: Optional[float] = None,
        **kwargs,
    ) -> Tensor:
        """Alias for forward(); returns (batch, T, n_nodes) activity tensor."""
        return self.forward(x, dt=dt, **kwargs)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def n_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def clamp_weights_(self, min_abs: float = 0.0) -> None:
        """In-place clamp on log-weight so |w| >= min_abs."""
        with torch.no_grad():
            self.log_weight_abs.clamp_(min=np.log(min_abs + 1e-8))

    def extra_repr(self) -> str:
        return (
            f"n_nodes={self.n_nodes}, n_edges={self.n_edges}, "
            f"dynamics={self.dynamics.__class__.__name__}, "
            f"trainable_params={self.n_parameters()}"
        )

