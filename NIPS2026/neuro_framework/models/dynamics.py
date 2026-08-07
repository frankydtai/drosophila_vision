"""
Neuron & Synapse Dynamics
=========================
Three neuron models with a common interface:

  VoltageModel   – DMN/FlyVis-style leaky-integrator (voltage activity, torch)
  LIFModel       – Leaky Integrate-and-Fire (torch)
  HHModel        – Hodgkin-Huxley 4-state ODE (torch)

All models expose:
  write_initial_state(batch, n_nodes, dt)  -> dict[str, Tensor]
  state_velocity(state, params, x_t, dt, target_sum)  -> dict[str, Tensor]

This mirrors the flyvis NetworkDynamics interface so the same Network wrapper
can swap dynamics at construction time.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

__all__ = [
    "BaseDynamics",
    "VoltageModel",
    "LIFModel",
    "HHModel",
    "DYNAMICS_REGISTRY",
    "build_dynamics",
]

_ACTIVATIONS = {
    "relu":       nn.ReLU,
    "elu":        nn.ELU,
    "softplus":   nn.Softplus,
    "leakyrelu":  nn.LeakyReLU,
    "sigmoid":    nn.Sigmoid,
    "tanh":       nn.Tanh,
}


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class BaseDynamics:
    """Abstract base class for neuron dynamics models."""

    def write_initial_state(
        self,
        batch_size: int,
        n_nodes: int,
        dt: float,
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, Tensor]:
        """Return a dict of zero-initialised state tensors."""
        raise NotImplementedError

    def state_velocity(
        self,
        state: Dict[str, Tensor],
        params: Dict[str, Tensor],
        x_t: Tensor,
        dt: float,
        target_sum: Callable[[Tensor], Tensor],
    ) -> Dict[str, Tensor]:
        """Return velocity dict matching the keys returned by write_initial_state."""
        raise NotImplementedError

    def step(
        self,
        state: Dict[str, Tensor],
        params: Dict[str, Tensor],
        x_t: Tensor,
        dt: float,
        target_sum: Callable[[Tensor], Tensor],
    ) -> Dict[str, Tensor]:
        """Euler integration: state + dt * velocity."""
        vel = self.state_velocity(state, params, x_t, dt, target_sum)
        return {k: state[k] + dt * vel[k] for k in vel}


# ---------------------------------------------------------------------------
# 1. Voltage / DMN-style model  (matches flyvis PPNeuronIGRSynapses)
# ---------------------------------------------------------------------------

class VoltageModel(BaseDynamics):
    """
    Leaky-integrator voltage model (DMN / FlyVis style).

    State  : activity  (batch, n_nodes)
    Params : time_const, bias  per node
             weight            per edge

    ODE:
        d(activity)/dt = (1/tau) * (-activity + bias
                          + sum_j(w_ij * activation(activity_j)) + x_t)

    Reference: Lappalainen et al. Nature 2024 (FlyVis / DMN).
    """

    def __init__(self, activation: str = "relu"):
        self.activation: nn.Module = _ACTIVATIONS[activation]()

    def write_initial_state(
        self, batch_size, n_nodes, dt, device=torch.device("cpu")
    ) -> Dict[str, Tensor]:
        return {"activity": torch.zeros(batch_size, n_nodes, device=device)}

    def state_velocity(
        self, state, params, x_t, dt, target_sum
    ) -> Dict[str, Tensor]:
        tau   = torch.clamp(params["time_const"], min=dt)
        pre_idx = params["pre_idx"]
        act_j = self.activation(state["activity"])          # (B, N)
        # params["weight"] is (B, E), act_j[:, pre_idx] is (B, E)
        syn   = target_sum(params["weight"] * act_j[:, pre_idx])  # (B, N)
        dv = (1.0 / tau) * (-state["activity"] + params["bias"] + syn + x_t)
        return {"activity": dv}

    def currents(
        self, state: Dict[str, Tensor], params: Dict[str, Tensor]
    ) -> Tensor:
        """Return synaptic current per edge (for analysis)."""
        return params["weight"] * self.activation(state["activity"])


# ---------------------------------------------------------------------------
# 2. Leaky Integrate-and-Fire
# ---------------------------------------------------------------------------

class LIFModel(BaseDynamics):
    """
    Leaky Integrate-and-Fire neuron model.

    State  : v   (membrane voltage, batch x n_nodes)
              z   (spike indicator,  batch x n_nodes)  – non-differentiable
              ref (refractory counter, batch x n_nodes)

    Params (per node): v_rest, v_thresh, tau_m, r_m
    Params (per edge): weight

    Spike surrogate: straight-through estimator for backprop.
    """

    def __init__(
        self,
        v_reset: float = -70.0,
        surrogate: str = "fast_sigmoid",
        refractory_steps: int = 2,
    ):
        self.v_reset = v_reset
        self.refractory_steps = refractory_steps
        self._surrogate = surrogate

    # Spike surrogate gradient (fast sigmoid)
    @staticmethod
    def _spike(v: Tensor, thresh: Tensor) -> Tensor:
        """Heaviside with straight-through surrogate gradient."""
        class _STE(torch.autograd.Function):
            @staticmethod
            def forward(ctx, v, thresh):
                ctx.save_for_backward(v - thresh)
                return (v >= thresh).float()
            @staticmethod
            def backward(ctx, grad):
                (diff,) = ctx.saved_tensors
                # fast-sigmoid surrogate
                sg = grad / (1.0 + torch.abs(diff)) ** 2
                return sg, -sg
        return _STE.apply(v, thresh)

    def write_initial_state(
        self, batch_size, n_nodes, dt, device=torch.device("cpu")
    ) -> Dict[str, Tensor]:
        return {
            "v":   torch.full((batch_size, n_nodes), -70.0, device=device),
            "z":   torch.zeros(batch_size, n_nodes, device=device),
            "ref": torch.zeros(batch_size, n_nodes, device=device),
        }

    def state_velocity(
        self, state, params, x_t, dt, target_sum
    ) -> Dict[str, Tensor]:
        v      = state["v"]
        z_pre  = state["z"]                                    # previous spikes
        ref    = state["ref"]

        tau   = torch.clamp(params.get("tau_m",  torch.tensor(20.0)), min=dt)
        v_rest = params.get("v_rest",   torch.tensor(-70.0))
        r_m    = params.get("r_m",      torch.tensor(1.0))
        thresh = params.get("v_thresh", torch.tensor(-55.0))

        # Synaptic input: weighted spikes from pre-synaptic neurons
        # params["weight"] is (B, E), z_pre is (B, N)
        pre_idx = params["pre_idx"]
        z_at_pre = z_pre[:, pre_idx]  # (B, E) - spikes at pre-synaptic neurons
        syn = target_sum(params["weight"] * z_at_pre)  # (B, N)
        I_total = syn + x_t                                    # (B, N)

        # Sub-threshold dynamics (masked by refractory period)
        in_ref  = (ref > 0).float()
        dv = (1.0 - in_ref) * (1.0 / tau) * (-(v - v_rest) + r_m * I_total)

        # Spikes & reset (handled externally in step() override)
        return {"v": dv, "z": torch.zeros_like(z_pre), "ref": -torch.ones_like(ref)}

    def step(self, state, params, x_t, dt, target_sum) -> Dict[str, Tensor]:
        """Override to handle spike / reset logic."""
        vel    = self.state_velocity(state, params, x_t, dt, target_sum)
        v_new  = state["v"] + dt * vel["v"]
        thresh = params.get("v_thresh", torch.tensor(-55.0))
        z_new  = self._spike(v_new, thresh)

        # Reset after spike
        v_reset_mask = z_new * self.v_reset + (1 - z_new) * v_new
        ref_new = torch.clamp(
            state["ref"] + dt * vel["ref"] + z_new * self.refractory_steps,
            min=0.0
        )
        return {"v": v_reset_mask, "z": z_new, "ref": ref_new}


# ---------------------------------------------------------------------------
# 3. Hodgkin-Huxley (single-compartment)
# ---------------------------------------------------------------------------

class HHModel(BaseDynamics):
    """
    Single-compartment Hodgkin-Huxley model.

    State  : v (mV), m, h, n  – all (batch, n_nodes)
    Params (per node): g_Na, g_K, g_L, E_Na, E_K, E_L, C_m
    Params (per edge): weight  (synaptic conductance)

    Standard HH kinetics; can be extended to include A-type, Ca2+, etc.
    """

    def write_initial_state(
        self, batch_size, n_nodes, dt, device=torch.device("cpu")
    ) -> Dict[str, Tensor]:
        v0 = torch.full((batch_size, n_nodes), -65.0, device=device)
        return {
            "v": v0,
            "m": self._m_inf(v0),
            "h": self._h_inf(v0),
            "n": self._n_inf(v0),
        }

    # -- Steady-state gating variables (vectorised) --------------------------
    @staticmethod
    def _alpha_m(v): return 0.1 * (v + 40.0) / (1.0 - torch.exp(-(v + 40.0) / 10.0) + 1e-7)
    @staticmethod
    def _beta_m(v):  return 4.0  * torch.exp(-(v + 65.0) / 18.0)
    @staticmethod
    def _alpha_h(v): return 0.07 * torch.exp(-(v + 65.0) / 20.0)
    @staticmethod
    def _beta_h(v):  return 1.0  / (1.0 + torch.exp(-(v + 35.0) / 10.0))
    @staticmethod
    def _alpha_n(v): return 0.01 * (v + 55.0) / (1.0 - torch.exp(-(v + 55.0) / 10.0) + 1e-7)
    @staticmethod
    def _beta_n(v):  return 0.125 * torch.exp(-(v + 65.0) / 80.0)

    def _m_inf(self, v): a = self._alpha_m(v); return a / (a + self._beta_m(v))
    def _h_inf(self, v): a = self._alpha_h(v); return a / (a + self._beta_h(v))
    def _n_inf(self, v): a = self._alpha_n(v); return a / (a + self._beta_n(v))

    def state_velocity(
        self, state, params, x_t, dt, target_sum
    ) -> Dict[str, Tensor]:
        v, m, h, n = state["v"], state["m"], state["h"], state["n"]

        # Default HH params (can be overridden via params dict)
        g_Na = params.get("g_Na", torch.tensor(120.0))
        g_K  = params.get("g_K",  torch.tensor(36.0))
        g_L  = params.get("g_L",  torch.tensor(0.3))
        E_Na = params.get("E_Na", torch.tensor(50.0))
        E_K  = params.get("E_K",  torch.tensor(-77.0))
        E_L  = params.get("E_L",  torch.tensor(-54.4))
        C_m  = params.get("C_m",  torch.tensor(1.0))

        # Ionic currents
        I_Na = g_Na * m**3 * h * (v - E_Na)
        I_K  = g_K  * n**4       * (v - E_K)
        I_L  = g_L               * (v - E_L)

        # Synaptic input (rate-coded pre-synaptic activity weighted by synapse)
        # params["weight"] is (B, E), v is (B, N)
        pre_idx = params["pre_idx"]
        v_at_pre = state["v"][:, pre_idx]  # (B, E) - voltage at pre-synaptic neurons
        I_syn = target_sum(params.get("weight", torch.tensor(1.0)) * v_at_pre)
        I_ext = x_t

        dv = (1.0 / C_m) * (-I_Na - I_K - I_L + I_syn + I_ext)

        # Gating variable kinetics
        am, bm = self._alpha_m(v), self._beta_m(v)
        ah, bh = self._alpha_h(v), self._beta_h(v)
        an, bn = self._alpha_n(v), self._beta_n(v)

        dm = am * (1 - m) - bm * m
        dh = ah * (1 - h) - bh * h
        dn = an * (1 - n) - bn * n

        return {"v": dv, "m": dm, "h": dh, "n": dn}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DYNAMICS_REGISTRY: Dict[str, type] = {
    "voltage": VoltageModel,
    "dmn":     VoltageModel,   # alias
    "lif":     LIFModel,
    "hh":      HHModel,
}


def build_dynamics(name: str, **kwargs) -> BaseDynamics:
    """Instantiate a dynamics model by name.

    Parameters
    ----------
    name : str
        One of ``'voltage'``, ``'dmn'``, ``'lif'``, ``'hh'``.
    **kwargs :
        Forwarded to the model constructor.
    """
    if name not in DYNAMICS_REGISTRY:
        raise ValueError(f"Unknown dynamics '{name}'. Available: {list(DYNAMICS_REGISTRY)}.")
    return DYNAMICS_REGISTRY[name](**kwargs)
