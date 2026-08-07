"""
Synapse Models
==============
PyTorch implementations of various synapse models inspired by Jaxley.

These models compute synaptic currents based on pre- and post-synaptic voltages.

Available models:
- TanhRateSynapse: Simple tanh-based rate synapse (no state)
- TanhConductanceSynapse: Tanh synapse with conductance-based current
- IonotropicSynapse: Biophysical ionotropic synapse with state variable
- NMDASynapse: Voltage-dependent NMDA receptor with Mg²⁺ block
- GABAaSynapse: Fast inhibitory GABA_A receptor synapse

References:
- Jaxley: https://jaxley.readthedocs.io/
- Abbott & Marder (1998): "Modeling Small Networks"
- Jahr & Stevens (1990): Voltage dependence of NMDA channels
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Optional, Tuple

__all__ = [
    "BaseSynapse",
    "TanhRateSynapse",
    "TanhConductanceSynapse",
    "IonotropicSynapse",
    "NMDASynapse",
    "GABAaSynapse",
]


class BaseSynapse(nn.Module):
    """
    Base class for synapse models.
    
    All synapse models should implement:
    - compute_current(): Calculate synaptic current
    - update_states(): Update synapse state variables (if any)
    """
    
    def __init__(self, n_edges: int):
        super().__init__()
        self.n_edges = n_edges
    
    def compute_current(
        self,
        states: Dict[str, Tensor],
        pre_voltage: Tensor,
        post_voltage: Tensor,
        params: Dict[str, Tensor],
    ) -> Tensor:
        """
        Compute synaptic current.
        
        Parameters
        ----------
        states : dict
            Synapse state variables (e.g., {'s': ...})
        pre_voltage : Tensor (batch, n_edges)
            Pre-synaptic voltages
        post_voltage : Tensor (batch, n_edges)
            Post-synaptic voltages
        params : dict
            Synapse parameters (e.g., {'gS': ..., 'e_syn': ...})
        
        Returns
        -------
        current : Tensor (batch, n_edges)
            Synaptic current in nA
        """
        raise NotImplementedError
    
    def update_states(
        self,
        states: Dict[str, Tensor],
        delta_t: float,
        pre_voltage: Tensor,
        post_voltage: Tensor,
        params: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """
        Update synapse state variables.
        
        Parameters
        ----------
        states : dict
            Current synapse states
        delta_t : float
            Time step in ms
        pre_voltage : Tensor (batch, n_edges)
            Pre-synaptic voltages
        post_voltage : Tensor (batch, n_edges)
            Post-synaptic voltages
        params : dict
            Synapse parameters
        
        Returns
        -------
        new_states : dict
            Updated synapse states
        """
        return states  # Default: no state update


class TanhRateSynapse(BaseSynapse):
    """
    Tanh-based rate synapse (no state variables).
    
    Current is computed as:
        I = -gS * tanh((V_pre - x_offset) * slope)
    
    Parameters
    ----------
    n_edges : int
        Number of synaptic connections
    gS : float
        Maximal synaptic conductance (default: 1e-4)
    x_offset : float
        Voltage offset for tanh (default: -70.0 mV)
    slope : float
        Slope of tanh activation (default: 1.0)
    learn_params : bool
        Whether synapse parameters are learnable (default: True)
    """
    
    def __init__(
        self,
        n_edges: int,
        gS: float = 1e-4,
        x_offset: float = -70.0,
        slope: float = 1.0,
        learn_params: bool = True,
    ):
        super().__init__(n_edges)
        
        # Learnable parameters
        self.log_gS = nn.Parameter(
            torch.full((n_edges,), torch.log(torch.tensor(gS))),
            requires_grad=learn_params
        )
        self.x_offset = nn.Parameter(
            torch.full((n_edges,), x_offset),
            requires_grad=learn_params
        )
        self.slope = nn.Parameter(
            torch.full((n_edges,), slope),
            requires_grad=learn_params
        )
    
    def compute_current(
        self,
        states: Dict[str, Tensor],
        pre_voltage: Tensor,
        post_voltage: Tensor,
        params: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        """
        Compute synaptic current.
        
        Parameters
        ----------
        states : dict
            Unused (no state variables)
        pre_voltage : Tensor (batch, n_edges)
            Pre-synaptic voltages
        post_voltage : Tensor (batch, n_edges)
            Post-synaptic voltages (unused)
        params : dict, optional
            Override parameters
        
        Returns
        -------
        current : Tensor (batch, n_edges)
            Synaptic current in nA
        """
        gS = torch.exp(self.log_gS) if params is None else params['gS']
        x_offset = self.x_offset if params is None else params['x_offset']
        slope = self.slope if params is None else params['slope']
        
        # I = -gS * tanh((V_pre - x_offset) * slope)
        current = -gS * torch.tanh((pre_voltage - x_offset) * slope)
        return current


class TanhConductanceSynapse(BaseSynapse):
    """
    Tanh conductance-based synapse (no state variables).
    
    Current is computed as:
        I = tanh((V_pre - x_offset) * slope) * gS * (V_post - e_syn)
    
    This is similar to TanhRateSynapse but includes conductance-based
    driving force (V_post - e_syn).
    
    Parameters
    ----------
    n_edges : int
        Number of synaptic connections
    gS : float
        Maximal synaptic conductance (default: 1e-4 uS)
    e_syn : float
        Reversal potential (default: 0.0 mV)
    x_offset : float
        Voltage offset for tanh (default: -70.0 mV)
    slope : float
        Slope of tanh activation (default: 1.0)
    learn_params : bool
        Whether synapse parameters are learnable (default: True)
    """
    
    def __init__(
        self,
        n_edges: int,
        gS: float = 1e-4,
        e_syn: float = 0.0,
        x_offset: float = -70.0,
        slope: float = 1.0,
        learn_params: bool = True,
    ):
        super().__init__(n_edges)
        
        # Learnable parameters
        self.log_gS = nn.Parameter(
            torch.full((n_edges,), torch.log(torch.tensor(gS))),
            requires_grad=learn_params
        )
        self.e_syn = nn.Parameter(
            torch.full((n_edges,), e_syn),
            requires_grad=learn_params
        )
        self.x_offset = nn.Parameter(
            torch.full((n_edges,), x_offset),
            requires_grad=learn_params
        )
        self.slope = nn.Parameter(
            torch.full((n_edges,), slope),
            requires_grad=learn_params
        )
    
    def compute_current(
        self,
        states: Dict[str, Tensor],
        pre_voltage: Tensor,
        post_voltage: Tensor,
        params: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        """
        Compute synaptic current.
        
        Parameters
        ----------
        states : dict
            Unused (no state variables)
        pre_voltage : Tensor (batch, n_edges)
            Pre-synaptic voltages
        post_voltage : Tensor (batch, n_edges)
            Post-synaptic voltages
        params : dict, optional
            Override parameters
        
        Returns
        -------
        current : Tensor (batch, n_edges)
            Synaptic current in nA
        """
        gS = torch.exp(self.log_gS) if params is None else params['gS']
        e_syn = self.e_syn if params is None else params['e_syn']
        x_offset = self.x_offset if params is None else params['x_offset']
        slope = self.slope if params is None else params['slope']
        
        # Activation based on pre-synaptic voltage
        activation = torch.tanh((pre_voltage - x_offset) * slope)
        
        # Conductance-based current
        current = activation * gS * (post_voltage - e_syn)
        return current


class IonotropicSynapse(BaseSynapse):
    """
    Biophysical ionotropic synapse with state variable.
    
    The synapse state 's' represents the probability that a postsynaptic
    receptor channel is open, depending on neurotransmitter release from
    the presynaptic terminal.
    
    State dynamics:
        s_inf = 1 / (1 + exp((v_th - V_pre) / delta))
        tau_s = (1 - s_inf) / k_minus
        ds/dt = (s_inf - s) / tau_s
    
    Current:
        I = gS * s * (V_post - e_syn)
    
    Parameters
    ----------
    n_edges : int
        Number of synaptic connections
    gS : float
        Maximal synaptic conductance (default: 1e-4 uS)
    e_syn : float
        Reversal potential (default: 0.0 mV)
    k_minus : float
        Unbinding rate constant (default: 0.025 s^-1)
    v_th : float
        Voltage threshold for activation (default: -35.0 mV)
    delta : float
        Voltage sensitivity (default: 10.0 mV)
    s_init : float
        Initial state value (default: 0.2)
    learn_params : bool
        Whether synapse parameters are learnable (default: True)
    
    References
    ----------
    Abbott & Marder (1998): "Modeling Small Networks" in Methods in
    Neuronal Modeling, MIT Press.
    """
    
    def __init__(
        self,
        n_edges: int,
        gS: float = 1e-4,
        e_syn: float = 0.0,
        k_minus: float = 0.025,
        v_th: float = -35.0,
        delta: float = 10.0,
        s_init: float = 0.2,
        learn_params: bool = True,
    ):
        super().__init__(n_edges)
        
        # Learnable parameters
        self.log_gS = nn.Parameter(
            torch.full((n_edges,), torch.log(torch.tensor(gS))),
            requires_grad=learn_params
        )
        self.e_syn = nn.Parameter(
            torch.full((n_edges,), e_syn),
            requires_grad=learn_params
        )
        self.k_minus = nn.Parameter(
            torch.full((n_edges,), k_minus),
            requires_grad=learn_params
        )
        self.v_th = nn.Parameter(
            torch.full((n_edges,), v_th),
            requires_grad=learn_params
        )
        self.delta = nn.Parameter(
            torch.full((n_edges,), delta),
            requires_grad=learn_params
        )
        
        # Initial state
        self.s_init = s_init
    
    def init_states(self, batch_size: int, device: torch.device) -> Dict[str, Tensor]:
        """Initialize synapse states."""
        return {
            's': torch.full((batch_size, self.n_edges), self.s_init, device=device)
        }
    
    def update_states(
        self,
        states: Dict[str, Tensor],
        delta_t: float,
        pre_voltage: Tensor,
        post_voltage: Tensor,
        params: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """
        Update synapse state variable 's'.
        
        Uses exponential Euler integration for stability.
        """
        v_th = self.v_th if params is None else params['v_th']
        delta = self.delta if params is None else params['delta']
        k_minus = self.k_minus if params is None else params['k_minus']
        
        s = states['s']
        
        # Steady-state activation
        s_inf = 1.0 / (1.0 + torch.exp((v_th - pre_voltage) / delta))
        
        # Time constant
        tau_s = (1.0 - s_inf) / (k_minus + 1e-8)
        
        # Exponential Euler update
        slope = -1.0 / (tau_s + 1e-8)
        exp_term = torch.exp(slope * delta_t)
        new_s = s * exp_term + s_inf * (1.0 - exp_term)
        
        return {'s': new_s}
    
    def compute_current(
        self,
        states: Dict[str, Tensor],
        pre_voltage: Tensor,
        post_voltage: Tensor,
        params: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        """
        Compute synaptic current.
        
        Current is proportional to the open probability 's' and the
        driving force (V_post - e_syn).
        """
        gS = torch.exp(self.log_gS) if params is None else params['gS']
        e_syn = self.e_syn if params is None else params['e_syn']
        
        s = states['s']
        
        # Conductance-based current
        g_syn = gS * s
        current = g_syn * (post_voltage - e_syn)
        return current


class NMDASynapse(BaseSynapse):
    """
    NMDA receptor synapse with voltage-dependent Mg²⁺ block.

    The NMDA receptor current is gated by both a ligand-binding state
    variable *s* and a voltage-dependent magnesium block factor *B(V)*:

        B(V) = 1 / (1 + [Mg²⁺]/3.57 · exp(−0.062 · V))
        I = gS · s · B(V_post) · (V_post − e_syn)

    NMDA receptors have slow kinetics (τ_rise ≈ 2 ms, τ_decay ≈ 50–100 ms)
    and are permeable to Ca²⁺, making them crucial for learning.

    Parameters
    ----------
    n_edges : int
    gS : float
        Maximal conductance (default: 1e-4 μS).
    e_syn : float
        Reversal potential (default: 0.0 mV).
    mg_conc : float
        Extracellular Mg²⁺ concentration in mM (default: 1.0).
    k_minus : float
        Unbinding rate (default: 0.01 s⁻¹, slower than ionotropic).
    v_th : float
        Pre-synaptic activation threshold (default: −35.0 mV).
    delta : float
        Pre-synaptic activation slope (default: 10.0 mV).

    References
    ----------
    Jahr & Stevens (1990), J. Neurosci.
    """

    def __init__(
        self,
        n_edges: int,
        gS: float = 1e-4,
        e_syn: float = 0.0,
        mg_conc: float = 1.0,
        k_minus: float = 0.01,
        v_th: float = -35.0,
        delta: float = 10.0,
        s_init: float = 0.0,
        learn_params: bool = True,
    ):
        super().__init__(n_edges)
        self.log_gS = nn.Parameter(
            torch.full((n_edges,), torch.log(torch.tensor(gS))),
            requires_grad=learn_params,
        )
        self.e_syn = nn.Parameter(
            torch.full((n_edges,), e_syn), requires_grad=learn_params,
        )
        self.mg_conc = nn.Parameter(
            torch.tensor(mg_conc), requires_grad=False,
        )
        self.k_minus = nn.Parameter(
            torch.full((n_edges,), k_minus), requires_grad=learn_params,
        )
        self.v_th = nn.Parameter(
            torch.full((n_edges,), v_th), requires_grad=learn_params,
        )
        self.delta = nn.Parameter(
            torch.full((n_edges,), delta), requires_grad=learn_params,
        )
        self.s_init = s_init

    def init_states(self, batch_size: int, device: torch.device) -> Dict[str, Tensor]:
        return {
            "s": torch.full(
                (batch_size, self.n_edges), self.s_init, device=device,
            )
        }

    def _mg_block(self, V_post: Tensor) -> Tensor:
        """Voltage-dependent Mg²⁺ block factor B(V)."""
        return 1.0 / (1.0 + (self.mg_conc / 3.57) * torch.exp(-0.062 * V_post))

    def update_states(
        self,
        states: Dict[str, Tensor],
        delta_t: float,
        pre_voltage: Tensor,
        post_voltage: Tensor,
        params: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        s = states["s"]
        s_inf = 1.0 / (1.0 + torch.exp((self.v_th - pre_voltage) / self.delta))
        tau_s = (1.0 - s_inf) / (self.k_minus + 1e-8)
        slope = -1.0 / (tau_s + 1e-8)
        exp_term = torch.exp(slope * delta_t)
        return {"s": s * exp_term + s_inf * (1.0 - exp_term)}

    def compute_current(
        self,
        states: Dict[str, Tensor],
        pre_voltage: Tensor,
        post_voltage: Tensor,
        params: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        gS = torch.exp(self.log_gS) if params is None else params["gS"]
        e_syn = self.e_syn if params is None else params["e_syn"]
        s = states["s"]
        B = self._mg_block(post_voltage)
        return gS * s * B * (post_voltage - e_syn)


class GABAaSynapse(BaseSynapse):
    """
    Fast inhibitory GABA_A receptor synapse.

    GABA_A receptors are ligand-gated Cl⁻ channels with fast kinetics
    (τ_rise ≈ 0.5 ms, τ_decay ≈ 5–10 ms) and a reversal potential
    near the Cl⁻ equilibrium (≈ −80 mV), providing fast inhibition.

        I = gS · s · (V_post − e_syn)

    Parameters
    ----------
    n_edges : int
    gS : float
        Maximal conductance (default: 1e-4 μS).
    e_syn : float
        Reversal potential (default: −80.0 mV, Cl⁻ equilibrium).
    k_minus : float
        Unbinding rate (default: 0.1 s⁻¹, fast).
    v_th : float
        Pre-synaptic activation threshold (default: −35.0 mV).
    delta : float
        Pre-synaptic activation slope (default: 10.0 mV).
    """

    def __init__(
        self,
        n_edges: int,
        gS: float = 1e-4,
        e_syn: float = -80.0,
        k_minus: float = 0.1,
        v_th: float = -35.0,
        delta: float = 10.0,
        s_init: float = 0.0,
        learn_params: bool = True,
    ):
        super().__init__(n_edges)
        self.log_gS = nn.Parameter(
            torch.full((n_edges,), torch.log(torch.tensor(gS))),
            requires_grad=learn_params,
        )
        self.e_syn = nn.Parameter(
            torch.full((n_edges,), e_syn), requires_grad=learn_params,
        )
        self.k_minus = nn.Parameter(
            torch.full((n_edges,), k_minus), requires_grad=learn_params,
        )
        self.v_th = nn.Parameter(
            torch.full((n_edges,), v_th), requires_grad=learn_params,
        )
        self.delta = nn.Parameter(
            torch.full((n_edges,), delta), requires_grad=learn_params,
        )
        self.s_init = s_init

    def init_states(self, batch_size: int, device: torch.device) -> Dict[str, Tensor]:
        return {
            "s": torch.full(
                (batch_size, self.n_edges), self.s_init, device=device,
            )
        }

    def update_states(
        self,
        states: Dict[str, Tensor],
        delta_t: float,
        pre_voltage: Tensor,
        post_voltage: Tensor,
        params: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        s = states["s"]
        s_inf = 1.0 / (1.0 + torch.exp((self.v_th - pre_voltage) / self.delta))
        tau_s = (1.0 - s_inf) / (self.k_minus + 1e-8)
        slope = -1.0 / (tau_s + 1e-8)
        exp_term = torch.exp(slope * delta_t)
        return {"s": s * exp_term + s_inf * (1.0 - exp_term)}

    def compute_current(
        self,
        states: Dict[str, Tensor],
        pre_voltage: Tensor,
        post_voltage: Tensor,
        params: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        gS = torch.exp(self.log_gS) if params is None else params["gS"]
        e_syn = self.e_syn if params is None else params["e_syn"]
        s = states["s"]
        return gS * s * (post_voltage - e_syn)
