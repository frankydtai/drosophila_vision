"""
Jaxley Connectome-Constrained Network
======================================
Builds a biophysical network using the Jaxley library (JAX backend).

Jaxley provides:
  - Compartmental / point-neuron models with HH-style channels
  - Differentiable simulation via JAX (jit, vmap, value_and_grad)
  - Native support for optic-lobe-scale connectomes

This module wraps Jaxley into a unified API consistent with
``network_torch.ConnectomeNetwork`` so the training loop can swap backends.

Dependencies
------------
    pip install jaxley optax

Quick start
-----------
    from neuro_framework.connectome.loader import ConnectomeLoader
    from neuro_framework.models.network_jax import JaxleyNetwork

    loader = ConnectomeLoader.from_fafb(cell_types=['T4a','T4b','LC4'])
    net = JaxleyNetwork.from_loader(loader, channel='lif')
    params = net.get_parameters()
    recordings, params = net.simulate(params, stimuli, dt=0.025)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["JaxleyNetwork"]

# ---------------------------------------------------------------------------
# Optional import guard
# ---------------------------------------------------------------------------
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap, value_and_grad
    import jaxley as jx
    from jaxley.synapses import IonotropicSynapse
    _JAXLEY_AVAILABLE = True
except ImportError:
    _JAXLEY_AVAILABLE = False
    logger.warning(
        "Jaxley not installed. JaxleyNetwork will raise ImportError at runtime. "
        "Install with: pip install jaxley"
    )

try:
    import optax
    _OPTAX_AVAILABLE = True
except ImportError:
    _OPTAX_AVAILABLE = False


def _require_jaxley():
    if not _JAXLEY_AVAILABLE:
        raise ImportError(
            "Jaxley is required for JaxleyNetwork. "
            "Install with: pip install git+https://github.com/jaxleyverse/jaxley.git"
        )


# ---------------------------------------------------------------------------
# Channel definitions
# ---------------------------------------------------------------------------

def _make_leak_channel():
    """Return Jaxley Leak channel (always available)."""
    _require_jaxley()
    from jaxley.channels import Leak
    return Leak()


def _make_hh_channels():
    """Return list of Jaxley HH channels."""
    _require_jaxley()
    from jaxley.channels import Na, K, Leak
    return [Na(), K(), Leak()]


_CHANNEL_BUILDERS = {
    "leak":    lambda: [_make_leak_channel()],
    "lif":     lambda: [_make_leak_channel()],   # point-neuron LIF via Leak + threshold
    "hh":      _make_hh_channels,
}


# ---------------------------------------------------------------------------
# JaxleyNetwork
# ---------------------------------------------------------------------------

class JaxleyNetwork:
    """
    Connectome-constrained network backed by Jaxley.

    Parameters
    ----------
    nodes_df : pd.DataFrame
        Node metadata (must contain ``node_idx`` and ``cell_type`` columns).
    edges_df : pd.DataFrame
        Edge table with ``pre_idx``, ``post_idx``, ``syn_count``.
    nt_sign : np.ndarray  (n_nodes,)
        +1 excit / -1 inhib / 0 unknown per pre-synaptic neuron.
    channel : str
        Channel type: ``'leak'``, ``'lif'``, or ``'hh'``.
    synapse_type : str
        Jaxley synapse class to use (default ``'IonotropicSynapse'``).
    dt : float
        Default integration timestep (ms).
    """

    def __init__(
        self,
        nodes_df: pd.DataFrame,
        edges_df: pd.DataFrame,
        nt_sign: np.ndarray,
        channel: str = "leak",
        dt: float = 0.025,
    ):
        _require_jaxley()
        self.nodes_df  = nodes_df
        self.edges_df  = edges_df
        self.nt_sign   = nt_sign
        self.channel   = channel
        self.dt        = dt
        self._net: Optional[Any] = None   # jx.Network
        self._params: Optional[List] = None

        self._build()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def from_loader(
        cls,
        loader,   # ConnectomeLoader
        channel: str = "leak",
        **kwargs,
    ) -> "JaxleyNetwork":
        """Build from a ConnectomeLoader."""
        nodes, edges = loader.load()
        nt_sign = loader.nt_sign()
        return cls(nodes, edges, nt_sign, channel=channel, **kwargs)

    # ------------------------------------------------------------------
    # Build Jaxley network
    # ------------------------------------------------------------------
    def _build(self) -> None:
        """Construct the jx.Network from nodes and edges."""
        _require_jaxley()

        n_nodes = len(self.nodes_df)
        logger.info("Building Jaxley network with %d neurons ...", n_nodes)

        # Each neuron is a single-compartment point cell
        point_cell = jx.Cell()
        cells = [point_cell for _ in range(n_nodes)]
        net = jx.Network(cells)

        # Add channels to all cells
        channels = _CHANNEL_BUILDERS.get(self.channel, _CHANNEL_BUILDERS["leak"])()
        for ch in channels:
            net.insert(ch)

        # Group cells by neurotransmitter type (excitatory / inhibitory)
        excit_idx = np.where(self.nt_sign > 0)[0].tolist()
        inhib_idx = np.where(self.nt_sign < 0)[0].tolist()
        if excit_idx:
            net.cell(excit_idx).add_to_group("excitatory")
        if inhib_idx:
            net.cell(inhib_idx).add_to_group("inhibitory")

        # Group cells by cell type
        if "cell_type" in self.nodes_df.columns:
            for ct, grp in self.nodes_df.groupby("cell_type"):
                idxs = grp["node_idx"].tolist()
                safe_name = str(ct).replace("/", "_").replace(" ", "_")
                net.cell(idxs).add_to_group(safe_name)

        # Connect with synapses
        self._add_synapses(net)

        self._net = net
        logger.info("Jaxley network built.")

    def _add_synapses(self, net) -> None:
        """Add synaptic connections from edge table."""
        _require_jaxley()
        from jaxley.connect import connect
        from jaxley.synapses import IonotropicSynapse

        pre_idx  = self.edges_df["pre_idx"].to_numpy()
        post_idx = self.edges_df["post_idx"].to_numpy()

        logger.info("Adding %d synapses ...", len(pre_idx))
        pre_cells  = net.select(nodes=pre_idx)
        post_cells = net.select(nodes=post_idx)
        connect(pre_cells, post_cells, IonotropicSynapse())

    # ------------------------------------------------------------------
    # Parameter interface (Jaxley style)
    # ------------------------------------------------------------------
    def get_parameters(self) -> List[Dict]:
        """
        Return trainable parameter list in Jaxley format.

        Each entry is a dict like ``{'IonotropicSynapse_gS': jnp.array(...)}``,
        compatible with ``optax`` optimizers.
        """
        if self._net is None:
            raise RuntimeError("Network not built yet.")
        # Mark synapse conductance as trainable
        self._net.make_trainable("IonotropicSynapse_gS")
        # Mark leak conductance if present
        try:
            self._net.make_trainable("Leak_gLeak")
        except Exception:
            pass
        params = self._net.get_parameters()
        self._params = params
        return params

    # ------------------------------------------------------------------
    # Stimulus helpers
    # ------------------------------------------------------------------
    def make_current_clamp(
        self,
        node_indices: List[int],
        i_amp: float,
        t_start: float,
        t_end: float,
        dt: Optional[float] = None,
    ):
        """
        Return a Jaxley-style current stimulus dict for specified nodes.

        Parameters
        ----------
        node_indices : list of int
            Indices of neurons to stimulate.
        i_amp : float
            Current amplitude (nA).
        t_start, t_end : float
            Stimulus window (ms).
        dt : float
            Timestep (ms). Defaults to self.dt.
        """
        dt = dt or self.dt
        t_stim = np.arange(t_start, t_end, dt)
        i_stim = np.full_like(t_stim, i_amp)
        return {"node_indices": node_indices, "t": t_stim, "i": i_stim}

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    def simulate(
        self,
        params: List[Dict],
        t_max: float,
        stimuli: Optional[List[Dict]] = None,
        dt: Optional[float] = None,
        record_nodes: Optional[List[int]] = None,
    ) -> Tuple[Any, Any]:
        """
        Run a differentiable simulation.

        Parameters
        ----------
        params : list of dict
            Trainable parameters returned by ``get_parameters()``.
        t_max : float
            Total simulation time (ms).
        stimuli : list of current-clamp dicts, optional
            Each produced by ``make_current_clamp()``.
        dt : float, optional
            Timestep (ms).
        record_nodes : list of int, optional
            Neuron indices to record. Defaults to all.

        Returns
        -------
        v_rec : jnp.ndarray  (T, n_recorded)
            Recorded membrane voltages.
        state : Jaxley state dict
        """
        _require_jaxley()
        dt = dt or self.dt
        net = self._net

        # Apply stimuli
        if stimuli:
            for stim in stimuli:
                node_sel = net.select(nodes=stim["node_indices"])
                node_sel.stimulate(jnp.array(stim["i"]))

        # Set up recordings
        if record_nodes is None:
            net.record("v")
        else:
            net.select(nodes=record_nodes).record("v")

        # Jaxley-style differentiable simulation
        v_rec, state = jx.integrate(net, params=params, t_max=t_max, delta_t=dt)
        return v_rec, state

    def simulate_batch(
        self,
        params: List[Dict],
        stimuli_batch: List[List[Dict]],
        t_max: float,
        dt: Optional[float] = None,
    ):
        """
        Parallelise simulation over a batch of stimuli using ``jax.vmap``.

        Parameters
        ----------
        stimuli_batch : list of stimulus lists
            Outer list = batch; inner list = per-stimulus dicts.

        Returns
        -------
        v_batch : jnp.ndarray  (batch, T, n_recorded)
        """
        _require_jaxley()
        dt = dt or self.dt

        def _single(stims):
            v, _ = self.simulate(params, t_max=t_max, stimuli=stims, dt=dt)
            return v

        return jax.vmap(_single)(stimuli_batch)

    # ------------------------------------------------------------------
    # Gradient computation (Jaxley + Optax)
    # ------------------------------------------------------------------
    def make_loss_and_grad(
        self,
        loss_fn,
        t_max: float,
        stimuli: Optional[List[Dict]] = None,
        dt: Optional[float] = None,
    ):
        """
        Return a (loss, grad) function compatible with Optax.

        Parameters
        ----------
        loss_fn : callable
            ``loss_fn(v_rec) -> scalar`` where v_rec is the voltage recording.
        t_max : float
            Simulation duration (ms).

        Returns
        -------
        value_and_grad_fn : callable
            Takes ``params`` and returns ``(loss_value, grads)``.
        """
        _require_jaxley()

        def _loss(params):
            v_rec, _ = self.simulate(params, t_max=t_max, stimuli=stimuli, dt=dt)
            return loss_fn(v_rec)

        return value_and_grad(_loss)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def network(self):
        """Underlying jx.Network object."""
        return self._net

    @property
    def n_nodes(self) -> int:
        return len(self.nodes_df)

    @property
    def n_edges(self) -> int:
        return len(self.edges_df)

    def __repr__(self) -> str:
        return (
            f"JaxleyNetwork(n_nodes={self.n_nodes}, n_edges={self.n_edges}, "
            f"channel='{self.channel}', dt={self.dt}ms)"
        )
