"""
Trainer
=======
High-level training loop for connectome-constrained networks.

Supports two backends:
  - PyTorch (ConnectomeNetwork)  via standard autograd + Adam / AdamW
  - Jaxley   (JaxleyNetwork)     via JAX value_and_grad + Optax

Training methods
----------------
  Method A – DMN-style constraint training:
      Includes knockout / silencing losses alongside a supervised objective.
  Method B – Layer-wise progressive training:
      Upstream layers are frozen; downstream layers are trained sequentially.

Quick start (PyTorch)
---------------------
    from neuro_framework.training.trainer import TorchTrainer

    trainer = TorchTrainer(network, optimizer_cfg={'lr': 1e-3})
    history = trainer.train(
        x_train,         # (batch, T, n_nodes)
        y_train,         # (batch, T, n_output)
        n_epochs=100,
        loss_name='mse',
    )

Quick start (Jaxley)
--------------------
    from neuro_framework.training.trainer import JaxTrainer

    trainer = JaxTrainer(jax_network, lr=1e-3)
    params, history = trainer.train(stimuli, targets, t_max=500.0, n_steps=1000)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from .losses import LossRegistry, combined_loss, mse_loss

logger = logging.getLogger(__name__)

__all__ = [
    "TorchTrainer",
    "JaxTrainer",
    "TrainingHistory",
]


# ---------------------------------------------------------------------------
# Training history container
# ---------------------------------------------------------------------------

@dataclass
class TrainingHistory:
    """Lightweight container for training metrics."""
    train_loss: List[float]       = field(default_factory=list)
    val_loss:   List[float]       = field(default_factory=list)
    extra:      Dict[str, List]   = field(default_factory=dict)

    def log(self, train: float, val: Optional[float] = None, **kwargs):
        self.train_loss.append(float(train))
        if val is not None:
            self.val_loss.append(float(val))
        for k, v in kwargs.items():
            self.extra.setdefault(k, []).append(v)

    def __repr__(self) -> str:
        n = len(self.train_loss)
        last = f"{self.train_loss[-1]:.4f}" if n else "–"
        return f"TrainingHistory(steps={n}, last_train_loss={last})"


# ---------------------------------------------------------------------------
# PyTorch Trainer
# ---------------------------------------------------------------------------

class TorchTrainer:
    """
    Training loop for PyTorch-based ConnectomeNetwork.

    Parameters
    ----------
    network : nn.Module
        The ConnectomeNetwork (or any nn.Module).
    optimizer_cfg : dict
        Kwargs forwarded to ``torch.optim.AdamW``.
        Example: ``{'lr': 1e-3, 'weight_decay': 1e-5}``.
    device : str or torch.device
        Computation device.
    clip_grad : float or None
        Global gradient norm clipping. None = no clipping.
    """

    def __init__(
        self,
        network: nn.Module,
        optimizer_cfg: Optional[Dict] = None,
        device: Union[str, torch.device] = "cpu",
        clip_grad: Optional[float] = 1.0,
    ):
        self.network   = network.to(device)
        self.device    = torch.device(device)
        self.clip_grad = clip_grad
        cfg = optimizer_cfg or {"lr": 1e-3}
        self.optimizer = torch.optim.AdamW(network.parameters(), **cfg)
        self.history   = TrainingHistory()

    # ------------------------------------------------------------------
    # Core training step
    # ------------------------------------------------------------------
    def _step(
        self,
        x: Tensor,
        y: Tensor,
        loss_fn: Callable,
        output_node_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Single forward + backward pass.

        Parameters
        ----------
        x : (batch, T, n_nodes)  stimulus
        y : (batch, T, n_output)  ground-truth (e.g. calcium / voltage traces)
        loss_fn : callable taking (pred, target) -> scalar Tensor
        output_node_mask : (n_nodes,) bool – which nodes to compare with y
        """
        self.optimizer.zero_grad()
        pred = self.network(x)                          # (batch, T, n_nodes)

        if output_node_mask is not None:
            pred_out = pred[:, :, output_node_mask]     # (batch, T, n_output)
        else:
            pred_out = pred

        loss = loss_fn(pred_out, y)
        loss.backward()

        if self.clip_grad is not None:
            nn.utils.clip_grad_norm_(self.network.parameters(), self.clip_grad)

        self.optimizer.step()
        return loss

    # ------------------------------------------------------------------
    # Method A: DMN-style training with knockout
    # ------------------------------------------------------------------
    def step_with_knockout(
        self,
        x_full:     Tensor,
        x_knockout: Tensor,
        y:          Tensor,
        knockout_node_mask: Tensor,
        output_node_mask:   Optional[Tensor] = None,
        supervised_weight:  float = 1.0,
        knockout_weight:    float = 0.5,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        DMN-style step (Method A): supervised loss + knockout consistency.

        Parameters
        ----------
        x_full     : (B, T, N) stimulus without silencing
        x_knockout : (B, T, N) stimulus with specified nodes zeroed out
        y          : (B, T, n_output) ground-truth activity
        knockout_node_mask : (N,) bool – nodes to silence in x_knockout
        """
        from .losses import knockout_consistency_loss

        self.optimizer.zero_grad()

        pred_full = self.network(x_full)
        pred_ko   = self.network(x_knockout)

        out_mask = output_node_mask
        p_full_out = pred_full[:, :, out_mask] if out_mask is not None else pred_full
        p_ko_out   = pred_ko[:, :, out_mask]   if out_mask is not None else pred_ko

        l_sup = mse_loss(p_full_out, y)
        l_ko  = knockout_consistency_loss(p_full_out, p_ko_out)

        total, breakdown = combined_loss({
            "supervised": (l_sup, supervised_weight),
            "knockout":   (l_ko,  knockout_weight),
        })
        total.backward()

        if self.clip_grad is not None:
            nn.utils.clip_grad_norm_(self.network.parameters(), self.clip_grad)
        self.optimizer.step()
        return total, breakdown

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    def train(
        self,
        x_train: Tensor,
        y_train: Tensor,
        n_epochs: int = 100,
        batch_size: int = 4,
        loss_name: str = "mse",
        x_val: Optional[Tensor] = None,
        y_val: Optional[Tensor] = None,
        output_node_mask: Optional[Tensor] = None,
        log_every: int = 10,
        checkpoint_every: Optional[int] = None,
        checkpoint_path: str = "checkpoint.pt",
    ) -> TrainingHistory:
        """
        Full training loop.

        Parameters
        ----------
        x_train : (N_samples, T, n_nodes)
        y_train : (N_samples, T, n_output)
        n_epochs : int
        batch_size : int
        loss_name : str  – key in LossRegistry (``'mse'``, ``'correlation'``, etc.)
        x_val / y_val : optional validation tensors
        output_node_mask : (n_nodes,) bool – restrict output comparison
        log_every : print every N epochs
        checkpoint_every : save checkpoint every N epochs (None = disabled)
        checkpoint_path : file path for checkpoints

        Returns
        -------
        TrainingHistory
        """
        loss_fn = LossRegistry.get(loss_name, mse_loss)
        dataset  = TensorDataset(
            x_train.to(self.device),
            y_train.to(self.device),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.network.train()
        t0 = time.time()

        for epoch in range(1, n_epochs + 1):
            epoch_losses = []
            for xb, yb in loader:
                loss = self._step(xb, yb, loss_fn, output_node_mask)
                epoch_losses.append(loss.item())

            train_loss = float(np.mean(epoch_losses))
            val_loss   = None

            if x_val is not None and y_val is not None:
                self.network.eval()
                with torch.no_grad():
                    pv = self.network(x_val.to(self.device))
                    if output_node_mask is not None:
                        pv = pv[:, :, output_node_mask]
                    val_loss = loss_fn(pv, y_val.to(self.device)).item()
                self.network.train()

            self.history.log(train_loss, val_loss)

            if epoch % log_every == 0:
                elapsed = time.time() - t0
                val_str = f"  val={val_loss:.4f}" if val_loss is not None else ""
                logger.info(
                    "Epoch %4d/%d  train=%.4f%s  [%.1fs]",
                    epoch, n_epochs, train_loss, val_str, elapsed,
                )

            if checkpoint_every and epoch % checkpoint_every == 0:
                self.save(checkpoint_path)
                logger.info("Checkpoint saved to %s", checkpoint_path)

        return self.history

    # ------------------------------------------------------------------
    # Method B: Layer-wise progressive training
    # ------------------------------------------------------------------
    def layerwise_train(
        self,
        layer_groups: List[List[nn.Parameter]],
        x_train: Tensor,
        y_train: Tensor,
        n_epochs_per_layer: int = 50,
        **train_kwargs,
    ) -> TrainingHistory:
        """
        Method B: freeze all layers; unfreeze and train one group at a time.

        Parameters
        ----------
        layer_groups : list of parameter groups (from upstream to downstream)
            Example:
                [
                    list(net.photoreceptor_params()),
                    list(net.t4_t5_params()),
                    list(net.lc_params()),
                ]
        n_epochs_per_layer : epochs per group
        """
        # Start with all frozen
        for p in self.network.parameters():
            p.requires_grad_(False)

        for i, group in enumerate(layer_groups):
            logger.info(
                "Layer-wise training: unfreezing group %d/%d (%d params) ...",
                i + 1, len(layer_groups), sum(p.numel() for p in group),
            )
            for p in group:
                p.requires_grad_(True)

            # Rebuild optimizer with only currently-active params
            trainable = [p for p in self.network.parameters() if p.requires_grad]
            self.optimizer = torch.optim.AdamW(trainable, lr=1e-3)

            self.train(
                x_train, y_train,
                n_epochs=n_epochs_per_layer,
                **train_kwargs,
            )

        return self.history

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save({
            "model_state":     self.network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "history":         self.history,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.network.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.history = ckpt.get("history", TrainingHistory())
        logger.info("Loaded checkpoint from %s", path)


# ---------------------------------------------------------------------------
# Jaxley / JAX Trainer
# ---------------------------------------------------------------------------

class JaxTrainer:
    """
    Training loop for Jaxley-based JaxleyNetwork.

    Requires ``optax`` (``pip install optax``).

    Parameters
    ----------
    network : JaxleyNetwork
    lr : float  learning rate
    optimizer : str  ``'adam'`` or ``'adamw'``
    """

    def __init__(
        self,
        network,             # JaxleyNetwork
        lr: float = 1e-3,
        optimizer: str = "adam",
    ):
        try:
            import optax
        except ImportError:
            raise ImportError("optax is required for JaxTrainer. pip install optax")
        import optax as _optax

        self.network  = network
        self.lr       = lr
        self.history  = TrainingHistory()

        if optimizer == "adamw":
            self._opt = _optax.adamw(lr)
        else:
            self._opt = _optax.adam(lr)

    def train(
        self,
        stimuli: Any,
        targets: Any,         # jnp.ndarray  (T, n_output)
        t_max: float,
        n_steps: int = 500,
        loss_fn: Optional[Callable] = None,
        record_nodes: Optional[List[int]] = None,
        log_every: int = 50,
        dt: Optional[float] = None,
    ) -> Tuple[Any, TrainingHistory]:
        """
        Gradient-descent training loop for a Jaxley network.

        Parameters
        ----------
        stimuli : current-clamp dicts (passed to JaxleyNetwork.simulate)
        targets : jnp.ndarray  (T, n_recorded)
            Ground-truth voltage / activity to match.
        t_max   : float  total simulation time (ms)
        n_steps : int    number of gradient steps
        loss_fn : callable  v_rec -> scalar  (defaults to MSE)
        record_nodes : list of int  neurons to record
        log_every : int  logging frequency
        dt : float  timestep (ms)

        Returns
        -------
        params : trained parameter list
        history : TrainingHistory
        """
        try:
            import jax.numpy as jnp
            import optax
        except ImportError:
            raise ImportError("jax and optax are required.")

        params = self.network.get_parameters()
        opt_state = self._opt.init(params)

        if loss_fn is None:
            def loss_fn(v_rec):
                return jnp.mean((v_rec - targets) ** 2)

        vg_fn = self.network.make_loss_and_grad(
            loss_fn, t_max=t_max, stimuli=stimuli, dt=dt
        )

        import jax

        t0 = time.time()
        for step in range(1, n_steps + 1):
            loss_val, grads = vg_fn(params)
            updates, opt_state = self._opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

            self.history.log(float(loss_val))

            if step % log_every == 0:
                elapsed = time.time() - t0
                logger.info(
                    "Step %4d/%d  loss=%.4f  [%.1fs]",
                    step, n_steps, float(loss_val), elapsed,
                )

        return params, self.history
