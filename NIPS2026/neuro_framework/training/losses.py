"""
Loss Functions
==============
Loss functions for training connectome-constrained networks.

All losses take predicted activity and ground-truth tensors and return a
scalar.  They are compatible with both PyTorch autograd and (where noted)
JAX / Optax.

Available losses
----------------
  mse_loss                 – mean squared error on activity traces
  correlation_loss         – 1 - Pearson correlation (maximise similarity)
  spike_rate_loss          – MSE between firing-rate targets and mean activity
  direction_selectivity_loss – DMN-style: preferred-direction vs null-direction
  knockout_consistency_loss  – neuron silencing (method A from project plan)
  combined_loss            – weighted sum of multiple losses
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "mse_loss",
    "correlation_loss",
    "spike_rate_loss",
    "direction_selectivity_loss",
    "knockout_consistency_loss",
    "combined_loss",
    "LossRegistry",
]


# ---------------------------------------------------------------------------
# Basic losses
# ---------------------------------------------------------------------------

def mse_loss(pred: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """
    Mean squared error between predicted and target activity.

    Parameters
    ----------
    pred   : (batch, T, n_nodes)  or  (batch, T)
    target : same shape as pred
    mask   : optional boolean tensor of same shape; if given, loss is computed
             only on True entries.
    """
    diff = (pred - target) ** 2
    if mask is not None:
        diff = diff * mask.float()
        return diff.sum() / (mask.float().sum() + 1e-8)
    return diff.mean()


def correlation_loss(
    pred: Tensor,
    target: Tensor,
    dim: int = 1,
    eps: float = 1e-8,
) -> Tensor:
    """
    1 - Pearson correlation averaged over batch & nodes.

    Minimising this loss maximises temporal correlation with the target trace.

    Parameters
    ----------
    pred   : (batch, T, n_neurons)
    target : (batch, T, n_neurons)
    dim    : dimension along which correlation is computed (default: time)
    """
    pred_c   = pred   - pred.mean(dim=dim, keepdim=True)
    target_c = target - target.mean(dim=dim, keepdim=True)
    num  = (pred_c * target_c).sum(dim=dim)          # (batch, n_neurons)
    denom = (
        pred_c.pow(2).sum(dim=dim).sqrt()
        * target_c.pow(2).sum(dim=dim).sqrt()
        + eps
    )
    corr = (num / denom).mean()                       # scalar
    return 1.0 - corr


def spike_rate_loss(
    pred: Tensor,
    target_rates: Tensor,
    node_mask: Optional[Tensor] = None,
) -> Tensor:
    """
    MSE between mean activity across time and target firing rates.

    Parameters
    ----------
    pred         : (batch, T, n_nodes)
    target_rates : (n_nodes,)  or  (batch, n_nodes)
    node_mask    : (n_nodes,) boolean – restrict to subset of neurons.
    """
    mean_act = pred.mean(dim=1)                       # (batch, n_nodes)
    if target_rates.dim() == 1:
        target_rates = target_rates.unsqueeze(0)      # (1, n_nodes)
    diff = (mean_act - target_rates) ** 2
    if node_mask is not None:
        diff = diff[:, node_mask]
    return diff.mean()


# ---------------------------------------------------------------------------
# Neuroscience-specific losses
# ---------------------------------------------------------------------------

def direction_selectivity_loss(
    activity_preferred: Tensor,
    activity_null:      Tensor,
    target_dsi:         Optional[Tensor] = None,
    eps: float = 1e-8,
) -> Tensor:
    """
    Direction Selectivity Index (DSI) loss, DMN-style.

    Encourages neurons to respond more strongly to the preferred direction
    than the null direction.

    DSI = (R_pref - R_null) / (R_pref + R_null + eps)

    Loss = MSE(DSI, target_dsi)  if target_dsi is given,
           else  -mean(DSI)      (maximise selectivity).

    Parameters
    ----------
    activity_preferred : (batch, T, n_nodes)  responses to preferred direction
    activity_null      : (batch, T, n_nodes)  responses to null direction
    target_dsi         : (n_nodes,) or None   target DSI values per neuron
    """
    r_pref = activity_preferred.mean(dim=(0, 1))  # (n_nodes,)
    r_null = activity_null.mean(dim=(0, 1))
    dsi = (r_pref - r_null) / (r_pref + r_null + eps)
    if target_dsi is not None:
        return F.mse_loss(dsi, target_dsi)
    return -dsi.mean()


def knockout_consistency_loss(
    activity_full:    Tensor,
    activity_knockout: Tensor,
    ko_weight: float = 1.0,
) -> Tensor:
    """
    DMN-style knockout (silencing) constraint (method A in project plan).

    Encourages the network output to change in a specific way when a
    given neuron type is silenced, consistent with experimental observations.

    This loss penalises the network if the silenced-run activity is too
    similar to the full-run activity (i.e. the silenced neuron had no effect).

    Parameters
    ----------
    activity_full     : (batch, T, n_output)  full network responses
    activity_knockout : (batch, T, n_output)  responses with neuron(s) silenced
    ko_weight         : relative weight of the knockout vs full-run terms
    """
    # We want the knockout to produce a measurable change: penalise similarity
    similarity = F.cosine_similarity(
        activity_full.flatten(1),
        activity_knockout.flatten(1),
        dim=1,
    ).mean()  # ranges [-1, 1]
    return ko_weight * similarity  # minimise similarity ⟹ encourage divergence


# ---------------------------------------------------------------------------
# Composite loss
# ---------------------------------------------------------------------------

def combined_loss(
    loss_dict: Dict[str, Tuple[Tensor, float]],
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """
    Weighted sum of multiple losses.

    Parameters
    ----------
    loss_dict : dict mapping name -> (loss_tensor, weight)

    Returns
    -------
    total : scalar Tensor
    breakdown : dict of individual (unweighted) losses for logging
    """
    total = torch.tensor(0.0)
    breakdown = {}
    for name, (loss_val, weight) in loss_dict.items():
        total = total + weight * loss_val
        breakdown[name] = loss_val.detach()
    return total, breakdown


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

LossRegistry: Dict[str, callable] = {
    "mse":                    mse_loss,
    "correlation":            correlation_loss,
    "spike_rate":             spike_rate_loss,
    "direction_selectivity":  direction_selectivity_loss,
    "knockout":               knockout_consistency_loss,
}
