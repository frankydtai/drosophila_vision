# -*- coding: utf-8 -*-
"""The one Ca readout filter and its exact inverse (shared by all models).

Forward (``ca_readout_step``) is a first-order low-pass on ``v - v_ref`` with
``alpha = deltat / Ca_tau``. ``ca_to_v_delta`` is its algebraic inverse, used to
turn a Ca-proxy trace (ImpR-based target, or a plotted reference cube) back into
delta-Vm. This is the ONLY Ca filter in the codebase; spot data bandpass/lowpass
(ImpR shaping) is a different, target-only signal path and lives in
``task.spot.data``.
"""
from __future__ import annotations

from neuron_model.param import Ca_tau, deltat

# One-step low-pass coefficient: ca[t] = (1-alpha)*ca[t-1] + alpha*(v[t]-v_ref).
CA_ALPHA = deltat / Ca_tau  # = 0.2 at deltat=10 ms, Ca_tau=50 ms


def ca_readout_step(ca, v, v_ref):
    """One Ca low-pass step on ``v - v_ref`` (shared by all models)."""
    return deltat / Ca_tau * (v - v_ref - ca) + ca


def ca_to_v_delta(ca, *, t_on=0):
    """Invert the Ca low-pass: recover ``v - v_ref`` from a Ca-proxy trace.

    ``ca`` has time on the last axis. ``v_delta[t] = (ca[t] - (1-alpha)*ca[t-1])
    / alpha`` with ``ca[t_on-1] = 0`` (the integrator resets at ``t_on``).

    ``t_on`` is the absolute onset index for full-length traces (entries before
    it are forced to 0). For post-onset arrays already sliced at the onset, pass
    ``t_on=0`` (default): index 0 then uses ``ca[-1]=0``.

    Works on numpy arrays and torch tensors.
    """
    alpha = CA_ALPHA
    t_on = int(t_on)
    try:
        import torch
        is_torch = torch.is_tensor(ca)
    except ImportError:  # pragma: no cover - torch always present in this repo
        is_torch = False

    if is_torch:
        prev = torch.zeros_like(ca)
        prev[..., 1:] = ca[..., :-1]
    else:
        import numpy as np
        ca = np.asarray(ca, dtype=np.float64)
        prev = np.zeros_like(ca)
        prev[..., 1:] = ca[..., :-1]

    if t_on > 0:
        prev[..., t_on] = 0.0  # no carry across the onset reset

    v_delta = (ca - (1.0 - alpha) * prev) / alpha

    if t_on > 0:
        v_delta[..., :t_on] = 0.0
    return v_delta
