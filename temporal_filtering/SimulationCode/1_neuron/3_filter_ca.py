# -*- coding: utf-8 -*-
"""The one Ca filter (kept; unused by forward / training / plotting).

``ca_filter`` is a first-order low-pass on ``v - v_onset`` with
``alpha = delta_ms / Ca_tau``. Training and plots use ``v`` (``v - v_onset``)
directly; ImpR / RecF targets are used as-is (no Ca conversion).

``delta_ms`` / ``Ca_tau`` are injected scalars.
"""
from __future__ import annotations


def ca_alpha(*, delta_ms: float, Ca_tau: float) -> float:
    """One-step low-pass coefficient ``delta_ms / Ca_tau``."""
    return float(delta_ms) / float(Ca_tau)


def ca_filter(ca, v, v_onset, *, delta_ms: float, Ca_tau: float):
    """One Ca low-pass step on ``v - v_onset`` (unused by current forward)."""
    return float(delta_ms) / float(Ca_tau) * (v - v_onset - ca) + ca
