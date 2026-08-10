# -*- coding: utf-8 -*-
"""The one Ca filter.

``filter_ca`` is a first-order low-pass on ``v_ca`` with
``dt_over_tau_ca = delta_ms / tau_ca`` (same pattern as ``hp_dt_over_tau`` /
``lp_dt_over_tau`` / ``membrane_dt_over_c``). Output state is ``f_ca``.

``delta_ms`` is an injected scalar; ``tau_ca`` may be a schema tensor
(``p["tau_ca"]``) so the ratio stays in the graph.
"""
from __future__ import annotations


def filter_ca(f_ca, v_ca, *, delta_ms: float, tau_ca):
    """One Ca low-pass step: ``f_ca ← f_ca + (delta_ms/tau_ca) (v_ca − f_ca)``."""
    dt_over_tau_ca = float(delta_ms) / tau_ca
    return dt_over_tau_ca * (v_ca - f_ca) + f_ca
