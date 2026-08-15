# -*- coding: utf-8 -*-
"""The one Ca filter.

``filter_ca`` is a first-order low-pass on ``v_ca`` with
``dt_over_tau_ca = delta_ms / tau_ca`` (same ``Δt/τ`` pattern as
``dt_over_tau_lp`` / ``membrane_dt_over_c``). Output state is ``ca``.

``delta_ms`` is an injected scalar; ``tau_ca`` may be a schema tensor
(``p["tau_ca"]``) so the ratio stays in the graph.
"""
from __future__ import annotations


def filter_ca(ca, v_ca, *, delta_ms: float, tau_ca):
    """One Ca low-pass step: ``ca ← ca + (delta_ms/tau_ca) (v_ca − ca)``."""
    dt_over_tau_ca = float(delta_ms) / tau_ca
    return dt_over_tau_ca * (v_ca - ca) + ca
