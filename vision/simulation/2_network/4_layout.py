# -*- coding: utf-8 -*-
"""Shared column helpers for training targets."""
from __future__ import annotations


def normalize_cost_extent(cost_extent=None):
    """``None`` or ``-1`` → unrestricted (all columns); else non-negative int."""
    if cost_extent is None:
        return None
    v = int(cost_extent)
    if v == -1:
        return None
    return v


def column_in_cost_extent(u, v, cost_extent=None) -> bool:
    """True when axial ``(u, v)`` lies in the cost hex disc (``None`` = all columns)."""
    cost_extent = normalize_cost_extent(cost_extent)
    if cost_extent is None:
        return True
    import build_hex
    return bool(build_hex.inside_mask(int(u), int(v), int(cost_extent)))
