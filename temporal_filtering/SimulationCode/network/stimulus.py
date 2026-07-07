# -*- coding: utf-8 -*-
"""Shared column helpers for training targets."""
from __future__ import annotations


def column_in_cost_extent(u, v, cost_extent=None) -> bool:
    """True when axial ``(u, v)`` lies in the cost hex disc (``None`` = all columns)."""
    if cost_extent is None:
        return True
    import column_mapper
    return bool(column_mapper.inside_mask(int(u), int(v), int(cost_extent)))
