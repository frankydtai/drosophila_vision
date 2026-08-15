# -*- coding: utf-8 -*-
"""Shared helpers for ``experiment`` mirror-fit experiment scripts (no plotting)."""
from __future__ import annotations

from default_params import (
    NEURON_SCHEMA,
)

import train


def resolve_i_h_param_modes(train_kw):
    """Pop CLI ``param_modes`` from *train_kw* and merge default i_h/hp splits.

    Borst: ``a_h`` / ``a_h_rev``.
    hp_lp: ``a_h`` / ``tau_hp_rise`` / ``tau_hp_fall``.
    Later tokens win: ``fixed`` then ``h_cells`` → ``indi``.
    """
    segments = (
        ("a_h", "a_h_rev")
        if train_kw["model"] == "borst"
        else ("a_h", "tau_hp_rise", "tau_hp_fall")
    )
    existing = train_kw.pop('param_modes', None) or {}
    h_cells = list(NEURON_SCHEMA['h_cells'])
    return {
        **existing,
        **{
            segment: [(None, "fixed"), (h_cells, "indi")]
            for segment in segments
            if segment not in existing
        },
    }


def spot_tasks_from(tasks):
    """Return tasks in *tasks* that are spot tasks."""
    return [t for t in tasks if t in train.SPOT_TASKS]


def normalize_mirror_fit(mirror_fit, mirror_sign):
    specs = mirror_fit if isinstance(mirror_fit, list) else [mirror_fit]
    return [
        {
            'mirror_types': list(spec['mirror_types']),
            'mirror_fit': spec['mirror_fit'],
            'mirror_sign': float(spec.get('mirror_sign', mirror_sign)),
        }
        for spec in specs
    ]


def spot_pack_mirror_fits(tasks, mirror_fit, mirror_sign=-1.0):
    """``{spot_task: pack_mirror_fit}`` for each spot task in *tasks*."""
    fits = normalize_mirror_fit(mirror_fit, mirror_sign)
    return {t: {'mirror_fit': fits} for t in spot_tasks_from(tasks)}
