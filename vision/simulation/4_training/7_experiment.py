# -*- coding: utf-8 -*-
"""Shared helpers for ``experiment`` mirror-fit experiment scripts (no plotting)."""
from __future__ import annotations

import training
from param_defaults import H_CELLS


def merge_i_h_train_modes(train_kw):
    """Pop CLI ``train_modes`` from *train_kw* and merge default i_h/hp splits.

    Borst: ``h_g_max`` / ``h_g_max_off``.
    hp_lp: ``a_slow`` / ``tau_hp``.
    Indi cells: :data:`H_CELLS`; ``fixed=['all']``.
    """
    i_h_indi = list(H_CELLS)
    if train_kw['model'] == 'borst':
        names = ('h_g_max', 'h_g_max_off')
    else:
        names = ('a_slow', 'tau_hp')
    i_h_modes = {
        name: {'indi': i_h_indi, 'fixed': ['all']}
        for name in names
    }
    existing = train_kw.pop('train_modes', None) or {}
    return {**existing, **i_h_modes}


def spot_tasks_from(tasks):
    """Return tasks in *tasks* that are spot tasks."""
    return [t for t in tasks if t in training.SPOT_TASKS]


def _normalize_mirror_fits(mirror_fits, mirror_sign):
    return [
        {
            'mirror_types': list(spec['mirror_types']),
            'mirror_fit': spec['mirror_fit'],
            'mirror_sign': float(spec.get('mirror_sign', mirror_sign)),
        }
        for spec in mirror_fits
    ]


def spot_pack_overrides(tasks, mirror_fits, mirror_sign=-1.0):
    """``{spot_task: mirror_fits override}`` for each spot task in *tasks*."""
    mirror = {'mirror_fits': _normalize_mirror_fits(mirror_fits, mirror_sign)}
    return {t: dict(mirror) for t in spot_tasks_from(tasks)}
