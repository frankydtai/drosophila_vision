# -*- coding: utf-8 -*-
"""Shared helpers for ``experiment`` mirror-fit experiment scripts (no plotting)."""
from __future__ import annotations

from default_params import (
    NEURON_SCHEMA_DEFAULT,
)

import train
from default_params import NEURON_SCHEMA_DEFAULT['h_cells']


def merge_i_h_train_modes(train_kw):
    """Pop CLI ``train_modes`` from *train_kw* and merge default i_h/hp splits.

    Borst: ``a_h`` / ``a_h_rev``.
    hp_lp: ``a_h`` / ``tau_hp``.
    Indi cells: :data:`NEURON_SCHEMA_DEFAULT['h_cells']`; ``fixed=['all']``.
    """
    names = ('a_h', 'a_h_rev') if train_kw['model'] == 'borst' else ('a_h', 'tau_hp')
    existing = train_kw.pop('train_modes', None) or {}
    return {
        **existing,
        **{name: {'indi': list(NEURON_SCHEMA_DEFAULT['h_cells']), 'fixed': ['all']} for name in names},
    }


def spot_tasks_from(tasks):
    """Return tasks in *tasks* that are spot tasks."""
    return [t for t in tasks if t in train.SPOT_TASKS]


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
    fits = _normalize_mirror_fits(mirror_fits, mirror_sign)
    return {t: {'mirror_fits': fits} for t in spot_tasks_from(tasks)}
