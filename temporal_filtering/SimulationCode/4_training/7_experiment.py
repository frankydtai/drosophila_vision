# -*- coding: utf-8 -*-
"""Shared helpers for ``experiment`` mirror-fit experiment scripts (no plotting)."""
from __future__ import annotations

import training
from training.defaults import IH_GMAX_INDI_NAMES


def merge_ih_param_partitions(train_kw):
    """Pop CLI ``param_partitions`` from *train_kw* and merge default Ih/hp splits.

    Borst: ``Ih_gmax`` / ``Ih_gmax_off``.
    hp_lp: ``hp_gain`` / ``tau_hp``.
    Indi names: :data:`IH_GMAX_INDI_NAMES`; ``fixed=['all']``.
    """
    ih_indi = list(IH_GMAX_INDI_NAMES)
    if train_kw['model'] == 'borst':
        names = ('Ih_gmax', 'Ih_gmax_off')
    else:
        names = ('hp_gain', 'tau_hp')
    ih_parts = {
        name: {'indi': ih_indi, 'fixed': ['all']}
        for name in names
    }
    existing = train_kw.pop('param_partitions', None) or {}
    return {**existing, **ih_parts}


def spot_targets_from(target_list):
    """Return targets in *target_list* that are spot targets."""
    return [t for t in target_list if t in training.SPOT_TARGETS]


def _normalize_mirror_fits(mirror_fits, mirror_sign):
    return [
        {
            'mirror_types': list(spec['mirror_types']),
            'mirror_fit': spec['mirror_fit'],
            'mirror_sign': float(spec.get('mirror_sign', mirror_sign)),
        }
        for spec in mirror_fits
    ]


def spot_pack_overrides(target_list, mirror_fits, mirror_sign=-1.0):
    """``{spot_target: mirror_fits override}`` for each spot target in *target_list*."""
    mirror = {'mirror_fits': _normalize_mirror_fits(mirror_fits, mirror_sign)}
    return {t: dict(mirror) for t in spot_targets_from(target_list)}
