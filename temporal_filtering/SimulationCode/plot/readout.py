#!/usr/bin/env python
"""Model-data readout cell selection from session + target."""

from __future__ import annotations

import numpy as np
import torch

import Medulla_Library as ml
from network.moving_bar_target import BORST_READOUT_SUBTYPES
from t4_t5_preference import READOUT_SUBTYPES

DEFAULT_MVD_GROUPS = [
    np.array(['L1', 'L2', 'L3', 'L4', 'L5']),
    np.array(['Mi1', 'Mi4', 'Mi9']),
    np.array(['Tm1', 'Tm2', 'Tm3', 'Tm4', 'Tm9']),
]


def mvd_groups(groups=None):
    src = DEFAULT_MVD_GROUPS if groups is None else groups
    return [np.asarray(g) for g in src if len(g) > 0]


def _pack_for(session, target):
    if target is None:
        return session.primary_pack
    return session.pack_for(target)


def _type_names_for_units(session, unit_indices):
    u = unit_indices
    if torch.is_tensor(u):
        u = u.detach().cpu().numpy()
    u = np.asarray(u, dtype=np.int64)
    backend = session.backend
    if backend.network is not None:
        C = backend.network
        node_type = C.node_type[u]
        if torch.is_tensor(node_type):
            node_type = node_type.detach().cpu().numpy()
        names = list(C.type_names)
        return [str(names[int(t)]) for t in node_type]
    node_type = backend.conn.node_type[u]
    if torch.is_tensor(node_type):
        node_type = node_type.detach().cpu().numpy()
    ctype = ml.ctype
    return [str(ctype[int(t)]) for t in node_type]


def pack_readout_types(session, target=None):
    """Unique cell-type names on pack.readout_unit, pack order."""
    pack = _pack_for(session, target)
    seen = set()
    out = []
    for name in _type_names_for_units(session, pack.readout_unit):
        if name not in seen:
            seen.add(name)
            out.append(name)
    return tuple(out)


def _mirror_ref_specs_from_override(override):
    if not override:
        return []
    specs = []
    if 'mirror_fits' in override:
        for spec in override['mirror_fits']:
            specs.append((
                [str(t) for t in spec['mirror_types']],
                str(spec['mirror_fit']),
                float(spec.get('mirror_sign', -1.0)),
            ))
    elif 'mirror_fit' in override:
        spec = override['mirror_fit']
        if isinstance(spec, dict) and 'mirror_types' in spec:
            specs.append((
                [str(t) for t in spec['mirror_types']],
                str(spec['mirror_fit']),
                float(spec.get('mirror_sign', -1.0)),
            ))
    return specs


def borst_ref_cubes(dark=False):
    """RecF reference cubes for the 13 Borst fit cell types."""
    data = ml.read_RecF_data_dark() if dark else ml.read_RecF_data()
    ref = data * ml.DATA_AMP
    return {str(name): ref[i] for i, name in enumerate(ml.cell_list)}


def spot_ref_cubes(session, target=None, dark=False):
    """Spot model-data reference cubes from RecF + pack_overrides mirror specs."""
    target = target or session.primary_pack.name
    ref = dict(borst_ref_cubes(dark=dark))
    overrides = (session.train_opts or {}).get('pack_overrides') or {}
    for mirror_types, mirror_fit, mirror_sign in _mirror_ref_specs_from_override(
        overrides.get(target),
    ):
        if mirror_fit not in ref:
            continue
        mirrored = mirror_sign * ref[mirror_fit]
        for name in mirror_types:
            ref[name] = mirrored
    return ref


def _mirror_type_groups_from_override(override, present):
    if not override:
        return []
    groups = []
    if 'mirror_fits' in override:
        for spec in override['mirror_fits']:
            row = [str(t) for t in spec.get('mirror_types', []) if str(t) in present]
            if row:
                groups.append(np.array(row))
    elif 'mirror_fit' in override:
        spec = override['mirror_fit']
        if isinstance(spec, dict) and 'mirror_types' in spec:
            row = [str(t) for t in spec['mirror_types'] if str(t) in present]
            if row:
                groups.append(np.array(row))
    return groups


def spot_model_data_groups(session, target=None, group_list=None):
    """Row groups for spot model-data plots."""
    target = target or session.primary_pack.name
    present = set(pack_readout_types(session, target))
    if group_list is not None:
        out = []
        for g in mvd_groups(group_list):
            row = [str(n) for n in g if str(n) in present]
            if row:
                out.append(np.array(row))
        return out

    out = []
    used = set()
    overrides = (session.train_opts or {}).get('pack_overrides') or {}
    for g in _mirror_type_groups_from_override(overrides.get(target), present):
        out.append(g)
        used.update(g.tolist())

    for g in DEFAULT_MVD_GROUPS:
        row = [str(n) for n in g if str(n) in present and str(n) not in used]
        if row:
            out.append(np.array(row))
            used.update(row)

    for name in pack_readout_types(session, target):
        if name not in used:
            out.append(np.array([name]))
            used.add(name)
    return out


def spot_model_data_names(session, target=None, group_list=None):
    """Flat cell names for network spot model-data grid."""
    names = []
    for g in spot_model_data_groups(session, target, group_list):
        names.extend(str(n) for n in g)
    return names


def moving_bar_row_types(session, target):
    """Row labels for moving-bar model-data plots."""
    present = pack_readout_types(session, target)
    present_set = set(present)
    canonical = (
        READOUT_SUBTYPES
        if session.backend.network is not None
        else BORST_READOUT_SUBTYPES
    )
    ordered = [str(t) for t in canonical if str(t) in present_set]
    used = set(ordered)
    for name in present:
        if name not in used:
            ordered.append(name)
    return tuple(ordered)
