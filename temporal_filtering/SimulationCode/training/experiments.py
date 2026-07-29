# -*- coding: utf-8 -*-
"""Shared helpers for ``add_data`` mirror-fit experiment scripts."""
from __future__ import annotations

import training as fc
import training.train as train
from neuron_model.param_defaults import DEFAULT_IH_GMAX_INDI_NAMES
from plot.readout import fit_ref_cubes


def merge_ih_param_partitions(train_kw):
    """Pop CLI ``param_partitions`` from *train_kw* and merge default Ih/hp splits.

    Conductance: ``Ih_gmax`` / ``Ih_gmax_off``.
    hp_lp: ``hp_gain`` / ``tau_hp``.
    Indi names: :data:`DEFAULT_IH_GMAX_INDI_NAMES`; ``fixed=['all']``.
    """
    ih_indi = list(DEFAULT_IH_GMAX_INDI_NAMES)
    if train_kw['model'] == 'conductance':
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
    return [t for t in target_list if t in fc.SPOT_TARGETS]


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
    """``pack_overrides`` map: each spot target → ``mirror_fits`` override."""
    mirror = {'mirror_fits': _normalize_mirror_fits(mirror_fits, mirror_sign)}
    return {t: dict(mirror) for t in spot_targets_from(target_list)}


def make_mirror_ref_cubes(mirror_fits, mirror_sign=-1.0):
    """Return ``mirror_ref_cubes(dark=...)`` from *mirror_fits* specs."""
    specs = _normalize_mirror_fits(mirror_fits, mirror_sign)

    def mirror_ref_cubes(dark=False):
        ref = fit_ref_cubes(dark=dark)
        for spec in specs:
            src = ref[spec['mirror_fit']]
            sign = spec['mirror_sign']
            for name in spec['mirror_types']:
                ref[name] = sign * src
        return ref

    return mirror_ref_cubes


def resolve_spot_plot_ref_cubes(spot_targets, mirror_ref_cubes):
    """Pick ``plot_ref_cubes`` / ``plot_ref_cubes_2`` from bright/dark spot targets."""
    plot_ref_cubes = plot_ref_cubes_2 = None
    if 'spot_bright' in spot_targets and 'spot_dark' in spot_targets:
        plot_ref_cubes = mirror_ref_cubes(dark=False)
        plot_ref_cubes_2 = mirror_ref_cubes(dark=True)
    elif 'spot_dark' in spot_targets:
        plot_ref_cubes = mirror_ref_cubes(dark=True)
    elif 'spot_bright' in spot_targets:
        plot_ref_cubes = mirror_ref_cubes(dark=False)
    return plot_ref_cubes, plot_ref_cubes_2


def run_mirror_spot_experiment(
    description,
    script_stem,
    mirror_fits,
    *,
    mirror_sign=-1.0,
    configure_parser=None,
):
    """CLI entry for spot mirror-fit experiments.

    *script_stem* / *mirror_fits*: value or ``callable(args) ->``.
    *mirror_fits* entries: ``{mirror_types, mirror_fit[, mirror_sign]}``.
    *configure_parser*: optional ``callable(ap)`` to add experiment-specific args.
    """
    ap = train.make_training_argparser(description)
    if configure_parser is not None:
        configure_parser(ap)
    args = ap.parse_args()
    stem = script_stem(args) if callable(script_stem) else script_stem
    try:
        train_kw = train.training_kwargs_from_args(args, script_stem=stem)
    except ValueError as exc:
        ap.error(str(exc))

    fits = mirror_fits(args) if callable(mirror_fits) else mirror_fits
    target_list = train_kw['target_list']
    pack_overrides = spot_pack_overrides(target_list, fits, mirror_sign)
    param_partitions = merge_ih_param_partitions(train_kw)
    spot_targets = spot_targets_from(target_list)
    plot_ref_cubes, plot_ref_cubes_2 = resolve_spot_plot_ref_cubes(
        spot_targets, make_mirror_ref_cubes(fits, mirror_sign),
    )

    run_kw = dict(train_kw)
    fname, outdir, session = train.run_training(
        **run_kw,
        pack_overrides=pack_overrides,
        param_partitions=param_partitions,
        plot_ref_cubes=plot_ref_cubes,
        plot_ref_cubes_2=plot_ref_cubes_2,
    )
    for tname in spot_targets:
        print(f'{tname} cost cells:', int(session.pack_for(tname).readout_unit.shape[0]))
    print('done ->', outdir)
    return fname, outdir, session
