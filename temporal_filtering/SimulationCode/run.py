#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train-then-plot orchestrator (above ``training`` and ``plot``).

Dependency direction:

    6_run.py  →  training  (pure train + artifacts)
    6_run.py  →  plot      (PNG / checkpoint figures)
    training  ✗→  plot

Usage (from ``SimulationCode/``, project ``.venv``):

    ../.venv/bin/python 6_run.py --model hp_lp --nofsteps 30 --lrs 0.1
    ../.venv/bin/python 6_run.py --target spot_bright --network right_min_neuron1_extent2 \\
        --nofsteps 5 --lrs 0.1

Re-plot an existing run without training:

    ../.venv/bin/python -m figure.plot_run <model>/<run_name>
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import import_bootstrap  # noqa: F401
import training.driver as train
from figure.plot_run import (
    add_plot_arguments,
    plot_kwargs_from_args,
    plot_param_set,
)

_CHECKPOINT_PNG_STEMS = (
    "spot_trained_ca",
    "spot_trained_v",
    "spot_all_ca",
    "spot_all_v",
    "bar_trained_ca",
    "bar_trained_v",
    "bar_all_ca",
    "bar_all_v",
)


def build_plot_kwargs(
    *,
    data_cubes=None,
    plot_right_only=True,
    at_x=None,
    at_y=None,
    align_at_x=None,
    align_at_y=None,
    show_pre=True,
):
    return dict(
        data_cubes=data_cubes,
        plot_right_only=plot_right_only,
        at_x=at_x,
        at_y=at_y,
        align_at_x=align_at_x,
        align_at_y=align_at_y,
        show_pre=show_pre,
    )


def make_plots(
    fname,
    outdir,
    session,
    result=None,
    *,
    data_cubes=None,
    plot_right_only=True,
    at_x=None,
    at_y=None,
    align_at_x=None,
    align_at_y=None,
    show_pre=True,
):
    """Cost curve + model-vs-data + all-cell-types."""
    plot_kw = build_plot_kwargs(
        data_cubes=data_cubes,
        plot_right_only=plot_right_only,
        at_x=at_x,
        at_y=at_y,
        align_at_x=align_at_x,
        align_at_y=align_at_y,
        show_pre=show_pre,
    )
    if result is not None:
        plot_param_set(
            result.all_params,
            outdir,
            session=session,
            final_costs=result.final_costs,
            cost_curve=result.cost_curve,
            costs_by_target=result.cost_curves_by_target,
            best_i=result.best_i,
            save_artifacts=False,
            **plot_kw,
        )
        return
    params = np.load(train.params_path(outdir, fname))
    final_costs, cost_curve, costs_by_target, _ = train.load_stored_costs(
        outdir, fname, np.atleast_2d(params).shape[0],
    )
    plot_param_set(
        params,
        outdir,
        session=session,
        final_costs=final_costs,
        cost_curve=cost_curve,
        costs_by_target=costs_by_target,
        save_artifacts=False,
        **plot_kw,
    )


def _rename_checkpoint_pngs(png_dir, tag):
    for stem in _CHECKPOINT_PNG_STEMS:
        src = os.path.join(png_dir, f"{stem}.png")
        if os.path.isfile(src):
            dst = os.path.join(png_dir, f"{stem}_{tag}.png")
            os.replace(src, dst)


def write_checkpoint_png(outdir, step, z_best, cost_best, session, plot_kw):
    tag = train.checkpoint_step_tag(step)
    png_dir = os.path.join(outdir, "png")
    os.makedirs(png_dir, exist_ok=True)
    z_np = z_best.detach().cpu().numpy()
    plot_param_set(
        np.array([z_np]),
        png_dir,
        session=session,
        final_costs=np.array([cost_best]),
        best_i=0,
        save_artifacts=False,
        cost_curve=None,
        **plot_kw,
    )
    _rename_checkpoint_pngs(png_dir, tag)
    print(f"wrote checkpoint png: {png_dir}/*_{tag}.png")


def make_checkpoint_on_png(plot_kw):
    """Bound *plot_kw* into a ``checkpoint_on_png`` callback for ``run_training``."""
    plot_kw = plot_kw or {}

    def on_png(outdir, step, z_best, cost_best, session):
        write_checkpoint_png(outdir, step, z_best, cost_best, session, plot_kw)

    return on_png


def run_training_and_plot(
    *,
    plot_data_cubes=None,
    plot_right_only=True,
    at_x=None,
    at_y=None,
    align_at_x=None,
    align_at_y=None,
    show_pre=True,
    **train_kw,
):
    """Train (``training.driver.run_training``) then plot. Returns ``(fname, outdir, session)``."""
    plot_kw = build_plot_kwargs(
        data_cubes=plot_data_cubes,
        plot_right_only=plot_right_only,
        at_x=at_x,
        at_y=at_y,
        align_at_x=align_at_x,
        align_at_y=align_at_y,
        show_pre=show_pre,
    )
    checkpoint_on_png = None
    if train_kw.get("checkpoint_interval") is not None:
        checkpoint_on_png = make_checkpoint_on_png(plot_kw)
    fname, outdir, session, result = train.run_training(
        **train_kw,
        checkpoint_on_png=checkpoint_on_png,
    )
    make_plots(
        fname,
        outdir,
        session,
        result=result,
        **plot_kw,
    )
    return fname, outdir, session


def make_run_argparser(description=None):
    """Training CLI + plot flags."""
    description = description or __doc__
    common = argparse.ArgumentParser(add_help=False)
    train.add_training_arguments(common)
    add_plot_arguments(common)
    return argparse.ArgumentParser(
        description=description,
        parents=[common],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )


def run_kwargs_from_args(args, *, script_stem="run"):
    """Merge training kwargs with plot kwargs for :func:`run_training_and_plot`."""
    train_kw = train.training_kwargs_from_args(args, script_stem=script_stem)
    train_kw.update(plot_kwargs_from_args(args))
    return train_kw


def run_mirror_spot_experiment(
    description,
    script_stem,
    mirror_fits,
    *,
    mirror_sign=-1.0,
    configure_parser=None,
):
    """CLI entry for spot mirror-fit experiments (train + plot)."""
    from training.defaults import DELTA_MS, PRE_MS, RESPONSE_MS
    from figure.readout import fit_data_cubes
    from neuron.params import ms_to_t
    from training.experiment import (
        merge_ih_param_partitions,
        spot_pack_overrides,
        spot_targets_from,
        _normalize_mirror_fits,
    )

    def make_mirror_data_cubes(fits, sign):
        specs = _normalize_mirror_fits(fits, sign)
        t_onset = ms_to_t(PRE_MS, delta_ms=DELTA_MS)
        n_t = t_onset + ms_to_t(RESPONSE_MS, delta_ms=DELTA_MS) + 1

        def mirror_data_cubes(contrasts):
            base = fit_data_cubes(
                contrasts=contrasts,
                t_onset=t_onset,
                n_t=n_t,
                delta_ms=DELTA_MS,
            )
            out = {}
            for contrast, cells in base.items():
                cells = dict(cells)
                for spec in specs:
                    src = cells[spec["mirror_fit"]]
                    s = spec["mirror_sign"]
                    for name in spec["mirror_types"]:
                        cells[name] = s * src
                out[contrast] = cells
            return out

        return mirror_data_cubes

    def resolve_spot_plot_data_cubes(spot_targets, mirror_data_cubes):
        contrasts = []
        if "spot_bright" in spot_targets:
            contrasts.append("bright")
        if "spot_dark" in spot_targets:
            contrasts.append("dark")
        if not contrasts:
            return None
        return mirror_data_cubes(tuple(contrasts))

    ap = make_run_argparser(description)
    if configure_parser is not None:
        configure_parser(ap)
    args = ap.parse_args()
    stem = script_stem(args) if callable(script_stem) else script_stem
    try:
        run_kw = run_kwargs_from_args(args, script_stem=stem)
    except ValueError as exc:
        ap.error(str(exc))

    fits = mirror_fits(args) if callable(mirror_fits) else mirror_fits
    target_list = run_kw["target_list"]
    pack_overrides = spot_pack_overrides(target_list, fits, mirror_sign)
    param_partitions = merge_ih_param_partitions(run_kw)
    spot_targets = spot_targets_from(target_list)
    plot_data_cubes = resolve_spot_plot_data_cubes(
        spot_targets, make_mirror_data_cubes(fits, mirror_sign),
    )

    fname, outdir, session = run_training_and_plot(
        **run_kw,
        pack_overrides=pack_overrides,
        param_partitions=param_partitions,
        plot_data_cubes=plot_data_cubes,
    )
    for tname in spot_targets:
        print(f"{tname} cost cells:", int(session.pack_for(tname).readout_unit.shape[0]))
    print("done ->", outdir)
    return fname, outdir, session


def main(argv=None):
    parser = make_run_argparser()
    args = parser.parse_args(argv)
    try:
        kw = run_kwargs_from_args(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    run_training_and_plot(**kw)


if __name__ == "__main__":
    main()
