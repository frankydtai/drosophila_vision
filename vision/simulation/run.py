#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train-then-plot orchestrator (above ``training``, ``figure``, ``analyze``).

Dependency direction:

    run.py  →  training  (pure train + artifacts)
    run.py  →  figure    (PNG / checkpoint figures)
    run.py  →  analyze   (optional post-plot, e.g. syn_sign)
    training  ✗→  figure

Usage (from ``simulation/``, project ``.venv``):

    ../.venv/bin/python run.py --model hp_lp --nofsteps 30 --lrs 0.1
    ../.venv/bin/python run.py --task spot_bright --network right_min_neuron1_extent2 \\
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
from import_bootstrap import parse_bool
import training.implement as train
from figure.plot_run import (
    add_plot_arguments,
    plot_kwargs_from_args,
    plot_param_set,
)

_CHECKPOINT_PNG_STEMS = (
    "spot_gt_ca",
    "spot_gt_v",
    "spot_all_ca",
    "spot_all_v",
    "bar_gt_ca",
    "bar_gt_v",
    "bar_all_ca",
    "bar_all_v",
)

_PLOT_KEYS = (
    "plot_right_only",
    "show_pre",
    "center_only",
    "at_x",
    "at_y",
    "align_at_x",
    "align_at_y",
    "html",
    "ms_shown",
)


def _take_plot_kw(kw, *, gt_cubes=None):
    """Pop plot keys from *kw* (values come from CLI via ``plot_kwargs_from_args``)."""
    plot_kw = {k: kw.pop(k) for k in _PLOT_KEYS}
    plot_kw["gt_cubes"] = gt_cubes
    return plot_kw


def make_plots(outdir, session, result=None, **plot_kw):
    """Cost curve + model-vs-gt + all-cells."""
    if result is not None:
        plot_param_set(
            result.all_params,
            outdir,
            session=session,
            final_costs=result.final_costs,
            cost_curve=result.cost_curve,
            costs_by_part=result.cost_curves_by_part,
            save_artifacts=False,
            **plot_kw,
        )
        return
    z = train.load_best_param(outdir, session)
    final_costs, cost_curve, costs_by_part, _ = train.load_stored_costs(outdir)
    best_cost = None
    if final_costs is not None and len(final_costs) > 0:
        best_cost = float(final_costs[int(np.argmin(final_costs))])
    plot_param_set(
        np.atleast_2d(z),
        outdir,
        session=session,
        final_costs=np.array([best_cost]) if best_cost is not None else None,
        cost_curve=cost_curve,
        costs_by_part=costs_by_part,
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


def run_training_and_plot(*, plot_gt_cubes=None, **kw):
    """Train (``training.implement.run_training``) then plot. Returns ``(fname, outdir, session)``."""
    syn_sign = kw.pop("syn_sign")
    plot_kw = _take_plot_kw(kw, gt_cubes=plot_gt_cubes)
    checkpoint_on_png = None
    if kw.get("checkpoint_interval") is not None:
        checkpoint_on_png = make_checkpoint_on_png(plot_kw)
    fname, outdir, session, result = train.run_training(
        **kw,
        checkpoint_on_png=checkpoint_on_png,
    )
    make_plots(outdir, session, result=result, **plot_kw)
    if syn_sign:
        from analyze.syn_sign import write_syn_sign_plots
        write_syn_sign_plots(outdir)
    return fname, outdir, session

def make_run_argparser(description=None):
    """Training CLI + plot flags."""
    description = description or __doc__
    common = argparse.ArgumentParser(add_help=False)
    train.add_training_arguments(common)
    add_plot_arguments(common)
    common.add_argument(
        '--syn-sign',
        nargs='?',
        const=True,
        default=False,
        type=parse_bool,
        metavar='BOOL',
        help='after plots, write pre_syn/syn_gt.png and syn_all.png',
    )
    return argparse.ArgumentParser(
        description=description,
        parents=[common],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )


def run_kwargs_from_args(args, *, script_stem="run"):
    """Merge training kwargs with plot kwargs for :func:`run_training_and_plot`."""
    train_kw = train.training_kwargs_from_args(args, script_stem=script_stem)
    train_kw.update(plot_kwargs_from_args(args))
    train_kw['syn_sign'] = args.syn_sign
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
    from param_defaults import DELTA_MS, DELTA_MS_PRE, MS_PRE, MS_RESPONSE
    from figure.readout import fit_gt_cubes
    from neuron.params import ms_to_t
    from training.experiment import (
        merge_i_h_train_modes,
        spot_pack_overrides,
        spot_tasks_from,
        _normalize_mirror_fits,
    )

    def make_mirror_gt_cubes(fits, sign):
        specs = _normalize_mirror_fits(fits, sign)
        t_onset = ms_to_t(MS_PRE, delta_ms=DELTA_MS_PRE)
        n_t = t_onset + ms_to_t(MS_RESPONSE, delta_ms=DELTA_MS) + 1

        def mirror_gt_cubes(contrasts):
            base = fit_gt_cubes(
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

        return mirror_gt_cubes

    def resolve_spot_plot_gt_cubes(spot_tasks, mirror_gt_cubes):
        contrasts = []
        if "spot_bright" in spot_tasks:
            contrasts.append("bright")
        if "spot_dark" in spot_tasks:
            contrasts.append("dark")
        if not contrasts:
            return None
        return mirror_gt_cubes(tuple(contrasts))

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
    tasks = run_kw["tasks"]
    pack_overrides = spot_pack_overrides(tasks, fits, mirror_sign)
    train_modes = merge_i_h_train_modes(run_kw)
    spot_tasks = spot_tasks_from(tasks)
    plot_gt_cubes = resolve_spot_plot_gt_cubes(
        spot_tasks, make_mirror_gt_cubes(fits, mirror_sign),
    )

    fname, outdir, session = run_training_and_plot(
        **run_kw,
        pack_overrides=pack_overrides,
        train_modes=train_modes,
        plot_gt_cubes=plot_gt_cubes,
    )
    for tname in spot_tasks:
        print(f"{tname} cost nodes:", int(session.pack_for(tname).readout_node.shape[0]))
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
