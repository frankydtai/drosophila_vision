#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train-then-plot orchestrator (above ``train``, ``figure``, ``analyze``).

Dependency direction:

    run.py  →  train  (pure train + data)
    run.py  →  figure    (PNG / checkpoint figures)
    run.py  →  analyze   (optional post-plot, e.g. syn_sign)
    train  ✗→  figure

Usage (from ``simulation/``, project ``.venv``):

    ../.venv/bin/python run.py --model hp_lp --n-iter 30 --lrs 0.1
    ../.venv/bin/python run.py --task spot --contrast bright --network right_min_neuron1_r2 \\
        --n-iter 5 --lrs 0.1

Re-plot an existing run without train:

    ../.venv/bin/python -m figure.plot <model>/<run>
"""
from __future__ import annotations

import argparse
import inspect
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import import_bootstrap  # noqa: F401
from import_bootstrap import parse_bool
import train
import train.implementation as implementation
import train.cli as cli
from figure.plot import (
    add_figure_arguments,
    resolve_figure_kwargs,
    plot_rf_t,
)
from figure.panel import plot_cost, figure_file_ext

_CHECKPOINT_PNG_STEM_PREFIXES = (
    "spot_gt",
    "spot_all",
    "bar_gt",
    "bar_all",
)

_FIGURE_KEYS = (
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


def _take_figure_kwargs(kwargs, *, gts=None):
    """Pop figure keys from *kwargs* (values come from CLI via ``resolve_figure_kwargs``)."""
    figure_kwargs = {k: kwargs.pop(k) for k in _FIGURE_KEYS}
    figure_kwargs["gts"] = gts
    return figure_kwargs


def plot_figures(outdir, session, result=None, **figure_kwargs):
    """Cost curve (train path only) + model-vs-gt + all-cells."""
    if result is not None:
        if result.cost_curve is not None and len(result.cost_curve) > 0:
            plot_cost(
                result.cost_curve,
                os.path.join(outdir, f'cost_curve{figure_file_ext(html=figure_kwargs.get("html"))}'),
                costs_by_part=result.cost_curves_by_part,
                part_order=list(train.session_cost_part_keys(session.tasks, session=session)),
            )
        plot_rf_t(
            result.run_params,
            outdir,
            session=session,
            final_costs=result.final_costs,
            save_data=False,
            **figure_kwargs,
        )
        return
    z = implementation.load_best_param(outdir, session)
    final_costs, _, _, _ = implementation.load_stored_costs(outdir)
    best_cost = None
    if final_costs is not None and len(final_costs) > 0:
        best_cost = float(final_costs[int(np.argmin(final_costs))])
    plot_rf_t(
        np.atleast_2d(z),
        outdir,
        session=session,
        final_costs=np.array([best_cost]) if best_cost is not None else None,
        save_data=False,
        **figure_kwargs,
    )




def _rename_checkpoint_pngs(png_dir, tag, *, filter_figure="v", file_suffix=""):
    for prefix in _CHECKPOINT_PNG_STEM_PREFIXES:
        stem = f"{prefix}_{filter_figure}"
        src = os.path.join(png_dir, f"{stem}{file_suffix}.png")
        if os.path.isfile(src):
            dst = os.path.join(png_dir, f"{stem}{file_suffix}_{tag}.png")
            os.replace(src, dst)


def save_checkpoint_png(outdir, iter, z_best, cost_best, session, figure_kwargs):
    tag = implementation.checkpoint_iter_tag(iter)
    png_dir = os.path.join(outdir, "png")
    os.makedirs(png_dir, exist_ok=True)
    z = z_best.detach().cpu().numpy()
    plot_rf_t(
        np.array([z]),
        png_dir,
        session=session,
        final_costs=np.array([cost_best]),
        save_data=False,
        **figure_kwargs,
    )
    from figure.panel import filter_figure
    _rename_checkpoint_pngs(
        png_dir, tag,
        filter_figure=filter_figure((session.train_opts or {}).get("filter")),
        file_suffix=figure_kwargs.get("file_suffix") or "",
    )
    print(f"wrote checkpoint png: {png_dir}/*_{tag}.png")


def build_checkpoint_on_png(figure_kwargs):
    """Bind *figure_kwargs* into a ``checkpoint_on_png`` callback for ``run_train``."""
    figure_kwargs = figure_kwargs or {}

    def on_png(outdir, iter, z_best, cost_best, session):
        save_checkpoint_png(outdir, iter, z_best, cost_best, session, figure_kwargs)

    return on_png


def run_train_and_plot(*, figure_gts=None, **kwargs):
    """Run train (``train.implementation.run_train``) then plot. Returns ``(fname, outdir, session)``."""
    syn_sign = kwargs.pop("syn_sign")
    figure_kwargs = _take_figure_kwargs(kwargs, gts=figure_gts)
    checkpoint_on_png = None
    if kwargs.get("checkpoint_interval") is not None:
        checkpoint_on_png = build_checkpoint_on_png(figure_kwargs)
    fname, outdir, session, result = implementation.run_train(
        **kwargs,
        checkpoint_on_png=checkpoint_on_png,
    )
    plot_figures(outdir, session, result=result, **figure_kwargs)
    if syn_sign:
        from analyze.syn_sign import save_syn_sign_figures
        save_syn_sign_figures(outdir)
    return fname, outdir, session

def build_run_argparser(description=None):
    """Train CLI + plot flags."""
    description = description or __doc__
    common = argparse.ArgumentParser(add_help=False)
    cli.add_train_arguments(common)
    add_figure_arguments(common)
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


def resolve_run_kwargs(args, *, script_stem="run"):
    """Merge train kwargs with figure kwargs for :func:`run_train_and_plot`."""
    train_kwargs = cli.resolve_train_kwargs(args, script_stem=script_stem)
    train_kwargs.update(resolve_figure_kwargs(args))
    train_kwargs['syn_sign'] = args.syn_sign
    return train_kwargs


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    print(f"cli: {' '.join(argv)}")
    parser = build_run_argparser()
    args = parser.parse_args(argv)
    try:
        kwargs = resolve_run_kwargs(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    run_train_and_plot(**kwargs)


if __name__ == "__main__":
    main()
