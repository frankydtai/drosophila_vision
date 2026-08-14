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
    ../.venv/bin/python run.py --task spot_bright --network right_min_neuron1_r2 \\
        --n-iter 5 --lrs 0.1

Re-plot an existing run without train:

    ../.venv/bin/python -m figure.plot <model>/<run_name>
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
    add_plot_arguments,
    plot_kwargs_from_args,
    plot_param_set,
)
from figure.util import plot_cost, plot_file_ext

_CHECKPOINT_PNG_STEM_PREFIXES = (
    "spot_gt",
    "spot_all",
    "bar_gt",
    "bar_all",
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


def _take_plot_kw(kw, *, gts=None):
    """Pop plot keys from *kw* (values come from CLI via ``plot_kwargs_from_args``)."""
    plot_kw = {k: kw.pop(k) for k in _PLOT_KEYS}
    plot_kw["gts"] = gts
    return plot_kw


def plot_figures(outdir, session, result=None, **plot_kw):
    """Cost curve (train path only) + model-vs-gt + all-cells."""
    if result is not None:
        if result.cost_curve is not None and len(result.cost_curve) > 0:
            plot_cost(
                result.cost_curve,
                os.path.join(outdir, f'cost_curve{plot_file_ext(html=plot_kw.get("html"))}'),
                costs_by_part=result.cost_curves_by_part,
                part_order=list(train.session_cost_part_keys(session.tasks, session=session)),
            )
        plot_param_set(
            result.run_params,
            outdir,
            session=session,
            final_costs=result.final_costs,
            save_data=False,
            **plot_kw,
        )
        return
    z = implementation.load_best_param(outdir, session)
    final_costs, _, _, _ = implementation.load_stored_costs(outdir)
    best_cost = None
    if final_costs is not None and len(final_costs) > 0:
        best_cost = float(final_costs[int(np.argmin(final_costs))])
    plot_param_set(
        np.atleast_2d(z),
        outdir,
        session=session,
        final_costs=np.array([best_cost]) if best_cost is not None else None,
        save_data=False,
        **plot_kw,
    )




def _rename_checkpoint_pngs(png_dir, tag, *, filter_token="v", file_suffix=""):
    for prefix in _CHECKPOINT_PNG_STEM_PREFIXES:
        stem = f"{prefix}_{filter_token}"
        src = os.path.join(png_dir, f"{stem}{file_suffix}.png")
        if os.path.isfile(src):
            dst = os.path.join(png_dir, f"{stem}{file_suffix}_{tag}.png")
            os.replace(src, dst)


def write_checkpoint_png(outdir, iter, z_best, cost_best, session, plot_kw):
    tag = implementation.checkpoint_iter_tag(iter)
    png_dir = os.path.join(outdir, "png")
    os.makedirs(png_dir, exist_ok=True)
    z_np = z_best.detach().cpu().numpy()
    plot_param_set(
        np.array([z_np]),
        png_dir,
        session=session,
        final_costs=np.array([cost_best]),
        save_data=False,
        **plot_kw,
    )
    from figure.util import filter_plot_token
    _rename_checkpoint_pngs(
        png_dir, tag,
        filter_token=filter_plot_token((session.train_opts or {}).get("filter")),
        file_suffix=plot_kw.get("file_suffix") or "",
    )
    print(f"wrote checkpoint png: {png_dir}/*_{tag}.png")


def build_checkpoint_on_png(plot_kw):
    """Bound *plot_kw* into a ``checkpoint_on_png`` callback for ``run_train``."""
    plot_kw = plot_kw or {}

    def on_png(outdir, iter, z_best, cost_best, session):
        write_checkpoint_png(outdir, iter, z_best, cost_best, session, plot_kw)

    return on_png


def run_train_and_plot(*, plot_gts=None, **kw):
    """Run train (``train.implementation.run_train``) then plot. Returns ``(fname, outdir, session)``."""
    syn_sign = kw.pop("syn_sign")
    plot_kw = _take_plot_kw(kw, gts=plot_gts)
    checkpoint_on_png = None
    if kw.get("checkpoint_interval") is not None:
        checkpoint_on_png = build_checkpoint_on_png(plot_kw)
    fname, outdir, session, result = implementation.run_train(
        **kw,
        checkpoint_on_png=checkpoint_on_png,
    )
    plot_figures(outdir, session, result=result, **plot_kw)
    if syn_sign:
        from analyze.syn_sign import save_syn_sign_plots
        save_syn_sign_plots(outdir)
    return fname, outdir, session

def build_run_argparser(description=None):
    """Train CLI + plot flags."""
    description = description or __doc__
    common = argparse.ArgumentParser(add_help=False)
    cli.add_train_arguments(common)
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
    """Merge train kwargs with plot kwargs for :func:`run_train_and_plot`."""
    train_kw = cli.train_kwargs_from_args(args, script_stem=script_stem)
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
    from task.spot.sti_spec import sti_timing_from_opts
    from figure.gt import fit_gts
    from train.experiment import (
        merge_i_h_param_modes,
        spot_pack_overrides,
        spot_tasks_from,
        _normalize_mirror_fits,
    )

    def build_mirror_gts(fits, sign, session):
        specs = _normalize_mirror_fits(fits, sign)
        def mirror_gts(contrasts):
            pack_name = f"spot_{str(contrasts[0])}"
            timing = sti_timing_from_opts(
                (session.train_opts or {}).get(f"{pack_name}_sti_opts") or {}
            )
            base = fit_gts(
                contrasts=contrasts,
                t_onset=timing.t_onset,
                n_t=timing.n_t_gt,
                delta_ms=session.delta_ms,
                gt_amp=session.gt_amp,
                filter=str((session.train_opts or {}).get("filter", "none")),
                spot_gt_mode=str((session.train_opts or {}).get("spot_gt_mode", "all")),
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

        return mirror_gts

    def resolve_spot_plot_gts(spot_tasks, mirror_gts):
        contrasts = []
        if "spot_bright" in spot_tasks:
            contrasts.append("bright")
        if "spot_dark" in spot_tasks:
            contrasts.append("dark")
        if not contrasts:
            return None
        return mirror_gts(tuple(contrasts))

    ap = build_run_argparser(description)
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
    param_modes = merge_i_h_param_modes(run_kw)
    spot_tasks = spot_tasks_from(tasks)
    build_session_params = inspect.signature(implementation.build_session).parameters
    preview_session = implementation.build_session(
        run_kw["model"],
        **{
            key: val
            for key, val in run_kw.items()
            if key != "model" and key in build_session_params
        },
        pack_overrides=pack_overrides,
        param_modes=param_modes,
    )
    plot_gts = resolve_spot_plot_gts(
        spot_tasks, build_mirror_gts(fits, mirror_sign, preview_session),
    )

    fname, outdir, session = run_train_and_plot(
        **run_kw,
        pack_overrides=pack_overrides,
        param_modes=param_modes,
        plot_gts=plot_gts,
    )
    for tname in spot_tasks:
        print(f"{tname} cost nodes:", int(session.pack_for(tname).entry_nodes.shape[0]))
    return fname, outdir, session


def main(argv=None):
    argv_list = list(sys.argv[1:] if argv is None else argv)
    print(f"cli: {' '.join(argv_list)}")
    parser = build_run_argparser()
    args = parser.parse_args(argv)
    try:
        kw = run_kwargs_from_args(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    run_train_and_plot(**kw)


if __name__ == "__main__":
    main()
