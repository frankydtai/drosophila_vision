#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train-then-plot orchestrator (Hydra config → train → figure).

Usage (from ``simulation/``, project ``.venv``):

    ../.venv/bin/python run.py
    ../.venv/bin/python run.py n_iter_gpu=300
    ../.venv/bin/python run.py tasks=spot spot_radius=1.5

Re-plot an existing run without train:

    ../.venv/bin/python -m figure.plot
    ../.venv/bin/python -m figure.plot html=true

All defaults live in ``conf/config.yaml`` (single file). CLI overrides use
Hydra ``key=value`` syntax.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import import_bootstrap  # noqa: F401

import hydra
import numpy as np

from config import resolve_run_kwargs


def run_train_and_plot(**kwargs):
    import train
    import train.implementation as implementation
    from figure.plot import plot_rf_t
    from figure.panel import plot_cost, figure_file_ext

    syn_sign = kwargs.pop("syn_sign")
    figure_kwargs = kwargs.pop("figure_kwargs")

    checkpoint_on_png = None
    if kwargs.get("checkpoint_interval") is not None:
        def checkpoint_on_png(outdir, iter, z_best, cost_best, session):
            token = implementation.checkpoint_iter_token(iter)
            png_dir = os.path.join(outdir, "png")
            os.makedirs(png_dir, exist_ok=True)
            base_suffix = figure_kwargs.get("file_suffix") or ""
            plot_rf_t(
                np.array([z_best.detach().cpu().numpy()]),
                png_dir,
                session=session,
                final_costs=np.array([cost_best]),
                save_data=False,
                **figure_kwargs,
                file_suffix=f"{base_suffix}_{token}",
            )
            print(f"wrote checkpoint png: {png_dir}/*_{token}.png")

    fname, outdir, session, result = implementation.run_train(
        **kwargs,
        checkpoint_on_png=checkpoint_on_png,
    )
    if result.cost_curve is not None and len(result.cost_curve) > 0:
        plot_cost(
            result.cost_curve,
            os.path.join(outdir, f'cost_curve{figure_file_ext(html=figure_kwargs.get("html"))}'),
            costs_by_part=result.cost_curves_by_part,
            part_order=list(train.session_cost_part_keys(session)),
        )
    plot_rf_t(
        result.run_params,
        outdir,
        session=session,
        final_costs=result.final_costs,
        save_data=False,
        **figure_kwargs,
    )
    if syn_sign:
        from analyze.syn_sign import save_syn_sign_figures
        save_syn_sign_figures(outdir)
    return fname, outdir, session


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg) -> None:
    try:
        kwargs = resolve_run_kwargs(cfg)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"config: {cfg}")
    run_train_and_plot(**kwargs)


if __name__ == "__main__":
    main()
