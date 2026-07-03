#!/usr/bin/env python3
"""<3 min micro-benchmark: synthetic traces, extrapolate to 512 panels."""
from __future__ import annotations

import os, sys, tempfile, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from training_config import COST_WINDOW_STEPS
from plot.moving_bar import _MOVING_BAR_T, _plot_moving_bar_cell, _style_moving_bar_time_axis
from plot.utils import nice_ylim as _nice_ylim

ANCHOR_512_S = 297.3  # measured full cold replot
NR, NC = 4, 4         # 16 panels — fast; scale x32 -> 512
REPS = 1


def _fake(nrows, ncols, seed=0):
    rng = np.random.default_rng(seed)
    mm, ms, dm = {}, {}, {}
    types = [f"T{i}" for i in range(nrows)]
    specs = [f"s{j}" for j in range(ncols)]
    for t in types:
        for s in specs:
            m = rng.normal(0, 0.2, COST_WINDOW_STEPS)
            mm[(t, s)] = m
            ms[(t, s)] = np.abs(rng.normal(0, 0.05, COST_WINDOW_STEPS))
            dm[(t, s)] = rng.normal(0, 0.15, COST_WINDOW_STEPS)
    return types, specs, mm, ms, dm


def _ylim(mm, ms, dm):
    c = []
    for k in mm:
        m, s, d = mm[k], ms[k], dm[k]
        c += [m, d, m + s, m - s]
    return _nice_ylim(*c)


def _bench(nrows, ncols, mode, dpi=100, rasterize=True, fmt="png", fig_scale=1.0):
    types, specs, mm, ms, dm = _fake(nrows, ncols)
    ylo, yhi = _ylim(mm, ms, dm)
    fig, axes = plt.subplots(nrows, ncols,
        figsize=(fig_scale * 1.4 * ncols, fig_scale * 0.85 * nrows),
        sharex=True, sharey=True)
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes[:, None] if ncols > 1 else axes[None, :]

    t0 = time.perf_counter()
    for ri, t in enumerate(types):
        for ci, s in enumerate(specs):
            ax = axes[ri, ci]
            m, sem, d = mm[(t, s)], ms[(t, s)], dm[(t, s)]
            if mode == "minimal":
                ax.plot(d, color="gray", linewidth=1.5)
                ax.plot(m, color="red", linewidth=1.5)
                ax.set_ylim(ylo, yhi)
            elif mode == "no_sem":
                _plot_moving_bar_cell(ax, m, sem, d, s, show_sem=False, ylim=(ylo, yhi))
            elif mode == "no_title":
                _plot_moving_bar_cell(ax, m, sem, d, "", show_sem=True, ylim=(ylo, yhi))
            elif mode == "baseline":
                _plot_moving_bar_cell(ax, m, sem, d, s, show_sem=True, ylim=None)
            else:  # production
                _plot_moving_bar_cell(ax, m, sem, d, s, show_sem=True, ylim=(ylo, yhi))
    for ci in range(ncols):
        _style_moving_bar_time_axis(axes[-1, ci], show_xlabel=(mode != "minimal"))
    fig.subplots_adjust(top=0.95, bottom=0.08, hspace=0.5, wspace=0.35)
    t_draw = time.perf_counter() - t0

    if rasterize:
        for ax in fig.axes:
            ax.set_rasterized(True)
    t1 = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=True) as f:
        kw = {"dpi": dpi}
        if fmt == "jpg":
            kw.update(format="jpg", pil_kwargs={"quality": 85})
        fig.savefig(f.name, **kw)
    t_save = time.perf_counter() - t1
    plt.close(fig)
    return t_draw, t_save


def _est512(t_draw, t_save, dpi, fig_scale=1.0):
    axis_r = (32 * 16) / (NR * NC)
    pix_r = ((1.4 * 16) * (0.85 * 32)) / ((fig_scale * 1.4 * NC) * (fig_scale * 0.85 * NR))
    return t_draw * axis_r + t_save * pix_r, t_draw * axis_r, t_save * pix_r


def main():
    t_start = time.perf_counter()
    cases = [
        ("production dpi100 raster", "production", 100, True, "png", 1.0),
        ("no SEM", "no_sem", 100, True, "png", 1.0),
        ("no title", "no_title", 100, True, "png", 1.0),
        ("minimal lines only", "minimal", 100, True, "png", 1.0),
        ("dpi 150", "production", 150, True, "png", 1.0),
        ("dpi 72", "production", 72, True, "png", 1.0),
        ("no rasterize", "production", 100, False, "png", 1.0),
        ("JPEG", "production", 100, True, "jpg", 1.0),
        ("fig 0.75x", "production", 100, True, "png", 0.75),
        ("minimal+dpi72+jpg", "minimal", 72, True, "jpg", 1.0),
    ]

    prod_draw, prod_save = _bench(NR, NC, "production")
    prod_est, _, _ = _est512(prod_draw, prod_save, 100)

    print(f"grid {NR}x{NC} x{REPS}  anchor 512 cold={ANCHOR_512_S:.0f}s\n")
    print(f"{'variant':<24} {'4x4 ms':>7} {'est512s':>8} {'save':>8}")
    print("-" * 52)
    for label, mode, dpi, rast, fmt, fs in cases:
        dr, sv = prod_draw, prod_save
        for _ in range(REPS):
            dr, sv = _bench(NR, NC, mode, dpi, rast, fmt, fs)
        est, _, _ = _est512(dr, sv, dpi, fs)
        saved = prod_est - est
        print(f"{label:<24} {(dr+sv)*1e3:7.0f} {est:8.1f} {saved:+8.1f}s")

    print("-" * 52)
    print(f"production est 512: {prod_est:.1f}s  (+0.2s trace -> full plot)")
    print(f"elapsed: {time.perf_counter()-t_start:.1f}s")


if __name__ == "__main__":
    main()
