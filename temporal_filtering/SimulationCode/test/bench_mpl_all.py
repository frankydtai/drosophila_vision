#!/usr/bin/env python3
"""One-shot benchmark: all matplotlib variants, ~3 min via large grid."""
from __future__ import annotations

import os
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from training_config import COST_WINDOW_STEPS
from plot.moving_bar import _plot_moving_bar_cell, _style_moving_bar_time_axis
from plot.utils import nice_ylim as _nice_ylim

ANCHOR_512_S = 297.3
NROWS, NCOLS = 12, 12
PANELS = NROWS * NCOLS
PANELS_512 = 32 * 16
SCALE_512 = PANELS_512 / PANELS


def _fake_traces(nrows, ncols, seed=0):
    rng = np.random.default_rng(seed)
    mm, ms, dm = {}, {}, {}
    for ri in range(nrows):
        for ci in range(ncols):
            m = rng.normal(0, 0.2, COST_WINDOW_STEPS)
            mm[(ri, ci)] = m
            ms[(ri, ci)] = np.abs(rng.normal(0, 0.05, COST_WINDOW_STEPS))
            dm[(ri, ci)] = rng.normal(0, 0.15, COST_WINDOW_STEPS)
    curves = []
    for key in mm:
        m, s, d = mm[key], ms[key], dm[key]
        curves.extend([m, d, m + s, m - s])
    return mm, ms, dm, _nice_ylim(*curves)


def _bench_variant(mm, ms, dm, ylim, mode, dpi=100, rasterize=True, fmt="png", fig_scale=1.0):
    fig, axes = plt.subplots(
        NROWS,
        NCOLS,
        figsize=(fig_scale * 1.4 * NCOLS, fig_scale * 0.85 * NROWS),
        sharex=True,
        sharey=True,
    )
    axes = np.asarray(axes)

    t0 = time.perf_counter()
    for ri in range(NROWS):
        for ci in range(NCOLS):
            ax = axes[ri, ci]
            m, sem, d = mm[(ri, ci)], ms[(ri, ci)], dm[(ri, ci)]
            if mode == "minimal":
                if d is not None:
                    ax.plot(d, color="gray", linewidth=1.5)
                ax.plot(m, color="red", linewidth=1.5)
                ax.set_ylim(ylim)
            elif mode == "no_sem":
                _plot_moving_bar_cell(ax, m, sem, d, f"s{ci}", show_sem=False, ylim=ylim)
            elif mode == "no_title":
                _plot_moving_bar_cell(ax, m, sem, d, "", show_sem=True, ylim=ylim)
            else:
                _plot_moving_bar_cell(ax, m, sem, d, f"s{ci}", show_sem=True, ylim=ylim)
        if NCOLS:
            axes[ri, 0].set_ylabel(f"T{ri}", fontsize=6, labelpad=4)
    for ci in range(NCOLS):
        _style_moving_bar_time_axis(axes[-1, ci], show_xlabel=(mode != "minimal"))
    fig.subplots_adjust(top=0.96, bottom=0.05, hspace=0.55, wspace=0.3)

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


def _est512_linear(t_draw, t_save):
    return t_draw * SCALE_512, t_save * SCALE_512, (t_draw + t_save) * SCALE_512


CASES = [
    ("production dpi100 raster", "production", 100, True, "png", 1.0),
    ("no SEM", "no_sem", 100, True, "png", 1.0),
    ("no title", "no_title", 100, True, "png", 1.0),
    ("minimal lines only", "minimal", 100, True, "png", 1.0),
    ("dpi 150", "production", 150, True, "png", 1.0),
    ("dpi 72", "production", 72, True, "png", 1.0),
    ("no rasterize", "production", 100, False, "png", 1.0),
    ("JPEG dpi100", "production", 100, True, "jpg", 1.0),
    ("fig 0.75x", "production", 100, True, "png", 0.75),
    ("minimal dpi72 jpg", "minimal", 72, True, "jpg", 0.75),
]


def main():
    t_start = time.perf_counter()
    mm, ms, dm, ylim = _fake_traces(NROWS, NCOLS)

    print(
        f"grid {NROWS}x{NCOLS}={PANELS} panels  linear x{SCALE_512:.3f} -> 512  "
        f"(anchor {ANCHOR_512_S:.0f}s = measured 512 cold replot, for calibration only)\n",
        flush=True,
    )
    print(
        f"{'variant':<22} {'draw':>7} {'save':>7} {'total':>7} "
        f"{'lin512':>7} {'cal512':>7} {'save s':>7}",
        flush=True,
    )
    print("-" * 78, flush=True)

    baseline_lin = None
    cal_factor = None
    rows = []

    for label, mode, dpi, rasterize, fmt, fig_scale in CASES:
        t0 = time.perf_counter()
        t_draw, t_save = _bench_variant(
            mm, ms, dm, ylim, mode, dpi=dpi, rasterize=rasterize, fmt=fmt, fig_scale=fig_scale,
        )
        lin_draw, lin_save, lin_total = _est512_linear(t_draw, t_save)
        if label == "production dpi100 raster":
            baseline_lin = lin_total
            cal_factor = ANCHOR_512_S / lin_total if lin_total > 0 else 1.0
        cal_total = lin_total * cal_factor if cal_factor else lin_total
        rows.append((label, t_draw, t_save, lin_total, cal_total))
        elapsed = time.perf_counter() - t_start
        print(
            f"{label:<22} {t_draw:7.2f}s {t_save:7.2f}s {t_draw + t_save:7.2f}s "
            f"{lin_total:7.1f}s {cal_total:7.1f}s {elapsed:6.0f}s",
            flush=True,
        )

    print("-" * 78, flush=True)
    prod = rows[0]
    print(
        f"production linear512={prod[3]:.1f}s  calibrated512={prod[4]:.1f}s  "
        f"(anchor {ANCHOR_512_S:.0f}s / linear = {cal_factor:.2f}x)",
        flush=True,
    )
    print(f"\nvs production (calibrated512 savings):", flush=True)
    for label, _, _, _, cal_total in rows:
        saved = prod[4] - cal_total
        print(f"  {label:<22} {saved:+6.1f}s", flush=True)
    print(f"\nelapsed {time.perf_counter() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
