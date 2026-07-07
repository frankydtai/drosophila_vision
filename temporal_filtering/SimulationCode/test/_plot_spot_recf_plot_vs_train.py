"""Compare spot RF data curves: plot pipeline (solid) vs training interp (dashed).

Reuses plot.spot._scale_curve and network.spot_target._recf_at — no re-implementation.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import Medulla_Library as ml
from plot.spot import _scale_curve, CENTER_BIN
from network.spot_target import _recf_at

OUT_PATH = os.path.join(HERE, 'spot_recf_plot_vs_train.png')
NSAMPLES = 45
RF_CENTER = 22
SQRT3_R = np.sqrt(3)
SQRT3_PLOT_X = RF_CENTER + 5 * SQRT3_R - 2  # after roll(-2)


def _rf_train_curve(recf_row, impr_row, maxt, center_amp):
    """45-point RF from training ``RecF(r)*ImpR(t_peak)*DATA_AMP``, plot normalization."""
    impr_at_peak = float(impr_row[maxt]) * ml.DATA_AMP
    rf = np.empty(NSAMPLES, dtype=np.float64)
    for s in range(NSAMPLES):
        r = (s - RF_CENTER) / 5.0
        rf[s] = _recf_at(recf_row, r) * impr_at_peak
    peak = float(np.max(np.abs(rf)))
    if peak > 0.0:
        rf = rf / peak * center_amp
    return np.roll(rf, -2)


def _y_at_plot_x(curve, plot_x):
    return float(np.interp(plot_x, np.arange(len(curve)), curve))


def main():
    recf_data, impr_data = ml.read_RecF_ImpR()
    ref9 = ml.read_RecF_data() * ml.DATA_AMP
    names = [str(c) for c in ml.cell_list]
    n = len(names)
    ncols = 5
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.2 * nrows), squeeze=False)
    fig.suptitle(
        'spot data RF: plot pipeline (solid) vs training interp (dashed)\n'
        f'vertical line: r=$\\sqrt{{3}}$  (plot_x={SQRT3_PLOT_X:.2f})',
        fontsize=11,
    )

    for i, name in enumerate(names):
        row, col = divmod(i, ncols)
        ax = axes[row, col]
        xt = ref9[i]
        _imp, rf_plot = _scale_curve(xt, CENTER_BIN)
        maxt = int(np.argmax(np.abs(xt[CENTER_BIN])))
        center_amp = float(np.max(np.abs(xt[CENTER_BIN])))
        rf_train = _rf_train_curve(recf_data[i], impr_data[i], maxt, center_amp)

        x = np.arange(NSAMPLES)
        ax.plot(x, rf_plot, color='0.45', lw=1.8, solid_capstyle='round', label='plot data')
        ax.plot(x, rf_train, color='0.45', lw=1.8, ls='--', label='train interp')
        ax.axvline(SQRT3_PLOT_X, color='crimson', lw=0.8, ls=':', alpha=0.85)

        y_plot = _y_at_plot_x(rf_plot, SQRT3_PLOT_X)
        y_train = _y_at_plot_x(rf_train, SQRT3_PLOT_X)
        ax.scatter([SQRT3_PLOT_X], [y_plot], s=18, color='0.2', zorder=5)
        ax.scatter([SQRT3_PLOT_X], [y_train], s=18, facecolors='none', edgecolors='crimson', zorder=5)
        ax.set_title(name, fontsize=9, fontweight='bold')
        ax.set_xlim(0, 40)
        ax.set_xticks([0, 20, 40])
        ax.set_xticklabels(['-20', '0', '20'], fontsize=7)
        if col == 0:
            ax.set_ylabel('mV', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.text(
            0.03, 0.97,
            f'$\\sqrt{{3}}$: {y_plot:.2f} / {y_train:.2f}',
            transform=ax.transAxes, fontsize=6.5, va='top',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'),
        )

    for j in range(n, nrows * ncols):
        row, col = divmod(j, ncols)
        axes[row, col].axis('off')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=8, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
    print(f'wrote {OUT_PATH}')

    print(f'\nr=sqrt(3)  plot_x={SQRT3_PLOT_X:.4f}')
    print(f'{"cell":<6} {"plot_y":>10} {"train_y":>10} {"diff":>10}')
    for i, name in enumerate(names):
        xt = ref9[i]
        _imp, rf_plot = _scale_curve(xt, CENTER_BIN)
        maxt = int(np.argmax(np.abs(xt[CENTER_BIN])))
        rf_train = _rf_train_curve(recf_data[i], impr_data[i], maxt, float(np.max(np.abs(xt[CENTER_BIN]))))
        yp = _y_at_plot_x(rf_plot, SQRT3_PLOT_X)
        yt = _y_at_plot_x(rf_train, SQRT3_PLOT_X)
        print(f'{name:<6} {yp:10.4f} {yt:10.4f} {yp - yt:10.4f}')


if __name__ == '__main__':
    main()
