"""Shared plotting helpers (no target-specific logic)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def nice_ylim(*curves, margin=1.25, step=5.0, floor=5.0, min_pad=3.0):
    """Symmetric y-limits that comfortably contain all provided curves."""
    vals = [np.asarray(c).ravel() for c in curves if c is not None]
    if not vals:
        return -floor, floor
    peak = float(np.max(np.abs(np.concatenate(vals))))
    ymax = max(peak * margin, peak + min_pad, floor)
    ymax = float(np.ceil(ymax / step) * step)
    return -ymax, ymax


def plot_cost(costs, path, *, costs_by_target=None, target_order=None):
    """Plot training cost; total + one subplot per target when ``costs_by_target`` is given."""
    if costs_by_target:
        names = list(target_order) if target_order else list(costs_by_target.keys())
        names = [n for n in names if n in costs_by_target and len(costs_by_target[n])]
        if names and costs is not None and len(costs):
            rows = [('total (weighted)', np.asarray(costs, dtype=np.float64))]
            rows.extend((name, costs_by_target[name]) for name in names)
            n = len(rows)
            fig, axes = plt.subplots(n, 1, figsize=(8, 2.8 * n), sharex=True)
            if n == 1:
                axes = [axes]
            nsteps = len(costs)
            for ax, (label, curve) in zip(axes, rows):
                ax.plot(curve, color='steelblue', linewidth=2)
                ax.set_ylabel('cost [% data power]')
                ax.set_title(label)
                ax.grid(True, alpha=0.3)
            axes[-1].set_xlabel('step')
            fig.suptitle(f'Training cost ({nsteps} steps)', fontsize=12, y=1.01)
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)
            return
        if len(names) == 1:
            costs = costs_by_target[names[0]]
    plt.figure(figsize=(8, 4))
    plt.plot(costs, color='steelblue', linewidth=2)
    plt.xlabel('step')
    plt.ylabel('cost [% data power]')
    plt.title(f'Training cost ({len(costs)} steps)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

