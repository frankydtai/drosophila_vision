"""Plot spot RecF gt time courses for all fit cells; report first nonzero.

Timing: ``ms_pre=100``, ``ms_pulse=50``, ``ms_response=400``; ``--delta-ms``
(default ``DELTA_MS`` from ``param_defaults``). Uses ``figure.spot.plot_cell_time``
only (no RF panel).

Usage (from ``SimulationCode/``):

    ../.venv/bin/python test/plot_spot_gt_time.py
    ../.venv/bin/python test/plot_spot_gt_time.py --delta-ms 5
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import import_bootstrap  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

from network.build import cell_family_rows, cell_names_in_family_order
from figure.gt import fit_gt_cubes
from figure.spot import CENTER_BIN, _pulse_end_from_opts, plot_cell_time
from figure.util import save_figure
from task.spot.input import spot_timing_t
from param_defaults import DELTA_MS

MS_PRE = 100.0
MS_PULSE = 50.0
MS_RESPONSE = 400.0
DEFAULT_SAVE = os.path.join(HERE, "spot_time.png")
ABS_TOL = 1e-12


def first_nonzero_t(y: np.ndarray, *, atol: float = ABS_TOL) -> int | None:
    """First sample index with ``|y| > atol``, or ``None`` if all ~0."""
    y = np.asarray(y, dtype=np.float64)
    hit = np.flatnonzero(np.abs(y) > atol)
    if hit.size == 0:
        return None
    return int(hit[0])


def report_first_nonzero(cells: dict[str, np.ndarray], *, t_onset: int, delta_ms: float) -> None:
    """Print center-bin first nonzero vs global t=0 and vs onset."""
    print(
        f"timing: ms_pre={MS_PRE:g} ms_pulse={MS_PULSE:g} "
        f"ms_response={MS_RESPONSE:g} delta_ms={delta_ms:g}  "
        f"t_onset={t_onset} ({t_onset * delta_ms:g} ms)"
    )
    print(f"{'cell':<6} {'t':>5} {'ms':>8} {'t-t_onset':>10} {'ms-onset':>10}")
    for name in cells:
        cube = np.asarray(cells[name], dtype=np.float64)
        t = first_nonzero_t(cube[CENTER_BIN])
        if t is None:
            print(f"{name:<6} {'(all ~0)':>5}")
            continue
        rel = t - t_onset
        print(
            f"{name:<6} {t:5d} {t * delta_ms:8.1f} "
            f"{rel:10d} {rel * delta_ms:10.1f}"
        )


def plot_all_cells(
    path: str,
    *,
    cells: dict[str, np.ndarray],
    t_onset: int,
    n_t: int,
    pulse_end: int,
    delta_ms: float,
) -> None:
    """Time-only grid for every fit cell via ``plot_cell_time``."""
    groups = [np.array(row) for row in cell_family_rows(list(cells))]
    names = cell_names_in_family_order(list(cells))
    nrows = len(groups)
    ncols = 5
    fig = plt.figure(figsize=(3.0 * ncols, 2.0 * nrows))
    gs = fig.add_gridspec(
        nrows, ncols,
        hspace=0.55, wspace=0.55, top=0.90, bottom=0.08, left=0.07, right=0.98,
    )

    for gi, row_names in enumerate(groups):
        row_idx = [names.index(str(n)) for n in row_names]
        start = (ncols - len(row_idx)) // 2
        for j, ni in enumerate(row_idx):
            name = names[ni]
            col = start + j
            ax = fig.add_subplot(gs[gi, col])
            # pre_end=0: draw full gt including pre-onset zeros.
            plot_cell_time(
                ax, None, cells[name],
                title=name,
                show_xlabels=True,
                show_ylabel=(j == 0),
                n_t=n_t,
                t_onset=t_onset,
                pre_end=0,
                show_pre=False,
                pulse_end=pulse_end,
                delta_ms=delta_ms,
            )

    fig.suptitle(
        f"spot bright gt  (ms_pre={MS_PRE:g}, ms_pulse={MS_PULSE:g}, "
        f"ms_response={MS_RESPONSE:g}, delta_ms={delta_ms:g}; t_onset={t_onset})",
        fontsize=12,
    )
    save_figure(fig, path, dpi=150)
    print(f"saved {path}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--save", default=DEFAULT_SAVE, help="output PNG path")
    p.add_argument(
        "--delta-ms",
        type=float,
        default=DELTA_MS,
        metavar="MS",
        help=f"time step in ms (default: {DELTA_MS})",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    delta_ms = float(args.delta_ms)
    if delta_ms <= 0:
        raise SystemExit("--delta-ms must be > 0")
    t_onset, n_t = spot_timing_t(
        ms_pre=MS_PRE, ms_response=MS_RESPONSE, delta_ms=delta_ms,
    )
    pulse_end = _pulse_end_from_opts(
        {"ms_pulse": MS_PULSE, "delta_ms": delta_ms}, t_onset, n_t,
    )
    cubes = fit_gt_cubes(
        contrasts=("bright",),
        t_onset=t_onset,
        n_t=n_t,
        ms_pulse=MS_PULSE,
        delta_ms=delta_ms,
    )
    cells = cubes["bright"]
    report_first_nonzero(cells, t_onset=t_onset, delta_ms=delta_ms)
    plot_all_cells(
        args.save, cells=cells, t_onset=t_onset, n_t=n_t, pulse_end=pulse_end,
        delta_ms=delta_ms,
    )


if __name__ == "__main__":
    main()
