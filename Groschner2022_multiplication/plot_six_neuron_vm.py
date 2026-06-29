#!/usr/bin/env python3
"""Plot measured Vm for all six neurons (Mi9, Tm3, Mi1, Mi4, C3, T4).

Layout matches Extended Data Fig. 7: 12 rows (6 cell types × ON/OFF edge)
× 18 columns (36 motion directions, every 20°). Presynaptic Vm uses
direction-dependent time shifts on cell-averaged traces (raw mV, not
normalized). T4 uses measured Vm at PD/ND (fig3_T4v.npy); other directions
use the conductance model Vm so every column has a trace.

Usage (from repo root, with .venv active):
    python plot_six_neuron_vm.py
    python plot_six_neuron_vm.py -o six_neuron_vm.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# --- paths ---
REPO = Path(__file__).resolve().parent
FIG3 = REPO / "A_biophysical_account_of_multiplication_by_a_single_neuron" / "Fig. 3"

# --- experiment constants (Fig. 3 / ED Fig. 7) ---
FS = 1000
TF = 8000
NDIRS = 36
STIMS = ("on", "off")
DIRS = ("pd", "nd")

NEURONS = ("Mi9", "Tm3", "Mi1", "Mi4", "C3", "T4")
COLORS = {
    "Mi9": "forestgreen",
    "Tm3": "orange",
    "Mi1": "C3",
    "Mi4": "C0",
    "C3": "cadetblue",
    "T4": "#C20078",
}
# fallback if data span is degenerate
Y_FALLBACK = (-90, -15)

# model params (ED Fig. 7)
GAINS = (0.92, 0.35, 0.65, 1.1, 1.49)
THRLDS = (0.20, 0.35, 0.88, 0.44, 0.70)
GLEAK = 0.50
ELEAK = -65.0
EGlu = -71.0
EGABA = -68.0
EnAChR = -21.0
PARAMS = np.array([*GAINS, *THRLDS, ELEAK, GLEAK])

mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False


def shift_data(data: np.ndarray, shift: int) -> np.ndarray:
    shifted = np.roll(data, shift)
    if shift > 0:
        shifted[:shift] = data[0]
    elif shift < 0:
        shifted[shift:] = data[-1]
    return shifted


def normalize(a: np.ndarray) -> np.ndarray:
    anorm = a - np.min(a)
    return anorm / np.max(anorm)


def dir_shift_vm(
    traces: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Raw mV traces → (2, 36, 8000) with direction-dependent delays."""
    out = {
        name: np.full((len(STIMS), NDIRS, TF), np.nan, dtype=float)
        for name in ("Mi9", "Tm3", "Mi1", "Mi4", "C3")
    }
    for s in range(len(STIMS)):
        for i in range(NDIRS):
            direction = i * 360 / NDIRS * np.pi / 180
            delay = int(4.8 * FS / 30 * np.cos(direction))
            out["Mi9"][s, i] = shift_data(traces["Mi9"][s], -delay)
            out["Tm3"][s, i] = traces["Tm3"][s]
            out["Mi1"][s, i] = traces["Mi1"][s]
            out["Mi4"][s, i] = shift_data(traces["Mi4"][s], +delay)
            out["C3"][s, i] = shift_data(traces["C3"][s], +delay)
    return out


def rect(x: np.ndarray, thrld: float) -> np.ndarray:
    result = x - thrld
    result = result * (result > 0)
    return result + thrld


def t4_model_vm(traces_norm: dict[str, np.ndarray]) -> np.ndarray:
    """T4 model Vm (2, 36, 8000) from normalized, direction-shifted inputs."""
    shifted = dir_shift_vm(traces_norm)
    vs = [shifted[n] for n in ("Mi9", "Tm3", "Mi1", "Mi4", "C3")]
    gs = []
    for idx, v in enumerate(vs):
        gs.append(PARAMS[idx] * rect(v - PARAMS[idx + 5], 0))
    gmi9, gtm3, gmi1, gmi4, gc3 = gs
    gtotal = gmi9 + gtm3 + gmi1 + gmi4 + gc3 + GLEAK
    t4 = (
        EGlu * gmi9
        + EnAChR * (gtm3 + gmi1)
        + EGABA * (gmi4 + gc3)
        + ELEAK * GLEAK
    ) / gtotal
    return t4


def t4_measured_on_grid(t4v: np.ndarray) -> np.ndarray:
    """Place measured T4 (2, 2, tf) PD/ND traces on the 36-dir grid.

    Dir index 0 matches PD shift geometry; index 18 matches ND (fig3 pdnd).
    """
    grid = np.full((len(STIMS), NDIRS, TF), np.nan)
    for s in range(len(STIMS)):
        grid[s, 0] = t4v[s, 0]   # PD
        grid[s, 18] = t4v[s, 1]  # ND
    return grid


def load_data(fig3_dir: Path) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Return direction-shifted Vm (6 neurons), T4 SEM grid, normalized traces."""
    avg = {}
    for name in ("Mi9", "Tm3", "Mi1", "Mi4", "C3"):
        arr = np.load(fig3_dir / f"fig3_{name}.npy")
        avg[name] = np.nanmean(arr, axis=1)

    t4va = np.load(fig3_dir / "fig3_T4v.npy")
    t4v = t4va.mean(0)
    t4_sem = stats.sem(t4va, axis=0, nan_policy="omit")

    vm = dir_shift_vm(avg)
    vm["T4"] = t4_measured_on_grid(t4v)

    t4_model = t4_model_vm({k: normalize(avg[k]) for k in avg})
    # fill directions without patch-clamp T4 using model Vm
    missing = np.isnan(vm["T4"])
    vm["T4"][missing] = t4_model[missing]

    sem = {n: None for n in NEURONS}
    sem["T4"] = t4_measured_on_grid(t4_sem)

    return vm, sem, avg


def roll_pd_center(vm: dict[str, np.ndarray], sem: dict[str, np.ndarray | None]) -> None:
    """In-place roll direction axis so PD sits at the centre column (ED Fig. 7)."""
    shift = NDIRS // 2
    for name in NEURONS:
        vm[name] = np.roll(vm[name], shift, axis=1)
        if sem[name] is not None:
            sem[name] = np.roll(sem[name], shift, axis=1)


def global_ylim(
    vm: dict[str, np.ndarray],
    sem: dict[str, np.ndarray | None],
    margin: int,
    pad: float = 2.0,
) -> tuple[float, float]:
    """Shared y-axis limits (mV) across all neurons and directions."""
    sl = slice(margin, -margin)
    chunks = [vm[n][:, :, sl].ravel() for n in NEURONS]
    if sem["T4"] is not None:
        t4 = vm["T4"][:, :, sl]
        s = sem["T4"][:, :, sl]
        chunks.append((t4 - s).ravel())
        chunks.append((t4 + s).ravel())
    vals = np.concatenate(chunks)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return Y_FALLBACK
    return float(np.min(vals) - pad), float(np.max(vals) + pad)


def plot_six_neuron_vm(
    vm: dict[str, np.ndarray],
    sem: dict[str, np.ndarray | None],
    *,
    margin: int = 3250,
    downsample: int = 15,
    output: Path | None = None,
    show: bool = True,
) -> plt.Figure:
    ncols = NDIRS // 2  # 18 columns, every 20°
    nrows = len(NEURONS) * len(STIMS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 1.15 * nrows), sharex=True, sharey=True)

    if nrows == 1:
        axes = axes[np.newaxis, :]
    if ncols == 1:
        axes = axes[:, np.newaxis]

    sl = slice(margin, -margin)
    t = np.arange(0, (TF - 2 * margin) / FS, 1 / FS)[::downsample]
    ylo, yhi = global_ylim(vm, sem, margin)

    for row, name in enumerate(NEURONS):
        for stim_idx, stim in enumerate(STIMS):
            ax_row = row * len(STIMS) + stim_idx
            color = COLORS[name]
            for col, dir_idx in enumerate(range(0, NDIRS, 2)):
                ax = axes[ax_row, col]
                y = vm[name][stim_idx, dir_idx, sl][::downsample]
                ax.plot(t, y, lw=1.5, c=color)
                if name == "T4" and sem["T4"] is not None:
                    s = sem["T4"][stim_idx, dir_idx, sl][::downsample]
                    if not np.all(np.isnan(s)):
                        ax.fill_between(t, y - s, y + s, color="grey", alpha=0.12, linewidth=0)
                if col > 0:
                    ax.tick_params(labelleft=False)
                if ax_row < nrows - 1:
                    ax.tick_params(labelbottom=False)
                if col == 0:
                    ax.set_ylabel(f"{name} {stim.upper()}\nVm (mV)", fontsize=8)
            if ax_row == nrows - 1:
                axes[ax_row, col].set_xlabel("Time (s)", fontsize=7)

    axes[0, 0].set_ylim(ylo, yhi)

    for col in range(ncols):
        orig_dir = (col * 2 - NDIRS // 2) % NDIRS
        deg = orig_dir * (360 // NDIRS)
        axes[0, col].set_title(f"{deg}°", fontsize=7, pad=2)

    fig.suptitle(
        "Six-neuron Vm (30°/s edges): presynaptic measured; T4 measured at PD/ND, model elsewhere",
        fontsize=10,
        y=1.002,
    )
    fig.tight_layout()

    if output is not None:
        fig.savefig(output, bbox_inches="tight", dpi=150)
        print(f"Saved {output}")
    if show:
        plt.show()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--fig3-dir",
        type=Path,
        default=FIG3,
        help="Directory containing fig3_*.npy files",
    )
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output PDF/PNG path")
    parser.add_argument("--margin", type=int, default=3250, help="Time margin (ms) cropped from each side")
    parser.add_argument("--downsample", type=int, default=15, help="Plot every Nth sample")
    parser.add_argument("--no-show", action="store_true", help="Do not open interactive window")
    args = parser.parse_args()

    vm, sem, _ = load_data(args.fig3_dir)
    roll_pd_center(vm, sem)
    plot_six_neuron_vm(
        vm,
        sem,
        margin=args.margin,
        downsample=args.downsample,
        output=args.output,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
