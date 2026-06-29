#!/usr/bin/env python3
"""2D spatial snapshots at multiple times: Groschner edge vs Gruntman moving bar."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent

GROSCHNER_SPEED = 30.0
GROSCHNER_ON = 1.0
GROSCHNER_OFF = 0.0

GRUNTMAN_LED_DEG = 2.25
GRUNTMAN_BAR_HEIGHT = 9
GRUNTMAN_STEP_MS = 40.0
GRUNTMAN_SPEED = GRUNTMAN_LED_DEG / (GRUNTMAN_STEP_MS / 1000.0)
GRUNTMAN_BG = 0.5
GRUNTMAN_BRIGHT = 1.0
GRUNTMAN_DARK = 0.0

GROSCHNER_SNAP_MS = (0, 250, 500, 750, 1000, 1250)
GRUNTMAN_SNAP_MS = (0, 180, 360, 540, 720, 900)

mpl.rcParams.update(
    {
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


def groschner_edge_2d(
    az_deg: np.ndarray,
    el_deg: np.ndarray,
    t_ms: float,
    polarity: str,
    direction: str = "pd",
) -> np.ndarray:
    """Azimuth × elevation snapshot. Edge is vertical; uniform along elevation."""
    v = GROSCHNER_SPEED * (1.0 if direction == "pd" else -1.0)
    x_edge = -15.0 + v * (t_ms / 1000.0)
    az = az_deg[None, :]
    if polarity == "on":
        # Bright half advances right (PD): already swept = left of edge = bright.
        bright = az <= x_edge if direction == "pd" else az >= x_edge
        field = np.where(bright, GROSCHNER_ON, GROSCHNER_OFF)
    else:
        # Dark half advances right (PD): already swept = left of edge = dark.
        dark = az <= x_edge if direction == "pd" else az >= x_edge
        field = np.where(dark, GROSCHNER_OFF, GROSCHNER_ON)
    return np.broadcast_to(field, (len(el_deg), len(az_deg))).copy()


def _gruntman_step_index(t_ms: float) -> int:
    return max(0, int(t_ms // GRUNTMAN_STEP_MS))


def gruntman_bar_2d(
    x_led: np.ndarray,
    y_led: np.ndarray,
    t_ms: float,
    width_led: int,
    contrast: str = "pc",
    direction: str = "pd",
    t4_style: bool = True,
) -> np.ndarray:
    """LED azimuth × elevation snapshot (motion along x)."""
    lum = np.full((len(y_led), len(x_led)), GRUNTMAN_BG, dtype=float)
    bar_val = GRUNTMAN_BRIGHT if contrast == "pc" else GRUNTMAN_DARK
    sign = 1 if direction == "pd" else -1
    start_led = -6
    k = _gruntman_step_index(t_ms)

    if t4_style:
        lead = start_led + sign * k
        trail = lead + sign * (width_led - 1)
        x_lo, x_hi = min(lead, trail), max(lead, trail)
        y_lo = y_led[0]
        y_hi = y_led[-1]
    else:
        lead = start_led + sign * k
        visible = min(width_led, max(1, k + 1))
        if sign > 0:
            x_lo, x_hi = lead, lead + visible - 1
        else:
            x_lo, x_hi = lead - visible + 1, lead
        y_lo = y_led[0]
        y_hi = y_led[-1]

    in_x = (x_led >= x_lo) & (x_led <= x_hi)
    in_y = (y_led >= y_lo) & (y_led <= y_hi)
    lum[np.ix_(in_y, in_x)] = bar_val
    return lum


def draw_2d_field(
    ax: plt.Axes,
    field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    x_label: str,
    y_label: str,
    title: str,
    *,
    show_rf: bool = True,
    show_axes: bool = False,
) -> None:
    extent = [x[0] - 0.5, x[-1] + 0.5, y[0] - 0.5, y[-1] + 0.5]
    ax.imshow(
        field,
        origin="lower",
        aspect="equal",
        extent=extent,
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    if show_rf:
        ax.plot(0, 0, "+", color="#C20078", ms=10, mew=1.5)
    ax.set_title(title, fontsize=8, pad=3)
    if show_axes:
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
    else:
        ax.set_xticks([])
        ax.set_yticks([])


def plot_comparison(out_path: Path) -> None:
    n_snap = len(GROSCHNER_SNAP_MS)
    fig = plt.figure(figsize=(2.2 * n_snap + 0.8, 15.5))
    gs = fig.add_gridspec(
        6,
        n_snap,
        height_ratios=[1.0, 1.0, 1.0, 0.85, 1.0, 0.85],
        hspace=0.35,
        wspace=0.12,
        left=0.07,
        right=0.98,
        top=0.96,
        bottom=0.03,
    )

    az = np.linspace(-18, 18, 181)
    el = np.linspace(-10, 10, 101)
    x_led = np.arange(-8, 9)
    y_led = np.arange(-(GRUNTMAN_BAR_HEIGHT // 2), GRUNTMAN_BAR_HEIGHT // 2 + 1)

    gruntman_title = (
        f"{GRUNTMAN_BAR_HEIGHT} LED tall, "
        f"{GRUNTMAN_STEP_MS:.0f} ms step ≈ {GRUNTMAN_SPEED:.0f}° s$^{{-1}}$, "
        f"0–900 ms"
    )
    row_specs = [
        ("Groschner 2022 — ON contrast edge, PD, 30° s$^{-1}$", "groschner", "on", 4),
        ("Groschner 2022 — OFF contrast edge, PD, 30° s$^{-1}$", "groschner", "off", 4),
        (f"Gruntman 2021 — bright bar (T4 PC), 4 LED, {gruntman_title}", "bar_time", "pc", 4),
        (f"Gruntman 2021 — bright bar (T4 PC), 1 LED, {gruntman_title}", "bar_time", "pc", 1),
        (f"Gruntman 2021 — dark bar (T4 NC), 4 LED, {gruntman_title}", "bar_time", "nc", 4),
        (f"Gruntman 2021 — dark bar (T4 NC), 1 LED, {gruntman_title}", "bar_time", "nc", 1),
    ]

    n_rows = len(row_specs)
    for row_idx, (row_title, kind, contrast, width_led) in enumerate(row_specs):
        fig.text(
            0.01,
            0.94 - row_idx * (0.90 / n_rows),
            row_title,
            fontsize=9,
            fontweight="bold",
            va="top",
            ha="left",
            rotation=90,
        )

        snap_times = GROSCHNER_SNAP_MS if kind == "groschner" else GRUNTMAN_SNAP_MS
        for col, t_ms in enumerate(snap_times):
            ax = fig.add_subplot(gs[row_idx, col])
            if kind == "groschner":
                field = groschner_edge_2d(az, el, t_ms, polarity=contrast, direction="pd")
                draw_2d_field(
                    ax,
                    field,
                    az,
                    el,
                    "Azimuth (°)",
                    "Elevation (°)",
                    f"t = {t_ms} ms",
                    show_axes=(row_idx == 0 and col == 0),
                )
            else:
                field = gruntman_bar_2d(
                    x_led,
                    y_led,
                    t_ms,
                    width_led=width_led,
                    contrast=contrast,
                    direction="pd",
                )
                draw_2d_field(
                    ax,
                    field,
                    x_led * GRUNTMAN_LED_DEG,
                    y_led * GRUNTMAN_LED_DEG,
                    "Azimuth (°)",
                    "Elevation (°)",
                    f"t = {t_ms} ms",
                    show_axes=(col == 0),
                )

    fig.suptitle(
        "2D spatial snapshots over time — Groschner edge vs Gruntman moving bar",
        fontsize=11,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.005,
        "Magenta + marks RF centre (0°, 0°).  Groschner: half-plane edge (black/white).  "
        "Gruntman: bright (PC) or dark (NC) bar on grey background.",
        ha="center",
        fontsize=7.5,
        color="#444",
    )

    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=REPO / "stimulus_comparison.png",
    )
    args = ap.parse_args()
    plot_comparison(args.output)


if __name__ == "__main__":
    main()
