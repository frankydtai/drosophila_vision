#!/usr/bin/env python3
"""2D LED-grid snapshots: Gruntman cardinal vs diagonal moving-bar paths.

Mirrors createMovingBarDiagCorrProtocolG4.m + generateBarFrameByInds.m +
makeRectangleMask.m (Reiser lab G4 pipeline).

Cardinal PD–ND (Fig. 3 cells): motion span 9 LED (w=1) along one axis.
Diagonal PD–ND (Fig. 1 cells): 13 steps for the same traverse (w=1) on a
rectangular grid; wider bars add trailing steps per protocol relPos.

Usage:
  python plot_cardinal_vs_diagonal.py
  python plot_cardinal_vs_diagonal.py -o cardinal_vs_diagonal.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import rotate

REPO = Path(__file__).resolve().parent

LED_DEG = 2.25
BAR_SPAN = 9
BAR_HEIGHT = 9
STEP_MS = 40.0
SPEED_DEG_S = LED_DEG / (STEP_MS / 1000.0)
BG = 0.5
BRIGHT = 1.0
DARK = 0.0

MASK_HW = BAR_SPAN // 2  # floor(barSpan/2) = 4
MASK_HH = BAR_HEIGHT // 2  # floor(barHeight/2) = 4
DIAG_RAD = round((2 * MASK_HW + 1) / np.sqrt(2))  # relDiagR = 6

# Canvas for makeRectangleMask (baseSiz in createMovingBarDiagCorrProtocolG4.m).
MASK_MAT_SIZE = 445

CARDINAL_ORI = 0
DIAGONAL_ORI = 1

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


def _divide_tot_square_to_cols(tot_width: int, sq_ori: int, sq_dim: int) -> list[np.ndarray]:
    """Port of divideTotSquareToCols.m; column coords are offsets from centre [0, 0]."""
    half = sq_dim // 2
    rel_wid = tot_width // 2
    flip = sq_ori > 3
    ori = sq_ori - 4 if flip else sq_ori
    cols: list[np.ndarray | None] = [None] * sq_dim

    if ori == 0:
        first = np.column_stack(
            [np.arange(half, -half - 1, -1, dtype=int), np.full(sq_dim, half, dtype=int)]
        )
        cols[0] = first
        for ii in range(1, sq_dim):
            cols[ii] = first - np.array([0, ii])
        cols = cols[::-1]
    elif ori == 1:
        cols[0] = np.column_stack([np.arange(-half, 1), np.arange(0, half + 1)])
        for ii in range(1, sq_dim):
            rel = cols[ii - 1]
            assert rel is not None
            if ii % 2 == 1:
                cols[ii] = (rel + np.array([1, 0]))[:-1]
            else:
                cols[ii] = np.unique(np.vstack([rel + [1, 0], rel + [0, -1]]), axis=0)
        cols = cols[::-1]
    elif ori == 2:
        first = np.column_stack([np.full(sq_dim, -half), np.arange(-half, half + 1)])
        cols[0] = first
        for ii in range(1, sq_dim):
            cols[ii] = first + np.array([ii, 0])
        cols = cols[::-1]
    elif ori == 3:
        cols[0] = np.column_stack([np.arange(0, half + 1), np.arange(half, -1, -1)])
        for ii in range(1, sq_dim):
            rel = cols[ii - 1]
            assert rel is not None
            if ii % 2 == 1:
                cols[ii] = (rel + np.array([-1, 0]))[1:]
            else:
                cols[ii] = np.unique(np.vstack([rel + [-1, 0], rel + [0, -1]]), axis=0)
    else:
        raise ValueError(f"unsupported orientation {sq_ori}")

    if flip:
        cols = [c[::-1] for c in cols if c is not None]

    cen = half + 1
    out = cols[cen - rel_wid - 1 : cen + rel_wid]
    return [c for c in out if c is not None]


def _expanded_sq_dim(sq_dim_in: int, ori: int) -> int:
    """generateBarFrameByInds.m: expand sqDim for 45° orientations."""
    if ori % 2 == 1:
        return 2 * round(sq_dim_in / np.sqrt(2)) + 1
    return sq_dim_in


def _cardinal_bar_leds(bar_pos: int, width_led: int) -> np.ndarray:
    """Axis-aligned 9×W bar for cardinal ori=0 (motion along azimuth)."""
    w_lo = (width_led - 1) // 2
    w_hi = width_led - 1 - w_lo
    h_lo = (BAR_HEIGHT - 1) // 2
    h_hi = BAR_HEIGHT - 1 - h_lo
    xs = np.arange(bar_pos - w_lo, bar_pos + w_hi + 1)
    ys = np.arange(-h_lo, h_hi + 1)
    return np.array([(x, y) for x in xs for y in ys], dtype=int)


def _diagonal_bar_leds(bar_pos: int, width_led: int) -> np.ndarray:
    """Diagonal bar via generateBarFrameByInds (sqDim 9→13, ori=1)."""
    sq_dim_in = 2 * MASK_HW + 1
    sq_dim = _expanded_sq_dim(sq_dim_in, DIAGONAL_ORI)
    cols = _divide_tot_square_to_cols(sq_dim, DIAGONAL_ORI, sq_dim)
    conv_pos = bar_pos + int(np.ceil(sq_dim / 2))
    rel = [i for i in range(conv_pos - width_led + 1, conv_pos + 1) if 1 <= i <= sq_dim]
    if not rel:
        return np.empty((0, 2), dtype=int)
    return np.vstack([cols[i - 1] for i in rel]).astype(int)


def _rectangle_mask_offsets(mask_h_w: int, mask_h_h: int, ori: int) -> set[tuple[int, int]]:
    """Port of makeRectangleMask.m; returns (x, y) offsets from RF centre."""
    mat_size = MASK_MAT_SIZE
    cen = int(np.ceil(mat_size / 2)) - 1
    mask = np.zeros((mat_size, mat_size), dtype=np.float64)
    mask[cen - mask_h_h : cen + mask_h_h + 1, cen - mask_h_w : cen + mask_h_w + 1] = 1.0

    if ori:
        rot = rotate(mask, 45 * ori, reshape=False, order=0, mode="constant", cval=0.0)
        temp_inds = _divide_tot_square_to_cols(2 * mask_h_w + 1, ori, mat_size)
        sec = np.zeros((mat_size, mat_size), dtype=np.float64)
        for col in temp_inds:
            for x_off, y_off in col:
                px = int(x_off + cen)
                py = int(y_off + cen)
                if 0 <= py < mat_size and 0 <= px < mat_size:
                    sec[py, px] = 1.0
        mask = rot * sec

    ys, xs = np.where(mask > 0)
    return {(int(x - cen), int(y - cen)) for y, x in zip(ys, xs)}


def _cardinal_mask_offsets() -> set[tuple[int, int]]:
    return {(x, y) for x in range(-MASK_HW, MASK_HW + 1) for y in range(-MASK_HH, MASK_HH + 1)}


def _motion_params(alignment: str) -> tuple[int, int, int]:
    """Return rel_rad, ori, mask_h_w for cardinal or diagonal."""
    if alignment == "cardinal":
        return MASK_HW, CARDINAL_ORI, MASK_HW
    return DIAG_RAD, DIAGONAL_ORI, DIAG_RAD


def _bar_positions(alignment: str, width_led: int, direction: str) -> list[int]:
    """relPos = -relRad : relRad+barW-1 from createMovingBarDiagCorrProtocolG4.m.

    PD is left-to-right in all figures (Gruntman Fig. 1B). For diagonal ori=1
    the native pos order runs ND in azimuth, so PD reverses the sequence.
    """
    rel_rad, _, _ = _motion_params(alignment)
    positions = list(range(-rel_rad, rel_rad + width_led))
    if alignment == "diagonal":
        if direction == "pd":
            positions = positions[::-1]
    elif direction == "nd":
        positions = positions[::-1]
    return positions


def n_steps_for(alignment: str, width_led: int) -> int:
    return len(_bar_positions(alignment, width_led, "pd"))


def moving_bar_field(
    x_led: np.ndarray,
    y_led: np.ndarray,
    step_k: int,
    *,
    alignment: str,
    width_led: int,
    contrast: str = "pc",
    direction: str = "pd",
) -> np.ndarray:
    """Return luminance field for step k (bar × rectangular presentation mask)."""
    positions = _bar_positions(alignment, width_led, direction)
    k = int(np.clip(step_k, 0, len(positions) - 1))
    bar_pos = positions[k]

    _, ori, mask_h_w = _motion_params(alignment)
    if alignment == "cardinal":
        bar_px = _cardinal_bar_leds(bar_pos, width_led)
        mask_px = _cardinal_mask_offsets()
    else:
        bar_px = _diagonal_bar_leds(bar_pos, width_led)
        mask_px = _rectangle_mask_offsets(mask_h_w, MASK_HH, ori)

    val = BRIGHT if contrast == "pc" else DARK
    lum = np.full((len(y_led), len(x_led)), BG, dtype=float)
    x_index = {int(v): i for i, v in enumerate(x_led)}
    y_index = {int(v): j for j, v in enumerate(y_led)}

    for dx, dy in bar_px:
        if (int(dx), int(dy)) not in mask_px:
            continue
        if int(dx) not in x_index or int(dy) not in y_index:
            continue
        lum[y_index[int(dy)], x_index[int(dx)]] = val
    return lum


def draw_field(
    ax: plt.Axes,
    field: np.ndarray,
    x_led: np.ndarray,
    y_led: np.ndarray,
    title: str,
    *,
    show_axes: bool = False,
) -> None:
    x_deg = x_led * LED_DEG
    y_deg = y_led * LED_DEG
    extent = [
        x_deg[0] - LED_DEG / 2,
        x_deg[-1] + LED_DEG / 2,
        y_deg[0] - LED_DEG / 2,
        y_deg[-1] + LED_DEG / 2,
    ]
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
    ax.plot(0, 0, "+", color="#C20078", ms=10, mew=1.5)

    ax.set_title(title, fontsize=7, pad=2)
    if show_axes:
        ax.set_xlabel("Azimuth (°)")
        ax.set_ylabel("Elevation (°)")
    else:
        ax.set_xticks([])
        ax.set_yticks([])


def plot_figure(out_path: Path) -> None:
    widths = (4, 1)
    panel_w = 1.35
    n_rows = len(widths) * 2
    max_steps = max(n_steps_for(al, w) for w in widths for al in ("cardinal", "diagonal"))
    fig = plt.figure(figsize=(panel_w * max_steps + 0.8, panel_w * (1.35 * n_rows) + 1.4))
    outer = fig.add_gridspec(
        n_rows, 1, height_ratios=[1] * n_rows, hspace=0.52, left=0.06, right=0.99, top=0.93, bottom=0.06
    )

    x_led = np.arange(-10, 11)
    y_led = np.arange(-10, 11)

    row_idx = 0
    for width_led in widths:
        w_deg = width_led * LED_DEG
        for alignment, fig_ref in (("cardinal", "Fig. 3"), ("diagonal", "Fig. 1")):
            n_steps = n_steps_for(alignment, width_led)
            gs_row = outer[row_idx].subgridspec(1, max_steps, wspace=0.06)
            row_title = (
                f"{alignment.capitalize()} PD–ND  ({fig_ref})  —  "
                f"{n_steps} steps × {STEP_MS:.0f} ms  —  {width_led} LED ({w_deg:.2g}°)"
            )
            for k in range(n_steps):
                t_ms = k * STEP_MS
                field = moving_bar_field(
                    x_led,
                    y_led,
                    k,
                    alignment=alignment,
                    width_led=width_led,
                    contrast="pc",
                    direction="pd",
                )
                ax = fig.add_subplot(gs_row[0, k])
                draw_field(
                    ax,
                    field,
                    x_led,
                    y_led,
                    f"step {k + 1}\n{t_ms:.0f} ms",
                    show_axes=(row_idx == n_rows - 1 and k == 0),
                )
                if k == 0:
                    ax.annotate(
                        row_title,
                        xy=(0.0, 1.22),
                        xycoords="axes fraction",
                        ha="left",
                        va="bottom",
                        fontsize=8.5,
                        fontweight="bold",
                    )
            row_idx += 1

    width_str = ", ".join(f"{w} LED ({w * LED_DEG:.2g}°)" for w in widths)
    fig.suptitle(
        f"Gruntman 2021 — Cardinal vs diagonal moving bar  "
        f"(T4 PC bright, bar widths: {width_str}; "
        f"{BAR_HEIGHT} LED tall, {SPEED_DEG_S:.0f}° s$^{{-1}}$)",
        fontsize=10,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "Magenta + = RF centre.  "
        f"Mask: cardinal [{MASK_HW},{MASK_HH}] LED half-size; "
        f"diagonal [{DIAG_RAD},{MASK_HH}].  "
        "Bar via generateBarFrameByInds + makeRectangleMask (G4 protocol).  "
        "w=1: 9 / 13 steps; w=4: 12 / 16 steps.",
        ha="center",
        fontsize=7.5,
        color="#444",
    )
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", type=Path, default=REPO / "cardinal_vs_diagonal.png")
    args = ap.parse_args()
    plot_figure(args.output)


if __name__ == "__main__":
    main()
