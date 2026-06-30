"""Digitize the Fig. 2I / 2J 2D size-tuning heat maps of Klapoetke et al. 2022
(Neuron) into numbers.

The raw response matrices for Fig. 2I/J were never deposited (only the LC18
model and the connectome analysis are public). This script recovers an
approximate version of the underlying data straight from the published figure
in ``paper.pdf``:

  Fig 2I (looming-responsive) : LPLC2, LC4, LPLC1, LC17, LC12
  Fig 2J (object-responsive)  : LC18, LC21, LC11, LC25, LC15

Each panel is a filled-contour map of normalized peak dF/F (magma colormap,
0-100%) over a 7x7 grid of object width (parallel to motion) x object height
(orthogonal to motion), with sizes [2, 4.5, 9, 15, 30, 45, 90] deg on both axes.

Method
------
1. Render the figure page at high DPI.
2. Auto-detect, inside each row's y-band, the 5 panel boxes (non-white columns)
   and the per-panel x-tick / per-row y-tick pixel positions (tick marks just
   outside the plot box). Ticks are the true sampling grid (the axis spacing is
   non-uniform, so we use measured tick pixels rather than assuming linear).
3. Build a colour->value lookup table from the figure's own printed colorbar
   (self-consistent with the figure's rendering), with pure black -> 0%.
4. At every (width_tick, height_tick) intersection, sample a small pixel window,
   map each pixel through the LUT and take the median -> dF/F %.
5. Write one CSV per cell type (rows = height 90..2, cols = width 2..90) plus a
   long-format combined CSV and a verification PNG.

Run with the project venv:  .venv/bin/python fig_digitization/2ij.py
"""

from __future__ import annotations

import os
import numpy as np
import fitz  # PyMuPDF

# --------------------------------------------------------------------------- #
# Configuration (measured against this paper.pdf at DPI below)                 #
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(_HERE, "..", "paper.pdf")  # LC18_model/paper.pdf
PAGE_INDEX = 3          # 0-based page that contains the Fig. 2 graphic
DPI = 300
OUT_DIR = os.path.join(_HERE, "2ij")

# size values for both axes (deg); order = ascending on x, the y axis is the
# same set printed top->bottom as 90..2
SIZES = [2.0, 4.5, 9.0, 15.0, 30.0, 45.0, 90.0]
WIDTHS = SIZES                      # x axis, left->right
HEIGHTS = list(reversed(SIZES))     # y axis, top->bottom (90 at top)

# Each row: cell types (left->right) and an approximate y-band to search in.
ROWS = [
    {"tag": "I", "cells": ["LPLC2", "LC4", "LPLC1", "LC17", "LC12"],
     "y_band": (1610, 1910)},
    {"tag": "J", "cells": ["LC18", "LC21", "LC11", "LC25", "LC15"],
     "y_band": (1960, 2245)},
]

# colorbar (shared geometry for both rows; x strip is the same, y tracks row)
COLORBAR_X = (2182, 2197)

WHITE_SUM = 680         # rgb.sum < WHITE_SUM  => "not white page"
INK_SUM = 300           # rgb.sum < INK_SUM    => "black ink" (ticks/frame)
SAMPLE_HALFWIN = 6      # +/- px window for sampling a grid intersection


# --------------------------------------------------------------------------- #
# Rendering                                                                    #
# --------------------------------------------------------------------------- #
def render_page(pdf_path: str, page_index: int, dpi: int) -> np.ndarray:
    doc = fitz.open(pdf_path)
    pix = doc[page_index].get_pixmap(dpi=dpi)
    arr = np.frombuffer(pix.samples, np.uint8)
    arr = arr.reshape(pix.height, pix.width, pix.n)[:, :, :3].astype(int)
    return arr


# --------------------------------------------------------------------------- #
# Geometry detection                                                           #
# --------------------------------------------------------------------------- #
def _group(values, gap=6):
    """Collapse runs of nearby pixel coordinates into their mean centers."""
    if len(values) == 0:
        return []
    out, cur = [], [values[0]]
    for v in values[1:]:
        if v - cur[-1] <= gap:
            cur.append(v)
        else:
            out.append(int(round(np.mean(cur))))
            cur = [v]
    out.append(int(round(np.mean(cur))))
    return out


def detect_panels_x(nonwhite: np.ndarray, y0: int, y1: int,
                    x_range=(700, 2200), min_w=120):
    """Return list of (x_left, x_right) for the panel boxes in a row band."""
    col_density = nonwhite[y0:y1, :].mean(0)
    inside = col_density > 0.6
    segs, start = [], None
    for x in range(*x_range):
        if inside[x] and start is None:
            start = x
        elif not inside[x] and start is not None:
            if x - start > min_w:
                segs.append((start, x))
            start = None
    if start is not None and x_range[1] - start > min_w:
        segs.append((start, x_range[1]))
    return segs


def refine_panel_y(nonwhite: np.ndarray, x0: int, x1: int,
                   y_band) -> tuple[int, int]:
    """Tighten a panel's top/bottom using its own column density."""
    yb0, yb1 = y_band
    row_density = nonwhite[yb0:yb1, x0:x1].mean(1)
    ys = np.where(row_density > 0.6)[0]
    return (yb0 + int(ys.min()), yb0 + int(ys.max()))


def detect_xticks(rgb: np.ndarray, x0: int, x1: int, y_bottom: int):
    """X tick centers = black marks in the white strip just below the box."""
    strip = (rgb[y_bottom + 3:y_bottom + 11, :].sum(2) < INK_SUM).mean(0)
    xs = [x for x in range(x0 - 2, x1 + 3) if strip[x] > 0.4]
    return _group(xs)


def detect_yticks(rgb: np.ndarray, x_left: int, y0: int, y1: int):
    """Y tick centers = black marks in the white strip just left of the box."""
    strip = (rgb[:, x_left - 7:x_left - 1].sum(2) < INK_SUM).mean(1)
    ys = [y for y in range(y0 - 2, y1 + 3) if strip[y] > 0.4]
    return _group(ys)


# --------------------------------------------------------------------------- #
# Colorbar -> value LUT                                                        #
# --------------------------------------------------------------------------- #
def build_colorbar_lut(rgb: np.ndarray, y_band, x_strip=COLORBAR_X):
    """Sample the figure's printed magma colorbar -> (colors, values 0..1).

    The bar spans the data range 0% (bottom) to 100% (top). Rows that are
    near-neutral gray/white (the surrounding % text or border) are dropped so
    the LUT is a clean magma gradient. Pure black is appended as value 0.
    """
    yb0, yb1 = y_band
    xs0, xs1 = x_strip
    center = rgb[yb0:yb1, xs0:xs1].mean(1)  # average across bar width
    ys = np.arange(yb0, yb1)

    r, g, b = center[:, 0], center[:, 1], center[:, 2]
    brightness = center.sum(1)
    # neutral gray/white text: channels nearly equal and not dark
    neutral = (np.abs(r - g) < 12) & (np.abs(g - b) < 12) & (brightness > 280)
    not_white = brightness < (3 * 250)
    keep = (~neutral) & not_white
    ys, cols = ys[keep], center[keep]
    if len(ys) < 10:
        raise RuntimeError("colorbar detection failed")

    # value: linear, top of kept strip = 100%, bottom = 0%
    y_top, y_bot = ys.min(), ys.max()
    values = (y_bot - ys) / float(y_bot - y_top)

    colors = np.vstack([cols, [0, 0, 0]])
    values = np.concatenate([values, [0.0]])
    return colors, values


def colors_to_values(pixels: np.ndarray, lut_colors, lut_values):
    """Map an (N,3) array of RGB pixels to values via nearest LUT color."""
    px = pixels.reshape(-1, 1, 3).astype(float)
    d = ((px - lut_colors.reshape(1, -1, 3)) ** 2).sum(2)
    idx = d.argmin(1)
    return lut_values[idx]


# --------------------------------------------------------------------------- #
# Sampling                                                                     #
# --------------------------------------------------------------------------- #
def _fill_mask(win: np.ndarray) -> np.ndarray:
    """True for pixels that are genuine magma fill (not page white nor the
    light contour lines). The magma ramp never exceeds blue ~191, while the
    white page and the pale contour lines have blue >> 200, so the blue
    channel cleanly separates them from even the brightest (100%) fill."""
    return win[..., 2] < 210


def sample_panel(rgb, xticks, yticks, x0, x1, y0, y1,
                 lut_colors, lut_values, half=SAMPLE_HALFWIN, inset=2):
    """Return a (n_height, n_width) matrix of dF/F % for one panel.

    Windows are clipped to stay strictly inside the plot box and page-white /
    contour-line pixels are dropped before taking the median, so boundary ticks
    (width/height = 2 or 90, which sit on the box edge) are not contaminated."""
    nH, nW = len(yticks), len(xticks)
    out = np.full((nH, nW), np.nan)
    xa, xb = x0 + inset, x1 - inset
    ya, yb = y0 + inset, y1 - inset
    for j, yc in enumerate(yticks):
        for i, xc in enumerate(xticks):
            xs = max(xa, xc - half); xe = min(xb, xc + half + 1)
            ys = max(ya, yc - half); ye = min(yb, yc + half + 1)
            win = rgb[ys:ye, xs:xe]
            m = _fill_mask(win)
            pix = win[m] if m.any() else win.reshape(-1, 3)
            vals = colors_to_values(pix.reshape(-1, 3), lut_colors, lut_values)
            out[j, i] = np.median(vals) * 100.0
    return out


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rgb = render_page(PDF_PATH, PAGE_INDEX, DPI)
    nonwhite = (rgb.sum(2) < WHITE_SUM)
    print(f"rendered page {PAGE_INDEX} at {DPI} dpi -> {rgb.shape[1]}x{rgb.shape[0]}")

    long_rows = []   # (row_tag, cell, width, height, dff_pct)
    matrices = {}    # cell -> (matrix, xticks, yticks)

    for row in ROWS:
        tag, cells, y_band = row["tag"], row["cells"], row["y_band"]
        lut_colors, lut_values = build_colorbar_lut(rgb, y_band)
        panels = detect_panels_x(nonwhite, y_band[0] + 30, y_band[1] - 30)
        if len(panels) != len(cells):
            raise RuntimeError(
                f"row {tag}: detected {len(panels)} panels, expected {len(cells)}")

        for cell, (x0, x1) in zip(cells, panels):
            y0, y1 = refine_panel_y(nonwhite, x0, x1, y_band)
            xticks = detect_xticks(rgb, x0, x1, y1)
            yticks = detect_yticks(rgb, x0, y0, y1)
            if len(xticks) != len(WIDTHS) or len(yticks) != len(HEIGHTS):
                raise RuntimeError(
                    f"{cell}: ticks x={len(xticks)} y={len(yticks)} "
                    f"(expected {len(WIDTHS)}x{len(HEIGHTS)})")

            mat = sample_panel(rgb, xticks, yticks, x0, x1, y0, y1,
                               lut_colors, lut_values)
            matrices[cell] = (mat, xticks, yticks)

            # per-cell-type CSV: rows = height (90..2), cols = width (2..90)
            path = os.path.join(OUT_DIR, f"Fig2{tag}_{cell}_sizeTuning.csv")
            header = "height_deg\\width_deg," + ",".join(str(w) for w in WIDTHS)
            with open(path, "w") as f:
                f.write(header + "\n")
                for j, h in enumerate(HEIGHTS):
                    f.write(str(h) + "," +
                            ",".join(f"{mat[j, i]:.1f}" for i in range(len(WIDTHS)))
                            + "\n")
            for j, h in enumerate(HEIGHTS):
                for i, w in enumerate(WIDTHS):
                    long_rows.append((tag, cell, w, h, mat[j, i]))
            print(f"  {tag} {cell:6s} box x[{x0},{x1}] y[{y0},{y1}]  "
                  f"peak={np.nanmax(mat):5.1f}%  ->  {os.path.basename(path)}")

    # combined long-format CSV
    combo = os.path.join(OUT_DIR, "fig2_ij_all_celltypes_long.csv")
    with open(combo, "w") as f:
        f.write("figure_panel,cell_type,object_width_deg,object_height_deg,dff_percent\n")
        for tag, cell, w, h, v in long_rows:
            f.write(f"2{tag},{cell},{w},{h},{v:.1f}\n")
    print(f"wrote {combo}  ({len(long_rows)} rows)")

    _save_verification_png(matrices)


def _save_verification_png(matrices):
    """Re-plot the digitized matrices for a side-by-side eyeball check."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print("matplotlib unavailable, skipping verification png:", e)
        return

    order = [c for row in ROWS for c in row["cells"]]
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for ax, cell in zip(axes.ravel(), order):
        mat = matrices[cell][0]
        im = ax.imshow(mat, cmap="magma", vmin=0, vmax=100, aspect="auto")
        ax.set_title(cell)
        ax.set_xticks(range(len(WIDTHS)))
        ax.set_xticklabels([str(w) for w in WIDTHS], fontsize=7)
        ax.set_yticks(range(len(HEIGHTS)))
        ax.set_yticklabels([str(h) for h in HEIGHTS], fontsize=7)
        ax.set_xlabel("width (deg)", fontsize=8)
        ax.set_ylabel("height (deg)", fontsize=8)
    fig.colorbar(im, ax=axes.ravel().tolist(), label="dF/F %", shrink=0.6)
    out = os.path.join(OUT_DIR, "fig2_ij_digitized_verification.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
