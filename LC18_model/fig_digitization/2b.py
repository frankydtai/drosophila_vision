"""Digitize the black population-average calcium traces of Fig. 2B of Klapoetke
et al. 2022 (Neuron) into numbers.

Fig. 2B is a 10 (cell type) x 7 (motion stimulus) grid of calcium traces; each
panel overlays single-cell traces (in the cell type's colour) with the
population-average trace in BLACK. Only the black population-average traces are
extracted here, and -- per request -- only the first two stimulus blocks:

    Looming        : dark, bright
    Moving square  : 4.5 deg, 15 deg

giving a 10 x 4 set of traces.

Method
------
1. Render the figure page at high DPI.
2. Geometry (measured against this paper.pdf at the DPI below): the 7 trace
   panels are ~101 px wide, grouped into blocks separated by thin white gutters;
   the 10 rows are evenly spaced. We keep the first 4 columns (blocks 1-2).
3. In each panel, the black trace = dark + low-saturation pixels (the coloured
   single-cell traces are saturated, the background is light gray). For every
   pixel column we take the median y of the black pixels -> trace y(x).
4. Calibrate with the figure's own scale bars: 2 s = 20 px (-> 10 px/s) and
   100% dF/F = 56 px. Time starts at the panel's left edge; dF/F is measured
   upward from each panel's own resting baseline (median trace level).
5. Write one CSV per (cell type, stimulus) with columns time_s, dff_percent,
   plus a combined long CSV and a verification PNG.

Run with the project venv:  .venv/bin/python fig_digitization/2b.py
"""

from __future__ import annotations

import os
import numpy as np
import fitz  # PyMuPDF

# --------------------------------------------------------------------------- #
# Configuration (measured against this paper.pdf at DPI below)                 #
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(_HERE, "..", "paper.pdf")   # LC18_model/paper.pdf
PAGE_INDEX = 3
DPI = 300
OUT_DIR = os.path.join(_HERE, "2b")

# cell types, top -> bottom (matches the dendrogram order in Fig. 2A/B)
CELLS = ["LC18", "LC21", "LC11", "LC25", "LC15",
         "LPLC2", "LC4", "LPLC1", "LC17", "LC12"]
ROW_Y0, ROW_Y1 = 630, 1429          # center y of first and last row
ROW_HALF = 42                        # +/- y window around a row center

# the 4 requested stimulus columns: (label, x_left, x_right) of the plot box
COLUMNS = [
    ("Looming_dark",       521, 622),
    ("Looming_bright",     622, 723),
    ("MovingSquare_4.5deg", 730, 831),
    ("MovingSquare_15deg",  831, 932),
]
COL_INSET = 3                        # trim a few px off each panel side

# scale-bar calibration (px)
PX_PER_SEC = 20.0 / 2.0              # 2 s scale bar = 20 px
PX_PER_PCT = 56.0 / 100.0           # 100% dF/F scale bar = 56 px


# --------------------------------------------------------------------------- #
def render_page(pdf_path, page_index, dpi):
    pix = fitz.open(pdf_path)[page_index].get_pixmap(dpi=dpi)
    arr = np.frombuffer(pix.samples, np.uint8)
    return arr.reshape(pix.height, pix.width, pix.n)[:, :, :3].astype(int)


def black_mask(rgb):
    """True where a pixel is the black trace: dark and near-neutral (so the
    saturated single-cell colours and the light-gray background are excluded)."""
    mx = rgb.max(2)
    mn = rgb.min(2)
    return (mx < 110) & ((mx - mn) < 45)


def extract_trace(bmask, x0, x1, yc, yhalf):
    """Per-column median y of black pixels inside the panel's row slot.

    Returns (xs, ys) in image pixel coords; columns without a black pixel are
    skipped and linearly interpolated afterwards.
    """
    ya, yb = yc - yhalf, yc + yhalf
    xs, ys = [], []
    for x in range(x0, x1):
        col = np.where(bmask[ya:yb, x])[0]
        if len(col):
            xs.append(x)
            ys.append(ya + float(np.median(col)))
    if not xs:
        return None, None
    xs = np.array(xs)
    ys = np.array(ys)
    full_x = np.arange(x0, x1)
    full_y = np.interp(full_x, xs, ys)
    return full_x, full_y


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rgb = render_page(PDF_PATH, PAGE_INDEX, DPI)
    bmask = black_mask(rgb)
    print(f"rendered page {PAGE_INDEX} @ {DPI} dpi -> {rgb.shape[1]}x{rgb.shape[0]}")

    row_centers = np.linspace(ROW_Y0, ROW_Y1, len(CELLS))
    long_rows = []
    traces = {}  # (cell, col) -> (t, dff)

    for ci, (clabel, cx0, cx1) in enumerate(COLUMNS):
        x0, x1 = cx0 + COL_INSET, cx1 - COL_INSET
        for ri, cell in enumerate(CELLS):
            yc = int(round(row_centers[ri]))
            xs, ys = extract_trace(bmask, x0, x1, yc, ROW_HALF)
            if xs is None:
                print(f"  !! no black trace for {cell} / {clabel}")
                continue
            # calibrate: baseline = resting level (median y of this trace)
            baseline = np.median(ys)
            t = (xs - x0) / PX_PER_SEC
            dff = (baseline - ys) / PX_PER_PCT       # up = positive dF/F
            traces[(cell, clabel)] = (t, dff)

            path = os.path.join(OUT_DIR, f"Fig2B_{cell}_{clabel}.csv")
            with open(path, "w") as f:
                f.write("time_s,dff_percent\n")
                for ti, vi in zip(t, dff):
                    f.write(f"{ti:.3f},{vi:.2f}\n")
            for ti, vi in zip(t, dff):
                long_rows.append((cell, clabel, ti, vi))
            print(f"  {cell:6s} {clabel:20s} n={len(t):3d} "
                  f"peak={dff.max():6.1f}%  -> {os.path.basename(path)}")

    combo = os.path.join(OUT_DIR, "fig2b_black_traces_long.csv")
    with open(combo, "w") as f:
        f.write("cell_type,stimulus,time_s,dff_percent\n")
        for cell, clabel, ti, vi in long_rows:
            f.write(f"{cell},{clabel},{ti:.3f},{vi:.2f}\n")
    print(f"wrote {combo}  ({len(long_rows)} rows)")

    _save_verification_png(traces)


def _save_verification_png(traces):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print("matplotlib unavailable, skipping verification png:", e)
        return
    col_labels = [c[0] for c in COLUMNS]
    fig, axes = plt.subplots(len(CELLS), len(col_labels),
                             figsize=(11, 16), sharex=True)
    ymax = max((d.max() for _, d in traces.values()), default=100)
    for ri, cell in enumerate(CELLS):
        for ci, clabel in enumerate(col_labels):
            ax = axes[ri, ci]
            if (cell, clabel) in traces:
                t, d = traces[(cell, clabel)]
                ax.plot(t, d, "k", lw=1)
            ax.set_ylim(-20, ymax * 1.05)
            ax.set_xticks([]); ax.set_yticks([])
            if ci == 0:
                ax.set_ylabel(cell, fontsize=9, rotation=0, ha="right", va="center")
            if ri == 0:
                ax.set_title(clabel, fontsize=8)
    fig.suptitle("Fig 2B black population-average traces (digitized)", y=0.995)
    out = os.path.join(OUT_DIR, "fig2b_black_traces_verification.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
