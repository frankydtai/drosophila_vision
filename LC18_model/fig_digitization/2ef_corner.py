"""Digitize the Fig. 2E/F dark (purple square) markers with a *corner / edge*
detector instead of blob template matching.

Why a separate algorithm
-------------------------
The dark markers are squares (open OR filled) of a FIXED pixel side length.
A blob/area template is fooled by two things that share the purple colour but
are not squares:
    * the thin horizontal poly-line that connects consecutive markers, and
    * the purple error-bar whiskers,
and by the green circles drawn ON TOP of a square (occluding its lower half).

This detector keys on the geometry that ONLY a square has: a top edge plus a
left AND a right vertical edge of the known side length (i.e. the two top
corners).  A horizontal connecting line has no vertical edges; a green circle
has no straight purple edges -- so both are rejected.  Because the side length
is known, once the top edge + verticals (top corners) are located the whole
square is reconstructed downward, which is robust even when a green circle
hides the bottom edge.

x is taken from the fixed speed grid (the bright green circles' columns, or the
log-axis calibration as a fallback), so the connecting line can never move a
point in x.  Only the vertical position cy is searched.

Bright (green circle) detection is reused unchanged from 2ef.py; only the dark
detection is replaced.  Output overwrites the same files in fig_digitization/2ef.

Run:  .venv/bin/python fig_digitization/2ef_corner.py
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("ef", os.path.join(_HERE, "2ef.py"))
ef = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ef)

# --------------------------------------------------------------------------- #
# Dark square geometry (px, measured against this paper.pdf at 300 dpi)        #
# --------------------------------------------------------------------------- #
SIDE = 11               # square side length (measured y-extent of a glyph)
H = SIDE // 2           # half side -> corners at center +/- H
EDGE_T = 2              # edge band thickness sampled along each side
DX_SEARCH = 4           # +/- px search around the nominal x column

TOP_MIN = 0.55          # min purple fraction along the (unoccluded) top edge
VERT_MIN = 0.40         # min purple fraction along each (unoccluded) vertical edge
ABOVE_MAX = 0.35        # max purple fraction just above the top edge (must be clear)


def _frac(seg: np.ndarray, occ: np.ndarray):
    """Purple fraction of an edge band over the pixels NOT hidden by green.

    Returns None when the band is fully occluded (no evidence either way)."""
    visible = ~occ
    n = int(visible.sum())
    if n == 0:
        return None
    return float((seg & visible).sum()) / n


def detect_square(purple, green_occ, cx0, y_lo, y_hi):
    """Locate one dark square near column ``cx0`` by its top two corners.

    Searches (cy, cx) and scores the square border of side SIDE.  A candidate
    is valid only if the top edge and BOTH vertical edges are purple (the
    corner condition that rejects the horizontal connecting line) and the row
    just above the top edge is clear (anchors the top boundary so the detector
    cannot slide down into a filled interior or onto a green circle).

    Returns ``(score, cy, cx)`` of the best square, or None if none qualifies.
    """
    best = None
    for cx in range(cx0 - DX_SEARCH, cx0 + DX_SEARCH + 1):
        for cy in range(y_lo + H + 3, y_hi - H):
            sl_top = (slice(cy - H, cy - H + EDGE_T), slice(cx - H, cx + H + 1))
            sl_bot = (slice(cy + H - EDGE_T + 1, cy + H + 1), slice(cx - H, cx + H + 1))
            sl_lft = (slice(cy - H, cy + H + 1), slice(cx - H, cx - H + EDGE_T))
            sl_rgt = (slice(cy - H, cy + H + 1), slice(cx + H - EDGE_T + 1, cx + H + 1))

            ht = _frac(purple[sl_top], green_occ[sl_top])
            vl = _frac(purple[sl_lft], green_occ[sl_lft])
            vr = _frac(purple[sl_rgt], green_occ[sl_rgt])
            if ht is None or vl is None or vr is None:
                continue
            if not (ht >= TOP_MIN and vl >= VERT_MIN and vr >= VERT_MIN):
                continue

            above = purple[cy - H - 3:cy - H - 1, cx - H:cx + H + 1]
            if above.size and above.mean() > ABOVE_MAX:
                continue

            hb = _frac(purple[sl_bot], green_occ[sl_bot])
            score = ht + vl + vr + (hb if hb is not None else 0.0)
            if best is None or score > best[0]:
                best = (score, cy, cx)
    return best


def _nominal_x(sp, col):
    return int(round(col["x10"] + col["decade"] * (np.log10(sp) - 1.0)))


def main():
    out_dir = ef.OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    rgb = ef.render_page()
    print(f"rendered page {ef.PAGE_INDEX} at {ef.DPI} dpi -> "
          f"{rgb.shape[1]}x{rgb.shape[0]}")
    green, purple = ef.colour_masks(rgb)
    green_occ = ef.dilate(green, 1)
    rows = ef.row_y_calibration(rgb)
    Tcr = ef.circle_ring()

    records = []   # (panel, cell, polarity, speed, dff, score, xpx, ypx, filled)
    for ri, cell in enumerate(ef.CELLS):
        y100, y0 = rows[ri]
        y_lo, y_hi = int(y100 - 16), int(y0 + 10)
        for tag, col in ef.COLUMNS.items():
            xa, xb = col["x_range"]

            # bright (green circles): reuse the existing template matcher; its
            # peak x-positions also give us the precise per-speed x grid.
            gresp, gyb = ef.green_response(green, xa, xb, y_lo, y_hi, Tcr)
            bright_x = {}
            for xi in ef.peak_pick(gresp, ef.MARKER_MIN_SEP, ef.GREEN_ACCEPT):
                cx, cy = xa + xi, int(gyb[xi])
                sp = ef.snap_speed(ef.x_to_speed(cx, col), tag)
                records.append((tag, cell, "bright", sp,
                                ef.y_to_dff(cy, y100, y0),
                                float(gresp[xi]), cx, cy, False))
                bright_x[sp] = cx

            # dark (purple squares): corner detector at each fixed speed column
            ndark = 0
            for sp in ef.CANON_SPEEDS[tag]:
                cx0 = bright_x.get(sp, _nominal_x(sp, col))
                det = detect_square(purple, green_occ, cx0, y_lo, y_hi)
                if det is None:
                    continue
                score, cy, cx = det
                records.append((tag, cell, "dark", sp,
                                ef.y_to_dff(cy, y100, y0),
                                score, cx, cy, False))
                ndark += 1
            nb = sum(1 for r in records if r[1] == cell and r[0] == tag
                     and r[2] == "bright")
            print(f"  {cell:6s} 2{tag}: {nb} bright, {ndark} dark")

    records = ef._dedup_and_fill(records)

    combo = os.path.join(out_dir, "fig2_ef_all_celltypes_long.csv")
    with open(combo, "w") as f:
        f.write("figure_panel,cell_type,polarity,speed_dps,dff_percent,filled\n")
        for r in records:
            f.write(f"2{r[0]},{r[1]},{r[2]},{r[3]:.1f},{r[4]:.1f},{int(r[8])}\n")
    print(f"wrote {combo}  ({len(records)} points)")

    for cell in ef.CELLS:
        path = os.path.join(out_dir, f"Fig2EF_{cell}.csv")
        rows_c = sorted([r for r in records if r[1] == cell],
                        key=lambda r: (r[0], r[2], r[3]))
        with open(path, "w") as f:
            f.write("figure_panel,polarity,speed_dps,dff_percent,filled\n")
            for r in rows_c:
                f.write(f"2{r[0]},{r[2]},{r[3]:.1f},{r[4]:.1f},{int(r[8])}\n")
        print(f"  {cell:6s} {len(rows_c):2d} pts -> {os.path.basename(path)}")

    ef._save_wide_table(records)
    _save_overlay(rgb, records, out_dir)


def _save_overlay(rgb, records, out_dir):
    """Verification overlay written to its OWN file (does not overwrite the
    other algorithm's fig2_ef_detected_overlay.png)."""
    try:
        from PIL import Image, ImageDraw
    except Exception as e:  # pragma: no cover
        print("PIL unavailable, skipping overlay:", e)
        return
    x0, y0, x1, y1 = 1690, 560, 2160, 1470
    im = Image.fromarray(rgb.astype(np.uint8)).crop((x0, y0, x1, y1)).convert("RGB")
    d = ImageDraw.Draw(im)
    for tag, cell, pol, sp, dff, score, xpx, ypx, filled in records:
        if filled:                      # do not mark copied (filled) points
            continue
        cx, cy = xpx - x0, ypx - y0
        if pol == "bright":
            d.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], outline=(255, 0, 0), width=2)
        else:
            d.rectangle([cx - H, cy - H, cx + H, cy + H], outline=(0, 0, 255), width=2)
    out = os.path.join(out_dir, "fig2_ef_corner_overlay.png")
    im.resize((im.width * 2, im.height * 2)).save(out)
    print(f"wrote {out}  (red circle=bright, blue square=dark)")


if __name__ == "__main__":
    main()
