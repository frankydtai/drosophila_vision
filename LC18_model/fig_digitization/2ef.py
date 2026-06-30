"""Digitize the Fig. 2E (speed tuning) / 2F (looming) line plots of Klapoetke
et al. 2022 (Neuron) into numbers -- means only (no error bars).

Layout (one row per LC cell type, dendrogram order):
    LC18, LC21, LC11, LC25, LC15, LPLC2, LC4, LPLC1, LC17, LC12
Each row has an E panel (speed tuning) and an F panel (looming). Every panel
plots normalized peak dF/F (0-100%, y) against object speed (deg/s, log x) with
two curves:
    bright = green  circles  (open / filled = significance, ignored here)
    dark   = purple squares

Algorithm (2D shape template matching at fixed speed columns)
-------------------------------------------------------------
The speeds (x positions) are FIXED, so we do NOT search for x. Instead:

1. Axis calibration:
   * Y, per row: the gray "[" bracket -> top = 100%, bottom = 0% (E/F share rows).
   * X, per column: gray decade ticks give pixel positions of 10/100/1000 deg/s
     (a log mapping). The shared speed grid is detected once per column by
     stacking all rows' coloured-pixel columns and peak-picking.
2. Marker centre by template matching (handles open/filled AND occlusion):
   * bright -> a circle-ring template matched on the green mask.
   * dark   -> a square-ring template matched on the purple mask, scored only
     over template pixels NOT occluded by green (green is drawn on top of
     purple), so green overlap no longer biases the dark estimate.
   At each fixed speed column we slide the template in a small (cy, cx) window
   and take the position of maximum normalized response; a point is recorded
   only if the best score clears a threshold (so empty panels, e.g. LPLC2's
   missing speed tuning, and rows with fewer points produce fewer points).
3. Map (x, y) -> (speed, dF/F %) and write CSVs + a verification overlay.

Run with the project venv:  .venv/bin/python fig_digitization/2ef.py
"""

from __future__ import annotations

import os
import numpy as np
import fitz  # PyMuPDF

# --------------------------------------------------------------------------- #
# Configuration (measured against this paper.pdf at DPI below)                 #
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(_HERE, "..", "paper.pdf")
PAGE_INDEX = 3
DPI = 300
OUT_DIR = os.path.join(_HERE, "2ef")

CELLS = ["LC18", "LC21", "LC11", "LC25", "LC15",
         "LPLC2", "LC4", "LPLC1", "LC17", "LC12"]

ROW_TOP = 591
ROW_BOTTOM_TOP = 1393
ROW_PITCH = (ROW_BOTTOM_TOP - ROW_TOP) / 9.0
BRACKET_X_E = 1786

COLUMNS = {
    "E": {"x10": 1796.0, "decade": 70.5, "x_range": (1798, 1948)},
    "F": {"x10": 1992.5, "decade": 70.75, "x_range": (1986, 2140)},
}

# marker glyph geometry (px, measured) and template ring thickness
SQUARE_SIZE = 11        # filled-square template (solid detector)
HOLLOW_RING_SIZE = 13   # square-ring template (original hollow detector)
CIRCLE_SIZE = 13
RING_THICK = 3
HALF = CIRCLE_SIZE // 2

# template-match acceptance and peak spacing
GREEN_ACCEPT = 0.45    # min fraction of circle ring that is green
SOLID_ACCEPT = 0.30    # min purple fraction (filled/ring max) for the solid detector
HOLLOW_ACCEPT = 0.50   # min visible-ring purple fraction for the hollow detector
PURPLE_MIN_VISIBLE = 0.45  # min fraction of square ring not occluded by green (hollow)
MARKER_MIN_SEP = 12    # min x-separation between two data points (px)


# --------------------------------------------------------------------------- #
# Rendering / colour masks / templates                                         #
# --------------------------------------------------------------------------- #
def render_page() -> np.ndarray:
    doc = fitz.open(PDF_PATH)
    pix = doc[PAGE_INDEX].get_pixmap(dpi=DPI)
    arr = np.frombuffer(pix.samples, np.uint8)
    return arr.reshape(pix.height, pix.width, pix.n)[:, :, :3].astype(int)


def colour_masks(rgb: np.ndarray):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    s = rgb.sum(2)
    not_extreme = (s > 90) & (s < 720)
    green = not_extreme & (g > r + 8) & (g > b + 8)
    purple = not_extreme & (r > g + 8) & (b > g + 8)
    return green, purple


def dilate(mask: np.ndarray, k: int = 1) -> np.ndarray:
    """Cheap binary dilation by k px (4-neighbourhood, separable)."""
    out = mask.copy()
    for _ in range(k):
        out[:-1, :] |= mask[1:, :]
        out[1:, :] |= mask[:-1, :]
        out[:, :-1] |= mask[:, 1:]
        out[:, 1:] |= mask[:, :-1]
        mask = out.copy()
    return out


def square_ring(size=SQUARE_SIZE, t=RING_THICK) -> np.ndarray:
    T = np.zeros((size, size), bool)
    T[:t, :] = T[-t:, :] = T[:, :t] = T[:, -t:] = True
    return T


def square_filled(size=SQUARE_SIZE) -> np.ndarray:
    return np.ones((size, size), bool)


def circle_ring(size=CIRCLE_SIZE, t=RING_THICK) -> np.ndarray:
    r = (size - 1) / 2.0
    yy, xx = np.mgrid[0:size, 0:size]
    d = np.sqrt((xx - r) ** 2 + (yy - r) ** 2)
    return (d <= r + 0.5) & (d >= r - t)


# --------------------------------------------------------------------------- #
# Y calibration (brackets)                                                     #
# --------------------------------------------------------------------------- #
def detect_brackets(rgb: np.ndarray, xcol: int):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    gray = (np.abs(r - g) < 22) & (np.abs(g - b) < 22) & \
           (rgb.sum(2) > 250) & (rgb.sum(2) < 640)
    col = gray[:, xcol - 1:xcol + 2].any(1)
    ys = np.where(col[470:1500])[0] + 470
    segs, s, prev = [], None, None
    for y in ys:
        if s is None:
            s = prev = y
        elif y - prev <= 12:
            prev = y
        else:
            if prev - s > 25:
                segs.append((s, prev))
            s = prev = y
    if s is not None and prev - s > 25:
        segs.append((s, prev))
    return segs


def row_y_calibration(rgb: np.ndarray):
    segs = detect_brackets(rgb, BRACKET_X_E)
    height = int(np.median([b - a for a, b in segs]))
    rows = []
    for k in range(10):
        y_top = ROW_TOP + k * ROW_PITCH
        match = [s for s in segs if abs(s[0] - y_top) < ROW_PITCH * 0.4]
        if match:
            rows.append(match[0])
        else:
            yt = int(round(y_top))
            rows.append((yt, yt + height))
    return rows


# --------------------------------------------------------------------------- #
# Shared speed grid (x positions) per column                                   #
# --------------------------------------------------------------------------- #
def peak_pick(resp: np.ndarray, min_sep: int, min_h: float):
    """Greedy peak picking: take the strongest x, suppress its +/-min_sep
    neighbourhood, repeat. Guarantees one peak per marker (no doubles)."""
    resp = resp.copy()
    chosen = []
    while True:
        x = int(resp.argmax())
        if resp[x] < min_h:
            break
        chosen.append(x)
        resp[max(0, x - min_sep):x + min_sep + 1] = -1
    return sorted(chosen)


# --------------------------------------------------------------------------- #
# Template matching for one marker at a fixed speed column                     #
# --------------------------------------------------------------------------- #
def green_response(green, xa, xb, y_lo, y_hi, T):
    """For each x, best circle-ring match score over cy (and the best cy)."""
    Tn = T.sum()
    resp = np.zeros(xb - xa)
    ybest = np.zeros(xb - xa, int)
    for xi, cx in enumerate(range(xa, xb)):
        best, by = 0.0, 0
        for cy in range(y_lo + HALF, y_hi - HALF):
            patch = green[cy - HALF:cy + HALF + 1, cx - HALF:cx + HALF + 1]
            if patch.shape != T.shape:
                continue
            sc = (patch & T).sum() / Tn
            if sc > best:
                best, by = sc, cy
        resp[xi], ybest[xi] = best, by
    return resp, ybest


def purple_response_solid(purple, green_occ, xa, xb, y_lo, y_hi, Tfill, Tring):
    """Solid-square detector (unchanged from the filled-marker fix).

    Scores ``max(filled_frac, ring_frac)`` with both fractions normalised by the
    *full* template area, so the peak lands on the dense purple block of a
    filled lavender square even when it overlaps a green circle.  Used ONLY for
    markers classified as solid; hollow squares are handled separately.
    """
    half = Tfill.shape[0] // 2
    nf, nr = Tfill.sum(), Tring.sum()
    resp = np.zeros(xb - xa)
    ybest = np.zeros(xb - xa, int)
    for xi, cx in enumerate(range(xa, xb)):
        best, by = 0.0, 0
        for cy in range(y_lo + half, y_hi - half):
            sl = (slice(cy - half, cy + half + 1), slice(cx - half, cx + half + 1))
            pp = purple[sl]
            if pp.shape != Tfill.shape:
                continue
            sc = max((pp & Tfill).sum() / nf, (pp & Tring).sum() / nr)
            if sc > best:
                best, by = sc, cy
        resp[xi], ybest[xi] = best, by
    return resp, ybest


def purple_response_hollow(purple, green_occ, xa, xb, y_lo, y_hi, T):
    """Hollow-square detector (the ORIGINAL method, kept verbatim).

    Scores the fraction of the *visible* (green-unoccluded) square ring that is
    purple.  The square-ring template is exactly the shape of a hollow marker,
    so it locks sharply onto hollow squares and -- with the high HOLLOW_ACCEPT
    threshold -- the thin connecting line never clears the bar.  Used ONLY for
    markers classified as hollow, so its known downward bias on solid squares
    that overlap green circles is irrelevant here.
    """
    half = T.shape[0] // 2
    Tn = T.sum()
    resp = np.zeros(xb - xa)
    ybest = np.zeros(xb - xa, int)
    for xi, cx in enumerate(range(xa, xb)):
        best, by = 0.0, 0
        for cy in range(y_lo + half, y_hi - half):
            sl = (slice(cy - half, cy + half + 1), slice(cx - half, cx + half + 1))
            pp = purple[sl]
            if pp.shape != T.shape:
                continue
            visible = T & ~green_occ[sl]
            denom = visible.sum()
            if denom < PURPLE_MIN_VISIBLE * Tn:
                continue
            sc = (pp & visible).sum() / denom
            if sc > best:
                best, by = sc, cy
        resp[xi], ybest[xi] = best, by
    return resp, ybest


# --------------------------------------------------------------------------- #
# Pixel -> data mappings                                                       #
# --------------------------------------------------------------------------- #
# the speeds are a fixed grid (same for every cell); pixel-derived values only
# jitter around it, so we snap each detection to the nominal speed.
CANON_SPEEDS = {"E": [25, 50, 100, 200, 400, 800], "F": [25, 100]}

# Everything is detected with the original hollow (square-ring) algorithm.
# These six solid squares sit on top of a green circle where the ring detector
# slides onto the green rim, so they alone are taken from the solid (filled)
# detector.  Keys are (panel_tag, cell, snapped_speed).
OVERRIDE_SOLID = {
    ("E", "LPLC1", 25.0), ("E", "LPLC1", 100.0),
    ("E", "LC12", 25.0),  ("E", "LC12", 400.0),
    ("F", "LC21", 25.0),  ("F", "LC21", 100.0),
    ("F", "LC12", 100.0),
}


def x_to_speed(x, col):
    return 10.0 ** (1.0 + (x - col["x10"]) / col["decade"])


def snap_speed(sp, tag):
    """Snap a measured speed to the nearest nominal speed (nearest in log)."""
    cands = CANON_SPEEDS[tag]
    return min(cands, key=lambda c: abs(np.log10(sp) - np.log10(c)))


def y_to_dff(y, y100, y0):
    return (y0 - y) / (y0 - y100) * 100.0


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rgb = render_page()
    print(f"rendered page {PAGE_INDEX} at {DPI} dpi -> {rgb.shape[1]}x{rgb.shape[0]}")
    green, purple = colour_masks(rgb)
    green_occ = dilate(green, 1)                   # green footprint that occludes purple
    rows = row_y_calibration(rgb)
    Tfill, Tring, Tcr = square_filled(), square_ring(), circle_ring()
    Tring_h = square_ring(HOLLOW_RING_SIZE)        # original hollow-square template

    # ------------------------------------------------------------------ #
    # Detect bright once, and dark with TWO independent detectors:        #
    #   solid  detector -> reproduces the current (filled-fix) output     #
    #   hollow detector -> reproduces the original (ring) output          #
    # Each detector is fed through the SAME pipeline, then we keep solid  #
    # squares from the solid pipeline and hollow squares from the hollow  #
    # pipeline, so neither set of results changes.                        #
    # ------------------------------------------------------------------ #
    def panel_bounds(ri):
        y100, y0 = rows[ri]
        return y100, y0, int(y100 - 16), int(y0 + 10)

    bright_records, dark_solid, dark_hollow = [], [], []
    for ri, cell in enumerate(CELLS):
        y100, y0, y_lo, y_hi = panel_bounds(ri)
        for tag, col in COLUMNS.items():
            xa, xb = col["x_range"]
            gr, gy = green_response(green, xa, xb, y_lo, y_hi, Tcr)
            for xi in peak_pick(gr, MARKER_MIN_SEP, GREEN_ACCEPT):
                cx, cy = xa + xi, int(gy[xi])
                bright_records.append(("bright", tag, cell, cx, cy,
                                       float(gr[xi]), y100, y0, col))
            sr, sy = purple_response_solid(purple, green_occ, xa, xb,
                                           y_lo, y_hi, Tfill, Tring)
            for xi in peak_pick(sr, MARKER_MIN_SEP, SOLID_ACCEPT):
                cx, cy = xa + xi, int(sy[xi])
                dark_solid.append(("dark", tag, cell, cx, cy,
                                   float(sr[xi]), y100, y0, col))
            hr, hy = purple_response_hollow(purple, green_occ, xa, xb,
                                            y_lo, y_hi, Tring_h)
            for xi in peak_pick(hr, MARKER_MIN_SEP, HOLLOW_ACCEPT):
                cx, cy = xa + xi, int(hy[xi])
                dark_hollow.append(("dark", tag, cell, cx, cy,
                                    float(hr[xi]), y100, y0, col))

    def to_records(raw):
        out = []
        for pol, tag, cell, cx, cy, score, y100, y0, col in raw:
            out.append((tag, cell, pol, snap_speed(x_to_speed(cx, col), tag),
                        y_to_dff(cy, y100, y0), score, cx, cy, False))
        return out

    cur = _dedup_and_fill(to_records(bright_records + dark_solid))    # solid algo
    orig = _dedup_and_fill(to_records(bright_records + dark_hollow))  # hollow algo
    cur_dark = {(r[0], r[1], r[3]): r for r in cur if r[2] == "dark"}

    # Default: every point from the original hollow pipeline.  Only the six
    # listed solid squares are overridden with the solid-detector result.
    bright = [r for r in orig if r[2] == "bright"]
    dark = {(r[0], r[1], r[3]): r for r in orig if r[2] == "dark"}
    for key in OVERRIDE_SOLID:
        if key in cur_dark:
            dark[key] = cur_dark[key]
    records = bright + list(dark.values())

    cell_order = {c: i for i, c in enumerate(CELLS)}
    records.sort(key=lambda r: (cell_order[r[1]], r[0], r[2], r[3]))
    print(f"dark points: {len(dark)} ({len(OVERRIDE_SOLID)} overridden with solid algo)")

    combo = os.path.join(OUT_DIR, "fig2_ef_all_celltypes_long.csv")
    with open(combo, "w") as f:
        f.write("figure_panel,cell_type,polarity,speed_dps,dff_percent,filled\n")
        for r in records:
            f.write(f"2{r[0]},{r[1]},{r[2]},{r[3]:.1f},{r[4]:.1f},{int(r[8])}\n")
    print(f"wrote {combo}  ({len(records)} points)")

    for cell in CELLS:
        path = os.path.join(OUT_DIR, f"Fig2EF_{cell}.csv")
        rows_c = sorted([r for r in records if r[1] == cell],
                        key=lambda r: (r[0], r[2], r[3]))
        with open(path, "w") as f:
            f.write("figure_panel,polarity,speed_dps,dff_percent,filled\n")
            for r in rows_c:
                f.write(f"2{r[0]},{r[2]},{r[3]:.1f},{r[4]:.1f},{int(r[8])}\n")
        print(f"  {cell:6s} {len(rows_c):2d} pts -> {os.path.basename(path)}")

    _save_wide_table(records)
    _save_overlay(rgb, records)


def _save_wide_table(records):
    """Wide pivot: rows = condition x speed, columns = cell types, cells dF/F%."""
    val = {(r[0], r[2], r[3], r[1]): r[4] for r in records}  # panel,pol,speed,cell
    conditions = [
        ("speed (bright)", "E", "bright", CANON_SPEEDS["E"]),
        ("speed (dark)",   "E", "dark",   CANON_SPEEDS["E"]),
        ("looming (bright)", "F", "bright", CANON_SPEEDS["F"]),
        ("looming (dark)",   "F", "dark",   CANON_SPEEDS["F"]),
    ]
    path = os.path.join(OUT_DIR, "fig2_ef_wide_by_type.csv")
    with open(path, "w") as f:
        f.write("condition,speed_dps," + ",".join(CELLS) + "\n")
        for label, tag, pol, speeds in conditions:
            for sp in speeds:
                cells = [val.get((tag, pol, float(sp), c), "")
                         for c in CELLS]
                cells = [f"{v:.1f}" if v != "" else "" for v in cells]
                f.write(f"{label},{sp}," + ",".join(cells) + "\n")
    print(f"wrote {path}")


def _dedup_and_fill(records):
    """Collapse snap collisions (keep highest score per speed) and fill any
    speed where bright has a point but dark does not, copying the bright dF/F
    (same number as green) so both curves share the identical speed grid."""
    best = {}
    for r in records:
        k = (r[0], r[1], r[2], r[3])          # panel, cell, polarity, speed
        if k not in best or r[5] > best[k][5]:
            best[k] = r
    records = list(best.values())

    bright = {(r[0], r[1], r[3]): r for r in records if r[2] == "bright"}
    dark_keys = {(r[0], r[1], r[3]) for r in records if r[2] == "dark"}
    for (tag, cell, sp), br in bright.items():
        if (tag, cell, sp) not in dark_keys:
            records.append((tag, cell, "dark", sp, br[4], br[5],
                            br[6], br[7], True))   # filled from bright

    cell_order = {c: i for i, c in enumerate(CELLS)}
    records.sort(key=lambda r: (cell_order[r[1]], r[0], r[2], r[3]))
    return records


def _save_overlay(rgb, records):
    try:
        from PIL import Image, ImageDraw
    except Exception as e:  # pragma: no cover
        print("PIL unavailable, skipping overlay:", e)
        return
    x0, y0, x1, y1 = 1690, 560, 2160, 1470
    im = Image.fromarray(rgb.astype(np.uint8)).crop((x0, y0, x1, y1)).convert("RGB")
    d = ImageDraw.Draw(im)
    h = HALF
    for tag, cell, pol, sp, dff, score, xpx, ypx, filled in records:
        if filled:                      # do not circle copied (filled) points
            continue
        cx, cy = xpx - x0, ypx - y0
        if pol == "bright":
            d.ellipse([cx - h, cy - h, cx + h, cy + h], outline=(255, 0, 0), width=2)
        else:
            d.rectangle([cx - h, cy - h, cx + h, cy + h], outline=(0, 0, 255), width=2)
    out = os.path.join(OUT_DIR, "fig2_ef_detected_overlay.png")
    im.resize((im.width * 2, im.height * 2)).save(out)
    print(f"wrote {out}  (red circle=bright, blue square=dark)")


if __name__ == "__main__":
    main()
