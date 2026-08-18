"""Digitize Gruntman et al. 2018 Fig. 2B Vm traces from 2b.png.

Fig. 2B shows mean T4 responses at five RF positions (−2 … +2)
for four flash durations (160, 80, 40, 20 ms), aligned to stimulus onset
(gray/red bars). Values are approximate raster digitization, not raw data.

Method
------
1. Skip left single-trial noise (mean traces begin near x ≈ 500).
2. Track the core raster colour of each trace. Short crossings and occlusions
   are repaired below with Fig. 2B-specific manual patches.
3. Time: 160-ms gray bar = 289 px, onset x = 518. Voltage uses the separately
   measured 5-mV scale bar in each row.
4. Per trace, rest_y = median y before onset; subtract median pre-stimulus Vm
   (t < −2 ms) so baseline is flat at 0.

Run:  ../.venv/bin/python 2b.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import median_filter

HERE = Path(__file__).resolve().parent
DEFAULT_IMAGE = HERE / "2b.png"
FIGURE = "2b"
OUT_STEM = HERE / f"{FIGURE}_digitized"

TRACE_X0, TRACE_X1 = 500, 1230
STIM_ONSET_X = 518
PX_PER_MS = 289.0 / 160.0
PX_PER_MV = {
    160: 75.0 / 5.0,
    80: 75.0 / 5.0,
    40: 149.0 / 5.0,
    20: 152.0 / 5.0,
}
MAX_DY_PX = 28
MAX_DY_TIERS = (MAX_DY_PX, 56, 96, 160)
PRE_STIM_MS = -2.0

CORE_RGB = {
    "dark_green": np.array((103, 169, 63), dtype=float),
    "light_green": np.array((193, 223, 144), dtype=float),
    "dark_pink": np.array((191, 53, 136), dtype=float),
    "light_pink": np.array((232, 185, 216), dtype=float),
}
CORE_MAX_DISTANCE = 22.0

# Hand patches are deliberately specific to this one published raster. They
# bridge short occlusions without trying to make a general-purpose digitizer.
MANUAL_LINEAR_PATCH_X = {
    (160, "dark_green"): ((575, 680),),
    (160, "light_green"): ((834, 848), (1084, 1110)),
    (80, "dark_green"): ((558, 650), (992, 1075)),
    (80, "light_green"): ((766, 781), (935, 995)),
    (40, "black"): ((565, 576),),
    (40, "dark_green"): ((555, 625), (975, 995)),
    (40, "light_green"): ((787, 807),),
    (20, "dark_green"): ((558, 615),),
}

MANUAL_PIXEL_ANCHORS = {
    (20, "light_green"): (
        (1088, 249.0),
        (1138, 249.0),
        (1145, 243.0),
        (1152, 241.5),
        (1160, 242.5),
        (1168, 247.0),
    ),
}

MANUAL_TRACE_END_X = {
    (160, "light_pink"): 1094,
    (20, "light_green"): 1168,
}

PANELS = [
    (160, 40, 400),
    (80, 410, 840),
    (40, 850, 1275),
    (20, 1285, 1660),
]

TRACES = [
    "black",
    "dark_green",
    "light_green",
    "dark_pink",
    "light_pink",
]

# Output position index keyed by extracted trace colour.
POSITION_IDX = {
    "black": 0,
    "dark_green": -1,
    "light_green": -2,
    "dark_pink": 1,
    "light_pink": 2,
}

def black_mask(rgb: np.ndarray) -> np.ndarray:
    mx = rgb.max(2)
    mn = rgb.min(2)
    return (mx < 90) & ((mx - mn) < 40)


def green_mask(rgb: np.ndarray, *, dark: bool) -> np.ndarray:
    color = "dark_green" if dark else "light_green"
    delta = rgb.astype(float) - CORE_RGB[color]
    return np.sqrt(np.sum(delta * delta, axis=2)) <= CORE_MAX_DISTANCE


def pink_mask(rgb: np.ndarray, *, dark: bool) -> np.ndarray:
    color = "dark_pink" if dark else "light_pink"
    delta = rgb.astype(float) - CORE_RGB[color]
    return np.sqrt(np.sum(delta * delta, axis=2)) <= CORE_MAX_DISTANCE


def line_mask(rgb: np.ndarray, color: str) -> np.ndarray:
    if color == "black":
        return black_mask(rgb)
    if color == "dark_green":
        return green_mask(rgb, dark=True)
    if color == "light_green":
        return green_mask(rgb, dark=False)
    if color == "dark_pink":
        return pink_mask(rgb, dark=True)
    return pink_mask(rgb, dark=False)


def _pick_y(rows: np.ndarray, prev: float | None, color: str) -> float | None:
    del color
    if len(rows) == 0:
        return None
    # Track the centre of the printed ink run. Tracking its nearest edge clamps
    # rounded peaks because the rising and falling thick strokes overlap in y.
    runs = np.split(rows, np.where(np.diff(rows) > 1)[0] + 1)
    centers = np.asarray([np.median(run) for run in runs], dtype=float)
    if prev is None:
        target = float(np.median(rows))
        return float(centers[np.argmin(np.abs(centers - target))])
    for max_dy in MAX_DY_TIERS:
        near = centers[np.abs(centers - prev) <= max_dy]
        if len(near) == 0:
            continue
        idx = int(np.argmin(np.abs(near - prev)))
        return float(near[idx])
    return None


def _seed_x(mask: np.ndarray) -> int | None:
    hits = [x for x in range(TRACE_X0, TRACE_X1) if mask[:, x].any()]
    if not hits:
        return None
    return min(hits, key=lambda x: abs(x - STIM_ONSET_X))


def _walk(mask: np.ndarray, seed_x: int, seed_y: float, color: str) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []

    prev = seed_y
    for x in range(seed_x - 1, TRACE_X0 - 1, -1):
        y = _pick_y(np.where(mask[:, x])[0], prev, color)
        if y is None:
            break
        xs.append(float(x))
        ys.append(y)
        prev = y

    prev = seed_y
    right_xs = [float(seed_x)]
    right_ys = [seed_y]
    for x in range(seed_x + 1, TRACE_X1):
        y = _pick_y(np.where(mask[:, x])[0], prev, color)
        if y is None:
            continue
        right_xs.append(float(x))
        right_ys.append(y)
        prev = y

    return xs[::-1] + right_xs, ys[::-1] + right_ys


def _repair_trace_pixels(
    full_x: np.ndarray,
    full_y: np.ndarray,
    flash_ms: int,
    color: str,
) -> tuple[np.ndarray, np.ndarray]:
    for x0, x1 in MANUAL_LINEAR_PATCH_X.get((flash_ms, color), ()):
        left = int(np.searchsorted(full_x, x0, side="left")) - 1
        right = int(np.searchsorted(full_x, x1, side="right"))
        if left < 0 or right >= len(full_x):
            continue
        patch = slice(left + 1, right)
        full_y[patch] = np.interp(
            full_x[patch],
            (full_x[left], full_x[right]),
            (full_y[left], full_y[right]),
        )

    anchors = MANUAL_PIXEL_ANCHORS.get((flash_ms, color))
    if anchors:
        anchor_x = np.asarray([anchor[0] for anchor in anchors], dtype=float)
        anchor_y = np.asarray([anchor[1] for anchor in anchors], dtype=float)
        use = (full_x >= anchor_x[0]) & (full_x <= anchor_x[-1])
        full_y[use] = np.interp(full_x[use], anchor_x, anchor_y)

    end_x = MANUAL_TRACE_END_X.get((flash_ms, color))
    if end_x is not None:
        keep = full_x <= end_x
        full_x = full_x[keep]
        full_y = full_y[keep]
    return full_x, full_y


def _to_vm(
    full_x: np.ndarray,
    full_y: np.ndarray,
    flash_ms: int,
) -> tuple[np.ndarray, np.ndarray]:
    pre_mask = full_x < STIM_ONSET_X
    rest_y = float(np.median(full_y[pre_mask])) if int(pre_mask.sum()) >= 3 else float(full_y[0])

    if full_x[0] > TRACE_X0:
        pad_x = np.arange(TRACE_X0, full_x[0], dtype=float)
        pad_y = np.full(len(pad_x), rest_y, dtype=float)
        full_x = np.concatenate([pad_x, full_x])
        full_y = np.concatenate([pad_y, full_y])

    time_ms = (full_x - STIM_ONSET_X) / PX_PER_MS
    vm_mv = (rest_y - full_y) / PX_PER_MV[flash_ms]
    pre_vm = vm_mv[time_ms < PRE_STIM_MS]
    if len(pre_vm) >= 3:
        vm_mv = vm_mv - float(np.median(pre_vm))
    return time_ms, vm_mv


def extract_trace(
    panel: np.ndarray,
    color: str,
    flash_ms: int,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    mask = line_mask(panel, color)
    seed_x = _seed_x(mask)
    if seed_x is None:
        return None, None
    seed_y = _pick_y(np.where(mask[:, seed_x])[0], None, color)
    if seed_y is None:
        return None, None
    xs, ys = _walk(mask, seed_x, seed_y, color)
    if len(xs) < 40:
        return None, None

    xs_a = np.asarray(xs, dtype=float)
    ys_a = median_filter(np.asarray(ys, dtype=float), size=5)
    full_x = np.arange(int(xs_a.min()), int(xs_a.max()) + 1, dtype=float)
    full_y = np.interp(full_x, xs_a, ys_a)
    full_x, full_y = _repair_trace_pixels(full_x, full_y, flash_ms, color)
    return _to_vm(full_x, full_y, flash_ms)


def digitize(img: np.ndarray) -> pd.DataFrame:
    rows: list[dict] = []
    for flash_ms, y0, y1 in PANELS:
        panel = img[y0:y1]
        for color in TRACES:
            position_idx = POSITION_IDX[color]
            time_ms, vm_mv = extract_trace(panel, color, flash_ms)
            if time_ms is None:
                print(f"  !! no trace for {flash_ms} ms / {color}")
                continue
            trace_id = f"{flash_ms}ms_pos{position_idx:+d}"
            pre = vm_mv[time_ms < PRE_STIM_MS]
            pre_std = float(np.std(pre)) if len(pre) >= 3 else float("nan")
            for ti, vi in zip(time_ms, vm_mv):
                rows.append(
                    dict(
                        trace_id=trace_id,
                        flash_duration_ms=int(flash_ms),
                        position_idx=int(position_idx),
                        color=color,
                        time_ms=float(ti),
                        vm_mv=float(vi),
                    )
                )
            print(
                f"  {flash_ms:3d} ms pos{position_idx:+d} n={len(time_ms):4d} "
                f"t=[{time_ms[0]:.0f},{time_ms[-1]:.0f}] ms "
                f"pre_std={pre_std:4.2f} mV peak={vm_mv.max():5.1f} mV "
                f"end={vm_mv[-1]:5.1f} mV"
            )
    return pd.DataFrame(rows)


def plot_check(df: pd.DataFrame, path: Path) -> None:
    colors = {
        "black": "#111111",
        "dark_green": "#2d6b1f",
        "light_green": "#b8e06a",
        "dark_pink": "#b03078",
        "light_pink": "#e8b0d0",
    }
    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    for ax, flash_ms in zip(axes, [160, 80, 40, 20]):
        for position_idx in [-2, -1, 0, 1, 2]:
            trace_id = f"{flash_ms}ms_pos{position_idx:+d}"
            sub = df[df.trace_id == trace_id].sort_values("time_ms")
            if sub.empty:
                continue
            ax.plot(
                sub.time_ms,
                sub.vm_mv,
                color=colors[sub.color.iloc[0]],
                lw=1.2,
                label=f"{position_idx:+d}",
            )
        ax.axvspan(0, flash_ms, color="0.75", alpha=0.12, zorder=0)
        ax.axvline(0, color="0.55", lw=0.8)
        ax.axvline(flash_ms, color="0.55", lw=0.8, ls="--")
        ax.annotate(
            "flash start\n0 ms",
            xy=(0, 0.02),
            xycoords=("data", "axes fraction"),
            xytext=(3, 2),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=7,
            color="0.30",
        )
        ax.annotate(
            f"flash end\n{flash_ms} ms",
            xy=(flash_ms, 0.14),
            xycoords=("data", "axes fraction"),
            xytext=(3, 2),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=7,
            color="0.30",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.70, pad=0.5),
        )
        ax.axhline(0, color="0.75", lw=0.5)
        ax.set_ylabel(f"{flash_ms} ms")
    axes[-1].set_xlabel("time (ms)")
    axes[0].legend(fontsize=7, ncol=5, loc="upper right")
    fig.suptitle("Gruntman 2018 Fig. 2B (digitized Vm)", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> int:
    img = np.array(Image.open(DEFAULT_IMAGE).convert("RGB"))
    print(f"loaded {DEFAULT_IMAGE.name}  {img.shape[1]}x{img.shape[0]}")
    df = digitize(img)
    csv_path = OUT_STEM.with_suffix(".csv")
    png_path = OUT_STEM.with_suffix(".png")
    df.to_csv(csv_path, index=False)
    plot_check(df, png_path)
    n_trace = df.trace_id.nunique() if not df.empty else 0
    print(f"wrote {csv_path}  ({len(df)} rows, {n_trace} traces)")
    print(f"wrote {png_path}")
    return 0 if n_trace >= 20 else 1


if __name__ == "__main__":
    raise SystemExit(main())
