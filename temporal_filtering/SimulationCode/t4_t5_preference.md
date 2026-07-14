# T4 / T5 moving-bar preference (PD / ND × PC / NC)

Gruntman-style cardinal bars: 4 directions × 2 contrasts × 2 widths → 16 conditions per eye.

## Data source

Experimental reference traces come from **[`MatlabFunctions/fig1_ci_digitized.csv`](../../MatlabFunctions/fig1_ci_digitized.csv)** — digitized population Vm from Gruntman et al. (2021) Figure 1 panels **Ci** (T4) and **Cii** (T5). Produced by [`MatlabFunctions/digitize_fig1_ci.py`](../../MatlabFunctions/digitize_fig1_ci.py) from a rendered figure PNG (approximate raster values, not raw lab data).

| CSV column   | Meaning |
| ------------ | ------- |
| `trace_id`   | Lookup key, e.g. `T4_PC_w1_PD` |
| `cell_type`  | `T4` or `T5` (pathway; not subtype a–d) |
| `panel`      | `Ci` (T4) or `Cii` (T5) |
| `contrast`   | `PC` or `NC` (preferred / non-preferred contrast for that pathway) |
| `width_led`  | Bar width in LED units: `1` → `w1`, `4` → `w4` |
| `direction`  | `PD` or `ND` (preferred / null direction for that trace) |
| `time_ms`    | Time relative to bar onset (ms) |
| `vm_mv`      | Population membrane potential (mV) |

The CSV holds **16 traces** per pathway panel: 2 contrasts × 2 widths × 2 directions. Subtype-specific tables below map each `(eye, stimulus)` to one of those keys via `t4_t5_preference.py` (`fig1_key_for_stimulus` → `T4_PC_w1_PD` style ids). Training loads the same data as [`fig1_ci_digitized.npz`](../../MatlabFunctions/fig1_ci_digitized.npz) (`training_config.FIG1_CI_NPZ`).

## Contrast (pathway)

| Pathway            | PC (preferred contrast) | NC (non-preferred contrast) |
| ------------------ | ----------------------- | --------------------------- |
| **T4** (ON)  | bright                  | dark                        |
| **T5** (OFF) | dark                    | bright                      |

## Subtype preferred direction (PD)

Same for T4 and T5 within one eye (Maisak et al. 2013):

| Subtype     | PD (visual field)     |
| ----------- | --------------------- |
| **a** | anterior → posterior |
| **b** | posterior → anterior |
| **c** | dorsal (up)           |
| **d** | ventral (down)        |

## Plot / stimulus axis convention

`moving_bar_stimulus` directions (`right`, `left`, `up`, `down`) are in column **degree** coordinates
(`uv_to_xy` then ``× DEG``: ``x_deg = DEG×v``, ``y_deg = DEG×(u+v/2)``; ``DEG = 4.5``).

| Eye             | anterior → posterior on retina                        | posterior → anterior             |
| --------------- | ------------------------------------------------------ | --------------------------------- |
| **Right** | bar moves**`right`** (+x, left→right on plot) | bar moves**`left`** (−x) |
| **Left**  | bar moves**`left`** (−x, right→left on plot) | bar moves**`right`** (+x) |

**Up / down** are the same for both eyes: **`up`** = dorsal (+y), **`down`** = ventral (−y).

Orthogonal motion (e.g. horizontal bar for T4c/T4d) is not aligned with that subtype’s PD axis; responses are typically weak (marked **—**).

## How to fill the table (two independent axes)

1. **PD vs ND** — from **motion direction only** (does it match that subtype’s PD?).
2. **PC vs NC** — from **contrast + pathway** (T4: bright=PC; T5: dark=PC).

Examples on the **PD** axis for T5a (PD = right on right eye):

| Stimulus       | Motion vs T5a | Contrast           | →                         |
| -------------- | ------------- | ------------------ | -------------------------- |
| right + bright | **PD**  | bright = NC for T5 | **PD**, NC           |
| right + dark   | **PD**  | dark = PC for T5   | **PD**, **PC** |

Do **not** treat “bright + T5” as ND; bright is only NC, not a direction flip.

Tables below: **PD** and **PC** are bold; ND and NC are plain.

---

## Right eye

| Stimulus       | T4a                        | T4b                        | T4c                        | T4d                        | T5a                        | T5b                        | T5c                        | T5d                        |
| -------------- | -------------------------- | -------------------------- | -------------------------- | -------------------------- | -------------------------- | -------------------------- | -------------------------- | -------------------------- |
| right + bright | **PD**, **PC** | ND,**PC**            | —                         | —                         | **PD**, NC           | ND, NC                     | —                         | —                         |
| right + dark   | **PD**, NC           | ND, NC                     | —                         | —                         | **PD**, **PC** | ND,**PC**            | —                         | —                         |
| left + bright  | ND,**PC**            | **PD**, **PC** | —                         | —                         | ND, NC                     | **PD**, NC           | —                         | —                         |
| left + dark    | ND, NC                     | **PD**, NC           | —                         | —                         | ND,**PC**            | **PD**, **PC** | —                         | —                         |
| up + bright    | —                         | —                         | **PD**, **PC** | ND,**PC**            | —                         | —                         | **PD**, NC           | ND, NC                     |
| up + dark      | —                         | —                         | **PD**, NC           | ND, NC                     | —                         | —                         | **PD**, **PC** | ND,**PC**            |
| down + bright  | —                         | —                         | ND,**PC**            | **PD**, **PC** | —                         | —                         | ND, NC                     | **PD**, NC           |
| down + dark    | —                         | —                         | ND, NC                     | **PD**, NC           | —                         | —                         | ND,**PC**            | **PD**, **PC** |

---

## Left eye

| Stimulus       | T4a                        | T4b                        | T4c                        | T4d                        | T5a                        | T5b                        | T5c                        | T5d                        |
| -------------- | -------------------------- | -------------------------- | -------------------------- | -------------------------- | -------------------------- | -------------------------- | -------------------------- | -------------------------- |
| left + bright  | **PD**, **PC** | ND,**PC**            | —                         | —                         | **PD**, NC           | ND, NC                     | —                         | —                         |
| left + dark    | **PD**, NC           | ND, NC                     | —                         | —                         | **PD**, **PC** | ND,**PC**            | —                         | —                         |
| right + bright | ND,**PC**            | **PD**, **PC** | —                         | —                         | ND, NC                     | **PD**, NC           | —                         | —                         |
| right + dark   | ND, NC                     | **PD**, NC           | —                         | —                         | ND,**PC**            | **PD**, **PC** | —                         | —                         |
| up + bright    | —                         | —                         | **PD**, **PC** | ND,**PC**            | —                         | —                         | **PD**, NC           | ND, NC                     |
| up + dark      | —                         | —                         | **PD**, NC           | ND, NC                     | —                         | —                         | **PD**, **PC** | ND,**PC**            |
| down + bright  | —                         | —                         | ND,**PC**            | **PD**, **PC** | —                         | —                         | ND, NC                     | **PD**, NC           |
| down + dark    | —                         | —                         | ND, NC                     | **PD**, NC           | —                         | —                         | ND,**PC**            | **PD**, **PC** |

---

## Quick symmetry

On a subtype’s **PD** axis:

| Stimulus | T4                         | T5                         |
| -------- | -------------------------- | -------------------------- |
| bright   | **PD**, **PC** | **PD**, NC           |
| dark     | **PD**, NC           | **PD**, **PC** |

Left vs right eye: only the stimulus direction for A→P vs P→A is mirrored (`right` ↔ `left`); up/down unchanged.

## References

- **Data:** [`MatlabFunctions/fig1_ci_digitized.csv`](../../MatlabFunctions/fig1_ci_digitized.csv) — Figure 1 Ci/Cii traces (Gruntman et al. 2021), digitized by `digitize_fig1_ci.py`.
- Maisak et al. (2013): T4a–d PD = front-to-back, back-to-front, up, down.
- Gruntman et al. (2021): T4 PC = bright, NC = dark; T5 PC = dark, NC = bright.
- Shinomiya et al. (2019) eLife; Takemura et al. (2017) eLife.
