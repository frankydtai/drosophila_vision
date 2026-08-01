# T4 / T5 moving-bar preference (PD / ND × PC / NC)

Gruntman-style cardinal bars: 4 directions × 2 contrasts × 2 widths → 16 conditions per eye.

## Data source

Experimental reference traces come from **[`MatlabFunctions/fig1_ci_digitized.csv`](../../MatlabFunctions/fig1_ci_digitized.csv)** — digitized population Vm from Gruntman et al. (2021) Figure 1 panels **Ci** (T4) and **Cii** (T5). Produced by [`MatlabFunctions/digitize_fig1_ci.py`](../../MatlabFunctions/digitize_fig1_ci.py) from a rendered figure PNG (approximate raster values, not raw lab data).

| CSV column   | Meaning |
| ------------ | ------- |
| `trace_id`   | Lookup key, e.g. `T4_PC_w1_PD` |
| `cell`  | `T4` or `T5` (pathway; not subtype a–d) |
| `panel`      | `Ci` (T4) or `Cii` (T5) |
| `contrast`   | `PC` or `NC` (preferred / non-preferred contrast for that pathway) |
| `width_led`  | Bar width in LED units: `1` → `w1`, `4` → `w4` |
| `direction`  | `PD` or `ND` (preferred / null direction for that trace) |
| `time_ms`    | Time relative to bar onset (ms) |
| `vm_mv`      | Population membrane potential (mV) |

The CSV holds **16 traces** per pathway panel: 2 contrasts × 2 widths × 2 directions. Subtype-specific tables below map each `(eye, stimulus)` to one of those keys via `t4_t5_dsi.py` (`fig1_key_for_stimulus` → `T4_PC_w1_PD` style ids). Training loads the same data as [`fig1_ci_digitized.npz`](../../MatlabFunctions/fig1_ci_digitized.npz) (`training_config.FIG1_CI_NPZ`).

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
| **Right** | bar moves **`right`** (+x, left→right on plot) | bar moves **`left`** (−x) |
| **Left**  | bar moves **`left`** (−x, right→left on plot) | bar moves **`right`** (+x) |

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
Each on-axis cell is `PD|ND, PC|NC; peak_w1/peak_w4; DSI_w1/DSI_w4`.
**Fill rules are mandatory — see [DSI rules](#dsi-rules-read-before-editing-eye-tables).** Orthogonal = `—`.

---

## Right eye

| Stimulus       | T4a | T4b | T4c | T4d | T5a | T5b | T5c | T5d |
| -------------- | --- | --- | --- | --- | --- | --- | --- | --- |
| right + bright | **PD**, **PC**; 18.25/23.70; +0.412/+0.514 | ND, **PC**; 7.60/7.60; −0.412/−0.514 | — | — | **PD**, NC; 8.05/13.55; +0.263/+0.246 | ND, NC; 4.70/8.20; −0.263/−0.246 | — | — |
| right + dark   | **PD**, NC; 3.15/16.35; +0.024/+0.389 | ND, NC; 3.00/7.20; −0.024/−0.389 | — | — | **PD**, **PC**; 11.40/25.60; +0.322/+0.469 | ND, **PC**; 5.85/9.25; −0.322/−0.469 | — | — |
| left + bright  | ND, **PC**; 7.60/7.60; −0.412/−0.514 | **PD**, **PC**; 18.25/23.70; +0.412/+0.514 | — | — | ND, NC; 4.70/8.20; −0.263/−0.246 | **PD**, NC; 8.05/13.55; +0.263/+0.246 | — | — |
| left + dark    | ND, NC; 3.00/7.20; −0.024/−0.389 | **PD**, NC; 3.15/16.35; +0.024/+0.389 | — | — | ND, **PC**; 5.85/9.25; −0.322/−0.469 | **PD**, **PC**; 11.40/25.60; +0.322/+0.469 | — | — |
| up + bright    | — | — | **PD**, **PC**; 18.25/23.70; +0.412/+0.514 | ND, **PC**; 7.60/7.60; −0.412/−0.514 | — | — | **PD**, NC; 8.05/13.55; +0.263/+0.246 | ND, NC; 4.70/8.20; −0.263/−0.246 |
| up + dark      | — | — | **PD**, NC; 3.15/16.35; +0.024/+0.389 | ND, NC; 3.00/7.20; −0.024/−0.389 | — | — | **PD**, **PC**; 11.40/25.60; +0.322/+0.469 | ND, **PC**; 5.85/9.25; −0.322/−0.469 |
| down + bright  | — | — | ND, **PC**; 7.60/7.60; −0.412/−0.514 | **PD**, **PC**; 18.25/23.70; +0.412/+0.514 | — | — | ND, NC; 4.70/8.20; −0.263/−0.246 | **PD**, NC; 8.05/13.55; +0.263/+0.246 |
| down + dark    | — | — | ND, NC; 3.00/7.20; −0.024/−0.389 | **PD**, NC; 3.15/16.35; +0.024/+0.389 | — | — | ND, **PC**; 5.85/9.25; −0.322/−0.469 | **PD**, **PC**; 11.40/25.60; +0.322/+0.469 |

---

## Left eye

| Stimulus       | T4a | T4b | T4c | T4d | T5a | T5b | T5c | T5d |
| -------------- | --- | --- | --- | --- | --- | --- | --- | --- |
| left + bright  | **PD**, **PC**; 18.25/23.70; +0.412/+0.514 | ND, **PC**; 7.60/7.60; −0.412/−0.514 | — | — | **PD**, NC; 8.05/13.55; +0.263/+0.246 | ND, NC; 4.70/8.20; −0.263/−0.246 | — | — |
| left + dark    | **PD**, NC; 3.15/16.35; +0.024/+0.389 | ND, NC; 3.00/7.20; −0.024/−0.389 | — | — | **PD**, **PC**; 11.40/25.60; +0.322/+0.469 | ND, **PC**; 5.85/9.25; −0.322/−0.469 | — | — |
| right + bright | ND, **PC**; 7.60/7.60; −0.412/−0.514 | **PD**, **PC**; 18.25/23.70; +0.412/+0.514 | — | — | ND, NC; 4.70/8.20; −0.263/−0.246 | **PD**, NC; 8.05/13.55; +0.263/+0.246 | — | — |
| right + dark   | ND, NC; 3.00/7.20; −0.024/−0.389 | **PD**, NC; 3.15/16.35; +0.024/+0.389 | — | — | ND, **PC**; 5.85/9.25; −0.322/−0.469 | **PD**, **PC**; 11.40/25.60; +0.322/+0.469 | — | — |
| up + bright    | — | — | **PD**, **PC**; 18.25/23.70; +0.412/+0.514 | ND, **PC**; 7.60/7.60; −0.412/−0.514 | — | — | **PD**, NC; 8.05/13.55; +0.263/+0.246 | ND, NC; 4.70/8.20; −0.263/−0.246 |
| up + dark      | — | — | **PD**, NC; 3.15/16.35; +0.024/+0.389 | ND, NC; 3.00/7.20; −0.024/−0.389 | — | — | **PD**, **PC**; 11.40/25.60; +0.322/+0.469 | ND, **PC**; 5.85/9.25; −0.322/−0.469 |
| down + bright  | — | — | ND, **PC**; 7.60/7.60; −0.412/−0.514 | **PD**, **PC**; 18.25/23.70; +0.412/+0.514 | — | — | ND, NC; 4.70/8.20; −0.263/−0.246 | **PD**, NC; 8.05/13.55; +0.263/+0.246 |
| down + dark    | — | — | ND, NC; 3.00/7.20; −0.024/−0.389 | **PD**, NC; 3.15/16.35; +0.024/+0.389 | — | — | ND, **PC**; 5.85/9.25; −0.322/−0.469 | **PD**, **PC**; 11.40/25.60; +0.322/+0.469 |

---

## Quick symmetry

On a subtype’s **PD** axis:

| Stimulus | T4                         | T5                         |
| -------- | -------------------------- | -------------------------- |
| bright   | **PD**, **PC** | **PD**, NC           |
| dark     | **PD**, NC           | **PD**, **PC** |

Left vs right eye: only the stimulus direction for A→P vs P→A is mirrored (`right` ↔ `left`); up/down unchanged.

## DSI rules (read before editing eye tables)

These rules govern every number in the Right/Left eye tables and any hardcoded training
target derived from them. **Do not invent another definition.**

### Cell format

```text
PD|ND, PC|NC; <peak_w1>/<peak_w4>; <DSI_w1>/<DSI_w4>
```

Example: `**PD**, **PC**; 18.25/23.70; +0.412/+0.514`

### Step 1 — labels (independent of DSI)

Use `motion_preference` / [How to fill the table](#how-to-fill-the-table-two-independent-axes):

1. **PD vs ND** from motion vs subtype PD only.
2. **PC vs NC** from contrast × pathway only (T4 bright=PC; T5 dark=PC).

### Step 2 — peak Vm in that cell

Each cell maps to **one** fig1 trace via `fig1_key_for_stimulus` →
`{T4|T5}_{PC|NC}_{w1|w4}_{PD|ND}`.

| Cell’s stimulus vs subtype | Peak to write (`w1/w4`) |
| -------------------------- | ----------------------- |
| this direction is **PD** | pathway×contrast **peak PD** from the source table below |
| this direction is ND | pathway×contrast **peak ND** from the source table below |

**PD and ND cells must have different peaks.** Never copy the same peak into both.

### Step 3 — DSI for that cell (canonical definition)

**DSI is always: this cell’s stimulus direction minus the opposite direction.**

$$
\mathrm{DSI} = \frac{\mathrm{peak}_{\mathrm{this\ dir}} - \mathrm{peak}_{\mathrm{opposite\ dir}}}{\mathrm{peak}_{\mathrm{this\ dir}} + \mathrm{peak}_{\mathrm{opposite\ dir}}}
$$

| Stimulus row direction | Formula |
| ---------------------- | ------- |
| `right` | `(peak_right − peak_left) / (peak_right + peak_left)` |
| `left` | `(peak_left − peak_right) / (peak_left + peak_right)` |
| `up` | `(peak_up − peak_down) / (peak_up + peak_down)` |
| `down` | `(peak_down − peak_up) / (peak_down + peak_up)` |

Peaks for `this` / `opposite` are the fig1 peaks for that subtype×contrast×width on those
two directions (Step 2). Write both widths as `DSI_w1/DSI_w4`.

**Not the definition:** `(peak_PD − peak_ND) / (peak_PD + peak_ND)` plus a separate ± from the
PD/ND label. That is only the special case when `this dir` is PD. When `this dir` is ND, the
formula above already yields a **negative** DSI.

Consequence (check only): ND cells are negative, PD cells positive, equal |DSI| for a given
pathway×contrast×width. If an ND cell shows a positive DSI, Step 3 was applied wrong.

### Worked example (Right eye, T4, bright, w1)

Peaks: for T4a, right↔PD = 18.25, left↔ND = 7.60; for T4b those are swapped.

| Stimulus | Subtype | Labels | Peak (this) | DSI computation | DSI |
| -------- | ------- | ------ | ----------- | --------------- | --- |
| right + bright | T4a | **PD**, **PC** | 18.25 | `(18.25−7.60)/(18.25+7.60)` | **+0.412** |
| right + bright | T4b | ND, **PC** | 7.60 | `(7.60−18.25)/(7.60+18.25)` | **−0.412** |
| left + bright | T4a | ND, **PC** | 7.60 | `(7.60−18.25)/(7.60+18.25)` | **−0.412** |
| left + bright | T4b | **PD**, **PC** | 18.25 | `(18.25−7.60)/(18.25+7.60)` | **+0.412** |

### Forbidden mistakes

1. Defining DSI as `(PD−ND)/(PD+ND)` and then stamping `+`/`−` from the PD/ND label.
2. Putting the **same signed DSI** on every cell of a subtype×contrast (ignoring row direction).
3. Using `(right−left)/(right+left)` for **left** or **down** rows (those must use
   this−opposite with **this = left** or **down**).
4. Same peak Vm on a PD cell and an ND cell.
5. Putting only peaks, or only DSI, when the table requires **both**.
6. Recomputing peaks/DSI from cost-window resampled traces instead of the source table below.
7. Treating PC/NC as a direction flip.

## Direction selectivity index (DSI) — source peaks

Population traces in [`fig1_ci_digitized.csv`](../../MatlabFunctions/fig1_ci_digitized.csv)
are already split by pathway, contrast (PC/NC), bar width, and motion direction (PD/ND).
For each condition, take the **peak** `vm_mv` over time. These peaks feed
[DSI rules](#dsi-rules-read-before-editing-eye-tables) Steps 2–3.

Map stimulus contrast to fig1 keys: **T4 bright** → `T4_PC`, **T4 dark** → `T4_NC`;
**T5 dark** → `T5_PC`, **T5 bright** → `T5_NC`.

The last column is only a **convenience** `| (PD−ND)/(PD+ND) |` when this dir = PD;
cell DSI must still use Step 3 (this − opposite).

| Pathway | Stimulus contrast | Width | fig1 key  | peak PD (mV) | peak ND (mV) | \|(PD−ND)/(PD+ND)\| |
| ------- | ----------------- | ----- | --------- | ------------ | ------------ | ------------------- |
| T4      | bright (PC)       | w1    | `T4_PC` | 18.25        | 7.60         | 0.412 |
| T4      | bright (PC)       | w4    | `T4_PC` | 23.70        | 7.60         | 0.514 |
| T4      | dark (NC)         | w1    | `T4_NC` | 3.15         | 3.00         | 0.024 |
| T4      | dark (NC)         | w4    | `T4_NC` | 16.35        | 7.20         | 0.389 |
| T5      | dark (PC)         | w1    | `T5_PC` | 11.40        | 5.85         | 0.322 |
| T5      | dark (PC)         | w4    | `T5_PC` | 25.60        | 9.25         | 0.469 |
| T5      | bright (NC)       | w1    | `T5_NC` | 8.05         | 4.70         | 0.263 |
| T5      | bright (NC)       | w4    | `T5_NC` | 13.55        | 8.20         | 0.246 |

T4 dark at w1 is nearly non-selective (PD ≈ ND). w4 traces are more direction-selective than w1 for both pathways.

## References

- **Data:** [`MatlabFunctions/fig1_ci_digitized.csv`](../../MatlabFunctions/fig1_ci_digitized.csv) — Figure 1 Ci/Cii traces (Gruntman et al. 2021), digitized by `digitize_fig1_ci.py`.
- Maisak et al. (2013): T4a–d PD = front-to-back, back-to-front, up, down.
- Gruntman et al. (2021): T4 PC = bright, NC = dark; T5 PC = dark, NC = bright.
- Shinomiya et al. (2019) eLife; Takemura et al. (2017) eLife.
