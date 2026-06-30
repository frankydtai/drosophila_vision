# Reproduction Status: Klapoetke et al. 2022 (Neuron)

Figure-by-figure assessment of whether each panel can be reproduced **from the contents of this `LC18_model/` folder**.

**Related:** [`paper_summary.md`](paper_summary.md) | [`paper.txt`](paper.txt) | [`paper.pdf`](paper.pdf)

## Scope of this folder

This is the **LC18 computational-model release only**. It contains the forward-model code, the model's pre-processed stimuli, the trained parameters, and an LC18 calcium *training* subset. It does **NOT** contain:

- the calcium-imaging dataset for the 10 LC types (RF maps, tuning curves),
- the imaging analysis pipeline,
- the hemibrain connectome analysis (separate Zenodo `10.5281/zenodo.5950008`),
- the parameter-fitting code, or the raw-stimulus → `pr_mov` generation code.

**Requirement to run:** MATLAB + Signal Processing Toolbox (`butter`/`filter`). The `.venv/` here is an empty Python stub (only pip/setuptools) and is irrelevant.

### Inventory

| Code                                                 | Role                                                                                                         |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `simulateXcontrast.m`                              | Full forward model (ON/OFF, DoG, adaptation, gain+saturation, crossover inhibition, E−I, pooling, GCaMP LP) |
| `convX2param.m`                                    | 25-element parameter vector`x` → `param` struct                                                         |
| `plot_figures.m`                                   | Entry point: runs model over`vstim/`, exports Fig 3 / S5 model panels (+ noXinh/noSat ablations)           |
| `Plots/.../plot_2d_tuning.m`, `colormap_magma.m` | Renders the **Fig S5G/J size×speed heatmaps** — reads the five `2Dsize_*dps/*_signal_peaks.csv` → `2Dsize_analysis/2Dsize_*dps.eps` |

| Data                               | Content                                                                            |
| ---------------------------------- | ---------------------------------------------------------------------------------- |
| `lc18_model_param.mat`           | Trained parameter vector`x`                                                      |
| `Table8_model_param.xlsx`        | Parameter names / bounds / linear constraints (Table S8)                           |
| `vstim/*.mat` (13)               | **Pre-processed** stimulus movies `pr_mov` (eye-optics sampled + 27 Hz LP) |
| `training_data/lc18_ca_data.mat` | Measured LC18 calcium (training subset)                                            |
| `training_data/vstim_set.mat`    | Training stimulus set                                                              |
| `Plots/`                         | Pre-generated outputs (`.eps` + `*_signal_peaks.csv`)                          |

Legend: ✅ fully reproducible here · 🟡 partial · ❌ not reproducible here (data/code absent or in external deposit)

---

## Figure 1 — LC anatomy & receptive-field mapping ❌

| Panel | Needs | `vstim/*.mat` | `*_signal_peaks.csv` |
|-------|-------|---------------|----------------------|
| A–C anatomy (schematic, LC18 population, single cells) | confocal/MCFO stacks, registration | — | — |
| D imaging setup | photo | — | — |
| E–G single-cell RF mapping (ROIs, contours, Ca traces) | two-photon Ca imaging + RF analysis | — | — |
| H RF FWHM vs dendrite size (r = 0.86) | per-cell RF + anatomy (Table S4) | — | — |

→ No raw imaging data and no analysis code in this folder.

## Figure 2 — Each LC type encodes distinct features ❌

| Panel | Needs | `vstim/*.mat` | `*_signal_peaks.csv` |
|-------|-------|---------------|----------------------|
| A cluster dendrogram | responses of all 10 LC types | `2DsizeTuning_*dps` (size part); looming/grating — | — |
| B motion-battery Ca responses | imaging | — | — |
| C RF shape | imaging | — | — |
| D directional tuning (DSi) | imaging | — | — |
| E speed tuning (bright/dark) | imaging | `FigS5_speedTuning.mat` | — |
| F looming | imaging | — | — |
| G–J 2D size tuning (incl. LC18 heatmap) | imaging | `2DsizeTuning_100dps.mat` (+ `_40/80/200/400dps` for size×speed) | — |

→ Entirely experimental, multi-cell-type. Not reproducible here.

## Figure 3 — LC18 small-object / temporal-contrast detection 🟡 (core of this folder)

| Panel | Needs | `vstim/*.mat` | `*_signal_peaks.csv` |
|-------|-------|---------------|----------------------|
| A facet cartoon | schematic | — | — |
| B dark/bright object motion @75°/s (Δt) | imaging (Ca) | `Fig3_barTuning_experiment.mat` | — |
| C, D flicker size/freq | imaging (Ca) | `Fig3_squareOnOffFlicker.mat` | — |
| E Ca traces, contrast-matched dark | imaging (Ca) | `Fig3_barTuning_experiment.mat` | — |
| F size vs contrast tuning (dark) | imaging | `Fig3_barTuning_model.mat` + `Fig3_smallSquareContrastTuning_model.mat` | — |
| G, H bright object Ca + tuning | imaging | `Fig3_barTuning(bright)_model.mat` + `Fig3_smallSquareContrastTuning_model.mat` | — |
| I model schematic | diagram | — | — |
| **J** modeled 15° dark bar | model run | `Fig3_barTuning_experiment.mat` | `F3_barWidthTuning_experiment` |
| **K** modeled 2° dark bar | model run | `Fig3_barTuning_experiment.mat` | `F3_barWidthTuning_experiment` |
| **L** modeled 2°, crossover removed | model run (ablation `x(20)=x(22)=0`) | `Fig3_barTuning_experiment.mat` | `F3_barWidthTuning_experiment_noXinh` |
| **M** measured avg trace + model (±crossover) | model run + measured overlay | `Fig3_barTuning_experiment.mat` (+ `Fig3_barTuning_model.mat`) | `F3_barWidthTuning_experiment` + `_noXinh` |
| **N** modeled bar-width tuning + intermediate signals | model run | `Fig3_barTuning_model.mat` | `F3_barWidthTuning_model` |
| **O** modeled contrast tuning ±saturation | model run (ablation `x(7)=Inf`) | `Fig3_smallSquareContrastTuning_model.mat` | `contrastTuning_smallSquare` + `_noSat` |

→ All model panels (J, K, L, N, O and the model curves of M) reproducible by running `plot_figures`. Experimental panels (B–H, and M's measured trace) are not.

**Data flow (verified against paper — model results are scoped to Fig 3 & S5 only; Table S8 caption "Related to Figures 3, S5"):**

- `vstim/*.mat` → `plot_figures.m` → per-protocol `prot_*.eps` (directly) **and** `*_signal_peaks.csv` (peak tables).
- The five `2Dsize_*dps/*_signal_peaks.csv` → `plot_2d_tuning.m` → `2Dsize_analysis/*.eps` heatmaps (**Fig S5G/J**).
- **All tuning-curve line plots — Fig 3M/N/O and Fig S5 B–F, H, I, K — are drawn from the `*_signal_peaks.csv` peaks, but NO repo script draws them** (plotted externally for the paper).
- **No figure outside Fig 3 and Fig S5 uses these CSVs (directly or indirectly).** Fig 1/2/4 and S1–S4, S6–S8 are experimental / anatomy / connectome; where S5 says "using Fig. 2E / 2G-H / 3C-D / S2D-E stimuli" it borrows only the *stimuli*, and the model curves are plotted in S5.

## Figure 4 — Topographic map & downstream integration ❌

| Panel | Needs | `vstim/*.mat` | `*_signal_peaks.csv` |
|-------|-------|---------------|----------------------|
| A glomeruli LM image | confocal | — | — |
| B–D PCA of responses vs glomerulus position | full LC functional dataset | `2DsizeTuning_*dps` (size part); looming — | — |
| E–M connectome integration (hemibrain) | connectome analysis (Zenodo `5950008`) | — | — |

→ Not reproducible here.

---

## Supplementary figures

### Figure S1 — anatomic & functional RF (→ Fig 1) ❌

| Panel | Needs | `vstim/*.mat` | `*_signal_peaks.csv` |
|-------|-------|---------------|----------------------|
| A resampled confocal single-cell anatomy | confocal anatomy | — | — |
| B RF FWHM (D–V axis) vs dendrite size (r = 0.61) | per-cell RF + anatomy | — | — |
| C interpolated peak responses along RF midline to RF-mapping stimuli | imaging (RF mapping) | — | — |

### Figure S2 — additional LC visual responses (→ Fig 2) ❌

| Panel | Needs | `vstim/*.mat` | `*_signal_peaks.csv` |
|-------|-------|---------------|----------------------|
| A measured reference fluorescence (per-neuron normalization) | imaging | — | — |
| B response trace to constant approach-speed looming | imaging | — | — |
| C population-avg Ca traces for Fig 2I/2J size tuning (preferred size) | imaging | `2DsizeTuning_100dps.mat` | — |
| D, E LC11 & LC18 size×speed: moving dark bars at four speeds | imaging | `2DsizeTuning_40/80/100/200/400dps.mat` | — |

### Figure S3 — direction selectivity (→ Fig 2D) ❌

| Panel | Needs | `vstim/*.mat` | `*_signal_peaks.csv` |
|-------|-------|---------------|----------------------|
| left (per cell type) individual neuron traces around polar DSi plot; custom dark-bar size/speed, multiple directions | imaging | — | — |
| right (per cell type) permutation test: null vs measured DSi distribution | analysis | — | — |

### Figure S4 — LC18 flicker & motion (→ Fig 3) ❌ (model counterparts in S5)

| Panel | Needs | `vstim/*.mat` | `*_signal_peaks.csv` |
|-------|-------|---------------|----------------------|
| A, B all measured responses to full-contrast ON/OFF square flicker (sizes × freqs) | imaging | `Fig3_squareOnOffFlicker.mat` | — |
| C, D dark / bright bar flicker at indicated frequencies | imaging | `FigS5_temporalTuning_flicker.mat` (+ `Fig3_squareOnOffFlicker.mat`) | — |
| E position invariance: 4.5°×9° flicker at azimuthal RF positions | imaging | — | — |
| F stimulus-on-RF cartoon (one-bar / two-bars motion, 100°/s) | diagram | — | — |
| G bar-motion width tuning (leading/lagging edge, width along motion) | imaging | `FigS5_temporalTuning_motion.mat` | — |
| H replotted bar motion vs flicker temporal tuning | analysis | `FigS5_temporalTuning_motion.mat` + `FigS5_temporalTuning_flicker.mat` | — |
| I–M putative T2 / T3 anatomy (EM vs LM) | anatomy | — | — |

### Figure S5 — LC18 model details & simulations (→ Fig 3) ✅ (all reproducible)

| Panel | Needs | `vstim/*.mat` | `*_signal_peaks.csv` |
|-------|-------|---------------|----------------------|
| A detailed model schematic (single columnar unit) | diagram | — | — |
| B–D modeled moving bar (75°/s, 4.5° h) with model modifications (±saturation / ±crossover) | model run | `Fig3_barTuning_model.mat` (+ `Fig3_barTuning(bright)_model.mat`) | `F3_barWidthTuning_model` (+ `_noSat`, `_noXinh`); `F3_widthTuning_brightBar_model` (+ `_noSat`, `_noXinh`) |
| E contrast tuning (4.5° square) ±saturation | model run | `Fig3_smallSquareContrastTuning_model.mat` | `contrastTuning_smallSquare` (+ `_noSat`) |
| F bar width tuning (moving bar 75°/s) | model run | `Fig3_barTuning_model.mat` (+ `Fig3_barTuning(bright)_model.mat`) | `F3_barWidthTuning_model`; `F3_widthTuning_brightBar_model` |
| G size tuning | model run | `2DsizeTuning_*dps.mat` | `2Dsize_40/80/100/200/400dps` |
| H ON/OFF flicker (Fig 3C-D stimuli) | model run | `Fig3_squareOnOffFlicker.mat` | `F3_squareOnOffFlicker` |
| I temporal tuning, flicker vs motion (4.5° dark square) | model run | `FigS5_temporalTuning_flicker.mat` + `FigS5_temporalTuning_motion.mat` | `temporalTuning_flicker`, `temporalTuning_motion` |
| J size×speed | model run | `2DsizeTuning_40/80/100/200/400dps.mat` | `2Dsize_40/80/100/200/400dps` |
| K speed tuning (Fig 2E stimuli) | model run | `FigS5_speedTuning.mat` | `speedTuning` |

### Figure S6 — LC25 line motion (→ Fig 2) ❌ (LC25, not in this folder)

| Panel | Needs | `vstim/*.mat` | `*_signal_peaks.csv` |
|-------|-------|---------------|----------------------|
| A LC25 Ca image / single-cell ROI | anatomy | — | — |
| B receptive field | imaging | — | — |
| C Ca traces to constant edge-speed looming (100°/s) | imaging | — | — |
| D space-time diagram, line-motion stimulus conditions i–iv | imaging | — | — |
| E Ca traces to (D) + dI/dt schematic at x1, x2 | imaging | — | — |
| F peak responses over a range of line-motion stimuli | imaging | — | — |
| G, H further line-motion / contrast-polarity conditions | imaging | — | — |

### Figure S7 — additional connectome analysis (→ Fig 4) ❌ (no visual input)

| Panel | Needs | `vstim/*.mat` | `*_signal_peaks.csv` |
|-------|-------|---------------|----------------------|
| A individual LC T-bars geometry (EM vs LM) | connectome | — | — |
| B, C glomeruli-pair integration vs null (looming/object grouping) | connectome | — | — |
| D–G LC output synapse statistics in central brain | connectome | — | — |

### Figure S8 — downstream-neuron anatomy (→ Fig 4) ❌ (no visual input)

| Panel | Needs | `vstim/*.mat` | `*_signal_peaks.csv` |
|-------|-------|---------------|----------------------|
| A downstream integration neurons projecting out of central brain | connectome | — | — |
| B table of name / bodyID / inputs / outputs | connectome | — | — |

## Gaps even within the model scope

1. **Parameter-fitting code is missing** — no `ga` / `patternsearch` / cost-function script (confirmed by search). You can run with the trained `lc18_model_param.mat`, but cannot re-derive parameters from `training_data/lc18_ca_data.mat`.
2. **Stimulus pre-processing code is missing** — `vstim/*.mat` are already `pr_mov` (eye-optics sampled + 27 Hz LP). The raw-movie → `pr_mov` generator is not included, so new stimuli / regeneration of `vstim` is not possible.

## How to run what *is* reproducible

```matlab
% in MATLAB, from this folder (requires Signal Processing Toolbox)
plot_figures
% → writes .eps + *_signal_peaks.csv under Plots/lc18_model_param/
```

## Bottom line

| Scope                                                          | Status                          |
| -------------------------------------------------------------- | ------------------------------- |
| Fig 3 J, K, L, N, O (+ model curves of M)                      | ✅ reproducible                 |
| Fig S5 B–K (all model panels)                                 | ✅ reproducible                 |
| Fig 3 B–H (experimental), Fig 1, Fig 2, Fig 4, S1–S4, S6–S8 | ❌ data/code not in this folder |
| Parameter optimization & stimulus generation                   | ❌ missing scripts              |
