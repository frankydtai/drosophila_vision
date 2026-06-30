# Paper Summary: Klapoetke et al. 2022 (Neuron)

**Full text:** [`paper.txt`](paper.txt) | **PDF:** [`paper.pdf`](paper.pdf)

> Focus of this summary: **visual stimulus conditions and recording details** (imaging setup, display, every stimulus protocol and its parameters). Model/connectome results are condensed at the end.

## Citation

Klapoetke, N. C., Nern, A., Rogers, E. M., Rubin, G. M., Reiser, M. B., & Card, G. M. (2022). A functionally ordered visual feature map in the Drosophila brain. *Neuron*, 110(10), 1700–1711.  
DOI: [10.1016/j.neuron.2022.02.013](https://doi.org/10.1016/j.neuron.2022.02.013)

## One-line takeaway

In vivo two-photon calcium imaging of **10 LC visual projection neuron types** with **receptive-field-aligned, adaptively designed stimuli** maps each type's object/looming/size/speed tuning; LC18 detects sub-facet objects via temporal contrast comparison.

---

## Recording details

### Animal preparation
| Parameter | Value |
|-----------|-------|
| **Sex / age** | Female, 2–5 days post-eclosion |
| **Husbandry** | 21.8 °C, 55% humidity, 16 h light / 8 h dark, cornmeal/molasses food |
| **Collection** | 12–24 h before imaging |
| **Prep** | Anesthetized on ice; head glued so **eye equator faces middle of screen**; proboscis glued; cuticle removed and **muscle 16 severed** |
| **Drivers** | Split-GAL4 lines (mostly from Wu et al., 2016); genetic calcium indicator (GCaMP) per Table S1 |
| **Side** | Stimuli to **right eye**; imaging from **right** side of brain |
| **Throughput** | ~24 h imaging per fly; ~1–2 neurons/fly |

### Two-photon imaging
| Parameter | Value |
|-----------|-------|
| **Microscope** | Thorlabs two-photon (as in Klapoetke et al., 2017) |
| **Laser** | Ti:Sapphire (Spectra-Physics Mai Tai eHP DS) @ **920 nm**, **<20 mW** at sample |
| **Objective** | 16× water-immersion (Nikon CFI75, NA 0.8) |
| **Detection** | GaAsP PMT (Hamamatsu H10770PB-40 SEL); bandpass 503/40 nm (Semrock) |
| **Volume rate** | **≥5.6 Hz** |
| **Saline** | Oxygenated, circulated throughout |
| **Motion correction** | None applied |

### Dataset / sampling
- **91 RF-centered neurons from 50 flies** for size/speed/direction tuning; typically **n = 7–11 neurons from 5 flies** per cell type (Table S2).
- LC18 sub-experiments: n = 8 neurons / 5 flies (most), n = 9 / 7 flies (flicker).
- No flies excluded; sample sizes are field-typical, not power-calculated; experimenters not blinded.

### Analysis
- Custom MATLAB scripts parse fluorescence and temporally align imaging frames to the stimulus.
- Stats: RM-ANOVA + Bonferroni/post hoc; KS test; Pearson correlation; Prism8 + MATLAB.

---

## Visual display (stimulus hardware)

| Parameter | Value |
|-----------|-------|
| **Projector** | Green **532 nm** projector setup (as in Klapoetke et al., 2017) |
| **Screen coverage** | ~**90° × 90°** of the fly's field of view (right eye) |
| **Background** | Always neutral **0.5** intensity |
| **Object intensity** | Black = **0**, white = **1** (unless contrast-matched) |
| **Refresh / framerate** | **180 Hz** (per flicker table) |
| **Model movies** | Grayscale −0.5…0.5, background 0, **0.5°/pixel** |

### Experiment structure
- **RF mapped first** for each fly; neurons selected near screen center; all later stimuli **spatially aligned (RF-centered)** to that neuron.
- Stimuli grouped into **blocks** = one run of all protocols; each neuron got **3 block repeats**; protocol order **randomly shuffled** per block.
- **Inter-stimulus interval (ISI) = 5 s** between protocols (except RF mapping, see Table S3).
- **Adaptive design:** stimuli spanned two log ranges in spatial and temporal parameters, generated online and aligned to each neuron's measured RF. Spatiotemporal parameters were **customized per LC cell type** (Table S3).

---

## Stimulus protocols (visual conditions)

### 1. Receptive-field mapping (Fig. 1; Table S3)
Drifting square-wave grating inside a small aperture at randomized grid positions; **2 repeats**; aperture & motion direction randomized.

| Cell | Spatial period (°/cyc) | Speed (°/s) | Aperture W×H (°) | Grid spacing (°) | Duration (s) | ISI (s) | Directions |
|------|------------------------|-------------|------------------|------------------|--------------|---------|------------|
| LC11 | 8 | 96 | 8×8 | 8 | 1 | 2 | Prog/Reg |
| LC12 | 8 | 96 | 8×8 | 8 | 1 | 2 | Prog/Reg |
| LC15 | 7.5 | 45 | 7.5×15 | 7.5 | 1 | 2 | Prog/Reg |
| LC17 | 8 | 96 | 8×8 | 8 | 1 | 2 | Prog/Reg |
| **LC18** | **8** | **96** | **8×8** | **8** | **1** | **2** | **Prog/Reg** |
| LC21 | 8 | 192 | 8×8 | 8 | 1 | 2 | Prog/Reg |
| LC25 | 8 | 96 | 8×8 | 8 | 1 | 2 | Prog/Reg |
| LPLC1 | 16 | 64 | 8×8 | 8 | 1 | 2 | Up/Down (edge-aligned) |
| LPLC2 | 30 | 20 | 15×15 | 7.5 | 1 | 2 | Prog/Reg (edge-aligned) |

(LC4 mapping: spatial period 30, speed 200, edge-based.) Bar/aperture range used for RF size estimate: 7.5–15° aperture.

### 2. Feature-screen motion battery (Fig. 2A, B)
Constant **edge speed**: looming **100°/s**, moving squares & bars **100°/s**, full-field drifting grating **25°/s**. Used for the cluster dendrogram (object vs looming groups).

### 3. 2D size tuning (Fig. 2G–J; Fig. 3 for LC18)
Moving **dark rectangle**, **edge speed 100°/s**, swept across the **entire RF**; widths × heights **systematically varied 2–90°** in both dimensions. Motion **front-to-back** (LC4: bottom-to-top). RF-aligned. One direction per object size (most LCs non-DS).

### 4. Speed tuning (Fig. 2E)
Object **size customized per cell type** (Table S3, "directional/speed tuning" sizes below); both **bright and dark** objects across a range of speeds (~1 order of magnitude). Filled symbols = significant bright vs dark difference.

| Cell | Object W×H (°) used for direction/speed tuning |
|------|----------------------------------------------|
| LC4 | 9×30 | 
| LC11 | 4.5×4.5 |
| LC12 | 9×9 |
| LC15 | 4.5×45 |
| LC17 | 4.5×4.5 |
| **LC18** | **4.5×4.5** |
| LC21 | 4.5×4.5 |
| LC25 | 4.5×15 |
| LPLC1 | 9×9 |
| LPLC2 | n/a (weak; speed/direction not run) |

### 5. Directional tuning (Fig. 2D)
**Dark object** moved every **45°** across RF center; object size & speed customized per cell type. LC17/LPLC2 omitted (weak non-looming responses). DSi = average response vector magnitude.

### 6. Looming (Fig. 2B, F; Fig. 4C)
Dark disk, **constant edge speed 100°/s** (feature screen); also r/v looming series (disk diameter 0→60°) at edge speed 25°/s for the dendrogram/ΔF/F-ref protocols (Table S3).

### 7. LC18 small-object / temporal-contrast motion (Fig. 3B)
Dark or bright **object motion at 75°/s**; **Δt** = interval between OFF and ON contrast changes seen by a single facet. Strong response at **Δt = 60 ms** (small object), weak at **Δt = 120 ms** (2× wider). Speed of **75°/s** chosen to give ~50% response to a moving 4.5° square.

### 8. LC18 flicker / apparent motion (Fig. 3C, D; Table S7)
Non-moving flicker patch at RF center; **framerate 180 Hz**; duration fixed to 266 ms or one full cycle. Frequencies span **0.94–90 Hz**:

| # frames/half-cycle | # cycles | Framerate (Hz) | Duration (s) | Stim freq (Hz) | Flash dur (ms) |
|---------------------|----------|----------------|--------------|----------------|----------------|
| 96 | 1 | 180 | 1.067 | 0.9375 | 533 |
| 48 | 1 | 180 | 0.533 | 1.875 | 267 |
| 24 | 1 | 180 | 0.267 | 3.75 | 133 |
| 12 | 2 | 180 | 0.267 | 7.5 | 67 |
| 6 | 4 | 180 | 0.267 | 15 | 33 |
| 3 | 8 | 180 | 0.267 | 30 | 17 |
| 2 | 12 | 180 | 0.267 | 45 | 11 |
| 1 | 24 | 180 | 0.267 | 90 | 6 |

Minimal response to **0.94 Hz** flicker (all sizes); large response to **9°/15° squares at 30 Hz** → confirms requirement for fast sequential contrast change (Δt ≲ 60 ms), not sub-4.5° spatial tuning.

### 9. LC18 size-vs-contrast (Fig. 3E–H)
Edge speed **75°/s** (gives 50% response to a 4.5° square). **Size tuning:** fixed black intensity, varied bar width w. **Contrast tuning:** fixed **4.5° square**, intensity matched to the size stimulus via
`Ic = (w/4.5)·Is + (1 − w/4.5)·Ib`, background `Ib = 0.5`. Both dark and bright tested. Contrast tuning mirrors size tuning (optical-blurring effect).

### ΔF/F normalization (Table S3)
Each cell type's responses normalized using a per-type reference stimulus (e.g. LC18: speed-tuning 4.5° square, black, edge speeds 100/200°/s) for cross-cell comparison.

---

## Key experimental results (brief)

- LCs split into **object-motion** (LC11, LC15, LC18, LC21, LC25) and **looming** (LC4, LC12, LC17, LPLC1, LPLC2) groups; most are **not direction-selective** (LC4 excepted).
- RF size set by dendritic arbor size (r = 0.86); RFs ~15–40°.
- **LC18 & LC11** are small-object selective (respond to objects as small as **2×4.5°**, below the ~5° facet acceptance angle) despite large RFs; selectivity preserved across ~1 order of magnitude of speed (peak shifts smaller at slow / larger at fast speeds).
- LC18 detects tiny objects by comparing **opposite-polarity contrast changes in time** at one location and prefers **intermediate contrast** (to beat optical blur).

---

## Model & downstream (condensed)

- **LC18 model** (Fig. 3I–O, Fig. S5, Table S8) — code in this folder. Columnar ON/OFF units: eye optics (5° acceptance, 4.5° interommatidial; 13×19 lattice) → photoreceptor LP (27 Hz, τ≈5.9 ms) → HP 1 Hz → rectify → DoG center-surround → adaptation → 4 signals with gain+saturation → faster/slower LP split → **crossover divisive inhibition** → subtraction `(1−w_off)·max(ON_f−ON_s,0)+w_off·max(OFF_f−OFF_s,0)` → spatial Gaussian pool → calcium LP (0.4 Hz, τ≈400 ms). Fit by GA + patternsearch to LC18 calcium recordings. Reproduces the 2° peak, size, contrast, and flicker tunings. (Files: `simulateXcontrast.m`, `convX2param.m`, `plot_figures.m`, `lc18_model_param.mat`; Zenodo [10.5281/zenodo.5950022](https://doi.org/10.5281/zenodo.5950022).)
- **Topographic map** (Fig. 4): PCA of responses (PC1 ≈ looming vs object, PC2 ≈ size) recapitulates glomerular anatomical order. Hemibrain connectome shows downstream neurons preferentially integrate from **spatially adjacent, functionally similar** glomeruli.

## Key references

- Klapoetke et al. 2017 (imaging + projector setup, LPLC2); Wu et al. 2016 (LC anatomy / driver lines); Tanaka & Clark 2020 (T2/T3 front end); Wiederman et al. 2013 (dragonfly small-target detector); Werblin 2010 (crossover inhibition); Scheffer et al. 2020 (hemibrain connectome).
