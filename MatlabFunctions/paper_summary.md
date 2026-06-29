# Paper Summary: Gruntman et al. 2021 (Current Biology)

**Full text:** [`paper.txt`](paper.txt) | **PDF:** [`paper.pdf`](paper.pdf)

## Citation

Gruntman, E., Reimers, P., Romani, S., & Reiser, M. B. (2021). Non-preferred contrast responses in the Drosophila motion pathways reveal a receptive field structure that explains a common visual illusion. *Current Biology*, 31, 5286–5298.  
DOI: [10.1016/j.cub.2021.09.072](https://doi.org/10.1016/j.cub.2021.09.072)

## One-line takeaway

T4 (ON) and T5 (OFF) share a similar spatiotemporal receptive field for **non-preferred contrast (NC)** inputs; a **4-conductance model** fit to static bar flashes predicts directional selectivity for both preferred and NC moving bars and explains the **reverse-phi illusion** at cellular and behavioral levels.

## Main question

Does motion computation in the fly ON/OFF pathways depend on crosstalk between channels, and can that crosstalk explain the reverse-phi illusion?

## Key findings

1. **NC moving bars are directionally selective** in T4 and T5 when bars are wide and slow; fast narrow NC bars do not evoke DS responses (Fig. 1).
2. **NC responses are not just trailing-edge PC responses** — three dynamic differences between PC and NC moving-bar responses (polarity order, decay rate, peak timing) imply additional mechanisms (Fig. 1E).
3. **T4 and T5 share a similar NC receptive field structure**: depolarization on trailing side, hyperpolarization toward center; onset hyperpolarization + offset depolarization at RF center (Fig. 2).
4. **4-conductance model** (Einc/Iinc for luminance increments, Edec/Idec for decrements) fit to static flashes predicts moving-bar DS for both contrasts (Fig. 3).
5. **Reverse-phi**: NC–PC bar pairs invert directional preference on the trailing side; model predicts this (Fig. 4).
6. **Behavior**: tethered flies show stronger reverse-phi turning with wider bars, less overlap, and slower steps — matching model predictions (Fig. 5).
7. **vs. prior 2-conductance model**: new model responds to bright gratings and shows enhanced DSI (Fig. 6).

## Terminology & conventions

| Term | Meaning |
|------|---------|
| **T4** | ON pathway; PC = bright, NC = dark |
| **T5** | OFF pathway; PC = dark, NC = bright |
| **PD / ND** | Preferred / non-preferred direction of motion |
| **PC / NC** | Preferred / non-preferred contrast |
| **RF** | Receptive field, mapped along PD–ND axis |
| **Reverse-phi** | Contrast inversion between sequential bar positions → perceived motion opposite to displacement |

In all figures, **PD is left-to-right**.

## Figure guide

| Figure | Content | Key n |
|--------|---------|-------|
| **Fig. 1** | NC moving-bar DS; distinct dynamics vs PC | T4 n=9–14; T5 n=12–17 |
| **Fig. 2** | Static ON/OFF RF mapping (bar flashes) | T4 n=15; T5 n=17 |
| **Fig. 3** | 4-conductance model architecture, fit, moving-bar predictions | 5 cardinally aligned cells each |
| **Fig. 4** | Moving edges; minimal reverse-phi bar pairs; model inversion | T4 n=7–9; T5 similar |
| **Fig. 5** | Tethered flight optomotor + reverse-phi behavior | n=17 flies |
| **Fig. 6** | 2-conductance vs 4-conductance grating responses | Model comparison |
| **Fig. S1** | NC moving bars across widths/speeds | — |
| **Fig. S2** | Current injection (intrinsic nonlinearity ruled out) | — |
| **Fig. S3–S4** | Full vs reduced model details | — |
| **Fig. S5** | Additional reverse-phi stimuli + model | — |

## Model summary (4-conductance)

### Membrane dynamics (small τ_M limit)

```
V ≈ (V_L + (E_tot - α·I_tot)·(V_E - V_L)) / (1 + E_tot + I_tot)
```

- **E_tot** = Einc + Edec (excitatory conductances)
- **I_tot** = Iinc + Idec (inhibitory conductances)
- Each conductance has Gaussian spatial RF + dual temporal filters (τ_rise, τ_decay)

### Input types

See **Model visual input** under [T4/T5 recording & visual input](#t4t5-recording--visual-input) for the full bright/dark → conductance mapping.

| Input | Role |
|-------|------|
| **Pulse** | Stimulus duration (ON/OFF while bar present) |
| **Delta** | Instantaneous contrast change at stimulus offset; amplitude scales with preceding duration (ReLU nonlinearity) |

### T4-optimized layout (dark = NC example)

- **Einc, Iinc**: increment pair (bright onset / offset)
- **Edec, Idec**: decrement pair (dark onset / offset)
- **Idec** on leading side → NC onset hyperpolarization
- **Edec** minor, far trailing

### T5-optimized layout

- **Edec, Idec** in center/trailing → PC (dark) DS
- **Iinc** on leading → NC (bright) onset hyperpolarization
- **Einc** further leading → weak bright-onset depolarization

### DS mechanism (T5 example, slow bars)

- **PD motion**: excitation and inhibition peaks separated in time → less cancellation
- **ND motion**: E and I overlap → suppression
- **NC bars**: trailing-edge delta drives dominant conductances (Edec/Idec for T5)

### Optimization

- **Reduced model** (main text): fit only to static bar flashes (widths 2–4 LED, durations 40/160 ms); fewer parameters; selected from top 1% solutions
- **Full model** (supplement): also includes moving bars in loss
- Optimizer: MATLAB `fmincon`, interior-point; 30 ms transmission delay
- Prior 2-conductance model (Gruntman et al. 2019 eLife): Edec/Idec only; blind to bright stimuli

## T4/T5 recording & visual input

### Animals and cell targeting

| | **T4** | **T5** |
|---|--------|--------|
| **Age/sex** | 1–2 day old females | same |
| **Split-GAL4** | SS02344 (VT015785-p65ADZp × R42F06-ZpGdbd) | SS25175 (VT055812-AD × R47H05-DBD) |
| **Label** | 10×UAS-GFP (attP2) | 10×UAS-GFP (attP2) |
| **n cells** | 15 (new dataset) | 17 (from Gruntman et al. 2019 eLife) |
| **Side** | Left optic lobe only | same |

GFP-labeled **somata** patched under visual control (60× water objective, Sutter SOM). IR + 460 nm GFP excitation (Micro-Manager).

### Patch-clamp setup

| Parameter | Value |
|-----------|-------|
| **Mode** | Current clamp, in vivo whole-cell |
| **Sample rate** | 20 kHz; low-pass 10 kHz (MultiClamp 700B + NI PCIe-7842R) |
| **Vm target** | −60 to −55 mV via 0–3 pA holding current (set before flash data) |
| **Electrode** | 9.5–10.5 MΩ; KAsp internal + Alexa 594 (265 mOsm, pH 7.3) |
| **Extracellular** | Standard fly saline, 275 mOsm, pH 7.3, 95% O₂ / 5% CO₂ |
| **Prep** | Head down; proboscis glued; perineural sheath removed; muscle 1657 cut to reduce brain motion |
| **Quality check** | Current-step injections at start of experiment |

### Per-cell calibration workflow

Each recorded cell follows this sequence; all later stimuli use that cell's **RF center** and **PD–ND axis**:

```
1. RF center mapping (3-step PC flash grids)
2. PD estimation (8-direction PC moving bars)
3. Static bright + dark bar flashes along PD–ND axis (widths 1, 2, 4 LED)
4. Additional stimuli: moving bars, moving edges, reverse-phi bar pairs
```

Stimuli pseudorandom within blocks. Repeats: **3×** moving/other stimuli; **5×** single bar flashes. ISI: **500 ms** moving, **800 ms** flashes.

### Visual display (electrophysiology)

| Parameter | Value |
|-----------|-------|
| **Hardware** | Modular LED panels (Reiser lab); FPGA controller on same PCIe card as ephys |
| **Geometry** | ~half cylinder: 216° azimuth × ~72° elevation |
| **Pixel size** | ≤ 2.25° per LED |
| **Wavelength** | Green, peak ~565 nm |
| **Background** | ~31 cd/m² (intermediate gray) |
| **Control** | Custom MATLAB stimulus gen → LabVIEW streams patterns from disk |

Bright = above background; dark = below background.

### Stimulus protocol (in presentation order)

#### Step 1 — RF center mapping

Three grids of **PC** flashing squares (200 ms each):

1. **6×7 grid** of non-overlapping 5×5 LED squares (~11° × 11°)
2. **3×3 grid**, 50% overlap, PC + NC squares (verify cell polarity)
3. **5×5 grid** of 3×3 LED squares (~7° × 7°), 1-pixel shifts → peak response = RF center

#### Step 2 — Preferred direction (PD)

- **8 directions**, 80 ms per LED step (~28°/s)
- Bar: **9 LEDs tall** × **1, 2, or 4 LEDs wide**
- Cardinal motion: 9 steps; diagonal: 13 steps (same spatial distance on rectangular grid)
- PD chosen visually from **PC responses only**
- Cells grouped by **cardinal** vs **diagonal** PD–ND alignment (Fig. 1: diagonal; Fig. 3: cardinal)

#### Step 3 — Static bar flashes (Fig. 2 RF mapping)

Along PD–ND axis, ≥13 positions:

| Parameter | Values |
|-----------|--------|
| **Widths** | 1, 2, 4 LEDs (2.25°, 4.5°, 9°) |
| **Durations** | 40 ms, 160 ms |
| **Contrasts** | Bright and dark at every position |
| **Repeats** | 5×; ISI 800 ms |

#### Step 4 — Moving bars (Figs 1, 3, S3, S4)

| Parameter | Values |
|-----------|--------|
| **Axis** | PD–ND only |
| **Widths** | 1, 2, 4 LEDs |
| **Step duration** | 40 ms → 56°/s; 160 ms → 14°/s |
| **Height** | 9 LEDs |
| **Contrasts** | Bright and dark |
| **Repeats** | 3×; ISI 500 ms |

**T4 vs T5 moving-bar geometry (critical difference):**

| | **T4** | **T5** |
|---|--------|--------|
| **Appearance** | Full-width bar appears and disappears | Bar **grows/shrinks** at presentation window edge |
| **4-LED example** | Always 4 LEDs wide on screen | Starts as 1 LED; reaches full width when trailing edge enters window |

Same leading-edge travel distance; different spatial envelope.

#### Step 5 — Moving edges (Fig. 4A)

Same windows and step durations (40 ms, 160 ms) as moving bars. Single contrast edge traverses window; entire window returns to background after edge exits.

#### Step 6 — Minimal reverse-phi bar pairs (Figs 4, S5)

Sequential two-bar flashes along PD–ND axis (160 ms per bar):

| Cell | Order | Configuration |
|------|-------|---------------|
| **T4** | Dark (NC) → Bright (PC) | 4-LED bars; second bar immediately after first |
| **T5** | Bright (NC) → Dark (PC) | Width-2 with speed-corrected delay, **or** width-4 back-to-back |

Fig. S5 adds overlapping 4-LED bar pairs (6- vs 8-LED span).

### Leading vs trailing edge in moving bars

A moving bar has two contrast transitions:

```
Bright bar moving right (PD):
  leading edge = dark→bright (increment)
  trailing edge = bright→dark (decrement)

Dark bar moving right:
  leading edge = bright→dark (decrement)
  trailing edge = dark→bright (increment)
```

For **T4 + dark (NC) moving bars**, responses are not explained by trailing-edge PC alone. Three observed differences vs bright (PC) bars:

1. Hyperpolarization **precedes** depolarization (opposite order)
2. Dark-bar response decay is **slower**
3. Dark-bar peak is **delayed** even when aligned to the increment edge

This motivated composite ON/OFF RF mapping and the 4-conductance model.

### NC moving-bar selectivity (Fig. 1)

- **Fast narrow NC bars** (e.g. 1 LED, 56°/s): weak or non-DS
- **Slow wide NC bars** (e.g. 4 LED, 14°/s): strong DS, comparable PD–ND peak difference to PC bars
- NC DS requires a **trailing luminance-change boundary** (confirmed by moving-edge experiments, Fig. 4A)

### Model visual input (stimulus → conductances)

The model maps visual events at each RF position **x** to four conductance streams (does not simulate full upstream pathway):

| Visual event | Input type | Drives |
|--------------|------------|--------|
| **Stimulus ON** (bar present) | **Pulse** `I(t,x) ∈ {0,1}` | Increment pair (bright) or decrement pair (dark) |
| **Stimulus OFF** (contrast reversal) | **Delta** `δ(t − t_off)` | Opposite pair; amplitude = ReLU(m·duration + b) |

**Bright flash at x:**
- `k_Einc = k_Iinc = I_bright(t,x)` while bright
- At bright offset → filtered delta into `Edec`, `Idec`

**Dark flash at x:**
- `k_Edec = k_Idec = I_dark(t,x)` while dark
- At dark offset → filtered delta into `Einc`, `Iinc`

Moving bars decompose into pulse/delta events at each spatial step. Simulations include **30 ms transmission delay** (ommatidia → T4/T5).

### Ephys analysis notes

- Trials excluded if pre-stimulus baseline drifts >10 mV (or >15/25 mV for strong slow-bar responses)
- Responses aligned to bar appearance (or central position for 8-dir mapping)
- Depolarization: 0.999 quantile in response window (200 ms + duration for PC; 375 ms + duration for NC)
- Hyperpolarization: longer window for PC (800 ms + duration); NC uses same window as depolarization (onset hyperpolarization)
- Rise start: time to 10% of max (arena delay corrected); decay: 80% → 20% of peak

### Behavior (separate arena)

Different display from ephys: 270° azimuth × ~120° elevation; pixel ~1.875°; ~570 nm LEDs.

| Parameter | Values |
|-----------|--------|
| **Gratings** | 30° cycle; bar widths 1.875°–9.375° (duty-cycle varied) |
| **Step duration** | 20, 40, 160 ms (≈120°, 60°, 30°/s) |
| **Conditions** | Standard (bright/gray) vs reverse-phi (contrast inverts each step) |
| **Trials** | 1.7 s open-loop rotation; interleaved with 3.6 s closed-loop stripe fixation |
| **Measure** | L−R wing-beat amplitude (yaw torque proxy); normalized per fly |
| **Flies** | Empty split × Kir2.1 control line (2–5 day old females; vigorous fliers) |

## Data & code

| Resource | Link |
|----------|------|
| **Data + analysis + model results** | [Figshare 10.25378/janelia.c.5629663.v1](https://doi.org/10.25378/janelia.c.5629663.v1) |
| **MATLAB analysis code** | [github.com/gruyal/MatlabFunctions](https://github.com/gruyal/MatlabFunctions) |
| **Prior T5 model params (Fig. 6)** | [Figshare 10.25378/janelia.c.4771805.v1](https://doi.org/10.25378/janelia.c.4771805.v1) |

## Related prior work (same lab)

- **Gruntman et al. 2018** (*Nat Neurosci*): 2-conductance T4 model (fast E + slow offset I)
- **Gruntman et al. 2019** (*eLife*): 2-conductance T5 model; PC-only RF
- This paper extends both with NC conductances (Einc/Iinc pair) and delta inputs at offset

## Reverse-phi intuition (minimal bar pair)

**Trailing-side pair (PD displacement, inverted DS):**  
NC bar in center → strong Idec; PC bar on trailing side → strong Iinc → net inhibition.

**Leading-side pair (ND displacement, excitatory):**  
NC on trailing → Einc at offset; PC in center → Einc at onset → net excitation.

Wider/slower bars → stronger NC conductances → stronger behavioral inversion.

## Discussion highlights

- NC hyperpolarization resembles **crossover inhibition** (mammalian retina)
- **Mi9** (glutamatergic, OFF-responding T4 input) candidate for Idec, but synapse location vs RF data unresolved
- Model converges on **3-arm E–I–I architecture** similar to Zavatone-Veth et al. 2020 and Borst 2018, but constrained by whole-cell RF measurements
- Static RF structure alone sufficient to predict complex dynamic stimuli (moving bars, reverse-phi, gratings)
