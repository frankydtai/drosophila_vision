# Paper Summary: Groschner et al. 2022 (Nature)

**Full text:** [`paper.txt`](paper.txt) | **PDF:** [`Paper.pdf`](Paper.pdf)

## Citation

Groschner, L. N., Malis, J. G., Zuidinga, B., & Borst, A. (2022). A biophysical account of multiplication by a single neuron. *Nature*, 603, 119–123.  
DOI: [10.1038/s41586-022-04428-3](https://doi.org/10.1038/s41586-022-04428-3)

## One-line takeaway

In vivo patch-clamp of **T4** and its **columnar inputs** (Mi9, Tm3, Mi1, Mi4, C3) shows that **multiplication-like DS** in T4 arises from **coincident cholinergic excitation and release from glutamatergic shunting inhibition** via **GluClα** on T4 dendrites.

## Main question

What is the biophysical implementation of the multiplication-like nonlinearity in the fly ON motion detector (T4), and does glutamatergic disinhibition via GluClα sharpen directional tuning and behavior?

## Circuit context (T4 inputs)

Each T4 dendrite spans ~7 columns (~3 along PD). Synaptic segregation along dendrite (distal → proximal):

| Neuron | Transmitter | Dendrite location | RF polarity vs luminance |
|--------|-------------|-------------------|--------------------------|
| **Mi9** | Glutamate → GluClα (inhibitory) | Distal (PD-first) | **Anticorrelated** — hyperpolarizes to ON increments |
| **Tm3, Mi1** | Acetylcholine (excitatory) | Centre | Positively correlated with luminance |
| **Mi4, C3** | GABA (inhibitory) | Proximal | Positively correlated with luminance |

Proposed motif: **×** glutamatergic/cholinergic (multiplicative disinhibition); **÷** cholinergic/GABAergic (divisive).

---

## Neuron recordings (electrophysiology)

### Animals and genotypes

| Target | Split-GAL4 driver | Label | n (typical) |
|--------|-------------------|-------|-------------|
| **Mi9** | R48A07-p65.AD × VT046779-GAL4.DBD | mCD8::GFP | 22 (RF avg) |
| **Tm3** | R13E12-p65.AD × R59C10-GAL4.DBD | mCD8::GFP | 11 |
| **Mi1** | R19F01-p65.AD × R71D01-GAL4.DBD | mCD8::GFP | 22 |
| **Mi4** | R48A07-p65.AD × R13F11-GAL4.DBD | mCD8::GFP | 10 |
| **C3** | R26H02-p65.AD × R29G11-GAL4.DBD | mCD8::GFP | 16 |
| **T4** | R42F06-p65.AD × VT037588-GAL4.DBD (**T4 > GFP**) | mCD8::GFP | 46 (RF); subtypes T4c/T4d enriched | 
| **T4 > GluClαRNAi** | Same T4 driver + TRiP.HMC03585 | GFP + RNAi | 14–17 |
| **T4 > Nmdar1RNAi** | Same T4 driver + TRiP.HMS02199 | GFP + RNAi | control |

- **Age/sex:** 2–24 h post-eclosion females; ≥1 wild-type `white` allele
- **Side:** Left dorsal optic lobe
- **Husbandry:** 25 °C, 60% humidity, 12:12 light:dark

### Patch-clamp setup

| Parameter | Value |
|-----------|-------|
| **Mode** | Whole-cell in vivo (current clamp primary; voltage clamp for pharmacology) |
| **Microscope** | Scientifica InVivo SliceScope or Zeiss Axio Scope.A1; ×60/1.0 NA water objective |
| **Visualization** | Bright-field + epifluorescence (GFP somata); white LED transillumination via light guide |
| **Amplifier** | MultiClamp 700B; Digidata 1550B; pCLAMP 11 |
| **Sample rate** | 10 kHz (analyzed at 1 kHz after alignment) |
| **Temperature** | Room temp 21–23 °C |
| **Electrode** | 15–20 MΩ; borosilicate; K-aspartate internal + biocytin (265 mOsm, pH 7.3) |
| **Extracellular** | Wilson/Laurent-style saline (280 mOsm, pH 7.3, 95% O₂ / 5% CO₂) |
| **Prep** | Head fixed in POM mount; cuticle/trachea removed; perineural sheath incised |
| **Inclusion** | Resting potential more negative than −25 mV (dark, no holding current, within 2 min of break-in) |
| **Resting Vm** | Wild-type T4: ~−65 mV; GluClαRNAi: ~11.9 mV depolarized |
| **Input resistance** | Wild-type: 5.28 ± 0.12 GΩ; GluClαRNAi: 6.70 ± 0.16 GΩ (dark) |

### Recording workflow

```
1. Break-in → verify resting Vm
2. White-noise RF mapping (3–20 min)
3. PD from square-wave gratings (8 directions, 1 Hz)
4. Optional: temporal frequency tuning (0.5–16 Hz)
5. Moving ON/OFF edges at 30°/s (Figs 3–4) or 36 directions (Fig 5)
6. Input-resistance tracking: repeated edges + varying holding current (−5 to 0 pA)
7. Pharmacology: glutamate / ACh / GABA puffs to dendrites (Fig 2)
```

Biocytin fills recovered post hoc for morphology (Extended Data Fig. 1, 3).

### Pharmacology (dendritic puff, layer 10 medulla)

| Drug | Concentration | Pulse | Purpose |
|------|---------------|-------|---------|
| **Glutamate** | 1 mM | 100 ms (500 ms for sustained Rin) | Test GluClα-mediated shunting |
| **Acetylcholine** | 1 mM | 100 ms | Measure E_ACh |
| **GABA** | 1 mM | 100 ms | Measure E_GABA |
| **Delivery** | 5 µm pipette, 50 kPa, PDES-02DX | aimed at GFP-labeled T4 dendrites |

Glutamate hyperpolarizes wild-type T4 by 3.72 ± 0.61 mV and reduces Rin by 25.27%; abolished by GluClαRNAi.

### Input-resistance measurement during visual motion (Fig. 4)

- Identical edge stimuli repeated with different holding currents (−5 to 0 pA)
- Rin(t) = slope of V vs I (linear regression across repetitions)
- Smoothed with Gaussian (13 ms σ)
- **ON edge, PD:** Rin peak ~147% of baseline, preceding depolarization ("window of opportunity")
- GluClαRNAi: no peak; baseline Rin already at wild-type peak level

---

## Visual input (electrophysiology arena)

### Display hardware

| Parameter | Value |
|-----------|-------|
| **Projectors** | 2× DLP Lightcrafter 3000 pico projectors → mirrors → cylindrical screen |
| **Coverage** | 180° azimuth × 105° elevation (left frontal visual field); screen doubles as Faraday shield |
| **Channel** | Green only (500–600 nm) → 180 Hz refresh, 8-bit |
| **Max luminance** | 1,274 cd/m² |
| **Mean stimulus luminance** | 8-bit grey = 128 → ~637 cd/m² (full-contrast stimuli) |
| **Software** | Panda3D (Python 2.7); predistorted for cylinder curvature |
| **Sync** | Continuous trigger signal time-locks stimulus and Vm |

### Stimulus 1 — White-noise RF mapping (primary RF characterization)

| Parameter | Value |
|-----------|-------|
| **Type** | Binary spatiotemporal white noise |
| **Pixel size** | 2.8° × 2.8° |
| **Update rate** | 60 Hz |
| **Duration** | 3–20 min per recording |
| **Analysis** | Reverse correlation: K(x,τ) = ∫ S(x, t−τ) × Vm(t) dt; τ ∈ [−0.5, +3.0] s |
| **Drift correction** | Subtract Gaussian low-pass (60 s σ) post hoc |
| **Inclusion** | RF peak >4 s.d.; RF centre >8 px (22.48°) from screen bezel |
| **Alignment** | Extremum of standard-score map → 0°; T4 RFs rotated to PD axis (Fig. 1e) |

**Polarity findings (Fig. 1d):**
- Tm3, Mi1, Mi4, C3: Vm **positively** correlated with luminance
- Mi9: **anticorrelated** — rapid hyperpolarization to ON increments (not OFF depolarization; Extended Data Fig. 2)
- T4: ON-excited with delayed GABAergic inhibition visible in space-time RF (Fig. 1e–g)

### Stimulus 2 — Square-wave gratings (PD + tuning)

| Parameter | Value |
|-----------|-------|
| **Spatial wavelength** | 30° (full screen) |
| **Motion** | 1 Hz temporal frequency; 8 directions, 45° apart |
| **PD definition** | Vector sum of peak Vm responses (Euclidean) across directions |
| **Temporal tuning** | Same grating, 0.5–16 Hz, alternating PD and ND (PD+180°) |
| **Readout** | ΔVm = max − min Vm (1 s prestimulus baseline subtracted) |

### Stimulus 3 — Moving contrast edges (Figs 3, 4, model alignment)

| Parameter | Value |
|-----------|-------|
| **Polarities** | Bright **ON** edges; dark **OFF** edges |
| **Speed** | **30° s−1** |
| **Directions** | PD and ND (per cell); also 36 directions for tuning (Fig. 5) |
| **Alignment** | Input neuron responses time-shifted using cross-correlation of dVm/dt with template T4 ON-edge PD response; RF centre positions verified with photodiode |

**Expected input sequence during ON edge in PD (Fig. 3):**
1. Mi9 glutamatergic inhibition drops (disinhibition) → Rin peak
2. Coincident Tm3/Mi1 cholinergic excitation → supralinear T4 depolarization
3. Later Mi4/C3 GABAergic inhibition on ND-related timing

Model converts measured input Vm → conductances via threshold + gain per cell type; Mi9/Mi4/C3 advanced/delayed by Δt = θ·cos(φ)/v (θ = 4.8° interommatidial angle).

### Stimulus 4 — Fine directional tuning (Fig. 5)

| Parameter | Value |
|-----------|-------|
| **Stimulus** | ON edges only |
| **Speed** | 30° s−1 |
| **Directions** | 36 evenly spaced (10° steps) |
| **Holding current** | −1 pA (stable long recordings) |
| **Readout** | Voigt fit to 700 ms window around peak; L_dir = |Σφ v(φ)| / Σ|v(φ)| |
| **Genotypes compared** | T4>GFP, T4>GluClαRNAi, T4>Nmdar1RNAi |

GluClαRNAi broadens tuning (89.62% of PD response at ±60° vs ~73% in controls).

---

## Model visual input → T4 Vm (single-compartment)

Passive conductance model (no capacitance at steady state):

```
Vm = [EGlu·gMi9 + EACh·(gTm3 + gMi1) + EGABA·(gMi4 + gC3) + Eleak·gleak]
     / [gMi9 + gTm3 + gMi1 + gMi4 + gC3 + gleak]
```

| Reversal potential | Value |
|--------------------|-------|
| E_Glu (GluClα) | −71 mV |
| E_ACh | −21 mV |
| E_GABA | −68 mV |
| E_leak | −65 mV (fit) |

Input Vm → conductance: rectilinear transfer (threshold + gain per neuron). Supralinearity = excitation coincident with **release from shunting inhibition** (Extended Data Fig. 5).

---

## Key recording results (visual)

| Finding | Condition | Result |
|---------|-----------|--------|
| Mi9 polarity | White-noise / ON flashes | Hyperpolarizes to luminance **increments** |
| GluClα shunting | Dark resting | Tonic glutamate keeps Cl channels open → low Rin, hyperpolarized rest |
| Window of opportunity | ON edge, PD | Rin peak ~147% then Vm depolarization |
| DS mechanism | Model + data | Disinhibition (Mi9 off) + ACh excitation (Tm3/Mi1 on) → supralinear |
| GluClαRNAi | 36-dir ON edges | Broader directional tuning; reduced PD response amplitude |
| GluClαRNAi | ON edge Rin | No PD peak; elevated baseline Rin |

---

## Behavior (separate arena — brief)

Not the focus of ephys, but uses different displays:

| | **Ephys projector** | **Behavior LCD arena** |
|---|---------------------|------------------------|
| Coverage | 180° × 105° | ~270° × 120° |
| Refresh | 180 Hz | 120 Hz |
| Mean luminance | ~637 cd/m² (grey 128) | Michelson 50%; grey 100 (~131 cd/m² max) |
| Open-loop edges | 30°/s (ephys); 60°/s (behavior) | 16 directions, 5 s trials |
| Closed-loop | — | 10° dark bar, fly-controlled |

GluClαRNAi in T4/T5 → exaggerated optomotor response to ON edges; impaired bar fixation.

---

## Data & code

| Resource | Link |
|----------|------|
| **Data** | [Edmond 10.17617/3.8g](https://doi.org/10.17617/3.8g) |
| **Code** | Same repository |

---

## Comparison to Gruntman et al. 2021 (Reiser lab)

| | **Groschner 2022** | **Gruntman 2021** |
|---|-------------------|-------------------|
| **Cells** | T4 + 5 input types | T4 and T5 postsynaptic only |
| **Display** | DLP projector cylinder | LED panel arena |
| **RF method** | 60 Hz binary white-noise reverse correlation | Structured bar flashes + moving bars |
| **NC/ON-OFF focus** | Mi9 glutamatergic disinhibition via GluClα | Composite PC/NC RF; 4-conductance model |
| **Key nonlinearity** | Shunting disinhibition (×) + GABA divisive (÷) | Pulse/delta ON-OFF conductances |
| **Rin measurements** | Yes — central to multiplication claim | Not emphasized |

## Related references

- **Borst 2018** (*PLoS Comput Biol*): 3-arm biophysical DS model (Mi9 × Tm3/Mi1, Mi4/C3 ÷)
- **Takemura et al. 2017** (*eLife*): T4 connectome — synapse positions along dendrite
- **Arenz et al. 2017** (*Curr Biol*): temporal tuning of motion detector inputs (same projector setup)
