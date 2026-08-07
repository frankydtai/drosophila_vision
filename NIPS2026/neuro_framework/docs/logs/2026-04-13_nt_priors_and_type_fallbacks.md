# Neurotransmitter Priors and Type-Fallback Rules

## Date
2026-04-13

## Goal
Externalize two kinds of initialization heuristics from `FAFBMCNetwork` into standalone configuration:

1. neurotransmitter-dependent ion-parameter priors
2. similar-neuron-type fallback / transfer rules

This keeps the model code cleaner and makes the biological assumptions explicit,
versioned, and easier to update when new literature or fitted parameters become available.

## Added files
- `neuro_framework/data/fafb_nt_ion_priors.csv`
- `neuro_framework/data/fafb_type_fallback_rules.csv`
- `neuro_framework/models/fafb_param_config.py`

## How the hierarchy now works
For ion parameters, initialization priority is:

1. neurotransmitter prior (`fafb_nt_ion_priors.csv`)
2. neuron-type fitted/default rule (`fafb_ion_channel_rules.csv`)
3. root-id override (`neuron_ion_overrides_path`)

Important detail:
- neurotransmitter prior is only used as a fallback when the neuron type has no direct fitted rule (or no configured fallback rule that resolves to one)
- root-level overrides always win

## Literature basis
### 1. Lamina transmitter evidence
Kolodziejczyk et al. 2008, *PLoS ONE*:
- photoreceptors use **histamine**
- **C2** and **C3** are likely **GABAergic**
- **L4** is suggested to be **cholinergic**
- lamina monopolar cells such as **L1/L2** show evidence connected to **glutamatergic** signaling markers

Source:
- Kolodziejczyk A, Sun X, Meinertzhagen IA, Nässel DR (2008)
  *Glutamate, GABA and Acetylcholine Signaling Components in the Lamina of the Drosophila Visual System*
  https://doi.org/10.1371/journal.pone.0002110

### 2. Broad glutamatergic candidate map across the visual system
Raghu et al. 2011, *PLoS ONE*:
- identifies many candidate **glutamatergic** neurons across lamina / medulla / lobula / lobula plate
- includes examples from **L**, **Tm**, **TmY**, **Mi**, **Dm**, **Pm** families
- supports the idea that family-level transmitter/intrinsic priors are more reasonable than treating all unknown visual neurons identically

Source:
- Raghu SV, Borst A (2011)
  *Candidate Glutamatergic Neurons in the Visual System of Drosophila*
  https://doi.org/10.1371/journal.pone.0019472

### 3. Modern connectome-level transmitter annotation
Recent visual-system connectome work explicitly integrates neurotransmitter identity into optic-lobe cell typing.
That supports using neurotransmitter class as a first-pass prior for unfit or sparsely constrained types.

Useful contemporary sources:
- *Connectome-driven neural inventory of a complete visual system* (Nature, 2025)
- *Neuronal parts list and wiring diagram for a visual system* (Nature, 2024)
- *Neurotransmitter classification from electron microscopy images at synaptic sites in Drosophila melanogaster* (Cell, 2024)

## What is direct evidence vs heuristic
### Directly literature-supported or dataset-supported
- histamine for photoreceptor-driven visual input
- GABA identity for C2 / C3 in lamina literature
- cholinergic identity for some lamina visual neurons such as L4
- strong evidence that transmitter identity varies systematically across visual neuron families

### Explicit heuristics used only for initialization
- translating transmitter class into HH parameters (`gNa`, `gK`, `gLeak`, `eLeak`)
- merging T4/T5 subtypes into a shared motion-pathway prior
- using nearby family members as fallback (`Tm1 -> Tm3`, `L2 -> L1`, etc.)
- initializing T2/T2a/T3 from Mi1 as pathway proxies

These rules are **initial priors**, not claims that the final fitted biophysical parameters are identical.

## Current fallback rationale in `fafb_type_fallback_rules.csv`
### Aliases
- `R1-R6 -> R1-6`

### Family merges / close-subtype transfers
- `T4b/c/d -> T4a`
- `T5a/b/c/d -> T4a`

Rationale:
- all are closely related motion-pathway cell classes
- useful as a shared prior until subtype-specific fitting is available

### Family-neighbor fallbacks
- `L2 -> L1`
- `L4 -> L3`
- `Tm1/2/4/9/20/21 -> Tm3`

Rationale:
- within-family initialization is preferable to a global default

### Pathway proxies
- `T1 -> Mi1`
- `T2/T2a/T3 -> Mi1`
- `C2/C3 -> L5`

Rationale:
- these are practical optic-lobe initialization rules, not direct biophysical claims
- they should be replaced by fitted type-specific parameters once available

## Why CSV instead of hard-coded dictionaries
- easier to inspect and edit
- lets us version biological assumptions separately from solver code
- simplifies future replacement with transcriptomic or literature-updated priors
- easier to export and compare in notebooks

## Recommended next step
The next clean extension is to add a notebook cell or utility that prints, for every type in the current optic-lobe network:
- inferred dominant neurotransmitter
- whether it used direct type rule, fallback type rule, nt prior, or root override
- which fallback target was used if any
