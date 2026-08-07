# TODO & Milestones

> `neuro_framework` — NeurIPS 2026 submission  
> Last updated: 2026-03-30

---

## Deadlines

| Milestone | Date | Days left |
|-----------|------|-----------|
| Abstract submission | 2026-05-04 | ~35 days |
| Full paper submission | 2026-05-06 | ~37 days |

---

## Phase 1 — Framework Foundation ✅ COMPLETE

- [x] `connectome/loader.py` — BANC + FAFB + optic_lobe + FlyVis unified loader
- [x] `models/dynamics.py` — VoltageModel, LIFModel, HHModel (fixed pre_idx indexing)
- [x] `models/network_torch.py` — PyTorch ConnectomeNetwork (fixed empty edge handling)
- [x] `models/network_jax.py` — Jaxley JaxleyNetwork
- [x] `stimulus/visual.py` — Flash, MovingBar, MovingEdge, Grating
- [x] `training/losses.py` — MSE, correlation, DSI, knockout losses
- [x] `training/trainer.py` — TorchTrainer (Method A/B), JaxTrainer
- [x] `utils/logging.py` — centralised logging to `logs/`
- [x] `tests/test_loader.py` — 20 passing tests (BANC, optic_lobe, FlyVis, network build)
- [x] `notebooks/01_connectome_and_network.ipynb` — demo notebook
- [x] `docs/` — CHANGELOG, architecture, todo, implementation_summary

**Test results**: 20 passed, 2 skipped (FAFB data not downloaded)  
**Date completed**: 2026-03-30

---

## Phase 2 — Data Integration (priority: HIGH)

### 2.1 FAFB LC Connectome
- [ ] Download / place FlyWire FAFB v783 files under  
      `Connectome Dataset/FAFB/` (e.g. `consolidated_cell_types.csv.gz`, `classification.csv.gz`, `neurons.csv.gz`, `connections_princeton.csv.gz`)
- [ ] Identify LC neuron root IDs for subtypes:  
      `LC4`, `LC6`, `LC9`, `LC10`, `LC11`, `LC13`, `LC15`, `LC16`, `LC17`, `LC18`, `LC21`, `LC22`, `LC26`
- [ ] Test `ConnectomeLoader.from_fafb(cell_types=['LC4', 'LC6'])` end-to-end
- [ ] Verify `nt_sign()` correctly maps LC subtypes (most are cholinergic)
- [ ] Compare subgraph statistics with DMN paper Table S1

### 2.2 Ground-Truth Neural Activity
- [ ] Locate / obtain LC calcium imaging dataset  
      (referenced in "LC bottleneck paper" — confirm data source with Jizheng)
- [ ] Write `data/calcium/loader.py`:  
      loads traces, aligns to stimulus timing, returns `(n_trials, T, n_lc)` tensor
- [ ] Verify temporal resolution matches simulation `dt`
- [ ] Write unit test: loader produces finite, normalised traces

### 2.3 BANC Integration Check
- [ ] Confirm BANC `neurons.csv.gz` contains optic-lobe cell types
- [ ] Test `ConnectomeLoader.from_banc(cell_types=['T4a','T4b','LC4'])` 
- [ ] Document which neuropils to filter for the visual pathway

---

## Phase 3 — Training Experiments (priority: HIGH)

### 3.1 Method A — DMN-style knockout training
- [ ] Select target LC subtypes and matching calcium imaging data
- [ ] Build `x_full` and `x_knockout` stimulus pairs (silence T4/T5 types)
- [ ] Run `TorchTrainer.step_with_knockout()` for 200 epochs
- [ ] Convergence check: train loss < 0.05 within 100 epochs
- [ ] Evaluate: compare DSI to DMN baseline and RF baseline
- [ ] Hyperparameter sweep: supervised_weight vs knockout_weight

### 3.2 Method B — Layer-wise progressive training
- [ ] Define layer groups: photoreceptors → T4/T5 → Tm → LC → DN
- [ ] Run `TorchTrainer.layerwise_train()` with 50 epochs per layer
- [ ] Compare final performance to Method A
- [ ] Ablation: skip layer-wise (train all at once) as control

### 3.3 Dynamics ablation
- [ ] Train same connectome with VoltageModel / LIFModel / HHModel
- [ ] Metric: correlation with calcium ground truth, DSI score
- [ ] Report training wall-clock time per model

---

## Phase 4 — Jaxley / Biophysics Track (priority: MEDIUM)

- [ ] Benchmark `JaxleyNetwork` (HH channels) on optic-lobe subset  
      (use `malecns_09_optic_lobe_hex_08` feather files from Jaxley tutorial)
- [ ] Fit synapse conductance `g_S` to match spike rates from calcium data
- [ ] Compare HH vs voltage-model on direction selectivity task
- [ ] Profile memory and speed on M1 Mac vs GPU server

---

## Phase 5 — Evaluation & Baselines (priority: HIGH)

- [ ] Implement DMN baseline runner (use `flyvis/` code directly)
- [ ] Implement RF baseline runner (use `Shiu/` notebook)
- [ ] Define evaluation metrics:
  - Pearson r with calcium traces
  - Direction Selectivity Index (DSI)
  - Null-direction suppression ratio
  - Knockout prediction accuracy
- [ ] Statistical testing: bootstrap CIs over model ensembles
- [ ] Produce comparison table: neuro_framework vs DMN vs RF

---

## Phase 6 — Figures & Paper (priority: HIGH after Phase 3)

- [ ] Figure 1: Architecture diagram (retina → T4/T5 → LC → DN)
- [ ] Figure 2: Connectome statistics (real FAFB LC vs DMN simplified)
- [ ] Figure 3: Training curves, Method A vs B
- [ ] Figure 4: DSI heatmaps across LC subtypes
- [ ] Figure 5: Calcium trace prediction (model vs ground truth)
- [ ] Figure 6 (optional): NeuroMechFly demo
- [ ] Supplementary: HH vs voltage model ablation
- [ ] Write paper draft in LaTeX (NeurIPS 2026 template)

---

## Phase 7 — Demo & Code Release (priority: LOW)

- [ ] NeuroMechFly DN interface: map DN activity → body control signals
- [ ] Demo notebook: end-to-end visual input → behaviour output
- [ ] Clean up code for open-source release
- [ ] Write `setup.py` / `pyproject.toml`
- [ ] Add unit tests for loader, dynamics, stimulus, trainer
- [ ] CI pipeline (GitHub Actions)

---

## Known Issues / Blockers

| Issue | Owner | Status |
|-------|-------|--------|
| FAFB LC root IDs not yet confirmed | Jizheng | Pending |
| Calcium imaging data source unclear | Jizheng | Pending |
| Jaxley version pinning (sjcabs branch) | Frank | Pending |
| BANC cell_type column format varies | Frank | Needs test |
| GPU server access for large-scale runs | Team | TBD |

---

## Questions to Resolve

1. **Which LC subtypes** are in the calcium imaging dataset? (LC4? LC6? All?)
2. **Stimulus protocol** for ground truth: moving bar only, or also flashes/gratings?
3. **Simulation dt**: 1.0 ms (DMN default) or 0.025 ms (Jaxley HH default)?
4. **BANC vs FAFB**: use FAFB for LC layer (denser, better annotated), BANC as validation?
5. **Knockout ground truth**: which silencing experiments are published?

---

## Task Assignment (v3.0)

| Task | Owner | Target |
|------|-------|--------|
| LC root ID extraction from FAFB | Jizheng | 2026-04-05 |
| Calcium data loader | Jizheng | 2026-04-07 |
| Method A training run | Frank | 2026-04-10 |
| Method B training run | Frank | 2026-04-17 |
| DMN / RF baseline eval | Frank | 2026-04-12 |
| Jaxley HH benchmark | Frank | 2026-04-20 |
| Figures 1–5 | Jizheng + Frank | 2026-04-28 |
| Paper draft | Jizheng | 2026-05-01 |
| Submission | Team | **2026-05-06** |
