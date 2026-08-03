# SimulationCode ↔ FAFB cell-type map

Scope: Borst / flyvis **SimulationCode** (`Circuits/ctype.npy`, 65 types) vs FAFB v783 built network **`right_min_neuron1`** (Matsliah FlyWire optic-lobe names) and its hex crop **`right_min_neuron1_extent10`**.

Counts below are **right** hemisphere in `4_built_networks/right_min_neuron1/`.

---

## 1. Neuron types in SimulationCode

### 1.1 Full inventory (`ctype.npy`, 65)

Order is the SimulationCode / flyvis index (0…64). Photoreceptors R1–R6 are separate slots; CT1 is split into two compartments.

| idx | name | idx | name | idx | name |
|----:|------|----:|------|----:|------|
| 0 | R1 | 22 | Mi3 | 44 | Tm2 |
| 1 | R2 | 23 | Mi4 | 45 | Tm3 |
| 2 | R3 | 24 | Mi9 | 46 | Tm4 |
| 3 | R4 | 25 | Mi10 | 47 | Tm5Y |
| 4 | R5 | 26 | Mi11 | 48 | Tm5a |
| 5 | R6 | 27 | Mi12 | 49 | Tm5b |
| 6 | R7 | 28 | Mi13 | 50 | Tm5c |
| 7 | R8 | 29 | Mi14 | 51 | Tm9 |
| 8 | L1 | 30 | Mi15 | 52 | Tm16 |
| 9 | L2 | 31 | T1 | 53 | Tm20 |
| 10 | L3 | 32 | T2 | 54 | Tm28 |
| 11 | L4 | 33 | T2a | 55 | Tm30 |
| 12 | L5 | 34 | T3 | 56 | TmY3 |
| 13 | Lawf1 | 35 | T4a | 57 | TmY4 |
| 14 | Lawf2 | 36 | T4b | 58 | TmY5a |
| 15 | Am | 37 | T4c | 59 | TmY9 |
| 16 | C2 | 38 | T4d | 60 | TmY10 |
| 17 | C3 | 39 | T5a | 61 | TmY13 |
| 18 | CT1(Lo1) | 40 | T5b | 62 | TmY14 |
| 19 | CT1(M10) | 41 | T5c | 63 | TmY15 |
| 20 | Mi1 | 42 | T5d | 64 | TmY18 |
| 21 | Mi2 | 43 | Tm1 | | |

Notes:

- **CT1(Lo1)** / **CT1(M10)** = lobula Lo1 vs medulla M10 terminals of one CT1 neuron (flyvis / Borst convention; counted twice in the 65).
- `FiveCol_MedSim_Python.py` sometimes aliases them as `CT1L` / `CT1M`.

### 1.2 Fit / gt subset (`Medulla_Library.cell_list`, 13)

These are the only types with impulse-response gt used in the classic Borst training loop:

`L1, L2, L3, L4, L5, Mi1, Tm3, Mi4, Mi9, Tm1, Tm2, Tm4, Tm9`

---

## 2. Neuron types in FAFB (`right_min_neuron1`)

### 2.1 Built network size

| quantity | value |
|---|---:|
| cells (types) | 682 |
| neurons (nodes) | 47291 |
| with hex `u,v` in `network.json` | 34 types |
| in `right_min_neuron1_extent10` | same 34 types (hex-disc crop) |

FAFB names follow **Matsliah et al.** FlyWire optic-lobe typing (`visual_neuron_types.csv` → `type`).

### 2.2 Types **with** column position (`u,v` in built `network.json`)

Two sources feed hex coordinates:

1. **Raw** FAFB `column_assignment` (`columns.csv.gz`) — 31 types.
2. **Assigned** via sole list `ASSIGNED_COLUMN_CELLS` in `3_assign_column.py` (imported by `4_build_network.py`) — partner-vote columns for **R1-6** (`post`), **Lawf1/Lawf2**, and SimulationCode-mapped FAFB types that lack native assignment (`Lai`, `Mi2`, `Mi10`, `Mi13–15`, `Tm5f`, `Tm5a–c`, `Tm16`, `Dm3v`, `Tm31`, `TmY3–5a`, `TmY9q` / `TmY9q__perp`, `TmY10`, `TmY11`, `TmY14`, `TmY15`, `Tm27`; all `pre` except R1-6). CT1 is **not** assigned (wide-field; stays without `u,v`).

After rebuild + `5_add_extent`, extent networks include raw columnar types **plus** every successfully located entry from that list. The snapshot below is the **pre-rebuild** 34-type set (raw 31 + R1-6/Lawf1/Lawf2 only):

| cell | n | with `u,v` | without `u,v` | column source |
|------|--:|--:|--:|---|
| R1-6 | 3437 | 3435 | 2 | assigned (`R1-6`, post) |
| R7 | 659 | 659 | 0 | raw |
| R8 | 655 | 654 | 1 | raw |
| L1 | 793 | 789 | 4 | raw |
| L2 | 791 | 789 | 2 | raw |
| L3 | 738 | 737 | 1 | raw |
| L4 | 724 | 711 | 13 | raw |
| L5 | 785 | 778 | 7 | raw |
| Lawf1 | 152 | 137 | 15 | assigned (`Lawf1`, pre) |
| Lawf2 | 168 | 149 | 19 | assigned (`Lawf2`, pre) |
| C2 | 743 | 743 | 0 | raw |
| C3 | 768 | 767 | 1 | raw |
| Mi1 | 796 | 796 | 0 | raw |
| Mi4 | 765 | 764 | 1 | raw |
| Mi9 | 770 | 766 | 4 | raw |
| T1 | 738 | 735 | 3 | raw |
| T2 | 725 | 717 | 8 | raw |
| T2a | 866 | 765 | 101 | raw |
| T3 | 823 | 729 | 94 | raw |
| T4a–T4d | (see network) | mostly yes | some | raw |
| T5a–T5d | (see network) | mostly yes | some | raw |
| Tm1 | 775 | 773 | 2 | raw |
| Tm2 | 767 | 766 | 1 | raw |
| Tm3 | 858 | 743 | 115 | raw |
| Tm4 | 734 | 719 | 15 | raw |
| Tm9 | 755 | 753 | 2 | raw |
| Tm20 | 744 | 740 | 4 | raw |
| Tm21 | 623 | 619 | 4 | raw |

T4/T5 with `u,v` (detail):

| cell | n | with `u,v` | without |
|------|--:|--:|--:|
| T4a | 737 | 728 | 9 |
| T4b | 748 | 743 | 5 |
| T4c | 842 | 778 | 64 |
| T4d | 777 | 750 | 27 |
| T5a | 743 | 735 | 8 |
| T5b | 760 | 744 | 16 |
| T5c | 766 | 739 | 27 |
| T5d | 727 | 704 | 23 |

**`extent10` cell list (34):**  
`C2, C3, L1–L5, Lawf1, Lawf2, Mi1, Mi4, Mi9, R1-6, R7, R8, T1, T2, T2a, T3, T4a–d, T5a–d, Tm1, Tm2, Tm20, Tm21, Tm3, Tm4, Tm9`

### 2.3 Types formerly without column (now on `ASSIGNED_COLUMN_CELLS`)

These SimulationCode-mapped FAFB names had no native `column_assignment` and were dropped by extent crops until partner assignment. They are listed in `ASSIGNED_COLUMN_CELLS` (`3_assign_column.py`); rebuild `4_build_network` + `5_add_extent` to place them. Counts below are still total neurons in `right_min_neuron1` (right):

| FAFB name | n | notes |
|-----------|--:|------|
| Lai | 231 | maps from SC `Am` |
| CT1 | 1 | maps from `CT1(Lo1)` / `CT1(M10)`; **not** on assign list |
| Mi2 | 427 | |
| Mi10 | 207 | |
| Mi13 | 350 | also absorbs SC `Mi12` |
| Mi14 | 127 | |
| Mi15 | 480 | |
| Tm5f | 397 | maps from `Tm5Y` |
| Tm5a | 219 | |
| Tm5b | 226 | |
| Tm5c | 308 | |
| Tm16 | 174 | |
| Dm3v | 320 | maps from `Tm28` |
| Tm31 | 54 | maps from `Tm30` |
| TmY3 | 359 | |
| TmY4 | 211 | |
| TmY5a | 572 | |
| TmY9q | 173 | half of SC `TmY9` |
| TmY9q__perp | 189 | other half of SC `TmY9` |
| TmY10 | 222 | |
| TmY11 | 178 | maps from `TmY13` |
| TmY14 | 185 | |
| TmY15 | 135 | |
| Tm27 | 559 | maps from `TmY18` |

Plus ~650 other FAFB types not in SimulationCode’s 65.

---

## 3. Name mapping (SimulationCode → FAFB) with evidence

### 3.1 Evidence sources

| shorthand | what it is | URL / path |
|-----------|------------|------------|
| **olmatching** | Official type↔type matching table across Nern male OL (`OL_type`), Matsliah FlyWire (`Matsliah_type` = FAFB names here), and Schlegel FlyWire (`Schlegel_type`) | [ol_annotations `data/olmatching.tsv`](https://github.com/flyconnectome/ol_annotations/blob/main/data/olmatching.tsv) |
| **CTE AKA** | Reiser Cell Type Explorer page header `AKA: … (Flywire, CTE-FAFB)` | e.g. [male CNS CTE](https://reiserlab.github.io/celltype-explorer-drosophila-male-cns/) |
| **Schlegel** | The `Schlegel_type` column *inside* olmatching (not a separate tool) | same TSV |
| **FIB connectivity** | Partner-weight cosine between flyvis `fib25-fib19_v2.2.json` and FAFB `right_min_neuron1` connections — used only when olmatching/CTE have **no row** | local connectomes |

How to read olmatching for our purpose: look up the SimulationCode / Nern / Schlegel name, then take **`Matsliah_type`** as the FAFB string in this repo.

### 3.2 Full map for the 65 SimulationCode types

| SimulationCode | FAFB name(s) | n (right) | with `u,v`? | evidence |
|----------------|--------------|----------:|:-----------:|----------|
| R1…R6 | R1-6 | 3437 | yes (assigned) | FAFB merges R1–R6 |
| R7 | R7 | 659 | yes | same name |
| R8 | R8 | 655 | yes | same name |
| L1…L5 | L1…L5 | (table §2.2) | yes | same name |
| Lawf1, Lawf2 | Lawf1, Lawf2 | 152, 168 | yes (assigned) | same name |
| **Am** | **Lai** | 231 | no | olmatching: `Schlegel_type=Am` → `Matsliah/OL=Lai`; FIB connectivity cosine ≈0.995 vs Lai (**not** Am1) |
| C2, C3 | C2, C3 | 743, 768 | yes | same name |
| **CT1(Lo1), CT1(M10)** | **CT1** | 1 | no | one neuron, two compartments (flyvis paper); FAFB type `CT1` |
| Mi1, Mi2, Mi4, Mi9, Mi10, Mi13–15 | same | (table) | Mi1/4/9 yes; others no | same name / olmatching 1-to-1 |
| **Mi3** | *(none)* | — | — | absent from olmatching & CTE; FIB edges too sparse to map |
| **Mi11** | *(none)* | — | — | same |
| **Mi12** | **Mi13** | 350 | no | no modern Mi12; FIB Mi12 connectivity ≈ Mi13 (merged with FIB Mi13) |
| T1, T2, T2a, T3, T4*, T5* | same | (table §2.2) | yes | same name |
| Tm1–4, Tm9, Tm16, Tm20 | same | (table) | Tm1–4,9,20 yes; Tm16 no | same name |
| **Tm5Y** | **Tm5f** | 397 | no | olmatching `Tm5Y→Tm5f`; [CTE Tm5Y AKA Tm5f](https://reiserlab.github.io/celltype-explorer-drosophila-male-cns/types/Tm5Y_R.html) |
| Tm5a, Tm5b, Tm5c | same | 219, 226, 308 | no | same name |
| **Tm28** | **Dm3v** | 320 | no | no modern Tm28 in olmatching/CTE; FIB Tm28 wiring (Tm1/L3/Tm2 → X → TmY4) matches Dm3v |
| **Tm30** | **Tm31** | 54 | no | olmatching `Tm30→Tm31`; [CTE Tm30 AKA Tm31](https://reiserlab.github.io/celltype-explorer-drosophila-male-cns/types/Tm30.html) |
| TmY3, TmY4, TmY5a, TmY10, TmY14, TmY15 | same | (table §2.3) | no | same name |
| **TmY9** | **TmY9q** + **TmY9q__perp** | 173 + 189 | no | olmatching: Schlegel `TmY9` ↔ `TmY9b→TmY9q`, `TmY9a→TmY9q__perp` |
| **TmY13** | **TmY11** | 178 | no | olmatching `TmY13→TmY11`; [CTE TmY13 AKA TmY11](https://reiserlab.github.io/celltype-explorer-drosophila-male-cns/types/TmY13.html) |
| **TmY18** | **Tm27** | 559 | no | olmatching `TmY18→Tm27`; [CTE TmY18 AKA Tm27](https://reiserlab.github.io/celltype-explorer-drosophila-male-cns/types/TmY18_L.html) |

### 3.3 Worked examples of online evidence

**Tm5Y → Tm5f (olmatching + CTE)**

```text
OL_type=Tm5Y   Matsliah_type=Tm5f   Schlegel_type=Tm5Y   matched as=1-to-1
```

CTE page title type `Tm5Y` with `AKA: Tm5f (Flywire, CTE-FAFB)`.

**Am → Lai (Schlegel column)**

```text
OL_type=Lai   Matsliah_type=Lai   Schlegel_type=Am   matched as=1-to-1
```

Do **not** confuse with `Am1` (separate wide-field ME–LO–LOP amacrine; n=1 in FAFB).

**TmY9 split**

```text
OL_type=TmY9a   Matsliah_type=TmY9q__perp   Schlegel_type=TmY9
OL_type=TmY9b   Matsliah_type=TmY9q         Schlegel_type=TmY9
```

**CT1 compartments**

flyvis / Nature paper wording: medulla vs lobula terminals listed as `CT1(M10)` and `CT1(Lo1)`; FAFB has a single cell type `CT1` (n=1, no column → excluded from extent networks).

### 3.4 Confidence tiers

| tier | mappings | basis |
|------|----------|--------|
| A — catalog | Tm5Y→Tm5f, Tm30→Tm31, TmY13→TmY11, TmY18→Tm27, TmY9→TmY9q(+__perp), Am→Lai, CT1 compartments→CT1, all same-name pairs | olmatching and/or CTE AKA |
| B — connectivity | Tm28→Dm3v, Mi12→Mi13 | FIB↔FAFB partner profiles; no olmatching row |
| C — unresolved | Mi3, Mi11 | no catalog row; FIB too sparse |

---

## 4. Practical checklist

1. Read SimulationCode name from `SimulationCode/Circuits/ctype.npy`.
2. Apply §3.2 map → Matsliah / FAFB string(s).
3. Look up counts in `4_built_networks/right_min_neuron1/cell_counts.csv`.
4. Column / extent eligibility:
   - has `u,v` in `network.json` → can appear in `*_extentN`;
   - else present only in full `right_min_neuron1`, not in extent crops.
5. If a name is missing, check **olmatching** `Matsliah_type` and **CTE AKA** before assuming absence.

---

## 5. Key paths in this repo

| path | role |
|------|------|
| `SimulationCode/Circuits/ctype.npy` | 65 Borst/flyvis type names |
| `SimulationCode/Medulla_Library.py` | `cell_list` (13 gt types) |
| `vision/connectome/FAFBv783/4_built_networks/right_min_neuron1/` | full right network + `cell_counts.csv` |
| `…/right_min_neuron1_extent10/network.json` | hex-cropped 34-type network |
| `vision/connectome/FAFBv783/3_assign_column.py` | sole `ASSIGNED_COLUMN_CELLS` (R1-6, Lawf1/2 + SC-mapped extras) |
| `vision/connectome/FAFBv783/4_build_network.py` | imports `ASSIGNED_COLUMN_CELLS` from `assign_column` |
| `flyvis/flyvis/connectome/fib25-fib19_v2.2.json` | FIB-25 type graph used for connectivity matching |
