# Vision Code Lexicon (English)

This document defines a strict, non-overlapping meaning for selected terms used across `vision/`.
Only words with an unambiguous single definition are included here.

## Nouns (A-Z)

### `batch` / `batches`

Definition: A batch dimension/index used to represent parallel stimulus samples (e.g., which stimulus instance a cost entry belongs to).
Example: `active_batches = pack.readout_batch.unique(sorted=True)`

### `cell` / `cells`

Definition: A neuron type label from the connectome (e.g. `"L1"`, `"T4a"`), grouping multiple nodes that share the same biological identity.
Example: `C.cell_names` (connectome vocabulary mapping node indices to cell types)
Forbidden: do not use `node` to refer to a cell type. `cell` is a connectome type label; `node` is a specific numbered instance in the network.

### `column` / `columns`

Definition: A named field in a CSV output table (e.g. a parameter name in `param.csv`, or a target cell type in the `syn_strength_cell` matrix).
Example: `"per-cell column names"` in `save_param_table`
Forbidden: do not use `hex` to refer to a CSV column. `column` is a table field; `hex` is a spatial position in the connectome.

### `connectome` / `connectomes`

Definition: The loaded biological neural-circuit dataset (FAFB), instantiated as a `Network` object `C`; the source of `cell`, `syn`, `hex` vocabulary and connectivity structure.
Example: `C = backend.network` (the connectome object used throughout training)
Forbidden: do not use `network` to refer to the connectome object `C` or its biological contents (`cell`, `syn`, `hex`). `network` names the simulation graph, not the biological dataset.

### `cost` / `costs`

Definition: The scalar optimization objective value computed from readouts versus ground truth (loss).
Example: `best_cost = float(final_costs[int(np.argmin(final_costs))])`

### `data`

Definition: The set of trained-parameter output files (`.npy`, `.npz`, `train_opts.json`, etc.) saved under the `data/` subfolder of a run directory.
Example: `os.makedirs(run_data_dir(outdir), exist_ok=True)`
Forbidden: `artifact`, `output` — use `data` instead.

### `edge` / `edges`

Definition: A directed connection between two nodes in the network, carrying a signed weight derived from connectome synapse counts (`syn_sign * n_syn` or `syn_sign`).
Example: `source_index = np.empty(len(edges), dtype=np.int64)`
Forbidden: do not use `syn` to refer to a network edge or its index. `edge` is a network connection instance; `syn` is a connectome biological synapse.

### `entry` / `entries`

Definition: One cost comparison unit in a `ReadoutPack`: a (cell × radius) or (cell × PD/ND) record carrying its own `readout_node`, `gt`, `cost_weight`, and `batch`. The cost MSE is a weighted sum over all active entries.
Example: `entries = _active_entry_indices(work, session, batch_idx=batch_idx)`
Forbidden: do not use `node` to refer to a cost entry. `entry` is a cost-layer record; `node` is a network neuron index.

### `degree` / `degrees`

Definition: An angular or visual-field quantity measured in degrees (°) — used for bar width, lane pitch, hex vertex positions, and motion-axis field bounds.
Example: `x_deg, y_deg = build_hex.uv_to_xy_deg(u, v)`
Forbidden: do not use `extent` to refer to a degree quantity. `degree` is angular; `radius` is a hex ring count.

### `gt` / `gts`

Definition: Ground truth target signals/labels used as the training/evaluation reference (e.g., spot/moving-bar response waveforms).
Example: `gt_affine_for_nodes(p, pack.readout_node, backend, session=session)`

### `hex` / `hexes`

Definition: A single hexagonal spatial position in the connectome, identified by its axial coordinates `(u, v)`.
Example: `for hex in moving_bar_cost_hexes(C, cost_radius=cost_radius): ...`
Forbidden: do not use `column` to refer to a hex position. `hex` is a spatial location; `column` is a CSV table field.

### `map` / `maps`

Definition: A lookup table from one key space to another — a dict or index-array pair (e.g. node index → cell name, hex index → node index, cell name → integer index).
Example: `_node_to_cell_map(nodes_by_cell)` → `dict[int, str]`

### `mode` / `modes`

Definition: A string token selected from a fixed enumeration, identifying which variant of a configurable behaviour is active (e.g. `syn_mode ∈ {"per_cell","per_edge"}`, `train_mode ∈ {"indi","shared","fixed","frozen"}`, `pre_steady_mode ∈ {"probe","solve"}`).
Example: `mode = normalize_syn_mode(syn_mode)`

### `moving_bar` / `moving_bars`

Definition: The moving-bar stimulus paradigm (and its related task variants) used to generate inputs and corresponding targets.
Example: `if pack.name in MOVING_BAR_TASKS: ...`

### `network` / `networks`

Definition: The simulation graph used during training — the set of nodes and directed edges instantiated from the connectome into `ScatterConn` tensors for forward computation.
Example: `net = backend.network`
Forbidden: do not use `connectome` to refer to the simulation graph or its tensor representation (`node`, `edge`). `connectome` names the biological source dataset, not the in-memory training structure.

### `node` / `nodes`

Definition: A specific neuron instance in the network (indexed 0..N-1), used in tensors shaped `(N,)` or `(B,T,N)`.
Example: `pack.readout_node.shape[0]`
Forbidden: do not use `cell` to refer to a node index or node tensor. `node` is a numbered network instance; `cell` is a connectome type label.

### `opt` / `opts`

Definition: A set of input options (often parsed from CLI or sidecar JSON) used to configure sessions, stimuli, or costs.
Example: `opt = session.train_opts or {}`

### `param` / `params`

Definition: A single physical parameter value (single named scalar/vector quantity in the model).
Example: `p["a_gt"]` (a parameter entry used during cost computation)

### `part` / `parts`

Definition: One named sub-cost of the total training cost, identified by a `part_key` string (e.g. `"spot_bright_R8_r0"`, `"moving_bar_bright_PD"`) and holding its own scalar MSE tensor. The total cost is `Σ W·part / Σ W` over all active parts.
Example: `parts[part_key] = part` in `calc_cost_parts`

### `radius` / `radii`

Definition: An integer hex ring count labelling a position or area size — used for connectome geometry (`C.meta["radius"]`), spot footprint (`spot_radius`), sub-spot shift neighbourhood (`shift_radius`), bar width+spacing (`bar_radius`), and cost hex-disc (`cost_radius`). Does not refer to Euclidean or angular distances.
Example: `connectome_radius = int(C.meta.get("radius", -1))` (JSON key `"radius"` in `network.json`)
Forbidden: do not use `extent` to refer to any hex ring count. `radius` is the sole term for hex ring counts.

### `iter` / `iters`

Definition: One optimizer iteration (a single Adam gradient update), used to count training progress (`nofiters`, `global_iter`, `checkpoint_interval`).
Example: `nofiters = NOFITERS_GPU if cuda_available else NOFITERS_CPU`

### `run` / `runs`

Definition: One independent training execution from initialization to convergence, producing one set of fitted parameters.
Example: `for i in range(nofruns): ...`

### `schema` / `schemas`

Definition: The parameter schema (segment list) describing how physical parameters are packed into the trainable space `z` and mapped back.
Example: `for seg, start, stop in schema_segments(schema): ...`

### `session` / `sessions`

Definition: A run-time context object that holds assembled configuration/state for one training/evaluation run (schema/backend/tasks/readouts/opts).
Example: `session = open_session_from_outdir(...)`

### `spot` / `spots`

Definition: The spot stimulus paradigm (and its related task variants) used to generate inputs and corresponding targets.
Example: `if task_name in SPOT_TASKS: ...`

### `syn` / `syns`

Definition: A synapse in the connectome: the biological connection between two cells, characterized by its sign (`syn_sign`) and count (`n_syn`), from which edge weights are derived.
Example: `syn_sign = float(e["syn_sign"])` — from `network.json` connectome data
Forbidden: do not use `edge` to refer to a synapse or its biological properties. `syn` is a connectome biological entity; `edge` is its network instantiation.

### `task` / `tasks`

Definition: A training/evaluation task type (e.g., `spot_bright`, `spot_dark`, `moving_bar_bright`, `moving_bar_dark`).
Example: `for name in session.tasks: ...`

### `trace` / `traces`

Definition: A time sequence of a signal (e.g., membrane voltage readout waveform) represented as a tensor over time indices.
Example: `trace_full, onset_trace = _forward_readout_and_onset_trace(...)`

### `val` / `vals`

Definition: A short-lived scalar value in a loop or inline expression (not a named physical parameter).
Example: `for key, val in overrides.items(): out[key] = float(val)`

## Verbs (A-Z)

### `add`

Definition: Register a new argument or option onto a CLI parser object.
Example: `add_training_arguments(parser)`, `add_plot_filter_argument(parser)`
Forbidden: do not use `add` for tensor arithmetic or dict insertion — `add` is exclusively for parser argument registration.

### `apply`

Definition: Merge a set of override values into an existing object, mutating or replacing fields to produce the updated version.
Example: `apply_pack_override(pack, override, backend)`, `apply_train_modes(schema, train_modes_by_name, node_names_for_seg)`
Forbidden: `inject` — use `apply` only for override/merge operations; use `inject` for arithmetic injection into a tensor.

### `build`

Definition: Create/construct a new object/structure in memory from inputs (not necessarily persistent).
Example: `spot = build_spot(C, spot_radius=..., multi_spot=..., fully_inside=...)`
Forbidden: `make` — use `build` instead.

### `expand`

Definition: Expand shorthand/alias inputs into a concrete list/dictionary of explicit targets.
Example: `tl = expand_tasks(list(tasks))`

### `inject`

Definition: Arithmetically add a parameter-scaled contribution into an existing tensor in-place (returning the modified tensor).
Example: `i_sti = inject_a_sti_radius(i_sti, p, pack)`
Forbidden: `apply` — use `inject` only for arithmetic tensor injection; use `apply` for override/merge operations.

### `forward`

Definition: Run the full time-loop simulation on `i_sti` to produce a `(B, T, N)` readout tensor (`v` or `ca`).
Example: `v = forward_v(session, p, i_sti, pack=pack)`
Forbidden: `simulate`, `compute` — use `forward` instead.

### `load`

Definition: Load a saved artifact or serialized object from storage and reconstruct it for use.
Example: `z = train.load_best_param(outdir, session)`
Forbidden: `read` — use `load` instead.

### `normalize`

Definition: Convert an input value into a canonical/valid form (e.g., clamp rules, ordering, or standard representation).
Example: `mode = normalize_syn_mode(syn_mode)`
Forbidden: `canonicalize`, `standardize` — use `normalize` instead.

### `parse`

Definition: Decode a raw string or token sequence into a structured in-memory value.
Example: `parse_moving_bar_spec(sname)` → `(color, direction, variant)`

### `plot`

Definition: Produce a visualization output (figures/curves/images) from run results or computed traces.
Example: `plot_cost(result.cost_curve, out_path, costs_by_part=...)`
Forbidden: `draw`, `render`, `visualize` — use `plot` instead.

### `resolve`

Definition: Resolve abstract references/aliases/defaults into final concrete values used by execution.
Example: `out[tname] = int(expanded[tname])`
Forbidden: `lookup`, `fetch` — use `resolve` instead.

### `run`

Definition: Execute a procedure end-to-end (training, evaluation, or full pipeline) and return or persist its result.
Example: `fname, outdir, session = run_training_and_plot(plot_gt_cubes=..., **kw)`
Forbidden: `execute`, `invoke` — use `run` instead.

### `save`

Definition: Persist in-memory arrays/objects to disk in a machine-recoverable form (e.g., `.npy/.npz` or serialized state).
Example: `np.savez_compressed(path, hex_current=hex_current)`
Forbidden: `write`, `dump`, `export` — use `save` instead.

### `step`

Definition: Advance the neuron state by one time sample (`t-1 → t`) using the model dynamics.
Example: `state, v = drv.step(state, v, p, i_sti[:, t-1], session, delta_ms=...)`
Forbidden: do not use `step` to mean an optimizer iteration — use `iter` instead.

### `train`

Definition: Run an optimization procedure that updates trainable parameters to minimize `cost`.
Example: `z_fit, opt_state = train_staged(z, cost_fn, bounds, lrs, nofiters, ...)`
Forbidden: `fit`, `optimize` — use `train` instead.

## Prepositions (A-Z)

### `_from_`
Definition: The sole directional preposition in function names that derive a result from a source — whether the source is a container/config (opts dict, outdir, args, state_dict) or a quantity (signal, tensor, index space).
Example: `open_session_from_outdir(outdir)`, `t_from_ms(ms, delta_ms)`, `v_ca_from_v(v, p, session)`, `node_values_from_z(z, schema)`, `hex_from_uv(u, v)`
Forbidden: `_to_` — never use `_to_` as a directional preposition in function names.
