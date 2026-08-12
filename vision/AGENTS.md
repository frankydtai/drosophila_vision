# Vision project instructions

Scope: `vision/**`.

## Hard stops

1. Before creating or renaming any function, method, class, parameter, variable, attribute, constant, CLI flag, file, or directory, read `.agents/references/vision-lexicon.md` from the repository root and search sibling code for established terminology. Use the lexicon's canonical term and never use a forbidden alternative.
2. Do not preserve backward compatibility unless the user explicitly asks for it. A request to change something to X means keep only X: no old path, shim, dual parser, deprecated parameter, or loader for old runs or sidecars.
3. When changing core code, do not update `simulation/test/**` merely to make it pass. Edit a test only when the user names that test or file, or when the requested task specifically requires it.
4. Use `vision/.venv/bin/python` for Python and pip operations in this subproject. Never use bare `python`, `python3`, `pip`, or `pip3`.
5. Do not add a new `.py`, `.sh`, `.ipynb`, or `.m` file unless the user has approved its path and responsibility.
6. A rename must update every corresponding function, method, attribute, constant, CLI surface, option key, docstring, comment, and call site in the same change.
7. Keep new code minimal. Prefer extending or replacing an existing function over creating a parallel implementation. Remove dead branches, unused locals, redundant recomputation, impossible defensive handlers, and unused symmetry APIs.
8. Keep detailed vocabulary definitions and forbidden alternatives only in the lexicon; do not duplicate them here.

## Interpreter map

| Script location | Interpreter |
|---|---|
| `vision/**` | `vision/.venv/bin/python` |
| `figure_digitization/**` | `figure_digitization/.venv/bin/python` |

## General script rules

1. Implement one canonical form and replace the old form; do not add a second path.
2. Do not accept aliases for the same concept. Existing group shorthand belongs only in `TASK_ALIASES`, `PART_COST_SCALE_ALIASES`, or `I_CLI_*_TASKS`.
3. Do not add duplicate functions or thin wrappers.
4. CLI list values use one comma-separated token. Exceptions are fixed-length pairs and top-level `KEY=VALUE` or bare-token bags. Use `parse_comma_list` from `import_bootstrap`; do not add another splitter.
5. Name CLI alias expanders `expand_<cli_name>_list` or `expand_<cli_name>_dict`, converting kebab case to snake case.
6. Use full words in parameters, attributes, locals, functions, files, and directories. Preserve an abbreviated user-facing CLI spelling only when the user explicitly requires it.
7. Join compound English words in identifiers with `_`.

## Simulation core architecture

The numbered simulation core is `1_neuron/`, `2_network/`, `3_task/`, `4_train/`, `5_figure/`, and `6_analyze/`. Non-core directories such as `experiment/` and `test/` are unnumbered. Imports use logical names only. Enable logical imports with `import import_bootstrap` and the project `simulation_sorted.pth`; renumbering changes disk names, not logical import strings or `vision/import_bootstrap.py`.

Number core modules by import and completion order. If A must be imported or completed before B, require `N(A) <= N(B)`, normally `N(A) < N(B)`. Share a number only for independent same-layer modules.

1. Add parameters through the schema; do not scatter dictionary indexing throughout the code.
2. Keep `run.py` a general driver. Importing train code must not parse arguments or touch CUDA.
3. Put path constants in logical `path.py` or `train/config.py`.
4. Keep CSV outputs rectangular; represent global scalars as constant columns.
5. Keep core code flexible enough for `experiment/` and `test/` to import and override without editing core code.
6. Core code must not import plotting layers. Shared logic needed by plotting and core belongs in core.
7. Numeric defaults have one source: `vision/default_params.py`. Keep it to literals and constant bags, without functions or formulas. Only train, figure, analyze, and run layers may import it; neuron, network, and task layers receive numbers by injection, except established moving-bar paradigm constants.
8. Never import or call `plot`, `plot_trained`, or other plot-layer modules from core modules.

## Non-core rules

1. Non-core train variants import core code.
2. Put train variants under `0_runs/<model>/run_<id>` with the complete train artifact set and established naming.
3. Do not rename or redirect files from old run directories after core changes; old artifacts may remain incompatible.

## Naming rules

1. Use singular names for one entity or concept and plural names for collections.
2. A script path contains exactly one verb across its parent directories and basename; do not duplicate the verb.
3. Use plural nouns for output directories.
4. Follow sibling naming patterns instead of inventing a new token for an existing idea.
5. Index, mask, lookup, and gathered-value locals must name both the object and its role; avoid opaque names.
6. Never concatenate two English words without `_`.
7. Do not use single-letter identifiers except the lexicon's explicitly allowed single-letter nouns.
