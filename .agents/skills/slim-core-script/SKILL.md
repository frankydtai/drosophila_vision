---
name: slim-core-script
description: Audit and remove redundancy from exactly one `vision` simulation core Python file at a time, including dead code, thin wrappers, duplicate branches, and redundant computation. Use when the user says slim-core-script, slim next core, slim-core auto, or asks to audit one core file for redundancy. Do not use for broad multi-file refactors or behavior changes.
---

# Slim one core script

## Enforce scope

1. Process exactly one target `.py` file per turn. Use the first unchecked path in `.codex/slim-core-queue.md` unless the user names a path.
2. Edit only the target file and, after successful checks, its queue checkbox. Search other files only to prove a symbol unused or find call sites.
3. Read `vision/AGENTS.md` and the complete target file before editing.
4. End the turn after reporting on the one file. In auto mode, let the Codex Stop hook request the next continuation.

## Check before deleting or renaming

Fail closed: if a check fails, restore the removed or renamed symbol in the target, leave the queue item unchecked, report the blocker, and end the turn. Never add a compatibility alias.

1. Search all of `vision/` for every module-level function, class, or constant to remove or rename, including imports and attribute uses.
2. If any external call site exists, keep the public name. Limit the pass to dead internal code unless the user explicitly scopes a multi-file rename.
3. Permit deletion only when the repository search finds no external uses and the smoke checks pass.

## Slim method

Inventory and remove:

- Dead helpers and unused imports.
- Thin wrappers with one call site; inline at that call site when clearer.
- Near-duplicate functions differing only in constants; parameterize one implementation.
- Branches for unsupported or already-mandatory backends.
- Copy-pasted apply or finalize chains.
- Redundant recomputation, unused locals, and exception handlers that cannot fire.

Prefer deletion and merging over new helpers. Preserve public behavior.

## Run mandatory smoke checks

Use `vision/.venv/bin/python` only.

1. Import the target module through its logical name after `import import_bootstrap` from `vision/`.
2. For targets under task, train, or modules imported by the main run path, also run:

```bash
cd vision && .venv/bin/python -c "import import_bootstrap; import train; import figure.plot"
```

3. Fix an import failure only by restoring symbols in the target file. If that is insufficient, undo the breaking slim change and leave the queue item unchecked.
4. Mark only the completed target `[x]` after every mandatory check passes.

## Report

Report the target path, line count before and after, removed or merged redundancy, and smoke results. Then end the turn.

## Auto mode

Arm auto-continuation with:

```bash
touch .codex/slim-core-auto.on
```

Invoke `$slim-core-script` for one queued file. The trusted repository Stop hook continues while unchecked queue items remain and removes the marker when the queue is empty.

Disarm with:

```bash
rm -f .codex/slim-core-auto.on
```

Review and trust the project hook with `/hooks` before expecting auto mode to run.

## Keep out of scope

- `__init__.py` unless named by the user.
- `experiment/`, `scratch/`, `0_runs/`, and `0_logs/`.
- Opportunistic public behavior changes.
- Multi-file renames unless the user names every file in the request.
