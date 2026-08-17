---
name: slim-core-script
description: >-
  Audit and delete redundancy in one vision simulation core script at a time
  (dead code, thin wrappers, duplicate branches), matching the 5_session slim
  method. Use when the user says slim-core-script, slim next core, slim-core
  auto, or audit one core file for redundancy.
---

# Slim one core script

## Hard limits (violate = stop)

1. **Exactly one** target `.py` file per turn. Prefer the first unchecked path in `.cursor/slim-core-queue.md`. If the user names a path, use that instead.
2. Edit **only** that file (plus the queue checkbox). Grep other files only to prove a symbol is unused / find call sites before deleting an export. Do **not** edit a second core script in the same turn.
3. After finishing this one file: mark it `[x]` in the queue **only if** the mandatory checks below pass; report; **end the turn**. Do **not** open the next file yourself. Continuation (if any) is done by the `stop` hook when auto is armed.
4. Obey `.cursor/rules/coding-rules.mdc` (especially HARD STOP 6 minimal code, 7 full internal names, no backward-compat shims).

## Mandatory before deleting / renaming any module-level name

Fail closed — if any step fails, **revert the deletion/rename in the target file**, leave queue unchecked (or leave `[ ]`), report the blocker, end turn. Do **not** invent a shim alias in the target file.

1. **Repo Grep** under `vision/` for every name you will remove or rename (defs, constants, functions, classes). Include `from … import name` and attribute uses.
2. If **any call site exists outside the target file**:
   - **Do not delete/rename that symbol this turn.** Keep the old public/module-level name. Slim only dead *internal* code, or parameterize without changing the export name.
   - Multi-file rename is **out of scope** for auto slim unless the user named every file in the same message.
3. If Grep shows **zero** external uses, deletion is allowed only after the smoke below.

## Mandatory smoke (not optional)

Use `vision/.venv/bin/python` only (never bare `python` / `python3`).

1. Import the target module via logical name (with `import import_bootstrap` from `vision/`).
2. If the target is under `3_task/`, `4_training/`, or anything `run.py` imports at startup, also:

```bash
cd vision && .venv/bin/python -c "import import_bootstrap; import figure.plot_run"
```

   Fix import failures **only by restoring symbols in the target file** this turn (do not edit figure/other cores). If restore is impossible without a second file, undo the slim change that broke the import and leave the queue item unchecked.

3. Do **not** mark the queue `[x]` when smoke fails.

## Method (same as `5_session` slim)

1. Read the **entire** target file.
2. Inventory redundancy:
   - Dead helpers / unused imports
   - Thin wrappers with a single call site → inline or lambda at the only use
   - Near-duplicate functions (same logic, different constants) → one parameterized function
   - Dead branches (e.g. backend already required to be `"network"`)
   - Copy-paste apply_/finalize chains → one place
   - Recompute / unused locals / try-except that cannot fire
3. Apply the smallest correct diff. Prefer delete/merge over new helpers.
4. Run **Mandatory before deleting** + **Mandatory smoke**.
5. Update `.cursor/slim-core-queue.md`: mark **only** this path `[x]` if checks passed.
6. Reply with: path, lines before→after, bullet list of what was removed/merged, smoke result. End turn.

## Auto mode (leave and continue)

- Arm (bind **one** chat by conversation id):

```bash
echo 'd916db21-1c8e-4df7-8cd1-07d9efa3a4e3' > .cursor/slim-core-auto.on
```

  File contents = that chat’s `conversation_id`. Other Agent windows get no follow-up.
- Start/continue **only** in that chat with the slim-core prompt.
- Each turn does **one** file; `stop` hook injects the next prompt until the queue is empty, then removes `slim-core-auto.on`.
- Disarm early: `rm -f .cursor/slim-core-auto.on`
- Log: `.cursor/slim-core-auto.log`
- If smoke fails twice in a row on related files, **disarm auto** and stop so a human can fix cross-file breakage.

## Out of scope

- `__init__.py` unless the user names it
- `experiment/`, `scratch/`, `0_runs/`, `0_logs/`
- Refactors that change public behavior “while you’re there”
- Multi-file renames (auto must not rename exports that still have external call sites)
