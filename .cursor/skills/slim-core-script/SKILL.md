---
name: slim-core-script
description: >-
  One vision simulation core .py per turn: inline single-use locals into
  correct expressions; delete dead code. No new nouns outside lexicon.
  Use when the user says slim-core-script, slim next core, slim-core auto,
  or inline garbage locals in one core file.
---

# Inline one core script

Queue: `.cursor/skills/slim-core-script/slim-core-queue.md`.

## What this task is (only this)

**Primary:** find **single-use locals** and **delete the assignment** by putting the RHS expression at the sole use site (or into a return / call argument). The result must be a correct logical expression — not a rename.

**Secondary (only if still present after inlining):**
- Dead helpers / unused imports
- Thin wrapper with exactly one call site → inline at that site
- Unused locals / recompute already in hand / try-except that cannot fire

**Not this task:**
- Renaming for taste
- Coining new identifiers
- Multi-file renames
- Behavior / API changes
- Lexicon inventoring (lexicon already owns that)

## Hard stops (violate = revert + leave `[ ]` + stop)

1. **Exactly one** target `.py` per turn. First unchecked path in the queue, unless the user named a path.
2. Edit **only** that file + that queue checkbox. Grep elsewhere only to prove unused exports / call sites. Do **not** open the next queue file.
3. **No new nouns / verbs / adjectives** in identifiers. Allowed names:
   - lexicon headwords and compounds built only from them (see `.cursor/rules/lexicon.mdc`);
   - names **already present** in the target file that you keep unchanged;
   - parameter / attribute / method names you did not invent this turn.
   Forbidden this turn: inventing a substitute for a lexicon concept; inventing a synonym for an existing local (`vals` → `param_vals`, `base` → something you made up, opaque shorts → other opaque shorts). **Inline deletes the name; it does not replace it.**
4. If a local is used **more than once**, keep the assignment. Do **not** invent a “better” name for it. Only if the existing name is a **lexicon Forbidden** substitute or a banned abbreviation may you rename it — and then **only** to the exact lexicon headword for that concept (or the file’s existing canonical name for that concept). If the concept is not in the lexicon and the current name is already full English + `_`, leave it.
5. Shadowing a parameter with its payload (`param = params.get(param, …)`) = do not ship. Inline or keep a name that already exists in-file / lexicon for that bag — do not mint a new one.
6. Obey `.cursor/rules/coding-rules.mdc` §③ **Forbidden bindings** (abbreviation, opaque short locals, lexicon-headword misuse, non-lexicon glue locals); no fake `resolve_*`; no backward-compat shims.
7. After this file: mark `[x]` only if checks pass; report; **end turn**. Auto continuation is the `stop` hook only.

## Single-use local — definition and transform

A local is **single-use** when, after its assignment, it is read **exactly once** in that function / method / comprehension scope (including as a return operand or call argument).

Transform:
1. Delete `name = RHS`.
2. At the sole use, write `RHS` (or the same subexpression already required there).
3. RHS must still evaluate **once**; evaluation order must not change; side effects must not duplicate or disappear.
4. Do **not** introduce a new identifier in the process.

Keep the assignment when:
- used ≥2 times;
- RHS has side effects and you would evaluate it twice if inlined at multiple sites (already ≥2 uses);
- inlining would obscure control-flow (early return / branch) more than the assignment — then keep; still no rename.

## Mandatory before deleting / renaming any module-level name

Fail closed — revert, leave unchecked, report, end turn. No shim aliases.

1. **Repo Grep** under `vision/` for every module-level name you remove or rename.
2. Any external call site → **do not** delete/rename that export this turn.
3. Zero external uses → deletion only after smoke.

Local inlining inside a function does not need repo Grep for that local’s name.

## Mandatory smoke

Use `vision/.venv/bin/python` only.

1. Import the target module via logical name (`import import_bootstrap` from `vision/`).
2. If target is under `3_task/`, `4_train/`, or anything `run.py` imports at startup:

```bash
cd vision && .venv/bin/python -c "import import_bootstrap; import figure.plot"
```

   Fix import failures **only** by restoring symbols in the **target** file. If that is impossible without a second file, undo the slim and leave `[ ]`.

3. Do **not** mark `[x]` when smoke fails.

## Method (per file)

1. Read the **entire** target file.
2. Inventory **single-use locals** first; inline them.
3. Then inventory secondary dead code (above). Prefer delete/merge; **no new helpers** unless the user asked.
4. Scan the diff: every **added** identifier token must be lexicon-legal or already in the file before this turn. If you added a new noun, revert that hunk.
5. Run mandatory export Grep (if needed) + smoke.
6. Mark **only** this path `[x]` in `.cursor/skills/slim-core-script/slim-core-queue.md` if checks passed.
7. Reply: path, lines before→after, bullets of what was **inlined/deleted** (not renamed), smoke result. End turn.

## Auto mode

Arm (bind **one** chat; file contents = `conversation_id`):

```bash
echo '<conversation_id>' > .cursor/slim-core-auto.on
```

In **that** chat only, start with the followup prompt (same text the hook emits). Each turn = one file; `stop` hook (registered in **`.cursor/hooks.json`**) continues until queue empty, then removes `.cursor/slim-core-auto.on`.

Disarm: `rm -f .cursor/slim-core-auto.on`  
Log: `.cursor/slim-core-auto.log`  
Smoke fails twice in a row → disarm auto and stop.

## Out of scope

- `__init__.py` unless the user names it
- `experiment/`, `scratch/`, `simulation/analyze/`, `0_runs/`, `0_logs/`
- Behavior / public API changes
- Multi-file renames
- Coining identifiers for “clarity”
- Broad refactors that are not inline / dead-delete
