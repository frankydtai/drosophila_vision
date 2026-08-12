# Repository instructions

## Repository map

- `vision/`: vision simulation, training, analysis, and figures.
- `figure_digitization/`: paper-figure digitization tools.
- `SimulationCode/`: legacy simulation code.
- `NIPS2026/`: experimental research code.

## Scoped instructions

Before modifying a subproject, read its nearest `AGENTS.md` completely.

- For `vision/**`, read `vision/AGENTS.md`.
- For `figure_digitization/**`, read `figure_digitization/AGENTS.md`.

Instructions closer to a modified file take precedence over broader instructions.

## Working-tree safety

- Preserve unrelated user changes. Inspect `git status --short` before editing.
- Do not overwrite, revert, rename, or delete unrelated changed or untracked files.
- Keep `.cursor/` intact. It remains supported alongside Codex.
- Put repository Codex skills in `.agents/skills/` and lifecycle hooks in `.codex/`.

## Working method

- Make the smallest change that fully satisfies the request.
- Use the environment belonging to the relevant subproject.
- Run the narrowest relevant verification after editing.
- Report changed files, checks performed, and anything not verified.

