# Figure digitization instructions

Scope: `figure_digitization/**`.

## Environment

- Use `figure_digitization/.venv/bin/python` for every Python and pip operation.
- Never use bare `python`, `python3`, `pip`, or `pip3`.

## Re-fix and patch requests

When the user asks to re-fix, patch, or 修好 a digitization result:

1. Change only the panels, cells, or traces named by the user.
2. Prefer a narrow panel-local override, such as a per-panel dictionary, keyed branch, local ROI blank, or local `y_top`, over changing shared defaults.
3. Do not retune global extraction behavior in a way that can regress already-correct panels. If a shared helper must change, gate the new behavior behind the named panel keys so other panels keep the old path.
4. Verify that named panels improved and unnamed panels are unchanged, comparing peak time, point count, and shape against the pre-patch CSV or check image.
5. If an unnamed panel changes, remove the global portion and implement a local patch.

