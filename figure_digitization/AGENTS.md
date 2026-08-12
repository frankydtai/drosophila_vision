# Figure digitization instructions

Scope: `figure_digitization/**`.

## Environment

- Use `figure_digitization/.venv/bin/python` for every Python and pip operation.
- Never use bare `python`, `python3`, `pip`, or `pip3`.

## Building a digitization script

Follow this workflow for a new figure digitizer. Do not hand off a script after
only confirming that it runs.

1. Read the paper caption and nearby text to establish the panel meaning,
   stimulus duration, units, trace identities, position ordering, and expected
   number of traces.
2. Inspect the source raster at full resolution with multimodal vision. Count
   every small panel and identify legends, labels, scale bars, stimulus markers,
   SEM bands, adjacent-panel overlap, and all positive and negative excursions.
3. Measure calibration from pixels in the supplied raster. Derive time from the
   printed time scale or stimulus-onset/offset markers and voltage from the
   printed voltage scale. Measure panel origins and steps from repeated markers;
   do not estimate them from appearance or inherit values from another figure.
4. Choose ROIs that contain the complete traces, including the deepest
   downward trough and the full recovery tail. A baseline near a crop boundary
   is a warning that negative responses may be clipped. Inspect colour-pixel
   extents outside the provisional crop before accepting it.
5. Extract the printed mean line, not the SEM envelope. Treat antialiasing and
   colour changes at crossings explicitly. Use panel-local masks or bounds to
   exclude captions, legends, scale bars, and neighbouring-panel tails.
6. Track line continuity through crossings and short colour gaps. Never replace
   missing post-stimulus pixels with baseline or zero. Pre-stimulus baseline
   padding is allowed only when the two plotted traces visibly overlap there and
   the shared baseline is measured from visible pixels in the same panel.
7. Emit a tidy CSV with stable trace identifiers and explicit experimental
   metadata, plus a check PNG that preserves the source panel ordering and uses
   shared, correctly calibrated axes.

## Required validation before handoff

Validation is an iterative visual and numerical process. Repeat extraction and
inspection until all checks pass.

1. Produce a pixel overlay for every panel: draw the extracted mean path over
   the original raster in a high-contrast diagnostic colour. Inspect the overlay
   with multimodal vision, at enlarged resolution when necessary.
2. Compare every trace against the source, checking stimulus onset and offset,
   peak and trough timing, crossings, downward excursions below zero, recovery
   tails, and the last visible sample. A separate digitized-only check plot is
   useful but does not replace the source-raster overlay.
3. Specifically look for premature endpoints, flat zero segments, straight
   interpolation across real curvature, jumps to text or scale bars, and traces
   that follow an SEM edge instead of the mean. Any of these fails validation.
4. Report per-trace point count, time extent, minimum and maximum value, and
   peak and trough time. Assert the expected trace count, finite numeric values,
   monotonic time within each trace, and no duplicate `(trace_id, time)` rows.
5. Run the script with `figure_digitization/.venv/bin/python` and compile it with
   the same interpreter. Regenerate both CSV and check PNG from the final code.
6. Do not claim completion while any panel is visibly truncated, clamped to
   zero, contaminated by annotations, or inconsistent with the source raster.

## Re-fix and patch requests

When the user asks to re-fix, patch, or 修好 a digitization result:

1. Save the pre-patch CSV and check image as temporary comparison artifacts
   before regenerating outputs.
2. Change only the panels, cells, or traces named by the user.
3. Diagnose the failure in source-pixel coordinates before editing. Check crop
   extent, mask coverage, extracted endpoint, trough value/time, and whether a
   label or adjacent trace entered the ROI.
4. Prefer a narrow panel-local override, such as a per-panel dictionary, keyed
   branch, local ROI blank, colour-specific bound, or local `y_top`/`y_bottom`,
   over changing shared defaults.
5. Do not retune global extraction behavior in a way that can regress already
   correct panels. If a shared helper must change, gate the new behavior behind
   the named panel or trace keys so other panels keep the old path.
6. Regenerate the outputs and repeat the original-raster pixel-overlay review
   for every changed trace. Continue patching while any named trace remains
   clipped, clamped, discontinuous, or visually misaligned.
7. Verify quantitatively that named traces improved by comparing point count,
   time extent, minimum/maximum, peak/trough time, and shape with the pre-patch
   CSV and check image.
8. Prove that unnamed traces are unchanged, preferably by exact dataframe
   comparison. If an unnamed trace changes, remove the global portion and
   implement a more local patch.
9. Report the changed files, the named traces repaired, the visual comparisons
   performed, numerical before/after evidence, unchanged-trace result, and all
   commands used for final verification.
