---
name: paper-conversion
description: Convert a local research `paper.pdf` into raw `paper.txt` and a detailed `paper_summary.md`, emphasizing visual-stimulus protocols and recording conditions. Use when asked to extract, convert, or summarize a paper PDF in a model or experiment folder. Do not use for ordinary documentation summaries or when no accessible PDF is available.
---

# Convert a paper PDF

## Produce outputs

Create `paper.txt` and `paper_summary.md` next to the source PDF.

## Extract full text

1. Run `pdftotext paper.pdf paper.txt`; do not hand-transcribe the PDF.
2. Preserve the raw `pdftotext` output.
3. Read the complete `paper.txt` before summarizing; do not rely on the abstract alone.

## Write the summary

Start with:

```md
**Full text:** [`paper.txt`](paper.txt) | **PDF:** [`paper.pdf`](paper.pdf)
```

Include, in order:

1. `## Citation`: authors, year, journal, and DOI as a Markdown link.
2. `## One-line takeaway`.
3. Recording and stimulus sections first, using Markdown tables where practical.
4. A short model, connectome, and theory-results section at the end.

Make recording coverage exhaustive: animal preparation, microscope, laser, objective, detector, sample or volume rate, number of neurons, and number of flies.

Make display coverage exhaustive: projector, wavelength, screen coverage, background, and refresh rate.

List every stimulus protocol with concrete parameters such as speed, size, inter-stimulus interval, frequency, and duration. Check supplementary tables for these values.

If the folder contains model code, add a compact table mapping relevant files to their roles and include the data or code DOI.

## Verify

- Confirm that both output files are beside the PDF.
- Confirm that the summary was derived from the full extracted text.
- Check citation and DOI values against the paper text or authoritative metadata when web access is allowed.
- Report any recording or stimulus parameter that the source does not establish instead of guessing.

