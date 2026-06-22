# Innovation Library Corpus Audit

Generated at: `2026-06-20T17:40:48+00:00`

## Source Counts

- Source items scraped: 1527
- Download links found: 1609
- Downloads completed: 1607
- Downloads failed or blocked: 2

## Extraction Counts

- Source documents seen by extractor: 1609
- Successful extraction count: 1591
- Unsupported/skipped source files: 16
- Scanned/image-only or no-text documents skipped: 153
- Documents skipped after low-information filtering: 5

Latest extraction manifest statuses:
- `ok`: 1591
- `skipped_download_error`: 2
- `unsupported`: 16

Unsupported file types:
- `.mp4`: 16

Included file types:
- `.doc`: 4
- `.docx`: 12
- `.pdf`: 1417

Skipped no-text/image-only file types:
- `.jpeg`: 8
- `.jpg`: 82
- `.pdf`: 60
- `.png`: 3

Skipped low-information file types:
- `.html`: 2
- `.pdf`: 3

## Chunk Counts

- Raw chunks before filtering: 65612
- Chunks dropped as low-information: 5329
- Oversized chunks split: 3032
- Additional split chunk parts: 3454
- Final manifest chunks after filtering/splitting: 63737

## Final Manifest

- Final manifest document count: 1433
- Manifest path: `/Users/ben/Documents/UNDP/SGP/sgp ai/sgp_ai/corpus/manifest.yaml`

Language distribution:
- `ar`: 2
- `en`: 1063
- `es`: 180
- `fr`: 94
- `hy`: 2
- `pt`: 33
- `ru`: 18
- `sw`: 2
- `th`: 1
- `ur`: 1
- `vi`: 34
- `zh`: 3

Summary sources:
- `chunk`: 951
- `metadata`: 476
- `title`: 6

## Normalization Choices

- Source ID is `gef_sgp_innovation_library` for all included documents.
- Innovation Library detail URLs are used as document URLs; local file paths are excluded from YAML.
- Local filesystem path strings found inside source text are replaced with `[local file path omitted]`.
- The scraped `Languages` field is blank for all source items, so language is inferred from download labels, filenames, titles, and document text; unresolved cases default to `en`.
- `Kyrgystan` and `Kyrgyz Republic` are normalized to `Kyrgyzstan`; `Vietnam` is normalized to `Viet Nam`.
- Obvious `Adaption` typos are normalized to `Adaptation` in metadata-derived fields and extracted chunk text.
- Region labels are normalized to lowercase route-safe values such as `africa`, `asia_pacific`, and `latin_america_caribbean`.
- Priority groups are mapped into `audience_tags` and `programme teams` is added to every SGP document.
- Chunks under about 80 characters are dropped unless they contain meaningful factual/numeric content.
- Bare URLs, page numbers, table-of-contents fragments, copyright lines, photo credits, source labels, read counts, and last-modified lines are dropped.
- Chunks over roughly 2,000 characters are split into stable same-page subchunks.

## Remaining Decisions Or Exclusions

- Video files are excluded because transcription is not implemented.
- Image-only/scanned files with no extracted text are skipped; OCR is not implemented in this manifest build.
- Remote/shared backend deployments still require valid storage credentials for the configured LanceDB backend.
