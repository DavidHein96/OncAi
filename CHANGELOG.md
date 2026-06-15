# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-06-07

### Added

- **Physician-review bundles** (`oncai.review`). A completed function-calling
  extraction batch can now be turned into a single self-contained
  `*.review_pkg.json` — everything the local review app
  (`review_app_reference/`) needs to adjudicate findings (approve / reject /
  edit each event against its source note), with no DuckDB, lake, or repo
  required on the reviewer's side.
  - `oncai fc run-single` writes the review package next to the batch JSONL
    automatically when the run completes (best-effort — a packaging failure
    warns but never fails the run).
  - `oncai fc review-package <batch.jsonl>` builds a package for an
    already-completed batch, reading its `_manifest.json` sidecar to recover the
    note source. Supports `--definition` (richer field schema), `--output`, and
    `--db` overrides.
  - The package's `field_schema` is derived from each tool's Pydantic model, so
    the app renders the right control per field (enum dropdowns, ApproxDate
    widgets, checkboxes, evidence snippets), including across all phases of a
    gated registry. A registry-free fallback infers a usable schema from the
    observed events when a definition has since changed.

## [0.1.0] - 2026-06-03

### Added

- Inital public release
