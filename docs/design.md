# Design Philosophy

Key decisions and why they were made.

## Parquet over CSV

CSV is the lingua franca of clinical data exports, but it's a terrible storage format — no types, no compression, slow to scan. Parquet gives us columnar compression (zstd), embedded schemas, and native DuckDB/Polars support. CSVs enter through the inbox and never persist beyond the ingest step.

## Content Hashing for Incremental Updates

Every row gets two hashes:

- **key_hash** (Blake2b-128) — computed from identity columns (e.g., `report_id`). Determines if a row already exists.
- **content_hash** — computed from value columns (e.g., `report_text`). Determines if an existing row has changed.

This makes re-ingestion safe and cheap — only genuinely new or changed rows are written. No timestamps, no "last modified" tracking, no coordination with the source system. After ingestion, source files are archived to `inbox/archive/` so they are never re-processed, eliminating redundant work on re-runs.

## JSONL as LLM Output Format

Extraction writes JSONL (one JSON object per line). This is deliberate:

- **Crash-safe** — each record is flushed immediately. If the process dies mid-batch, you lose one record, not the whole run.
- **Resumable** — on restart, read existing `(note_id, source_content_hash)` pairs from the JSONL and skip them. Addenda (same `note_id`, new content) are caught and re-extracted instead of silently skipped.
- **Inspectable** — `cat output.jsonl | jq .` to debug any record.
- **Promotable** — a separate ingest step flattens JSONL into wide Parquet columns (events / finish / run_meta preserved verbatim as JSON strings). This decouples extraction from schema evolution; relational reshaping happens at db-build time via per-batch SQL sidecars.

## Function-Calling over Chat Completions

Structured extraction uses LLM function-calling (tool use), not free-text parsing. Each extraction target is a Pydantic model that becomes an OpenAI-compatible tool schema. The LLM "calls" tools like `record_diagnosis(histologic_type="clear_cell", laterality="left", ...)`.

Why:
- **Constrained outputs** — enum fields restrict the LLM to valid clinical values (no "Clear Cell" vs "clear cell" vs "ccRCC" ambiguity).
- **Validated** — Pydantic validates every tool call before it's recorded. Bad data fails fast.
- **Multi-event** — the LLM can call multiple tools per note (e.g., record a diagnosis AND a treatment in one pass).
- **Composable** — definitions are just Pydantic models. Add a new extraction target by writing a new model and registering it.

## Single-Note Scope

This toolkit deliberately targets the **one-report-in / one-record-out** case (pathology reports, radiology reports, etc.) rather than longitudinal multi-note extraction. Single-note workflows have a simpler contract: every input has a stable `record_id`, the extraction is a pure function of the report text + the system prompt, and resumes/incrementals reduce to a hash anti-join. Multi-note workflows (where a "record" is a patient timeline assembled across dozens of encounters) need a different state model and are out of scope here.

## DuckDB for Everything

No Postgres, no data warehouse, no cloud services. One DuckDB file holds the entire queryable dataset. It reads directly from Parquet, supports full SQL, and runs on a laptop. Clinical data projects are typically small enough (thousands to low millions of rows) that this is more than sufficient, and the operational simplicity is worth it.

## Definitions as Configuration

Extraction behavior is defined by Pydantic models + system prompts, not application code. Adding a new extraction target means:

1. Define Pydantic models with enum-constrained fields
2. Write a system prompt with domain knowledge
3. Register tools in a `ToolRegistry`

No changes to the extraction engine, CLI, or pipeline code. The same engine handles pathology staging, treatment timelines, IHC results — the definitions are the only difference.
