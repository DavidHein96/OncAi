# Design Philosophy

Key decisions and why they were made.

## Parquet over CSV

CSV is the lingua franca of clinical data exports, but it's a terrible storage format — no types, no compression, slow to scan. Parquet gives us columnar compression (zstd), embedded schemas, and native DuckDB/Polars support. CSVs enter through the inbox and never persist beyond the ingest step.

## Content Hashing for Incremental Updates

Every row gets two hashes:

- **key_hash** (Blake2b-128) — computed from identity columns (e.g., `report_id`). Determines if a row already exists.
- **content_hash** — computed from value columns (e.g., `report_text`). Determines if an existing row has changed.

This makes re-ingestion safe and cheap — only genuinely new or changed rows are written. No timestamps, no "last modified" tracking, no coordination with the source system. The inbox is the permanent source of truth: every ingest replays it from scratch, and because the content-hash merge turns unchanged rows into no-ops, re-running is idempotent rather than redundant — there's no archive step to manage.

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

## Remote Storage Is Just a Filesystem Path

`remote_path` is a plain directory. `pull`/`push` (`lake.py`) move data with ordinary filesystem operations — `shutil.copy2`, `Path.glob`, `Path.exists()`. There is **no built-in SFTP, S3, or cloud client**, and that is a deliberate decision rather than a missing feature.

We considered adding a native SFTP transport (the config once carried `remote_type` / `sftp_*` stubs) and chose not to, for several reasons:

- **The OS already solved it.** `sshfs`, `rclone mount`, Box Drive, and SMB/NFS all expose remote storage as a local directory. Pointing `remote_path` at such a mount gives you SFTP/S3/cloud sync with **zero** transport code in oncai — the existing `copy2`/`glob` path just works.
- **Even the "SFTP is the only sanctioned transport" case is covered.** Hospital/clinical environments that mandate SFTP can use `rclone mount sftp:…` (userspace, cross-platform, no admin) and still point oncai at the mount. Mounting locks no one out.
- **Native SFTP is a subsystem, not a flag.** Doing it properly means abstracting *every* path operation (exists, glob, copy, sidecar read/write, atomic writes) behind a virtual-filesystem interface, then taking on a heavyweight SSH dependency, host-key verification, auth, retries, and partial-transfer handling. That is a large, security-sensitive surface to maintain in a project whose value is the data model, not the transport.
- **Dead config is a trap.** A `remote_type: sftp` switch that silently falls back to local behavior generates "I configured SFTP and nothing happened" confusion. Better to have no switch than a switch that lies.

The filesystem-path boundary is the right extension point: if a real requirement ever demands in-process SFTP, it can be added behind a transport interface at that point, driven by a concrete use case. Building it speculatively buys nothing the OS doesn't already provide.

## Definitions as Configuration

Extraction behavior is defined by Pydantic models + system prompts, not application code. Adding a new extraction target means:

1. Define Pydantic models with enum-constrained fields
2. Write a system prompt with domain knowledge
3. Register tools in a `ToolRegistry`

No changes to the extraction engine, CLI, or pipeline code. The same engine handles pathology staging, treatment timelines, IHC results — the definitions are the only difference.
