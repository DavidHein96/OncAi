# Incremental FC Extraction

The `--incremental` flag on `oncai fc run-single` lets you drop a new batch
of source documents in, ingest, and re-extract just the new and changed
rows — without manually figuring out which IDs to run on. Each incremental
run is pinned to a specific baseline batch and produces a versioned JSONL
that merges back into that batch's lake parquet at ingest time.

## The flow

```
1. Baseline:    oncai fc run-single <defn> --batch foo
                → fc_outputs/<Defn>/foo.jsonl

   Curate:      cp foo.jsonl + foo_manifest.json → inbox/fc_extractions/
                oncai ingest fc_extractions
                oncai build-db
                → lake/fc_extractions/foo.parquet
                → extractions_raw.foo

2. New source data lands.
                oncai ingest pathology
                oncai build-db

3. Incremental: oncai fc run-single <defn> --batch foo --incremental
                → reads extractions_raw.foo, anti-joins against current
                  raw.pathology, runs FC on the delta only
                → fc_outputs/<Defn>/foo.v2.jsonl  (auto-bumped from inbox)

   Curate:      cp foo.v2.jsonl + foo.v2_manifest.json → inbox/fc_extractions/
                oncai ingest fc_extractions
                → ingest groups foo.jsonl + foo.v2.jsonl, merges them
                  (latest extracted_at per record_id wins)
                → lake/fc_extractions/foo.parquet (rewritten in place)

   oncai build-db rebuilds extractions_raw.foo from the merged parquet.

4. Repeat. Next incremental → foo.v3.jsonl. Etc.
```

The user-facing batch name (`foo`) is stable. Versions are a transport
detail — they live in the JSONL filename and in each row's `batch_name`
column for provenance, but the lake parquet and DuckDB table are always
just `foo`.

## Why batch-pinned

Each `--incremental` run is tied to one `--batch`. There's no automatic
"latest gold across all batches" view. The reason: a single definition
gets run with different models, different prompts, different cohorts, and
those are peer batches the user wants to keep distinct for hand-curation.
Implicitly merging them would be silently destructive.

Multiple batches coexist as peers in `extractions_raw.*`. Hand-curated
gold tables — written as per-base `.sql` transforms or downstream
queries — are the user's call.

## How the delta is computed

The lake already does the hard part. Every pathology row carries a stable
`content_hash` from `transforms/collate.py`. That hash is the join key.

`_compute_incremental_delta` in `cli/fc_cmds.py` is a DuckDB anti-join:

1. Verify `extractions_raw.<batch>` exists. Error out otherwise —
   `--incremental` only makes sense as an extension of an existing baseline.
2. Pull `(record_id, source_content_hash, system_prompt_hash)` from
   `extractions_raw.<batch>` (success-only, all versions).
3. Query `(id_col, content_hash)` from the source table, filtered by
   `--where` / `--cohort` / `--id-file`.
4. Anti-join. Each source row falls into one of:

   - **new** — `record_id` not in the batch table.
   - **changed** — same `record_id`, different `content_hash` (addendum).
   - **prompt_changed** — same `(record_id, content_hash)`, different
     `system_prompt_hash` (only with `--reextract-on-prompt-change`).
   - **skipped** — exact match.

Pre-run preview makes the decision visible:

```
Incremental delta: 142 new + 7 changed (addenda) + 0 prompt-changed + 0 forced; skipping 9,318
  Output batch:  foo.v2.jsonl
```

### Forcing specific ids

`--force-rerun PATH` accepts a CSV (column auto-detected: `note_id`,
`report_id`, or `id`) of report ids that should be re-extracted even if
their hash matches the baseline. They land in a separate `forced` bucket
in the count.

When to reach for it:

- You spotted a bug in a few extractions during spot-checks and want to
  redo just those reports — not the whole batch, not everything that's
  technically "changed."
- You tweaked a small piece of the definition (a single tool's enum, a
  prompt clarification) and want to re-extract a hand-picked sample
  without bumping `system_prompt_hash` to invalidate everything.
- You're debugging an extraction failure mode and want to iterate on the
  same handful of reports without globbing.

Example:

```bash
echo "report_id\nP12345\nP67890\nP24680" > /tmp/redo.csv
oncai fc run-single path_kidney_nephrectomy \
    --batch foo --incremental \
    --source raw.pathology --backend vllm-local \
    --force-rerun /tmp/redo.csv
```

Pre-run preview (with three forced ids on top of an otherwise quiet delta):

```
Incremental delta: 0 new + 0 changed (addenda) + 0 prompt-changed + 3 forced; skipping 9,317
  Output batch:  foo.v2.jsonl
```

`--force-rerun` requires `--incremental` and composes with `--cohort` /
`--id-file`: those still scope the source query, so a forced id outside
the scope can't be extracted (no source row available). Those are
reported as `forced_missing` with a yellow warning so a typo or
mis-scoped run doesn't slip through silently:

```
Warning: 2 force-rerun id(s) not present in source query result
(check --cohort/--id-file/--where scope, or that the id exists in
raw.pathology): P99999, PXYZ
```

Forced ids that are already in `new` / `changed` / `prompt_changed` (i.e.
they would have been extracted anyway) are not double-counted — `forced`
only counts ids whose extraction was *added* by `--force-rerun`.

For this to work, every JSONL record carries a top-level
`source_content_hash` field (the hex `content_hash` of the source row at
extraction time), and `jsonl_to_wide_parquet` promotes it to a column on
the wide parquet. That's what `extractions_raw.<batch>` reads from.

## Versioned JSONLs

`--incremental` writes to `<batch>.v<N>.jsonl`. The version is auto-bumped
from inbox state — `_next_version_jsonl` scans
`inbox/fc_extractions/<batch>.v*.jsonl` for the highest `N` and adds one.
The baseline `<batch>.jsonl` is implicit v1.

The scan is deliberately on `inbox/`, not `fc_outputs/`. Rationale:
`inbox/` is permanent and synced to the team's shared store; `fc_outputs/`
is local-only and varies per machine. Different team members would
otherwise compute different next-version numbers for the same batch.

## Inbox-side merge at ingest

`_ingest_fc_extractions` groups inbox JSONLs by base batch name (strips
`\.v\d+$` from the stem) and merges each group into one lake parquet via
`merge_versioned_jsonls_to_parquet`:

- Concatenate all rows from all versions.
- Drop failures (same as before).
- Keep the latest `extracted_at` per `record_id`. Identical timestamps
  tie-break to higher `batch_name` lexicographically (`foo.v2` beats `foo`).
- Write `lake/fc_extractions/<batch>.parquet`.

Each surviving row's `batch_name` column tracks its **source JSONL stem**.
After a merge, `extractions_raw.foo` has rows with `batch_name IN ('foo',
'foo.v2', 'foo.v3', …)` so per-version provenance survives. Per-version
manifests stay separate in the lake folder under their own names.

The inbox is permanent (no archive step). Every `oncai ingest
fc_extractions` run sees the full inbox and reproduces lake parquets from
scratch — the inbox is the single source of truth.

## Hash-aware resume within a run

Resume inside an in-progress `<batch>.v<N>.jsonl` matches on `(note_id,
source_content_hash)` rather than `note_id` alone. So if the same batch
JSONL was partially completed and the source addended in between, the
addendum gets re-extracted instead of being silently skipped.

Records produced by pre-incremental code carry `source_content_hash=null`.
A `legacy_done_ids` fallback skips those by `note_id` alone, preserving
prior behavior.

## Cohort composability

Cohorts and `--incremental` compose because they operate on different
steps. `--cohort` restricts which source rows are considered (the SQL
filter in step 3 of the delta). The resume set comes from
`extractions_raw.<batch>` and is cohort-agnostic.

- **Cohort grows** → exactly the new rows are extracted.
- **Cohort shrinks** → removed rows aren't re-extracted; their old
  extractions stay in the batch table. JOIN to `cohort.<my_cohort>` at
  query time to filter to current members.
- **Different cohort, same batch** → source rows not yet in the batch
  table get extracted, regardless of which cohort was active before.
- **Source content changed since last run** → caught as `changed`;
  cohort membership is a row filter, not a content snapshot.

To force a full re-run within a cohort, drop `--incremental` and run
with just `--cohort` (today's behavior).

## Files

- `src/oncai/fc_extraction/batch_single.py` — `source_content_hash` is
  pulled in `_load_notes` and threaded through to JSONL records;
  `_get_existing_extraction_keys` keys resume by `(note_id, hash)`.
- `src/oncai/fc_extraction/load.py` — `jsonl_to_wide_parquet` promotes
  hashes to columns and stamps `batch_name` from source JSONL stem;
  `merge_versioned_jsonls_to_parquet` collapses versions into one parquet.
- `src/oncai/cli/fc_cmds.py` — `--incremental` flag,
  `_compute_incremental_delta` does the DuckDB anti-join,
  `_next_version_jsonl` auto-bumps the output filename.
- `src/oncai/ingest.py` — `_ingest_fc_extractions` groups inbox JSONLs by
  base batch name and calls the merge function once per group.

The lake side (`transforms/collate.py`, `lake.py`, `hashing.py`) is
unchanged — its existing `key_hash` / `content_hash` columns were already
incremental-friendly.

## Out of scope

- **Multi-note / patient-timeline workflows** — where identity is `mrn`-based
  and the "source content hash" would be a hash of the patient's whole
  timeline. The same anti-join idea applies, but the state model is
  different. This toolkit is single-note only (see `design.md`).
- **`note_id` → `record_id` rename.** The JSONL field is hardcoded to
  `note_id` even when `--id-col` is `report_id` (the value held is the
  `report_id`, but the JSON key is `note_id`). The wide parquet already
  uses `record_id`. Rename worth doing as its own focused change —
  touches JSONL schema, parquet schema, and downstream `.sql` transforms.
