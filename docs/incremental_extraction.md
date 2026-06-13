# Incremental FC Extraction

`oncai fc run-single` is **incremental by default**. A *batch* is a folder of
immutable, numbered **segments**; each run appends the next segment, extracting
only the rows that are new or changed since the segments already there. You
never figure out which IDs to run — the run diffs the source against the batch's
own history.

```
inbox/fc_extractions/<batch>/
├── batch.json        # pins definition + source + id_col (identity guard)
├── 001.jsonl         # first run  (immutable, write-once)
├── 002.jsonl         # a later run's delta
└── 003.jsonl
```

For each `record_id`, the **highest-numbered segment that contains it wins** —
explicit integer order, no timestamps. A re-extraction in `003` supersedes the
same record's row in `001`; a record only in `001` stays.

## The flow

```
1. First run:   oncai fc run-single <defn> --batch foo --source raw.pathology
                → extracts all matching rows
                → inbox/fc_extractions/foo/001.jsonl   (promoted automatically)

   Ingest:      oncai ingest fc_extractions
                oncai build-db
                → lake/fc_extractions/foo.parquet  →  extractions_raw.foo

2. New / corrected source data lands.
                oncai ingest pathology && oncai build-db

3. Re-run:      oncai fc run-single <defn> --batch foo --source raw.pathology
                → diffs raw.pathology against foo's existing segments,
                  runs FC on the delta only
                → inbox/fc_extractions/foo/002.jsonl

   Ingest:      oncai ingest fc_extractions
                → merges foo/001.jsonl + foo/002.jsonl, highest segment per
                  record_id wins → lake/fc_extractions/foo.parquet (rewritten)
                oncai build-db → rebuilds extractions_raw.foo

4. Repeat. Next run → foo/003.jsonl. Etc.
```

The run writes its working JSONL into `fc_outputs/<Defn>/` (resumable scratch);
on success it is promoted into the batch folder as `NNN.jsonl` via an atomic
rename, so a crashed run never leaves a partial segment in the inbox. A crashed
run resumes its working file on the next invocation (same segment number, since
the inbox doesn't yet have it).

## The delta

Each source row is bucketed against the batch's history — read straight from the
inbox segments, so the delta is correct **without** a `build-db` first:

| Bucket | Meaning |
|---|---|
| **new** | `record_id` not in any segment yet |
| **changed** | id present, but no prior row's `source_content_hash` matches the current source `content_hash` (an addendum / edit) |
| **definition_changed** | content matches, but no matching row used the current `definition_hash` — the system prompt **or** a tool's Pydantic fields/enums changed. Only with `--reextract-on-prompt-change` |
| **forced** | id listed in `--force-rerun <csv>` that would otherwise skip |
| **skipped** | hash (and, if asked, definition) already match |

Only new + changed + definition_changed + forced are extracted into the next
segment. If the delta is empty, the run reports "up to date" and writes nothing.

The `definition_hash` (stamped on every record's `run_meta`) is a hash of the
system prompt **plus** every task tool's JSON schema — so re-running after you
add/rename a field or edit an enum re-extracts those records, which a
prompt-only hash would miss.

## Flags

- `--full` — re-extract every matching row into a new segment, ignoring what
  prior segments cover (e.g. after a definition change you want applied to all
  rows). Incompatible with the delta-refinement flags below.
- `--reextract-on-prompt-change` — also re-extract rows whose `definition_hash`
  differs: the system prompt **or** a tool's Pydantic fields/enums changed
  (`--source` runs only).
- `--force-rerun <csv>` — re-extract the listed ids even if unchanged
  (`--source` runs only).
- `--scratch` — **one-off test run.** Extracts to
  `fc_outputs/<Def>/<batch>.scratch.jsonl` and does **not** promote a segment
  into the inbox (no `batch.json`, no run log); runs fresh each time. Inspect it
  with `oncai fc peek`. Use this while iterating on a prompt or definition
  without writing anything to the canonical batch.

## Batch identity (`batch.json`)

The batch name is just a label. On the first run, `batch.json` pins the
`(definition, source, id_col)` the batch was created with; every later run
asserts they still match and refuses otherwise. This stops a typo'd `--source`
or a different definition from silently merging unrelated records into one
table. To extract a different source or definition, use a new `--batch` name.

## Notes

- `--jsonl` mode (notes from a file instead of a DuckDB table) always extracts
  everything in the file into a new segment — there's no SQL source to diff, so
  the delta flags don't apply.
- Within a single run, extraction also resumes per-note from its working file,
  so an interrupted run picks up where it left off.
- Failed records are dropped from the lake parquet (they're not in the
  success-only history, so a later run retries them); diagnostics live in the
  per-segment `NNN_manifest.json` and `oncai fc status`.
