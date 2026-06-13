# Architecture

A pathology data lake + LLM extraction toolkit. Raw EHR exports go in, queryable structured findings come out via single-note function-calling extraction.

## Data Flow

```
Raw CSVs ──→ Inbox ──→ Lake (Parquet) ──→ DuckDB
                          │                    ▲
                          │                    │
                          └── FC Extraction ───┘
                                  │
                                  └── JSONL ──→ ingest ──→ Lake
```

A user-facing batch goes through five steps: `pull` → `ingest` → `build-db` → `fc run-single` → `ingest fc_extractions` (to land the JSONL back in the lake) → `push` (to share with the team).

`pull`/`push` move **inbox files only**. The lake is a disposable projection rebuilt locally by `ingest` + `build-db`, so it is never transferred — which is what makes whole-file clobber between machines impossible. The inbox is canonical and append-only: raw drops plus pipeline outputs (extraction JSONLs, review logs, run manifests) all live there and ride the inbox push.

Because the inbox is append-only and the sync is additive, **removal** can't be a physical delete — it would be silently resurrected by the next `pull`. Forgetting data is instead an *appended* tombstone that propagates like any other inbox file; ingest reconciles the lake to `inbox − tombstones`. See [tombstones.md](tombstones.md) (design).

`pull`/`push` treat `remote_path` as a plain filesystem path — there is no built-in SFTP/cloud client. For remote storage, mount it locally (sshfs, rclone, Box Drive) and point `remote_path` at the mount. See [design.md](design.md#remote-storage-is-just-a-filesystem-path) for why.

## Layers

### 1. Ingest layer

**Commands**: `pull`, `ingest`, `build-db`

Inbox CSVs are replayed into a versioned parquet lake. Each folder has a fixed ingest mode declared in `config.FOLDER_MODES`:

| Mode | Folder | What ingest does |
|---|---|---|
| DATED | `pathology` | Replay all `YYYY-MM-DD_*.csv` files in date order, rebuilding `lake/<folder>/<folder>.parquet` from scratch via key/content-hash merge. |
| STATIC | `fc_extractions` | Per-batch segment folders (`<batch>/NNN.jsonl`) merge into one `extractions_raw.<batch>` parquet. Optional `<batch>/<batch>.sql` sidecars reshape raw outputs into `extractions_transformed`. |
| STATIC | `fc_reviews` | Per-segment `<batch>/<batch>.NNN.review_pkg.json` + `.reviews.jsonl` pairs merge (highest segment per note) into one `extractions_silver.<batch>` parquet. Optional `<batch>/<batch>.sql` sidecars reshape silver into `extractions_gold`. |
| NAMED | `cohorts` | Filename = identity; one parquet per CSV. |
| MANIFEST | `runs` | Each `fc run-single` writes an immutable `inbox/runs/<run_id>.run.json` (started → completed in place); ingest unions the manifests into `lake/runs/runs.parquet`. |

Per-folder transforms live in `transforms/`:
- **Pathology collation** (`transforms/collate.py`) — multi-line CSV reports are reassembled into one row per `report_id`, then generic text cleaning is applied (Unicode normalization, whitespace and line-ending standardization). Two cleaning steps are opt-in and off by default, since they encode source-specific assumptions: site-specific boilerplate stripping (`PATHOLOGY_BOILERPLATE_PATTERNS`, ships empty) and decoding double-spaces back into line breaks (`DECODE_DOUBLE_SPACE_LINEBREAKS`). Reports that are already one-row-per-report (a `report_text` column and no `mult_ln_val_storage`) are auto-detected and passed through untouched — hashes only, no collation or cleaning. Each row gets `key_hash` (Blake2b of identity cols) and `content_hash` (Blake2b of content cols) so subsequent re-ingests are incremental.
- **Passthrough** (`transforms/passthrough.py`) — identity transform with schema validation; not currently used by shipped folders but available for callers that just need column-type enforcement.

`build-db` reads all lake parquets into a single DuckDB file organized by schema. Optionally per-batch SQL transforms run after the base load so each curated batch can declare its own derived tables.

### 2. Extraction layer

**Command**: `oncai fc run-single <definition>`

Single-note function-calling extraction. Each report is processed independently — the model receives one report at a time plus the registered tools, then calls them to record findings. Tools are Pydantic models, so every call is validated before being recorded.

The flow per note:
1. Load the row (DuckDB query against `--source` table, or JSONL line).
2. Send the report text + system prompt + tool schemas to the LLM.
3. The LLM calls tools (potentially many per note — one per finding).
4. Each tool call is Pydantic-validated. Failures trigger up to N validation retries with the error fed back to the model.
5. The LLM calls `finish_single_extraction` to terminate.
6. Each record is written to a resumable working JSONL in `fc_outputs/<DefinitionName>/` (crash-safe). On completion the file is promoted into the batch folder as `inbox/fc_extractions/<batch>/NNN.jsonl` — the next immutable segment.

A definition module is a Pydantic-models-plus-system-prompt unit. It exports:
- `DEFINITION_NAME` — used as the output subdirectory and JSONL prefix.
- `SYSTEM_PROMPT` — instructions for the LLM.
- `create_<name>_registry()` — factory returning a `ToolRegistry(single_note=True)` with the task-specific tools registered.

Adding a new extraction target means writing a new definition module + registering it in `cli/fc_cmds.py`'s `_DEFINITIONS` dict. No changes to the extraction engine.

### 3. Query layer

**Command**: direct DuckDB SQL.

The DuckDB is rebuilt from the lake on demand. Schemas:

| Schema | Source | Contents |
|---|---|---|
| `raw` | `pathology` | Per-report pathology after collation |
| `cohort` | `cohorts` | One table per named cohort + `cohort.meta` registry |
| `extractions_raw` | `fc_extractions` | One table per FC batch, wide row-per-note layout |
| `extractions_silver` | `fc_reviews` | One table per completed review batch, approved events with edits applied (event-grain, sparse) |
| `extractions_gold` | `fc_reviews/<batch>/<batch>.sql` sidecars | Dense per-concept tables reshaped from a batch's silver table |
| `extractions_transformed` | `fc_extractions/<batch>/<batch>.sql` sidecars | User-declared derived tables (typed events from `events_json`, etc.) |
| `scratch` | `oncai fc peek` | Throwaway per-event layout for a quick look (cleared on rebuild) |
| `runs` | `runs` | Run-log history (one row per `fc run-single` invocation) |

The wide layout in `extractions_raw.<batch>` keeps `events_json`, `finish_json`, `run_meta_json` as JSON strings — schema doesn't evolve with new event types. Relational reshaping into typed tables happens at `build-db` time via batch-local transform sidecars.

Review packages are intentionally selective. A normal run auto-builds a package
for reports that called `flag_report_for_review`; `oncai fc review-package` can
also build queues from cross-run disagreement (`--scope disagreements
--compare-with <other.jsonl>`). Disagreement comparison ignores plain free-text
strings and only compares schema-backed categorical strings, numbers, booleans,
and structured date fields when the definition is available. See
[review_system.md](review_system.md) for the full review package, review app,
silver-table ingest, and gold reshape workflow.

## Incremental updates

Both ingest and extraction are incremental:

- **Lake ingest** uses content-hash dedup. Re-ingesting an inbox file only writes the rows whose `(key_hash, content_hash)` pair isn't already in the lake parquet — so addenda (same `report_id`, different `report_text`) become new rows.
- **FC extraction** is incremental by default. A batch is a folder of numbered segments (`inbox/fc_extractions/<batch>/NNN.jsonl`); re-running diffs the source against the batch's existing segments (read from the inbox, not the built DB) and extracts only new rows + addenda + (optionally) prompt-changed rows into the next segment. At ingest the segments merge into one lake parquet, **highest segment per `record_id` wins** — explicit integer order, no timestamps. `--full` ignores prior segments. See `incremental_extraction.md` for details.

## Package Structure

```
src/oncai/
├── cli/                    # Typer CLI
│   ├── main_cmds.py        # init, pull, push, ingest, build-db, status, schemas, version
│   ├── fc_cmds.py          # fc run-single, list, status, stage, unstage, manifest
│   ├── cohort_cmds.py      # cohort add, list, info, remove
│   ├── runs_cmds.py        # runs list, show, compare
│   └── db_cmds.py          # db update
├── fc_extraction/          # Single-note FC extraction engine
│   ├── client.py           #   LLM client (Azure Responses / vLLM Responses / vLLM Chat)
│   ├── batch_single.py     #   Per-note batch runner (resumable, parallel)
│   ├── tools.py            #   Pydantic → OpenAI tool schema + validation
│   ├── load.py             #   JSONL → wide lake parquet
│   ├── manifest.py         #   Git / version / hash helpers
│   ├── models.py           #   ExtractionEvent, ApproxDate, FinishExtraction, ...
│   └── definitions/        #   Shipped definitions (path_kidney_*, example)
├── review/                 # Review package generation, selection, and silver load
├── transforms/             # Ingest transforms
│   ├── collate.py          #   Multi-line pathology collation + text cleaning
│   └── passthrough.py      #   Identity transform (schema validation)
├── schemas/                # Dataset column definitions
│   └── pathology.py
├── config.py               # OncaiConfig (oncai.yaml) + FOLDER_MODES
├── db.py                   # Lake → DuckDB builder
├── lake.py                 # Remote ↔ local sync, content-hash merge
├── ingest.py               # Inbox → lake replay pipeline
├── cohort.py               # Cohort management
├── runs.py                 # Run logging
├── hashing.py              # Blake2b content hashing
├── sidecar.py              # SHA-256 inbox sidecars
└── lake_check.py           # Lake data validation
```
