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

`pull`/`push` treat `remote_path` as a plain filesystem path — there is no built-in SFTP/cloud client. For remote storage, mount it locally (sshfs, rclone, Box Drive) and point `remote_path` at the mount. See [design.md](design.md#remote-storage-is-just-a-filesystem-path) for why.

## Layers

### 1. Ingest layer

**Commands**: `pull`, `ingest`, `build-db`

Inbox CSVs are replayed into a versioned parquet lake. Each folder has a fixed ingest mode declared in `config.FOLDER_MODES`:

| Mode | Folder | What ingest does |
|---|---|---|
| DATED | `pathology` | Replay all `YYYY-MM-DD_*.csv` files in date order, rebuilding `lake/<folder>/<folder>.parquet` from scratch via key/content-hash merge. |
| STATIC | `fc_extractions` | Each inbox file maps to its own lake parquet (filename stem becomes the parquet/table name). |
| NAMED | `cohorts` | Filename = identity; one parquet per CSV. |
| LAKE_ONLY | `runs` | No inbox path; populated by `fc run-single` writes. |

Per-folder transforms live in `transforms/`:
- **Pathology collation** (`transforms/collate.py`) — multi-line CSV reports are reassembled into one row per `report_id`, then generic text cleaning is applied (Unicode normalization, whitespace and line-ending standardization). Two cleaning steps are opt-in and off by default, since they encode source-specific assumptions: site-specific boilerplate stripping (`PATHOLOGY_BOILERPLATE_PATTERNS`, ships empty) and decoding double-spaces back into line breaks (`DECODE_DOUBLE_SPACE_LINEBREAKS`). Reports that are already one-row-per-report (a `report_text` column and no `mult_ln_val_storage`) are auto-detected and passed through untouched — hashes only, no collation or cleaning. Each row gets `key_hash` (Blake2b of identity cols) and `content_hash` (Blake2b of content cols) so subsequent re-ingests are incremental.
- **Passthrough** (`transforms/passthrough.py`) — identity transform with schema validation; not currently used by shipped folders but available for callers that just need column-type enforcement.

`build-db` reads all lake parquets into a single DuckDB file organized by schema. Optionally per-folder `<batch>.sql` transforms run after the base load so each curated batch can declare its own derived tables.

### 2. Extraction layer

**Command**: `oncai fc run-single <definition>`

Single-note function-calling extraction. Each report is processed independently — the model receives one report at a time plus the registered tools, then calls them to record findings. Tools are Pydantic models, so every call is validated before being recorded.

The flow per note:
1. Load the row (DuckDB query against `--source` table, or JSONL line).
2. Send the report text + system prompt + tool schemas to the LLM.
3. The LLM calls tools (potentially many per note — one per finding).
4. Each tool call is Pydantic-validated. Failures trigger up to N validation retries with the error fed back to the model.
5. The LLM calls `finish_single_extraction` to terminate.
6. The full record is written to `fc_outputs/<DefinitionName>/<batch>.jsonl` immediately (crash-safe).

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
| `extractions_transformed` | per-batch `<batch>.sql` files | User-declared derived tables (typed events from `events_json`, etc.) |
| `extractions_staging` | `oncai fc stage` | Ad-hoc per-event flat layout for exploration |
| `runs` | `runs` | Run-log history (one row per `fc run-single` invocation) |

The wide layout in `extractions_raw.<batch>` keeps `events_json`, `finish_json`, `run_meta_json` as JSON strings — schema doesn't evolve with new event types. Relational reshaping into typed tables happens at `build-db` time via the `<batch>.sql` transform sidecars.

## Incremental updates

Both ingest and extraction are incremental:

- **Lake ingest** uses content-hash dedup. Re-ingesting an inbox file only writes the rows whose `(key_hash, content_hash)` pair isn't already in the lake parquet — so addenda (same `report_id`, different `report_text`) become new rows.
- **FC extraction** has `--incremental`. Given an existing baseline `extractions_raw.<batch>`, an incremental run anti-joins against the source table to extract only new rows + addenda + (optionally) rows whose `system_prompt_hash` differs. The output is a versioned `<batch>.v<N>.jsonl` that merges back into the same lake parquet at ingest time. See `incremental_extraction.md` for details.

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
