# OncAI — Oncology Data Lake + Single-Note FC Extraction

A small data-lake + LLM-extraction toolkit for pathology reports. Ingests raw CSVs into a versioned parquet lake, builds a queryable DuckDB, and runs **single-note function-calling extraction** against each report — the model calls Pydantic-validated tools to record structured findings.

Designed for one-report-in / one-set-of-findings-out workloads (e.g. nephrectomy pathology reports → tumor type + stage + margins + IHC results).

> See [docs/architecture.md](docs/architecture.md) for how the system works, [docs/design.md](docs/design.md) for why, and [docs/incremental_extraction.md](docs/incremental_extraction.md) for incremental / addenda re-extraction.

## Quickstart

```bash
# Install
uv sync

# Initialize folder structure + write oncai.yaml
oncai init --remote /path/to/shared/data

# Pull data from remote (parquet lake + inbox files)
oncai pull

# Preview an ingest
oncai ingest --dry-run

# Ingest inbox CSVs into the lake
oncai ingest

# Build DuckDB from the lake parquets
oncai build-db

# Run a single-note FC extraction over the pathology table
oncai fc run-single path_kidney_basic \
    --batch v1 \
    --source raw.pathology \
    --backend gpt5mini \
    --limit 20

# Inspect / iterate the resulting JSONL
oncai fc stage fc_outputs/PathKidneyBasic/v1.jsonl
oncai fc status

# Push results back to remote
oncai push fc_extractions
```

## Architecture

```
Remote (Box / shared drive)    Lake (local parquet)         DuckDB (queryable)
┌──────────────────┐  pull  ┌─────────────────────┐  build  ┌─────────────────┐
│ pathology/       │ ─────► │ lake/pathology/     │ ──────► │ raw.pathology   │
│ cohorts/         │        │ lake/cohorts/       │ ──────► │ cohort.<name>   │
│ fc_extractions/  │        │ lake/fc_extractions/│ ──────► │ extractions_raw.│
│ runs/            │        │ lake/runs/          │         │   <batch>       │
└──────────────────┘        └─────────────────────┘         │ runs.runs       │
                                     ▲                       └─────────────────┘
                            ingest   │
                            ┌────────┴────────────┐
                            │ inbox/<folder>/     │
                            └─────────────────────┘
```

**Data flow:** Remote → `pull` → Lake (parquet) → `build-db` → DuckDB → `fc run-single` → JSONL → `ingest fc_extractions` → Lake → `push` → Remote.

**Why a lake + DuckDB:** the parquet lake is the source of truth (versioned, content-hashed merges). DuckDB is rebuilt from the lake on demand — no migrations, no schema drift between extractions and source data.

## CLI Reference

### Core Commands

| Command | Description |
|---|---|
| `oncai init` | Initialize folder structure and write `oncai.yaml` |
| `oncai pull [FOLDER...]` | Pull parquet lake + inbox files from remote |
| `oncai push [FOLDER...]` | Push local lake + inbox files to remote |
| `oncai ingest [DATASET]` | Replay inbox files into lake parquets |
| `oncai build-db` | Build DuckDB from lake parquets |
| `oncai status` | Show inbox / lake / database status (`--check` for sidecar + health checks) |
| `oncai schemas` | List registered dataset schemas |
| `oncai version` | Show version |

Options: `--dry-run` previews `pull` / `push` / `ingest` without writing. `--force` recreates the DuckDB.

### Database Management (`oncai db`)

| Command | Description |
|---|---|
| `oncai db update <folder>` | Refresh one folder's tables in the DuckDB without a full rebuild |

### Function-Calling Extraction (`oncai fc`)

Single-note extraction: the model receives one report at a time + the registered tools, then calls them to record findings (each tool call is Pydantic-validated). Results land as one JSON record per note in a batch JSONL.

| Command | Description |
|---|---|
| `oncai fc list` | List the shipped single-note definitions |
| `oncai fc run-single <definition>` | Run a definition over a source table or JSONL |
| `oncai fc status [PATH]` | Tally success/failure across JSONL outputs |
| `oncai fc stage <jsonl>` | Stage a JSONL into `extractions_staging.<stem>` for ad-hoc SQL |
| `oncai fc unstage <stem>` | Drop a staging table (or `--all`) |
| `oncai fc manifest <path>` | Show the run manifest for a batch |

Shipped definitions:

| Name | What it extracts |
|---|---|
| `example` | Template — diagnosis + treatment events. Copy + modify to build your own. |
| `path_kidney_basic` | Triage routing pass — coarse cancer status + which downstream workflows apply |
| `path_kidney_nephrectomy` | Primary nephrectomy pathology — type, histology, grade, stage, margins |
| `path_kidney_ihc` | Immunohistochemistry markers from path reports |
| `path_kidney_proc_site_hist` | Per-specimen procedure / site / histology breakdown |

Key options for `fc run-single`:

```
--batch <name>             Batch name (becomes the extractions_raw table name)
--source <table>           DuckDB source table (e.g. raw.pathology)
--jsonl <file>             Load notes from a JSONL instead of DuckDB (mutex with --source)
--backend <name>           Named LLM backend from oncai.yaml
--text-col <col>           Column / JSON key with the note text (default report_text)
--id-col <col>             Column / JSON key with the row id (default report_id)
--limit <n>                Max notes to process
--where <sql>              SQL WHERE filter (DuckDB mode only)
--cohort <name>            Filter to ids in a named cohort
--id-file <csv>            Filter to ids from a CSV
--workers <n>              Concurrent workers (default 1)
--incremental              Only run rows whose source content_hash differs from
                           the existing extractions_raw.<batch> baseline. Output
                           bumps to <batch>.v<N>.jsonl.
--retry-failed             Drop previously-failed records and rerun just those
--reasoning-effort <lvl>   Azure reasoning depth: low / medium / high / none
```

`--incremental` re-extracts only new and changed reports (addenda) against a baseline batch, writing a versioned `<batch>.v<N>.jsonl` that merges back into the same lake parquet. See [docs/incremental_extraction.md](docs/incremental_extraction.md) for the full workflow, versioned batches, and `--force-rerun`.

### Cohorts (`oncai cohort`)

Named lists of report_ids or mrns used to scope FC runs.

| Command | Description |
|---|---|
| `oncai cohort add <csv> --name <name>` | Add a CSV as a named cohort (auto-detects key column) |
| `oncai cohort list` | List all cohorts |
| `oncai cohort info <name>` | Show cohort details |
| `oncai cohort remove <name>` | Remove a cohort |

CSVs can also be dropped into `inbox/cohorts/` and ingested via `oncai ingest cohorts` — the cohort name comes from the filename. Either way the cohort lands in `lake/cohorts/<name>.parquet` and the DuckDB `cohort.<name>` table.

### Run History (`oncai runs`)

Every `fc run-single` invocation logs metadata to `lake/runs/runs.parquet` (and the DuckDB `runs.runs` table).

| Command | Description |
|---|---|
| `oncai runs list [--type <t>]` | List recent runs |
| `oncai runs show <id>` | Show full details for one run |
| `oncai runs compare <id1> <id2>` | Side-by-side diff of two runs |

## Ingesting data

Data enters through `inbox/<folder>/` and is replayed into the parquet lake by `oncai ingest`. Preview first with `oncai ingest --dry-run`; ingest reports exactly what it did (column renames, inferred fields, collation vs passthrough) so nothing happens silently. Re-running is idempotent — the content-hash merge writes only new or changed rows.

### Pathology reports

Drop dated CSVs into `inbox/pathology/` named `YYYY-MM-DD_<label>.csv` (the date orders replay; the label is free text). Ingest accepts two shapes and auto-detects which you have.

**1. Raw multi-line export** — one row per report *line*, collated into one row per report:

| Column | Required | Notes |
|---|---|---|
| `report_id` | yes | Report identity (also accepts Epic `case_num`). |
| `mult_ln_val_storage` | yes | Report text, one fragment per row. |
| `row_id` | no | Line order within a report; inferred from CSV order if absent. |
| `mrn` | no | Patient identifier (also accepts Epic `pat_mrn_id`). |
| `ordering_date` | no | Normalized to `YYYY-MM-DD`. |
| `external_name` | no | Section tag; `gross` / `micro` / `addlinfo` rows are dropped. |

Fragments are joined with no delimiter (they may be cut mid-word), then generic text cleaning is applied (Unicode / whitespace / line-ending normalization).

**2. Already-clean reports** — one row per report, text ready to go:

| Column | Required | Notes |
|---|---|---|
| `report_id` | yes | Report identity. |
| `report_text` | yes | Final report text, used verbatim. |
| `mrn`, `ordering_date` | no | Carried through if present. |

If `report_text` is present and there's no `mult_ln_val_storage`, ingest treats the file as already clean and **passes it through untouched** — no collation, no text cleaning — adding only the content hashes.

#### Cleaning knobs (pathology)

The multi-line path always applies generic cleaning. Two source-specific steps are **off by default** so one site's format isn't forced on everyone; enable them in `src/oncai/transforms/collate.py`:

- `PATHOLOGY_BOILERPLATE_PATTERNS` — regexes stripped from report text (ships empty). Add your institution's attestation / disclaimer boilerplate here.
- `DECODE_DOUBLE_SPACE_LINEBREAKS` — set `True` if your source encodes line breaks as double spaces, to decode them back into newlines.

### Other folders

- `inbox/cohorts/<name>.csv` → `oncai ingest cohorts` (one parquet per file; the filename is the cohort name).
- `inbox/fc_extractions/<batch>.jsonl` → `oncai ingest fc_extractions` (extraction outputs merged back into the lake).

### Skip the lake for ready-made notes

If you already have clean notes in a JSONL and only want extraction, you don't need to ingest at all — point `fc run-single` straight at the file:

```bash
oncai fc run-single path_kidney_basic --jsonl notes.jsonl --batch v1 --backend gpt5mini
```

`--jsonl` reads notes directly (no DuckDB, no lake); `--text-col` / `--id-col` locate the note text and id (default `report_text` / `report_id`).

## DuckDB Schemas

| Schema | Source | Contents |
|---|---|---|
| `raw` | `pathology` | Pathology reports (collated multi-line CSVs → one row per report) |
| `cohort` | `cohorts` | One table per named cohort + `cohort.meta` |
| `extractions_raw` | `fc_extractions` | One table per FC batch, wide row-per-note layout |
| `extractions_transformed` | (per-batch `.sql` files) | Materialized derived tables from `<batch>.sql` |
| `extractions_staging` | `oncai fc stage` | Ad-hoc per-event flat layout for exploration |
| `runs` | `runs` | Run-log history (one row per `fc run-single`) |

Per-batch SQL transforms: dropping `<batch>.sql` next to `lake/fc_extractions/<batch>.parquet` runs at `build-db` time. Use it to reshape `events_json` columns into typed relational tables for downstream queries.

## Adding a New Definition

Copy `src/oncai/fc_extraction/definitions/example.py`, rename, edit the tool models + system prompt. Then register it in `src/oncai/cli/fc_cmds.py` by adding to the `_DEFINITIONS` dict:

```python
_DEFINITIONS: dict[str, tuple[str, str]] = {
    "example": ("oncai.fc_extraction.definitions.example", "create_example_registry"),
    "my_definition": (
        "oncai.fc_extraction.definitions.my_definition",
        "create_my_definition_registry",
    ),
}
```

A definition module exposes three things: `DEFINITION_NAME` (used as the output subdir), `SYSTEM_PROMPT`, and a `create_<name>_registry()` factory returning a `ToolRegistry(single_note=True)` with your tools registered.

## Package Structure

```
src/oncai/
├── cli/                        # Typer CLI
│   ├── main_cmds.py            # init, pull, push, ingest, build-db, status, schemas, version
│   ├── fc_cmds.py              # fc run-single, list, status, stage, unstage, manifest
│   ├── cohort_cmds.py          # cohort add, list, info, remove
│   ├── runs_cmds.py            # runs list, show, compare
│   ├── db_cmds.py              # db update
│   └── _shared.py              # console, get_config, MRN/ID/cohort filter loaders
├── fc_extraction/              # Single-note function-calling extraction
│   ├── client.py               # LLM client (Azure OpenAI Responses / vLLM)
│   ├── batch_single.py         # Per-note batch runner (resumable, parallel)
│   ├── tools.py                # Pydantic → OpenAI tool schema + validation
│   ├── load.py                 # JSONL → wide lake parquet
│   ├── manifest.py             # Git / version / hash helpers
│   ├── models.py               # ExtractionEvent, ExtractionPlan, ApproxDate, ...
│   └── definitions/            # Shipped extraction definitions
│       ├── example.py
│       ├── path_kidney_basic.py
│       ├── path_kidney_nephrectomy.py
│       ├── path_kidney_ihc.py
│       └── path_kidney_proc_site_hist.py
├── transforms/                 # Ingestion transforms
│   ├── collate.py              # Multi-line pathology collation + text cleaning
│   └── passthrough.py          # Identity transform (schema validation only)
├── schemas/                    # Dataset column definitions
│   └── pathology.py
├── config.py                   # OncaiConfig (oncai.yaml)
├── db.py                       # Lake → DuckDB builder
├── lake.py                     # Remote ↔ local sync + merge with content-hash dedup
├── ingest.py                   # Inbox → lake replay pipeline
├── cohort.py                   # Cohort management
├── runs.py                     # Run logging
├── hashing.py                  # Blake2b content hashing
├── sidecar.py                  # SHA-256 sidecar files for inbox provenance
└── lake_check.py               # Lake data validation
```

## Development

```bash
# Install with dev dependencies
uv sync --extra dev

# Tests
uv run python -m pytest tests/ -v

# Lint
uv run ruff check src/

# Type check
uv run ty check src/
```

## Configuration

`oncai init` writes a default `oncai.yaml`:

```yaml
remote_path: /path/to/shared/data    # Box mount, network drive, or local test folder
lake_path: oncai_data/lake
inbox_path: oncai_data/inbox
db_path: oncai_data/oncai.duckdb
remote_type: local                    # or sftp

llm_backends:
  gpt5mini:
    type: azure-responses
    endpoint: https://your-endpoint.openai.azure.com/
    deployment: gpt-5-mini
    api_key_env: AZURE_OPENAI_API_KEY
  vllm-local:
    type: vllm
    base_url: http://localhost:8000/v1
    model: meta-llama/Llama-3-70b
    api_key_env: VLLM_API_KEY
```

LLM backends are named configurations referenced by `--backend <name>`. API keys are read from environment variables (set in `.env` or your shell).

## License

OncAI is licensed under the **GNU Affero General Public License v3.0** — see [LICENSE](LICENSE) for the full text. Under the AGPL's network-use clause (section 13), running a modified version as a network service obligates you to offer those users the corresponding source.
