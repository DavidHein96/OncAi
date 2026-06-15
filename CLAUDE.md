# CLAUDE.md

Orientation for Claude (and other LLM agents) working in this repo. If you're
reading this in a conversation, you're most likely here to **add a new
function-calling extraction workflow** — that's the primary contribution
shape. Everything else (lake, ingest, DuckDB plumbing) is the supporting cast.

## What this repo is

A pathology data lake + LLM extraction toolkit. Raw EHR CSVs are ingested into
a versioned parquet lake, materialised as a queryable DuckDB, and then run
through single-note function-calling extraction: one report in, one set of
structured findings out. Pydantic models constrain the LLM's outputs; events
are written to JSONL crash-safely and merged back into the lake.

See [README.md](README.md), [docs/architecture.md](docs/architecture.md), and
[docs/design.md](docs/design.md) for the full picture.

## The mental model of an FC workflow

A workflow is a *definition module* + the wiring to expose it as a CLI subcommand.
Each definition module exports three things:

- `DEFINITION_NAME` — used as the output subdirectory under `fc_outputs/`.
- `SYSTEM_PROMPT` — task instructions for the LLM.
- `create_<name>_registry()` — factory returning a `ToolRegistry(single_note=True)`
  with task-specific tools registered.

The tools are Pydantic models inheriting from `ExtractionEvent` (or
`ExtractionPlan` for orientation tools). Each field becomes part of the
JSON-schema sent to the LLM; enums constrain the LLM to valid values; Pydantic
validates every tool call before it's recorded.

The shipped example: `src/oncai/fc_extraction/definitions/example.py`.

## Adding a new FC workflow

Use the `/new-fc-workflow` slash skill (in `.claude/skills/new-fc-workflow/`).
It walks through copy → edit → register and verifies with `oncai fc list`.

Don't invent your own scaffolding flow when the user asks "add a new
extraction" / "new FC workflow" / "new definition" — invoke the skill.

## FC building blocks (where things live)

```
src/oncai/fc_extraction/
├── models.py         # ExtractionEvent, ExtractionPlan, ApproxDate,
│                     # FinishSingleExtraction — base classes for tools
├── tools.py          # ToolRegistry + Pydantic→OpenAI schema conversion
├── client.py         # FunctionCallingClient (Azure Responses / vLLM /
│                     # vLLM Chat). This is the LLM cut-point.
├── batch_single.py   # run_fc_single_batch — per-note batch runner with
│                     # resumable JSONL, hash-based skip, parallel workers
├── load.py           # batch segments → wide parquet (highest segment per
│                     # record_id wins; events_json/finish_json/run_meta_json
│                     # preserved verbatim as JSON strings)
├── manifest.py       # git/version/hash helpers + definition_hash
│                     # (prompt + tool schemas) for change detection
└── definitions/      # One file per workflow (example.py + path_kidney_*)
```

Registered in `src/oncai/cli/fc_cmds.py:_DEFINITIONS`. The dict maps a
short name (used as `oncai fc run-single <name>`) to
`(module_path, factory_name)`. New definitions MUST be added here or they
won't appear in `oncai fc list`.

## Supporting plumbing (skim)

- **Lake**: `ingest.py` + `transforms/collate.py` replay inbox files into
  versioned lake parquets (each row gets Blake2b `key_hash` / `content_hash`
  for incremental dedup). `lake.py` syncs the **inbox only** between local and
  remote (`pull_inbox_from_remote` / `push_inbox_to_remote`) — the lake is a
  disposable projection rebuilt from the inbox, never transferred.
- **DuckDB**: `db.py` rebuilds a queryable database from the lake. Schemas:
  `raw` (pathology), `cohort`, `extractions_raw` (one table per FC batch),
  `extractions_silver` (one table per reviewed batch from `fc_reviews` — the
  adjudicated, event-grain output), `extractions_gold` (dense per-concept tables
  a review batch's `<batch>.sql` sidecar reshapes from its silver table),
  `extractions_transformed` (user `.sql` sidecars), `scratch`
  (`oncai fc peek`), `runs` (run log).
- **Run log**: every `fc run-single` invocation writes an immutable manifest to
  `inbox/runs/<run_id>.run.json` via `runs.py` (`start_run` → `complete_run`,
  started → completed in place). `ingest runs` projects the manifests into
  `lake/runs/runs.parquet`; `runs list/show/compare` read the manifests direct.

## Conventions you MUST follow

These are non-obvious and were established intentionally. Don't drift from
them when editing or generating code.

- **Strict lint**: `ruff check src/oncai` and `ty check src/oncai` must pass.
  The configured rule set is broad (S, ANN, BLE, PTH, PERF, TRY, etc.); see
  `pyproject.toml` for ignores. New `# noqa: ...` comments need a *reason*.
- **Python 3.12+**: `requires-python = ">=3.12"`. Use `enum.StrEnum`, not
  `class Foo(str, Enum)`. Use `pathlib.Path`, not `os.path`.
- **DuckDB SQL safety**: identifiers (schema/table names) go through the
  `_q()` helper in `db.py` for double-quote escaping. Values use `?` bind
  parameters. The S608 lint warning is per-file-ignored in `db.py`,
  `cli/fc_cmds.py`, and `fc_extraction/batch_single.py` because of this
  pattern — see the pyproject.toml comments.
- **User-visible logging from ingest**: append to `FolderResult.notes` rather
  than printing directly. The CLI renders notes in yellow via
  `_print_ingest_results` — that's the "bright sugar" surface.
- **Subprocess**: resolve binaries via `shutil.which(...)` at import time and
  pass `check=False` explicitly. See `runs.py:_GIT_BIN` for the pattern.
- **Exception handling**: catching `Exception` is allowed (`BLE001` ignored)
  for optional integrations / parquet readers — `parquet_readable` is the
  one place we report failures, downstream checks skip silently.
- **No `# type: ignore` without a code**: use `# type: ignore[union-attr]`
  etc. ty respects mypy-style ignore comments.

## Banned terms

Private product names, institution-specific system names, and old internal
project labels **must not appear anywhere** in the codebase, docs, or tests.
If you see one during a search, it's a regression — remove it.

## Where the tests live

- `tests/unit/` — fast, isolated unit tests. Run constantly.
- `tests/e2e/test_pipeline_roundtrip.py` — end-to-end pipeline with a fake
  `FunctionCallingClient`. Read this to understand how the layers fit together
  and how to write integration tests that don't hit a real LLM.
- `tests/fixtures/sample_pathology.csv` — small synthetic kidney pathology
  CSV (multi-line reports, multi-specimen reports, IHC panels). Drop into
  `inbox/pathology/` as `YYYY-MM-DD_<label>.csv` for manual smoke tests.

## Commands

```bash
# Install
uv sync --extra dev

# Lint + type-check (both must be clean)
uv run ruff check src/oncai
uv run ty check src/oncai

# Tests
uv run pytest tests/unit -q             # fast unit suite
uv run pytest tests/e2e -q              # e2e with fake LLM
uv run pytest tests -q                  # all

# CLI smoke
uv run oncai --help
uv run oncai fc list                    # see registered definitions
```
