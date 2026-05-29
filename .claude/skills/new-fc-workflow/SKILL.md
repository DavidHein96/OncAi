---
name: new-fc-workflow
description: Scaffold a new single-note function-calling extraction workflow. Use when the user says "add a new fc workflow", "new extraction definition", "new FC definition", "add an extraction task", or otherwise asks to create a new oncai function-calling target. Copies the example definition module, helps the user fill in tools/prompt, and registers the new workflow so `oncai fc run-single <name>` works.
allowed-tools: Bash, Read, Write, Edit, Grep
---

# Add a new FC extraction workflow

A workflow = a Pydantic-models + system-prompt definition module + a one-line
registration in `cli/fc_cmds.py:_DEFINITIONS`. Background: see `CLAUDE.md`
and `src/oncai/fc_extraction/definitions/example.py`.

## Before you start

Ask the user for:

- **Short name** (snake_case, used in CLI: `oncai fc run-single <name>`).
  Examples in repo: `path_kidney_basic`, `path_kidney_ihc`.
- **DEFINITION_NAME** (display / output subdir name, CamelCase by convention).
  Examples in repo: `KidneyPathBasic`, `KidneyPathIHC`.
- **What the workflow extracts** in one or two sentences. Used to shape the
  system prompt and the tool models.

If any are unclear, ask before generating files.

## Steps

### 1. Copy the example module

```bash
cp src/oncai/fc_extraction/definitions/example.py \
   src/oncai/fc_extraction/definitions/<short_name>.py
```

### 2. Edit the new module

Open `src/oncai/fc_extraction/definitions/<short_name>.py` and change:

- `DEFINITION_NAME = "Example"` → `"<DefinitionName>"`
- The enums (e.g. `DiagnosisType`, `TreatmentIntent`) — replace with whatever
  categorical fields the workflow needs. Keep them as `StrEnum` (Python
  3.12+); the values become the LLM's allowed tokens.
- The tool models (`RecordDiagnosis`, `RecordTreatment`) — replace with the
  Pydantic models for *your* workflow. They MUST inherit from
  `ExtractionEvent` (which provides `note_id` and `comment` fields). Each
  field needs a `Field(..., description=...)` — the description is read by
  the LLM, so make it specific. Use `ApproxDate` for variable-precision
  dates. Use `enum_field: MyEnum | None = Field(None, ...)` for optional
  fields.
- `SYSTEM_PROMPT` — task-specific instructions. Keep the EXTRACTION RULES /
  DATE FORMAT / IMPORTANT blocks; replace the "extract" list with what your
  workflow records. Always include the line about calling
  `finish_single_extraction` when done.
- `create_example_registry()` → `create_<short_name>_registry()`. Inside,
  `registry.register(name=..., description=..., model=...)` once per tool.
  The `name` becomes the tool name the LLM calls; the `description` is what
  the LLM sees as the tool's purpose.

### 3. Register the workflow

Edit `src/oncai/cli/fc_cmds.py` and add an entry to `_DEFINITIONS`
(alphabetical order by short name):

```python
_DEFINITIONS: dict[str, tuple[str, str]] = {
    ...,
    "<short_name>": (
        "oncai.fc_extraction.definitions.<short_name>",
        "create_<short_name>_registry",
    ),
}
```

### 4. Verify

```bash
uv run oncai fc list                # new <short_name> should appear
uv run ruff check src/oncai         # must be clean
uv run ty check src/oncai           # must be clean
```

Tell the user how to dry-run their workflow once the lake has data:

```bash
oncai fc run-single <short_name> --batch v1 \
    --source raw.pathology --backend <their_backend> --limit 5
```

## What NOT to do

- Don't subclass `BaseModel` directly for tool models — inherit from
  `ExtractionEvent` (or `ExtractionPlan` for orientation tools). The base
  provides `note_id` / `comment` and the registry needs to recognise the
  type to route events vs plans.
- Don't add the workflow to `MULTI_TABLE_FOLDERS` or `SCHEMA_MAPPING` in
  `db.py` — FC workflows write into the existing `extractions_raw` schema
  via the standard `fc_extractions` ingest path. No db.py changes are
  needed for a new workflow.
- Don't invent new banned terms — see `CLAUDE.md` for the list
  (`beaker`, `copath`, `legacy ai`, `templater`).
