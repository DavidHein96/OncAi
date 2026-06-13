# Review System

The review system closes the loop from function-calling extraction to an
adjudicated **silver** table in DuckDB — and, optionally, to dense **gold**
tables reshaped from it with a per-batch SQL sidecar.

It has four artifacts:

1. An extraction batch JSONL from `oncai fc run-single`.
2. A frozen review package, `<batch>.review_pkg.json`.
3. An append-only reviewer verdict log, `<batch>.reviews.jsonl`.
4. A reviewed (silver) parquet, `lake/fc_reviews/<batch>.parquet`, loaded into
   DuckDB as `extractions_silver.<batch>`.

The package is the handoff artifact for a reviewer. The review log is the
completed adjudication. The silver parquet is derived from the package plus the
review log; it is never edited by hand.

Why "silver" and not "gold"? The reviewed table is trustworthy but still
*event-grain and sparse* — a definition's distinct event tools share one wide
table, so each row is null across every column owned by a different tool. That's
a classic silver layer: validated, not yet shaped for consumption. Projecting it
into dense, per-concept **gold** tables is a separate, optional SQL step
(["Gold Reshape"](#gold-reshape)).

## End-to-End Workflow

Run an extraction. By default, `run-single` builds a review package containing
only reports that called `flag_report_for_review`:

```bash
oncai fc run-single path_kidney_basic \
    --batch v1 \
    --source raw.pathology \
    --backend gpt5mini
```

The extraction JSONL streams into `fc_outputs/`, but the review package is
written straight into the review inbox, under a batch-named folder the run
creates idempotently (mirroring `inbox/fc_extractions/<batch>/`):

```text
fc_outputs/PathKidneyBasic/v1.001.jsonl          # the raw extraction segment
inbox/fc_reviews/v1/v1.001.review_pkg.json       # the handoff package
```

Open the package in the review app:

```bash
python apps/review_app/server.py \
    --package inbox/fc_reviews/v1/v1.001.review_pkg.json \
    --reviewer "Reviewer Name"
```

When launched with `--package`, the review app writes the verdict log beside
the package by default — i.e. back into the same inbox batch folder:

```text
inbox/fc_reviews/v1/v1.001.reviews.jsonl
```

Because the package already lives in the inbox, there is no copy step. Once the
completed `*.reviews.jsonl` is dropped beside it, ingest and rebuild:

```bash
oncai ingest fc_reviews
oncai build-db
```

The reviewed rows are then queryable as:

```sql
SELECT *
FROM extractions_silver.v1;
```

For a smaller database refresh after ingesting reviews, use:

```bash
oncai db update fc_reviews
```

## What Gets Reviewed

Review packages contain extraction `events`, not `plans`. `ExtractionPlan`
tools are orientation/control-flow outputs and are intentionally excluded from
review. There is no legacy fallback to `plans.flag_report_for_review`; review
selection only looks at `events.flag_report_for_review`.

Every available FC workflow must expose a `flag_report_for_review` event. Use
the reserved `review_anchor: list[str]` field for exact report text snippets
that explain why the report was flagged.

Source-note jump targets are explicit provenance metadata, not inferred from
ordinary field types:

- `evidence`: event-level snippets shown as extraction evidence.
- `review_anchor`: event-level snippets shown as review anchors on flagged reports.

Both forms are highlighted in the source note and clickable when the quote can
be found. They are not editable fields and do not become silver-table values.

The default package scope is flagged reports only:

```bash
oncai fc run-single path_kidney_basic \
    --batch v1 \
    --source raw.pathology \
    --backend gpt5mini \
    --review-package flagged
```

Other run-time scopes:

```bash
--review-package all      # package every successful report
--review-package none     # do not build a package after the run
```

## Building Packages After a Run

Use `oncai fc review-package` when the JSONL already exists or when you want a
different review scope:

```bash
oncai fc review-package fc_outputs/PathKidneyBasic/v1.jsonl --scope all
oncai fc review-package fc_outputs/PathKidneyBasic/v1.jsonl --scope flagged
```

The command reads the batch manifest sidecar when available, reloads source
notes from DuckDB or the original JSONL source, and writes
`<batch>.review_pkg.json` beside the JSONL unless `--output` is supplied.

If the originating definition is known, pass it explicitly to get richer review
controls:

```bash
oncai fc review-package fc_outputs/PathKidneyBasic/v1.jsonl \
    --definition path_kidney_basic
```

With a definition, the field schema includes enum options, required fields,
number controls, boolean controls, approximate-date controls, and evidence
snippet fields. Without a definition, the package builder infers a minimal
schema from the observed event values.

## Disagreement Review

A common pattern is to run the same report set multiple times, possibly with
different models or prompts, and only send disagreements to review.

```bash
oncai fc review-package run_a.jsonl \
    --scope disagreements \
    --compare-with run_b.jsonl

oncai fc review-package run_a.jsonl \
    --scope flagged-or-disagreements \
    --compare-with run_b.jsonl \
    --compare-with run_c.jsonl
```

Disagreement selection compares structured event outputs by `note_id`. Missing
records in a comparison run count as disagreements. Event order does not
matter.

Plain `str` fields are treated as free text and ignored by default. The fields
that participate are:

- schema-backed categorical strings such as enums/literals,
- floats and integers,
- booleans,
- structured approximate-date fields.

`comment` is ignored by default. Add more ignored fields with:

```bash
oncai fc review-package run_a.jsonl \
    --scope disagreements \
    --compare-with run_b.jsonl \
    --agreement-ignore-field rationale
```

### The adjudication hash — are two runs even comparable?

Every package carries an `adjudication_hash`: a hash of the *comparability
contract* the two runs must share for a field-by-field diff to mean anything. It
**includes** `definition_name`, the event/tool names, each field's name and
control/type, enum option sets, the per-event identity-key and comparison config
(`event_identity_fields` + `comparison_fields`), and the normalization-rules
version. It deliberately
**excludes** everything that varies run-to-run without affecting comparability:
the system prompt, model/backend, sampling params (temperature / top_p / top_k),
reasoning effort, git commit, run date, and batch name.

So two runs of the same definition with **different models or prompts** share an
`adjudication_hash` and can be compared; a run whose definition changed a field,
enum, or identity key gets a **different** hash and must not be diffed against
the old one. This is distinct from `definition_hash` (which *includes* the prompt
and drives re-extraction) — `adjudication_hash` is the schema-only contract for
cross-run review. It's computed by `oncai.review.adjudication_hash`.

## Review Package Shape

A review package is self-contained JSON. It carries the rendering schema, the
source note text, and the events to adjudicate:

```json
{
  "definition_name": "PathKidneyBasic",
  "batch": "v1",
  "generated_at": "2026-06-12T12:00:00+00:00",
  "adjudication_hash": "8e981e992c03dd74",
  "field_schema": {
    "triage_report": {
      "label": "Triage Report",
      "fields": [
        {
          "name": "has_kidney_cancer",
          "label": "Has Kidney Cancer",
          "control": "enum",
          "options": ["yes", "no", "unknown"],
          "required": true
        }
      ]
    }
  },
  "patients": [
    {
      "mrn": "M0001",
      "notes": {
        "R0001": {
          "note_text": "...",
          "mrn": "M0001",
          "note_date": "2025-01-15",
          "note_type": "Pathology",
          "department": null
        }
      },
      "events": [
        {
          "event_key": "R0001::triage_report::1",
          "event_type": "triage_report",
          "note_id": "R0001",
          "fingerprint": "a1b2c3d4e5f6",
          "fields": {}
        }
      ]
    }
  ]
}
```

`event_key` is the stable join key between the package and the review log. It is
**identity-addressed** — the *slot* a finding occupies:

```text
<note_id>::<event_type>::<identity>
   identity = the declared event_identity_fields where present,
              else the finding's ordinal for its type in the report (1, 2, …)
```

Do not edit event keys by hand. Keying by **identity** (not content) is what lets
two runs' versions of the same finding line up for adjudication — and it keeps a
verdict attached to the finding as its value is edited. For events without
declared identity, the ordinal pairs once-per-report findings across runs (a lone
`record_nephrectomy_specimen` is always `::1`). Each event also carries a
`fingerprint` (a hash of its field values): a changed value changes the
fingerprint — driving re-review and cross-run agreement — without moving the slot.

## Review Log Shape

The review app writes one JSON object per verdict. The file is an append-only
event record; if an event is reviewed more than once, the entry with the latest
`reviewed_at` for that `event_key` wins.

```json
{
  "event_key": "R0001::triage_report::1",
  "mrn": "M0001",
  "event_type": "triage_report",
  "note_id": "R0001",
  "verdict": "approved",
  "edits": {
    "has_kidney_cancer": "yes"
  },
  "comment": "",
  "reviewer": "Reviewer Name",
  "reviewed_at": "2026-06-12T12:34:56.000Z"
}
```

Supported verdicts are:

- `approved`: included in silver, with edits applied.
- `rejected`: excluded from silver.

All events in the package must have a review verdict before ingestion writes
that batch's silver table. The rule is **per batch**, so one batch's state never
blocks another:

- a package with **no reviews log yet** is skipped — you simply can't build its
  silver until the reviewer sends one back, and that stops nothing;
- a **complete** review log builds silver;
- a **present-but-incomplete** (or invalid) review log fails the ingest with an
  error naming it — an unfinished log shouldn't have been dropped in — but only
  that batch is blocked; the complete batches alongside it still build.

## Silver Table Semantics

Each package already sits in the review inbox under its batch folder; the
reviewer just drops the matching `*.reviews.jsonl` beside it. A batch is
reviewed one segment at a time, so a batch with several segments has several
pairs in the one folder:

```text
inbox/fc_reviews/<batch>/<batch>.001.review_pkg.json
inbox/fc_reviews/<batch>/<batch>.001.reviews.jsonl
inbox/fc_reviews/<batch>/<batch>.002.review_pkg.json   # a later segment's review
inbox/fc_reviews/<batch>/<batch>.002.reviews.jsonl
```

The loader discovers these recursively, so a flat
`inbox/fc_reviews/<batch>.001.review_pkg.json` still works; the nested,
batch-named layout is just the default the run hook writes.

Then run:

```bash
oncai ingest fc_reviews
```

The loader joins package events to review verdicts by `event_key`:

- approved events become rows,
- reviewer edits override original extracted fields,
- rejected events are counted and excluded,
- unreviewed events fail the ingest,
- review lines with no matching package event are ignored with a note.

A batch's segment reviews then **merge into one silver table** the way the raw
side merges segments — for each `note_id`, the highest segment that reviewed it
wins (a note re-reviewed in a later segment supersedes its earlier row). Each
row's `batch_name` records which segment it came from. The output is one clean
parquet per batch:

```text
lake/fc_reviews/<batch>.parquet   →   extractions_silver.<batch>
```

`build-db` and `oncai db update fc_reviews` load those parquets into the
`extractions_silver` schema, one DuckDB table per batch.

Each silver row includes fixed metadata columns:

```text
event_key
event_type
tool_name
note_id
mrn
definition_name
batch_name
package_generated_at
review_verdict
reviewer
reviewed_at
review_comment
note_date
note_type
department
original_fields_json
edits_json
reviewed_fields_json
key_hash
content_hash
```

The reviewed event fields are also flattened into columns. Sparse event types
can therefore share one table while preserving the complete original, edit, and
reviewed field JSON. The flat columns hold the **post-review values**
(`reviewed_fields = original_fields | edits`, edits winning); diff
`original_fields_json` against `reviewed_fields_json` to see exactly what a
reviewer changed.

`key_hash` is derived from `event_key`. `content_hash` is derived from the
reviewed fields and review metadata, so changing a verdict, edit, comment,
reviewer, or timestamp produces a new content hash.

## Gold Reshape

The silver table is event-grain and sparse on purpose — it preserves every
reviewed event losslessly. But an analyst usually wants a *dense* table: one row
per concept, only the relevant columns, no JSON blobs. That projection is a
**per-batch SQL sidecar**, and it's an inbox artifact like everything else.

Drop a `<batch>.sql` next to the lake parquet's source and `oncai build-db` runs
it right after building `extractions_silver.<batch>`, writing into the
`extractions_gold` schema:

```bash
# author the reshape for one batch, referencing that batch's silver table
cp docs/examples/ihc_gold_reshape.sql inbox/fc_reviews/kidney_v1.sql
# (edit it to point at extractions_silver."kidney_v1")

oncai ingest fc_reviews     # mirrors *.sql from inbox/fc_reviews/ into the lake
oncai build-db              # builds silver, then runs lake/fc_reviews/kidney_v1.sql
```

The sidecar is deliberately **aligned with one batch**: it names that batch's
silver table directly, so it can be exact about which rows it reshapes. It lands
its output in `extractions_gold` (a schema `build-db` auto-creates):

```sql
CREATE OR REPLACE TABLE extractions_gold.ihc_results AS
SELECT mrn, note_id, specimen_id, standardized_test_name, standardized_test_status, ...
FROM extractions_silver."kidney_v1"
WHERE event_type = 'record_ihc_result';
```

See [`docs/examples/ihc_gold_reshape.sql`](examples/ihc_gold_reshape.sql) for a
worked PathKidneyIhc reshape: it splits `record_ihc_result` and
`flag_report_for_review` into separate tables and `PIVOT`s the markers into a
per-specimen matrix (one column per marker). A reshape that fails is reported in
the build summary without aborting the rebuild — the same contract as the raw
side's per-base `.sql` transforms.

If you reshape several batches, each sidecar `CREATE OR REPLACE`s its own gold
tables, so either give them batch-qualified names or `INSERT` into a shared table
you created once — pooling across batches is yours to decide, since the reshape
is per batch by design.

## Authoring Requirements for FC Definitions

For a workflow to participate cleanly in review:

- Put reviewable findings in `ExtractionEvent` tools.
- Use `ExtractionPlan` only for orientation/control-flow data that should not
  be reviewed.
- Register `flag_report_for_review` as an `ExtractionEvent`.
- Include required `review_anchor: list[str]` snippets on the flag tool.
- Use `evidence` for exact source-note snippets supporting ordinary extractions.
- Prefer enums/literals, numeric fields, booleans, and structured dates for
  fields that should be compared across runs.
- Keep plain `str` fields for true free text. They render as text areas in the
  app and are ignored by disagreement selection.

The review app derives its controls from the Pydantic JSON schema:

| Schema shape | Review control |
|---|---|
| enum/literal | dropdown |
| integer/number | numeric input |
| boolean | checkbox |
| `ApproxDate` object | date + precision + anchor |
| `list` | read-only |
| plain string | text area |
| other object/list | read-only |

## Troubleshooting

- Empty package: no successful records matched the selected scope. Use
  `--scope all` or `--review-package all` if the goal is exhaustive review.
- Missing note text: the package builder could not reload the source table or
  source JSONL. The package still renders, but the app shows note text as
  unavailable.
- Ingest says a peer file is missing: a batch needs both
  `<batch>.review_pkg.json` and `<batch>.reviews.jsonl` with the same batch
  stem, in the same `inbox/fc_reviews/<batch>/` folder (the package is put
  there by the run; the reviews log must land beside it).
- Ingest fails on unreviewed events: finish every event in the package before
  loading silver, or rebuild a smaller package with the intended scope.
- A highlighted quote does not jump: `review_anchor` or `evidence` text
  must be an exact report snippet. Matching is whitespace-flexible and
  case-insensitive, but it still needs to be text present in the source note.
