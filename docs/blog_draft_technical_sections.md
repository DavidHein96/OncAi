# Draft prose — technical sections

First-person draft of the four most technical sections (§1–§4). Voice is meant
to match a portfolio/engineering blog: confident, concrete, no hand-waving.
Code blocks are trimmed for the page; link to the real files in the post.

---

## One Pydantic model, four jobs

Most extraction pipelines I've seen keep four things in sync by hand: the schema
you send the model, the validator that checks what comes back, the database
columns you land it in, and the document that tells a human what any of it
means. Four artifacts, four places to drift.

I collapsed them into one. The unit of extraction is a Pydantic model, and that
single class *is* all four:

```python
class RecordDiagnosis(ExtractionEvent):
    """Record a diagnosis finding. The docstring becomes the tool description."""
    diagnosis_date: ApproxDate = Field(..., description="Date of diagnosis (YYYY, YYYY-MM, or YYYY-MM-DD).")
    diagnosis_type: DiagnosisType = Field(..., description="primary, secondary, or recurrence.")
    diagnosis_name: str = Field(..., description="e.g., 'renal cell carcinoma'.")
    stage: str | None = Field(None, description="Cancer stage if documented.")
```

When the model decides to record a diagnosis, this class becomes the function
schema in the tool-call request — the docstring is the tool description, every
`Field(description=...)` is an inline instruction, and `DiagnosisType` (a
`StrEnum`) hard-constrains the value to a legal option. When the call comes
back, the *same* class validates it before a single byte touches disk. And when
I later query the results, the same field names are the columns. Change the
model and all four move together, because there is only one of them.

The part I want to flag is that those `description` strings are doing real work.
They aren't documentation that rots — they're the prompt. "Tighten the
extraction" usually means editing a field description, not a megaprompt off in
another file. The schema and the instructions live in the same place because
they're the same thing.

### The `$ref` war story

Here's the detail that only shows up once you run this against models that
aren't OpenAI's.

Pydantic, very reasonably, factors enums out into a `$defs` block and references
them with `$ref`:

```json
{ "diagnosis_type": { "$ref": "#/$defs/DiagnosisType" } }
```

OpenAI's tool layer dereferences that fine. Several of the open models I wanted
to serve through vLLM did not — their tool-call parsers saw a `$ref` they didn't
resolve, which meant the enum's allowed values were *invisible to the model*. It
would happily invent a `diagnosis_type` that no human had ever sanctioned,
because as far as it could tell the field was a free string.

The fix is to inline the references so every schema is self-contained:

```python
def _inline_refs(schema):
    """Pydantic emits enums as $ref into $defs. Many local LLMs / vLLM parsers
    don't dereference $ref, so enum values become invisible to the model.
    Inlining yields a self-contained schema."""
    defs = schema.pop("$defs", {})
    def resolve(node):
        # walk the tree; wherever a node is {"$ref": "#/$defs/X"},
        # replace it with a deep copy of defs["X"], recursively.
        ...
    return resolve(schema)
```

It's a small function, but it's the kind of thing you can't design up front —
you find it because you tested against the models you actually intended to run,
not the one with the friendliest API. That's the whole reason the client layer
is an abstraction over Azure, vLLM, and vLLM-chat instead of a thin wrapper
around one SDK.

---

## Crash-safe batches without an orchestrator

A batch here is a few thousand independent LLM calls: one pathology report in,
one set of structured findings out, no shared state between reports. That shape
is exactly what an orchestrator is *for* — and exactly where one would have been
overkill. I didn't need a scheduler, a worker pool I don't control, or a
metadata database. I needed three properties: don't lose work if the process
dies, don't redo work I've already paid for, and run a few of them at once.

All three fall out of one decision: the output file is append-only, and it's the
state.

Each finished note is serialized to a line of JSONL and flushed immediately:

```python
def _write_record(out_file, record):
    line = json.dumps(record) + "\n"
    with write_lock:
        out_file.write(line)
        out_file.flush()        # survive a crash mid-batch
```

If the machine dies at note 3,000 of 5,000, the first 3,000 are already on disk,
intact, one complete JSON object per line. There's no partial-write to clean up
because a record is either fully written and newline-terminated or it isn't
there at all.

Resume isn't a feature I bolted on; it's just *reading the file back*. Before a
run, I load the `note_id`s already present and skip them. The output is the
checkpoint.

Parallelism is a thread pool and the same write lock — about twenty lines, not a
worker fleet:

```python
with ThreadPoolExecutor(max_workers=workers) as pool:
    futures = {pool.submit(_process_single_note, note=n, ...): n for n in pending}
    for fut in as_completed(futures):
        _write_record(out_file, fut.result())   # lock serializes the appends
```

The last piece is knowing when a batch is *done*, as opposed to interrupted. I
write a manifest — counts, config, git info — but only after the JSONL is closed.
Its presence is the done-marker:

```python
# Written only after the JSONL is closed, so the manifest existing
# means the batch finished cleanly.
_write_single_note_manifest(..., result=result)
```

A half-finished run has a JSONL and no manifest, which is precisely the state
"resume me." A finished run has both. I rebuilt the maybe-20% of an orchestrator
I actually needed, and in exchange the entire state of a batch is two files I can
open in a text editor.

---

## The good idea: content hashes, or LLM extraction as a build system

This is the part I'd put on the whiteboard.

The orchestrators I skipped track what's been processed in a database. I track it
in the *inputs*. Every extraction is content-addressed by three hashes: the
source note's text, the system prompt, and the code version that produced it.

```python
# each output record carries the hashes of what produced it
"source_content_hash": record.get("source_content_hash"),   # hash of the note text
"system_prompt_hash":  run_meta.get("system_prompt_hash"),  # hash of the prompt
"content_hash": blake2b_128(events_json + (finish_json or "")),
```

Once your unit of work is keyed by the hash of its inputs, "what do I need to
re-run?" stops being a bookkeeping question and becomes a set difference. On an
incremental run I bucket every source row against what's already been extracted:

```python
prior = existing.get(note_id)
if not prior:
    new_ids.add(note_id)                      # never seen
elif content_hash not in {p.hash for p in prior}:
    changed_ids.add(note_id)                  # note text edited, or an addendum landed
elif reextract_on_prompt_change and prompt_hash not in {p.prompt for p in prior}:
    prompt_changed_ids.add(note_id)           # I changed the prompt, not the data
else:
    skipped += 1                              # nothing changed → don't re-pay
```

This is `make` for LLM extraction. `make` doesn't recompile a `.o` file whose
`.c` source is untouched; this doesn't re-call a model on a report whose text —
and whose prompt — are unchanged. The dependency graph is just "output depends
on (note text, prompt, code)," and the hashes are the timestamps.

What that buys you, concretely:

- **A new addendum on one report** changes that report's `content_hash`. It falls
  into `changed`, gets re-extracted, and the other 4,999 reports don't move.
- **You edit one sentence of the system prompt** to fix a systematic error. Every
  report's `prompt_hash` now differs, so with `--reextract-on-prompt-change` you
  re-run exactly the affected set and nothing else. Without that flag, a prompt
  tweak is explicitly *not* a reason to re-pay for unchanged data — your call,
  encoded in a flag rather than a comment.
- **Everything else is skipped**, and the run tells you so:

  ```
  Incremental delta: 3 new + 1 changed (addenda) + 0 prompt-changed + 0 forced;
  skipping 4,996 already-extracted
  ```

The line I'd land the section on: the same three hashes that decide what to
*skip* are also the complete record of what *produced* every value I keep. The
mechanism that controls cost and the mechanism that gives me reproducibility are
the same mechanism. I didn't build incremental processing and provenance as two
systems; I content-addressed the work and got both. A database would have given
me the first and not the second.

---

## A warehouse I can delete: parquet lake + disposable DuckDB

The data can't leave the machine, so the entire "ship it to a warehouse" branch
of data engineering was closed to me from the start. What I built instead has a
shape worth stealing even when PHI isn't the constraint: a durable source of
truth that's *only* files, and a query engine that owns no irreplaceable state.

Raw EHR exports get ingested into a versioned parquet lake. That lake is the
source of truth — every row carries a Blake2b `key_hash` and `content_hash`, so
re-ingesting the same data is a no-op and changed rows are detected without a
change-data-capture system. Writes are atomic, because a corrupt lake is the one
failure you can't `rm` your way out of:

```python
# tmp → fsync → rename, so a crash mid-write never leaves a torn parquet
tmp = output_path.with_name(output_path.name + ".tmp")
df.write_parquet(tmp, compression="zstd")
fd = os.open(tmp, os.O_RDONLY); os.fsync(fd); os.close(fd)
tmp.replace(output_path)        # rename is atomic on POSIX
```

DuckDB sits on top, and the mental unlock is that **it's disposable**. It isn't a
database I migrate and protect; it's a materialized view I rebuild from the lake
on demand. Schemas for raw data, cohorts, extractions, and run logs all get
assembled from parquet, with per-workflow `.sql` sidecar files doing the
relational reshaping at build time. If a build goes wrong, the fix is `rm
oncai.duckdb` and rebuild — never a migration, never a stuck state, because none
of the truth lives there.

That division — files are the truth, the database is a cache — is what lets the
whole thing run on a laptop. Nothing moves, nothing phones home, and "back up my
data" means "copy a folder of parquet." The PHI constraint forced the
architecture, but I'd reach for it again on any project where I want the data
layer to be boring, inspectable, and impossible to corrupt by accident.
