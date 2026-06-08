# Outline: *What I Didn't Use, and Why — Building a Clinical LLM Extraction Pipeline Without Frameworks*

**Alt titles:** "Make for LLM Extraction" · "A PHI-Safe Data Platform That Runs on a Laptop" · "Boring Primitives, On Purpose"

---

## 0. Hook (½ page)

Open with the punchline, not the background:

> I built a pipeline that turns raw pathology reports into physician-validated structured data. It has no orchestrator, no warehouse, no web framework, and no ORM. Not because I'm a minimalist — because three constraints made frameworks the *wrong* tool, and primitives the right one.

Then state the three constraints up front, because every later decision traces back to them:

1. **PHI can't leave the machine.** No cloud warehouse, no managed queue.
2. **Every result must be reproducible.** I have to explain exactly what produced any extracted value.
3. **A pathologist with no Python has to use the output.** The last mile is a human, not a dashboard.

End the intro with a one-line table of contents: *"Here's what I reached for instead of each framework, and where a framework would actually have won."*

---

## 1. Instead of a schema DSL → one Pydantic model doing four jobs

**The reach:** a config language / JSON schema files / separate validation layer.

**What I did:** the extraction unit *is* a Pydantic model. The same class is (1) the tool definition the LLM sees, (2) the validator for every tool call, (3) the output data shape, and (4) the human-readable spec.

```python
# definitions/example.py  (trimmed)
class RecordDiagnosis(ExtractionEvent):
    """Record a diagnosis finding. The docstring becomes the tool description."""
    diagnosis_date: ApproxDate = Field(..., description="Date of diagnosis (YYYY, YYYY-MM, or YYYY-MM-DD).")
    diagnosis_type: DiagnosisType = Field(..., description="primary, secondary, or recurrence.")
    diagnosis_name: str = Field(..., description="e.g., 'renal cell carcinoma'.")
    stage: str | None = Field(None, description="Cancer stage if documented.")
```

**Talking points:**
- The field `description=` strings are simultaneously prompt engineering and documentation. Enums constrain the model to valid values for free.
- One source of truth: change the model, and the prompt schema, the validator, and the DB columns all move together.

**The war story (this is the credibility moment):** local/vLLM tool-call parsers don't dereference JSON-schema `$ref`, so Pydantic's enum-as-`$ref` output made enum values *invisible* to the model. Had to inline them:

```python
# tools.py  (trimmed)
def _inline_refs(schema):
    """Pydantic emits enums as $ref into $defs. Many local LLMs / vLLM parsers
    don't dereference $ref, so enum values become invisible to the model.
    Inlining yields a self-contained schema."""
    defs = schema.pop("$defs", {})
    def resolve(node): ...  # walk, replace #/$defs/... with the actual subschema
    return resolve(schema)
```

> Lesson to state plainly: "OpenAI tolerates `$ref`; the open models I wanted to run didn't. You only learn that by shipping against both."

---

## 2. Instead of an orchestrator → append-only JSONL + a manifest done-marker

**The reach:** Airflow / Prefect / Celery for a batch of N independent LLM calls.

**What I did:** crash-safe, resumable batch processing in a single file. Each finished note is flushed to JSONL immediately; the run isn't "done" until a manifest is written *after* the JSONL closes.

```python
# batch_single.py  (trimmed)
def _write_record(out_file, record):
    line = json.dumps(record) + "\n"
    with write_lock:
        out_file.write(line)
        out_file.flush()        # survive a crash mid-batch
```

```python
# the manifest is the "done marker" — written only after the JSONL is closed,
# so its *presence* means the batch finished cleanly.
_write_single_note_manifest(..., result=result)
```

**Talking points:**
- Resume = "read which `note_id`s are already in the JSONL, skip them." No external state store.
- Parallelism is a `ThreadPoolExecutor` with a write lock — ~20 lines, not a worker fleet.
- The "20% of an orchestrator I actually needed" framing goes here.

---

## 3. The centerpiece — instead of a job-state DB → content hashes (LLM extraction as a build system)

**The reach:** a database tracking what's been processed.

**The insight:** treat extraction like `make`. Hash the *inputs* — source note content, the system prompt, the code version — and only re-run what changed.

```python
# load.py: each record is content-addressed
"source_content_hash": record.get("source_content_hash"),   # hash of the note text
"system_prompt_hash":  run_meta.get("system_prompt_hash"),  # hash of the prompt
"content_hash": blake2b_128(events_json + (finish_json or "")),
```

On an incremental run, every source row is bucketed:

```python
# cli/fc_cmds.py: _categorize_delta  (trimmed to the spine)
prior = existing.get(note_id)
if not prior:
    new_ids.add(note_id)                      # never seen
elif content_hash not in {p.hash for p in prior}:
    changed_ids.add(note_id)                  # note text edited / addendum
elif reextract_on_prompt_change and prompt_hash not in {p.prompt for p in prior}:
    prompt_changed_ids.add(note_id)           # I edited the prompt
else:
    skipped += 1                              # nothing changed → don't re-pay
```

**Talking points (this section sells the whole post):**
- Edit one sentence of your prompt → re-extract only the affected notes, not all 50k. **Cost control + reproducibility in one mechanism.**
- This is Make/Bazel thinking, applied to a non-deterministic LLM step. Frame it explicitly: *"The orchestrators I skipped track state in a DB. I track it in the inputs themselves — which is also exactly what makes runs reproducible."*
- Show the CLI output buckets ("3 new + 1 changed + 0 prompt-changed; skipping 4,996 already-extracted") — it makes the idea concrete.

---

## 4. Instead of a warehouse → versioned parquet lake + DuckDB as a disposable view

**The reach:** Snowflake / BigQuery / Spark.

**What I did:** parquet is the source of truth (versioned, content-hashed for dedup); DuckDB is a *materialized view you can delete and rebuild anytime*; SQL sidecar files reshape per workflow.

```python
# load.py: atomic write — tmp → fsync → rename, so a crash never corrupts the lake
tmp = output_path.with_name(output_path.name + ".tmp")
df.write_parquet(tmp, compression="zstd")
fd = os.open(tmp, os.O_RDONLY); os.fsync(fd); os.close(fd)
tmp.replace(output_path)        # atomic
```

**Talking points:**
- Runs on a laptop. **The data never moves** — which is the whole PHI ballgame.
- "Disposable DuckDB" is the mental unlock: the DB carries no irreplaceable state, so a bad build is `rm` + rebuild, never a migration.
- Blake2b `key_hash` / `content_hash` per row give incremental dedup without a CDC system.

---

## 5. Instead of provenance tooling → record everything, every run

**The reach:** MLflow / W&B / an experiment tracker.

**What I did:** every run stamps git commit + dirty flag, code version, prompt hash, model, and sampling params into the manifest and a run log.

```python
# batch_single.py: _build_run_meta  (trimmed)
return {
    "backend": ..., "model": ...,
    "git_commit": git_info["commit"], "git_dirty": git_info["dirty"],
    "code_version": get_code_version(),
    "system_prompt_hash": hash_string(system_prompt),
}
```

**Talking points:**
- Point at any extracted value → reconstruct the exact prompt + model + code that produced it. That's constraint #2, satisfied by a dict.
- Tie it back: the same hashes that drive incremental runs (§3) *are* the provenance record. One mechanism, two payoffs.

---

## 6. Instead of a web framework → stdlib HTTP + vanilla JS, freezable to one exe

**The reach:** Flask/FastAPI + React for the review UI.

**What I did:** the physician review app imports nothing outside the standard library, so it freezes into a single shareable binary.

```python
# review_app_reference/server.py  (from the module docstring)
#   uv run pyinstaller --onefile --name oncai-review server.py
class Handler(BaseHTTPRequestHandler):
    def do_GET(self): ...    # /api/data serves the review package
    def do_POST(self): ...   # /api/review appends a verdict to a JSONL
```

**Talking points:**
- The user is a pathologist with no Python. Constraint #3 *forced* zero dependencies — "double-click an exe, open a file, adjudicate."
- The app consumes one self-contained `*.review_pkg.json` (schema + events + source notes), so it needs no DB or repo. Mention the package builder as the producer side of that contract.
- Verdicts are append-only JSONL too — same crash-safe primitive as §2, reused.
- Nice closer for the section: the evidence-highlighting (each extracted field links back to the exact quote in the note) — human-in-the-loop, closed.

---

## 7. Where a framework *would* have won (credibility section — don't skip)

Short and honest. This is what makes the thesis read as judgment, not dogma:

- **A team.** Shared infra, onboarding, and "the framework is the lingua franca" beat bespoke primitives once >1 person touches it.
- **SLAs / always-on.** A real scheduler, retries, alerting, backfills — my "manifest done-marker" is for batch jobs I babysit, not a 24/7 service.
- **Scale past one machine.** DuckDB-on-a-laptop is a feature *because* of PHI; lift that constraint and a warehouse earns its keep.

State the rule you actually followed: *"Frameworks trade transparency for leverage. With one developer, hard reproducibility, and data that can't move, I didn't need the leverage — and the transparency was the product."*

---

## 8. Close (¼ page)

- Reprise the three constraints → the three primitives (hashes, JSONL, parquet/DuckDB).
- One forward-looking line: the content-hash idea generalizes to any expensive non-deterministic step, not just LLMs.
- Link the repo; point readers at `definitions/example.py` as the "add your own extraction in one file" entry point.

---

## Notes for when you draft

- **Lead with §3 energy even in the intro** — tease "I track processing state in the inputs, not a database" early so readers know the good idea is coming.
- Every code block is short on purpose. Link to the file for the full version; the blog shows the *spine*.
- The two highest-signal "I actually shipped this" details are the `$ref` inlining (§1) and the prompt-hash delta bucket (§3). Don't bury them.
