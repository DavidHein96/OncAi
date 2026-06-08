# Draft prose — narrative sections, titles, and LinkedIn blurb

Companion to `blog_draft_technical_sections.md`. Covers the intro (§0), the
softer sections (§5 provenance, §6 review app, §7 tradeoffs), the close (§8),
plus title options and a LinkedIn post.

---

## §0 — Intro

I built a pipeline that turns raw pathology reports into physician-validated
structured data. It has no orchestrator, no data warehouse, no web framework,
and no ORM. That wasn't minimalism for its own sake. Three constraints made
frameworks the wrong tool and boring primitives the right one, and the whole
system is really just those three constraints followed to their conclusions:

1. **The data can't leave the machine.** It's PHI. No cloud warehouse, no
   managed queue, nothing that phones home.
2. **Every result has to be reproducible.** If a number ends up in a structured
   dataset, I need to say exactly what produced it — which note, which prompt,
   which model, which commit.
3. **The end user is a pathologist, not an engineer.** The last mile is a human
   reviewing the model's work, so the output has to be something a clinician
   with no Python can open and use.

What follows is a tour of what I reached for *instead* of each framework — and,
because I'm not trying to sell you a religion, an honest section on where a
framework would have won. The thread running through all of it: I track state in
the inputs, not in a database, and that one decision pays for itself about four
different ways.

---

## §5 — Provenance: record everything, every run

In research, an extracted value is only as trustworthy as your ability to
reproduce it. "The model said the tumor was grade 2" is not a fact until I can
tell you which note it read, which prompt I gave it, which model answered, and
which version of my code was running. Six months later, when someone asks why a
cohort number changed, "I think I tweaked the prompt around then" is not an
answer.

So every run stamps its full identity into the manifest and a run log:

```python
return {
    "backend": ..., "model": ...,
    "git_commit": git_info["commit"], "git_dirty": git_info["dirty"],
    "code_version": get_code_version(),
    "system_prompt_hash": hash_string(system_prompt),
}
```

The `git_dirty` flag is the small detail I'm proud of: it records whether I had
uncommitted changes when I ran the batch. A clean commit hash that doesn't
actually match what executed is worse than no hash at all, because it lies with
confidence. Recording "dirty" turns "this is reproducible" from a hope into a
checkable claim.

This is also where the build-system idea pays off a second time. I didn't write
a provenance system and an incremental-processing system. They're the *same*
hashes — `source_content_hash`, `system_prompt_hash`, the code version. The thing
that lets me skip work I've already done is the thing that lets me reconstruct
how any given value was produced. People reach for an experiment tracker to get
this. I got it as a side effect of content-addressing the work, and it lives in
plain parquet I can query with SQL instead of a service I have to run.

---

## §6 — The last mile: a review app with zero dependencies

A structured dataset that no clinician has checked is a hypothesis, not a result.
The pipeline's real output isn't the JSONL — it's a pathologist saying "yes,
that diagnosis is right; no, that date is wrong, here's the fix." So the last
component is a review tool, and its single hardest requirement is the user:
someone who has never opened a terminal.

That requirement wrote the architecture by itself. The review app imports nothing
outside the Python standard library — a `BaseHTTPRequestHandler`, some JSON, a
folder of static HTML and vanilla JS — specifically so it can be frozen into one
double-clickable executable:

```python
#   uv run pyinstaller --onefile --name oncai-review server.py
class Handler(BaseHTTPRequestHandler):
    def do_GET(self): ...    # /api/data serves the review package
    def do_POST(self): ...   # /api/review appends a verdict to a JSONL
```

No Flask, no React, no `npm install`, no DuckDB on the reviewer's machine. A
collaborator gets one binary and one file, opens the file, and starts
adjudicating. The "one file" is a self-contained review package — the events, the
schema that says how to render each field, and the source notes — so the app
needs no database and no copy of the repo to do its job.

Two details close the loop properly. First, evidence highlighting: each extracted
field links back to the exact quote in the source note, rendered inline, so the
reviewer is checking the model against the text and not against their memory.
Second, the verdicts themselves are written as append-only JSONL — the exact same
crash-safe primitive the extraction batches use. By this point in the system,
"append a line and flush" is just *how state is recorded*, whether it's a model's
output or a human's judgment. One good primitive, reused until it's boring.

---

## §7 — Where a framework would have won

I've spent this whole post explaining what I didn't use, so let me be honest
about when the other choice is right — because "no frameworks" as a blanket rule
is just a different kind of dogma.

- **A team.** The moment more than one person touches this, a framework stops
  being overhead and starts being a shared language. My append-only JSONL and
  manifest done-markers are obvious *to me*; a new hire would learn Airflow
  faster than they'd learn my conventions. Bespoke primitives have an onboarding
  tax that a solo project never pays.
- **Always-on, with SLAs.** My orchestration is "a human runs a batch and
  watches it." Real retries, alerting, backfills, and scheduled runs are exactly
  what a scheduler exists to provide, and reimplementing them well is a project
  in itself. The manifest-as-done-marker is great for jobs I babysit; it is not
  an on-call system.
- **Scale past one machine.** DuckDB-on-a-laptop is a *feature* here precisely
  because the data can't move. Lift that constraint — bigger data, distributed
  compute, many concurrent users — and a real warehouse earns its keep
  immediately.

The rule I actually followed isn't "frameworks bad." It's that frameworks trade
transparency for leverage. With one developer, a hard reproducibility
requirement, and data that can't leave the machine, I didn't need the leverage —
and the transparency turned out to be the product. The system is inspectable end
to end: every piece of state is a file I can open, and every result traces back
to the inputs that made it. For this problem, that was worth more than anything a
framework would have handed me.

---

## §8 — Close

Three constraints, three primitives. The data can't move, so the truth is a
folder of parquet and the database is a disposable cache. Everything has to be
reproducible, so I content-address the work — and get incremental processing for
free, or get reproducibility for free, depending on which way you read it. The
user is a human, so state is append-only files a person could audit by hand, all
the way down to the verdicts.

The one idea I'd carry to the next project isn't any of the specific tools — it's
content-addressing the expensive step. Hash a unit of work's inputs and you stop
asking a database what you've done and start asking a set difference, while
getting a perfect provenance trail as a byproduct. It happens to work beautifully
for non-deterministic, pay-per-call LLM extraction, but nothing about it is
LLM-specific.

If you want to see the shape of it, the repo's `definitions/example.py` is the
whole contribution surface: define a few Pydantic models, write a prompt, and you
have a new extraction workflow. One file, four jobs each.

---

## Title options

Ranked, with the angle each one leads with.

1. **"Make for LLM Extraction"** — short, technical, and it's the strongest idea
   in the post. Reads as a claim, which makes people click to see if you back it
   up. My pick if the audience is engineers.
2. **"What I Didn't Use, and Why: A Clinical LLM Pipeline Without Frameworks"** —
   your spine, stated plainly. Slightly longer but the "and why" signals judgment
   rather than contrarianism. Best all-rounder.
3. **"I Tracked Pipeline State in Hashes Instead of a Database"** — leads with the
   specific decision. Very clicky for a data/ML crowd.
4. **"A PHI-Safe Data Platform That Runs on a Laptop"** — leads with the
   constraint. Best if you want the healthcare/compliance audience.
5. **"Boring Primitives, On Purpose"** — vibe-first; pairs well as a subtitle
   under #1 or #2.

Suggested combo: **"Make for LLM Extraction — building a clinical data pipeline
without frameworks."** Idea first, context second.

---

## LinkedIn blurb

Two variants — a punchy one and a slightly longer one. Both first-person, both
end on a soft ask.

### Variant A (short, hook-forward)

> I built a clinical LLM pipeline with no orchestrator, no warehouse, and no web
> framework — and I think it's *better* for it.
>
> The core idea: treat LLM extraction like `make`. Hash each note's text, the
> prompt, and the code version, and only re-run what actually changed. Edit one
> sentence of a prompt and you re-extract the affected notes, not all 50,000. The
> same hashes that save you the API bill are also a perfect provenance trail.
>
> Three constraints drove every decision — the data can't leave the machine, every
> result has to be reproducible, and the end user is a pathologist with no Python.
> I wrote up what I reached for instead of each framework, and (honestly) where a
> framework would have won.
>
> New post: [link]

### Variant B (a little more context)

> Healthcare data can't leave the machine, every extracted value has to be
> reproducible, and the person checking the output is a clinician, not an engineer.
> Those three constraints meant the usual stack — an orchestrator, a warehouse, a
> React app — was the wrong tool. So I didn't use them.
>
> Instead: a versioned parquet lake as the source of truth, a DuckDB I can delete
> and rebuild, append-only JSONL as crash-safe state, and Pydantic models that
> serve as the LLM's schema, the validator, and the database columns all at once.
>
> The piece I'm most happy with is treating extraction like a build system —
> content-addressing each unit of work by its inputs, so re-running only touches
> what changed and provenance falls out for free.
>
> I also wrote an honest section on where a framework *would* have won, because
> "no frameworks" as a blanket rule is just a different dogma.
>
> [link]

### Notes on the blurb

- Lead with the contrarian hook (no orchestrator/warehouse/framework) — it stops
  the scroll. Then immediately pay it off with the `make` idea so it doesn't read
  as cheap contrarianism.
- The "same hashes save the bill *and* give provenance" line is your most
  shareable sentence. Keep it near the top.
- The honest-tradeoffs mention at the end does real work on LinkedIn — it signals
  seniority and preempts the "but what about scale?" reply-guys.
- Three to five lines is the sweet spot before "see more" truncates; Variant A is
  safer for reach, B for a more technical follower base.
