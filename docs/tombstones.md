# Tombstones — forgetting data in an inbox-canonical lake

This is the design for **removal** in oncai: how you "un-ingest" a completed
review, retire an old extraction batch, or drop a cohort you no longer want —
and have that removal hold *everywhere*, not just on the one machine you ran it
on.

Status: **v1 implemented for `fc_extractions`, `fc_reviews`,
`fc_adjudications`, and `cohorts`.**
Tombstones sync through `inbox/tombstones/`, ingest skips forgotten sources and
prunes their local lake projection, and `build-db` rebuilds from the pruned lake.
Physical garbage collection, remote deletion, `pathology`, and `runs`
tombstones are deferred.

## The problem

oncai is inbox-canonical (see [architecture.md](architecture.md)):

- the **inbox** is the immutable, content-hashed, append-only source of truth;
- the **lake** (parquet) and **DuckDB** are disposable projections rebuilt from
  the inbox by `oncai ingest` / `oncai build-db`;
- only the **inbox** syncs to remote, and the sync is **additive** —
  `pull_inbox_from_remote` / `push_inbox_to_remote` copy new/changed files and
  *never delete* (`oncai.lake._transfer_inbox_files`).

Those properties are exactly what makes naive deletion fail. Walk it through:

1. You `rm -rf inbox/fc_extractions/old_batch/` locally.
2. `oncai ingest` is additive per source — it builds parquets for the inbox
   sources that *exist* and never prunes. So `lake/fc_extractions/old_batch.parquet`
   is left stranded as an orphan, and `build-db` happily reloads it.
3. Even if ingest *did* prune the orphan, the **remote inbox still has
   `old_batch/`**. The next `pull` is additive and copies it straight back down;
   the next ingest rebuilds the parquet. The deletion silently undoes itself.

And you can't fix step 3 by teaching the sync to mirror deletions (`rsync
--delete`): an empty fresh clone could then push-delete the entire remote. That
footgun is precisely why the sync is additive in the first place.

**Conclusion:** removal cannot be a physical delete reconciled by the
projection. It has to be a change to the *canonical inbox* that **propagates
through the additive, copy-only sync** every other inbox file already uses.

## The principle: deletion is an append

Don't remove the source. **Append a marker that says the source is forgotten.**

A tombstone is just another immutable inbox file, so it rides the same additive
sync as everything else: `push` sends it up, every machine's `pull` brings it
down, and `ingest` honors it. The bytes of the original source stay put
(auditable, reversible); the projection simply stops including them.

This is the event-sourcing / Delta Lake / Iceberg answer, and it is the same
**append-only, latest-wins, content-addressed** shape oncai already uses for
**runs** (`inbox/runs/<run_id>.run.json`, started→completed) and **review
verdicts** (append-only log, latest `reviewed_at` per `event_key` wins).

## Tombstone records

A tombstone is a `forget` or `revive` **event** about a single inbox source.
Events are immutable, one per file, and resolved **latest-wins per target** —
so a `revive` after a `forget` propagates exactly the way the `forget` did.

```
inbox/tombstones/<event_id>.tombstone.json
```

```json
{
  "event_id": "a1b2c3d4e5f60718",
  "kind": "fc_extractions",
  "target": "old_batch",
  "action": "forget",
  "reason": "superseded by re-extraction with the v2 definition",
  "actor": "dave",
  "at": "2026-06-12T14:03:22.481Z"
}
```

- **`event_id`** — 16 hex chars, makes the filename unique and conflict-free so
  two machines never write the same file (mirrors `run_id`).
- **`kind`** — the dataset folder the target lives in (`fc_extractions`,
  `fc_reviews`, `cohorts`, `pathology`, …). One namespace per folder.
- **`target`** — the identity of the inbox source within that folder (see
  [target identity](#target-identity-per-folder-mode)).
- **`action`** — `forget` | `revive`.
- **`at`** — ISO-8601 UTC; the latest-`at` event for a `(kind, target)` wins.
- **`reason` / `actor`** — free-text provenance for the audit trail.

A flat `inbox/tombstones/` (kind inside the record, not the path) mirrors
`inbox/runs/` and keeps ingest's lookup trivial. The set of *active* tombstones
is computed by grouping events on `(kind, target)` and keeping the latest `at`;
a target is forgotten iff its latest action is `forget`. Order-independent:
membership is resolved from the full set every ingest, never from arrival order.

### Target identity per folder mode

"What can I forget, and how is it keyed" follows directly from each folder's
`IngestMode` (`oncai.config.FOLDER_MODES`):

| Folder | Mode | Inbox source unit | `target` | Effect of a `forget` |
|---|---|---|---|---|
| `fc_extractions` | STATIC | a batch folder `<batch>/` | `<batch>` | drop `lake/fc_extractions/<batch>.parquet` + its `extractions_raw.<batch>` table |
| `fc_reviews` | STATIC | a batch's review pairs `<batch>/` | `<batch>` | drop `lake/fc_reviews/<batch>.parquet` + `extractions_silver.<batch>` (and the gold reshape that reads it) |
| `fc_adjudications` | STATIC | an adjudication round folder `<round>/` | `<round>` | drop `lake/fc_adjudications/<round>.parquet` + `extractions_adjudicated.<round>` (and the gold reshape that reads it) |
| `cohorts` | NAMED | a `<name>.csv` | `<name>` | drop that cohort's parquet + table |
| `pathology` | DATED | a `YYYY-MM-DD_<label>.csv` | the filename stem | deferred for v1 |
| `runs` | MANIFEST | a `<run_id>.run.json` | `<run_id>` | deferred for v1 |

The unit is always "one inbox source as the user thinks of it." For the
per-source modes (STATIC/NAMED) a forget removes a whole parquet/table; for the
merged modes (DATED/MANIFEST) it removes that source's *contribution* to the
single merged parquet on the next replay.

## Ingest becomes a reconcile

Today each folder handler in `oncai.ingest` is additive: it writes parquets for
the inbox sources it finds and never removes anything. With tombstones, ingest
gains one rule, applied uniformly:

> **lake(folder) = projection of { inbox sources } − { active tombstones for folder }**

Concretely, at the top of `run_ingest` we resolve the active tombstone set once
(`{(kind, target)}`), then each handler:

1. **skips** building a parquet for any source whose `(folder, name)` is
   tombstoned; and
2. **prunes** any existing lake parquet whose source is tombstoned *or* no
   longer present in the inbox at all (a belt-and-suspenders orphan sweep —
   tombstones are the propagating mechanism, but a locally-`rm`'d source with no
   tombstone yet also shouldn't leave a stale parquet on *this* machine).

Pruned/skipped items are reported in `FolderResult.notes` (rendered yellow), so
a run that forgets things says so out loud rather than silently shrinking.

On the DuckDB side:

- **`oncai build-db`** does `db_path.unlink()` + full rebuild from the lake, so
  it self-corrects the moment the lake is correct. It is the **complete** reset.
- **`oncai db update <folder>`** is incremental: it `CREATE OR REPLACE`s each
  present parquet's table and **`DROP TABLE`s** any base table whose backing
  parquet was pruned (reported as a removal, not a zero-row update).

> **Caveat — derived tables need a full rebuild.** `db update` only reconciles a
> folder's own *base* tables (`extractions_raw.<batch>`,
> `extractions_silver.<batch>`, cohort tables). It does **not** drop the tables a
> forgotten batch's `.sql` sidecar built in the *transform* schemas
> (`extractions_transformed`, `extractions_gold`) — the incremental path has no
> record of which derived tables a now-pruned batch produced. So after
> forgetting a batch that had a reshape sidecar, run a full **`oncai build-db`**
> (which `DROP SCHEMA … CASCADE`s every owned schema) to be sure no stale
> derived rows survive. Rule of thumb: **tombstoning anything with a `.sql`
> reshape ⇒ do a full `build-db`, not just `db update`.**

## Commands

```bash
# Preview (default): show the inbox source, lake parquet, and DB table that
# would be affected — touches nothing.
oncai forget fc_extractions old_batch
oncai forget fc_reviews v1 --reason "bad prompt, re-reviewing"

# Apply: append the forget event + prune the local projection for instant
# feedback. The tombstone syncs on the next `push`.
oncai forget fc_reviews v1 --reason "..." --yes

# Undo — append a revive event; it propagates the same way.
oncai revive fc_reviews v1 --yes

# See what's forgotten (the tombstone log, projected like any other folder).
oncai forget --list
oncai ingest tombstones       # optional: project to lake/tombstones/tombstones.parquet
oncai build-db                # query as: SELECT * FROM meta.tombstones WHERE active
```

`oncai forget` is the guided front door; it writes the event and prunes the
local lake parquet/SQL sidecar immediately so you see the local lake effect
without waiting for a full ingest. Reconcile-on-ingest is the *invariant* that
makes the same thing happen on every other machine after they `pull`. Run
`oncai build-db` to rebuild the DuckDB projection; `oncai db update <folder>`
also drops stale base tables for the touched multi-table folder.

The tombstone log is itself projectable: a MANIFEST-style union into
`lake/tombstones/tombstones.parquet` → `meta.tombstones`, with a resolved
`active` boolean, gives a queryable "what have we forgotten, when, why, by whom."

## Logical now, physical later

A tombstone is a **logical** delete: the source bytes remain in the inbox. That
is deliberate —

- it's **instant and safe** (no destructive op in the hot path),
- it's **reversible** (`revive`),
- it's **auditable** (the original is still there, with a record of why it was
  dropped),
- and it **propagates** cleanly through additive sync.

Reclaiming the bytes is a separate, explicit, deliberately destructive step:

```bash
oncai gc            # preview: list tombstoned sources whose bytes are still present
oncai gc --yes      # physically delete the inbox bytes of forgotten sources
```

`oncai gc` is the **only** command that physically deletes inbox data. It acts
on locally-present bytes of targets whose latest action is `forget`. Remote
bytes are out of scope for v1 (see below) — the tombstone already makes the data
*logically* gone everywhere; `gc` is just space reclamation, and is the one
place where "are you sure" is warranted.

## Propagation guarantees

- **Forget propagates up and down.** The tombstone is a new inbox file; `push`
  sends it to remote, every machine's `pull` brings it down. No `--delete`, no
  resurrection.
- **Revive propagates identically.** Because resolution is latest-`at`-wins per
  target, appending a `revive` after a `forget` (or vice versa) reaches every
  machine and flips the projection on the next ingest.
- **Order-independent.** A machine that receives `old_batch/` and its tombstone
  in either order, or in the same pull, resolves the same state — membership is
  computed from the whole set, not from sequence.
- **Multi-writer safe.** Each event is its own file keyed by a random
  `event_id`; two machines forgetting/reviving concurrently produce two files,
  not a conflict. Latest-`at` decides the outcome deterministically.

## What this is explicitly NOT

- **Not `rsync --delete`.** The sync stays additive; we never mirror deletions.
- **Not history mutation.** We don't rewrite or remove the original inbox files
  to forget them — forgetting is an append. (`gc` is the separate, opt-in
  exception for reclaiming bytes.)
- **Not row-level surgery.** v1 forgets *sources* (a batch, a cohort, a dated
  file), the unit the user already reasons about — not individual records inside
  a parquet.

## Open questions for v1

1. **Remote `gc`.** v1 reclaims **local** bytes only; the tombstone makes data
   logically gone everywhere, but the remote keeps the original bytes until
   someone runs a remote-aware reclaim. Do we want a guarded
   `oncai gc --remote` later, or is "logical-gone everywhere + manual remote
   cleanup" enough? (Deferred — the user asked to think on the remote angle.)
2. **Forget-then-reuse-the-name.** If `old_batch` is forgotten and a *new*
   `old_batch` is later created, the latest tombstone action governs. A `forget`
   at T1 does not forget a source authored at T2 > T1; the resolver must compare
   the source's own provenance time, or we simply require `revive` before
   reusing a name. Needs a decision.
3. **Should `runs` be forgettable at all?** It's the immutable audit log;
   tombstoning a run is arguably a contradiction. Likely allow it but discourage
   it (the tombstone is itself the audit trail).
4. **GC and content hashes.** After `gc` removes the bytes, a later re-ingest
   must not "miss" them as orphans-without-tombstone; the tombstone remains the
   record that they were intentionally gone.

## Relationship to existing pieces

- Mirrors the **runs** lifecycle (`oncai.runs`) and **review verdict** log
  (latest-wins) — same append-only, content-addressed, sync-additive discipline.
- Closes the loop opened by [review_system.md](review_system.md): forgetting a
  review batch retires its `extractions_silver.<batch>` table and the
  `.gold.sql` reshape that reads it, on every machine, via one appended file.
- Keeps the **lake disposable**: the reconcile rule makes the lake a faithful
  `inbox − tombstones` projection, restoring the "pure function of the inbox"
  property for deletions, not just additions.
