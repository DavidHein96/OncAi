# Design notes

A short tour of the decisions behind `oncai-review` and why they were made the
way they were. The guiding constraint throughout: a clinician should be able to
review extractions on a locked-down hospital workstation with **no install, no
network, and no data leaving the machine**.

## Standard library only

`server.py` imports nothing outside the Python standard library. That is a
deliberate, load-bearing constraint, not an aesthetic one:

- It can be frozen into a **single self-contained executable** with PyInstaller
  (`--onefile`). A collaborator runs one file — no Python, no `pip install`, no
  virtualenv, no database engine.
- There is no dependency supply chain to vet for a tool that touches clinical
  text, and nothing to drift or break over time.

The cost is writing a tiny HTTP layer by hand on top of `http.server` instead of
reaching for a framework. For an app with three endpoints, that is a good trade.

## Local web app instead of a desktop GUI

The UI is plain HTML/CSS/JS served over `http.server` and opened in the user's
default browser. This gets a rich, familiar review surface (scrolling notes,
inline highlights, copy-to-clipboard chips) without bundling a GUI toolkit or a
browser engine. The server binds to `127.0.0.1` only, so it is never reachable
off the machine, and it **auto-falls back to an open port** if the preferred one
is busy so the app always starts.

## Bundling web assets into the one-file exe

When PyInstaller freezes the app, the `web/` directory is bundled via
`--add-data` and unpacked at startup to a temporary dir exposed as
`sys._MEIPASS`. `_web_dir()` checks for that path first and falls back to the
sibling `web/` folder when running from source, so the exact same code serves
assets whether run from source or as a frozen binary.

## Append-only verdict log

Verdicts are written to a `*.reviews.jsonl` sidecar, one JSON object per save.
Reviewing is **append-only**: editing a verdict writes a new line rather than
mutating an old one. On load the log is replayed and the last write per
`event_key` wins (`ReviewState._load_existing_reviews`). This buys:

- **Crash safety** — a half-written session is just a shorter log; nothing is
  corrupted in place.
- **Auditability** — the full history of how a reviewer changed their mind is
  preserved, which matters for clinical data.
- **Trivial concurrency** — appends under a lock, no read-modify-write.

Malformed or blank lines are skipped on replay, so a log is never un-loadable.

## Security: path-traversal guard

Static files are served from `web/`, but the requested path is attacker-controlled
in principle. `_serve_static` resolves the target and verifies it is still inside
`web/` via `Path.relative_to`, returning `403` otherwise — so a request like
`/../server.py` cannot escape the asset directory. This is covered by a test
(`test_path_traversal_is_blocked`).

## Authoritative validation on the server

The browser validates edited dates as you type, but the server **re-validates**
before persisting (`_validate_review`), because client-side checks can be
bypassed. An `ApproxDate`'s `date` must be empty (unknown) or a real calendar
date in full `YYYY-MM-DD` form — a separate `precision` field conveys how much
is actually known (year / month / day), and an optional `anchor` must be one of
the allowed hints. Anything else is rejected with a `400` and a human-readable
message. The rule mirrors the extraction model's own date contract so reviewed
data stays round-trippable.

## Whitespace-flexible evidence highlighting

The model's explicit provenance snippets (`evidence` and `review_anchor`)
are highlighted inline in the source note. An exact substring match would
frequently miss, because note text is normalized on ingestion (runs of
whitespace collapse, double spaces become newlines). So the highlighter builds a
regex per snippet where any run of whitespace matches any run of whitespace,
case-insensitively, then merges overlapping match ranges before rendering. The
result is robust highlighting that survives reformatting without false-joining
adjacent spans.

## Testing strategy

The suite (`tests/test_server.py`) covers three layers without any test
dependencies beyond `pytest`:

1. **Pure helpers** — filename safety, date validation, path derivation.
2. **The verdict log** — replay, last-write-wins, append semantics, validation.
3. **Live HTTP round-trips** — a real server on an ephemeral port exercises
   static serving, the path-traversal guard, package load, and review save/reject.

The front end is tested too, without adding a build step or any npm
dependencies: `app.js` exports its pure helpers under Node (and boots the app in
the browser), and `web/app.test.js` exercises them with Node's built-in test
runner — the order-insensitive change detection (`canon`/`sameValue`) and the
whitespace-flexible evidence highlighter (`buildHighlightedNote`), which are the
parts most likely to break silently.

CI runs the Python suite on Linux, Windows, and macOS across Python 3.11 and
3.13, the front-end suite on Node, and gates every change on `ruff` (lint) and
`ty` (static type checking). The codebase is fully type-annotated, so `ty` runs
clean with no ignores.
