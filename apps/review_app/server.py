#!/usr/bin/env python3
"""Local physician review server for FC extraction outputs — standard library only.

Reads a ``*.review_pkg.json`` produced by ``oncai.review.package`` and serves a
localhost web UI for adjudicating extracted events (approve / reject / edit
fields), writing verdicts to a ``*.reviews.jsonl`` sidecar.

This module imports nothing outside the Python standard library so it can be
frozen into a single shareable executable:

    uv run pyinstaller --onefile --name oncai-review server.py
    # (bundles web/ via --add-data)

A collaborator then just runs ``oncai-review`` (no Python, DuckDB, or data lake)
and opens a ``*.review_pkg.json`` from the in-app file picker; verdicts are saved
under ``~/Documents/oncai_reviews/<batch>.reviews.jsonl``. Passing ``--package``
opens one immediately instead (the ``oncai fc review`` path), saving the reviews
next to the package.

Web assets (index.html / style.css / app.js) live in the sibling ``web/`` dir.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import mimetypes
import re
import sys
import threading
import tomllib
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Serve the SVG favicon as image/svg+xml everywhere. On Windows mimetypes reads
# the registry, which doesn't reliably map .svg, so register it explicitly.
mimetypes.add_type("image/svg+xml", ".svg")


def _web_dir() -> Path:
    """Directory holding the web assets (index.html / style.css / app.js).

    Running from source they sit next to this module in ``web/``. When frozen
    by PyInstaller (``--onefile``), bundled data is unpacked to ``sys._MEIPASS``
    at startup, so the one-file exe stays fully self-contained.
    """
    base = getattr(sys, "_MEIPASS", None)
    return (Path(base) / "web") if base else (Path(__file__).resolve().parent / "web")


def _read_version() -> str:
    """App's semantic version — ``pyproject.toml`` is the single source of truth.

    Running from source, ``pyproject.toml`` sits next to this module at the repo
    root. When frozen by PyInstaller it's bundled alongside ``web/`` (see the
    ``--add-data`` in the Makefile and build workflow), so the exe can still
    report its version. Falls back to ``"0+unknown"`` if it can't be read, so the
    app always starts.
    """
    base = getattr(sys, "_MEIPASS", None)
    root = Path(base) if base else Path(__file__).resolve().parent
    try:
        meta = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        return str(meta["project"]["version"])
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        return "0+unknown"


__version__ = _read_version()


# Where verdicts land when a package is opened from inside the app. The browser
# file picker can't reveal the chosen file's real folder, so we use a stable,
# predictable home-dir location keyed by the package's batch name.
SAVE_DIR = Path.home() / "Documents" / "oncai_reviews"
DEFAULT_REVIEWER = ""

# Identifies our server on /api/ping so a second launch can recognise an
# already-running instance (single-instance reconnect) rather than starting a
# duplicate. See _find_running_instance().
APP_ID = "oncai-review"


def _safe_name(name: str) -> str:
    """Filesystem-safe batch name for the reviews filename."""
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in name).strip("._")
    return safe or "review"


# ApproxDate.date is always a full YYYY-MM-DD (precision conveys granularity);
# a null date is allowed when unknown.
_APPROX_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ANCHORS = {None, "BOM", "EOM", "BOY", "EOY", "MID", "EXACT"}


def _validate_review(review: dict) -> str | None:
    """Return an error message if an edited ApproxDate is malformed, else None.

    Authoritative guard so bad data can't be persisted even if the browser
    validation is bypassed. Checks edited ApproxDate values ({"date": ...}):
    the date must be a real calendar date in YYYY-MM-DD (or null = unknown),
    and the anchor must be one of the allowed hints.
    """
    for field, value in (review.get("edits") or {}).items():
        if not (isinstance(value, dict) and "date" in value):
            continue
        date = value.get("date")
        if date is not None:
            date = str(date)
            if not _APPROX_DATE_RE.match(date):
                return (
                    f"Field '{field}' has an invalid date '{date}'. "
                    "Use YYYY-MM-DD (set precision for approximate dates)."
                )
            try:
                datetime.date.fromisoformat(date)
            except ValueError:
                return f"Field '{field}' has an impossible calendar date '{date}'."
        if value.get("anchor") not in _ANCHORS:
            return f"Field '{field}' has an invalid date anchor '{value.get('anchor')}'."
    return None


class ReviewState:
    """Loaded package + the verdicts written so far. Guarded by a lock."""

    def __init__(self, package: dict, reviews_path: Path, reviewer: str) -> None:
        self.package = package
        self.reviews_path = Path(reviews_path).expanduser().resolve()
        self.reviewer = reviewer
        self.lock = threading.Lock()
        self.reviews: dict[str, dict] = {}
        self._load_existing_reviews()

    @classmethod
    def from_path(cls, package_path: Path, reviews_path: Path, reviewer: str) -> ReviewState:
        """Build from a package file on disk (the --package / CLI path)."""
        return cls(json.loads(Path(package_path).read_text()), reviews_path, reviewer)

    def _load_existing_reviews(self) -> None:
        """Replay the append-only reviews log; last write per event_key wins."""
        if not self.reviews_path.exists():
            return
        for raw_line in self.reviews_path.read_text().splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = rec.get("event_key")
            if key:
                self.reviews[key] = rec

    def data_payload(self) -> dict:
        with self.lock:
            return {
                "loaded": True,
                "definition_name": self.package.get("definition_name"),
                "batch": self.package.get("batch"),
                "generated_at": self.package.get("generated_at"),
                "field_schema": self.package.get("field_schema", {}),
                "patients": _patients_with_reviewer_added_events(
                    self.package, self.reviews
                ),
                "reviews": self.reviews,
                "reviewer": self.reviewer,
                "reviews_path": str(self.reviews_path),
            }

    def save_review(self, review: dict) -> dict:
        """Append a verdict to the log and update memory. Returns it normalized."""
        key = review.get("event_key")
        if not key:
            raise ValueError("review is missing event_key")
        err = _validate_review(review)
        if err:
            raise ValueError(err)
        with self.lock:
            self.reviews_path.parent.mkdir(parents=True, exist_ok=True)
            with self.reviews_path.open("a") as fh:
                fh.write(json.dumps(review, default=str) + "\n")
            self.reviews[key] = review
            return {"ok": True, "reviewed_count": len(self.reviews)}


STATE: ReviewState | None = None


def _is_reviewer_added_event(review: dict) -> bool:
    """Whether a review log record represents an entity created in the app."""
    return review.get("is_new_event") is True


def _patients_with_reviewer_added_events(
    package: dict,
    reviews: dict[str, dict],
) -> list[dict]:
    """Return package patients plus saved reviewer-created events.

    Reviewer-created entities live in the append-only reviews log, not the
    immutable review package. Rehydrate them into the payload so refreshes and
    re-opens show those added cards alongside extracted events.
    """
    raw_patients = package.get("patients", [])
    patients = copy.deepcopy(raw_patients) if isinstance(raw_patients, list) else []
    known_keys = {
        str(event.get("event_key") or "")
        for patient in patients
        if isinstance(patient, dict)
        for event in patient.get("events", [])
        if isinstance(event, dict)
    }
    for review in reviews.values():
        if not _is_reviewer_added_event(review):
            continue
        event_key = str(review.get("event_key") or "")
        if not event_key or event_key in known_keys:
            continue
        patient = _patient_for_review(patients, review)
        patient.setdefault("events", []).append(_event_from_added_review(review))
        known_keys.add(event_key)
    return patients


def _patient_for_review(patients: list[dict], review: dict) -> dict:
    mrn = str(review.get("mrn") or "")
    note_id = str(review.get("note_id") or "")
    for patient in patients:
        if str(patient.get("mrn") or "") == mrn:
            return patient
        notes = patient.get("notes") or {}
        if isinstance(notes, dict) and note_id in notes:
            return patient
    patient = {"mrn": mrn or note_id or "reviewer-added", "notes": {}, "events": []}
    if note_id:
        patient["notes"][note_id] = {
            "note_text": None,
            "mrn": mrn or None,
            "note_date": None,
            "note_type": None,
            "department": None,
        }
    patients.append(patient)
    return patient


def _event_from_added_review(review: dict) -> dict:
    note_id = str(review.get("note_id") or "")
    edits = review.get("edits") or {}
    fields = {"note_id": note_id}
    if isinstance(edits, dict):
        fields.update(edits)
    return {
        "event_key": str(review.get("event_key") or ""),
        "event_type": str(review.get("event_type") or ""),
        "note_id": note_id,
        "fingerprint": "",
        "fields": fields,
        "is_new_event": True,
    }


class Handler(BaseHTTPRequestHandler):
    # Quieter logging — one line per request is plenty for a local tool.
    def log_message(self, format: str, *args: object) -> None:
        sys.stderr.write("  " + (format % args) + "\n")

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj: object, code: int = 200) -> None:
        self._send(code, json.dumps(obj, default=str).encode("utf-8"), "application/json")

    def do_GET(self) -> None:
        if self.path == "/api/ping":
            # Lightweight identity probe used for single-instance reconnect.
            self._send_json({"app": APP_ID, "version": __version__})
        elif self.path == "/api/data":
            if STATE is None:
                # No package opened yet — the UI shows the file picker.
                self._send_json({"loaded": False, "reviewer": DEFAULT_REVIEWER, "version": __version__})
            else:
                payload = STATE.data_payload()
                payload["version"] = __version__
                self._send_json(payload)
        else:
            self._serve_static(self.path)

    def _serve_static(self, path: str) -> None:
        """Serve a file from web/, guarding against path traversal."""
        if path == "/" or not path:
            path = "/index.html"
        web = _web_dir()
        target = (web / path.lstrip("/")).resolve()
        try:
            target.relative_to(web.resolve())
        except ValueError:
            self._send(403, b"forbidden", "text/plain")
            return
        if not target.is_file():
            self._send(404, b"not found", "text/plain")
            return
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        self._send(200, target.read_bytes(), ctype)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        if self.path == "/api/load":
            self._handle_load(raw)
        elif self.path == "/api/review":
            self._handle_review(raw)
        elif self.path == "/api/unload":
            self._handle_unload()
        elif self.path == "/api/quit":
            self._handle_quit()
        else:
            self._send(404, b"not found", "text/plain")

    def _handle_load(self, raw: bytes) -> None:
        """Open a review package uploaded from the browser file picker."""
        global STATE  # noqa: PLW0603 — module-level singleton is intentional for this local single-user server
        try:
            package = json.loads(raw)
        except json.JSONDecodeError as exc:
            self._send_json({"ok": False, "error": f"Not valid JSON: {exc}"}, code=400)
            return
        if not isinstance(package, dict) or "patients" not in package or "field_schema" not in package:
            self._send_json(
                {
                    "ok": False,
                    "error": "That file isn't a review package. Build one with "
                    "`oncai fc review-package <batch>` and open the .review_pkg.json it writes.",
                },
                code=400,
            )
            return
        batch = _safe_name(str(package.get("batch") or "review"))
        reviews_path = SAVE_DIR / f"{batch}.reviews.jsonl"
        STATE = ReviewState(package, reviews_path, DEFAULT_REVIEWER)
        print(f"Opened package (batch '{batch}') — reviews -> {reviews_path}")
        self._send_json(STATE.data_payload())

    def _handle_review(self, raw: bytes) -> None:
        if STATE is None:
            self._send_json({"ok": False, "error": "No package loaded"}, code=400)
            return
        try:
            review = json.loads(raw)
        except json.JSONDecodeError:
            self._send_json({"ok": False, "error": "invalid json"}, code=400)
            return
        try:
            result = STATE.save_review(review)
        except ValueError as exc:
            self._send_json({"ok": False, "error": str(exc)}, code=400)
            return
        self._send_json(result)

    def _handle_unload(self) -> None:
        """Close the active package without deleting any saved review log."""
        global STATE  # noqa: PLW0603 — module-level singleton is intentional for this local single-user server
        STATE = None
        self._send_json(
            {
                "ok": True,
                "loaded": False,
                "reviewer": DEFAULT_REVIEWER,
                "version": __version__,
            }
        )

    def _handle_quit(self) -> None:
        """Stop the server so the app fully closes.

        Lets a reviewer quit from the browser (the macOS .app and Windows build
        both run the server in a Terminal/console window; Ctrl-C there also
        works, but a button is friendlier). We answer first, then call
        ``shutdown()`` from another thread (it must not run on the serving
        thread) so ``serve_forever`` returns and the process exits cleanly.
        """
        self._send_json({"ok": True})
        threading.Thread(target=self.server.shutdown, daemon=True).start()


def _open_server(host: str, port: int) -> ThreadingHTTPServer:
    """Bind an HTTP server, picking an open port intelligently.

    Tries the requested ``port`` first, then a small range above it, and finally
    falls back to an OS-assigned ephemeral port (``0``) so the app always starts
    even when the preferred port is busy. Pass ``port=0`` to go straight to an
    OS-assigned port.
    """
    candidates = [0] if port == 0 else [port, *range(port + 1, port + 21), 0]
    last_err: OSError | None = None
    for candidate in candidates:
        try:
            return ThreadingHTTPServer((host, candidate), Handler)
        except OSError as exc:  # port in use / not permitted — try the next one
            last_err = exc
            continue
    raise OSError(f"Could not bind a port on {host}: {last_err}")


def _instance_url_if_ours(host: str, port: int) -> str | None:
    """Return ``http://host:port/`` if our app answers /api/ping there, else None."""
    url = f"http://{host}:{port}/"
    try:
        with urllib.request.urlopen(url + "api/ping", timeout=0.3) as resp:  # noqa: S310 — fixed localhost URL
            info = json.loads(resp.read())
    except (OSError, ValueError):  # refused / timeout / non-JSON — not our server
        return None
    return url if isinstance(info, dict) and info.get("app") == APP_ID else None


def _find_running_instance(host: str, port: int) -> str | None:
    """URL of an already-running oncai-review on ``host``, or None.

    Probes the preferred port and the same fallback range :func:`_open_server`
    would use, so a second launch (e.g. the reviewer re-opening the app after
    closing the browser tab) reconnects to the existing server instead of
    starting a duplicate. ``port == 0`` (always-ephemeral) opts out.
    """
    if not port:
        return None
    for candidate in [port, *range(port + 1, port + 21)]:
        url = _instance_url_if_ours(host, candidate)
        if url is not None:
            return url
    return None


def default_reviews_path(package_path: Path) -> Path:
    name = package_path.name
    if name.endswith(".review_pkg.json"):
        return package_path.with_name(name[: -len(".review_pkg.json")] + ".reviews.jsonl")
    return package_path.with_name(package_path.stem + ".reviews.jsonl")


def serve(  # noqa: PLR0913 — these are all explicit CLI-facing options, kept flat on purpose
    package_path: Path | None = None,
    *,
    reviews_path: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    reviewer: str = "",
    open_browser: bool = True,
) -> None:
    """Start the review server (blocking). Ctrl-C to stop.

    If ``package_path`` is given it's opened immediately (the ``oncai fc review``
    / ``--package`` path). If omitted, the server starts empty and the app shows
    a file picker to open a ``.review_pkg.json``; those reviews are written under
    ``SAVE_DIR`` keyed by the package's batch name.
    """
    global STATE, DEFAULT_REVIEWER  # noqa: PLW0603 — module-level singleton is intentional here
    DEFAULT_REVIEWER = reviewer

    # GUI flow (double-click, no package): if the app is already running, just
    # reopen its tab instead of starting a duplicate. This is the recovery path
    # when a reviewer closes the browser tab — re-launching brings them back to
    # their loaded package. (Skipped for the --package CLI path, which wants to
    # load that specific package into a fresh server.)
    if package_path is None:
        running = _find_running_instance(host, port)
        if running is not None:
            print(f"oncai-review is already running at {running} — reopening it.")
            print("  (Close the tab and click Quit in the app to stop it.)")
            if open_browser:
                webbrowser.open(running)
            return

    if package_path is not None:
        package_path = Path(package_path)
        if not package_path.exists():
            raise FileNotFoundError(f"Review package not found: {package_path}")
        reviews_path = Path(reviews_path) if reviews_path else default_reviews_path(package_path)
        STATE = ReviewState.from_path(package_path, reviews_path, reviewer)

    httpd = _open_server(host, port)
    actual_port = httpd.server_address[1]
    url = f"http://{host}:{actual_port}/"
    if port and actual_port != port:
        print(f"Port {port} was busy — using {actual_port} instead.")
    print(f"Review server v{__version__}: {url}")
    if STATE is not None:
        n_events = sum(len(p.get("events", [])) for p in STATE.package.get("patients", []))
        print(f"  package : {package_path}")
        print(f"  reviews : {STATE.reviews_path}")
        print(f"  patients: {len(STATE.package.get('patients', []))}   events: {n_events}")
    else:
        print("  no package opened — pick a .review_pkg.json in the browser")
        print(f"  reviews will be saved under: {SAVE_DIR}")
    print("  Quit from the button in the app, or Ctrl-C here.")
    if open_browser:
        threading.Thread(target=lambda: webbrowser.open(url), daemon=True).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        httpd.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local physician review server for FC extractions.")
    parser.add_argument(
        "--package",
        "-p",
        type=Path,
        default=None,
        help="Path to a *.review_pkg.json (optional; if omitted, pick one in the app)",
    )
    parser.add_argument(
        "--reviews", type=Path, default=None, help="Output reviews JSONL (default: alongside package)"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Preferred port; auto-falls back to an open one if busy (0 = OS-assigned)",
    )
    parser.add_argument("--reviewer", default="", help="Reviewer name stamped on verdicts")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open a browser")
    args = parser.parse_args(argv)
    serve(
        args.package,
        reviews_path=args.reviews,
        host=args.host,
        port=args.port,
        reviewer=args.reviewer,
        open_browser=not args.no_browser,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
