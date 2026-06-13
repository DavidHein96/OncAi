"""Centralized run logging for LLM-powered extraction runs.

Each run is an immutable JSON manifest at ``inbox/runs/<run_id>.run.json``: a
"started" record is written at launch and rewritten with the
completed/failed/cancelled sections when the run ends. The run owns its own
file, so the end-of-run rewrite is a safe single-writer update — no shared
parquet to clobber.

The inbox manifests are the source of truth. ``oncai ingest runs`` projects
them into ``lake/runs/runs.parquet`` (and from there the DuckDB ``runs.runs``
table) for SQL, but ``runs list/show/compare`` read the manifests directly so
they reflect in-flight runs without an ingest step.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import typing
from dataclasses import dataclass, fields
from pathlib import Path

import polars as pl

_GIT_BIN = shutil.which("git")

# Per-run manifest suffix. The inbox sync transports these like any other
# inbox file, so a run's provenance reaches the remote with the inbox push.
RUN_FILE_SUFFIX = ".run.json"


@dataclass
class RunLog:
    """One run manifest — captures everything about an LLM run."""

    # Identity
    run_id: str = ""
    run_type: str = ""  # "fc_workflow" | "fc_single" | "compression"
    name: str = ""
    batch_name: str = ""
    status: str = "started"  # "started" | "completed" | "failed" | "cancelled"

    # Timing
    started_at: str = ""
    completed_at: str | None = None
    duration_seconds: float | None = None

    # Git / Code
    git_commit: str | None = None
    git_branch: str | None = None
    git_dirty: bool | None = None
    code_version: str | None = None

    # Backend / Model
    backend: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None

    # FC single-note config
    source_table: str | None = None
    text_column: str | None = None
    workers: int | None = None
    definition_path: str | None = None

    # Prompts & Tools
    system_prompt: str | None = None
    system_prompt_hash: str | None = None
    tools_json: str | None = None
    tool_schemas_json: str | None = None

    # Input
    db_path: str | None = None
    mrn_source: str | None = None
    input_count: int = 0

    # Results
    items_processed: int = 0
    items_succeeded: int = 0
    items_failed: int = 0
    items_skipped: int = 0
    total_events: int = 0
    events_by_type_json: str | None = None

    # Tokens
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    # Output
    output_path: str | None = None
    errors_json: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Plain dict for JSON serialization / Polars construction."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict) -> RunLog:
        """Build a ``RunLog`` from a manifest dict, ignoring unknown keys.

        Missing keys fall back to dataclass defaults, so an older manifest
        without a newer field still loads (forward/backward tolerant).
        """
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def _generate_run_id(run_type: str, name: str, batch: str, started_at: str) -> str:
    """Generate an 8-char deterministic run ID."""
    key = f"{run_type}|{name}|{batch}|{started_at}"
    return hashlib.sha256(key.encode()).hexdigest()[:8]


class GitInfo:
    """Git repository information."""

    __slots__ = ("branch", "commit", "dirty")

    def __init__(self) -> None:
        self.commit: str | None = None
        self.branch: str | None = None
        self.dirty: bool | None = None


def _get_git_info() -> GitInfo:
    """Capture commit / branch / dirty state of the current repo.

    Returns an all-``None`` ``GitInfo`` if ``git`` is not on PATH or any of the
    invocations time out — the run log records what it can and never raises.
    """
    info = GitInfo()
    if _GIT_BIN is None:
        return info
    # Args are all literal subcommands and a path resolved from PATH at import.
    # No user-controlled input flows into argv here, so S603 is suppressed.
    try:
        result = subprocess.run(  # noqa: S603
            [_GIT_BIN, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            info.commit = result.stdout.strip()[:12]

        result = subprocess.run(  # noqa: S603
            [_GIT_BIN, "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            info.branch = result.stdout.strip()

        result = subprocess.run(  # noqa: S603
            [_GIT_BIN, "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            info.dirty = len(result.stdout.strip()) > 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return info


def _get_code_version() -> str | None:
    """Get the installed package version."""
    try:
        from importlib.metadata import version

        return version("oncai")
    except Exception:
        return None


def _hash_string(s: str) -> str:
    """SHA-256 prefix hash of a string."""
    return hashlib.sha256(s.encode()).hexdigest()[:16]


# --- manifest <-> parquet projection --------------------------------------


def _unwrap_optional(annotation: object) -> object:
    """Return ``X`` for an ``X | None`` annotation, else the annotation itself."""
    args = typing.get_args(annotation)
    if args:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return annotation


def _runs_schema() -> dict[str, type[pl.DataType] | pl.DataType]:
    """Derive a stable Polars schema from ``RunLog``'s field annotations.

    Pinning the dtypes keeps the projected parquet's columns predictable even
    when a field is null across every run in a batch (which would otherwise
    infer to a Null-typed column).
    """
    dtype_map: dict[type, type[pl.DataType]] = {
        str: pl.String,
        int: pl.Int64,
        float: pl.Float64,
        bool: pl.Boolean,
    }
    hints = typing.get_type_hints(RunLog)
    schema: dict[str, type[pl.DataType] | pl.DataType] = {}
    for f in fields(RunLog):
        base = _unwrap_optional(hints[f.name])
        schema[f.name] = dtype_map.get(base, pl.String)  # type: ignore[arg-type]
    return schema


_RUNS_SCHEMA = _runs_schema()


def _runs_dir(inbox_path: Path) -> Path:
    return inbox_path / "runs"


def _run_file(inbox_path: Path, run_id: str) -> Path:
    return _runs_dir(inbox_path) / f"{run_id}{RUN_FILE_SUFFIX}"


def _atomic_write_json(data: dict, path: Path) -> None:
    """Write ``data`` as JSON atomically: temp file → fsync → rename.

    A reader (or the inbox sync) never sees a half-written manifest.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    fd = os.open(tmp, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    tmp.replace(path)


def start_run(run: RunLog, inbox_path: Path) -> Path:
    """Write the "started" manifest for ``run`` to ``inbox/runs/``.

    Returns the manifest path.
    """
    path = _run_file(inbox_path, run.run_id)
    _atomic_write_json(run.to_dict(), path)
    return path


def complete_run(run_id: str, inbox_path: Path, **updates: object) -> bool:
    """Fill in the terminal sections of a run's manifest, in place.

    Reads the run's own ``<run_id>.run.json``, applies ``updates`` (restricted
    to known ``RunLog`` fields), and rewrites it atomically. Returns False if
    the manifest doesn't exist or can't be read.
    """
    path = _run_file(inbox_path, run_id)
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return False

    known = {f.name for f in fields(RunLog)}
    for key, value in updates.items():
        if key in known:
            data[key] = value

    _atomic_write_json(data, path)
    return True


def load_run_manifests(inbox_path: Path) -> list[dict]:
    """Read every run manifest in ``inbox/runs/`` as a dict (skips unreadable)."""
    runs_dir = _runs_dir(inbox_path)
    if not runs_dir.exists():
        return []
    out: list[dict] = []
    for f in sorted(runs_dir.glob(f"*{RUN_FILE_SUFFIX}")):
        try:
            out.append(json.loads(f.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    return out


def runs_to_dataframe(runs: list[dict]) -> pl.DataFrame:
    """Project a list of manifest dicts into a typed run-log DataFrame."""
    if not runs:
        return pl.DataFrame(schema=_RUNS_SCHEMA)
    rows = [RunLog.from_dict(r).to_dict() for r in runs]
    return pl.DataFrame(rows, schema=_RUNS_SCHEMA)


def list_runs(
    inbox_path: Path,
    run_type: str | None = None,
    limit: int | None = None,
) -> pl.DataFrame:
    """Read the run manifests, optionally filter by type, sort desc by started_at."""
    df = runs_to_dataframe(load_run_manifests(inbox_path))
    if df.height == 0:
        return df

    if run_type is not None:
        df = df.filter(pl.col("run_type") == run_type)

    df = df.sort("started_at", descending=True)

    if limit is not None:
        df = df.head(limit)

    return df


def get_run(inbox_path: Path, run_id: str) -> dict | None:
    """Prefix-match lookup of a run by its ID (like git short hashes)."""
    runs_dir = _runs_dir(inbox_path)
    if not runs_dir.exists():
        return None

    for f in sorted(runs_dir.glob(f"*{RUN_FILE_SUFFIX}")):
        rid = f.name[: -len(RUN_FILE_SUFFIX)]
        if rid.startswith(run_id):
            try:
                return RunLog.from_dict(json.loads(f.read_text())).to_dict()
            except (json.JSONDecodeError, OSError):
                return None
    return None
