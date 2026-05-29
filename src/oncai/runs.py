"""Centralized run logging for LLM-powered extraction and compression runs.

Stores one row per run in lake/runs/runs.parquet, queryable via DuckDB.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import dataclass, fields
from pathlib import Path

import polars as pl

_GIT_BIN = shutil.which("git")


@dataclass
class RunLog:
    """One row in the run log parquet — captures everything about an LLM run."""

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


def _runlog_to_dict(run: RunLog) -> dict:
    """Convert RunLog to a plain dict for Polars."""
    return {f.name: getattr(run, f.name) for f in fields(run)}


def log_run(run_log: RunLog, lake_path: Path) -> Path:
    """Append a RunLog to lake/runs/runs.parquet (read-concat-rewrite)."""
    runs_dir = lake_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = runs_dir / "runs.parquet"

    new_df = pl.DataFrame([_runlog_to_dict(run_log)])

    if parquet_path.exists():
        existing = pl.read_parquet(parquet_path)
        combined = pl.concat([existing, new_df], how="diagonal_relaxed")
    else:
        combined = new_df

    combined.write_parquet(parquet_path)
    return parquet_path


def update_run(run_id: str, lake_path: Path, **updates) -> bool:
    """Update fields on an existing run row, matched by run_id.

    Returns True if the run was found and updated, False otherwise.
    """
    parquet_path = lake_path / "runs" / "runs.parquet"
    if not parquet_path.exists():
        return False

    df = pl.read_parquet(parquet_path)
    if df.filter(pl.col("run_id") == run_id).height == 0:
        return False

    for key, value in updates.items():
        if key not in df.columns:
            continue
        df = df.with_columns(
            pl.when(pl.col("run_id") == run_id)
            .then(pl.lit(value))
            .otherwise(pl.col(key))
            .alias(key)
        )

    df.write_parquet(parquet_path)
    return True


def list_runs(
    lake_path: Path,
    run_type: str | None = None,
    limit: int | None = None,
) -> pl.DataFrame:
    """Read the run log, optionally filter by type, sort desc by started_at."""
    parquet_path = lake_path / "runs" / "runs.parquet"
    if not parquet_path.exists():
        return pl.DataFrame()

    df = pl.read_parquet(parquet_path)

    if run_type is not None:
        df = df.filter(pl.col("run_type") == run_type)

    df = df.sort("started_at", descending=True)

    if limit is not None:
        df = df.head(limit)

    return df


def get_run(lake_path: Path, run_id: str) -> dict | None:
    """Prefix-match lookup of a run by its ID (like git short hashes)."""
    parquet_path = lake_path / "runs" / "runs.parquet"
    if not parquet_path.exists():
        return None

    df = pl.read_parquet(parquet_path)
    matches = df.filter(pl.col("run_id").str.starts_with(run_id))

    if matches.height == 0:
        return None

    # Return first match (should be unique for 8-char IDs)
    return matches.row(0, named=True)
