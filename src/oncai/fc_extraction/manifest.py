"""
Helpers for FC extraction run manifests.

Single-note batch runs in ``batch_single.py`` build their own manifest dict
inline. The helpers here are shared utilities — git introspection, package
version lookup, and a short hash function for prompt fingerprinting.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from typing import Any

# Engine-provided tools that aren't part of a definition's extraction contract —
# excluded from the contract hash and tool-schema provenance so they don't make
# every record look "changed" when the engine's builtins shift.
_BUILTIN_TOOLS = frozenset(
    {"finish_note_extraction", "stop_workflow", "finish_single_extraction"}
)


def _run_git(*args: str) -> str | None:
    """Run ``git <args>`` and return stripped stdout, or None on any failure.

    Used for best-effort metadata capture — we don't care *why* git is
    unavailable (not installed, not a repo, timeout), only that we couldn't
    read the value. Callers treat ``None`` as "unknown".
    """
    # Best-effort version-control introspection; ``git`` on PATH is the
    # standard invocation and ``args`` come from hardcoded call sites.
    try:
        result = subprocess.run(  # noqa: S603
            ["git", *args],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def get_git_info() -> dict[str, Any]:
    """Get current git repository information.

    Returns a dict with ``commit`` (short SHA), ``branch``, and ``dirty``
    (bool). Each field is ``None`` when unavailable.
    """
    commit = _run_git("rev-parse", "HEAD")
    branch = _run_git("rev-parse", "--abbrev-ref", "HEAD")
    status = _run_git("status", "--porcelain")
    return {
        "commit": commit[:12] if commit else None,
        "branch": branch,
        "dirty": (len(status) > 0) if status is not None else None,
    }


def get_code_version() -> str | None:
    """Get the package version."""
    try:
        from importlib.metadata import version

        return version("oncai")
    except Exception:
        return None


def hash_string(s: str) -> str:
    """Create a short hash of a string."""
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def tool_schemas_for(registry: Any) -> dict[str, Any]:
    """JSON schema of each task tool (engine builtins excluded).

    The schema reflects the tool's Pydantic model — its fields, types, and enum
    options — so it captures the part of the extraction contract that ``hash_string``
    of the prompt alone misses.
    """
    out: dict[str, Any] = {}
    for name in registry.list_tools():
        if name in _BUILTIN_TOOLS:
            continue
        tool_def = registry.get(name)
        if tool_def is not None:
            out[name] = tool_def.model.model_json_schema()
    return out


def definition_hash(system_prompt: str, tool_schemas: dict[str, Any]) -> str:
    """Hash the full extraction contract: system prompt + tool JSON schemas.

    Unlike a prompt-only hash, this changes whenever a tool's Pydantic model
    changes — a field added/removed/renamed, an enum's options edited — so the
    incremental delta's change-detection re-extracts on *field* changes, not
    just prompt edits. Schemas are serialised with sorted keys for stability.
    """
    payload = json.dumps(
        {"system_prompt": system_prompt, "tool_schemas": tool_schemas},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def definition_hash_from_registry(system_prompt: str, registry: Any) -> str:
    """``definition_hash`` over a registry's tools — the single source of truth.

    Both the writer (run_meta, stamped per record) and the reader (the delta's
    change-detection) call this, so the two sides always hash identically.
    """
    return definition_hash(system_prompt, tool_schemas_for(registry))
