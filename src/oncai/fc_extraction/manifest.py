"""
Helpers for FC extraction run manifests.

Single-note batch runs in ``batch_single.py`` build their own manifest dict
inline. The helpers here are shared utilities — git introspection, package
version lookup, and a short hash function for prompt fingerprinting.
"""

from __future__ import annotations

import hashlib
import subprocess
from typing import Any


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
