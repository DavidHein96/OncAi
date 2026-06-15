"""SHA-256 sidecar files for inbox content.

Each inbox file gets a peer ``<filename>.sha256`` holding a small JSON record::

    {"sha256": "<hex digest>", "size": <bytes>, "mtime_ns": <int>}

The ``size``/``mtime_ns`` are a cheap freshness stamp: the cached digest is
trusted only while the file's current ``stat()`` still matches. A ``stat`` is
nearly free; hashing is not — so we keep the "don't re-hash on every call"
speedup, but the cache self-invalidates the moment a file is edited in place
(which would otherwise let a stale digest mask the change). Sidecars let
sync/push detect drift across machines without size-only heuristics.

Legacy bare-hex sidecars (digest only, no JSON) are still read; lacking stat
metadata they're treated as never-fresh, so they recompute and self-upgrade to
the JSON form on the next ``ensure_sidecar``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SIDECAR_SUFFIX = ".sha256"
_CHUNK = 1 << 20  # 1 MiB


def compute_sha256(path: Path) -> str:
    """Hex SHA-256 of the file at *path*."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(_CHUNK)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sidecar_path(file_path: Path) -> Path:
    """Path of the sidecar for *file_path* (e.g. foo.csv → foo.csv.sha256)."""
    return file_path.with_name(file_path.name + SIDECAR_SUFFIX)


def _read_record(file_path: Path) -> dict | None:
    """Parse *file_path*'s sidecar into a ``{sha256, size, mtime_ns}`` record.

    Returns None if the sidecar is missing or empty. A legacy bare-hex sidecar
    (or any non-object content) yields a record with the digest but
    ``size``/``mtime_ns`` of None, so it can never be judged fresh.
    """
    sp = sidecar_path(file_path)
    if not sp.exists():
        return None
    text = sp.read_text().strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict) and "sha256" in data:
        return {
            "sha256": data["sha256"],
            "size": data.get("size"),
            "mtime_ns": data.get("mtime_ns"),
        }
    # Legacy bare-hex (or anything else): treat the raw text as the digest.
    return {"sha256": text, "size": None, "mtime_ns": None}


def _is_fresh(file_path: Path, record: dict) -> bool:
    """True iff *record*'s stamped size + mtime match *file_path*'s current stat."""
    if record.get("size") is None or record.get("mtime_ns") is None:
        return False
    try:
        st = file_path.stat()
    except OSError:
        return False
    return st.st_size == record["size"] and st.st_mtime_ns == record["mtime_ns"]


def read_sidecar(file_path: Path) -> str | None:
    """Return the digest stored in *file_path*'s sidecar, or None.

    This is the raw stored digest — it does not stat-validate. The strict
    integrity check (``status --check``) recomputes and compares; the
    freshness-gated cache lives in ``ensure_sidecar`` / ``hash_for_compare``.
    """
    record = _read_record(file_path)
    return record["sha256"] if record is not None else None


def write_sidecar(file_path: Path, hash_hex: str) -> None:
    """Write *hash_hex* plus *file_path*'s current size/mtime to its sidecar."""
    try:
        st = file_path.stat()
        size: int | None = st.st_size
        mtime_ns: int | None = st.st_mtime_ns
    except OSError:
        size, mtime_ns = None, None
    record = {"sha256": hash_hex, "size": size, "mtime_ns": mtime_ns}
    sidecar_path(file_path).write_text(json.dumps(record) + "\n")


def ensure_sidecar(file_path: Path) -> str:
    """Return the file's digest, using a fresh sidecar or (re)computing one.

    Trusts a stored digest only while its stamped stat still matches; otherwise
    recomputes and rewrites the sidecar. (The hash is computed then the file is
    re-stat'd to stamp it — a negligible window for the inbox's immutable files.)
    """
    record = _read_record(file_path)
    if record is not None and _is_fresh(file_path, record):
        return record["sha256"]
    digest = compute_sha256(file_path)
    write_sidecar(file_path, digest)
    return digest


def hash_for_compare(file_path: Path) -> str:
    """Return the file's hash for comparison **without writing a sidecar**.

    Trusts a stored digest only while its stamped stat still matches; otherwise
    computes the digest fresh but does not persist it. Use this on read/planning
    paths — the sync conflict scan and any dry-run — so they stay
    side-effect-free (no ``.sha256`` materialised, in particular not onto a
    read-only or remote destination). ``ensure_sidecar`` is the write-through
    variant, for the actual copy path.
    """
    record = _read_record(file_path)
    if record is not None and _is_fresh(file_path, record):
        return record["sha256"]
    return compute_sha256(file_path)


def verify_sidecar(file_path: Path) -> bool:
    """True iff a sidecar exists and its digest matches the file's current bytes.

    The strong check: always recomputes, ignoring the stat stamp.
    """
    stored = read_sidecar(file_path)
    if stored is None:
        return False
    return stored == compute_sha256(file_path)
