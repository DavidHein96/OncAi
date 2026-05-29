"""SHA-256 sidecar files for inbox content.

Each inbox file gets a peer ``<filename>.sha256`` containing the hex digest of
the file's contents on a single line. Sidecars let sync/push detect drift
across machines without size-only heuristics.
"""

from __future__ import annotations

import hashlib
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


def read_sidecar(file_path: Path) -> str | None:
    """Return the hex digest stored in *file_path*'s sidecar, or None."""
    sp = sidecar_path(file_path)
    if not sp.exists():
        return None
    text = sp.read_text().strip()
    return text or None


def write_sidecar(file_path: Path, hash_hex: str) -> None:
    """Write *hash_hex* to *file_path*'s sidecar."""
    sidecar_path(file_path).write_text(hash_hex + "\n")


def ensure_sidecar(file_path: Path) -> str:
    """Read the sidecar; if missing, compute the hash, write it, and return it."""
    existing = read_sidecar(file_path)
    if existing is not None:
        return existing
    digest = compute_sha256(file_path)
    write_sidecar(file_path, digest)
    return digest


def verify_sidecar(file_path: Path) -> bool:
    """True iff a sidecar exists and matches the file's current contents."""
    stored = read_sidecar(file_path)
    if stored is None:
        return False
    return stored == compute_sha256(file_path)
