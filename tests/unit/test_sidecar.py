"""Tests for oncai.sidecar — SHA-256 sidecar files for inbox content."""

from __future__ import annotations

import json
from pathlib import Path

from oncai.sidecar import (
    compute_sha256,
    ensure_sidecar,
    hash_for_compare,
    read_sidecar,
    sidecar_path,
    verify_sidecar,
    write_sidecar,
)


class TestSidecarPath:
    def test_sidecar_path_appends_suffix(self):
        p = Path("/tmp/foo.csv")
        assert sidecar_path(p) == Path("/tmp/foo.csv.sha256")


class TestComputeSha256:
    def test_deterministic(self, tmp_path):
        p = tmp_path / "x.bin"
        p.write_bytes(b"hello world")
        assert compute_sha256(p) == compute_sha256(p)

    def test_changes_with_content(self, tmp_path):
        p = tmp_path / "x.bin"
        p.write_bytes(b"hello")
        h1 = compute_sha256(p)
        p.write_bytes(b"hello!")
        h2 = compute_sha256(p)
        assert h1 != h2


class TestSidecarRoundTrip:
    def test_read_returns_none_when_missing(self, tmp_path):
        p = tmp_path / "data.csv"
        p.write_text("a,b\n1,2")
        assert read_sidecar(p) is None

    def test_write_then_read(self, tmp_path):
        p = tmp_path / "data.csv"
        p.write_text("hi")
        write_sidecar(p, "deadbeef")
        assert read_sidecar(p) == "deadbeef"

    def test_ensure_creates_when_missing(self, tmp_path):
        p = tmp_path / "data.csv"
        p.write_text("hi")
        h = ensure_sidecar(p)
        assert read_sidecar(p) == h
        assert h == compute_sha256(p)

    def test_ensure_returns_stored_when_present(self, tmp_path):
        """ensure_sidecar trusts the stored hash; doesn't recompute on every call."""
        p = tmp_path / "data.csv"
        p.write_text("hi")
        write_sidecar(p, "stored_value_not_real_hash")
        assert ensure_sidecar(p) == "stored_value_not_real_hash"


class TestHashForCompare:
    def test_computes_without_writing_a_sidecar(self, tmp_path):
        p = tmp_path / "data.csv"
        p.write_text("hi")
        h = hash_for_compare(p)
        assert h == compute_sha256(p)
        # The defining property: no sidecar is left behind.
        assert not sidecar_path(p).exists()
        assert read_sidecar(p) is None

    def test_returns_stored_sidecar_when_present(self, tmp_path):
        p = tmp_path / "data.csv"
        p.write_text("hi")
        write_sidecar(p, "stored_value_not_real_hash")
        assert hash_for_compare(p) == "stored_value_not_real_hash"


class TestStatValidation:
    def test_stale_sidecar_recomputed_after_edit(self, tmp_path):
        p = tmp_path / "data.csv"
        p.write_text("original")
        stored_old = ensure_sidecar(p)

        # Edit in place WITHOUT updating the sidecar; a different size makes the
        # stat stamp drift regardless of mtime resolution.
        p.write_text("tampered-and-longer")
        fresh = compute_sha256(p)
        assert fresh != stored_old

        # The raw cached digest is stale...
        assert read_sidecar(p) == stored_old
        # ...but the freshness-gated readers detect the drift and recompute.
        assert hash_for_compare(p) == fresh
        assert ensure_sidecar(p) == fresh
        # ensure_sidecar rewrote the sidecar to the fresh digest.
        assert read_sidecar(p) == fresh

    def test_fresh_sidecar_trusted_without_recompute(self, tmp_path):
        # A digest that does NOT match the bytes is still returned while the
        # stat stamp is fresh — proving the fast path trusts the cache.
        p = tmp_path / "data.csv"
        p.write_text("hi")
        write_sidecar(p, "not_the_real_hash")  # stamps current size/mtime
        assert hash_for_compare(p) == "not_the_real_hash"
        assert ensure_sidecar(p) == "not_the_real_hash"

    def test_legacy_bare_hex_sidecar_self_upgrades(self, tmp_path):
        # An old-format sidecar (bare digest, no JSON) has no stat stamp, so it
        # is treated as never-fresh: read still works, reads recompute, and
        # ensure_sidecar upgrades it to the JSON form.
        p = tmp_path / "data.csv"
        p.write_text("hello")
        sidecar_path(p).write_text("deadbeefdeadbeef\n")  # legacy format

        assert read_sidecar(p) == "deadbeefdeadbeef"  # still readable
        real = compute_sha256(p)
        assert hash_for_compare(p) == real  # recomputed, not trusted
        assert ensure_sidecar(p) == real

        record = json.loads(sidecar_path(p).read_text())
        assert record["sha256"] == real
        assert record["size"] is not None


class TestVerifySidecar:
    def test_no_sidecar_means_unverified(self, tmp_path):
        p = tmp_path / "data.csv"
        p.write_text("hi")
        assert verify_sidecar(p) is False

    def test_matching_sidecar_verifies(self, tmp_path):
        p = tmp_path / "data.csv"
        p.write_text("hi")
        ensure_sidecar(p)
        assert verify_sidecar(p) is True

    def test_mismatched_sidecar_fails(self, tmp_path):
        p = tmp_path / "data.csv"
        p.write_text("original")
        ensure_sidecar(p)
        p.write_text("tampered")
        assert verify_sidecar(p) is False
