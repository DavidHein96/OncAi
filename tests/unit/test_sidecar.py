"""Tests for oncai.sidecar — SHA-256 sidecar files for inbox content."""

from __future__ import annotations

from pathlib import Path

from oncai.sidecar import (
    compute_sha256,
    ensure_sidecar,
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
