"""Tests for oncai.lake — inbox sync/push, inbox listing, and status."""

from __future__ import annotations

import polars as pl
import pytest

from oncai.lake import (
    SyncConflictError,
    get_inbox_files,
    get_lake_status,
    pull_inbox_from_remote,
    push_inbox_to_remote,
)
from oncai.sidecar import sidecar_path


def _result_for(results, folder):
    for r in results:
        if r.folder == folder:
            return r
    raise AssertionError(f"No result for folder: {folder}")


# ---------------------------------------------------------------------------
# pull_inbox_from_remote
# ---------------------------------------------------------------------------


class TestPullInboxFromRemote:
    def test_syncs_inbox_files_with_sidecars(self, oncai_config):
        remote_inbox = oncai_config.remote_path / "inbox" / "pathology"
        remote_inbox.mkdir(parents=True, exist_ok=True)
        (remote_inbox / "2024-11-09_path.csv").write_text("col1,col2\na,b")
        (remote_inbox / "2024-11-09_records.jsonl").write_text('{"id": 1}\n')

        results = pull_inbox_from_remote(oncai_config, folders=["pathology"])
        r = _result_for(results, "pathology")
        assert r.inbox_copied == 2

        local_inbox = oncai_config.inbox_path / "pathology"
        for name in ("2024-11-09_path.csv", "2024-11-09_records.jsonl"):
            assert (local_inbox / name).exists()
            assert sidecar_path(local_inbox / name).exists()

    def test_syncs_run_manifests(self, oncai_config):
        remote_runs = oncai_config.remote_path / "inbox" / "runs"
        remote_runs.mkdir(parents=True, exist_ok=True)
        (remote_runs / "abcd1234.run.json").write_text('{"run_id": "abcd1234"}')

        results = pull_inbox_from_remote(oncai_config, folders=["runs"])
        assert _result_for(results, "runs").inbox_copied == 1
        assert (oncai_config.inbox_path / "runs" / "abcd1234.run.json").exists()

    def test_does_not_pull_lake_parquets(self, oncai_config):
        # A parquet in the remote *lake* folder must NOT be pulled — the lake is
        # rebuilt locally via ingest, never transferred.
        remote_lake = oncai_config.remote_path / "pathology"
        remote_lake.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"report_id": ["R1"]}).write_parquet(
            remote_lake / "pathology.parquet"
        )

        pull_inbox_from_remote(oncai_config, folders=["pathology"])

        assert not (
            oncai_config.lake_path / "pathology" / "pathology.parquet"
        ).exists()

    def test_idempotent_resync(self, oncai_config):
        remote_inbox = oncai_config.remote_path / "inbox" / "pathology"
        remote_inbox.mkdir(parents=True, exist_ok=True)
        (remote_inbox / "2024-11-09_path.csv").write_text("col1,col2\na,b")

        pull_inbox_from_remote(oncai_config, folders=["pathology"])

        results = pull_inbox_from_remote(oncai_config, folders=["pathology"])
        assert _result_for(results, "pathology").inbox_copied == 0

    def test_hash_mismatch_raises(self, oncai_config):
        remote_inbox = oncai_config.remote_path / "inbox" / "pathology"
        local_inbox = oncai_config.inbox_path / "pathology"
        remote_inbox.mkdir(parents=True, exist_ok=True)
        local_inbox.mkdir(parents=True, exist_ok=True)

        # Same name, different content on each side
        (remote_inbox / "2024-11-09_path.csv").write_text("REMOTE")
        (local_inbox / "2024-11-09_path.csv").write_text("LOCAL")

        with pytest.raises(SyncConflictError) as excinfo:
            pull_inbox_from_remote(oncai_config, folders=["pathology"])
        assert len(excinfo.value.conflicts) == 1
        c = excinfo.value.conflicts[0]
        assert c.filename == "2024-11-09_path.csv"
        assert c.direction == "pull"


# ---------------------------------------------------------------------------
# push_inbox_to_remote
# ---------------------------------------------------------------------------


class TestPushInboxToRemote:
    def test_pushes_inbox_with_sidecars(self, oncai_config):
        local_inbox = oncai_config.inbox_path / "pathology"
        local_inbox.mkdir(parents=True, exist_ok=True)
        (local_inbox / "2024-11-09_path.csv").write_text("a,b\n1,2")

        results = push_inbox_to_remote(oncai_config, folders=["pathology"])
        r = _result_for(results, "pathology")
        assert r.inbox_copied == 1

        remote_inbox = oncai_config.remote_path / "inbox" / "pathology"
        assert (remote_inbox / "2024-11-09_path.csv").exists()
        assert (remote_inbox / "2024-11-09_path.csv.sha256").exists()

    def test_pushes_run_manifests(self, oncai_config):
        local_runs = oncai_config.inbox_path / "runs"
        local_runs.mkdir(parents=True, exist_ok=True)
        (local_runs / "abcd1234.run.json").write_text('{"run_id": "abcd1234"}')

        results = push_inbox_to_remote(oncai_config, folders=["runs"])
        assert _result_for(results, "runs").inbox_copied == 1
        assert (
            oncai_config.remote_path / "inbox" / "runs" / "abcd1234.run.json"
        ).exists()

    def test_does_not_push_lake_parquets(self, oncai_config):
        lake = oncai_config.lake_path / "pathology"
        lake.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"report_id": ["R1"]}).write_parquet(lake / "pathology.parquet")

        push_inbox_to_remote(oncai_config, folders=["pathology"])

        assert not (
            oncai_config.remote_path / "pathology" / "pathology.parquet"
        ).exists()


# ---------------------------------------------------------------------------
# sidecar side effects (planning paths must not write)
# ---------------------------------------------------------------------------


class TestSidecarSideEffects:
    def test_dry_run_writes_no_sidecars(self, oncai_config):
        remote_inbox = oncai_config.remote_path / "inbox" / "pathology"
        local_inbox = oncai_config.inbox_path / "pathology"
        remote_inbox.mkdir(parents=True, exist_ok=True)
        local_inbox.mkdir(parents=True, exist_ok=True)
        # Same file on both sides (matching content), no sidecars yet.
        (remote_inbox / "2024-11-09_path.csv").write_text("same")
        (local_inbox / "2024-11-09_path.csv").write_text("same")

        pull_inbox_from_remote(oncai_config, folders=["pathology"], dry_run=True)

        # The planning pass must not have materialised any .sha256 sidecar.
        assert not sidecar_path(remote_inbox / "2024-11-09_path.csv").exists()
        assert not sidecar_path(local_inbox / "2024-11-09_path.csv").exists()

    def test_conflict_scan_does_not_write_to_remote(self, oncai_config):
        # A real push's conflict pre-scan must not write sidecars onto the
        # remote before the push is judged safe. A matching file on both sides
        # needs no copy, so nothing — not even a sidecar — should land remotely.
        local_inbox = oncai_config.inbox_path / "pathology"
        remote_inbox = oncai_config.remote_path / "inbox" / "pathology"
        local_inbox.mkdir(parents=True, exist_ok=True)
        remote_inbox.mkdir(parents=True, exist_ok=True)
        (local_inbox / "2024-11-09_path.csv").write_text("same")
        (remote_inbox / "2024-11-09_path.csv").write_text("same")

        push_inbox_to_remote(oncai_config, folders=["pathology"])

        assert not sidecar_path(remote_inbox / "2024-11-09_path.csv").exists()


# ---------------------------------------------------------------------------
# get_inbox_files
# ---------------------------------------------------------------------------


class TestGetInboxFiles:
    def test_finds_csv_and_jsonl(self, oncai_config):
        inbox = oncai_config.inbox_path / "pathology"
        inbox.mkdir(parents=True, exist_ok=True)
        (inbox / "data.csv").write_text("a,b\n1,2")
        (inbox / "data.jsonl").write_text('{"x":1}\n')

        result = get_inbox_files(oncai_config)

        assert "pathology" in result
        names = {p.name for p in result["pathology"]}
        assert "data.csv" in names
        assert "data.jsonl" in names

    def test_finds_run_manifests(self, oncai_config):
        runs = oncai_config.inbox_path / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        (runs / "abcd1234.run.json").write_text('{"run_id": "abcd1234"}')

        result = get_inbox_files(oncai_config)
        assert "runs" in result
        assert {p.name for p in result["runs"]} == {"abcd1234.run.json"}

    def test_empty_inbox(self, oncai_config):
        result = get_inbox_files(oncai_config)
        assert result == {}


# ---------------------------------------------------------------------------
# get_lake_status
# ---------------------------------------------------------------------------


class TestGetLakeStatus:
    def test_returns_file_and_row_counts(self, oncai_config):
        lake_path = oncai_config.lake_path / "pathology"
        lake_path.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"report_id": ["R1", "R2", "R3"]}).write_parquet(
            lake_path / "pathology.parquet"
        )

        status = get_lake_status(oncai_config)

        assert "pathology" in status
        assert status["pathology"]["files"] == 1
        assert status["pathology"]["rows"] == 3
