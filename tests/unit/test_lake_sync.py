"""Tests for oncai.lake — sync, push, inbox, and status."""

from __future__ import annotations

import polars as pl
import pytest

from oncai.lake import (
    SyncConflictError,
    get_inbox_files,
    get_lake_status,
    push_lake_to_remote,
    sync_remote_to_lake,
)
from oncai.sidecar import sidecar_path


def _result_for(results, folder):
    for r in results:
        if r.folder == folder:
            return r
    raise AssertionError(f"No result for folder: {folder}")


# ---------------------------------------------------------------------------
# sync_remote_to_lake
# ---------------------------------------------------------------------------


class TestSyncRemoteToLake:
    def test_copies_parquets(self, oncai_config):
        remote_path = oncai_config.remote_path / "pathology"
        remote_path.mkdir(parents=True, exist_ok=True)
        df = pl.DataFrame({"report_id": ["R1"], "mrn": ["M1"]})
        df.write_parquet(remote_path / "path.parquet")

        results = sync_remote_to_lake(oncai_config, folders=["pathology"])

        r = _result_for(results, "pathology")
        assert r.lake_copied == 1
        assert r.inbox_copied == 0
        lake_file = oncai_config.lake_path / "pathology" / "path.parquet"
        assert lake_file.exists()

    def test_skips_up_to_date(self, oncai_config):
        remote_path = oncai_config.remote_path / "pathology"
        remote_path.mkdir(parents=True, exist_ok=True)
        df = pl.DataFrame({"report_id": ["R1"]})
        df.write_parquet(remote_path / "path.parquet")

        sync_remote_to_lake(oncai_config, folders=["pathology"])

        results = sync_remote_to_lake(oncai_config, folders=["pathology"])
        r = _result_for(results, "pathology")
        assert r.lake_copied == 0

    def test_syncs_inbox_files_with_sidecars(self, oncai_config):
        remote_inbox = oncai_config.remote_path / "inbox" / "pathology"
        remote_inbox.mkdir(parents=True, exist_ok=True)
        (remote_inbox / "2024-11-09_path.csv").write_text("col1,col2\na,b")
        (remote_inbox / "2024-11-09_records.jsonl").write_text('{"id": 1}\n')

        results = sync_remote_to_lake(oncai_config, folders=["pathology"])
        r = _result_for(results, "pathology")
        assert r.inbox_copied == 2

        local_inbox = oncai_config.inbox_path / "pathology"
        for name in ("2024-11-09_path.csv", "2024-11-09_records.jsonl"):
            assert (local_inbox / name).exists()
            assert sidecar_path(local_inbox / name).exists()

    def test_idempotent_resync(self, oncai_config):
        remote_inbox = oncai_config.remote_path / "inbox" / "pathology"
        remote_inbox.mkdir(parents=True, exist_ok=True)
        (remote_inbox / "2024-11-09_path.csv").write_text("col1,col2\na,b")

        sync_remote_to_lake(oncai_config, folders=["pathology"])

        results = sync_remote_to_lake(oncai_config, folders=["pathology"])
        r = _result_for(results, "pathology")
        assert r.inbox_copied == 0

    def test_hash_mismatch_raises(self, oncai_config):
        remote_inbox = oncai_config.remote_path / "inbox" / "pathology"
        local_inbox = oncai_config.inbox_path / "pathology"
        remote_inbox.mkdir(parents=True, exist_ok=True)
        local_inbox.mkdir(parents=True, exist_ok=True)

        # Same name, different content on each side
        (remote_inbox / "2024-11-09_path.csv").write_text("REMOTE")
        (local_inbox / "2024-11-09_path.csv").write_text("LOCAL")

        with pytest.raises(SyncConflictError) as excinfo:
            sync_remote_to_lake(oncai_config, folders=["pathology"])
        assert len(excinfo.value.conflicts) == 1
        c = excinfo.value.conflicts[0]
        assert c.filename == "2024-11-09_path.csv"
        assert c.direction == "pull"


# ---------------------------------------------------------------------------
# push_lake_to_remote
# ---------------------------------------------------------------------------


class TestPushLakeToRemote:
    def test_copies_to_remote(self, oncai_config):
        lake_path = oncai_config.lake_path / "pathology"
        lake_path.mkdir(parents=True, exist_ok=True)
        df = pl.DataFrame({"report_id": ["R1"]})
        df.write_parquet(lake_path / "path.parquet")

        results = push_lake_to_remote(oncai_config, folders=["pathology"])

        r = _result_for(results, "pathology")
        assert r.lake_copied == 1
        assert (oncai_config.remote_path / "pathology" / "path.parquet").exists()

    def test_skips_up_to_date(self, oncai_config):
        lake_path = oncai_config.lake_path / "pathology"
        lake_path.mkdir(parents=True, exist_ok=True)
        df = pl.DataFrame({"report_id": ["R1"]})
        df.write_parquet(lake_path / "path.parquet")

        push_lake_to_remote(oncai_config, folders=["pathology"])

        results = push_lake_to_remote(oncai_config, folders=["pathology"])
        r = _result_for(results, "pathology")
        assert r.lake_copied == 0

    def test_pushes_inbox_with_sidecars(self, oncai_config):
        local_inbox = oncai_config.inbox_path / "pathology"
        local_inbox.mkdir(parents=True, exist_ok=True)
        (local_inbox / "2024-11-09_path.csv").write_text("a,b\n1,2")

        results = push_lake_to_remote(oncai_config, folders=["pathology"])
        r = _result_for(results, "pathology")
        assert r.inbox_copied == 1

        remote_inbox = oncai_config.remote_path / "inbox" / "pathology"
        assert (remote_inbox / "2024-11-09_path.csv").exists()
        assert (remote_inbox / "2024-11-09_path.csv.sha256").exists()


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
