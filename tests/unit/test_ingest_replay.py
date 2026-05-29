"""Tests for oncai.ingest — dated replay-from-scratch and validation."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from oncai.ingest import run_ingest
from oncai.sidecar import sidecar_path


def _make_path_csv(path: Path, rows: list[dict]) -> None:
    pl.DataFrame(rows).write_csv(path)


def _path_row(report_id: str, mrn: str, text: str, row_id: int = 1) -> dict:
    return {
        "report_id": report_id,
        "row_id": row_id,
        "mrn": mrn,
        "mult_ln_val_storage": text,
        "ordering_date": "2024-01-15",
    }


class TestDatedReplay:
    def test_replay_two_files_in_date_order(self, oncai_config):
        inbox = oncai_config.inbox_path / "pathology"
        inbox.mkdir(parents=True, exist_ok=True)

        _make_path_csv(
            inbox / "2024-11-09_path.csv",
            [
                _path_row("A", "M1", "old A"),
                _path_row("B", "M2", "B text"),
            ],
        )
        _make_path_csv(
            inbox / "2024-12-09_path.csv",
            [
                _path_row("A", "M1", "new A"),  # updates A
                _path_row("C", "M3", "C text"),
            ],
        )

        results = run_ingest(oncai_config, folder="pathology")
        r = results[0]
        assert r.row_count == 3
        assert len(r.files) == 2

        # First file: 2 new
        assert r.files[0].name == "2024-11-09_path.csv"
        assert r.files[0].new_rows == 2
        # Second file: 1 new (C), 1 updated (A)
        assert r.files[1].name == "2024-12-09_path.csv"
        assert r.files[1].new_rows == 1
        assert r.files[1].updated_rows == 1

        # Final parquet should have the newer A.
        df = pl.read_parquet(oncai_config.lake_path / "pathology" / "pathology.parquet")
        a_row = df.filter(pl.col("report_id") == "A").row(0, named=True)
        assert a_row["report_text"] == "new A"

    def test_replay_is_idempotent_on_disk(self, oncai_config):
        inbox = oncai_config.inbox_path / "pathology"
        inbox.mkdir(parents=True, exist_ok=True)
        _make_path_csv(inbox / "2024-11-09_path.csv", [_path_row("A", "M1", "x")])

        run_ingest(oncai_config, folder="pathology")
        first = (
            oncai_config.lake_path / "pathology" / "pathology.parquet"
        ).read_bytes()

        run_ingest(oncai_config, folder="pathology")
        second = (
            oncai_config.lake_path / "pathology" / "pathology.parquet"
        ).read_bytes()

        # Replay-from-scratch produces the same result; output bytes match.
        assert first == second

    def test_creates_sidecars(self, oncai_config):
        inbox = oncai_config.inbox_path / "pathology"
        inbox.mkdir(parents=True, exist_ok=True)
        csv = inbox / "2024-11-09_path.csv"
        _make_path_csv(csv, [_path_row("A", "M1", "x")])

        assert not sidecar_path(csv).exists()
        run_ingest(oncai_config, folder="pathology")
        assert sidecar_path(csv).exists()


class TestDatedFilenameValidation:
    def test_non_iso_filename_rejected(self, oncai_config):
        inbox = oncai_config.inbox_path / "pathology"
        inbox.mkdir(parents=True, exist_ok=True)
        _make_path_csv(inbox / "stray.csv", [_path_row("A", "M1", "x")])

        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            run_ingest(oncai_config, folder="pathology")

    def test_lists_all_offenders(self, oncai_config):
        inbox = oncai_config.inbox_path / "pathology"
        inbox.mkdir(parents=True, exist_ok=True)
        _make_path_csv(inbox / "stray.csv", [_path_row("A", "M1", "x")])
        _make_path_csv(inbox / "another_one.csv", [_path_row("A", "M1", "x")])
        _make_path_csv(inbox / "2024-11-09_path.csv", [_path_row("A", "M1", "x")])

        with pytest.raises(ValueError) as excinfo:
            run_ingest(oncai_config, folder="pathology")
        # Both stray names should appear in the error message
        assert "stray.csv" in str(excinfo.value)
        assert "another_one.csv" in str(excinfo.value)
        # The valid file shouldn't be listed as an offender
        assert "2024-11-09_path.csv" not in str(excinfo.value).split("Offenders:")[1]


class TestLakeOnlyFolders:
    def test_lake_only_skipped(self, oncai_config):
        inbox = oncai_config.inbox_path / "runs"
        inbox.mkdir(parents=True, exist_ok=True)
        # Drop something that would otherwise trigger ingest
        (inbox / "anything.parquet").write_bytes(b"")

        results = run_ingest(oncai_config, folder="runs")
        assert results == []
