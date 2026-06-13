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


def _clean_path_row(report_id: str, mrn: str, text: str) -> dict:
    # Already-clean: one row per report with report_text, no mult_ln_val_storage.
    return {
        "report_id": report_id,
        "mrn": mrn,
        "report_text": text,
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


class TestAlreadyCleanReports:
    def test_clean_reports_skip_collation(self, oncai_config):
        inbox = oncai_config.inbox_path / "pathology"
        inbox.mkdir(parents=True, exist_ok=True)
        # Double spaces that collation would have reflowed into newlines.
        text = "DIAGNOSIS: Clear cell RCC.  Margins negative."
        _make_path_csv(
            inbox / "2024-11-09_clean.csv",
            [_clean_path_row("A", "M1", text), _clean_path_row("B", "M2", "Benign.")],
        )

        results = run_ingest(oncai_config, folder="pathology")
        r = results[0]
        assert r.row_count == 2
        # The skip is surfaced to the user via notes.
        assert any("skipping collation" in n for n in r.notes)

        df = pl.read_parquet(
            oncai_config.lake_path / "pathology" / "pathology.parquet"
        )
        a_text = df.filter(pl.col("report_id") == "A").row(0, named=True)["report_text"]
        # Preserved verbatim: double spaces not reflowed, nothing stripped.
        assert a_text == text
        assert {"key_hash", "content_hash"}.issubset(df.columns)


class TestIngestRuns:
    def test_manifests_union_into_one_parquet(self, oncai_config):
        import json

        runs_inbox = oncai_config.inbox_path / "runs"
        runs_inbox.mkdir(parents=True, exist_ok=True)
        (runs_inbox / "aaaa1111.run.json").write_text(
            json.dumps(
                {
                    "run_id": "aaaa1111",
                    "run_type": "fc_single",
                    "name": "x",
                    "batch_name": "v1",
                    "status": "completed",
                    "started_at": "2025-01-01",
                    "items_succeeded": 5,
                }
            )
        )
        (runs_inbox / "bbbb2222.run.json").write_text(
            json.dumps(
                {
                    "run_id": "bbbb2222",
                    "run_type": "fc_single",
                    "name": "y",
                    "batch_name": "v2",
                    "status": "started",
                    "started_at": "2025-01-02",
                }
            )
        )

        results = run_ingest(oncai_config, folder="runs")
        assert len(results) == 1

        out = oncai_config.lake_path / "runs" / "runs.parquet"
        assert out.exists()
        df = pl.read_parquet(out)
        assert df.height == 2
        assert set(df["run_id"].to_list()) == {"aaaa1111", "bbbb2222"}

    def test_empty_runs_inbox_skipped(self, oncai_config):
        # An empty inbox/runs has nothing to project — dispatch returns None.
        results = run_ingest(oncai_config, folder="runs")
        assert results == []
