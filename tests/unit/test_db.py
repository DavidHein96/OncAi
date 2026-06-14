"""Tests for oncai.db — database build and info."""

from __future__ import annotations

import polars as pl

from oncai.config import OncaiConfig
from oncai.db import build_database, get_database_info, update_database_folder


class TestBuildDatabase:
    def test_creates_pathology_table(self, tmp_path):
        lake = tmp_path / "lake"
        (lake / "pathology").mkdir(parents=True)

        pl.DataFrame(
            {"report_id": ["R1"], "mrn": ["M1"], "report_text": ["Hi"]}
        ).write_parquet(lake / "pathology" / "pathology.parquet")

        config = OncaiConfig(
            remote_path=tmp_path / "remote",
            lake_path=lake,
            inbox_path=tmp_path / "inbox",
            db_path=tmp_path / "test.duckdb",
        )
        results = build_database(config)

        assert "raw.pathology" in results
        assert results["raw.pathology"] == 1

    def test_force_recreates(self, tmp_path):
        lake = tmp_path / "lake"
        (lake / "pathology").mkdir(parents=True)
        pl.DataFrame({"report_id": ["R1"]}).write_parquet(
            lake / "pathology" / "pathology.parquet"
        )

        db_path = tmp_path / "test.duckdb"
        config = OncaiConfig(
            remote_path=tmp_path / "remote",
            lake_path=lake,
            inbox_path=tmp_path / "inbox",
            db_path=db_path,
        )

        build_database(config)
        assert db_path.exists()

        results = build_database(config, force=True)
        assert "raw.pathology" in results
        assert db_path.exists()

    def test_creates_gold_review_table(self, tmp_path):
        lake = tmp_path / "lake"
        (lake / "fc_reviews").mkdir(parents=True)
        pl.DataFrame(
            {
                "event_key": ["N1::record_diagnosis::0"],
                "event_type": ["record_diagnosis"],
                "note_id": ["N1"],
                "mrn": ["MRN001"],
                "review_verdict": ["approved"],
            }
        ).write_parquet(lake / "fc_reviews" / "demo.parquet")

        config = OncaiConfig(
            remote_path=tmp_path / "remote",
            lake_path=lake,
            inbox_path=tmp_path / "inbox",
            db_path=tmp_path / "test.duckdb",
        )
        results = build_database(config)

        assert results["extractions_silver.demo"] == 1

    def test_build_database_drops_stale_owned_tables(self, tmp_path):
        lake = tmp_path / "lake"
        reviews = lake / "fc_reviews"
        reviews.mkdir(parents=True)
        pl.DataFrame({"event_key": ["A"]}).write_parquet(reviews / "demo.parquet")

        config = OncaiConfig(
            remote_path=tmp_path / "remote",
            lake_path=lake,
            inbox_path=tmp_path / "inbox",
            db_path=tmp_path / "test.duckdb",
        )
        build_database(config)
        assert "extractions_silver.demo" in get_database_info(config)

        (reviews / "demo.parquet").unlink()
        build_database(config)

        assert "extractions_silver.demo" not in get_database_info(config)

    def test_update_database_folder_drops_missing_multi_table_parquet(self, tmp_path):
        lake = tmp_path / "lake"
        raw = lake / "fc_extractions"
        raw.mkdir(parents=True)
        pl.DataFrame({"record_id": ["A"]}).write_parquet(raw / "a.parquet")
        pl.DataFrame({"record_id": ["B"]}).write_parquet(raw / "b.parquet")

        config = OncaiConfig(
            remote_path=tmp_path / "remote",
            lake_path=lake,
            inbox_path=tmp_path / "inbox",
            db_path=tmp_path / "test.duckdb",
        )
        build_database(config)
        assert {"extractions_raw.a", "extractions_raw.b"}.issubset(
            get_database_info(config)
        )

        (raw / "b.parquet").unlink()
        result = update_database_folder(config, "fc_extractions")

        info = get_database_info(config)
        assert "extractions_raw.a" in info
        assert "extractions_raw.b" not in info
        # The dropped table is reported as a removal, not a zero-row update.
        assert result.dropped == ["extractions_raw.b"]
        assert "extractions_raw.b" not in result.updated
        assert result.updated["extractions_raw.a"] == 1


class TestGetDatabaseInfo:
    def test_returns_table_info(self, tmp_path):
        lake = tmp_path / "lake"
        (lake / "pathology").mkdir(parents=True)
        pl.DataFrame(
            {"report_id": ["R1", "R2"], "report_text": ["a", "b"]}
        ).write_parquet(lake / "pathology" / "pathology.parquet")

        db_path = tmp_path / "test.duckdb"
        config = OncaiConfig(
            remote_path=tmp_path / "remote",
            lake_path=lake,
            inbox_path=tmp_path / "inbox",
            db_path=db_path,
        )
        build_database(config)

        info = get_database_info(config)
        assert "raw.pathology" in info
        assert info["raw.pathology"]["rows"] == 2
        assert info["raw.pathology"]["columns"] == 2

    def test_empty_when_no_db(self, tmp_path):
        config = OncaiConfig(
            remote_path=tmp_path / "remote",
            lake_path=tmp_path / "lake",
            inbox_path=tmp_path / "inbox",
            db_path=tmp_path / "nonexistent.duckdb",
        )
        assert get_database_info(config) == {}
