"""Tests for oncai.db — database build and info."""

from __future__ import annotations

import polars as pl

from oncai.config import OncaiConfig
from oncai.db import build_database, get_database_info


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
