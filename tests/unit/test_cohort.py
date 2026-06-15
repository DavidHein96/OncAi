"""Tests for oncai.cohort module."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from oncai.cohort import add_cohort, get_cohort_info, list_cohorts, remove_cohort


@pytest.fixture
def lake_path(tmp_path) -> Path:
    lake = tmp_path / "lake"
    lake.mkdir()
    return lake


@pytest.fixture
def cohort_csv(tmp_path) -> Path:
    csv_path = tmp_path / "patients.csv"
    df = pl.DataFrame(
        {
            "mrn": ["MRN001", "MRN002", "MRN003"],
            "group": ["control", "treatment", "treatment"],
        }
    )
    df.write_csv(csv_path)
    return csv_path


class TestAddCohort:
    def test_creates_parquet_and_sidecar(self, cohort_csv, lake_path):
        metadata = add_cohort(
            csv_path=cohort_csv,
            lake_path=lake_path,
            name="test_cohort",
            key_column="mrn",
            description="Test cohort",
        )
        assert metadata.name == "test_cohort"
        assert metadata.row_count == 3
        assert metadata.key_column == "mrn"
        assert (lake_path / "cohorts" / "test_cohort.parquet").exists()
        assert (lake_path / "cohorts" / "test_cohort.cohort.json").exists()

    def test_invalid_key_column_raises(self, cohort_csv, lake_path):
        with pytest.raises(ValueError, match="not found"):
            add_cohort(
                csv_path=cohort_csv,
                lake_path=lake_path,
                name="bad",
                key_column="nonexistent",
            )

    def test_case_insensitive_key(self, tmp_path, lake_path):
        csv_path = tmp_path / "upper.csv"
        df = pl.DataFrame({"MRN": ["A", "B"], "val": [1, 2]})
        df.write_csv(csv_path)

        metadata = add_cohort(
            csv_path=csv_path,
            lake_path=lake_path,
            name="upper_test",
            key_column="mrn",
        )
        assert metadata.row_count == 2
        # Verify the parquet has normalized column name
        result_df = pl.read_parquet(lake_path / "cohorts" / "upper_test.parquet")
        assert "mrn" in result_df.columns

    def test_key_column_cast_to_string(self, tmp_path, lake_path):
        csv_path = tmp_path / "int_mrn.csv"
        df = pl.DataFrame({"mrn": [12345, 67890]})
        df.write_csv(csv_path)

        add_cohort(csv_path=csv_path, lake_path=lake_path, name="int_test")
        result_df = pl.read_parquet(lake_path / "cohorts" / "int_test.parquet")
        assert result_df["mrn"].dtype == pl.Utf8


class TestListCohorts:
    def test_empty(self, lake_path):
        result = list_cohorts(lake_path)
        assert result == []

    def test_returns_all(self, cohort_csv, lake_path):
        add_cohort(cohort_csv, lake_path, "cohort_a")
        add_cohort(cohort_csv, lake_path, "cohort_b")
        result = list_cohorts(lake_path)
        names = [c.name for c in result]
        assert "cohort_a" in names
        assert "cohort_b" in names


class TestGetCohortInfo:
    def test_found(self, cohort_csv, lake_path):
        add_cohort(cohort_csv, lake_path, "info_test", description="my desc")
        info = get_cohort_info(lake_path, "info_test")
        assert info is not None
        assert info.name == "info_test"
        assert info.description == "my desc"

    def test_not_found(self, lake_path):
        assert get_cohort_info(lake_path, "nope") is None


class TestRemoveCohort:
    def test_removes_files(self, cohort_csv, lake_path):
        add_cohort(cohort_csv, lake_path, "to_remove")
        assert (lake_path / "cohorts" / "to_remove.parquet").exists()

        removed = remove_cohort(lake_path, "to_remove")
        assert removed is True
        assert not (lake_path / "cohorts" / "to_remove.parquet").exists()
        assert not (lake_path / "cohorts" / "to_remove.cohort.json").exists()

    def test_remove_nonexistent(self, lake_path):
        removed = remove_cohort(lake_path, "nonexistent")
        assert removed is False

    def test_lifecycle(self, cohort_csv, lake_path):
        """Full add -> list -> info -> remove lifecycle."""
        add_cohort(cohort_csv, lake_path, "lifecycle")
        assert len(list_cohorts(lake_path)) == 1

        info = get_cohort_info(lake_path, "lifecycle")
        assert info.row_count == 3

        remove_cohort(lake_path, "lifecycle")
        assert len(list_cohorts(lake_path)) == 0
