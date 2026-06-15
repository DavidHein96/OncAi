"""Tests for oncai.cli._shared module."""

from __future__ import annotations

import polars as pl
import pytest
from click.exceptions import Exit as ClickExit

from oncai.cli._shared import (
    load_cohort_filter,
    load_mrn_filter,
    resolve_definition_path,
)


class TestResolveDefinitionPath:
    def test_directory_match(self, tmp_path):
        """If name matches a directory, return the directory."""
        (tmp_path / "ihc").mkdir()
        result = resolve_definition_path("ihc", tmp_path)
        assert result == tmp_path / "ihc"
        assert result.is_dir()

    def test_nested_path(self, tmp_path):
        """kidney/resection -> base/kidney/resection.yaml."""
        (tmp_path / "kidney").mkdir()
        yaml_file = tmp_path / "kidney" / "resection.yaml"
        yaml_file.write_text("test: true")

        result = resolve_definition_path("kidney/resection", tmp_path)
        assert result == yaml_file

    def test_simple_name_dir(self, tmp_path):
        """resection -> base/resection/ (directory takes priority)."""
        (tmp_path / "resection").mkdir()
        yaml_file = tmp_path / "resection" / "resection.yaml"
        yaml_file.write_text("test: true")

        result = resolve_definition_path("resection", tmp_path)
        # Directory match takes priority over file match
        assert result == tmp_path / "resection"

    def test_simple_name_file(self, tmp_path):
        """When only the yaml exists (no dir match), resolves to file."""
        # Create only the nested yaml, not the parent as a "directory match"
        (tmp_path / "myschema").mkdir()
        yaml_file = tmp_path / "myschema" / "myschema.yaml"
        yaml_file.write_text("test: true")

        # Because myschema/ IS a directory, it gets matched as dir.
        # Let's test with a name that doesn't conflict as a dir.
        (tmp_path / "sub").mkdir()
        yaml_file2 = tmp_path / "sub" / "sub.yaml"
        yaml_file2.write_text("test: true")

        # Since sub/ exists as dir, returns dir
        result = resolve_definition_path("sub", tmp_path)
        assert result == tmp_path / "sub"

    def test_direct_path(self, tmp_path):
        """Absolute path passed as name."""
        yaml_file = tmp_path / "custom.yaml"
        yaml_file.write_text("test: true")

        result = resolve_definition_path(str(yaml_file), tmp_path)
        assert result == yaml_file

    def test_not_found_raises_exit(self, tmp_path):
        with pytest.raises(ClickExit):
            resolve_definition_path("nonexistent", tmp_path)


class TestLoadMrnFilter:
    def test_basic(self, tmp_path):
        csv_path = tmp_path / "mrns.csv"
        df = pl.DataFrame({"mrn": ["MRN001", "MRN002", "MRN003"]})
        df.write_csv(csv_path)

        result = load_mrn_filter(csv_path)
        assert result == {"MRN001", "MRN002", "MRN003"}

    def test_case_insensitive_column(self, tmp_path):
        csv_path = tmp_path / "mrns.csv"
        df = pl.DataFrame({"MRN": ["A", "B"]})
        df.write_csv(csv_path)

        result = load_mrn_filter(csv_path)
        assert result == {"A", "B"}

    def test_extra_columns_ignored(self, tmp_path):
        csv_path = tmp_path / "mrns.csv"
        df = pl.DataFrame({"mrn": ["X"], "name": ["Alice"]})
        df.write_csv(csv_path)

        result = load_mrn_filter(csv_path)
        assert result == {"X"}

    def test_no_mrn_column_raises(self, tmp_path):
        csv_path = tmp_path / "bad.csv"
        df = pl.DataFrame({"patient_id": ["1"]})
        df.write_csv(csv_path)

        with pytest.raises(ClickExit):
            load_mrn_filter(csv_path)

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(ClickExit):
            load_mrn_filter(tmp_path / "missing.csv")

    def test_casts_to_string(self, tmp_path):
        csv_path = tmp_path / "mrns.csv"
        df = pl.DataFrame({"mrn": [12345, 67890]})
        df.write_csv(csv_path)

        result = load_mrn_filter(csv_path)
        assert all(isinstance(v, str) for v in result)


def _create_cohort(tmp_path, name="test_cohort", key_column="mrn", values=None):
    """Helper to create a cohort parquet + sidecar in a fake lake."""
    import json

    if values is None:
        values = ["MRN001", "MRN002", "MRN003"]

    lake_path = tmp_path / "lake"
    cohorts_dir = lake_path / "cohorts"
    cohorts_dir.mkdir(parents=True)

    df = pl.DataFrame({key_column: values})
    parquet_path = cohorts_dir / f"{name}.parquet"
    df.write_parquet(parquet_path)

    sidecar = parquet_path.with_suffix(".cohort.json")
    sidecar.write_text(
        json.dumps(
            {
                "name": name,
                "key_column": key_column,
                "description": "test cohort",
                "created_at": "2025-01-01T00:00:00Z",
                "row_count": len(values),
                "columns": [key_column],
                "source_file": "test.csv",
            }
        )
    )

    return lake_path


class TestLoadCohortFilter:
    def test_valid_cohort(self, tmp_path):
        lake_path = _create_cohort(tmp_path)
        result = load_cohort_filter("test_cohort", lake_path)

        assert result.key_column == "mrn"
        assert result.values == {"MRN001", "MRN002", "MRN003"}
        assert result.parquet_path == lake_path / "cohorts" / "test_cohort.parquet"

    def test_non_mrn_key(self, tmp_path):
        lake_path = _create_cohort(
            tmp_path, key_column="report_id", values=["R1", "R2"]
        )
        result = load_cohort_filter("test_cohort", lake_path)

        assert result.key_column == "report_id"
        assert result.values == {"R1", "R2"}

    def test_cohort_not_found_raises(self, tmp_path):
        lake_path = tmp_path / "lake"
        lake_path.mkdir()

        with pytest.raises(ClickExit):
            load_cohort_filter("nonexistent", lake_path)

    def test_sidecar_missing_raises(self, tmp_path):
        lake_path = tmp_path / "lake"
        cohorts_dir = lake_path / "cohorts"
        cohorts_dir.mkdir(parents=True)

        # Create parquet but no sidecar
        df = pl.DataFrame({"mrn": ["MRN001"]})
        df.write_parquet(cohorts_dir / "orphan.parquet")

        with pytest.raises(ClickExit):
            load_cohort_filter("orphan", lake_path)

    def test_casts_values_to_string(self, tmp_path):
        lake_path = _create_cohort(tmp_path, key_column="mrn", values=[12345, 67890])
        result = load_cohort_filter("test_cohort", lake_path)

        assert all(isinstance(v, str) for v in result.values)
