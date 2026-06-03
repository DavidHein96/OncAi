"""Tests for lake health check module."""

from __future__ import annotations

import duckdb
import polars as pl
import pytest

from oncai.config import OncaiConfig
from oncai.lake_check import (
    check_db_alignment,
    check_duplicate_keys,
    check_empty_datasets,
    check_jsonl_parseable,
    check_null_ids,
    check_parquet_readable,
    check_required_columns,
    run_lake_checks,
)


@pytest.fixture
def lake_path(tmp_path):
    """Create a minimal lake directory structure."""
    for folder in ["pathology", "cohorts", "fc_extractions"]:
        (tmp_path / "lake" / folder).mkdir(parents=True)
    return tmp_path / "lake"


@pytest.fixture
def config(tmp_path, lake_path):
    """Create a OncaiConfig pointing to temp directories."""
    return OncaiConfig(
        lake_path=lake_path,
        inbox_path=tmp_path / "inbox",
        remote_path=tmp_path / "remote",
        db_path=tmp_path / "test.duckdb",
    )


# --- parquet_readable ---


def test_readable_parquet_passes(lake_path):
    df = pl.DataFrame({"report_id": ["R1"], "mrn": ["M1"]})
    df.write_parquet(lake_path / "pathology" / "pathology.parquet")

    results = check_parquet_readable(lake_path)
    assert len(results) == 1
    assert results[0].passed is True
    assert results[0].name == "parquet_readable"


def test_corrupt_parquet_fails(lake_path):
    bad_file = lake_path / "pathology" / "pathology.parquet"
    bad_file.write_bytes(b"this is not a parquet file")

    results = check_parquet_readable(lake_path)
    assert len(results) == 1
    assert results[0].passed is False
    assert results[0].severity == "error"


def test_corrupt_parquet_silently_skipped_by_other_checks(lake_path):
    """Freezes current silent-skip behavior on a corrupted parquet.

    Only ``check_parquet_readable`` surfaces the failure; every other check_*
    helper continues past the broken file without emitting a CheckResult for
    it. If that policy changes, update this test alongside the implementation.
    """
    bad_file = lake_path / "pathology" / "pathology.parquet"
    bad_file.write_bytes(b"this is not a parquet file")
    scope = "pathology/pathology.parquet"

    # parquet_readable is the one place the failure is surfaced.
    readable = check_parquet_readable(lake_path)
    assert any(r.scope == scope and not r.passed for r in readable)

    # Every other check just skips — no result emitted for this scope.
    assert all(r.scope != scope for r in check_required_columns(lake_path))
    assert all(r.scope != scope for r in check_null_ids(lake_path))
    assert all(r.scope != scope for r in check_duplicate_keys(lake_path))
    assert all(r.scope != scope for r in check_empty_datasets(lake_path))


def test_corrupt_parquet_excluded_from_db_alignment(config, lake_path):
    """A corrupt parquet causes its table to be excluded from db_alignment.

    Treating an unreadable file as 0 rows would produce a phantom "DB has N,
    lake has 0" mismatch rooted in the read failure, not in real divergence.
    check_parquet_readable already reports the broken file at error severity,
    so the alignment check skips the table entirely rather than emitting a
    misleading mismatch.
    """
    bad_file = lake_path / "pathology" / "pathology.parquet"
    bad_file.write_bytes(b"this is not a parquet file")

    con = duckdb.connect(str(config.db_path))
    con.execute("CREATE SCHEMA IF NOT EXISTS raw")
    con.execute(
        "CREATE TABLE raw.pathology AS SELECT * FROM "
        "(VALUES ('R1', 'M1'), ('R2', 'M2')) t(report_id, mrn)"
    )
    con.close()

    results = check_db_alignment(config)
    assert all(r.scope != "raw.pathology" for r in results)


# --- required_columns ---


def test_required_columns_pass(lake_path):
    df = pl.DataFrame(
        {
            "report_id": ["R1"],
            "key_hash": [b"\x00"],
            "content_hash": [b"\x00"],
            "mrn": ["M1"],
            "report_text": ["text"],
            "ordering_date": [None],
            "external_name": [None],
        }
    )
    df.write_parquet(lake_path / "pathology" / "pathology.parquet")

    results = check_required_columns(lake_path)
    passed = [r for r in results if r.scope == "pathology/pathology.parquet"]
    assert len(passed) == 1
    assert passed[0].passed is True


def test_required_columns_missing(lake_path):
    # Missing key_hash column
    df = pl.DataFrame(
        {
            "report_id": ["R1"],
            "content_hash": [b"\x00"],
            "mrn": ["M1"],
            "report_text": ["text"],
        }
    )
    df.write_parquet(lake_path / "pathology" / "pathology.parquet")

    results = check_required_columns(lake_path)
    failures = [
        r for r in results if not r.passed and r.scope == "pathology/pathology.parquet"
    ]
    assert len(failures) >= 1
    messages = [r.message for r in failures]
    assert any("key_hash" in m for m in messages)


# --- null_ids ---


def test_null_ids_detected(lake_path):
    df = pl.DataFrame(
        {
            "report_id": [None, "R2"],
            "key_hash": [b"\x00", b"\x01"],
            "content_hash": [b"\x00", b"\x01"],
            "mrn": ["M1", "M2"],
            "report_text": ["text1", "text2"],
            "ordering_date": [None, None],
            "external_name": [None, None],
        }
    )
    df.write_parquet(lake_path / "pathology" / "pathology.parquet")

    results = check_null_ids(lake_path)
    failures = [r for r in results if not r.passed]
    assert len(failures) >= 1
    assert "report_id" in failures[0].message
    assert "1 null" in failures[0].message


def test_null_ids_passes_when_clean(lake_path):
    df = pl.DataFrame(
        {
            "report_id": ["R1", "R2"],
            "key_hash": [b"\x00", b"\x01"],
            "content_hash": [b"\x00", b"\x01"],
            "mrn": ["M1", "M2"],
            "report_text": ["text1", "text2"],
            "ordering_date": [None, None],
            "external_name": [None, None],
        }
    )
    df.write_parquet(lake_path / "pathology" / "pathology.parquet")

    results = check_null_ids(lake_path)
    passed = [r for r in results if r.passed]
    assert len(passed) >= 1


# --- duplicate_keys ---


def test_duplicate_keys_detected(lake_path):
    df = pl.DataFrame(
        {
            "report_id": ["R1", "R2"],
            "key_hash": [b"\x00", b"\x00"],  # duplicate
        }
    )
    df.write_parquet(lake_path / "pathology" / "pathology.parquet")

    results = check_duplicate_keys(lake_path)
    failures = [r for r in results if not r.passed]
    assert len(failures) == 1
    assert "duplicate" in failures[0].message
    assert failures[0].severity == "warning"


def test_duplicate_keys_passes_when_unique(lake_path):
    df = pl.DataFrame(
        {
            "report_id": ["R1", "R2"],
            "key_hash": [b"\x00", b"\x01"],
        }
    )
    df.write_parquet(lake_path / "pathology" / "pathology.parquet")

    results = check_duplicate_keys(lake_path)
    passed = [r for r in results if r.passed]
    assert len(passed) == 1


# --- empty_datasets ---


def test_empty_dataset_warning(lake_path):
    # Write an empty parquet
    df = pl.DataFrame({"report_id": pl.Series([], dtype=pl.String)})
    df.write_parquet(lake_path / "pathology" / "pathology.parquet")

    results = check_empty_datasets(lake_path)
    assert len(results) == 1
    assert results[0].passed is False
    assert results[0].severity == "warning"
    assert "0 rows" in results[0].message


def test_nonempty_dataset_no_warning(lake_path):
    df = pl.DataFrame({"report_id": ["R1"]})
    df.write_parquet(lake_path / "pathology" / "pathology.parquet")

    results = check_empty_datasets(lake_path)
    assert len(results) == 0  # only flags problems


# --- jsonl_parseable ---


def test_jsonl_valid_passes(tmp_path):
    output_dir = tmp_path / "fc_outputs"
    output_dir.mkdir()
    jsonl_file = output_dir / "results.jsonl"
    jsonl_file.write_text('{"key": "val1"}\n{"key": "val2"}\n')

    results = check_jsonl_parseable(tmp_path)
    assert len(results) == 1
    assert results[0].passed is True
    assert "2 lines" in results[0].message


def test_jsonl_invalid_line_fails(tmp_path):
    output_dir = tmp_path / "fc_outputs"
    output_dir.mkdir()
    jsonl_file = output_dir / "results.jsonl"
    jsonl_file.write_text('{"key": "val1"}\nthis is not json\n{"key": "val3"}\n')

    results = check_jsonl_parseable(tmp_path)
    assert len(results) == 1
    assert results[0].passed is False
    assert "1/3" in results[0].message


def test_jsonl_no_output_dirs(tmp_path):
    results = check_jsonl_parseable(tmp_path)
    assert len(results) == 0


# --- db_alignment ---


def test_db_alignment_mismatch(config, lake_path):
    # Write a parquet with 3 rows
    df = pl.DataFrame({"report_id": ["R1", "R2", "R3"], "mrn": ["M1", "M2", "M3"]})
    df.write_parquet(lake_path / "pathology" / "pathology.parquet")

    # Build a DB with only 2 rows
    con = duckdb.connect(str(config.db_path))
    con.execute("CREATE SCHEMA IF NOT EXISTS raw")
    con.execute(
        "CREATE TABLE raw.pathology AS SELECT * FROM (VALUES ('R1', 'M1'), ('R2', 'M2')) t(report_id, mrn)"
    )
    con.close()

    results = check_db_alignment(config)
    failures = [r for r in results if not r.passed and r.scope == "raw.pathology"]
    assert len(failures) == 1
    assert "DB has 2 rows" in failures[0].message
    assert "lake has 3 rows" in failures[0].message


def test_db_alignment_matches(config, lake_path):
    df = pl.DataFrame({"report_id": ["R1", "R2"], "mrn": ["M1", "M2"]})
    df.write_parquet(lake_path / "pathology" / "pathology.parquet")

    con = duckdb.connect(str(config.db_path))
    con.execute("CREATE SCHEMA IF NOT EXISTS raw")
    con.execute(
        "CREATE TABLE raw.pathology AS SELECT * FROM (VALUES ('R1', 'M1'), ('R2', 'M2')) t(report_id, mrn)"
    )
    con.close()

    results = check_db_alignment(config)
    passed = [r for r in results if r.passed and r.scope == "raw.pathology"]
    assert len(passed) == 1


def test_db_alignment_no_db(config):
    results = check_db_alignment(config)
    assert len(results) == 0


# --- run_lake_checks orchestrator ---


def test_run_lake_checks_returns_all(config, lake_path):
    # Create a valid pathology parquet
    df = pl.DataFrame(
        {
            "report_id": ["R1"],
            "key_hash": [b"\x00"],
            "content_hash": [b"\x00"],
            "mrn": ["M1"],
            "report_text": ["text"],
            "ordering_date": [None],
            "external_name": [None],
        }
    )
    df.write_parquet(lake_path / "pathology" / "pathology.parquet")

    results = run_lake_checks(config)

    # Should have results from multiple check types
    check_names = {r.name for r in results}
    assert "parquet_readable" in check_names
    assert "required_columns" in check_names
    assert "null_ids" in check_names
    assert "duplicate_keys" in check_names
