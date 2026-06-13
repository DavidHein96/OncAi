"""Shared pytest fixtures for oncai tests."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from oncai.config import OncaiConfig


@pytest.fixture
def tmp_oncai_env(tmp_path, monkeypatch):
    """
    Create a temporary oncai environment with proper directory structure.

    Changes working directory to tmp_path so config files are isolated.
    """
    monkeypatch.chdir(tmp_path)

    data_dir = tmp_path / "oncai_data"
    for folder in ["lake", "inbox", "remote"]:
        (data_dir / folder).mkdir(parents=True)

    for subfolder in [
        "pathology",
        "cohorts",
        "runs",
        "fc_extractions",
        "fc_reviews",
    ]:
        (data_dir / "lake" / subfolder).mkdir(parents=True)
        (data_dir / "inbox" / subfolder).mkdir(parents=True)

    return tmp_path


@pytest.fixture
def oncai_config(tmp_oncai_env) -> OncaiConfig:
    """Create a OncaiConfig pointing to the temp environment."""
    data_dir = tmp_oncai_env / "oncai_data"
    return OncaiConfig(
        remote_path=data_dir / "remote",
        lake_path=data_dir / "lake",
        inbox_path=data_dir / "inbox",
        db_path=data_dir / "oncai.duckdb",
    )


@pytest.fixture
def sample_pathology_csv(tmp_path) -> Path:
    """Create a sample pathology CSV with multi-line reports."""
    csv_path = tmp_path / "pathology.csv"

    data = [
        # Report 1: 3 lines
        {
            "report_id": "CO19-001",
            "row_id": 1,
            "mrn": "MRN001",
            "mult_ln_val_storage": "DIAGNOSIS: Clear cell renal cell carcinoma.",
            "ordering_date": "2024-01-15",
        },
        {
            "report_id": "CO19-001",
            "row_id": 2,
            "mrn": "MRN001",
            "mult_ln_val_storage": "Tumor size: 4.5 cm greatest dimension.",
            "ordering_date": "2024-01-15",
        },
        {
            "report_id": "CO19-001",
            "row_id": 3,
            "mrn": "MRN001",
            "mult_ln_val_storage": "Margins: Negative for carcinoma.",
            "ordering_date": "2024-01-15",
        },
        # Report 2: 2 lines
        {
            "report_id": "CO19-002",
            "row_id": 1,
            "mrn": "MRN002",
            "mult_ln_val_storage": "DIAGNOSIS: Papillary renal cell carcinoma, type 1.",
            "ordering_date": "2024-01-16",
        },
        {
            "report_id": "CO19-002",
            "row_id": 2,
            "mrn": "MRN002",
            "mult_ln_val_storage": "Fuhrman grade: 2.",
            "ordering_date": "2024-01-16",
        },
        # Report 3: 1 line
        {
            "report_id": "CO19-003",
            "row_id": 1,
            "mrn": "MRN003",
            "mult_ln_val_storage": "DIAGNOSIS: Oncocytoma.",
            "ordering_date": "2024-01-17",
        },
    ]

    df = pl.DataFrame(data)
    df.write_csv(csv_path)
    return csv_path
