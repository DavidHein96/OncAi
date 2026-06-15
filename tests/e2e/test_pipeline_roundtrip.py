"""End-to-end pipeline: ingest → build-db → fc run-single → ingest fc_extractions → build-db.

The whole pipeline runs for real EXCEPT the LLM call, which is replaced with a
fake ``FunctionCallingClient`` that returns canned tool-call sequences keyed by
note_id. This catches cross-layer regressions (JSONL ↔ wide-parquet schema
drift, version-merge logic, manifest plumbing, DuckDB table shape) that unit
tests can't see.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import duckdb
import polars as pl
import pytest

from oncai.config import OncaiConfig
from oncai.db import build_database
from oncai.fc_extraction.batch_single import (
    SingleNoteConfig,
    run_fc_single_batch,
)
from oncai.fc_extraction.client import FunctionCallingClient, NoteExtractionResult
from oncai.fc_extraction.definitions.example import (
    DiagnosisType,
    RecordDiagnosis,
    RecordTreatment,
    TreatmentIntent,
    create_example_registry,
)
from oncai.fc_extraction.models import ApproxDate, FinishSingleExtraction
from oncai.fc_extraction.tools import ToolRegistry
from oncai.ingest import run_ingest


class _FakeFCClient(FunctionCallingClient):
    """Stand-in for FunctionCallingClient that returns canned extractions.

    Skips the real ``__init__`` (no Azure endpoint needed) and overrides
    ``extract_single_note`` to look up a hand-crafted ``NoteExtractionResult``
    by note_id. ``_build_run_meta`` only reads optional attrs off the client
    via ``getattr(..., default)``, so leaving them unset is fine.
    """

    def __init__(self, canned: dict[str, list[tuple[str, Any]]]) -> None:
        # Intentionally skip super().__init__() — the real init wants Azure
        # credentials we don't have, and the only method we override here
        # doesn't touch the parent's attributes.
        self._canned = canned

    def extract_single_note(  # type: ignore[override]
        self,
        note_text: str,
        system_prompt: str,
        registry: ToolRegistry,
        note_id: str = "",
    ) -> NoteExtractionResult:
        events = self._canned.get(note_id, [])
        return NoteExtractionResult(
            note_id=note_id,
            success=True,
            finish=FinishSingleExtraction(
                note_id=note_id,
                reasoning=f"extracted {len(events)} event(s)",
                confidence=0.95,
            ),
            events=events,
            rounds=1,
            input_tokens=100,
            output_tokens=20,
            reasoning_tokens=0,
        )


@pytest.fixture
def pipeline_env(tmp_path: Path) -> tuple[Path, OncaiConfig]:
    """Stand up the oncai folder tree rooted at tmp_path."""
    base = tmp_path / "oncai_data"
    for sub in ("lake", "inbox", "remote"):
        (base / sub).mkdir(parents=True)
    for folder in ("pathology", "cohorts", "fc_extractions", "runs"):
        (base / "lake" / folder).mkdir(parents=True)
        (base / "inbox" / folder).mkdir(parents=True)
    cfg = OncaiConfig(
        remote_path=base / "remote",
        lake_path=base / "lake",
        inbox_path=base / "inbox",
        db_path=base / "test.duckdb",
    )
    return tmp_path, cfg


def _write_pathology_csv(inbox_dir: Path, name: str, rows: list[dict]) -> Path:
    p = inbox_dir / "pathology" / name
    pl.DataFrame(rows).write_csv(p)
    return p


def test_full_pipeline_with_fake_llm(pipeline_env: tuple[Path, OncaiConfig]) -> None:
    tmp_path, cfg = pipeline_env

    # === 1. Pathology CSV → lake parquet ==================================
    _write_pathology_csv(
        cfg.inbox_path,
        "2024-01-15_path.csv",
        [
            {
                "report_id": "R1",
                "row_id": 1,
                "mrn": "M1",
                "mult_ln_val_storage": "DIAGNOSIS: Clear cell renal cell carcinoma, T2N0M0.",
                "ordering_date": "2024-01-15",
            },
            {
                "report_id": "R2",
                "row_id": 1,
                "mrn": "M2",
                "mult_ln_val_storage": (
                    "DIAGNOSIS: Papillary RCC, type 1. "
                    "Patient underwent radical nephrectomy."
                ),
                "ordering_date": "2024-01-15",
            },
        ],
    )

    ingest_results = run_ingest(cfg, folder="pathology")
    assert len(ingest_results) == 1
    assert ingest_results[0].row_count == 2

    lake_path = cfg.lake_path / "pathology" / "pathology.parquet"
    assert lake_path.exists()
    df_lake = pl.read_parquet(lake_path)
    assert df_lake.height == 2
    assert set(df_lake["report_id"].to_list()) == {"R1", "R2"}
    assert {"key_hash", "content_hash"}.issubset(df_lake.columns)

    # === 2. Lake parquet → DuckDB ========================================
    db_counts = build_database(cfg)
    assert db_counts.get("raw.pathology") == 2

    # === 3. Fake-LLM extraction =========================================
    canned: dict[str, list[tuple[str, Any]]] = {
        "R1": [
            (
                "record_diagnosis",
                RecordDiagnosis(
                    note_id="R1",
                    evidence=["DIAGNOSIS: Clear cell renal cell carcinoma, T2N0M0."],
                    diagnosis_date=ApproxDate(date="2024-01-15", precision=3),
                    diagnosis_type=DiagnosisType.PRIMARY,
                    diagnosis_name="clear cell renal cell carcinoma",
                    stage="T2N0M0",
                ),
            ),
        ],
        "R2": [
            (
                "record_diagnosis",
                RecordDiagnosis(
                    note_id="R2",
                    evidence=["DIAGNOSIS: Papillary RCC, type 1."],
                    diagnosis_date=ApproxDate(date="2024-01-15", precision=3),
                    diagnosis_type=DiagnosisType.PRIMARY,
                    diagnosis_name="papillary RCC",
                ),
            ),
            (
                "record_treatment",
                RecordTreatment(
                    note_id="R2",
                    evidence=["Patient underwent radical nephrectomy."],
                    treatment_date=ApproxDate(date="2024-01-15", precision=3),
                    treatment_name="radical nephrectomy",
                    treatment_type="surgery",
                    intent=TreatmentIntent.CURATIVE,
                ),
            ),
        ],
    }
    fake_client = _FakeFCClient(canned=canned)
    fc_config = SingleNoteConfig(name="Example", system_prompt="test prompt")
    fc_output_dir = tmp_path / "fc_outputs"

    batch_result = run_fc_single_batch(
        registry=create_example_registry(),
        config=fc_config,
        client=fake_client,
        db_path=cfg.db_path,
        source_table="raw.pathology",
        output_dir=fc_output_dir,
        batch_name="v1",
        text_col="report_text",
        id_col="report_id",
        progress=False,
    )

    assert batch_result.total_notes == 2
    assert batch_result.successful == 2
    assert batch_result.failed == 0

    jsonl_path = fc_output_dir / "Example" / "v1.jsonl"
    manifest_path = jsonl_path.with_name("v1_manifest.json")
    assert jsonl_path.exists()
    assert manifest_path.exists()

    with jsonl_path.open() as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert len(records) == 2
    by_id = {r["note_id"]: r for r in records}
    assert by_id["R1"]["success"] is True
    assert (
        by_id["R1"]["events"]["record_diagnosis"][0]["diagnosis_name"]
        == "clear cell renal cell carcinoma"
    )
    assert (
        by_id["R2"]["events"]["record_treatment"][0]["treatment_name"]
        == "radical nephrectomy"
    )

    # === 4. Promote the run's JSONL into a batch folder as segment 001 ====
    # A batch is a folder of numbered segments: inbox/fc_extractions/<batch>/NNN.jsonl.
    batch_dir = cfg.inbox_path / "fc_extractions" / "v1"
    batch_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(jsonl_path, batch_dir / "001.jsonl")
    shutil.copy2(manifest_path, batch_dir / "001_manifest.json")

    fc_ingest_results = run_ingest(cfg, folder="fc_extractions")
    assert len(fc_ingest_results) == 1
    lake_fc_parquet = cfg.lake_path / "fc_extractions" / "v1.parquet"
    assert lake_fc_parquet.exists()

    # === 5. Rebuild DB and verify the extractions table is queryable =====
    db_counts2 = build_database(cfg, force=True)
    assert db_counts2.get("extractions_raw.v1") == 2
    assert db_counts2.get("raw.pathology") == 2

    con = duckdb.connect(str(cfg.db_path), read_only=True)
    try:
        rows = con.execute(
            "SELECT record_id, events_json FROM extractions_raw.v1 ORDER BY record_id"
        ).fetchall()
    finally:
        con.close()

    assert [r[0] for r in rows] == ["R1", "R2"]
    r1_events = json.loads(rows[0][1])
    r2_events = json.loads(rows[1][1])
    assert "record_diagnosis" in r1_events
    assert r1_events["record_diagnosis"][0]["diagnosis_name"] == (
        "clear cell renal cell carcinoma"
    )
    assert "record_treatment" in r2_events
    assert r2_events["record_treatment"][0]["treatment_name"] == "radical nephrectomy"
