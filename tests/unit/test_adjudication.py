"""Tests for two-batch adjudication package creation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from oncai.adjudication import (
    ADJUDICATION_DESCRIPTOR,
    ADJUDICATION_LOG_SUFFIX,
    ADJUDICATION_PACKAGE_SUFFIX,
    build_adjudication_package,
    package_from_jsonls,
)
from oncai.fc_extraction.models import ExtractionEvent
from oncai.fc_extraction.tools import ToolRegistry
from oncai.ingest import run_ingest
from oncai.review.schema import build_field_schema
from oncai.sidecar import sidecar_path


class _Marker(ExtractionEvent):
    marker: str = Field(..., description="marker")
    status: Literal["positive", "negative"] = Field(..., description="status")
    detail: str | None = Field(None, description="free text detail")


def _registry() -> ToolRegistry:
    reg = ToolRegistry(single_note=True)
    reg.register(
        name="record_marker",
        description="record marker",
        model=_Marker,
        event_identity_fields=("marker",),
        comparison_fields=("status",),
    )
    return reg


def _record(note_id: str, events: list[dict], *, mrn: str = "M1") -> dict:
    return {
        "note_id": note_id,
        "mrn": mrn,
        "definition_name": "D",
        "success": True,
        "events": {"record_marker": events},
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")


def test_build_adjudication_package_keeps_only_disagreements() -> None:
    field_schema = build_field_schema(_registry())
    left = [
        _record(
            "N1",
            [
                {"note_id": "N1", "marker": "CK7", "status": "positive"},
                {"note_id": "N1", "marker": "PAX8", "status": "positive"},
            ],
        )
    ]
    right = [
        _record(
            "N1",
            [
                {"note_id": "N1", "marker": "CK7", "status": "positive"},
                {"note_id": "N1", "marker": "PAX8", "status": "negative"},
                {"note_id": "N1", "marker": "CAIX", "status": "positive"},
            ],
        )
    ]

    package = build_adjudication_package(
        round_name="ihc_compare_v1",
        definition_name="D",
        left_records=left,
        right_records=right,
        left_batch="left",
        right_batch="right",
        left_label="gpt",
        right_label="gemma",
        left_jsonl=Path("left.jsonl"),
        right_jsonl=Path("right.jsonl"),
        field_schema=field_schema,
        notes={},
        generated_at="2026-06-14T00:00:00+00:00",
    )

    events = package["patients"][0]["events"]
    assert package["summary"]["disagreements"] == 2
    assert {event["status"] for event in events} == {"different", "missing_left"}
    assert all("CK7" not in json.dumps(event) for event in events)
    pax8 = next(event for event in events if event["status"] == "different")
    assert pax8["comparison"]["gpt"]["status"] == "positive"
    assert pax8["comparison"]["gemma"]["status"] == "negative"
    caix = next(event for event in events if event["status"] == "missing_left")
    assert caix["left"] is None
    assert caix["right"]["fields"]["marker"] == "CAIX"


def test_package_from_jsonls_writes_canonical_files(oncai_config) -> None:
    left = oncai_config.inbox_path / "fc_extractions" / "left" / "001.jsonl"
    right = oncai_config.inbox_path / "fc_extractions" / "right" / "001.jsonl"
    _write_jsonl(
        left,
        [_record("N1", [{"note_id": "N1", "marker": "CK7", "status": "positive"}])],
    )
    _write_jsonl(
        right,
        [_record("N1", [{"note_id": "N1", "marker": "CK7", "status": "negative"}])],
    )

    package_path, descriptor_path, package = package_from_jsonls(
        config=oncai_config,
        round_name="ihc_compare_v1",
        definition_name="D",
        left_jsonl=left,
        right_jsonl=right,
        field_schema=build_field_schema(_registry()),
        left_label="left",
        right_label="right",
    )

    assert package_path == (
        oncai_config.inbox_path
        / "fc_adjudications"
        / "ihc_compare_v1"
        / f"ihc_compare_v1{ADJUDICATION_PACKAGE_SUFFIX}"
    )
    assert descriptor_path == package_path.parent / ADJUDICATION_DESCRIPTOR
    assert package_path.exists()
    assert descriptor_path.exists()
    descriptor = json.loads(descriptor_path.read_text())
    assert descriptor["round"] == "ihc_compare_v1"
    assert descriptor["summary"]["disagreements"] == 1
    assert package["summary"]["disagreements"] == 1


def test_ingest_fc_adjudications_sidecars_package_and_waits_for_log(
    oncai_config,
) -> None:
    round_dir = oncai_config.inbox_path / "fc_adjudications" / "ihc_compare_v1"
    round_dir.mkdir(parents=True, exist_ok=True)
    package = round_dir / f"ihc_compare_v1{ADJUDICATION_PACKAGE_SUFFIX}"
    descriptor = round_dir / ADJUDICATION_DESCRIPTOR
    package.write_text(json.dumps({"round": "ihc_compare_v1"}))
    descriptor.write_text(json.dumps({"round": "ihc_compare_v1"}))

    results = run_ingest(oncai_config, folder="fc_adjudications")

    assert sidecar_path(package).exists()
    assert sidecar_path(descriptor).exists()
    assert not (
        oncai_config.lake_path / "fc_adjudications" / "ihc_compare_v1.parquet"
    ).exists()
    notes = " ".join(results[0].notes)
    assert f"waiting for *{ADJUDICATION_LOG_SUFFIX}" in notes
