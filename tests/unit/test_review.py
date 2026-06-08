"""Tests for the review-package builder (oncai.review)."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import Field

from oncai.fc_extraction.models import ExtractionEvent, ExtractionPlan
from oncai.fc_extraction.tools import GatedToolRegistry, ToolDefinition, ToolRegistry
from oncai.review import (
    build_field_schema,
    build_review_package,
    infer_field_schema_from_events,
    package_from_jsonl,
)
from oncai.review.package import default_package_path


# ---------------------------------------------------------------------------
# build_field_schema — controls inferred from Pydantic models
# ---------------------------------------------------------------------------


def _example_registry() -> ToolRegistry:
    from oncai.fc_extraction.definitions.example import create_example_registry

    return create_example_registry()


def _field(schema: dict, event_type: str, name: str) -> dict:
    fields = schema[event_type]["fields"]
    return next(f for f in fields if f["name"] == name)


def test_field_schema_covers_event_tools_only() -> None:
    schema = build_field_schema(_example_registry())
    assert set(schema) == {"record_diagnosis", "record_treatment"}
    # Built-in control-flow tools never appear.
    assert "finish_note_extraction" not in schema


def test_field_schema_controls() -> None:
    schema = build_field_schema(_example_registry())

    # enum -> dropdown with options + required
    dtype = _field(schema, "record_diagnosis", "diagnosis_type")
    assert dtype["control"] == "enum"
    assert dtype["options"] == ["primary", "secondary", "recurrence"]
    assert dtype["required"] is True

    # ApproxDate -> date widget
    assert _field(schema, "record_diagnosis", "diagnosis_date")["control"] == "approx_date"

    # optional str -> text, not required
    stage = _field(schema, "record_diagnosis", "stage")
    assert stage["control"] == "text"
    assert stage["required"] is False

    # Literal str union -> enum
    ttype = _field(schema, "record_treatment", "treatment_type")
    assert ttype["control"] == "enum"
    assert ttype["options"] == ["surgery", "systemic", "radiation", "other"]

    # comment (required free text) survives; note_id is dropped (header chip)
    names = {f["name"] for f in schema["record_diagnosis"]["fields"]}
    assert "comment" in names
    assert "note_id" not in names


def test_field_schema_label_humanized() -> None:
    schema = build_field_schema(_example_registry())
    assert schema["record_diagnosis"]["label"] == "Record Diagnosis"
    assert _field(schema, "record_diagnosis", "diagnosis_date")["label"] == "Diagnosis Date"


def test_field_schema_excludes_plan_tools() -> None:
    class MyPlan(ExtractionPlan):
        note: str = Field(..., description="plan")

    class MyEvent(ExtractionEvent):
        value: int = Field(..., description="v")

    reg = ToolRegistry(single_note=True)
    reg.register(name="orient", description="plan tool", model=MyPlan)
    reg.register(name="record", description="event tool", model=MyEvent)

    schema = build_field_schema(reg)
    assert "record" in schema
    assert "orient" not in schema  # ExtractionPlan subclasses are not reviewed
    assert _field(schema, "record", "value")["control"] == "number"


def test_field_schema_bool_control() -> None:
    schema = build_field_schema(_example_registry())
    # PathKidneyBasic has bool flags — but example doesn't; build a tiny one.

    class Flagged(ExtractionEvent):
        flag: bool = Field(..., description="a flag")
        snippets: list[str] = Field(default_factory=list, description="evidence")

    reg = ToolRegistry(single_note=True)
    reg.register(name="rec", description="d", model=Flagged)
    s = build_field_schema(reg)
    assert _field(s, "rec", "flag")["control"] == "bool"
    # string lists render as locatable evidence snippets
    assert _field(s, "rec", "snippets")["control"] == "evidence_verbatim"


def test_gated_registry_schema_covers_all_phases() -> None:
    class GateEvent(ExtractionEvent):
        report_type: str = Field(..., description="kind")

    class BiopsyEvent(ExtractionEvent):
        depth: int = Field(..., description="d")

    gate = {
        "classify": ToolDefinition(
            name="classify", description="gate", model=GateEvent
        )
    }
    phase_map = {
        "biopsy": {
            "record_biopsy": ToolDefinition(
                name="record_biopsy", description="x", model=BiopsyEvent
            )
        }
    }
    reg = GatedToolRegistry(gate_tools=gate, phase_map=phase_map)

    schema = build_field_schema(reg)
    # Even though the registry starts in the gate phase, the extract-phase tool
    # is still covered.
    assert "classify" in schema
    assert "record_biopsy" in schema
    assert _field(schema, "record_biopsy", "depth")["control"] == "number"


# ---------------------------------------------------------------------------
# infer_field_schema_from_events — registry-free fallback
# ---------------------------------------------------------------------------


def test_infer_schema_from_events() -> None:
    events = {
        "record_x": [
            {
                "note_id": "N1",
                "comment": "c",
                "when": {"date": "2023-01-01", "precision": 3},
                "count": 4,
                "flag": True,
                "quotes": ["a", "b"],
            }
        ]
    }
    schema = infer_field_schema_from_events(events)
    controls = {f["name"]: f["control"] for f in schema["record_x"]["fields"]}
    assert controls["when"] == "approx_date"
    assert controls["count"] == "number"
    assert controls["flag"] == "bool"
    assert controls["quotes"] == "evidence_verbatim"
    assert controls["comment"] == "text"
    assert "note_id" not in controls


# ---------------------------------------------------------------------------
# build_review_package — grouping, event_key, plans excluded, backfill
# ---------------------------------------------------------------------------


def _record(note_id: str, mrn: str | None, events: dict) -> dict:
    return {
        "note_id": note_id,
        "mrn": mrn,
        "success": True,
        "definition_name": "Example",
        "events": events,
        "plans": {},
    }


def test_build_package_groups_by_mrn_and_keys_events() -> None:
    records = [
        _record(
            "N1",
            "M1",
            {"record_diagnosis": [{"note_id": "N1", "diagnosis_name": "RCC"}]},
        ),
        _record(
            "N2",
            "M1",
            {"record_diagnosis": [{"note_id": "N2", "diagnosis_name": "x"}]},
        ),
        _record(
            "N3",
            "M2",
            {
                "record_diagnosis": [
                    {"note_id": "N3", "diagnosis_name": "a"},
                    {"note_id": "N3", "diagnosis_name": "b"},
                ]
            },
        ),
    ]
    pkg = build_review_package(
        definition_name="Example",
        batch="v1",
        records=records,
        field_schema=build_field_schema(_example_registry()),
        notes={"N1": {"note_text": "report one"}},
        generated_at="2026-01-01T00:00:00+00:00",
    )

    patients = {p["mrn"]: p for p in pkg["patients"]}
    assert set(patients) == {"M1", "M2"}
    assert len(patients["M1"]["events"]) == 2
    assert len(patients["M2"]["events"]) == 2

    # Two events on the same note get distinct, stable keys.
    keys = [e["event_key"] for e in patients["M2"]["events"]]
    assert keys == ["N3::record_diagnosis::0", "N3::record_diagnosis::1"]

    # Note text is attached where available.
    assert patients["M1"]["notes"]["N1"]["note_text"] == "report one"


def test_build_package_falls_back_to_note_id_without_mrn() -> None:
    records = [_record("N1", None, {"record_diagnosis": [{"diagnosis_name": "x"}]})]
    pkg = build_review_package(
        definition_name="Example",
        batch="v1",
        records=records,
        field_schema={},
        notes={},
    )
    assert pkg["patients"][0]["mrn"] == "N1"


def test_build_package_excludes_plans_and_backfills_schema() -> None:
    records = [
        {
            "note_id": "N1",
            "mrn": "M1",
            "success": True,
            "events": {"surprise_tool": [{"note_id": "N1", "score": 5}]},
            "plans": {"orient": [{"note_id": "N1", "thought": "x"}]},
        }
    ]
    pkg = build_review_package(
        definition_name="X",
        batch="v1",
        records=records,
        field_schema={},  # registry didn't know about surprise_tool
        notes={},
    )
    # Plans never become events.
    events = pkg["patients"][0]["events"]
    assert [e["event_type"] for e in events] == ["surprise_tool"]
    # Unknown event type is backfilled so the app can still render fields.
    assert "surprise_tool" in pkg["field_schema"]
    controls = {
        f["name"]: f["control"]
        for f in pkg["field_schema"]["surprise_tool"]["fields"]
    }
    assert controls == {"score": "number"}


# ---------------------------------------------------------------------------
# package_from_jsonl — end-to-end, JSONL note source
# ---------------------------------------------------------------------------


def test_package_from_jsonl_roundtrip(tmp_path: Path) -> None:
    # A batch JSONL with one success + one failure (the failure is dropped).
    batch = tmp_path / "v1.jsonl"
    records = [
        {
            "note_id": "KP-001",
            "mrn": "M0001",
            "success": True,
            "definition_name": "Example",
            "events": {
                "record_diagnosis": [
                    {
                        "note_id": "KP-001",
                        "comment": "clear cell RCC",
                        "diagnosis_name": "renal cell carcinoma",
                        "diagnosis_type": "primary",
                        "diagnosis_date": {"date": "2024-03-15", "precision": 3},
                    }
                ]
            },
        },
        {"note_id": "KP-002", "success": False, "events": {}},
    ]
    with batch.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    # A notes JSONL the package can re-read for source text.
    notes_jsonl = tmp_path / "notes.jsonl"
    with notes_jsonl.open("w") as fh:
        fh.write(
            json.dumps(
                {"report_id": "KP-001", "report_text": "DIAGNOSIS: RCC.", "mrn": "M0001"}
            )
            + "\n"
        )

    out = package_from_jsonl(
        jsonl_path=batch,
        registry=_example_registry(),
        source_table=f"jsonl:{notes_jsonl}",
        db_path=notes_jsonl,
    )
    assert out == default_package_path(batch)
    pkg = json.loads(out.read_text())

    assert pkg["definition_name"] == "Example"
    assert pkg["batch"] == "v1"
    # Only the successful record is in the package.
    assert len(pkg["patients"]) == 1
    patient = pkg["patients"][0]
    assert patient["mrn"] == "M0001"
    assert len(patient["events"]) == 1
    assert patient["notes"]["KP-001"]["note_text"] == "DIAGNOSIS: RCC."
    # Schema from the registry carries enum options.
    assert pkg["field_schema"]["record_diagnosis"]["fields"]


def test_package_from_jsonl_without_registry_or_notes(tmp_path: Path) -> None:
    batch = tmp_path / "v2.jsonl"
    with batch.open("w") as fh:
        fh.write(
            json.dumps(
                {
                    "note_id": "N1",
                    "mrn": "M1",
                    "success": True,
                    "definition_name": "Mystery",
                    "events": {"rec": [{"note_id": "N1", "n": 1}]},
                }
            )
            + "\n"
        )
    out = package_from_jsonl(jsonl_path=batch)
    pkg = json.loads(out.read_text())
    # Inferred schema still lets the app render the event.
    assert "rec" in pkg["field_schema"]
    assert pkg["patients"][0]["notes"]["N1"]["note_text"] is None
