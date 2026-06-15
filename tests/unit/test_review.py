"""Tests for the review-package builder (oncai.review)."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from oncai.fc_extraction.models import ExtractionEvent, ExtractionPlan
from oncai.fc_extraction.tools import GatedToolRegistry, ToolDefinition, ToolRegistry
from oncai.review import (
    SlotState,
    adjudication_hash,
    adjudication_hash_from_registry,
    build_field_schema,
    build_review_package,
    event_content_hash,
    infer_field_schema_from_events,
    package_from_jsonl,
    record_to_slots,
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
    assert set(schema) == {
        "record_diagnosis",
        "record_treatment",
        "flag_report_for_review",
    }
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
    assert (
        _field(schema, "record_diagnosis", "diagnosis_date")["control"] == "approx_date"
    )

    # optional str -> text, not required
    stage = _field(schema, "record_diagnosis", "stage")
    assert stage["control"] == "text"
    assert stage["required"] is False

    # Literal str union -> enum
    ttype = _field(schema, "record_treatment", "treatment_type")
    assert ttype["control"] == "enum"
    assert ttype["options"] == ["surgery", "systemic", "radiation", "other"]

    # Base provenance/header fields are hidden from editable field schema.
    names = {f["name"] for f in schema["record_diagnosis"]["fields"]}
    assert "evidence" not in names
    assert "note_id" not in names


def test_field_schema_label_humanized() -> None:
    schema = build_field_schema(_example_registry())
    assert schema["record_diagnosis"]["label"] == "Record Diagnosis"
    assert (
        _field(schema, "record_diagnosis", "diagnosis_date")["label"]
        == "Diagnosis Date"
    )


def test_field_schema_includes_comparison_metadata() -> None:
    class MyEvent(ExtractionEvent):
        finding_id: str = Field(..., description="finding key")
        value: int = Field(..., description="v")

    reg = ToolRegistry(single_note=True)
    reg.register(
        name="record",
        description="event tool",
        model=MyEvent,
        event_identity_fields=("finding_id",),
        comparison_fields=("value",),
    )

    schema = build_field_schema(reg)
    assert schema["record"]["event_identity_fields"] == ["finding_id"]
    assert schema["record"]["comparison_fields"] == ["value"]


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
    # PathKidneyBasic has bool flags — but example doesn't; build a tiny one.

    class Flagged(ExtractionEvent):
        flag: bool = Field(..., description="a flag")
        snippets: list[str] = Field(default_factory=list, description="evidence")

    reg = ToolRegistry(single_note=True)
    reg.register(name="rec", description="d", model=Flagged)
    s = build_field_schema(reg)
    assert _field(s, "rec", "flag")["control"] == "bool"
    # list fields are data, not implicit source-note anchors.
    assert _field(s, "rec", "snippets")["control"] == "readonly"


def test_registered_definitions_hide_provenance_fields() -> None:
    from oncai.cli.fc_cmds import _DEFINITIONS

    for _key, (module_path, factory_name) in _DEFINITIONS.items():
        module = importlib.import_module(module_path)
        registry = getattr(module, factory_name)()
        schema = build_field_schema(registry)

        assert "flag_report_for_review" in schema
        names = {f["name"] for f in schema["flag_report_for_review"]["fields"]}
        assert "review_anchor" not in names
        assert "evidence" not in names


def test_registered_definitions_teach_explicit_provenance() -> None:
    from oncai.cli.fc_cmds import _DEFINITIONS

    for _key, (module_path, _factory_name) in _DEFINITIONS.items():
        module = importlib.import_module(module_path)
        text = Path(module.__file__).read_text()

        assert "`evidence`" in text
        assert "field_evidence" not in text
        assert "review_anchor" in text
        assert "flag_for_review_anchor" not in text
        assert "supporting quotes" not in text


def test_ihc_definition_declares_event_identity_fields() -> None:
    from oncai.fc_extraction.definitions.path_kidney_ihc import (
        create_path_kidney_ihc_registry,
    )

    schema = build_field_schema(create_path_kidney_ihc_registry())

    assert schema["record_ihc_result"]["event_identity_fields"] == [
        "specimen_id",
        "standardized_test_name",
    ]
    assert schema["record_ihc_result"]["comparison_fields"] == [
        "flag_for_sub_specimen_heterogeneity",
        "given_result",
        "standardized_test_status",
        "standardized_test_intensity",
        "standardized_test_extent",
        "standardized_test_pattern",
    ]


def test_gated_registry_schema_covers_all_phases() -> None:
    class GateEvent(ExtractionEvent):
        report_type: str = Field(..., description="kind")

    class BiopsyEvent(ExtractionEvent):
        depth: int = Field(..., description="d")

    gate = {
        "classify": ToolDefinition(name="classify", description="gate", model=GateEvent)
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
    assert controls["quotes"] == "readonly"
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

    # Two findings on the same note, no declared identity → ordinal slot keys.
    keys = [e["event_key"] for e in patients["M2"]["events"]]
    assert keys == ["N3::record_diagnosis::1", "N3::record_diagnosis::2"]

    # Note text is attached where available.
    assert patients["M1"]["notes"]["N1"]["note_text"] == "report one"


def test_event_key_uses_ordinal_without_declared_identity() -> None:
    # No event_identity_fields → each finding keyed by its ordinal for its type.
    rec = _record("N1", "M1", {"record_diagnosis": [{"d": "a"}, {"d": "b"}]})
    pkg = build_review_package(
        definition_name="X", batch="v1", records=[rec], field_schema={}, notes={}
    )
    keys = [e["event_key"] for e in pkg["patients"][0]["events"]]
    assert keys == ["N1::record_diagnosis::1", "N1::record_diagnosis::2"]


def test_event_key_uses_declared_identity_order_independently() -> None:
    fs = {"record_x": {"label": "X", "event_identity_fields": ["sid"], "fields": []}}

    def keys(events: list[dict]) -> set[str]:
        rec = _record("N1", "M1", {"record_x": events})
        pkg = build_review_package(
            definition_name="X", batch="v1", records=[rec], field_schema=fs, notes={}
        )
        return {e["event_key"] for e in pkg["patients"][0]["events"]}

    a, b = {"sid": "A", "v": 1}, {"sid": "B", "v": 2}
    # Identity-derived keys (not positional) → stable across event order.
    assert keys([a, b]) == keys([b, a])
    assert all("::id-" in k for k in keys([a, b]))


def test_identity_collision_gets_disambiguating_suffix() -> None:
    fs = {"record_x": {"label": "X", "event_identity_fields": ["sid"], "fields": []}}
    # Two findings claim the SAME identity (a data clash) → distinct keys via suffix.
    rec = _record(
        "N1", "M1", {"record_x": [{"sid": "A", "v": 1}, {"sid": "A", "v": 2}]}
    )
    pkg = build_review_package(
        definition_name="X", batch="v1", records=[rec], field_schema=fs, notes={}
    )
    keys = [e["event_key"] for e in pkg["patients"][0]["events"]]
    assert len(set(keys)) == 2
    assert keys[1].endswith("::1")  # dup suffix on the collision


def test_package_events_carry_fingerprint() -> None:
    # Two identical findings: same value fingerprint, distinct ordinal slots.
    rec = _record("N1", "M1", {"record_diagnosis": [{"d": "a"}, {"d": "a"}]})
    evs = build_review_package(
        definition_name="X", batch="v1", records=[rec], field_schema={}, notes={}
    )["patients"][0]["events"]
    assert evs[0]["fingerprint"] == evs[1]["fingerprint"]  # same value
    assert evs[0]["event_key"] != evs[1]["event_key"]  # different slot (::1, ::2)


def test_event_fingerprint_ignores_provenance_fields() -> None:
    assert event_content_hash(
        {"diagnosis": "ccRCC", "review_anchor": ["quoted one"]}
    ) == event_content_hash({"diagnosis": "ccRCC", "review_anchor": ["quoted two"]})
    assert event_content_hash(
        {"diagnosis": "ccRCC", "evidence": ["quote"]}
    ) == event_content_hash({"diagnosis": "ccRCC"})


def test_record_to_slots_matches_package_events() -> None:
    field_schema = {
        "record_x": {"label": "X", "event_identity_fields": ["sid"], "fields": []}
    }
    record = _record(
        "N1",
        "M1",
        {"record_x": [{"sid": "A", "value": 1}, {"sid": "A", "value": 2}]},
    )

    slots = record_to_slots(record, field_schema)
    package = build_review_package(
        definition_name="X",
        batch="v1",
        records=[record],
        field_schema=field_schema,
        notes={},
    )

    assert [slot.as_package_event() for slot in slots] == package["patients"][0][
        "events"
    ]
    assert slots[0].fingerprint == event_content_hash({"sid": "A", "value": 1})
    assert slots[1].event_key.endswith("::1")


def test_record_to_slots_state_preserves_batch_key_sequence() -> None:
    records = [
        _record("N1", "M1", {"record_x": [{"value": "first"}]}),
        _record("N1", "M1", {"record_x": [{"value": "second"}]}),
    ]
    state = SlotState()
    slots = [
        slot for record in records for slot in record_to_slots(record, {}, state=state)
    ]

    package = build_review_package(
        definition_name="X",
        batch="v1",
        records=records,
        field_schema={},
        notes={},
    )

    assert [slot.event_key for slot in slots] == [
        "N1::record_x::1",
        "N1::record_x::2",
    ]
    assert [slot.as_package_event() for slot in slots] == package["patients"][0][
        "events"
    ]


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
        f["name"]: f["control"] for f in pkg["field_schema"]["surprise_tool"]["fields"]
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
                {
                    "report_id": "KP-001",
                    "report_text": "DIAGNOSIS: RCC.",
                    "mrn": "M0001",
                }
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


def test_default_package_path_segment_includes_batch_name(tmp_path: Path) -> None:
    segment = tmp_path / "inbox" / "fc_extractions" / "kidney" / "001.jsonl"
    assert default_package_path(segment) == segment.with_name(
        "kidney.001.review_pkg.json"
    )


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


def test_package_from_jsonl_only_flagged_reports(tmp_path: Path) -> None:
    batch = tmp_path / "v3.jsonl"
    records = [
        {
            "note_id": "N1",
            "mrn": "M1",
            "success": True,
            "definition_name": "Flagged",
            "events": {"rec": [{"note_id": "N1", "value": "routine"}]},
        },
        {
            "note_id": "N2",
            "mrn": "M2",
            "success": True,
            "definition_name": "Flagged",
            "events": {
                "rec": [{"note_id": "N2", "value": "ambiguous"}],
                "flag_report_for_review": [
                    {
                        "note_id": "N2",
                        "comment": "ambiguous text",
                        "reason": "conflicting information",
                    }
                ],
            },
        },
    ]
    with batch.open("w") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")

    out = package_from_jsonl(jsonl_path=batch, only_flagged=True)
    pkg = json.loads(out.read_text())

    assert [patient["mrn"] for patient in pkg["patients"]] == ["M2"]
    event_types = [event["event_type"] for event in pkg["patients"][0]["events"]]
    assert event_types == ["rec", "flag_report_for_review"]


def test_package_from_jsonl_note_id_filter(tmp_path: Path) -> None:
    batch = tmp_path / "v4.jsonl"
    records = [
        {
            "note_id": "N1",
            "mrn": "M1",
            "success": True,
            "definition_name": "Selected",
            "events": {"rec": [{"note_id": "N1", "value": "a"}]},
        },
        {
            "note_id": "N2",
            "mrn": "M2",
            "success": True,
            "definition_name": "Selected",
            "events": {"rec": [{"note_id": "N2", "value": "b"}]},
        },
    ]
    with batch.open("w") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")

    out = package_from_jsonl(jsonl_path=batch, note_ids={"N2"})
    pkg = json.loads(out.read_text())

    assert [patient["mrn"] for patient in pkg["patients"]] == ["M2"]


# ---------------------------------------------------------------------------
# adjudication_hash — the comparability contract for cross-run review
# ---------------------------------------------------------------------------


class _DxAB(ExtractionEvent):
    dx: Literal["a", "b"] = Field(..., description="diagnosis")


class _DxABC(ExtractionEvent):
    dx: Literal["a", "b", "c"] = Field(..., description="diagnosis")  # extra option


class _DxRenamed(ExtractionEvent):
    diagnosis: Literal["a", "b"] = Field(..., description="diagnosis")  # renamed


def _reg(model, *, identity=(), comparison=()) -> ToolRegistry:
    reg = ToolRegistry(single_note=True)
    reg.register(
        name="record",
        description="d",
        model=model,
        event_identity_fields=identity,
        comparison_fields=comparison,
    )
    return reg


def test_adjudication_hash_is_stable_and_schema_sensitive() -> None:
    base = adjudication_hash_from_registry("D", _reg(_DxAB))

    # Stable for the same schema.
    assert base == adjudication_hash_from_registry("D", _reg(_DxAB))
    # Sensitive to enum option sets, field names, identity + comparison config,
    # and definition name.
    assert base != adjudication_hash_from_registry("D", _reg(_DxABC))
    assert base != adjudication_hash_from_registry("D", _reg(_DxRenamed))
    assert base != adjudication_hash_from_registry("D", _reg(_DxAB, identity=("dx",)))
    assert base != adjudication_hash_from_registry("D", _reg(_DxAB, comparison=("dx",)))
    assert base != adjudication_hash_from_registry("OTHER", _reg(_DxAB))


def test_adjudication_hash_includes_normalization_version() -> None:
    fs = build_field_schema(_reg(_DxAB))
    assert adjudication_hash("D", fs, normalization_version="1") != adjudication_hash(
        "D", fs, normalization_version="2"
    )


def test_adjudication_hash_ignores_run_params() -> None:
    # The crux: same definition + same schema, but different batch name and run
    # date → the SAME adjudication hash. So two models' runs are comparable.
    records = [_record("N1", "M1", {"record_diagnosis": [{"diagnosis_name": "x"}]})]
    fs = build_field_schema(_example_registry())

    pkg_gpt = build_review_package(
        definition_name="Example",
        batch="kidney_gpt5mini",
        records=records,
        field_schema=fs,
        notes={},
        generated_at="2026-01-01T00:00:00+00:00",
    )
    pkg_gemma = build_review_package(
        definition_name="Example",
        batch="kidney_gemma4",
        records=records,
        field_schema=fs,
        notes={},
        generated_at="2026-02-02T00:00:00+00:00",
    )

    assert pkg_gpt["adjudication_hash"]  # present in the package
    assert pkg_gpt["adjudication_hash"] == pkg_gemma["adjudication_hash"]
