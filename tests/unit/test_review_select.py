"""Tests for selecting extraction records that need review."""

from __future__ import annotations

import json
from pathlib import Path

from oncai.review.select import (
    comparable_fields_from_field_schema,
    disagreement_note_ids,
    flagged_note_ids_from_jsonl,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")


def _record(note_id: str, events: dict) -> dict:
    return {
        "note_id": note_id,
        "mrn": f"M-{note_id}",
        "success": True,
        "events": events,
    }


def test_flagged_note_ids_from_jsonl(tmp_path: Path) -> None:
    batch = tmp_path / "run.jsonl"
    _write_jsonl(
        batch,
        [
            _record("N1", {"rec": [{"note_id": "N1", "value": "a"}]}),
            _record(
                "N2",
                {
                    "rec": [{"note_id": "N2", "value": "b"}],
                    "flag_report_for_review": [
                        {
                            "note_id": "N2",
                            "comment": "ambiguous",
                            "reason": "conflicting information",
                        }
                    ],
                },
            ),
        ],
    )

    assert flagged_note_ids_from_jsonl(batch) == {"N2"}


def test_disagreement_note_ids_uses_explicit_comparable_fields(tmp_path: Path) -> None:
    primary = tmp_path / "primary.jsonl"
    compare = tmp_path / "compare.jsonl"
    _write_jsonl(
        primary,
        [
            _record(
                "N1",
                {
                    "rec": [
                        {"note_id": "N1", "value": "a", "comment": "first"},
                        {"note_id": "N1", "value": "b", "comment": "second"},
                    ]
                },
            ),
            _record("N2", {"rec": [{"note_id": "N2", "score": 1}]}),
            _record("N3", {"rec": [{"note_id": "N3", "value": "present"}]}),
        ],
    )
    _write_jsonl(
        compare,
        [
            _record(
                "N1",
                {
                    "rec": [
                        {"note_id": "N1", "value": "b", "comment": "changed"},
                        {"note_id": "N1", "value": "a", "comment": "changed"},
                    ]
                },
            ),
            _record("N2", {"rec": [{"note_id": "N2", "score": 2}]}),
        ],
    )

    assert disagreement_note_ids(
        primary,
        [compare],
        comparable_fields={"rec": {"score"}},
    ) == {"N2", "N3"}


def test_disagreement_note_ids_ignores_unselected_primitive_fields(tmp_path: Path) -> None:
    primary = tmp_path / "primary.jsonl"
    compare = tmp_path / "compare.jsonl"
    _write_jsonl(primary, [_record("N1", {"rec": [{"note_id": "N1", "score": 1}]})])
    _write_jsonl(compare, [_record("N1", {"rec": [{"note_id": "N1", "score": 2}]})])

    assert disagreement_note_ids(primary, [compare]) == set()


def test_disagreement_note_ids_ignores_plain_strings(tmp_path: Path) -> None:
    primary = tmp_path / "primary.jsonl"
    compare = tmp_path / "compare.jsonl"
    _write_jsonl(primary, [_record("N1", {"rec": [{"note_id": "N1", "text": "a"}]})])
    _write_jsonl(compare, [_record("N1", {"rec": [{"note_id": "N1", "text": "b"}]})])

    assert disagreement_note_ids(primary, [compare]) == set()


def test_disagreement_note_ids_compares_schema_enum_strings(tmp_path: Path) -> None:
    primary = tmp_path / "primary.jsonl"
    compare = tmp_path / "compare.jsonl"
    _write_jsonl(
        primary,
        [_record("N1", {"rec": [{"note_id": "N1", "category": "positive"}]})],
    )
    _write_jsonl(
        compare,
        [_record("N1", {"rec": [{"note_id": "N1", "category": "negative"}]})],
    )

    assert disagreement_note_ids(
        primary,
        [compare],
        comparable_fields={"rec": {"category"}},
    ) == {"N1"}


def test_comparable_fields_include_text_identity_fields() -> None:
    field_schema = {
        "record_ihc_result": {
            "label": "Record Ihc Result",
            "event_identity_fields": ["specimen_id", "standardized_test_name"],
            "comparison_fields": ["standardized_test_status"],
            "fields": [
                {"name": "specimen_id", "control": "text"},
                {"name": "standardized_test_name", "control": "enum"},
                {"name": "standardized_test_status", "control": "enum"},
                {"name": "standardized_test_intensity", "control": "enum"},
                {"name": "given_result", "control": "text"},
            ],
        }
    }

    assert comparable_fields_from_field_schema(field_schema) == {
        "record_ihc_result": {
            "specimen_id",
            "standardized_test_name",
            "standardized_test_status",
        }
    }


def test_disagreement_note_ids_compares_text_identity_fields(tmp_path: Path) -> None:
    primary = tmp_path / "primary.jsonl"
    compare = tmp_path / "compare.jsonl"
    event = {
        "standardized_test_name": "CA-IX",
        "standardized_test_status": "Positive",
    }
    _write_jsonl(
        primary,
        [_record("N1", {"record_ihc_result": [event | {"specimen_id": "A"}]})],
    )
    _write_jsonl(
        compare,
        [_record("N1", {"record_ihc_result": [event | {"specimen_id": "B"}]})],
    )

    assert disagreement_note_ids(
        primary,
        [compare],
        comparable_fields={
            "record_ihc_result": {
                "specimen_id",
                "standardized_test_name",
                "standardized_test_status",
            }
        },
    ) == {"N1"}
