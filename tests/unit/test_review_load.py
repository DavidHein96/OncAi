"""Tests for loading completed review-app output into reviewed (silver) tables."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from oncai.ingest import run_ingest
from oncai.review.load import (
    REVIEW_LOG_SUFFIX,
    REVIEW_PACKAGE_SUFFIX,
    review_to_silver_df,
    review_to_silver_parquet,
)
from oncai.sidecar import sidecar_path


def _raw_record(
    note_id: str,
    diagnosis: str,
    *,
    mrn: str = "MRN001",
    second_diagnosis: str | None = None,
    include_provenance: bool = False,
) -> dict:
    events = [{"note_id": note_id, "diagnosis": diagnosis}]
    if include_provenance:
        events[0]["evidence"] = ["diagnosis quote"]
        events[0]["review_anchor"] = ["event quote"]
    if second_diagnosis is not None:
        events.append({"note_id": note_id, "diagnosis": second_diagnosis})
    return {
        "note_id": note_id,
        "mrn": mrn,
        "success": True,
        "definition_name": "PathKidneyBasic",
        "events": {"record_diagnosis": events},
    }


def _write_raw_jsonl(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    return path


def _review_package(*, include_second: bool = True, batch: str = "demo.001") -> dict:
    events = [
        {
            "event_key": "N1::record_diagnosis::1",
            "event_type": "record_diagnosis",
            "note_id": "N1",
            "fields": {
                "note_id": "N1",
                "comment": "source rationale",
                "diagnosis": "clear cell renal cell carcinoma",
                "diagnosis_date": {
                    "date": "2024",
                    "precision": 1,
                    "anchor": "BOY",
                },
                "evidence": ["clear cell", "renal cell carcinoma"],
            },
        },
    ]
    if include_second:
        events.append(
            {
                "event_key": "N1::record_diagnosis::2",
                "event_type": "record_diagnosis",
                "note_id": "N1",
                "fields": {
                    "note_id": "N1",
                    "comment": "uncertain duplicate",
                    "diagnosis": "oncocytoma",
                },
            }
        )

    return {
        "definition_name": "PathKidneyBasic",
        "batch": batch,
        "generated_at": "2026-06-07T12:00:00+00:00",
        "field_schema": {},
        "patients": [
            {
                "mrn": "MRN001",
                "notes": {
                    "N1": {
                        "note_text": "diagnosis text",
                        "mrn": "MRN001",
                        "note_date": "2024-02-03",
                        "note_type": "pathology",
                        "department": "Pathology",
                    }
                },
                "events": events,
            }
        ],
    }


def _reviews() -> list[dict]:
    return [
        {
            "event_key": "N1::record_diagnosis::1",
            "mrn": "MRN001",
            "event_type": "record_diagnosis",
            "note_id": "N1",
            "verdict": "approved",
            "edits": {"diagnosis": "old value"},
            "comment": "first pass",
            "reviewer": "reviewer-a",
            "reviewed_at": "2026-06-08T00:00:00Z",
        },
        {
            "event_key": "N1::record_diagnosis::1",
            "mrn": "MRN001",
            "event_type": "record_diagnosis",
            "note_id": "N1",
            "verdict": "approved",
            "edits": {"diagnosis": "edited clear cell carcinoma"},
            "comment": "fixed",
            "reviewer": "reviewer-b",
            "reviewed_at": "2026-06-08T01:00:00Z",
        },
        {
            "event_key": "N1::record_diagnosis::2",
            "mrn": "MRN001",
            "event_type": "record_diagnosis",
            "note_id": "N1",
            "verdict": "rejected",
            "edits": {},
            "comment": "duplicate",
            "reviewer": "reviewer-b",
            "reviewed_at": "2026-06-08T01:05:00Z",
        },
        {
            "event_key": "N9::unknown::1",
            "mrn": "MRN999",
            "event_type": "unknown",
            "note_id": "N9",
            "verdict": "approved",
            "edits": {},
            "comment": "",
            "reviewer": "reviewer-b",
            "reviewed_at": "2026-06-08T01:06:00Z",
        },
    ]


def _write_package_reviews_and_raw(tmp_path: Path) -> tuple[Path, Path, Path]:
    package_path = tmp_path / f"demo.001{REVIEW_PACKAGE_SUFFIX}"
    reviews_path = tmp_path / f"demo.001{REVIEW_LOG_SUFFIX}"
    raw_path = tmp_path / "001.jsonl"
    package_path.write_text(json.dumps(_review_package()))
    reviews_path.write_text("\n".join(json.dumps(r) for r in _reviews()) + "\n")
    _write_raw_jsonl(
        raw_path,
        [
            _raw_record(
                "N1",
                "clear cell renal cell carcinoma",
                second_diagnosis="oncocytoma",
            )
        ],
    )
    return package_path, reviews_path, raw_path


def test_review_to_silver_df_applies_last_approved_edits(tmp_path: Path) -> None:
    package_path, reviews_path, raw_path = _write_package_reviews_and_raw(tmp_path)

    result = review_to_silver_df(package_path, reviews_path, raw_path)

    assert result.total_events == 2
    assert result.reviewed_events == 2
    assert result.approved_events == 1
    assert result.auto_accepted_events == 0
    assert result.rejected_events == 1
    assert result.ignored_reviews == 1
    assert result.df is not None
    assert result.df.height == 1

    row = result.df.row(0, named=True)
    assert row["event_key"] == "N1::record_diagnosis::1"
    assert row["review_verdict"] == "approved"
    assert row["acceptance_reason"] == "flagged_reviewed"
    assert row["reviewer"] == "reviewer-b"
    assert row["review_comment"] == "fixed"
    assert row["diagnosis"] == "edited clear cell carcinoma"
    assert json.loads(row["reviewed_fields_json"])["diagnosis"] == (
        "edited clear cell carcinoma"
    )
    assert isinstance(row["key_hash"], bytes)
    assert isinstance(row["content_hash"], bytes)


def test_review_to_silver_df_auto_accepts_raw_slots_outside_worklist(
    tmp_path: Path,
) -> None:
    package_path = tmp_path / f"demo.001{REVIEW_PACKAGE_SUFFIX}"
    reviews_path = tmp_path / f"demo.001{REVIEW_LOG_SUFFIX}"
    raw_path = tmp_path / "001.jsonl"
    package_path.write_text(
        json.dumps(_review_package(include_second=False, batch="demo.001"))
    )
    reviews_path.write_text(
        json.dumps(
            {
                "event_key": "N1::record_diagnosis::1",
                "verdict": "approved",
                "edits": {},
                "reviewed_at": "2026-06-08T01:00:00Z",
            }
        )
        + "\n"
    )
    _write_raw_jsonl(
        raw_path,
        [
            _raw_record("N1", "reviewed diagnosis"),
            _raw_record("N2", "quiet diagnosis", mrn="MRN002"),
        ],
    )

    result = review_to_silver_df(package_path, reviews_path, raw_path)

    assert result.written == 2
    assert result.approved_events == 1
    assert result.auto_accepted_events == 1
    rows = {row["event_key"]: row for row in result.df.iter_rows(named=True)}
    assert rows["N1::record_diagnosis::1"]["acceptance_reason"] == "flagged_reviewed"
    assert rows["N2::record_diagnosis::1"]["acceptance_reason"] == (
        "unflagged_autoaccept"
    )
    assert rows["N2::record_diagnosis::1"]["review_verdict"] == ""
    assert rows["N2::record_diagnosis::1"]["diagnosis"] == "quiet diagnosis"


def test_review_to_silver_excludes_provenance_from_gold_fields(tmp_path: Path) -> None:
    package_path = tmp_path / f"demo.001{REVIEW_PACKAGE_SUFFIX}"
    reviews_path = tmp_path / f"demo.001{REVIEW_LOG_SUFFIX}"
    raw_path = tmp_path / "001.jsonl"
    package_path.write_text(
        json.dumps(_review_package(include_second=False, batch="demo.001"))
    )
    reviews_path.write_text(
        json.dumps(
            {
                "event_key": "N1::record_diagnosis::1",
                "verdict": "approved",
                "edits": {},
                "reviewed_at": "2026-06-08T01:00:00Z",
            }
        )
        + "\n"
    )
    _write_raw_jsonl(
        raw_path,
        [_raw_record("N1", "reviewed diagnosis", include_provenance=True)],
    )

    result = review_to_silver_df(package_path, reviews_path, raw_path)

    row = result.df.row(0, named=True)
    assert "evidence" not in result.df.columns
    assert "review_anchor" not in result.df.columns
    assert "evidence" not in json.loads(row["reviewed_fields_json"])
    assert "review_anchor" not in json.loads(row["reviewed_fields_json"])


def test_review_to_silver_includes_reviewer_added_events(tmp_path: Path) -> None:
    package_path = tmp_path / f"demo.001{REVIEW_PACKAGE_SUFFIX}"
    reviews_path = tmp_path / f"demo.001{REVIEW_LOG_SUFFIX}"
    raw_path = tmp_path / "001.jsonl"
    package_path.write_text(
        json.dumps(_review_package(include_second=False, batch="demo.001"))
    )
    reviews_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event_key": "N1::record_diagnosis::1",
                        "verdict": "approved",
                        "edits": {},
                        "reviewed_at": "2026-06-08T01:00:00Z",
                    }
                ),
                json.dumps(
                    {
                        "event_key": "__new__::N1::record_diagnosis::manual",
                        "mrn": "MRN001",
                        "event_type": "record_diagnosis",
                        "note_id": "N1",
                        "verdict": "approved",
                        "is_new_event": True,
                        "edits": {"diagnosis": "reviewer added diagnosis"},
                        "comment": "missed by extraction",
                        "reviewer": "reviewer-b",
                        "reviewed_at": "2026-06-08T01:05:00Z",
                    }
                ),
            ]
        )
        + "\n"
    )
    _write_raw_jsonl(raw_path, [_raw_record("N1", "reviewed diagnosis")])

    result = review_to_silver_df(package_path, reviews_path, raw_path)

    assert result.total_events == 2
    assert result.approved_events == 2
    assert result.ignored_reviews == 0
    rows = {row["event_key"]: row for row in result.df.iter_rows(named=True)}
    added = rows["__new__::N1::record_diagnosis::manual"]
    assert added["acceptance_reason"] == "reviewer_added"
    assert added["diagnosis"] == "reviewer added diagnosis"
    assert added["review_comment"] == "missed by extraction"


def test_review_to_silver_requires_complete_review(tmp_path: Path) -> None:
    package_path = tmp_path / f"demo.001{REVIEW_PACKAGE_SUFFIX}"
    reviews_path = tmp_path / f"demo.001{REVIEW_LOG_SUFFIX}"
    raw_path = tmp_path / "001.jsonl"
    package_path.write_text(json.dumps(_review_package()))
    reviews_path.write_text(json.dumps(_reviews()[0]) + "\n")
    _write_raw_jsonl(
        raw_path,
        [_raw_record("N1", "first", second_diagnosis="second")],
    )

    with pytest.raises(ValueError, match="unreviewed"):
        review_to_silver_df(package_path, reviews_path, raw_path)


def test_review_to_silver_fails_when_package_does_not_match_raw_segment(
    tmp_path: Path,
) -> None:
    package_path = tmp_path / f"demo.001{REVIEW_PACKAGE_SUFFIX}"
    reviews_path = tmp_path / f"demo.001{REVIEW_LOG_SUFFIX}"
    raw_path = tmp_path / "001.jsonl"
    package_path.write_text(json.dumps(_review_package(include_second=False)))
    reviews_path.write_text(json.dumps(_reviews()[0]) + "\n")
    _write_raw_jsonl(raw_path, [_raw_record("OTHER", "diagnosis")])

    with pytest.raises(ValueError, match="not present in raw segment"):
        review_to_silver_df(package_path, reviews_path, raw_path)


def test_review_to_silver_parquet_writes_gold_rows(tmp_path: Path) -> None:
    package_path, reviews_path, raw_path = _write_package_reviews_and_raw(tmp_path)
    out = tmp_path / "demo.parquet"

    result = review_to_silver_parquet(package_path, reviews_path, raw_path, out)

    assert result.written == 1
    df = pl.read_parquet(out)
    assert df.height == 1
    assert df.row(0, named=True)["batch_name"] == "demo.001"


def _review_dir(config, batch_stem: str):
    """The one canonical inbox folder for a review pair: fc_reviews/<base>/.

    ``batch_stem`` is the per-segment stem (e.g. ``demo.001``); the folder is
    named for the base batch (``demo``), matching what the run hook creates.
    """
    base = batch_stem.rsplit(".", 1)[0]
    d = config.inbox_path / "fc_reviews" / base
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_ingest_fc_reviews_writes_lake_parquet_and_sidecars(oncai_config) -> None:
    batch_dir = _review_dir(oncai_config, "demo.001")
    package_path = batch_dir / f"demo.001{REVIEW_PACKAGE_SUFFIX}"
    reviews_path = batch_dir / f"demo.001{REVIEW_LOG_SUFFIX}"
    raw_path = oncai_config.inbox_path / "fc_extractions" / "demo" / "001.jsonl"
    sql_path = batch_dir / "demo.sql"
    package_path.write_text(json.dumps(_review_package(batch="demo.001")))
    reviews_path.write_text("\n".join(json.dumps(r) for r in _reviews()) + "\n")
    sql_path.write_text(
        "CREATE OR REPLACE TABLE extractions_gold.demo AS SELECT 1 AS n;"
    )
    _write_raw_jsonl(
        raw_path,
        [
            _raw_record(
                "N1",
                "clear cell renal cell carcinoma",
                second_diagnosis="oncocytoma",
            )
        ],
    )

    results = run_ingest(oncai_config, folder="fc_reviews")

    assert len(results) == 1
    result = results[0]
    assert result.files[0].new_rows == 1
    assert result.files[0].unchanged_rows == 1
    assert any("rejected event" in note for note in result.notes)
    assert any("ignored 1 review" in note for note in result.notes)
    assert sidecar_path(package_path).exists()
    assert sidecar_path(reviews_path).exists()

    lake_parquet = oncai_config.lake_path / "fc_reviews" / "demo.parquet"
    assert lake_parquet.exists()
    df = pl.read_parquet(lake_parquet)
    assert df.height == 1
    assert df.row(0, named=True)["diagnosis"] == "edited clear cell carcinoma"

    lake_sql = oncai_config.lake_path / "fc_reviews" / "demo.sql"
    assert lake_sql.read_text() == sql_path.read_text()
    assert any(
        "demo/demo.sql: SQL transform mirrored to lake as demo.sql" in note
        for note in result.notes
    )


def test_ingest_fc_reviews_ignores_flat_layout(oncai_config) -> None:
    # The only supported layout is fc_reviews/<batch>/; a flat pair is not found.
    inbox = oncai_config.inbox_path / "fc_reviews"
    inbox.mkdir(parents=True, exist_ok=True)
    (inbox / f"demo.001{REVIEW_PACKAGE_SUFFIX}").write_text(
        json.dumps(_review_package(batch="demo.001"))
    )
    (inbox / f"demo.001{REVIEW_LOG_SUFFIX}").write_text(
        "\n".join(json.dumps(r) for r in _reviews()) + "\n"
    )

    results = run_ingest(oncai_config, folder="fc_reviews")

    assert results[0].files == []
    assert not (oncai_config.lake_path / "fc_reviews" / "demo.parquet").exists()


def test_ingest_fc_reviews_skips_and_prunes_tombstoned_batch(oncai_config) -> None:
    from oncai.tombstones import TombstoneAction, write_tombstone_event

    batch_dir = _review_dir(oncai_config, "demo.001")
    (batch_dir / f"demo.001{REVIEW_PACKAGE_SUFFIX}").write_text(
        json.dumps(_review_package(batch="demo.001"))
    )
    (batch_dir / f"demo.001{REVIEW_LOG_SUFFIX}").write_text(
        "\n".join(json.dumps(r) for r in _reviews()) + "\n"
    )
    raw_path = oncai_config.inbox_path / "fc_extractions" / "demo" / "001.jsonl"
    _write_raw_jsonl(
        raw_path,
        [_raw_record("N1", "clear cell renal cell carcinoma", second_diagnosis="onc")],
    )

    run_ingest(oncai_config, folder="fc_reviews")
    out = oncai_config.lake_path / "fc_reviews" / "demo.parquet"
    assert out.exists()

    # Forget by the base batch name (the fc_reviews/<batch>/ folder name).
    write_tombstone_event(
        oncai_config,
        kind="fc_reviews",
        target="demo",
        action=TombstoneAction.FORGET,
        actor="test",
        at="2026-06-12T12:00:00Z",
        event_id="aaaaaaaaaaaaaaaa",
    )
    results = run_ingest(oncai_config, folder="fc_reviews")

    assert not out.exists()
    notes = " ".join(results[0].notes)
    assert "demo: skipped because tombstoned" in notes
    assert "demo: pruned lake projection — tombstoned" in notes


def test_event_key_round_trips_to_gold(tmp_path: Path) -> None:
    from oncai.review.package import build_review_package

    rec = _raw_record("N1", "ccRCC", mrn="M1")
    pkg = build_review_package(
        definition_name="D", batch="v1.001", records=[rec], field_schema={}, notes={}
    )
    key = pkg["patients"][0]["events"][0]["event_key"]

    pkg_path = tmp_path / f"v1.001{REVIEW_PACKAGE_SUFFIX}"
    rev_path = tmp_path / f"v1.001{REVIEW_LOG_SUFFIX}"
    raw_path = tmp_path / "001.jsonl"
    pkg_path.write_text(json.dumps(pkg))
    rev_path.write_text(
        json.dumps(
            {"event_key": key, "verdict": "approved", "edits": {}, "reviewed_at": "t"}
        )
        + "\n"
    )
    _write_raw_jsonl(raw_path, [rec])

    result = review_to_silver_df(pkg_path, rev_path, raw_path)
    assert result.approved_events == 1
    row = result.df.row(0, named=True)
    assert row["event_key"] == key
    assert row["diagnosis"] == "ccRCC"


def test_latest_reviewed_at_wins_regardless_of_line_order(tmp_path: Path) -> None:
    package_path = tmp_path / f"demo.001{REVIEW_PACKAGE_SUFFIX}"
    reviews_path = tmp_path / f"demo.001{REVIEW_LOG_SUFFIX}"
    raw_path = tmp_path / "001.jsonl"
    package_path.write_text(json.dumps(_review_package(include_second=False)))
    later = {
        "event_key": "N1::record_diagnosis::1",
        "verdict": "approved",
        "edits": {"diagnosis": "FINAL"},
        "reviewer": "late",
        "reviewed_at": "2026-06-08T05:00:00Z",
    }
    earlier = {
        "event_key": "N1::record_diagnosis::1",
        "verdict": "approved",
        "edits": {"diagnosis": "OLD"},
        "reviewer": "early",
        "reviewed_at": "2026-06-08T01:00:00Z",
    }
    reviews_path.write_text(json.dumps(later) + "\n" + json.dumps(earlier) + "\n")
    _write_raw_jsonl(raw_path, [_raw_record("N1", "source")])

    row = review_to_silver_df(package_path, reviews_path, raw_path).df.row(0, named=True)
    assert row["reviewer"] == "late"
    assert row["diagnosis"] == "FINAL"


def _seg_pkg(batch: str, events: list[dict]) -> dict:
    return {
        "definition_name": "D",
        "batch": batch,
        "generated_at": "2026-01-01T00:00:00+00:00",
        "field_schema": {},
        "patients": [{"mrn": "M1", "notes": {}, "events": events}],
    }


def _seg_event(note_id: str, diagnosis: str) -> dict:
    return {
        "event_key": f"{note_id}::record_diagnosis::1",
        "event_type": "record_diagnosis",
        "note_id": note_id,
        "fields": {"diagnosis": diagnosis},
    }


def _approve(key: str, at: str) -> str:
    return json.dumps(
        {"event_key": key, "verdict": "approved", "edits": {}, "reviewed_at": at}
    )


def test_ingest_fc_reviews_merges_segments_into_one_gold_table(oncai_config) -> None:
    inbox = _review_dir(oncai_config, "kidney.001")
    raw_dir = oncai_config.inbox_path / "fc_extractions" / "kidney"

    _write_raw_jsonl(raw_dir / "001.jsonl", [_raw_record("N1", "OLD", mrn="M1")])
    _write_raw_jsonl(
        raw_dir / "002.jsonl",
        [_raw_record("N1", "NEW", mrn="M1"), _raw_record("N2", "Z", mrn="M1")],
    )
    (inbox / "kidney.001.review_pkg.json").write_text(
        json.dumps(_seg_pkg("kidney.001", [_seg_event("N1", "OLD")]))
    )
    (inbox / "kidney.001.reviews.jsonl").write_text(
        _approve("N1::record_diagnosis::1", "2026-01-01T00:00:00Z") + "\n"
    )
    (inbox / "kidney.002.review_pkg.json").write_text(
        json.dumps(
            _seg_pkg(
                "kidney.002",
                [_seg_event("N1", "NEW"), _seg_event("N2", "Z")],
            )
        )
    )
    (inbox / "kidney.002.reviews.jsonl").write_text(
        _approve("N1::record_diagnosis::1", "2026-01-02T00:00:00Z")
        + "\n"
        + _approve("N2::record_diagnosis::1", "2026-01-02T00:00:00Z")
        + "\n"
    )

    run_ingest(oncai_config, folder="fc_reviews")

    out = oncai_config.lake_path / "fc_reviews" / "kidney.parquet"
    assert out.exists()
    assert not (oncai_config.lake_path / "fc_reviews" / "kidney.001.parquet").exists()

    by_note = {r["note_id"]: r for r in pl.read_parquet(out).iter_rows(named=True)}
    assert set(by_note) == {"N1", "N2"}
    assert by_note["N1"]["diagnosis"] == "NEW"
    assert by_note["N1"]["batch_name"] == "kidney.002"


def test_ingest_fc_reviews_fails_naming_incomplete_batch(oncai_config) -> None:
    good_dir = _review_dir(oncai_config, "good.001")
    bad_dir = _review_dir(oncai_config, "bad.001")
    good_raw = oncai_config.inbox_path / "fc_extractions" / "good" / "001.jsonl"
    bad_raw = oncai_config.inbox_path / "fc_extractions" / "bad" / "001.jsonl"
    _write_raw_jsonl(good_raw, [_raw_record("N1", "good")])
    _write_raw_jsonl(bad_raw, [_raw_record("N1", "first", second_diagnosis="second")])

    (good_dir / "good.001.review_pkg.json").write_text(
        json.dumps(_review_package(include_second=False, batch="good.001"))
    )
    (good_dir / "good.001.reviews.jsonl").write_text(
        _approve("N1::record_diagnosis::1", "2026-01-01T00:00:00Z") + "\n"
    )
    (bad_dir / "bad.001.review_pkg.json").write_text(
        json.dumps(_review_package(batch="bad.001"))
    )
    (bad_dir / "bad.001.reviews.jsonl").write_text(
        _approve("N1::record_diagnosis::1", "2026-01-01T00:00:00Z") + "\n"
    )

    with pytest.raises(ValueError) as exc:
        run_ingest(oncai_config, folder="fc_reviews")
    msg = str(exc.value)
    assert "bad" in msg
    assert "good" not in msg
    assert (oncai_config.lake_path / "fc_reviews" / "good.parquet").exists()
    assert not (oncai_config.lake_path / "fc_reviews" / "bad.parquet").exists()


def test_ingest_fc_reviews_skips_package_without_reviews(oncai_config) -> None:
    pending_dir = _review_dir(oncai_config, "pending.001")
    done_dir = _review_dir(oncai_config, "done.001")
    done_raw = oncai_config.inbox_path / "fc_extractions" / "done" / "001.jsonl"
    _write_raw_jsonl(done_raw, [_raw_record("N1", "done")])
    (pending_dir / "pending.001.review_pkg.json").write_text(
        json.dumps(_review_package())
    )
    (done_dir / "done.001.review_pkg.json").write_text(
        json.dumps(_review_package(include_second=False, batch="done.001"))
    )
    (done_dir / "done.001.reviews.jsonl").write_text(
        _approve("N1::record_diagnosis::1", "2026-01-01T00:00:00Z") + "\n"
    )

    results = run_ingest(oncai_config, folder="fc_reviews")

    assert (oncai_config.lake_path / "fc_reviews" / "done.parquet").exists()
    assert not (oncai_config.lake_path / "fc_reviews" / "pending.parquet").exists()
    notes = " ".join(results[0].notes)
    assert "pending.001" in notes and "missing" in notes
