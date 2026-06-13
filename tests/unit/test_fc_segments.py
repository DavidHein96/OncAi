"""Tests for the batch-as-folder-of-segments model.

Covers the segment-order merge (load), the folder ingest, and the CLI helpers
that compute the delta / claim a segment / promote a working JSONL.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
import typer

from oncai.cli.fc_cmds import (
    _categorize_delta,
    _check_or_write_batch_descriptor,
    _finalize_segment,
    _load_batch_history,
    _next_segment,
)
from oncai.fc_extraction.load import merge_segments_to_parquet, segment_files
from oncai.fc_extraction.manifest import (
    definition_hash,
    definition_hash_from_registry,
    tool_schemas_for,
)
from oncai.ingest import run_ingest


def _record(
    note_id: str,
    diagnosis: str,
    *,
    success: bool = True,
    src_hash: str = "h1",
    prompt_hash: str = "p1",
) -> dict:
    return {
        "note_id": note_id,
        "definition_name": "Example",
        "success": success,
        "events": {"record_diagnosis": [{"diagnosis_name": diagnosis}]},
        "finish": None,
        "rounds": 1,
        "input_tokens": 1,
        "output_tokens": 1,
        "reasoning_tokens": 0,
        "extracted_at": "2025-01-01T00:00:00",
        "source_content_hash": src_hash,
        "run_meta": {
            "system_prompt_hash": prompt_hash,
            "definition_hash": prompt_hash,
        },
    }


class _FakeModel:
    def __init__(self, schema: dict):
        self._schema = schema

    def model_json_schema(self) -> dict:
        return self._schema


class _FakeTool:
    def __init__(self, schema: dict):
        self.model = _FakeModel(schema)


class _FakeRegistry:
    """Duck-typed stand-in: tools + an engine builtin that must be excluded."""

    def __init__(self, tools: dict[str, dict]):
        self._tools = tools

    def list_tools(self) -> list[str]:
        return [*self._tools, "finish_single_extraction"]

    def get(self, name: str):
        return _FakeTool(self._tools[name]) if name in self._tools else None


def _write_segment(batch_dir: Path, n: int, records: list[dict]) -> Path:
    batch_dir.mkdir(parents=True, exist_ok=True)
    path = batch_dir / f"{n:03d}.jsonl"
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


# ---------------------------------------------------------------------------
# segment_files + merge_segments_to_parquet
# ---------------------------------------------------------------------------


class TestSegmentMerge:
    def test_segment_files_sorted_by_int(self, tmp_path):
        bd = tmp_path / "v1"
        _write_segment(bd, 2, [_record("R1", "x")])
        _write_segment(bd, 10, [_record("R1", "x")])
        _write_segment(bd, 1, [_record("R1", "x")])
        # A non-segment file is ignored.
        (bd / "batch.json").write_text("{}")
        (bd / "001_manifest.json").write_text("{}")

        nums = [n for n, _ in segment_files(bd)]
        assert nums == [1, 2, 10]

    def test_highest_segment_wins_per_record(self, tmp_path):
        bd = tmp_path / "v1"
        # 001: R1 (old), R2.  002: R1 (re-extracted).
        _write_segment(bd, 1, [_record("R1", "old"), _record("R2", "r2only")])
        _write_segment(bd, 2, [_record("R1", "NEW")])

        result = merge_segments_to_parquet(bd, tmp_path / "out.parquet", dry_run=True)
        df = result.df
        assert df is not None
        assert df.height == 2  # unique record_ids
        rows = {r["record_id"]: r for r in df.iter_rows(named=True)}

        # R1 comes from the higher segment (002), R2 from its only segment (001).
        assert rows["R1"]["segment"] == 2
        assert rows["R2"]["segment"] == 1
        assert "NEW" in rows["R1"]["events_json"]
        assert "old" not in rows["R1"]["events_json"]
        # Provenance: batch_name is the folder, not the segment filename.
        assert rows["R1"]["batch_name"] == "v1"

    def test_failed_records_dropped(self, tmp_path):
        bd = tmp_path / "v1"
        _write_segment(
            bd, 1, [_record("R1", "x"), _record("R2", "y", success=False)]
        )
        result = merge_segments_to_parquet(bd, tmp_path / "out.parquet", dry_run=True)
        assert result.df is not None
        assert result.df.height == 1
        assert result.skipped_failed == 1

    def test_empty_batch_dir(self, tmp_path):
        result = merge_segments_to_parquet(
            tmp_path / "missing", tmp_path / "out.parquet", dry_run=True
        )
        assert result.df is None
        assert result.total_records == 0


# ---------------------------------------------------------------------------
# ingest: a batch folder → one lake parquet
# ---------------------------------------------------------------------------


class TestIngestFcExtractions:
    def test_folder_of_segments_merges_to_parquet(self, oncai_config):
        bd = oncai_config.inbox_path / "fc_extractions" / "mybatch"
        _write_segment(bd, 1, [_record("R1", "old"), _record("R2", "y")])
        _write_segment(bd, 2, [_record("R1", "NEW")])
        sql_path = bd / "mybatch.sql"
        sql_path.write_text(
            "CREATE OR REPLACE TABLE extractions_transformed.mybatch AS "
            "SELECT 1 AS n;"
        )

        results = run_ingest(oncai_config, folder="fc_extractions")
        assert len(results) == 1

        out = oncai_config.lake_path / "fc_extractions" / "mybatch.parquet"
        assert out.exists()
        df = pl.read_parquet(out)
        assert df.height == 2
        r1 = df.filter(pl.col("record_id") == "R1").row(0, named=True)
        assert r1["segment"] == 2
        assert "NEW" in r1["events_json"]

        lake_sql = oncai_config.lake_path / "fc_extractions" / "mybatch.sql"
        assert lake_sql.read_text() == sql_path.read_text()
        assert any(
            "mybatch/mybatch.sql: SQL transform mirrored to lake as mybatch.sql"
            in note
            for note in results[0].notes
        )


# ---------------------------------------------------------------------------
# CLI helpers: segment claim / history / descriptor / finalize
# ---------------------------------------------------------------------------


class TestNextSegment:
    def test_empty_is_one(self, tmp_path):
        assert _next_segment(tmp_path / "v1") == 1

    def test_max_plus_one(self, tmp_path):
        bd = tmp_path / "v1"
        _write_segment(bd, 1, [_record("R1", "x")])
        _write_segment(bd, 2, [_record("R1", "x")])
        assert _next_segment(bd) == 3


class TestLoadBatchHistory:
    def test_collects_success_hashes_across_segments(self, tmp_path):
        bd = tmp_path / "v1"
        _write_segment(
            bd, 1, [_record("R1", "x", src_hash="h1", prompt_hash="p1")]
        )
        _write_segment(
            bd, 2, [_record("R1", "x2", src_hash="h2", prompt_hash="p2")]
        )
        # A failed record contributes nothing to the resume set.
        _write_segment(bd, 3, [_record("R9", "z", success=False)])

        hist = _load_batch_history(bd)
        assert set(hist) == {"R1"}
        assert hist["R1"] == [("h1", "p1"), ("h2", "p2")]


class TestBatchDescriptor:
    def test_writes_then_matches(self, tmp_path):
        bd = tmp_path / "v1"
        _check_or_write_batch_descriptor(
            bd, definition="Example", source="raw.pathology", id_col="report_id"
        )
        assert (bd / "batch.json").exists()
        data = json.loads((bd / "batch.json").read_text())
        assert data["definition"] == "Example"
        assert data["source"] == "raw.pathology"

        # Re-running with identical params is fine (no raise).
        _check_or_write_batch_descriptor(
            bd, definition="Example", source="raw.pathology", id_col="report_id"
        )

    def test_mismatch_bails(self, tmp_path):
        bd = tmp_path / "v1"
        _check_or_write_batch_descriptor(
            bd, definition="Example", source="raw.pathology", id_col="report_id"
        )
        with pytest.raises(typer.Exit):
            _check_or_write_batch_descriptor(
                bd, definition="Other", source="raw.pathology", id_col="report_id"
            )


class TestDefinitionHash:
    def test_changes_with_prompt(self):
        assert definition_hash("p1", {}) != definition_hash("p2", {})

    def test_changes_with_tool_schema(self):
        # The whole point: a changed Pydantic field (here, an added field)
        # changes the hash even when the prompt is identical.
        a = definition_hash("p", {"t": {"fields": ["a"]}})
        b = definition_hash("p", {"t": {"fields": ["a", "b"]}})
        assert a != b

    def test_key_order_independent(self):
        assert definition_hash("p", {"a": 1, "b": 2}) == definition_hash(
            "p", {"b": 2, "a": 1}
        )

    def test_from_registry_excludes_builtins_and_matches(self):
        reg = _FakeRegistry({"record_diagnosis": {"type": "object", "x": 1}})
        schemas = tool_schemas_for(reg)
        # The engine builtin is excluded; the task tool is kept.
        assert set(schemas) == {"record_diagnosis"}
        assert definition_hash_from_registry("p", reg) == definition_hash("p", schemas)

    def test_registry_hash_tracks_field_change(self):
        before = _FakeRegistry({"record_diagnosis": {"fields": ["dx"]}})
        after = _FakeRegistry({"record_diagnosis": {"fields": ["dx", "grade"]}})
        assert definition_hash_from_registry(
            "same prompt", before
        ) != definition_hash_from_registry("same prompt", after)


class TestCategorizeDelta:
    def test_definition_change_re_extracts_when_enabled(self):
        # Same content (h1), but a prior row used an OLD definition hash.
        source_rows = [("R1", "h1")]
        history = {"R1": [("h1", "OLD_DEFN")]}

        delta, counts, _ = _categorize_delta(
            source_rows=source_rows,
            existing=history,
            note_id_set=None,
            definition_hash="NEW_DEFN",
            reextract_on_prompt_change=True,
            forced_rerun_ids=None,
        )
        assert "R1" in delta
        assert counts["definition_changed"] == 1

    def test_definition_change_skipped_when_disabled(self):
        source_rows = [("R1", "h1")]
        history = {"R1": [("h1", "OLD_DEFN")]}

        delta, counts, _ = _categorize_delta(
            source_rows=source_rows,
            existing=history,
            note_id_set=None,
            definition_hash="NEW_DEFN",
            reextract_on_prompt_change=False,
            forced_rerun_ids=None,
        )
        assert "R1" not in delta
        assert counts["skipped"] == 1

    def test_matching_definition_skips(self):
        source_rows = [("R1", "h1")]
        history = {"R1": [("h1", "DEFN")]}
        delta, counts, _ = _categorize_delta(
            source_rows=source_rows,
            existing=history,
            note_id_set=None,
            definition_hash="DEFN",
            reextract_on_prompt_change=True,
            forced_rerun_ids=None,
        )
        assert "R1" not in delta
        assert counts["skipped"] == 1


class TestFinalizeSegment:
    def test_promotes_working_jsonl_and_manifest(self, tmp_path):
        # A working file + manifest in the fc_outputs scratch area.
        work_dir = tmp_path / "fc_outputs" / "Example"
        work_dir.mkdir(parents=True)
        working = work_dir / "mybatch.003.jsonl"
        working.write_text(json.dumps(_record("R1", "x")) + "\n")
        working.with_name("mybatch.003_manifest.json").write_text('{"ok": true}')

        batch_dir = tmp_path / "inbox" / "fc_extractions" / "mybatch"
        final = _finalize_segment(working, batch_dir, 3)

        assert final == batch_dir / "003.jsonl"
        assert final.exists()
        assert (batch_dir / "003_manifest.json").exists()
        # No leftover temp files.
        assert not list(batch_dir.glob("*.tmp"))
