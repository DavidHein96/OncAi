"""Tests for the run logging system."""

from __future__ import annotations

import json

import polars as pl

from oncai.runs import (
    RunLog,
    _generate_run_id,
    _hash_string,
    get_run,
    list_runs,
    log_run,
    update_run,
)

# ---------- _generate_run_id ----------


class TestGenerateRunId:
    def test_deterministic(self):
        id1 = _generate_run_id("fc_workflow", "adj", "v1", "2025-01-01T00:00:00")
        id2 = _generate_run_id("fc_workflow", "adj", "v1", "2025-01-01T00:00:00")
        assert id1 == id2

    def test_length_8(self):
        rid = _generate_run_id("fc_workflow", "adj", "v1", "2025-01-01T00:00:00")
        assert len(rid) == 8

    def test_differs_by_input(self):
        id1 = _generate_run_id("fc_workflow", "adj", "v1", "2025-01-01T00:00:00")
        id2 = _generate_run_id("fc_single", "adj", "v1", "2025-01-01T00:00:00")
        id3 = _generate_run_id("fc_workflow", "adj", "v2", "2025-01-01T00:00:00")
        assert id1 != id2
        assert id1 != id3


# ---------- log_run ----------


class TestLogRun:
    def test_creates_parquet_when_none_exists(self, tmp_path):
        lake = tmp_path / "lake"
        lake.mkdir()

        run = RunLog(
            run_id="abcd1234",
            run_type="fc_workflow",
            name="test",
            batch_name="v1",
            started_at="2025-01-01T00:00:00+00:00",
        )
        path = log_run(run, lake)

        assert path.exists()
        df = pl.read_parquet(path)
        assert df.height == 1
        assert df["run_id"][0] == "abcd1234"

    def test_appends_to_existing(self, tmp_path):
        lake = tmp_path / "lake"
        lake.mkdir()

        run1 = RunLog(
            run_id="aaaa1111",
            run_type="fc_workflow",
            name="a",
            batch_name="v1",
            started_at="2025-01-01T00:00:00",
        )
        run2 = RunLog(
            run_id="bbbb2222",
            run_type="fc_single",
            name="b",
            batch_name="v2",
            started_at="2025-01-02T00:00:00",
        )

        log_run(run1, lake)
        log_run(run2, lake)

        df = pl.read_parquet(lake / "runs" / "runs.parquet")
        assert df.height == 2
        assert set(df["run_id"].to_list()) == {"aaaa1111", "bbbb2222"}

    def test_all_fields_roundtrip(self, tmp_path):
        lake = tmp_path / "lake"
        lake.mkdir()

        run = RunLog(
            run_id="full0001",
            run_type="compression",
            name="oncology_standard",
            batch_name="v3",
            started_at="2025-06-01T10:00:00+00:00",
            completed_at="2025-06-01T10:05:00+00:00",
            duration_seconds=300.5,
            git_commit="abc123def456",
            git_branch="main",
            git_dirty=True,
            code_version="0.1.0",
            backend="azure-responses",
            model="gpt-4o",
            reasoning_effort="medium",
            temperature=0.0,
            source_table="raw.pathology",
            text_column="report_text",
            workers=4,
            definition_path="/some/path.yaml",
            system_prompt="You are a medical AI.",
            system_prompt_hash=_hash_string("You are a medical AI."),
            tools_json=None,
            tool_schemas_json=None,
            db_path="/data/oncai.duckdb",
            mrn_source="limit:100",
            input_count=100,
            items_processed=100,
            items_succeeded=95,
            items_failed=5,
            items_skipped=0,
            total_input_tokens=50000,
            total_output_tokens=10000,
            output_path="/output/v3.jsonl",
            errors_json=json.dumps([{"note_id": "N1", "error": "timeout"}]),
        )
        log_run(run, lake)

        df = pl.read_parquet(lake / "runs" / "runs.parquet")
        row = df.row(0, named=True)

        assert row["run_id"] == "full0001"
        assert row["duration_seconds"] == 300.5
        assert row["items_succeeded"] == 95
        assert row["total_input_tokens"] == 50000
        assert row["git_dirty"] is True
        assert row["system_prompt"] == "You are a medical AI."

    def test_json_fields_stored_correctly(self, tmp_path):
        lake = tmp_path / "lake"
        lake.mkdir()

        tools = ["record_surgery", "record_pathology"]
        errors = [{"mrn": "M1", "error": "fail"}]

        run = RunLog(
            run_id="json0001",
            run_type="fc_workflow",
            name="test",
            batch_name="v1",
            started_at="2025-01-01T00:00:00",
            tools_json=json.dumps(tools),
            errors_json=json.dumps(errors),
            events_by_type_json=json.dumps({"surgery": 5}),
        )
        log_run(run, lake)

        row = get_run(lake, "json0001")
        assert json.loads(row["tools_json"]) == tools
        assert json.loads(row["errors_json"]) == errors
        assert json.loads(row["events_by_type_json"]) == {"surgery": 5}


# ---------- list_runs ----------


class TestListRuns:
    def test_empty_returns_empty_df(self, tmp_path):
        lake = tmp_path / "lake"
        lake.mkdir()
        df = list_runs(lake)
        assert df.height == 0

    def test_returns_sorted_by_started_at(self, tmp_path):
        lake = tmp_path / "lake"
        lake.mkdir()

        for i, ts in enumerate(["2025-01-03", "2025-01-01", "2025-01-02"]):
            run = RunLog(
                run_id=f"sort{i:04d}",
                run_type="fc_workflow",
                name="a",
                batch_name="v1",
                started_at=ts,
            )
            log_run(run, lake)

        df = list_runs(lake)
        assert df.height == 3
        # Descending order
        assert df["started_at"][0] == "2025-01-03"
        assert df["started_at"][2] == "2025-01-01"

    def test_filter_by_type(self, tmp_path):
        lake = tmp_path / "lake"
        lake.mkdir()

        log_run(
            RunLog(
                run_id="fc01",
                run_type="fc_workflow",
                name="a",
                batch_name="v1",
                started_at="2025-01-01",
            ),
            lake,
        )
        log_run(
            RunLog(
                run_id="comp01",
                run_type="compression",
                name="b",
                batch_name="v1",
                started_at="2025-01-02",
            ),
            lake,
        )

        df = list_runs(lake, run_type="compression")
        assert df.height == 1
        assert df["run_type"][0] == "compression"

    def test_limit(self, tmp_path):
        lake = tmp_path / "lake"
        lake.mkdir()

        for i in range(5):
            log_run(
                RunLog(
                    run_id=f"lim{i:05d}",
                    run_type="fc_workflow",
                    name="a",
                    batch_name="v1",
                    started_at=f"2025-01-0{i + 1}",
                ),
                lake,
            )

        df = list_runs(lake, limit=2)
        assert df.height == 2


# ---------- get_run ----------


class TestGetRun:
    def test_existing_run_found(self, tmp_path):
        lake = tmp_path / "lake"
        lake.mkdir()

        log_run(
            RunLog(
                run_id="findme01",
                run_type="fc_workflow",
                name="a",
                batch_name="v1",
                started_at="2025-01-01",
            ),
            lake,
        )

        row = get_run(lake, "findme01")
        assert row is not None
        assert row["run_id"] == "findme01"

    def test_nonexistent_returns_none(self, tmp_path):
        lake = tmp_path / "lake"
        lake.mkdir()

        log_run(
            RunLog(
                run_id="exist001",
                run_type="fc_workflow",
                name="a",
                batch_name="v1",
                started_at="2025-01-01",
            ),
            lake,
        )

        assert get_run(lake, "zzzzzzzz") is None

    def test_prefix_match(self, tmp_path):
        lake = tmp_path / "lake"
        lake.mkdir()

        log_run(
            RunLog(
                run_id="abcd1234",
                run_type="fc_workflow",
                name="a",
                batch_name="v1",
                started_at="2025-01-01",
            ),
            lake,
        )

        row = get_run(lake, "abcd")
        assert row is not None
        assert row["run_id"] == "abcd1234"

    def test_no_parquet_returns_none(self, tmp_path):
        lake = tmp_path / "lake"
        lake.mkdir()
        assert get_run(lake, "anything") is None


# ---------- update_run ----------


class TestUpdateRun:
    def test_updates_status_and_results(self, tmp_path):
        lake = tmp_path / "lake"
        lake.mkdir()

        log_run(
            RunLog(
                run_id="upd00001",
                run_type="fc_workflow",
                name="a",
                batch_name="v1",
                started_at="2025-01-01",
            ),
            lake,
        )

        updated = update_run(
            "upd00001",
            lake,
            status="completed",
            completed_at="2025-01-01T00:05:00",
            duration_seconds=300.0,
            items_processed=10,
            items_succeeded=9,
            items_failed=1,
        )
        assert updated is True

        row = get_run(lake, "upd00001")
        assert row["status"] == "completed"
        assert row["completed_at"] == "2025-01-01T00:05:00"
        assert row["duration_seconds"] == 300.0
        assert row["items_succeeded"] == 9
        assert row["items_failed"] == 1

    def test_update_nonexistent_returns_false(self, tmp_path):
        lake = tmp_path / "lake"
        lake.mkdir()

        log_run(
            RunLog(
                run_id="exists01",
                run_type="fc_workflow",
                name="a",
                batch_name="v1",
                started_at="2025-01-01",
            ),
            lake,
        )

        assert update_run("noexist1", lake, status="completed") is False

    def test_update_no_parquet_returns_false(self, tmp_path):
        lake = tmp_path / "lake"
        lake.mkdir()
        assert update_run("anything", lake, status="failed") is False

    def test_update_only_affects_target_row(self, tmp_path):
        lake = tmp_path / "lake"
        lake.mkdir()

        log_run(
            RunLog(
                run_id="row1keep",
                run_type="fc_workflow",
                name="a",
                batch_name="v1",
                started_at="2025-01-01",
            ),
            lake,
        )
        log_run(
            RunLog(
                run_id="row2updt",
                run_type="fc_single",
                name="b",
                batch_name="v2",
                started_at="2025-01-02",
            ),
            lake,
        )

        update_run("row2updt", lake, status="cancelled", duration_seconds=42.0)

        row1 = get_run(lake, "row1keep")
        row2 = get_run(lake, "row2updt")
        assert row1["status"] == "started"  # unchanged
        assert row2["status"] == "cancelled"
        assert row2["duration_seconds"] == 42.0

    def test_default_status_is_started(self, tmp_path):
        lake = tmp_path / "lake"
        lake.mkdir()

        log_run(
            RunLog(
                run_id="stat0001",
                run_type="fc_workflow",
                name="a",
                batch_name="v1",
                started_at="2025-01-01",
            ),
            lake,
        )

        row = get_run(lake, "stat0001")
        assert row["status"] == "started"
