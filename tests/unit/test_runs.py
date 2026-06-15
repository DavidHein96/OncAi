"""Tests for the run logging system (inbox-resident JSON manifests)."""

from __future__ import annotations

import json

from oncai.runs import (
    RUN_FILE_SUFFIX,
    RunLog,
    _generate_run_id,
    complete_run,
    get_run,
    list_runs,
    runs_to_dataframe,
    start_run,
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


# ---------- start_run ----------


class TestStartRun:
    def test_writes_started_manifest(self, tmp_path):
        inbox = tmp_path / "inbox"
        run = RunLog(
            run_id="abcd1234",
            run_type="fc_single",
            name="test",
            batch_name="v1",
            started_at="2025-01-01T00:00:00+00:00",
        )
        path = start_run(run, inbox)

        assert path.exists()
        assert path.name == f"abcd1234{RUN_FILE_SUFFIX}"
        data = json.loads(path.read_text())
        assert data["run_id"] == "abcd1234"
        assert data["status"] == "started"

    def test_one_file_per_run(self, tmp_path):
        inbox = tmp_path / "inbox"
        start_run(RunLog(run_id="aaaa1111", started_at="2025-01-01"), inbox)
        start_run(RunLog(run_id="bbbb2222", started_at="2025-01-02"), inbox)

        names = {f.name for f in (inbox / "runs").glob(f"*{RUN_FILE_SUFFIX}")}
        assert names == {f"aaaa1111{RUN_FILE_SUFFIX}", f"bbbb2222{RUN_FILE_SUFFIX}"}


# ---------- complete_run (the started → completed lifecycle) ----------


class TestCompleteRun:
    def test_rewrites_same_file_with_results(self, tmp_path):
        inbox = tmp_path / "inbox"
        start_run(
            RunLog(run_id="life0001", run_type="fc_single", started_at="2025-01-01"),
            inbox,
        )

        before = get_run(inbox, "life0001")
        assert before["status"] == "started"
        assert before["completed_at"] is None

        ok = complete_run(
            "life0001",
            inbox,
            status="completed",
            completed_at="2025-01-01T00:05:00",
            duration_seconds=300.0,
            items_succeeded=9,
            items_failed=1,
        )
        assert ok is True

        after = get_run(inbox, "life0001")
        assert after["status"] == "completed"
        assert after["completed_at"] == "2025-01-01T00:05:00"
        assert after["duration_seconds"] == 300.0
        assert after["items_succeeded"] == 9
        # The lifecycle stays in ONE file — no second record is created.
        assert len(list((inbox / "runs").glob(f"*{RUN_FILE_SUFFIX}"))) == 1

    def test_missing_manifest_returns_false(self, tmp_path):
        assert complete_run("nope0001", tmp_path / "inbox", status="failed") is False

    def test_ignores_unknown_update_fields(self, tmp_path):
        inbox = tmp_path / "inbox"
        start_run(RunLog(run_id="known001", started_at="2025-01-01"), inbox)

        complete_run("known001", inbox, status="completed", bogus_field="x")

        data = json.loads((inbox / "runs" / f"known001{RUN_FILE_SUFFIX}").read_text())
        assert data["status"] == "completed"
        assert "bogus_field" not in data

    def test_only_target_manifest_changes(self, tmp_path):
        inbox = tmp_path / "inbox"
        start_run(RunLog(run_id="keep0001", started_at="2025-01-01"), inbox)
        start_run(RunLog(run_id="updt0001", started_at="2025-01-02"), inbox)

        complete_run("updt0001", inbox, status="cancelled", duration_seconds=42.0)

        assert get_run(inbox, "keep0001")["status"] == "started"  # unchanged
        assert get_run(inbox, "updt0001")["status"] == "cancelled"
        assert get_run(inbox, "updt0001")["duration_seconds"] == 42.0


# ---------- list_runs ----------


class TestListRuns:
    def test_empty_returns_empty_df(self, tmp_path):
        df = list_runs(tmp_path / "inbox")
        assert df.height == 0

    def test_returns_sorted_by_started_at(self, tmp_path):
        inbox = tmp_path / "inbox"
        for i, ts in enumerate(["2025-01-03", "2025-01-01", "2025-01-02"]):
            start_run(
                RunLog(
                    run_id=f"sort{i:04d}",
                    run_type="fc_single",
                    name="a",
                    batch_name="v1",
                    started_at=ts,
                ),
                inbox,
            )

        df = list_runs(inbox)
        assert df.height == 3
        assert df["started_at"][0] == "2025-01-03"
        assert df["started_at"][2] == "2025-01-01"

    def test_filter_by_type(self, tmp_path):
        inbox = tmp_path / "inbox"
        start_run(
            RunLog(run_id="fc000001", run_type="fc_workflow", started_at="2025-01-01"),
            inbox,
        )
        start_run(
            RunLog(run_id="cmp00001", run_type="compression", started_at="2025-01-02"),
            inbox,
        )

        df = list_runs(inbox, run_type="compression")
        assert df.height == 1
        assert df["run_type"][0] == "compression"

    def test_limit(self, tmp_path):
        inbox = tmp_path / "inbox"
        for i in range(5):
            start_run(
                RunLog(run_id=f"lim{i:05d}", started_at=f"2025-01-0{i + 1}"), inbox
            )

        assert list_runs(inbox, limit=2).height == 2


# ---------- get_run ----------


class TestGetRun:
    def test_existing_run_found(self, tmp_path):
        inbox = tmp_path / "inbox"
        start_run(RunLog(run_id="findme01", started_at="2025-01-01"), inbox)

        row = get_run(inbox, "findme01")
        assert row is not None
        assert row["run_id"] == "findme01"

    def test_nonexistent_returns_none(self, tmp_path):
        inbox = tmp_path / "inbox"
        start_run(RunLog(run_id="exist001", started_at="2025-01-01"), inbox)
        assert get_run(inbox, "zzzzzzzz") is None

    def test_prefix_match(self, tmp_path):
        inbox = tmp_path / "inbox"
        start_run(RunLog(run_id="abcd1234", started_at="2025-01-01"), inbox)

        row = get_run(inbox, "abcd")
        assert row is not None
        assert row["run_id"] == "abcd1234"

    def test_no_runs_dir_returns_none(self, tmp_path):
        assert get_run(tmp_path / "inbox", "anything") is None


# ---------- runs_to_dataframe (the manifest → parquet projection) ----------


class TestRunsToDataframe:
    def test_empty_keeps_schema(self):
        df = runs_to_dataframe([])
        assert df.height == 0
        assert {"run_id", "status", "items_succeeded"}.issubset(df.columns)

    def test_all_fields_roundtrip(self):
        run = RunLog(
            run_id="full0001",
            run_type="compression",
            name="oncology_standard",
            batch_name="v3",
            started_at="2025-06-01T10:00:00+00:00",
            completed_at="2025-06-01T10:05:00+00:00",
            duration_seconds=300.5,
            git_dirty=True,
            temperature=0.0,
            workers=4,
            system_prompt="You are a medical AI.",
            tools_json=json.dumps(["record_surgery", "record_pathology"]),
            items_succeeded=95,
            total_input_tokens=50000,
        )
        df = runs_to_dataframe([run.to_dict()])
        row = df.row(0, named=True)

        assert row["run_id"] == "full0001"
        assert row["duration_seconds"] == 300.5
        assert row["items_succeeded"] == 95
        assert row["total_input_tokens"] == 50000
        assert row["git_dirty"] is True
        assert json.loads(row["tools_json"]) == ["record_surgery", "record_pathology"]

    def test_ignores_unknown_keys(self):
        df = runs_to_dataframe(
            [{"run_id": "x", "started_at": "2025-01-01", "unknown_key": "y"}]
        )
        assert df.height == 1
        assert "unknown_key" not in df.columns
