"""Tests for append-only tombstone resolution and projection."""

from __future__ import annotations

import polars as pl

from oncai.ingest import run_ingest
from oncai.tombstones import (
    TombstoneAction,
    prune_lake_target,
    resolve_tombstones,
    write_tombstone_event,
)


def test_tombstone_latest_event_wins(oncai_config) -> None:
    write_tombstone_event(
        oncai_config,
        kind="fc_extractions",
        target="old_batch",
        action=TombstoneAction.FORGET,
        reason="bad prompt",
        actor="a",
        at="2026-06-12T12:00:00Z",
        event_id="aaaaaaaaaaaaaaaa",
    )
    write_tombstone_event(
        oncai_config,
        kind="fc_extractions",
        target="old_batch",
        action=TombstoneAction.REVIVE,
        reason="keep it",
        actor="b",
        at="2026-06-12T13:00:00Z",
        event_id="bbbbbbbbbbbbbbbb",
    )

    state = resolve_tombstones(oncai_config)

    assert state.errors == ()
    assert state.active_targets("fc_extractions") == set()


def test_tombstones_ingest_projects_active_state(oncai_config) -> None:
    write_tombstone_event(
        oncai_config,
        kind="fc_reviews",
        target="review_a",
        action=TombstoneAction.FORGET,
        actor="a",
        at="2026-06-12T12:00:00Z",
        event_id="aaaaaaaaaaaaaaaa",
    )
    write_tombstone_event(
        oncai_config,
        kind="cohorts",
        target="cohort_a",
        action=TombstoneAction.FORGET,
        actor="a",
        at="2026-06-12T12:00:00Z",
        event_id="bbbbbbbbbbbbbbbb",
    )
    write_tombstone_event(
        oncai_config,
        kind="cohorts",
        target="cohort_a",
        action=TombstoneAction.REVIVE,
        actor="b",
        at="2026-06-12T13:00:00Z",
        event_id="cccccccccccccccc",
    )

    results = run_ingest(oncai_config, folder="tombstones")

    assert len(results) == 1
    out = oncai_config.lake_path / "tombstones" / "tombstones.parquet"
    df = pl.read_parquet(out)
    active = {
        (row["kind"], row["target"], row["action"]): row["active"]
        for row in df.iter_rows(named=True)
    }
    assert active[("fc_reviews", "review_a", "forget")] is True
    assert active[("cohorts", "cohort_a", "forget")] is False
    assert active[("cohorts", "cohort_a", "revive")] is False


def test_prune_lake_target_removes_known_projection_files(oncai_config) -> None:
    lake = oncai_config.lake_path / "cohorts"
    parquet = lake / "demo.parquet"
    sidecar = lake / "demo.cohort.json"
    sql = lake / "demo.sql"
    parquet.write_bytes(b"parquet")
    sidecar.write_text("{}")
    sql.write_text("SELECT 1;")

    pruned = prune_lake_target(oncai_config, "cohorts", "demo")

    assert {path.name for path in pruned} == {
        "demo.parquet",
        "demo.cohort.json",
        "demo.sql",
    }
    assert not parquet.exists()
    assert not sidecar.exists()
    assert not sql.exists()
