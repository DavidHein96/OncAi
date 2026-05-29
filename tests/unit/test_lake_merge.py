"""Tests for oncai.lake.merge_into_lake function."""

from __future__ import annotations

import polars as pl
import pytest

from oncai.lake import merge_into_lake


@pytest.fixture
def make_df():
    """Helper to create a DataFrame with key_hash and content_hash."""
    from oncai.hashing import blake2b_128

    def _make(rows: list[dict]) -> pl.DataFrame:
        key_hashes = []
        content_hashes = []
        for row in rows:
            key_hashes.append(blake2b_128(row["id"]))
            content_hashes.append(blake2b_128(f"{row['id']}|{row['value']}"))
        df = pl.DataFrame(rows)
        df = df.with_columns(
            [
                pl.Series("key_hash", key_hashes, dtype=pl.Binary),
                pl.Series("content_hash", content_hashes, dtype=pl.Binary),
            ]
        )
        return df

    return _make


class TestMergeIntoLake:
    def test_new_file(self, tmp_path, make_df):
        """When no existing parquet, all rows are new."""
        parquet_path = tmp_path / "test.parquet"
        df = make_df([{"id": "1", "value": "a"}, {"id": "2", "value": "b"}])

        merged, stats = merge_into_lake(df, parquet_path)
        assert stats["new_rows"] == 2
        assert stats["updated_rows"] == 0
        assert stats["unchanged_rows"] == 0
        assert len(merged) == 2

    def test_all_duplicates(self, tmp_path, make_df):
        """When all rows exist with same content, all unchanged."""
        parquet_path = tmp_path / "test.parquet"
        df = make_df([{"id": "1", "value": "a"}])

        # Write initial
        df.write_parquet(parquet_path)

        # Merge same data
        merged, stats = merge_into_lake(df, parquet_path)
        assert stats["new_rows"] == 0
        assert stats["updated_rows"] == 0
        assert stats["unchanged_rows"] == 1
        assert len(merged) == 1

    def test_content_update(self, tmp_path, make_df):
        """When key exists but content changed, row is updated."""
        parquet_path = tmp_path / "test.parquet"
        df_old = make_df([{"id": "1", "value": "old"}])
        df_old.write_parquet(parquet_path)

        df_new = make_df([{"id": "1", "value": "new"}])
        merged, stats = merge_into_lake(df_new, parquet_path)
        assert stats["updated_rows"] == 1
        assert stats["new_rows"] == 0
        assert len(merged) == 1
        # The updated value should be present
        assert merged["value"].to_list() == ["new"]

    def test_mix_new_and_existing(self, tmp_path, make_df):
        """Mix of new, updated, and unchanged."""
        parquet_path = tmp_path / "test.parquet"
        df_old = make_df(
            [
                {"id": "1", "value": "a"},
                {"id": "2", "value": "b"},
            ]
        )
        df_old.write_parquet(parquet_path)

        df_new = make_df(
            [
                {"id": "2", "value": "b"},  # unchanged
                {"id": "3", "value": "c"},  # new
            ]
        )
        merged, stats = merge_into_lake(df_new, parquet_path)
        assert stats["new_rows"] == 1
        assert stats["unchanged_rows"] == 1
        assert stats["updated_rows"] == 0
        assert len(merged) == 3  # old row 1 + unchanged row 2 + new row 3
