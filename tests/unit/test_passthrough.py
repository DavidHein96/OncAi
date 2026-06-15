"""Tests for oncai.transforms.passthrough module."""

from __future__ import annotations

import polars as pl
import pytest

from oncai.schemas import SchemaSpec
from oncai.transforms.passthrough import passthrough_transform


@pytest.fixture
def simple_schema():
    return SchemaSpec(
        name="test",
        columns={
            "id": pl.Int64,
            "value": pl.String,
            "key_hash": pl.Binary,
            "content_hash": pl.Binary,
        },
        row_key_cols=("id",),
        content_cols=("id", "value"),
        transform="passthrough",
    )


class TestPassthroughTransform:
    def test_adds_hash_columns(self, simple_schema):
        df = pl.DataFrame({"id": [1, 2], "value": ["a", "b"]}).lazy()
        result = passthrough_transform(df, simple_schema).collect()
        assert "key_hash" in result.columns
        assert "content_hash" in result.columns

    def test_preserves_data(self, simple_schema):
        df = pl.DataFrame({"id": [1, 2, 3], "value": ["x", "y", "z"]}).lazy()
        result = passthrough_transform(df, simple_schema).collect()
        assert len(result) == 3
        assert result["id"].to_list() == [1, 2, 3]
        assert result["value"].to_list() == ["x", "y", "z"]

    def test_hash_types(self, simple_schema):
        df = pl.DataFrame({"id": [1], "value": ["a"]}).lazy()
        result = passthrough_transform(df, simple_schema).collect()
        assert result["key_hash"].dtype == pl.Binary
        assert result["content_hash"].dtype == pl.Binary

    def test_different_content_different_hash(self, simple_schema):
        df = pl.DataFrame({"id": [1, 1], "value": ["a", "b"]}).lazy()
        result = passthrough_transform(df, simple_schema).collect()
        hashes = result["content_hash"].to_list()
        assert hashes[0] != hashes[1]

    def test_same_key_same_hash(self, simple_schema):
        df = pl.DataFrame({"id": [1, 1], "value": ["a", "b"]}).lazy()
        result = passthrough_transform(df, simple_schema).collect()
        key_hashes = result["key_hash"].to_list()
        assert key_hashes[0] == key_hashes[1]

    def test_sort_cols(self):
        schema = SchemaSpec(
            name="test_sorted",
            columns={"id": pl.Int64, "value": pl.String},
            row_key_cols=("id",),
            content_cols=("id", "value"),
            transform="passthrough",
            sort_cols=("id",),
        )
        df = pl.DataFrame({"id": [3, 1, 2], "value": ["c", "a", "b"]}).lazy()
        result = passthrough_transform(df, schema).collect()
        assert result["id"].to_list() == [1, 2, 3]
