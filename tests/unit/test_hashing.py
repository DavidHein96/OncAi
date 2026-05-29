"""Tests for oncai.hashing module."""

from __future__ import annotations

import polars as pl

from oncai.hashing import (
    add_hashes_to_dataframe,
    blake2b_128,
    compute_content_hash,
    compute_key_hash,
    now_utc,
)


class TestBlake2b128:
    def test_returns_16_bytes(self):
        result = blake2b_128("hello")
        assert len(result) == 16

    def test_deterministic(self):
        assert blake2b_128("test") == blake2b_128("test")

    def test_different_input_different_hash(self):
        assert blake2b_128("a") != blake2b_128("b")

    def test_bytes_input(self):
        result = blake2b_128(b"hello")
        assert len(result) == 16

    def test_string_and_bytes_same(self):
        assert blake2b_128("hello") == blake2b_128(b"hello")

    def test_empty_string(self):
        result = blake2b_128("")
        assert len(result) == 16


class TestComputeKeyHash:
    def test_single_value(self):
        result = compute_key_hash(("abc",))
        assert len(result) == 16

    def test_multiple_values(self):
        result = compute_key_hash(("a", "b", "c"))
        assert len(result) == 16

    def test_deterministic(self):
        assert compute_key_hash(("x", "y")) == compute_key_hash(("x", "y"))

    def test_order_sensitive(self):
        assert compute_key_hash(("a", "b")) != compute_key_hash(("b", "a"))

    def test_empty_tuple(self):
        result = compute_key_hash(())
        assert len(result) == 16


class TestComputeContentHash:
    def test_basic(self):
        result = compute_content_hash(("hello", "world"))
        assert len(result) == 16

    def test_none_handling(self):
        """None values should be converted to empty string."""
        result = compute_content_hash((None, "value"))
        assert len(result) == 16
        # None becomes "" in hash
        assert result == compute_content_hash(("", "value"))

    def test_order_sensitive(self):
        assert compute_content_hash(("a", "b")) != compute_content_hash(("b", "a"))


class TestNowUtc:
    def test_returns_naive_datetime(self):
        result = now_utc()
        assert result.tzinfo is None

    def test_returns_recent(self):
        from datetime import UTC, datetime

        result = now_utc()
        now = datetime.now(tz=UTC).replace(tzinfo=None)
        # Should be within 1 second
        delta = abs((now - result).total_seconds())
        assert delta < 1


class TestAddHashesToDataframe:
    def test_basic(self):
        df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        result = add_hashes_to_dataframe(df, key_cols=["a"], content_cols=["a", "b"])
        assert "key_hash" in result.columns
        assert "content_hash" in result.columns
        assert len(result) == 2
        assert result["key_hash"].dtype == pl.Binary

    def test_empty_dataframe(self):
        df = pl.DataFrame({"a": [], "b": []}).cast({"a": pl.Int64, "b": pl.Utf8})
        result = add_hashes_to_dataframe(df, key_cols=["a"])
        assert "key_hash" in result.columns
        assert "content_hash" in result.columns
        assert len(result) == 0

    def test_missing_columns_handled(self):
        """Columns not in DataFrame should be silently skipped."""
        df = pl.DataFrame({"a": [1]})
        result = add_hashes_to_dataframe(df, key_cols=["a", "nonexistent"])
        assert "key_hash" in result.columns
        assert len(result) == 1

    def test_content_cols_default_to_key_cols(self):
        df = pl.DataFrame({"a": [1]})
        result = add_hashes_to_dataframe(df, key_cols=["a"])
        assert "content_hash" in result.columns
