"""Content hashing utilities for row-level deduplication."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import polars as pl


def blake2b_128(data: str | bytes) -> bytes:
    """Compute Blake2b-128 hash of data."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.blake2b(data, digest_size=16).digest()


def compute_key_hash(values: tuple[str, ...]) -> bytes:
    """Compute hash from composite key values."""
    key_str = "|".join(str(v) for v in values)
    return blake2b_128(key_str)


def compute_content_hash(values: tuple[str, ...]) -> bytes:
    """Compute hash from content values for change detection."""
    content_str = "|".join(str(v) if v is not None else "" for v in values)
    return blake2b_128(content_str)


def now_utc() -> datetime:
    """Current UTC time (naive, for Arrow/Parquet compatibility)."""
    return datetime.now(tz=UTC).replace(tzinfo=None)


def add_hashes_to_dataframe(
    df: "pl.DataFrame",
    key_cols: list[str],
    content_cols: list[str] | None = None,
) -> "pl.DataFrame":
    """
    Add key_hash and content_hash columns to a Polars DataFrame.

    Args:
        df: Input DataFrame
        key_cols: Columns that form the composite business key
        content_cols: Columns to hash for change detection (defaults to key_cols)

    Returns:
        DataFrame with key_hash and content_hash columns added
    """

    if df.is_empty():
        return df.with_columns(
            [
                pl.lit(None).cast(pl.Binary).alias("key_hash"),
                pl.lit(None).cast(pl.Binary).alias("content_hash"),
            ]
        )

    content_cols = content_cols or key_cols

    # Filter to columns that actually exist
    existing_key_cols = [c for c in key_cols if c in df.columns]
    existing_content_cols = [c for c in content_cols if c in df.columns]

    # Compute hashes row by row
    key_hashes = []
    content_hashes = []

    for row in df.iter_rows(named=True):
        key_str = "|".join(str(row.get(c, "") or "") for c in existing_key_cols)
        key_hashes.append(blake2b_128(key_str))

        content_str = "|".join(str(row.get(c, "") or "") for c in existing_content_cols)
        content_hashes.append(blake2b_128(content_str))

    return df.with_columns(
        [
            pl.Series("key_hash", key_hashes, dtype=pl.Binary),
            pl.Series("content_hash", content_hashes, dtype=pl.Binary),
        ]
    )
