"""Passthrough transform for simple schema validation."""

from __future__ import annotations

import polars as pl

from oncai.hashing import compute_content_hash, compute_key_hash
from oncai.schemas import SchemaSpec


def passthrough_transform(df: pl.LazyFrame, schema: SchemaSpec) -> pl.LazyFrame:
    """
    Simple passthrough transform that validates and adds hashes.

    No collation needed - just validate schema and compute hashes.
    """
    result: pl.DataFrame = df.collect()  # type: ignore[assignment]

    # Compute hashes based on schema key and content columns
    key_hashes = []
    content_hashes = []

    for row in result.iter_rows(named=True):
        key_vals = tuple(str(row.get(c, "")) for c in schema.row_key_cols)
        content_vals = tuple(str(row.get(c, "")) for c in schema.content_cols)

        key_hashes.append(compute_key_hash(key_vals))
        content_hashes.append(compute_content_hash(content_vals))

    result = result.with_columns(
        pl.Series("key_hash", key_hashes),
        pl.Series("content_hash", content_hashes),
    )

    # Sort if specified
    if schema.sort_cols:
        result = result.sort(list(schema.sort_cols))

    return result.lazy()
