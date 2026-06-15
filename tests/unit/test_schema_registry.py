"""Tests for oncai.schemas registry."""

from __future__ import annotations

import pytest

from oncai.schemas import SchemaSpec, get_schema, list_schemas, register_schema


class TestSchemaRegistry:
    def test_list_schemas_includes_pathology(self):
        names = list_schemas()
        assert "pathology" in names

    def test_get_schema_returns_spec(self):
        schema = get_schema("pathology")
        assert isinstance(schema, SchemaSpec)
        assert schema.name == "pathology"

    def test_get_schema_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown schema"):
            get_schema("nonexistent_schema_xyz")

    def test_schema_spec_fields(self):
        schema = get_schema("pathology")
        assert isinstance(schema.row_key_cols, tuple)
        assert isinstance(schema.content_cols, tuple)
        assert schema.transform in ("collate", "passthrough", "collate_dedup")
        assert len(schema.columns) > 0

    def test_register_custom_schema(self):
        import polars as pl

        spec = SchemaSpec(
            name="test_custom",
            columns={"id": pl.Int64, "value": pl.String},
            row_key_cols=("id",),
            content_cols=("id", "value"),
            transform="passthrough",
        )
        register_schema(spec)
        result = get_schema("test_custom")
        assert result.name == "test_custom"
