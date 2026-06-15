"""Schema definitions and registry for OncAI datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import polars as pl


@dataclass
class SchemaSpec:
    """Specification for a dataset schema."""

    name: str
    columns: dict[str, type[pl.DataType] | pl.DataType]
    row_key_cols: tuple[str, ...]
    content_cols: tuple[str, ...]
    transform: Literal["collate", "passthrough", "collate_dedup"]
    sort_cols: tuple[str, ...] = ()
    date_cols: dict[str, str] = field(default_factory=dict)  # col -> strptime format
    rename_cols: dict[str, str] = field(default_factory=dict)  # csv_col -> schema_col

    def get_polars_schema(self) -> dict[str, type[pl.DataType] | pl.DataType]:
        """Return schema dict for Polars."""
        return self.columns.copy()


# Global schema registry
_REGISTRY: dict[str, SchemaSpec] = {}


def register_schema(spec: SchemaSpec) -> SchemaSpec:
    """Register a schema specification."""
    _REGISTRY[spec.name] = spec
    return spec


def get_schema(name: str) -> SchemaSpec:
    """Get a registered schema by name."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown schema: {name}. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]


def list_schemas() -> list[str]:
    """List all registered schema names."""
    return list(_REGISTRY.keys())


def get_all_schemas() -> dict[str, SchemaSpec]:
    """Get all registered schemas."""
    return _REGISTRY.copy()


# Import schemas to register them (must be after SchemaSpec/register_schema are defined)
from oncai.schemas import pathology  # noqa: E402,F401

__all__ = [
    "SchemaSpec",
    "register_schema",
    "get_schema",
    "list_schemas",
    "get_all_schemas",
]
