"""Pathology reports schema definition."""

import polars as pl

from oncai.schemas import SchemaSpec, register_schema

# Raw pathology CSV columns (before collation)
RAW_COLUMNS = {
    "mrn": pl.String,
    "ordering_date": pl.Date,
    "report_id": pl.String,
    "external_name": pl.String,
    "group_line": pl.Int32,  # Optional
    "value_line": pl.Int32,  # Optional
    "mult_ln_val_storage": pl.String,
    "row_id": pl.Int32,
}

# Collated pathology schema (after transform)
PATHOLOGY_SCHEMA = register_schema(
    SchemaSpec(
        name="pathology",
        columns={
            "mrn": pl.String,
            "ordering_date": pl.Date,
            "report_id": pl.String,
            "external_name": pl.String,
            "report_text": pl.String,
            "key_hash": pl.Binary,
            "content_hash": pl.Binary,
        },
        row_key_cols=("report_id",),
        content_cols=("report_text",),
        transform="collate",
        sort_cols=("report_id",),
        date_cols={"ordering_date": "%Y-%m-%d"},
    )
)
