"""DuckDB database builder from lake parquet files."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

from oncai.config import OncaiConfig, get_dataset_folders

# Folders where each parquet file becomes a separate table
MULTI_TABLE_FOLDERS = {
    "cohorts",
    "fc_extractions",
    "fc_reviews",
}

# Schema mapping: lake folder -> DuckDB schema name (where parquets are loaded).
# fc_extractions lands in extractions_raw; its per-base .sql transforms write to
# extractions_transformed. fc_reviews lands in extractions_silver — reviewed,
# event-grain rows — and a per-batch ``<batch>.sql`` sidecar (mirrored from the
# inbox, run by _run_sibling_sql) reshapes that batch's silver table into dense,
# per-concept tables in extractions_gold.
SCHEMA_MAPPING = {
    "pathology": "raw",
    "cohorts": "cohort",
    "fc_extractions": "extractions_raw",
    "fc_reviews": "extractions_silver",
    "runs": "runs",
}

# Schemas that exist purely as transform destinations (no parquet ever lands
# here). Auto-created by ``build_database`` so per-base .sql files can write
# to them without doing CREATE SCHEMA themselves. ``extractions_gold`` is where a
# review batch's ``<batch>.sql`` sidecar reshapes its reviewed (silver) table into
# dense, per-concept tables.
TRANSFORM_SCHEMAS: set[str] = {"extractions_transformed", "extractions_gold"}

# Scratch schema — a throwaway namespace populated only by ``oncai fc peek``
# for a quick SQL look at a JSONL. Disposable by design (a rebuild clears it);
# auto-created so peek can land tables without CREATE SCHEMA boilerplate.
STAGING_SCHEMAS: set[str] = {"scratch"}


def _q(name: str) -> str:
    """Quote a DuckDB identifier for safe interpolation into SQL.

    DuckDB doesn't allow parameter binding for identifiers (you can't say
    ``CREATE TABLE ?``), so the SQL-spec defence is double-quoting: wrap in
    ``"..."`` and escape any internal ``"`` by doubling it. Applied to every
    schema and table name in this module.
    """
    return '"' + name.replace('"', '""') + '"'


def _count_rows(con: duckdb.DuckDBPyConnection, qname: str) -> int:
    """``SELECT COUNT(*) FROM qname``. ``qname`` must already be _q()-quoted."""
    return con.execute(f"SELECT COUNT(*) FROM {qname}").fetchone()[0]  # type: ignore[index]


def _build_cohort_meta_table(con: duckdb.DuckDBPyConnection, lake_path: Path) -> int:
    """(Re)create ``cohort.meta`` from cohort sidecar JSON files.

    Lets users discover what cohorts exist and when they were made via plain
    SQL, without joining the inline ``cohort_created_at``/``cohort_name`` cols
    on every cohort table.
    """
    cohorts_dir = lake_path / "cohorts"
    sidecars = sorted(cohorts_dir.glob("*.cohort.json")) if cohorts_dir.exists() else []

    con.execute("CREATE SCHEMA IF NOT EXISTS cohort")
    con.execute("DROP TABLE IF EXISTS cohort.meta")
    con.execute("""
        CREATE TABLE cohort.meta (
            name VARCHAR,
            description VARCHAR,
            key_column VARCHAR,
            created_at VARCHAR,
            row_count BIGINT,
            columns VARCHAR[],
            source_file VARCHAR
        )
    """)

    for sidecar in sidecars:
        try:
            data = json.loads(sidecar.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        con.execute(
            "INSERT INTO cohort.meta VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                data.get("name"),
                data.get("description") or "",
                data.get("key_column"),
                data.get("created_at"),
                data.get("row_count"),
                list(data.get("columns") or []),
                data.get("source_file") or "",
            ],
        )

    return con.execute("SELECT COUNT(*) FROM cohort.meta").fetchone()[0]  # type: ignore[index]


def update_database_folder(config: OncaiConfig, folder: str) -> dict[str, int]:
    """
    Refresh a single lake folder's tables in the existing DuckDB database.

    Use after adding cohorts, promoting FC extractions, etc.

    Args:
        config: OncAI configuration
        folder: Lake folder name (e.g. "cohorts", "fc_extractions")

    Returns:
        Dict of schema.table_name -> row_count for affected tables

    Raises:
        FileNotFoundError: If DB or lake folder doesn't exist
    """
    db_path = config.db_path
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}. Run 'build-db' first.")

    lake_folder = config.lake_path / folder
    if not lake_folder.exists():
        raise FileNotFoundError(f"Lake folder not found: {lake_folder}")

    parquets = list(lake_folder.glob("*.parquet"))
    if not parquets:
        return {}

    schema = SCHEMA_MAPPING.get(folder)
    if schema is None:
        raise ValueError(f"No schema mapping configured for folder: {folder}")
    con = duckdb.connect(str(db_path))
    results = {}
    transform_errors: list[str] = []

    try:
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        # Per-base .sql files write to *_transformed schemas; staging is for
        # ``oncai fc peek``. Both auto-created so callers don't need their
        # own CREATE SCHEMA boilerplate.
        for s in TRANSFORM_SCHEMAS | STAGING_SCHEMAS:
            con.execute(f"CREATE SCHEMA IF NOT EXISTS {s}")

        if folder in MULTI_TABLE_FOLDERS:
            for parquet_path in parquets:
                table_name = parquet_path.stem
                full_name = f"{schema}.{table_name}"
                qname = f"{_q(schema)}.{_q(table_name)}"
                try:
                    con.execute(
                        f"CREATE OR REPLACE TABLE {qname} AS "
                        "SELECT * FROM read_parquet(?)",
                        [str(parquet_path)],
                    )
                    results[full_name] = _count_rows(con, qname)
                    _run_sibling_sql(con, parquet_path, transform_errors)
                except Exception as e:
                    print(f"Error creating table {full_name}: {e}")

            if folder == "cohorts":
                try:
                    meta_rows = _build_cohort_meta_table(con, config.lake_path)
                    results["cohort.meta"] = meta_rows
                except Exception as e:
                    print(f"Error creating cohort.meta: {e}")
        else:
            table_name = folder
            full_name = f"{schema}.{table_name}"
            qname = f"{_q(schema)}.{_q(table_name)}"
            parquet_path_str = (
                str(parquets[0])
                if len(parquets) == 1
                else str(lake_folder / "*.parquet")
            )
            try:
                con.execute(
                    f"CREATE OR REPLACE TABLE {qname} AS "
                    "SELECT * FROM read_parquet(?)",
                    [parquet_path_str],
                )
                results[full_name] = _count_rows(con, qname)
                _run_sibling_sql(con, lake_folder / f"{folder}.parquet", transform_errors)
            except Exception as e:
                print(f"Error creating table {full_name}: {e}")
    finally:
        con.close()

    if transform_errors:
        print(f"\n⚠ {len(transform_errors)} transform(s) failed:")
        for m in transform_errors:
            print(f"  - {m}")

    return results


def build_database(
    config: OncaiConfig, force: bool = False, strict: bool = False
) -> dict[str, int]:
    """
    Build DuckDB database from lake parquet files.

    Organizes tables into schemas:
    - raw: pathology
    - cohort: one table per cohort + meta
    - extractions_raw: one table per fc batch
    - extractions_silver: one table per completed review batch (reviewed,
      event-grain; sparse — a definition's event tools share a table)
    - extractions_gold: dense per-concept tables built by a review batch's
      ``<batch>.sql`` sidecar reshaping its silver table
    - runs: run log
    - extractions_transformed / scratch: created empty for
      per-base ``.sql`` transforms and ``oncai fc peek`` to write into.

    Args:
        config: OncAI configuration
        force: If True, recreate database even if it exists

    Returns:
        Dict of schema.table_name -> row_count
    """
    db_path = config.db_path

    if db_path.exists() and force:
        db_path.unlink()

    db_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(db_path))
    results = {}
    transform_errors: list[str] = []

    # Create all schemas upfront — including transform/scratch schemas so per-base
    # .sql files and ``oncai fc peek`` can write into them without their own
    # CREATE SCHEMA boilerplate.
    schemas_needed = set(SCHEMA_MAPPING.values()) | TRANSFORM_SCHEMAS | STAGING_SCHEMAS
    for schema in schemas_needed:
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")

    # Scan all known folders: dataset folders + any SCHEMA_MAPPING folders that exist
    all_folders = list(
        dict.fromkeys(list(get_dataset_folders()) + list(SCHEMA_MAPPING.keys()))
    )
    for folder in all_folders:
        lake_folder = config.lake_path / folder
        if not lake_folder.exists():
            continue

        parquets = list(lake_folder.glob("*.parquet"))
        if not parquets:
            continue

        schema = SCHEMA_MAPPING.get(folder)
        if schema is None:
            continue

        # Special handling for folders with multiple different-schema parquets
        if folder in MULTI_TABLE_FOLDERS:
            for parquet_path in parquets:
                # Table name = filename without .parquet
                table_name = parquet_path.stem
                full_name = f"{schema}.{table_name}"
                qname = f"{_q(schema)}.{_q(table_name)}"
                try:
                    con.execute(
                        f"CREATE OR REPLACE TABLE {qname} AS "
                        "SELECT * FROM read_parquet(?)",
                        [str(parquet_path)],
                    )
                    results[full_name] = _count_rows(con, qname)
                    _run_sibling_sql(con, parquet_path, transform_errors)
                except Exception as e:
                    print(f"Error creating table {full_name}: {e}")

            if folder == "cohorts":
                try:
                    meta_rows = _build_cohort_meta_table(con, config.lake_path)
                    results["cohort.meta"] = meta_rows
                except Exception as e:
                    print(f"Error creating cohort.meta: {e}")
        else:
            # Standard handling: one table per folder
            table_name = folder
            full_name = f"{schema}.{table_name}"
            qname = f"{_q(schema)}.{_q(table_name)}"

            if len(parquets) == 1:
                parquet_path_str = str(parquets[0])
            else:
                # Use glob pattern for multiple files with same schema
                parquet_path_str = str(lake_folder / "*.parquet")

            try:
                con.execute(
                    f"CREATE OR REPLACE TABLE {qname} AS "
                    "SELECT * FROM read_parquet(?)",
                    [parquet_path_str],
                )
                results[full_name] = _count_rows(con, qname)
                # Per-folder transform: <folder>/<folder>.sql can introduce
                # derived tables for the single-table folder.
                _run_sibling_sql(con, lake_folder / f"{folder}.parquet", transform_errors)
            except Exception as e:
                print(f"Error creating table {full_name}: {e}")

    con.close()

    if transform_errors:
        print(f"\n⚠ {len(transform_errors)} transform(s) failed:")
        for m in transform_errors:
            print(f"  - {m}")
        if strict:
            raise RuntimeError(
                f"{len(transform_errors)} SQL transform(s) failed (--strict)"
            )

    return results


def _run_sibling_sql(
    con: duckdb.DuckDBPyConnection,
    parquet_path: Path,
    errors: list[str] | None = None,
) -> None:
    """If ``<parquet_stem>.sql`` exists next to ``parquet_path``, execute it.

    Per-parquet SQL transforms live alongside their base parquet, so each
    curated run is self-contained: drop the segments + ``.sql`` together and
    ``oncai build-db`` materialises both the raw table and any derived
    tables/views the SQL declares. A failing transform doesn't abort the build
    (one bad transform shouldn't tank the whole rebuild) — but it's collected
    into ``errors`` so the caller can surface a summary instead of letting it
    scroll past mid-build.
    """
    sql_path = parquet_path.with_suffix(".sql")
    if not sql_path.exists():
        return
    try:
        con.execute(sql_path.read_text())
        print(f"  Ran transform: {sql_path.name}")
    except Exception as e:
        msg = f"{sql_path.name}: {type(e).__name__}: {e}"
        if errors is not None:
            errors.append(msg)
        print(f"  [transform error] {msg}")


def get_database_info(config: OncaiConfig) -> dict[str, dict]:
    """
    Get information about the DuckDB database.

    Returns dict of schema.table_name -> {rows, columns, schema, size_mb}
    """
    db_path = config.db_path

    if not db_path.exists():
        return {}

    con = duckdb.connect(str(db_path), read_only=True)
    results = {}

    try:
        # Get list of tables from all schemas (excluding system schemas)
        tables = con.execute("""
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
            ORDER BY table_schema, table_name
        """).fetchall()

        for schema, table_name in tables:
            full_name = f"{schema}.{table_name}"
            qname = f"{_q(schema)}.{_q(table_name)}"

            row_count = _count_rows(con, qname)

            col_count = con.execute(
                """
                SELECT COUNT(*)
                FROM information_schema.columns
                WHERE table_schema = ? AND table_name = ?
                """,
                [schema, table_name],
            ).fetchone()[0]  # type: ignore[index]

            results[full_name] = {
                "rows": row_count,
                "columns": col_count,
                "schema": schema,
            }

    finally:
        con.close()

    # Add file size
    if db_path.exists():
        size_mb = db_path.stat().st_size / (1024 * 1024)
        for table_info in results.values():
            table_info["db_size_mb"] = round(size_mb, 2)

    return results
