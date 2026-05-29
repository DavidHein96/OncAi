"""Function-calling single-note extraction CLI commands."""

from __future__ import annotations

import importlib
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

import typer
from rich.table import Table

from ._shared import console, get_config, load_cohort_filter

if TYPE_CHECKING:
    import duckdb

# DuckDB table identifier safety. Batch names become table names in
# extractions_raw, so they need to be safe ASCII identifiers.
_SAFE_BATCH_NAME = re.compile(r"^[A-Za-z0-9_]+$")

LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}

_BUILTIN_TOOLS = {
    "finish_note_extraction",
    "stop_workflow",
    "finish_single_extraction",
}


def _bail(msg: str) -> NoReturn:
    """Print a red error and exit with code 1."""
    console.print(f"[red]{msg}[/red]")
    raise typer.Exit(1)


def _configure_fc_logging(level_name: str) -> None:
    """Configure logging for the oncai.fc_extraction package."""
    level = LOG_LEVELS.get(level_name.lower(), logging.WARNING)
    fc_logger = logging.getLogger("oncai.fc_extraction")
    fc_logger.setLevel(level)
    if not fc_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        fc_logger.addHandler(handler)


# -----------------------------------------------------------------------------
# Definition registry
# -----------------------------------------------------------------------------
# Maps the CLI-facing definition name to (module_path, registry_factory_name).
# Each definition module exposes ``DEFINITION_NAME`` and ``SYSTEM_PROMPT`` at
# module level, plus a registry factory function. Add new definitions by
# dropping a module into ``fc_extraction/definitions/`` and registering it here.

_DEFINITIONS: dict[str, tuple[str, str]] = {
    "example": (
        "oncai.fc_extraction.definitions.example",
        "create_example_registry",
    ),
    "path_kidney_basic": (
        "oncai.fc_extraction.definitions.path_kidney_basic",
        "create_kidney_path_basic_registry",
    ),
    "path_kidney_ihc": (
        "oncai.fc_extraction.definitions.path_kidney_ihc",
        "create_ihc_advanced_registry",
    ),
    "path_kidney_nephrectomy": (
        "oncai.fc_extraction.definitions.path_kidney_nephrectomy",
        "create_kidney_path_nephrectomy_registry",
    ),
    "path_kidney_proc_site_hist": (
        "oncai.fc_extraction.definitions.path_kidney_proc_site_hist",
        "create_path_procedure_site_hist_registry",
    ),
}


def _load_definition(name: str):
    """Resolve ``name`` to ``(note_config, registry)``."""
    from oncai.fc_extraction.batch_single import SingleNoteConfig

    if name not in _DEFINITIONS:
        console.print(f"[red]Unknown definition: {name}[/red]")
        console.print(
            "Available definitions: " + ", ".join(sorted(_DEFINITIONS.keys()))
        )
        raise typer.Exit(1)

    module_path, factory_name = _DEFINITIONS[name]
    module = importlib.import_module(module_path)
    registry = getattr(module, factory_name)()
    note_config = SingleNoteConfig(
        name=module.DEFINITION_NAME,
        system_prompt=module.SYSTEM_PROMPT,
    )
    return note_config, registry


fc_app = typer.Typer(help="Function-calling single-note extraction")


# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------


def _validate_source_args(
    *,
    jsonl: Path | None,
    source: str | None,
    where: str | None,
    cohort: str | None,
    incremental: bool,
) -> None:
    """Validate --jsonl / --source mutual exclusion + jsonl-incompatible flags."""
    if jsonl and source:
        _bail("Cannot use both --jsonl and --source")
    if not jsonl and not source:
        _bail("Must supply either --source or --jsonl")
    if jsonl:
        if where:
            _bail("--where is not supported with --jsonl")
        if cohort:
            _bail("--cohort is not supported with --jsonl")
        if incremental:
            _bail("--incremental is not supported with --jsonl")
        if not jsonl.exists():
            _bail(f"JSONL file not found: {jsonl}")


def _validate_batch_args(
    *,
    batch: str | None,
    incremental: bool,
    reextract_on_prompt_change: bool,
    force_rerun: Path | None,
) -> str:
    """Validate --batch + incremental-only flags. Returns the validated batch name."""
    if not batch:
        _bail("--batch is required")
    if not _SAFE_BATCH_NAME.match(batch):
        _bail(f"--batch must be alphanumeric+underscore (got {batch!r})")
    if reextract_on_prompt_change and not incremental:
        _bail("--reextract-on-prompt-change requires --incremental")
    if force_rerun and not incremental:
        _bail("--force-rerun requires --incremental")
    return batch


def _resolve_backend(config, backend_name: str):
    """Look up a backend by name and convert its type to an ``FCBackend`` enum.

    Returns ``(backend_cfg, backend_enum)``.
    """
    from oncai.fc_extraction import FCBackend

    try:
        backend_cfg = config.get_backend(backend_name)
    except ValueError as e:
        _bail(str(e))

    try:
        backend_enum = FCBackend(backend_cfg.type)
    except ValueError:
        _bail(f"Invalid backend type '{backend_cfg.type}' in oncai.yaml")

    return backend_cfg, backend_enum


# -----------------------------------------------------------------------------
# Banner + filter helpers
# -----------------------------------------------------------------------------


def _print_run_banner(
    *,
    note_config,
    batch: str,
    backend_name: str,
    backend_cfg,
    backend_enum,
    jsonl: Path | None,
    source: str | None,
    text_col: str,
    id_col: str,
    reasoning_effort: str,
    verbosity: str,
    temperature: float,
    workers: int,
    limit: int | None,
    where: str | None,
    id_file: Path | None,
    cohort: str | None,
) -> None:
    """Print the configuration banner for a run."""
    from oncai.fc_extraction import FCBackend

    console.print("[blue]Single-Note FC Extraction Configuration[/blue]")
    console.print(f"  Definition: {note_config.name}")
    console.print(f"  Batch:      {batch}")
    console.print(f"  Backend:    {backend_name} ({backend_cfg.type})")
    console.print(f"  JSONL:      {jsonl}" if jsonl else f"  Source:     {source}")
    console.print(f"  Text col:   {text_col}")
    console.print(f"  ID col:     {id_col}")
    if backend_enum == FCBackend.AZURE_RESPONSES:
        console.print(f"  Deployment: {backend_cfg.deployment}")
        console.print(f"  Reasoning:  {reasoning_effort}")
        console.print(f"  Verbosity:  {verbosity}")
    else:
        console.print(f"  Model:      {backend_cfg.model}")
        console.print(f"  Temperature: {temperature}")
    if workers > 1:
        console.print(f"  Workers:    {workers}")
    if limit:
        console.print(f"  Limit:      {limit} notes")
    if where:
        console.print(f"  Where:      {where}")
    if id_file:
        console.print(f"  ID file:    {id_file}")
    if cohort:
        console.print(f"  Cohort:     {cohort}")


def _resolve_note_id_filter(
    *,
    id_file: Path | None,
    cohort: str | None,
    id_col: str,
    lake_path: Path,
) -> set[str] | None:
    """Load the note_id filter from --id-file or --cohort (mutually exclusive)."""
    if id_file and cohort:
        _bail("Cannot use both --id-file and --cohort")

    if id_file:
        from ._shared import load_id_filter

        return load_id_filter(id_file, id_col=id_col)

    if cohort:
        cf = load_cohort_filter(cohort, lake_path)
        if cf.key_column.lower() != id_col.lower():
            console.print(
                f"[red]Cohort key mismatch:[/red] cohort '{cohort}' is keyed by "
                f"'{cf.key_column}', but this run uses --id-col='{id_col}'."
            )
            console.print(
                "  Either pass a cohort keyed on the same column, or set "
                "--id-col to match."
            )
            raise typer.Exit(1)
        console.print(
            f"  Cohort '{cohort}' contributed {len(cf.values):,} {id_col}s "
            f"(keyed on '{cf.key_column}')"
        )
        return cf.values

    return None


# -----------------------------------------------------------------------------
# Incremental
# -----------------------------------------------------------------------------


def _next_version_jsonl(inbox_fc_dir: Path, batch: str) -> str:
    """Return the next ``<batch>.v<N>.jsonl`` filename based on inbox state.

    Scans ``inbox_fc_dir`` for files matching ``<batch>.v<N>.jsonl`` and
    returns ``<batch>.v<N+1>.jsonl``. If no versioned files exist (only the
    baseline ``<batch>.jsonl`` is present, or nothing at all), returns
    ``<batch>.v2.jsonl`` — the baseline is implicitly v1.
    """
    pattern = re.compile(rf"^{re.escape(batch)}\.v(\d+)\.jsonl$")
    highest = 1  # baseline is implicit v1
    if inbox_fc_dir.exists():
        for f in inbox_fc_dir.iterdir():
            m = pattern.match(f.name)
            if m:
                highest = max(highest, int(m.group(1)))
    return f"{batch}.v{highest + 1}.jsonl"


def _load_baseline_extractions(
    con: duckdb.DuckDBPyConnection, batch: str
) -> dict[str, list[tuple[str | None, str | None]]]:
    """Load the success-only resume set from ``extractions_raw.<batch>``.

    Returns a mapping ``record_id -> [(source_content_hash, system_prompt_hash), ...]``.
    The list-per-id shape preserves every prior version of each record so prompt-
    change detection can see all hashes that were used. The hash fields are
    ``None`` for legacy rows produced before those columns existed.

    Raises ``FileNotFoundError`` if the baseline batch table doesn't exist —
    incremental runs are only meaningful as an extension of an existing baseline.
    """
    baseline_exists = bool(
        con.execute(
            """
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = 'extractions_raw' AND table_name = ?
            """,
            [batch],
        ).fetchone()[0]  # type: ignore[index]
    )
    if not baseline_exists:
        raise FileNotFoundError(
            f"extractions_raw.{batch} not found — run a baseline batch "
            f"first (without --incremental), then ingest + build-db."
        )

    # Detect whether the baseline carries the hash columns. Older runs predate
    # them; the SELECT falls back to NULL so the post-load shape is uniform.
    cols = con.execute(
        """
        SELECT column_name FROM information_schema.columns
        WHERE table_schema = 'extractions_raw' AND table_name = ?
        """,
        [batch],
    ).fetchall()
    col_names = {c[0] for c in cols}
    src_expr = (
        '"source_content_hash"'
        if "source_content_hash" in col_names
        else "CAST(NULL AS VARCHAR)"
    )
    prompt_expr = (
        '"system_prompt_hash"'
        if "system_prompt_hash" in col_names
        else "CAST(NULL AS VARCHAR)"
    )
    rows = con.execute(
        f'''SELECT record_id, {src_expr}, {prompt_expr}
            FROM extractions_raw."{batch}"
            WHERE success = TRUE
        '''
    ).fetchall()

    existing: dict[str, list[tuple[str | None, str | None]]] = {}
    for rid, sh, ph in rows:
        existing.setdefault(str(rid), []).append((sh, ph))
    return existing


def _query_source_state(
    con: duckdb.DuckDBPyConnection,
    source_table: str,
    id_col: str,
    where: str | None,
) -> list[tuple[Any, Any]]:
    """Return current ``(id, content_hash)`` tuples from the source table.

    ``content_hash`` is selected as NULL when the source table lacks the column
    (legacy schemas), so the categorize step can treat all sources uniformly.
    """
    from oncai.fc_extraction.batch_single import _source_table_has_column

    has_hash = _source_table_has_column(con, source_table, "content_hash")
    hash_select = '"content_hash"' if has_hash else "CAST(NULL AS BLOB)"
    query = (
        f'SELECT "{id_col}" AS note_id, {hash_select} AS source_content_hash '
        f"FROM {source_table}"
    )
    if where:
        query += f" WHERE ({where})"
    return con.execute(query).fetchall()


def _normalize_content_hash(raw: Any) -> str | None:
    """Convert a DuckDB content_hash value to an optional hex string.

    DuckDB returns BLOB columns as ``bytes``/``bytearray``; older rows may carry
    strings; missing values are ``None``. Normalising up front keeps the
    categorize loop free of isinstance branches.
    """
    if isinstance(raw, (bytes, bytearray)):
        return raw.hex()
    if raw is not None:
        return str(raw)
    return None


def _categorize_delta(
    *,
    source_rows: list[tuple[Any, Any]],
    existing: dict[str, list[tuple[str | None, str | None]]],
    note_id_set: set[str] | None,
    system_prompt_hash: str,
    reextract_on_prompt_change: bool,
    forced_rerun_ids: set[str] | None,
) -> tuple[set[str], dict[str, int], set[str]]:
    """Bucket each source row into new / changed / prompt_changed / forced / skipped.

    The categorization for a given ``record_id``:
      - **new**: id not in the baseline at all.
      - **changed**: id is in baseline but no prior row matches the current
        content_hash (i.e. an addendum / content edit).
      - **prompt_changed**: id matches a baseline content_hash but none of the
        matching rows used the current ``system_prompt_hash``. Only counted
        when ``reextract_on_prompt_change`` is set.
      - **forced**: id would otherwise be skipped (hash + prompt match, or
        legacy row with no hash), but the caller explicitly listed it in
        ``forced_rerun_ids``.
      - **skipped**: id matches a prior extraction and the caller didn't
        force a re-run.

    ``note_id_set`` (if provided) further restricts the scope; forced ids
    constrained out by it are returned as ``forced_missing`` for caller warning.

    Returns ``(delta_id_set, counts, forced_missing_ids)``.
    """
    new_ids: set[str] = set()
    changed_ids: set[str] = set()
    prompt_changed_ids: set[str] = set()
    forced_ids: set[str] = set()
    skipped = 0

    forced_set = forced_rerun_ids or set()
    forced_seen: set[str] = set()

    for nid_raw, content_hash_raw in source_rows:
        nid = str(nid_raw)
        if note_id_set is not None and nid not in note_id_set:
            continue
        if nid in forced_set:
            forced_seen.add(nid)
        content_hash = _normalize_content_hash(content_hash_raw)

        prior = existing.get(nid)
        if not prior:
            new_ids.add(nid)
            continue

        # Legacy fallback: any prior row predates content_hash → treat as
        # already-done (or forced if explicitly requested).
        if any(p[0] is None for p in prior):
            if nid in forced_set:
                forced_ids.add(nid)
            else:
                skipped += 1
            continue

        matching_hash = [p for p in prior if p[0] == content_hash]
        if not matching_hash:
            changed_ids.add(nid)
            continue

        if reextract_on_prompt_change and not any(
            p[1] == system_prompt_hash for p in matching_hash
        ):
            prompt_changed_ids.add(nid)
            continue

        # Hash + (optionally) prompt match: would be skipped, unless forced.
        if nid in forced_set:
            forced_ids.add(nid)
        else:
            skipped += 1

    delta = new_ids | changed_ids | prompt_changed_ids | forced_ids
    forced_missing = forced_set - forced_seen
    return (
        delta,
        {
            "new": len(new_ids),
            "changed": len(changed_ids),
            "prompt_changed": len(prompt_changed_ids),
            "forced": len(forced_ids),
            "skipped": skipped,
        },
        forced_missing,
    )


def _compute_incremental_delta(
    *,
    db_path: Path,
    source_table: str,
    id_col: str,
    note_id_set: set[str] | None,
    where: str | None,
    batch: str,
    system_prompt_hash: str,
    reextract_on_prompt_change: bool,
    forced_rerun_ids: set[str] | None = None,
) -> tuple[set[str], dict[str, int], set[str]]:
    """Compute the source IDs that need extraction for a batch-pinned run.

    Reads the resume set from ``extractions_raw.<batch>`` (success-only),
    queries the source table for current ``(id_col, content_hash)``, and
    anti-joins to find new + changed + (when requested) prompt-changed rows.

    ``forced_rerun_ids`` adds any ids that would otherwise be skipped into a
    ``forced`` bucket (already-extracted but the user wants to redo them).
    Forced ids constrained out by ``note_id_set`` / ``where`` won't appear in
    the source query and are returned as ``forced_missing`` for warning.

    Errors out if ``extractions_raw.<batch>`` doesn't exist — incremental is
    only meaningful as an extension of an existing baseline batch.

    Returns:
        (delta_id_set, counts, forced_missing_ids) where counts has keys
        ``new``, ``changed``, ``prompt_changed``, ``forced``, ``skipped``.
    """
    import duckdb

    if not _SAFE_BATCH_NAME.match(batch):
        raise ValueError(
            f"--batch must be alphanumeric+underscore for incremental: {batch!r}"
        )

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        existing = _load_baseline_extractions(con, batch)
        source_rows = _query_source_state(con, source_table, id_col, where)
    finally:
        con.close()

    return _categorize_delta(
        source_rows=source_rows,
        existing=existing,
        note_id_set=note_id_set,
        system_prompt_hash=system_prompt_hash,
        reextract_on_prompt_change=reextract_on_prompt_change,
        forced_rerun_ids=forced_rerun_ids,
    )


def _run_incremental_step(
    *,
    config,
    note_config,
    batch: str,
    source: str,
    id_col: str,
    note_id_set: set[str] | None,
    where: str | None,
    reextract_on_prompt_change: bool,
    force_rerun: Path | None,
) -> tuple[set[str], str] | None:
    """Run the incremental delta + output-name-bump step.

    ``source`` is required (incremental needs a DuckDB source table — it is
    mutually exclusive with ``--jsonl``, enforced upstream by
    ``_validate_source_args``).

    Returns ``(new_note_id_set, output_batch_name)``, or ``None`` if the delta
    is empty (caller should treat as nothing-to-do and bail cleanly).
    """
    from oncai.fc_extraction.manifest import hash_string

    from ._shared import load_id_filter

    forced_rerun_ids: set[str] | None = None
    if force_rerun:
        forced_rerun_ids = load_id_filter(force_rerun, id_col=id_col)
        console.print(
            f"  Force-rerun:   {len(forced_rerun_ids):,} {id_col}(s) "
            f"from {force_rerun.name}"
        )

    sys_prompt_hash = hash_string(note_config.system_prompt)
    try:
        delta, counts, forced_missing = _compute_incremental_delta(
            db_path=config.db_path,
            source_table=source,
            id_col=id_col,
            note_id_set=note_id_set,
            where=where,
            batch=batch,
            system_prompt_hash=sys_prompt_hash,
            reextract_on_prompt_change=reextract_on_prompt_change,
            forced_rerun_ids=forced_rerun_ids,
        )
    except FileNotFoundError as e:
        _bail(str(e))

    console.print(
        f"[blue]Incremental delta:[/blue] {counts['new']} new + "
        f"{counts['changed']} changed (addenda) + "
        f"{counts['prompt_changed']} prompt-changed + "
        f"{counts['forced']} forced; "
        f"skipping {counts['skipped']} already-extracted"
    )
    if forced_missing:
        preview = ", ".join(sorted(forced_missing)[:10])
        extra = (
            f" (and {len(forced_missing) - 10} more)"
            if len(forced_missing) > 10
            else ""
        )
        console.print(
            f"[yellow]Warning:[/yellow] {len(forced_missing)} "
            f"force-rerun id(s) not present in source query result "
            f"(check --cohort/--id-file/--where scope, or that the id "
            f"exists in {source}): {preview}{extra}"
        )
    if not delta:
        return None

    inbox_fc_dir = config.inbox_path / "fc_extractions"
    next_name = _next_version_jsonl(inbox_fc_dir, batch)
    output_batch_name = next_name.removesuffix(".jsonl")
    console.print(f"  Output batch:  {output_batch_name}.jsonl")
    return delta, output_batch_name


# -----------------------------------------------------------------------------
# Client + run logging
# -----------------------------------------------------------------------------


def _build_fc_client(
    *,
    backend_enum,
    backend_cfg,
    config,
    reasoning_effort: str,
    verbosity: str,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    debug: bool,
):
    """Wire up an FC client from connection info + run-specific sampling params."""
    from oncai.fc_extraction import FCBackend, FCClientConfig, get_fc_client

    client_config = FCClientConfig(
        reasoning_effort=reasoning_effort,
        text_verbosity=verbosity,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_rounds_per_note=12,
    )

    client_kwargs: dict[str, Any] = {}
    if debug:
        debug_dir = config.lake_path.parent / "fc_debug"
        client_kwargs["debug_dir"] = str(debug_dir)
        console.print(
            f"  [yellow]Debug:    saving requests/responses to {debug_dir}[/yellow]"
        )

    if backend_enum == FCBackend.AZURE_RESPONSES:
        client_kwargs["endpoint"] = backend_cfg.endpoint
        client_kwargs["api_key"] = backend_cfg.resolve_api_key()
        client_kwargs["deployment"] = backend_cfg.deployment
        client_kwargs["api_version"] = backend_cfg.api_version
    elif backend_enum in (FCBackend.VLLM, FCBackend.VLLM_CHAT):
        client_kwargs["base_url"] = backend_cfg.base_url
        client_kwargs["model"] = backend_cfg.model
        api_key = (
            backend_cfg.resolve_api_key()
            if backend_cfg.api_key_env != "AZURE_OPENAI_API_KEY"
            else "not-needed"
        )
        client_kwargs["api_key"] = api_key

    return get_fc_client(backend=backend_enum, config=client_config, **client_kwargs)


def _describe_mrn_source(
    *,
    id_file: Path | None,
    cohort: str | None,
    limit: int | None,
) -> str:
    """Short string summarising where the run's IDs came from (for the run log)."""
    if id_file:
        return f"file:{id_file.name}"
    if cohort:
        return f"cohort:{cohort}"
    if limit:
        return f"limit:{limit}"
    return "all"


def _prelog_run(
    *,
    config,
    note_config,
    registry,
    fc_client,
    backend_name: str,
    backend_enum,
    batch_name: str,
    started_at: str,
    source: str | None,
    jsonl: Path | None,
    text_col: str,
    workers: int,
    reasoning_effort: str,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    mrn_source: str,
) -> str | None:
    """Write a "started" entry to lake/runs/. Returns run_id, or None on failure.

    Failures here are logged as warnings rather than aborting — losing the run
    log is unfortunate but shouldn't kill an otherwise-good batch.
    """
    from oncai.fc_extraction import FCBackend

    try:
        from oncai.runs import (
            RunLog,
            _generate_run_id,
            _get_code_version,
            _get_git_info,
            _hash_string,
            log_run,
        )

        git_info = _get_git_info()
        tool_names = [t for t in registry.list_tools() if t not in _BUILTIN_TOOLS]
        tool_schemas = {
            t: registry.get(t).model.model_json_schema()
            for t in tool_names
        }
        resolved_model = getattr(fc_client, "model", None) or getattr(
            fc_client, "deployment", None
        )

        run_id = _generate_run_id("fc_single", note_config.name, batch_name, started_at)
        is_azure = backend_enum == FCBackend.AZURE_RESPONSES
        is_vllm = backend_enum in (FCBackend.VLLM, FCBackend.VLLM_CHAT)
        run = RunLog(
            run_id=run_id,
            run_type="fc_single",
            name=note_config.name,
            batch_name=batch_name,
            status="started",
            started_at=started_at,
            git_commit=git_info.commit,
            git_branch=git_info.branch,
            git_dirty=git_info.dirty,
            code_version=_get_code_version(),
            backend=backend_name,
            model=resolved_model,
            reasoning_effort=reasoning_effort if is_azure else None,
            temperature=temperature if is_vllm else None,
            top_p=top_p if is_vllm else None,
            top_k=top_k if is_vllm else None,
            source_table=source or (f"jsonl:{jsonl.name}" if jsonl else None),
            text_column=text_col,
            workers=workers,
            system_prompt=note_config.system_prompt,
            system_prompt_hash=_hash_string(note_config.system_prompt),
            tools_json=json.dumps(tool_names),
            tool_schemas_json=json.dumps(tool_schemas),
            db_path=str(jsonl) if jsonl else str(config.db_path),
            mrn_source=mrn_source,
        )
        log_run(run, config.lake_path)
    except Exception as e:
        console.print(f"  [yellow]Warning: failed to pre-log run: {e}[/yellow]")
        return None
    else:
        return run_id


def _finalize_run(
    *,
    run_id: str | None,
    lake_path: Path,
    t0: float,
    status: str,
    **extra: Any,
) -> None:
    """Update an in-flight run log with terminal status + any result fields.

    Silently swallows logging failures so the user-facing flow isn't affected
    when the runs parquet has a problem.
    """
    if not run_id:
        return
    try:
        from oncai.runs import update_run

        update_run(
            run_id,
            lake_path,
            status=status,
            completed_at=datetime.now(timezone.utc).isoformat(),
            duration_seconds=round(time.monotonic() - t0, 2),
            **extra,
        )
    except Exception as e:
        if status == "completed":
            console.print(f"  [yellow]Warning: failed to update run log: {e}[/yellow]")


# -----------------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------------


@fc_app.command(name="run-single")
def fc_run_single(
    definition: str = typer.Argument(..., help="Definition name (see `oncai fc list`)"),
    batch: str | None = typer.Option(
        None,
        "--batch",
        help="Batch name for this run. Required. With --incremental, this "
        "pins to the existing extractions_raw.<batch> baseline; the actual "
        "output JSONL is auto-bumped to <batch>.v<N>.jsonl.",
    ),
    source: str | None = typer.Option(
        None,
        "--source",
        "-s",
        help="DuckDB source table (e.g., raw.pathology). Mutually exclusive with --jsonl.",
    ),
    jsonl: Path | None = typer.Option(
        None,
        "--jsonl",
        help="JSONL file to load notes from (one JSON object per line). "
        "Skips DuckDB entirely; incompatible with --source/--where/--cohort/--incremental.",
    ),
    text_col: str = typer.Option(
        "report_text", "--text-col", help="Column/JSON key containing the row's text"
    ),
    id_col: str = typer.Option(
        "report_id", "--id-col", help="Column/JSON key containing the row identifier"
    ),
    backend: str = typer.Option(
        ...,
        "--backend",
        "-b",
        help="Named LLM backend from oncai.yaml (e.g. 'azure-prod', 'vllm-local')",
    ),
    limit: int | None = typer.Option(
        None, "--limit", "-n", help="Max notes to process"
    ),
    where: str | None = typer.Option(
        None, "--where", "-w", help="SQL WHERE clause filter"
    ),
    id_file: Path | None = typer.Option(
        None,
        "--id-file",
        "-f",
        help="CSV file with note/report IDs to filter to (auto-detects column: note_id, report_id, id)",
    ),
    cohort: str | None = typer.Option(
        None,
        "--cohort",
        help="Named cohort from lake/cohorts/ to filter rows. Cohort's key "
        "column must match --id-col.",
    ),
    reasoning_effort: str = typer.Option(
        "medium",
        "--reasoning-effort",
        "-e",
        help="Reasoning effort: low, medium, high, or none to disable (Azure only)",
    ),
    verbosity: str = typer.Option(
        "low",
        "--verbosity",
        "-v",
        help="Text output verbosity: low, medium, high, or none to disable (Azure only)",
    ),
    temperature: float = typer.Option(
        0.0, "--temperature", "-t", help="Sampling temperature (vLLM only)"
    ),
    top_p: float | None = typer.Option(
        None, "--top-p", help="Nucleus sampling probability (vLLM only)"
    ),
    top_k: int | None = typer.Option(
        None, "--top-k", help="Top-k sampling (vLLM only; sent via extra_body)"
    ),
    no_resume: bool = typer.Option(
        False, "--no-resume", help="Don't skip already-processed notes"
    ),
    retry_failed: bool = typer.Option(
        False,
        "--retry-failed",
        help="Drop previously-failed records and retry just those notes. "
        "Backs up the original JSONL to <name>.jsonl.bak.",
    ),
    incremental: bool = typer.Option(
        False,
        "--incremental",
        help="Pin to an existing baseline batch and run only on rows that "
        "aren't yet in extractions_raw.<batch> (or whose source content_hash "
        "differs). Output goes to fc_outputs/<DefinitionName>/<batch>.v<N>.jsonl.",
    ),
    reextract_on_prompt_change: bool = typer.Option(
        False,
        "--reextract-on-prompt-change",
        help="When --incremental is set, also re-extract rows whose "
        "system_prompt_hash differs from the current definition's prompt.",
    ),
    force_rerun: Path | None = typer.Option(
        None,
        "--force-rerun",
        help="CSV of report_ids to force-rerun even if their hash matches "
        "the baseline. Requires --incremental.",
    ),
    rate_limit: float = typer.Option(
        1.0,
        "--rate-limit",
        "-r",
        help="Seconds to wait between notes (ignored with --workers > 1)",
    ),
    workers: int = typer.Option(
        1, "--workers", "-j", help="Number of concurrent workers"
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Save full request/response JSON for each API call to fc_debug/",
    ),
    log_level: str = typer.Option(
        "warning", "--log-level", "-l", help="Extraction log level"
    ),
):
    """
    Run single-note function-calling extraction.

    Each note is processed independently. Designed for pathology reports where
    one report = one extraction.

    Examples:
        oncai fc run-single path_kidney_basic --batch v1 --source raw.pathology --limit 5
        oncai fc run-single path_kidney_ihc --batch v1 --source raw.pathology
    """
    _configure_fc_logging(log_level)
    config = get_config()
    note_config, registry = _load_definition(definition)

    # ---- validate inputs -----------------------------------------------------
    _validate_source_args(
        jsonl=jsonl,
        source=source,
        where=where,
        cohort=cohort,
        incremental=incremental,
    )
    batch = _validate_batch_args(
        batch=batch,
        incremental=incremental,
        reextract_on_prompt_change=reextract_on_prompt_change,
        force_rerun=force_rerun,
    )
    backend_cfg, backend_enum = _resolve_backend(config, backend)

    # ---- banner + filters ----------------------------------------------------
    _print_run_banner(
        note_config=note_config,
        batch=batch,
        backend_name=backend,
        backend_cfg=backend_cfg,
        backend_enum=backend_enum,
        jsonl=jsonl,
        source=source,
        text_col=text_col,
        id_col=id_col,
        reasoning_effort=reasoning_effort,
        verbosity=verbosity,
        temperature=temperature,
        workers=workers,
        limit=limit,
        where=where,
        id_file=id_file,
        cohort=cohort,
    )

    note_id_set = _resolve_note_id_filter(
        id_file=id_file,
        cohort=cohort,
        id_col=id_col,
        lake_path=config.lake_path,
    )

    if not jsonl and not config.db_path.exists():
        console.print(f"[red]Database not found: {config.db_path}[/red]")
        console.print("Run 'oncai build-db' first.")
        raise typer.Exit(1)

    # ---- incremental delta (optional) ---------------------------------------
    output_batch_name = batch
    if incremental:
        # _validate_source_args enforces (jsonl XOR source) and forbids
        # incremental + jsonl; therefore source is non-None here. This guard
        # is for the type-checker; should be unreachable at runtime.
        if source is None:
            raise RuntimeError("incremental requires --source")
        step = _run_incremental_step(
            config=config,
            note_config=note_config,
            batch=batch,
            source=source,
            id_col=id_col,
            note_id_set=note_id_set,
            where=where,
            reextract_on_prompt_change=reextract_on_prompt_change,
            force_rerun=force_rerun,
        )
        if step is None:
            console.print("[green]Nothing to extract — all up to date.[/green]")
            return
        note_id_set, output_batch_name = step

    # ---- client + run logging ------------------------------------------------
    fc_client = _build_fc_client(
        backend_enum=backend_enum,
        backend_cfg=backend_cfg,
        config=config,
        reasoning_effort=reasoning_effort,
        verbosity=verbosity,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        debug=debug,
    )
    console.print(f"  Tools:      {', '.join(registry.list_tools())}")
    console.print("\n[blue]Starting single-note extraction...[/blue]")

    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.monotonic()
    run_id = _prelog_run(
        config=config,
        note_config=note_config,
        registry=registry,
        fc_client=fc_client,
        backend_name=backend,
        backend_enum=backend_enum,
        batch_name=output_batch_name,
        started_at=started_at,
        source=source,
        jsonl=jsonl,
        text_col=text_col,
        workers=workers,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        mrn_source=_describe_mrn_source(id_file=id_file, cohort=cohort, limit=limit),
    )

    # ---- run the batch -------------------------------------------------------
    from oncai.fc_extraction.batch_single import run_fc_single_batch

    output_dir = config.lake_path.parent / "fc_outputs"
    try:
        result = run_fc_single_batch(
            registry=registry,
            config=note_config,
            client=fc_client,
            db_path=config.db_path if not jsonl else None,
            source_table=source,
            output_dir=output_dir,
            batch_name=output_batch_name,
            text_col=text_col,
            id_col=id_col,
            limit=limit,
            where=where,
            note_ids=note_id_set,
            resume=not no_resume,
            retry_failed=retry_failed,
            rate_limit=rate_limit,
            workers=workers,
            backend_name=backend,
            jsonl_path=jsonl,
        )
    except KeyboardInterrupt:
        _finalize_run(
            run_id=run_id, lake_path=config.lake_path, t0=t0, status="cancelled"
        )
        console.print("\n[yellow]Run cancelled by user.[/yellow]")
        raise typer.Exit(1) from None
    except Exception:
        _finalize_run(run_id=run_id, lake_path=config.lake_path, t0=t0, status="failed")
        raise

    _finalize_run(
        run_id=run_id,
        lake_path=config.lake_path,
        t0=t0,
        status="completed",
        input_count=result.total_notes,
        items_processed=result.total_notes,
        items_succeeded=result.successful,
        items_failed=result.failed,
        items_skipped=result.skipped,
        total_input_tokens=result.total_input_tokens,
        total_output_tokens=result.total_output_tokens,
        output_path=result.output_path,
    )

    console.print(f"\n[green]✓[/green] {result}")
    console.print(f"  Output: {result.output_path}")


@fc_app.command(name="unstage")
def fc_unstage(
    batch: str = typer.Argument(
        ...,
        help="Batch stem to unstage (drops extractions_staging.<batch>). "
        "Pass --all to drop the whole staging schema.",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
    all: bool = typer.Option(False, "--all", help="Drop all staged tables"),
):
    """
    Drop a staged table from the ``extractions_staging`` schema.

    Examples:
        oncai fc unstage path_basic_v3
        oncai fc unstage path_basic_v3 --yes
        oncai fc unstage IGNORED --all          # drops every staged table
    """
    import duckdb

    config = get_config()
    schema_name = "extractions_staging"

    if not config.db_path.exists():
        _bail(f"Database not found: {config.db_path}")

    con = duckdb.connect(str(config.db_path))
    try:
        if all:
            tables = con.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = ?",
                [schema_name],
            ).fetchall()
            if not tables:
                console.print(f"[yellow]No tables in {schema_name}[/yellow]")
                return
            console.print(f"[blue]{schema_name}: {len(tables)} table(s)[/blue]")
            for (tbl,) in tables:
                row_count = con.execute(
                    f'SELECT COUNT(*) FROM "{schema_name}"."{tbl}"'
                ).fetchone()[0]  # type: ignore[index]
                console.print(f"  {tbl}: {row_count:,} rows")

            if not yes and not typer.confirm(
                f"Drop ALL {len(tables)} staged tables in {schema_name}?"
            ):
                console.print("[yellow]Cancelled[/yellow]")
                return

            for (tbl,) in tables:
                con.execute(f'DROP TABLE "{schema_name}"."{tbl}"')
            console.print(
                f"\n[green]✓[/green] Dropped {len(tables)} table(s) from {schema_name}"
            )
            return

        exists = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_schema = ? AND table_name = ?",
            [schema_name, batch],
        ).fetchone()[0]  # type: ignore[index]

        if not exists:
            console.print(
                f"[yellow]No staged table {schema_name}.{batch} found[/yellow]"
            )
            return

        row_count = con.execute(
            f'SELECT COUNT(*) FROM "{schema_name}"."{batch}"'
        ).fetchone()[0]  # type: ignore[index]
        console.print(f"[blue]{schema_name}.{batch}: {row_count:,} rows[/blue]")

        if not yes and not typer.confirm(f"Drop table {schema_name}.{batch}?"):
            console.print("[yellow]Cancelled[/yellow]")
            return

        con.execute(f'DROP TABLE "{schema_name}"."{batch}"')
    finally:
        con.close()

    console.print(f"\n[green]✓[/green] Dropped {schema_name}.{batch}")


@fc_app.command(name="status")
def fc_status(
    path: Path | None = typer.Argument(None, help="Path to JSONL file or directory"),
):
    """Show function-calling extraction results statistics."""
    from oncai.fc_extraction.batch_single import get_single_note_batch_status

    config = get_config()

    if path is None:
        path = config.lake_path.parent / "fc_outputs"

    if not path.exists():
        console.print(f"[yellow]No FC outputs found at {path}[/yellow]")
        return

    files = [path] if path.is_file() else list(path.glob("**/*.jsonl"))
    if not files:
        console.print(f"[yellow]No JSONL files found in {path}[/yellow]")
        return

    table = Table(title="Function-Calling Extraction Results")
    table.add_column("File")
    table.add_column("Total", justify="right")
    table.add_column("Success", justify="right")
    table.add_column("Failed", justify="right")
    table.add_column("Rate", justify="right")

    for jsonl_file in sorted(files):
        stats = get_single_note_batch_status(jsonl_file)
        total = stats["total"]
        success = stats["successful"]
        failed = stats["failed"]
        rate = f"{success / total:.1%}" if total > 0 else "N/A"
        table.add_row(
            str(jsonl_file.relative_to(path.parent if path.is_file() else path)),
            str(total),
            str(success),
            str(failed),
            rate,
        )

    console.print(table)


@fc_app.command(name="list")
def fc_list():
    """List available single-note definitions."""
    defn_table = Table(
        title="Single-Note Definitions  (oncai fc run-single <definition>)"
    )
    defn_table.add_column("Definition", style="cyan")
    for defn in sorted(_DEFINITIONS.keys()):
        defn_table.add_row(defn)
    console.print(defn_table)


@fc_app.command(name="manifest")
def fc_manifest(
    path: Path = typer.Argument(
        ..., help="Path to manifest JSON file or batch JSONL file"
    ),
):
    """View the manifest for a batch run.

    Shows all configuration, git info, and results summary that went into
    producing a batch of extractions.

    Examples:
        oncai fc manifest fc_outputs/KidneyPathBasic/v1_manifest.json
        oncai fc manifest fc_outputs/KidneyPathBasic/v1.jsonl
    """
    from rich.panel import Panel
    from rich.syntax import Syntax

    manifest_path = (
        path.with_name(path.stem + "_manifest.json")
        if path.suffix == ".jsonl"
        else path
    )
    if not manifest_path.exists():
        _bail(f"Manifest not found: {manifest_path}")

    with manifest_path.open() as f:
        manifest = json.load(f)

    json_str = json.dumps(manifest, indent=2)
    syntax = Syntax(json_str, "json", theme="monokai", line_numbers=False)
    console.print(
        Panel(
            syntax, title=f"[bold]Manifest: {manifest_path.name}[/bold]", expand=False
        )
    )

    results = manifest.get("results", {})
    git = manifest.get("git", {})

    console.print("\n[bold]Quick Summary:[/bold]")
    console.print(
        f"  Definition:  {manifest.get('definition_name') or manifest.get('workflow_name')}"
    )
    console.print(f"  Batch:       {manifest.get('batch_name')}")
    dirty = " (dirty)" if git.get("dirty") else ""
    console.print(f"  Git commit:  {git.get('commit', 'unknown')}{dirty}")
    console.print(f"  Started:     {manifest.get('started_at')}")
    console.print(f"  Completed:   {manifest.get('completed_at', 'in progress')}")
    console.print(
        f"  Notes:       {results.get('notes_succeeded', 0)}/"
        f"{results.get('notes_processed', 0)} successful"
        f" ({results.get('notes_skipped', 0)} skipped)"
    )
    console.print(f"  Events:      {results.get('total_events', 0)} total")
    console.print(
        f"  Tokens:      {results.get('total_input_tokens', 0):,} in / "
        f"{results.get('total_output_tokens', 0):,} out"
    )


@fc_app.command(name="stage")
def fc_stage(
    jsonl_path: Path = typer.Argument(..., help="Path to JSONL file to stage"),
    skip_failed: bool = typer.Option(
        True, "--skip-failed/--include-failed", help="Skip failed extractions"
    ),
    replace: bool = typer.Option(
        False, "--replace", help="Replace existing staging table instead of appending"
    ),
):
    """
    Stage FC extraction results into DuckDB for quick exploration.

    Lands at ``extractions_staging.<jsonl_stem>``.

    Drop one staged table with ``oncai fc unstage <stem>``, or all staged
    tables with ``oncai fc unstage --all``.

    Examples:
        oncai fc stage fc_outputs/KidneyPathBasic/path_basic_v3.jsonl
            → extractions_staging.path_basic_v3
    """
    import tempfile

    import duckdb

    from oncai.fc_extraction.load import _flatten_fc_single_record, _load_fc_records

    config = get_config()

    if not jsonl_path.exists():
        _bail(f"File not found: {jsonl_path}")
    if not config.db_path.exists():
        console.print(f"[red]Database not found: {config.db_path}[/red]")
        console.print("Run 'oncai build-db' first.")
        raise typer.Exit(1)

    records = _load_fc_records(jsonl_path)
    if not records:
        console.print("[yellow]No records found in JSONL[/yellow]")
        return

    all_rows: list[dict] = []
    skipped = 0
    for record in records:
        if skip_failed and not record.get("success", False):
            skipped += 1
            continue
        all_rows.extend(_flatten_fc_single_record(record))

    if not all_rows:
        console.print(
            f"[yellow]No events to stage ({skipped} failed records skipped)[/yellow]"
        )
        return

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        for row in all_rows:
            tmp.write(json.dumps(row, default=str) + "\n")
        tmp_path = tmp.name

    schema_name = "extractions_staging"
    table_name = jsonl_path.stem

    console.print("[blue]Staging FC extractions into DuckDB[/blue]")
    console.print(f"  Source:  {jsonl_path}")
    console.print(f"  Table:   {schema_name}.{table_name}")
    console.print(f"  Records: {len(records)} notes ({skipped} failed skipped)")
    console.print(f"  Rows:    {len(all_rows)} events")

    con = duckdb.connect(str(config.db_path))
    try:
        con.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema_name}"')
        con.execute(
            f'CREATE OR REPLACE TABLE "{schema_name}"."{table_name}" '
            f"AS SELECT * FROM read_json_auto('{tmp_path}')"
        )
        row_count = con.execute(
            f'SELECT COUNT(*) FROM "{schema_name}"."{table_name}"'
        ).fetchone()[0]  # type: ignore[index]
    finally:
        con.close()
        Path(tmp_path).unlink(missing_ok=True)

    console.print(
        f"\n[green]✓[/green] Staged {row_count} rows into {schema_name}.{table_name}"
    )
    console.print(f"  Query:   SELECT * FROM {schema_name}.{table_name} LIMIT 10")
