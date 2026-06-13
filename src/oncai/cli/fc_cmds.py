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

_RUN_REVIEW_PACKAGE_SCOPES = {"all", "flagged", "none"}
_REVIEW_SELECTION_SCOPES = {
    "all",
    "flagged",
    "disagreements",
    "flagged-or-disagreements",
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
        "create_path_kidney_basic_registry",
    ),
    "path_kidney_ihc": (
        "oncai.fc_extraction.definitions.path_kidney_ihc",
        "create_path_kidney_ihc_registry",
    ),
    "path_kidney_nephrectomy": (
        "oncai.fc_extraction.definitions.path_kidney_nephrectomy",
        "create_path_kidney_nephrectomy_registry",
    ),
    "path_kidney_proc_site_hist": (
        "oncai.fc_extraction.definitions.path_kidney_proc_site_hist",
        "create_path_kidney_proc_site_hist_registry",
    ),
}


def _resolve_definition_name(name: str) -> str | None:
    """Resolve a user-supplied definition identifier to a registered key.

    Accepts either the registered snake_case key (e.g. ``path_kidney_basic``)
    or the module's PascalCase ``DEFINITION_NAME`` (e.g. ``PathKidneyBasic``).
    Returns the snake_case key, or ``None`` if nothing matches. The
    PascalCase scan only runs when the direct key lookup misses, so the
    common case avoids importing every definition module.
    """
    if name in _DEFINITIONS:
        return name
    for key, (module_path, _factory) in _DEFINITIONS.items():
        module = importlib.import_module(module_path)
        if getattr(module, "DEFINITION_NAME", None) == name:
            return key
    return None


def _load_definition(name: str):
    """Resolve ``name`` to ``(note_config, registry)``."""
    from oncai.fc_extraction.batch_single import SingleNoteConfig

    resolved = _resolve_definition_name(name)
    if resolved is None:
        console.print(f"[red]Unknown definition: {name}[/red]")
        console.print(
            "Available definitions: " + ", ".join(sorted(_DEFINITIONS.keys()))
        )
        raise typer.Exit(1)

    module_path, factory_name = _DEFINITIONS[resolved]
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
        if not jsonl.exists():
            _bail(f"JSONL file not found: {jsonl}")


def _validate_batch_args(
    *,
    batch: str | None,
    jsonl: Path | None,
    full: bool,
    reextract_on_prompt_change: bool,
    force_rerun: Path | None,
) -> str:
    """Validate --batch + delta-refinement flags. Returns the validated batch name.

    The delta is always computed (against the batch's existing segments) for
    ``--source`` runs; ``--reextract-on-prompt-change`` and ``--force-rerun``
    refine it, so they only apply there — not in ``--jsonl`` mode (no SQL source
    to diff) and not with ``--full`` (which ignores prior segments by design).
    """
    if not batch:
        _bail("--batch is required")
    if not _SAFE_BATCH_NAME.match(batch):
        _bail(f"--batch must be alphanumeric+underscore (got {batch!r})")
    if (reextract_on_prompt_change or force_rerun) and jsonl:
        _bail("--reextract-on-prompt-change / --force-rerun require --source (not --jsonl)")
    if (reextract_on_prompt_change or force_rerun) and full:
        _bail("--reextract-on-prompt-change / --force-rerun don't apply with --full")
    return batch


def _validate_choice(value: str, *, option: str, choices: set[str]) -> str:
    normalized = value.strip().lower()
    if normalized not in choices:
        _bail(f"{option} must be one of {', '.join(sorted(choices))}")
    return normalized


def _definition_name_from_manifest(jsonl_path: Path) -> str | None:
    manifest_path = jsonl_path.with_name(jsonl_path.stem + "_manifest.json")
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict):
        return None
    definition = manifest.get("definition_name") or manifest.get("workflow_name")
    return str(definition) if definition else None


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


_DESCRIPTOR_NAME = "batch.json"


def _batch_dir(config, batch: str) -> Path:
    """The inbox folder for a batch: ``inbox/fc_extractions/<batch>/``."""
    return config.inbox_path / "fc_extractions" / batch


def _next_segment(batch_dir: Path) -> int:
    """Next segment number for a batch = max existing ``NNN`` + 1 (first run → 1)."""
    from oncai.fc_extraction.load import segment_files

    segs = segment_files(batch_dir)
    return (segs[-1][0] + 1) if segs else 1


def _load_batch_history(
    batch_dir: Path,
) -> dict[str, list[tuple[str | None, str | None]]]:
    """Read the success records across a batch's segments → the resume set.

    Returns ``{record_id: [(source_content_hash, definition_hash), ...]}`` built
    directly from the inbox segments — the canonical source of truth — so the
    incremental delta never depends on the built DuckDB being current. The
    ``definition_hash`` covers prompt + tool schemas, so a field change shows up
    as a definition change.
    """
    from oncai.fc_extraction.load import segment_files

    existing: dict[str, list[tuple[str | None, str | None]]] = {}
    for _, path in segment_files(batch_dir):
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not rec.get("success", False):
                    continue
                rid = str(rec.get("note_id", ""))
                src = rec.get("source_content_hash")
                run_meta = rec.get("run_meta") or {}
                # definition_hash (prompt + tool schemas) is the change key;
                # fall back to the prompt-only hash for records that predate it.
                defn = run_meta.get("definition_hash") or run_meta.get(
                    "system_prompt_hash"
                )
                existing.setdefault(rid, []).append((src, defn))
    return existing


def _read_batch_descriptor(batch_dir: Path) -> dict | None:
    path = batch_dir / _DESCRIPTOR_NAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _check_or_write_batch_descriptor(
    batch_dir: Path,
    *,
    definition: str,
    source: str | None,
    id_col: str,
) -> None:
    """Pin a batch's identity on first run; assert it matches on every re-run.

    The batch name alone is an unenforced label — without this guard, re-running
    ``--batch foo`` against a different source or definition would silently merge
    unrelated records into one table. ``batch.json`` makes the contract explicit.
    """
    fields = {"definition": definition, "source": source, "id_col": id_col}
    desc = _read_batch_descriptor(batch_dir)
    if desc is None:
        batch_dir.mkdir(parents=True, exist_ok=True)
        (batch_dir / _DESCRIPTOR_NAME).write_text(json.dumps(fields, indent=2))
        return
    mismatches = [
        f"{k}: batch={desc.get(k)!r} vs this run={fields[k]!r}"
        for k in fields
        if desc.get(k) != fields[k]
    ]
    if mismatches:
        _bail(
            f"--batch {batch_dir.name!r} was created with different parameters:\n  "
            + "\n  ".join(mismatches)
            + "\nUse a new --batch name, or match the original parameters."
        )


def _finalize_segment(working_jsonl: Path, batch_dir: Path, seg: int) -> Path:
    """Promote a completed working JSONL into the inbox batch folder as NNN.jsonl.

    A run extracts into ``fc_outputs/`` (resumable scratch); on success the
    segment is promoted into ``inbox/fc_extractions/<batch>/NNN.jsonl`` via a
    temp-then-rename so a partial file never appears under the final name. The
    sibling ``NNN_manifest.json`` is promoted alongside it.
    """
    import shutil

    batch_dir.mkdir(parents=True, exist_ok=True)
    final = batch_dir / f"{seg:03d}.jsonl"
    tmp = batch_dir / f"{seg:03d}.jsonl.tmp"
    shutil.copy2(working_jsonl, tmp)
    tmp.replace(final)

    work_manifest = working_jsonl.with_name(working_jsonl.stem + "_manifest.json")
    if work_manifest.exists():
        man_final = batch_dir / f"{seg:03d}_manifest.json"
        man_tmp = batch_dir / f"{seg:03d}_manifest.json.tmp"
        shutil.copy2(work_manifest, man_tmp)
        man_tmp.replace(man_final)
    return final


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
    definition_hash: str,
    reextract_on_prompt_change: bool,
    forced_rerun_ids: set[str] | None,
) -> tuple[set[str], dict[str, int], set[str]]:
    """Bucket each source row into new / changed / definition_changed / forced / skipped.

    The categorization for a given ``record_id``:
      - **new**: id not in the history at all.
      - **changed**: id is in history but no prior row matches the current
        content_hash (i.e. an addendum / content edit).
      - **definition_changed**: id matches a prior content_hash but none of the
        matching rows used the current ``definition_hash`` (prompt + tool
        schemas) — i.e. the prompt or a tool's fields changed. Only counted when
        ``reextract_on_prompt_change`` is set.
      - **forced**: id would otherwise be skipped (hash + definition match, or
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
    definition_changed_ids: set[str] = set()
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
            p[1] == definition_hash for p in matching_hash
        ):
            definition_changed_ids.add(nid)
            continue

        # Hash + (optionally) definition match: would be skipped, unless forced.
        if nid in forced_set:
            forced_ids.add(nid)
        else:
            skipped += 1

    delta = new_ids | changed_ids | definition_changed_ids | forced_ids
    forced_missing = forced_set - forced_seen
    return (
        delta,
        {
            "new": len(new_ids),
            "changed": len(changed_ids),
            "definition_changed": len(definition_changed_ids),
            "forced": len(forced_ids),
            "skipped": skipped,
        },
        forced_missing,
    )


def _compute_delta(
    *,
    config,
    batch_dir: Path,
    source: str,
    id_col: str,
    note_id_set: set[str] | None,
    where: str | None,
    definition_hash: str,
    reextract_on_prompt_change: bool,
    force_rerun: Path | None,
) -> tuple[set[str], dict[str, int], set[str]]:
    """Which source ids need extraction, vs the batch's existing inbox segments.

    History (already-extracted ids + their source/definition hashes) comes from
    the inbox segments — the canonical source of truth, so the delta is correct
    without a ``build-db`` first. Current source state comes from the DuckDB
    source table. Buckets each id into new / changed / definition_changed /
    forced / skipped.

    Returns ``(delta_id_set, counts, forced_missing_ids)``.
    """
    import duckdb

    from ._shared import load_id_filter

    forced_rerun_ids = (
        load_id_filter(force_rerun, id_col=id_col) if force_rerun else None
    )

    history = _load_batch_history(batch_dir)
    con = duckdb.connect(str(config.db_path), read_only=True)
    try:
        source_rows = _query_source_state(con, source, id_col, where)
    finally:
        con.close()

    return _categorize_delta(
        source_rows=source_rows,
        existing=history,
        note_id_set=note_id_set,
        definition_hash=definition_hash,
        reextract_on_prompt_change=reextract_on_prompt_change,
        forced_rerun_ids=forced_rerun_ids,
    )


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
    """Write a "started" manifest to inbox/runs/. Returns run_id, or None on failure.

    Failures here are logged as warnings rather than aborting — losing the run
    manifest is unfortunate but shouldn't kill an otherwise-good batch.
    """
    from oncai.fc_extraction import FCBackend

    try:
        from oncai.runs import (
            RunLog,
            _generate_run_id,
            _get_code_version,
            _get_git_info,
            _hash_string,
            start_run,
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
        start_run(run, config.inbox_path)
    except Exception as e:
        console.print(f"  [yellow]Warning: failed to pre-log run: {e}[/yellow]")
        return None
    else:
        return run_id


def _build_review_package_for_run(
    *,
    note_config,
    registry,
    output_path: str,
    inbox_path: Path,
    batch: str,
    source: str | None,
    jsonl: Path | None,
    db_path: Path,
    text_col: str,
    id_col: str,
    review_package_scope: str,
) -> None:
    """Build the ``*.review_pkg.json`` for a just-completed batch.

    Lands the package in ``inbox/fc_reviews/<batch>/`` (created idempotently) so
    the reviewer's completed ``<segment>.reviews.jsonl`` has an obvious folder to
    drop back into — mirroring ``inbox/fc_extractions/<batch>/``.

    Best-effort: a packaging failure is reported in yellow but never fails the
    run — the extraction JSONL is the source of truth and is already on disk.
    """
    from oncai.review import package_from_jsonl

    scope = _validate_choice(
        review_package_scope,
        option="--review-package",
        choices=_RUN_REVIEW_PACKAGE_SCOPES,
    )
    if scope == "none":
        return

    try:
        if jsonl is not None:
            source_table: str | None = f"jsonl:{jsonl}"
            notes_db: Path | None = jsonl
        else:
            source_table = source
            notes_db = db_path
        pkg_dir = inbox_path / "fc_reviews" / batch
        pkg_dir.mkdir(parents=True, exist_ok=True)
        dest = pkg_dir / f"{Path(output_path).stem}.review_pkg.json"
        pkg_path = package_from_jsonl(
            jsonl_path=Path(output_path),
            registry=registry,
            definition_name=note_config.name,
            source_table=source_table,
            db_path=notes_db,
            text_col=text_col,
            id_col=id_col,
            output_path=dest,
            only_flagged=scope == "flagged",
        )
    except Exception as e:
        console.print(f"  [yellow]Warning: failed to build review package: {e}[/yellow]")
    else:
        console.print(f"  Review package: {pkg_path}")
        console.print(
            f"  [dim]→ when reviewed, drop {Path(output_path).stem}.reviews.jsonl "
            f"into {pkg_dir}[/dim]"
        )


def _finalize_run(
    *,
    run_id: str | None,
    inbox_path: Path,
    t0: float,
    status: str,
    **extra: Any,
) -> None:
    """Fill in the terminal sections of a run's manifest (status + results).

    Silently swallows logging failures so the user-facing flow isn't affected
    when the run manifest has a problem.
    """
    if not run_id:
        return
    try:
        from oncai.runs import complete_run

        complete_run(
            run_id,
            inbox_path,
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
        help="Batch name. Required. A batch is a folder of numbered segments "
        "(inbox/fc_extractions/<batch>/NNN.jsonl); each run appends the next "
        "segment with only the new/changed rows.",
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
        "Skips DuckDB entirely; incompatible with --source/--where/--cohort.",
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
    full: bool = typer.Option(
        False,
        "--full",
        help="Re-extract every matching row into a new segment, ignoring what "
        "the batch's existing segments already cover. The default is "
        "incremental: only rows new or changed since the existing segments.",
    ),
    scratch: bool = typer.Option(
        False,
        "--scratch",
        help="One-off test run: extract to fc_outputs only and do NOT promote a "
        "segment into the inbox (no batch.json, no run log). Runs fresh each "
        "time; inspect the output with `oncai fc peek`.",
    ),
    reextract_on_prompt_change: bool = typer.Option(
        False,
        "--reextract-on-prompt-change",
        help="Also re-extract rows whose definition_hash differs — i.e. the "
        "prompt OR a tool's Pydantic fields/enums changed (--source runs only).",
    ),
    force_rerun: Path | None = typer.Option(
        None,
        "--force-rerun",
        help="CSV of ids to re-extract even if unchanged (--source runs only).",
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
    review_package: str = typer.Option(
        "flagged",
        "--review-package",
        help=(
            "Review package to build after the run: flagged, all, or none. "
            "flagged includes only reports that called flag_report_for_review."
        ),
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
    )
    batch = _validate_batch_args(
        batch=batch,
        jsonl=jsonl,
        full=full,
        reextract_on_prompt_change=reextract_on_prompt_change,
        force_rerun=force_rerun,
    )
    review_package = _validate_choice(
        review_package,
        option="--review-package",
        choices=_RUN_REVIEW_PACKAGE_SCOPES,
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

    # ---- batch folder + identity guard --------------------------------------
    batch_dir = _batch_dir(config, batch)
    output_dir = config.lake_path.parent / "fc_outputs"
    if not scratch:
        _check_or_write_batch_descriptor(
            batch_dir,
            definition=note_config.name,
            source=source if jsonl is None else f"jsonl:{jsonl.name}",
            id_col=id_col,
        )

    # ---- delta (skipped for --full, --jsonl, and --scratch: extract all) -----
    if jsonl is None and not full and not scratch:
        if source is None:  # _validate_source_args guarantees this; type guard
            raise RuntimeError("a --source is required without --jsonl")
        from oncai.fc_extraction.manifest import definition_hash_from_registry

        delta, counts, forced_missing = _compute_delta(
            config=config,
            batch_dir=batch_dir,
            source=source,
            id_col=id_col,
            note_id_set=note_id_set,
            where=where,
            definition_hash=definition_hash_from_registry(
                note_config.system_prompt, registry
            ),
            reextract_on_prompt_change=reextract_on_prompt_change,
            force_rerun=force_rerun,
        )
        console.print(
            f"[blue]Delta:[/blue] {counts['new']} new + {counts['changed']} changed "
            f"+ {counts['definition_changed']} definition-changed "
            f"+ {counts['forced']} forced; "
            f"skipping {counts['skipped']} already-extracted"
        )
        if forced_missing:
            preview = ", ".join(sorted(forced_missing)[:10])
            console.print(
                f"[yellow]Warning:[/yellow] {len(forced_missing)} force-rerun id(s) "
                f"not in the source scope (check --where/--id-file or {source}): "
                f"{preview}"
            )
        if not delta:
            console.print("[green]Nothing to extract — batch is up to date.[/green]")
            return
        note_id_set = delta

    # ---- choose the output target -------------------------------------------
    if scratch:
        # One-off: a fixed scratch name, started fresh each run (so a prompt /
        # definition edit isn't masked by resumed records), never promoted.
        run_label = f"{batch}.scratch"
        scratch_jsonl = output_dir / note_config.name / f"{run_label}.jsonl"
        scratch_jsonl.unlink(missing_ok=True)
        scratch_jsonl.with_name(scratch_jsonl.stem + "_manifest.json").unlink(
            missing_ok=True
        )
        console.print(
            "  [yellow]Scratch run[/yellow] — output stays in fc_outputs, "
            "not promoted to the inbox."
        )
    else:
        seg = _next_segment(batch_dir)
        run_label = f"{batch}.{seg:03d}"
        console.print(
            f"  Segment:    {seg:03d}  →  inbox/fc_extractions/{batch}/{seg:03d}.jsonl"
        )

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
    # Scratch runs leave no trace in the canonical run log.
    run_id = (
        None
        if scratch
        else _prelog_run(
            config=config,
            note_config=note_config,
            registry=registry,
            fc_client=fc_client,
            backend_name=backend,
            backend_enum=backend_enum,
            batch_name=run_label,
            started_at=started_at,
            source=source,
            jsonl=jsonl,
            text_col=text_col,
            workers=workers,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            mrn_source=_describe_mrn_source(
                id_file=id_file, cohort=cohort, limit=limit
            ),
        )
    )

    # ---- run the batch (working JSONL in fc_outputs scratch) -----------------
    from oncai.fc_extraction.batch_single import run_fc_single_batch

    try:
        result = run_fc_single_batch(
            registry=registry,
            config=note_config,
            client=fc_client,
            db_path=config.db_path if not jsonl else None,
            source_table=source,
            output_dir=output_dir,
            batch_name=run_label,
            text_col=text_col,
            id_col=id_col,
            limit=limit,
            where=where,
            note_ids=note_id_set,
            resume=(not no_resume) and not scratch,
            retry_failed=retry_failed,
            rate_limit=rate_limit,
            workers=workers,
            backend_name=backend,
            jsonl_path=jsonl,
        )
    except KeyboardInterrupt:
        _finalize_run(
            run_id=run_id, inbox_path=config.inbox_path, t0=t0, status="cancelled"
        )
        console.print("\n[yellow]Run cancelled by user.[/yellow]")
        raise typer.Exit(1) from None
    except Exception:
        _finalize_run(
            run_id=run_id, inbox_path=config.inbox_path, t0=t0, status="failed"
        )
        raise

    # ---- scratch: stop here, nothing touches the inbox -----------------------
    if scratch:
        console.print(f"\n[green]✓[/green] {result}")
        console.print(f"  Scratch output (not promoted): {result.output_path}")
        console.print(f"  Peek it:  oncai fc peek {result.output_path}")
        return

    # ---- promote the working JSONL into the inbox batch folder ---------------
    segment_path = _finalize_segment(Path(result.output_path), batch_dir, seg)

    _finalize_run(
        run_id=run_id,
        inbox_path=config.inbox_path,
        t0=t0,
        status="completed",
        input_count=result.total_notes,
        items_processed=result.total_notes,
        items_succeeded=result.successful,
        items_failed=result.failed,
        items_skipped=result.skipped,
        total_input_tokens=result.total_input_tokens,
        total_output_tokens=result.total_output_tokens,
        output_path=str(segment_path),
    )

    console.print(f"\n[green]✓[/green] {result}")
    console.print(f"  Segment: {segment_path}")

    # Build a physician-review package from the just-written segment so a
    # completed run is immediately review-ready (open it in the oncai review app).
    _build_review_package_for_run(
        note_config=note_config,
        registry=registry,
        output_path=result.output_path,
        inbox_path=config.inbox_path,
        batch=batch,
        source=source,
        jsonl=jsonl,
        db_path=config.db_path,
        text_col=text_col,
        id_col=id_col,
        review_package_scope=review_package,
    )


@fc_app.command(name="review-package")
def fc_review_package(
    batch: Path = typer.Argument(
        ...,
        help="Path to a completed batch JSONL "
        "(e.g. fc_outputs/PathKidneyBasic/v1.jsonl).",
    ),
    definition: str | None = typer.Option(
        None,
        "--definition",
        "-d",
        help="Definition name (see `oncai fc list`). Gives the package a richer "
        "field schema (enum options, required-ness). Auto-detected from the "
        "batch manifest when omitted.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Where to write the .review_pkg.json "
        "(default: <batch>.review_pkg.json beside the JSONL).",
    ),
    scope: str = typer.Option(
        "all",
        "--scope",
        help=(
            "Which reports to package: all, flagged, disagreements, or "
            "flagged-or-disagreements."
        ),
    ),
    compare_with: list[Path] | None = typer.Option(
        None,
        "--compare-with",
        help=(
            "Additional batch JSONL to compare against the primary batch. "
            "Use multiple times with --scope disagreements."
        ),
    ),
    agreement_ignore_field: list[str] | None = typer.Option(
        None,
        "--agreement-ignore-field",
        help=(
            "Field ignored when comparing runs. Defaults to comment. "
            "Use multiple times."
        ),
    ),
    db: Path | None = typer.Option(
        None,
        "--db",
        help="DuckDB path for loading source notes "
        "(default: from the batch manifest, then oncai.yaml).",
    ),
):
    """Build a physician-review package from an already-completed batch.

    Reads the batch JSONL and its ``_manifest.json`` sidecar, re-loads the
    source notes, and writes a ``*.review_pkg.json`` for the oncai review app.

    Examples:
        oncai fc review-package fc_outputs/PathKidneyBasic/v1.jsonl
        oncai fc review-package fc_outputs/PathKidneyBasic/v1.jsonl -d path_kidney_basic
        oncai fc review-package run_a.jsonl --scope disagreements --compare-with run_b.jsonl
    """
    from oncai.review import build_field_schema, package_from_batch
    from oncai.review.select import (
        DEFAULT_AGREEMENT_IGNORE_FIELDS,
        comparable_fields_from_field_schema,
        disagreement_note_ids,
        flagged_note_ids_from_jsonl,
    )

    if not batch.exists():
        _bail(f"Batch JSONL not found: {batch}")
    scope = _validate_choice(
        scope,
        option="--scope",
        choices=_REVIEW_SELECTION_SCOPES,
    )
    compare_paths = compare_with or []
    for path in compare_paths:
        if not path.exists():
            _bail(f"Comparison batch JSONL not found: {path}")
    if scope in {"disagreements", "flagged-or-disagreements"} and not compare_paths:
        _bail(f"--scope {scope} requires at least one --compare-with JSONL")

    config = get_config()
    registry = None
    definition_to_load = definition or _definition_name_from_manifest(batch)
    if definition is not None:
        _, registry = _load_definition(definition)
    elif (
        definition_to_load is not None
        and _resolve_definition_name(definition_to_load) is not None
    ):
        _, registry = _load_definition(definition_to_load)

    selected_note_ids: set[str] | None = None
    only_flagged = False
    if scope == "flagged":
        only_flagged = True
    elif scope in {"disagreements", "flagged-or-disagreements"}:
        ignored = agreement_ignore_field or list(DEFAULT_AGREEMENT_IGNORE_FIELDS)
        comparable_fields = (
            comparable_fields_from_field_schema(build_field_schema(registry))
            if registry is not None
            else None
        )
        selected_note_ids = disagreement_note_ids(
            batch,
            compare_paths,
            ignore_fields=ignored,
            comparable_fields=comparable_fields,
        )
        if scope == "flagged-or-disagreements":
            selected_note_ids |= flagged_note_ids_from_jsonl(batch)

    db_path = db or config.db_path
    try:
        pkg_path = package_from_batch(
            jsonl_path=batch,
            registry=registry,
            db_path=db_path if db_path.exists() else None,
            output_path=output,
            note_ids=selected_note_ids,
            only_flagged=only_flagged,
        )
    except FileNotFoundError as e:
        _bail(str(e))

    console.print(f"[green]✓[/green] Wrote review package: {pkg_path}")
    if scope != "all":
        selected_label = (
            "flagged reports"
            if scope == "flagged"
            else f"{len(selected_note_ids or set())} selected report(s)"
        )
        console.print(f"  Scope: {scope} ({selected_label})")
    console.print(
        "  Open it in the oncai review app "
        "(apps/review_app/server.py --package <file>)."
    )


@fc_app.command(name="unpeek")
def fc_unpeek(
    batch: str = typer.Argument(
        ...,
        help="Scratch table stem to drop (scratch.<batch>). "
        "Pass --all to drop the whole scratch schema.",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
    all: bool = typer.Option(False, "--all", help="Drop all scratch tables"),
):
    """
    Drop a table from the throwaway ``scratch`` schema (the ``fc peek`` target).

    Examples:
        oncai fc unpeek path_basic_v3
        oncai fc unpeek path_basic_v3 --yes
        oncai fc unpeek IGNORED --all          # drops every scratch table
    """
    import duckdb

    config = get_config()
    schema_name = "scratch"

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
        oncai fc manifest fc_outputs/PathKidneyBasic/v1_manifest.json
        oncai fc manifest fc_outputs/PathKidneyBasic/v1.jsonl
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


@fc_app.command(name="peek")
def fc_peek(
    jsonl_path: Path = typer.Argument(..., help="Path to a JSONL file to peek at"),
    skip_failed: bool = typer.Option(
        True, "--skip-failed/--include-failed", help="Skip failed extractions"
    ),
    replace: bool = typer.Option(
        False, "--replace", help="Replace existing scratch table instead of appending"
    ),
):
    """
    Load FC extraction results into a throwaway ``scratch`` DuckDB schema for a
    quick SQL look — handy for eyeballing a segment before you ingest it.

    Lands at ``scratch.<jsonl_stem>``. The scratch schema is disposable by
    design: a ``build-db`` clears it. Drop one peeked table with
    ``oncai fc unpeek <stem>``, or all of them with ``oncai fc unpeek --all``.

    Examples:
        oncai fc peek fc_outputs/PathKidneyBasic/path_basic_v3.jsonl
            → scratch.path_basic_v3
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

    schema_name = "scratch"
    table_name = jsonl_path.stem

    console.print("[blue]Peeking at FC extractions in DuckDB[/blue]")
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
        f"\n[green]✓[/green] Loaded {row_count} rows into {schema_name}.{table_name}"
    )
    console.print(f"  Query:   SELECT * FROM {schema_name}.{table_name} LIMIT 10")
