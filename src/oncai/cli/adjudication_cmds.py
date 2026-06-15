"""Adjudication CLI commands."""

from __future__ import annotations

from pathlib import Path

import typer

from oncai.adjudication import package_from_jsonls
from oncai.review.schema import build_field_schema

from ._shared import console, get_config
from .fc_cmds import _SAFE_BATCH_NAME, _bail, _load_definition

adjudication_app = typer.Typer(help="Create and manage cross-batch adjudication")


@adjudication_app.command("create")
def adjudication_create(
    round_name: str = typer.Argument(
        ...,
        help="Adjudication round name, e.g. ihc_compare_v1",
    ),
    left: Path = typer.Option(
        ...,
        "--left",
        help="Left extraction batch JSONL to compare",
    ),
    right: Path = typer.Option(
        ...,
        "--right",
        help="Right extraction batch JSONL to compare",
    ),
    definition: str = typer.Option(
        ...,
        "--definition",
        "-d",
        help="Definition name (see `oncai fc list`) for schema/comparability",
    ),
    left_label: str = typer.Option(
        "left",
        "--left-label",
        help="Display label for the left batch",
    ),
    right_label: str = typer.Option(
        "right",
        "--right-label",
        help="Display label for the right batch",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory (default: inbox/fc_adjudications/<round>)",
    ),
    db: Path | None = typer.Option(
        None,
        "--db",
        help="DuckDB path for loading source notes (default: manifest, then oncai.yaml)",
    ),
) -> None:
    """Create a two-batch adjudication package from extraction JSONLs."""
    if not _SAFE_BATCH_NAME.match(round_name):
        _bail(f"round name must be alphanumeric+underscore (got {round_name!r})")
    if left_label == right_label:
        _bail("--left-label and --right-label must differ")
    for label, path in (("left", left), ("right", right)):
        if not path.exists():
            _bail(f"{label} batch JSONL not found: {path}")

    config = get_config()
    note_config, registry = _load_definition(definition)
    db_path = db or config.db_path

    try:
        package_path, descriptor_path, package = package_from_jsonls(
            config=config,
            round_name=round_name,
            definition_name=note_config.name,
            left_jsonl=left,
            right_jsonl=right,
            field_schema=build_field_schema(registry),
            left_label=left_label,
            right_label=right_label,
            db_path=db_path if db_path.exists() else None,
            output_dir=output_dir,
        )
    except ValueError as exc:
        _bail(str(exc))

    console.print(f"[green]✓[/green] Wrote adjudication package: {package_path}")
    console.print(f"  Descriptor: {descriptor_path}")
    console.print(f"  Disagreements: {package['summary']['disagreements']}")
    console.print(
        "  Open it in the review app once adjudication-package support is wired."
    )
