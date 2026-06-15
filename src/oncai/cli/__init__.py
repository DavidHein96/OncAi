"""OncAI Command Line Interface."""

from __future__ import annotations

import typer

from .adjudication_cmds import adjudication_app
from .cohort_cmds import cohort_app
from .db_cmds import db_app
from .fc_cmds import fc_app
from .main_cmds import register_main_commands
from .runs_cmds import runs_app

app = typer.Typer(
    name="oncai",
    help="Oncology data lake + single-note function-calling extraction CLI",
    no_args_is_help=True,
)

# Register sub-apps
app.add_typer(fc_app, name="fc")
app.add_typer(cohort_app, name="cohort")
app.add_typer(runs_app, name="runs")
app.add_typer(db_app, name="db")
app.add_typer(adjudication_app, name="adjudication")

# Register main commands (init, pull, push, ingest, build-db, status, schemas, version)
register_main_commands(app)


def main():
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
