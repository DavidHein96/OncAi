"""
Single-note extraction definitions.

Each definition module provides:
- Pydantic models for the extraction tools
- A ``create_<name>_registry()`` factory returning a configured ToolRegistry
- ``DEFINITION_NAME`` and ``SYSTEM_PROMPT`` module-level constants

See ``example.py`` for a documented template. The shipped ``path_kidney_*``
modules are concrete examples of the same pattern for kidney pathology reports.

Import from each submodule directly, e.g.:

    from oncai.fc_extraction.definitions.path_kidney_basic import (
        DEFINITION_NAME,
        SYSTEM_PROMPT,
        create_kidney_path_basic_registry,
    )
"""
