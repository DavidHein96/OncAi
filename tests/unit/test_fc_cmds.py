"""Tests for the fc CLI definition-name resolution."""

from __future__ import annotations

import pytest

from oncai.cli.fc_cmds import _DEFINITIONS, _resolve_definition_name


def test_snake_case_key_resolves_to_itself() -> None:
    for key in _DEFINITIONS:
        assert _resolve_definition_name(key) == key


def test_pascal_case_definition_name_resolves_to_snake_key() -> None:
    # Each definition's PascalCase DEFINITION_NAME should resolve to its
    # registered snake_case key.
    assert _resolve_definition_name("PathKidneyBasic") == "path_kidney_basic"
    assert _resolve_definition_name("PathKidneyIhc") == "path_kidney_ihc"
    assert _resolve_definition_name("Example") == "example"


@pytest.mark.parametrize("name", ["bogus", "pathkidneybasic", "Path_Kidney_Basic", ""])
def test_unknown_name_returns_none(name: str) -> None:
    assert _resolve_definition_name(name) is None
