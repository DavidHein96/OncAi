"""Helpers for normalizing LLM-provided enum-like tool arguments."""

from __future__ import annotations

import logging
import re
from enum import StrEnum


def normalize_key(s: str) -> str:
    """Reduce a string to a canonical comparison key."""
    return re.sub(r"[\s\-_./()]+", "", s).lower()


def build_enum_lookup(enum_cls: type[StrEnum]) -> dict[str, str]:
    """Build a normalized-key lookup for all values in a StrEnum."""
    return {normalize_key(member.value): member.value for member in enum_cls}


def build_literal_lookup(values: tuple[str, ...]) -> dict[str, str]:
    """Build a normalized-key lookup for a fixed set of string values."""
    return {normalize_key(value): value for value in values}


def normalize_against(
    value: object,
    lookup: dict[str, str],
    field_name: str,
    *,
    logger: logging.Logger | None = None,
) -> str:
    """Match a value against a normalized lookup, logging any correction."""
    if isinstance(value, StrEnum):
        return value.value
    raw = str(value).strip()
    match = lookup.get(normalize_key(raw))
    if logger is not None and match is not None and match != raw:
        logger.debug("normalizer fixed %s: %r -> %r", field_name, raw, match)
    return match if match is not None else raw
