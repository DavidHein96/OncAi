"""Build review-ready bundles from completed FC extraction batches.

A *review package* (``*.review_pkg.json``) is a single self-contained file the
local physician-review app (``review_app_reference/``) opens to adjudicate
extracted events — approve / reject / edit each finding against the source note.
This package turns a finished batch (its JSONL + manifest + the source notes)
into that file.

Typical use is automatic — the end of ``oncai fc run-single`` calls
:func:`package_from_jsonl`. To (re)build one for an already-completed batch:

    from oncai.review import package_from_batch
    package_from_batch(jsonl_path=Path("fc_outputs/PathKidneyBasic/v1.jsonl"))

See :mod:`oncai.review.package` for the file format and the producer contract
with the review server.
"""

from __future__ import annotations

from .package import (
    build_review_package,
    default_package_path,
    package_from_batch,
    package_from_jsonl,
    write_review_package,
)
from .schema import build_field_schema, infer_field_schema_from_events

__all__ = [
    "build_field_schema",
    "infer_field_schema_from_events",
    "build_review_package",
    "write_review_package",
    "default_package_path",
    "package_from_jsonl",
    "package_from_batch",
]
