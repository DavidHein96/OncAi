"""Build review-ready bundles from completed FC extraction batches.

A *review package* (``*.review_pkg.json``) is a single self-contained file the
local physician-review app (``apps/review_app/``) opens to adjudicate
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

from .load import (
    ReviewLoadResult,
    review_batch_name,
    review_to_silver_df,
    review_to_silver_parquet,
)
from .package import (
    build_review_package,
    default_package_path,
    package_from_batch,
    package_from_jsonl,
    write_review_package,
)
from .schema import (
    adjudication_hash,
    adjudication_hash_from_registry,
    build_field_schema,
    infer_field_schema_from_events,
)
from .select import (
    NORMALIZATION_VERSION,
    disagreement_note_ids,
    flagged_note_ids,
    flagged_note_ids_from_jsonl,
)
from .slots import (
    Slot,
    SlotState,
    event_content_hash,
    events_from_record,
    record_to_slots,
)

__all__ = [
    "NORMALIZATION_VERSION",
    "adjudication_hash",
    "adjudication_hash_from_registry",
    "build_field_schema",
    "infer_field_schema_from_events",
    "ReviewLoadResult",
    "build_review_package",
    "write_review_package",
    "default_package_path",
    "package_from_jsonl",
    "package_from_batch",
    "review_batch_name",
    "review_to_silver_df",
    "review_to_silver_parquet",
    "flagged_note_ids",
    "flagged_note_ids_from_jsonl",
    "disagreement_note_ids",
    "Slot",
    "SlotState",
    "events_from_record",
    "event_content_hash",
    "record_to_slots",
]
