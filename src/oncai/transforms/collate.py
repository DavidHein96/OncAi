"""Multi-line record collation transforms."""

from __future__ import annotations

import re
import unicodedata
from datetime import date

import polars as pl
from dateutil import parser as dateutil_parser

from oncai.hashing import compute_content_hash, compute_key_hash


def normalize_date(value: str | date | None) -> date | None:
    """
    Normalize a date value to a Python date object.

    Handles various input formats using dateutil.parser.
    Returns None for empty/invalid values.
    """
    if value is None:
        return None

    # Already a date object
    if isinstance(value, date):
        return value

    # Empty string
    if not str(value).strip():
        return None

    try:
        dt = dateutil_parser.parse(str(value))
        return dt.date()
    except (ValueError, TypeError):
        return None


# Regex patterns to remove boilerplate from pathology reports
PATHOLOGY_BOILERPLATE_PATTERNS = [
    re.compile(
        r"Stain quality is acceptable\. The microscopic findings are reflected in the diagnosis rendered\."
    ),
    re.compile(
        r"The patient's name, specimen container\(s\) and cassettes all match\."
    ),
    re.compile(
        r"I,.*the senior physician, attest that I: \(i\) attended the biopsy procedure; \(ii\) immediately examined smears while the procedure was underway; and \(iii\) determined or confirmed the adequacy of the specimen\(s\)\."
    ),
    re.compile(
        r"I provided appropriate supervision to.*As the senior physician, I attest.* that.*I: \(i\) examined the relevant preparation\(s\) for the specimen\(s\); and \(ii\) rendered or confirmed the diagnosis\(es\)\."
    ),
    re.compile(
        r"The case was reviewed and discussed at.*conference on.* and the above interpretation represents a consensus opinion"
    ),
    re.compile(r"This case was reviewed with.* who concurs with the above impression"),
    re.compile(
        r"The case was reviewed in the.* The final diagnosis represents the opinion of the group\."
    ),
    re.compile(
        r"The case was reviewed and discussed at genitourinary pathology consensus conference on .*\."
    ),
    re.compile(
        r"Biopsy performed by.* at William P\. Clements Jr\. University Hospital on \d+/\d+/\d\d+"
    ),
    re.compile(
        r"Biopsy performed by.* at Clements Jr\. University Hospital on \d+/\d+/\d\d+"
    ),
    re.compile(r"Dr\. .*reviewed and concurred\."),
    re.compile(
        r"I,.* examined the preparations and rendered the diagnosis. I provided supervision to .* Reported findings to"
    ),
    re.compile(r"Time to fixation: <1 hour"),
    re.compile(r"Duration of fixation in formalin: Between 6 and 72 hours"),
    re.compile(
        r"Adequacy reported to Dr.*on \d+/\d+/\d\d+ at \d+:?\d\d[AaPp ]?[AaPpmM]?[mM]?"
    ),
    re.compile(r"Immediate on-site adequacy assessment was performed by:[ \w\.]*"),
    re.compile(r"\w+ \w+, MS, PA\(ASCP\)"),
    re.compile(r"PA\(ASCP\)"),
    re.compile(r"\d\d\d-\d\d\d-\d\d\d\d"),
    re.compile(r"\.--"),
    re.compile(
        r"Selective slides for grading are reviewed in consultation with.*who concurs\."
    ),
    re.compile(r"BAP1 loss \(BRCA1 associated protein-1\).*24382589"),
    re.compile(r"The microscopic findings are reflected in the diagnosis rendered"),
    re.compile(r"Received at the request of .* received for outside consultation"),
    re.compile(r"serially sectioned, entirely submitted"),
    re.compile(r"serially sectioned and entirely submitted"),
    re.compile(r"The case is reviewed at.*conference on \d+/\d+/\d\d+\."),
    re.compile(r"This case was reviewed with.*\."),
    re.compile(r"FDA Disclaimer:.*CAP eCC January 2018 Annual Release"),
]

# External names to skip (these are section headers, not content)
SKIP_EXTERNAL_NAMES = {
    "gross",
    "$gross",
    "micro",
    "$micro",
    "addlinfo",
    "$addlinfo",
    "$clindx",
}


def clean_text(text: str | None) -> str:
    """
    Basic text cleaning for encoding resilience and normalization.

    Ensures the same logical text produces the same hash regardless of:
    - Unicode normalization form (NFC/NFD/NFKC/NFKD)
    - Line ending style (CR/LF/CRLF)
    - BOM or zero-width characters
    - Whitespace variants (non-breaking space, em-space, etc.)
    - Trailing whitespace
    """
    if text is None:
        return ""

    # Remove BOM and zero-width characters
    text = text.lstrip("\ufeff")
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)

    # Normalize line endings to \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Unicode normalization (NFKC is most aggressive - handles compatibility chars)
    text = unicodedata.normalize("NFKC", text)

    # Normalize ALL Unicode whitespace variants to regular space
    # Includes: non-breaking space, en/em space, thin space, hair space, etc.
    text = re.sub(
        r"[\u00a0\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000]",
        " ",
        text,
    )

    # Replace bullets with hyphens
    text = text.replace("\u2022", "-")

    # Replace en-dashes and em-dashes with hyphens
    text = text.replace("\u2013", "-")
    text = text.replace("\u2014", "-")

    # Replace smart quotes with regular quotes
    text = text.replace("\u2018", "'")  # left single quote
    text = text.replace("\u2019", "'")  # right single quote
    text = text.replace("\u201c", '"')  # left double quote
    text = text.replace("\u201d", '"')  # right double quote

    # Collapse excessive whitespace (3+ spaces -> 2 spaces)
    text = re.sub(r" {3,}", "  ", text)

    # Collapse excessive dashes
    text = re.sub(r"-{10,}", "-------", text)

    # Strip trailing whitespace from each line
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    return text.strip()


def clean_pathology_text(text: str) -> str:
    """Clean pathology report text with boilerplate removal."""
    # First apply basic cleaning
    text = clean_text(text)

    # Remove boilerplate patterns
    for pattern in PATHOLOGY_BOILERPLATE_PATTERNS:
        text = pattern.sub("", text)

    # Convert double spaces to newlines (this is how line breaks are encoded)
    text = re.sub(r"  ", "\n", text)

    # Clean up multiple newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def collate_pathology(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Collate multi-line pathology reports into single rows.

    Rules:
    1. Sort by report_id and row_id only (row_id is required)
    2. Skip rows where external_name is "gross", "micro", or "addlinfo"
    3. Join lines with NO delimiter (lines may be cut mid-word)
    4. After collation, double spaces "  " become newlines
    5. Apply boilerplate removal patterns
    """
    # Collect to process row by row (needed for proper filtering and ordering)
    pdf: pl.DataFrame = df.collect()  # type: ignore[assignment]

    # Verify row_id exists
    if "row_id" not in pdf.columns:
        raise ValueError("row_id column is required for pathology collation")

    # Sort by report_id and row_id only
    pdf = pdf.sort(["report_id", "row_id"])

    # Group and collate manually to handle filtering and no-delimiter join
    reports: dict[str, dict] = {}

    for row in pdf.iter_rows(named=True):
        report_id = row["report_id"]
        external_name = (row.get("external_name") or "").lower().strip()
        text = row.get("mult_ln_val_storage") or ""

        # Skip rows with excluded external_name values
        if external_name in SKIP_EXTERNAL_NAMES:
            continue

        # Skip empty text
        if not text.strip():
            continue

        if report_id not in reports:
            reports[report_id] = {
                "mrn": row["mrn"],
                "ordering_date": row.get("ordering_date"),
                "external_name": row.get("external_name"),
                "text_parts": [],
            }

        # Append text with NO delimiter - lines may be cut mid-word
        reports[report_id]["text_parts"].append(text)

    # Build result dataframe
    result_rows = []
    for report_id, data in reports.items():
        # Join with no delimiter, then clean
        raw_text = "".join(data["text_parts"])
        report_text = clean_pathology_text(raw_text)

        # Normalize ordering_date to YYYY-MM-DD format
        ordering_date = normalize_date(data["ordering_date"])

        result_rows.append(
            {
                "report_id": report_id,
                "mrn": data["mrn"],
                "ordering_date": ordering_date,
                "external_name": data["external_name"],
                "report_text": report_text,
            }
        )

    if not result_rows:
        # Return empty dataframe with correct schema
        return pl.DataFrame(
            {
                "report_id": [],
                "mrn": [],
                "ordering_date": [],
                "external_name": [],
                "report_text": [],
                "key_hash": [],
                "content_hash": [],
            }
        ).lazy()

    result = pl.DataFrame(result_rows)

    # Compute hashes
    key_hashes = []
    content_hashes = []

    for row in result.iter_rows(named=True):
        key_hashes.append(compute_key_hash((row["report_id"],)))
        content_hashes.append(compute_content_hash((row["report_text"],)))

    result = result.with_columns(
        pl.Series("key_hash", key_hashes),
        pl.Series("content_hash", content_hashes),
    )

    return result.lazy()
