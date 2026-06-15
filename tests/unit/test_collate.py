"""Tests for oncai.transforms.collate — text cleaning and multi-line collation."""

from __future__ import annotations

import re
from datetime import date

import polars as pl

from oncai.transforms import collate as collate_mod
from oncai.transforms.collate import (
    PATHOLOGY_BOILERPLATE_PATTERNS,
    clean_pathology_text,
    clean_text,
    collate_pathology,
    normalize_date,
)

# ---------------------------------------------------------------------------
# normalize_date
# ---------------------------------------------------------------------------


class TestNormalizeDate:
    def test_valid_date_string(self):
        assert normalize_date("2024-01-15") == date(2024, 1, 15)

    def test_none_and_empty(self):
        assert normalize_date(None) is None
        assert normalize_date("") is None
        assert normalize_date("   ") is None

    def test_already_date_object(self):
        d = date(2024, 6, 1)
        assert normalize_date(d) is d


# ---------------------------------------------------------------------------
# clean_text
# ---------------------------------------------------------------------------


class TestCleanText:
    def test_unicode_normalization(self):
        # Smart quotes → regular quotes, em-dash → hyphen
        text = "\u201cHello\u201d \u2014 world"
        result = clean_text(text)
        assert '"Hello"' in result
        assert "-" in result
        assert "\u2014" not in result

    def test_line_endings(self):
        text = "line1\r\nline2\rline3"
        result = clean_text(text)
        assert "\r" not in result
        # trailing whitespace stripped per line
        text2 = "hello   \nworld   "
        result2 = clean_text(text2)
        assert result2 == "hello\nworld"

    def test_bom_and_zero_width(self):
        text = "\ufeffHello\u200bWorld"
        result = clean_text(text)
        assert result == "HelloWorld"

    def test_none_returns_empty(self):
        assert clean_text(None) == ""


# ---------------------------------------------------------------------------
# clean_pathology_text
# ---------------------------------------------------------------------------


class TestCleanPathologyText:
    def test_no_default_boilerplate_removal(self):
        # Ships with no site-specific patterns, so report content is preserved
        # verbatim — only generic cleaning applies.
        assert PATHOLOGY_BOILERPLATE_PATTERNS == []
        text = (
            "DIAGNOSIS: Clear cell RCC. Stain quality is acceptable. Margins negative."
        )
        result = clean_pathology_text(text)
        assert "Clear cell RCC" in result
        assert "Stain quality is acceptable" in result
        assert "Margins negative" in result

    def test_boilerplate_hook_strips_registered_patterns(self, monkeypatch):
        # Users opt in by registering their own site patterns on the hook;
        # clean_pathology_text reads the module global at call time.
        monkeypatch.setattr(
            collate_mod,
            "PATHOLOGY_BOILERPLATE_PATTERNS",
            [re.compile(r"Stain quality is acceptable\.")],
        )
        text = (
            "DIAGNOSIS: Clear cell RCC. Stain quality is acceptable. Margins negative."
        )
        result = collate_mod.clean_pathology_text(text)
        assert "Stain quality is acceptable" not in result
        assert "Clear cell RCC" in result
        assert "Margins negative" in result

    def test_double_space_linebreak_decoding_off_by_default(self):
        # Off by default: double spaces are left alone, not turned into newlines.
        assert collate_mod.DECODE_DOUBLE_SPACE_LINEBREAKS is False
        result = clean_pathology_text("Line one.  Line two.")
        assert "\n" not in result
        assert "Line one." in result
        assert "Line two." in result

    def test_double_space_linebreak_decoding_when_enabled(self, monkeypatch):
        # Opt-in for sources that encode line breaks as double spaces.
        monkeypatch.setattr(collate_mod, "DECODE_DOUBLE_SPACE_LINEBREAKS", True)
        result = collate_mod.clean_pathology_text("Line one.  Line two.")
        assert "\n" in result


# ---------------------------------------------------------------------------
# collate_pathology
# ---------------------------------------------------------------------------


class TestCollatePathology:
    def test_groups_by_report_id(self, sample_pathology_csv):
        df = pl.scan_csv(sample_pathology_csv)
        result = collate_pathology(df).collect()

        # 3 distinct report_ids → 3 rows
        assert result.height == 3
        assert set(result["report_id"].to_list()) == {
            "CO19-001",
            "CO19-002",
            "CO19-003",
        }

        # Hashes are present and non-empty
        assert "key_hash" in result.columns
        assert "content_hash" in result.columns
        assert all(h is not None and len(h) > 0 for h in result["key_hash"].to_list())

        # Report text is the concatenation of lines (no delimiter)
        row1 = result.filter(pl.col("report_id") == "CO19-001")
        text = row1["report_text"][0]
        assert "Clear cell renal cell carcinoma" in text

    def test_skips_excluded_external_names(self):
        data = [
            {
                "report_id": "R1",
                "row_id": 1,
                "mrn": "M1",
                "mult_ln_val_storage": "Content line",
                "external_name": "results",
                "ordering_date": "2024-01-01",
            },
            {
                "report_id": "R1",
                "row_id": 2,
                "mrn": "M1",
                "mult_ln_val_storage": "Gross description",
                "external_name": "gross",
                "ordering_date": "2024-01-01",
            },
            {
                "report_id": "R1",
                "row_id": 3,
                "mrn": "M1",
                "mult_ln_val_storage": "Micro description",
                "external_name": "$micro",
                "ordering_date": "2024-01-01",
            },
        ]
        df = pl.LazyFrame(data)
        result = collate_pathology(df).collect()
        assert result.height == 1
        text = result["report_text"][0]
        assert "Content line" in text
        assert "Gross description" not in text
        assert "Micro description" not in text
