"""Tests for oncai.config module."""

from __future__ import annotations

from pathlib import Path

from oncai.config import OncaiConfig, get_dataset_folders, load_config, save_config


class TestOncaiConfig:
    def test_defaults(self):
        config = OncaiConfig()
        assert config.lake_path == Path("oncai_data/lake")
        assert config.inbox_path == Path("oncai_data/inbox")
        assert config.remote_path == Path("oncai_data/remote")
        assert config.db_path == Path("oncai_data/oncai.duckdb")

    def test_custom_paths(self):
        config = OncaiConfig(
            lake_path=Path("/custom/lake"),
            inbox_path=Path("/custom/inbox"),
        )
        assert config.lake_path == Path("/custom/lake")
        assert config.inbox_path == Path("/custom/inbox")


class TestSaveLoadConfig:
    def test_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = OncaiConfig(
            remote_path=Path("my_remote"),
            lake_path=Path("my_lake"),
            inbox_path=Path("my_inbox"),
        )
        save_config(config)

        loaded = load_config()
        assert loaded.remote_path == Path("my_remote")
        assert loaded.lake_path == Path("my_lake")
        assert loaded.inbox_path == Path("my_inbox")

    def test_load_missing_config_returns_default(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = load_config()
        assert config == OncaiConfig()


class TestGetDatasetFolders:
    def test_returns_list(self):
        folders = get_dataset_folders()
        assert isinstance(folders, list)
        assert len(folders) > 0

    def test_contains_pathology(self):
        folders = get_dataset_folders()
        assert "pathology" in folders

    def test_contains_cohorts(self):
        folders = get_dataset_folders()
        assert "cohorts" in folders

    def test_contains_fc_extractions(self):
        folders = get_dataset_folders()
        assert "fc_extractions" in folders

    def test_contains_fc_reviews(self):
        folders = get_dataset_folders()
        assert "fc_reviews" in folders

    def test_contains_fc_adjudications(self):
        folders = get_dataset_folders()
        assert "fc_adjudications" in folders

    def test_contains_tombstones(self):
        folders = get_dataset_folders()
        assert "tombstones" in folders
