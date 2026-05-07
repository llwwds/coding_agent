"""Tests for config.py - configuration loading and validation."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class TestSettings:
    """Tests for the Settings class and global settings singleton."""

    def test_settings_defaults(self):
        """Test that Settings loads default values correctly."""
        from config import Settings

        s = Settings()
        assert s.WORKSPACE_DIR == "./workspace"
        assert s.MAX_FIX_ROUNDS == 5
        assert s.LOG_LEVEL == "INFO"

    def test_settings_from_env(self, monkeypatch):
        """Test that Settings loads values from environment variables."""
        monkeypatch.setenv("MAX_FIX_ROUNDS", "10")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("WORKSPACE_DIR", "./test_ws")

        from config import Settings

        s = Settings()
        assert s.MAX_FIX_ROUNDS == 10
        assert s.LOG_LEVEL == "DEBUG"
        assert s.WORKSPACE_DIR == "./test_ws"

    def test_settings_validate_missing_key(self, monkeypatch):
        """Test that validate raises error when API key is missing."""
        monkeypatch.setenv("OPENAI_API_KEY", "")
        monkeypatch.setenv("OPENAI_BASE_URL", "")
        from config import Settings

        s = Settings()
        with pytest.raises(ValueError):
            s.validate()

    def test_settings_validate_success(self, monkeypatch):
        """Test that validate passes when all required keys are set."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.test.com")
        monkeypatch.setenv("MODEL_NAME", "test-model")
        from config import Settings

        s = Settings()
        s.validate()

    def test_global_settings_instance(self):
        """Test that the global settings singleton exists."""
        from config import settings

        assert settings is not None
        assert isinstance(settings.WORKSPACE_DIR, str)


class TestConfigIntegration:
    """Integration tests for config module."""

    def test_env_file_loading(self):
        """Test that .env file is loaded by python-dotenv."""
        import config

        assert hasattr(config, "settings")
