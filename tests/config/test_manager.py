"""Tests for ConfigManager — config persistence, defaults, getters, validation."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from voca.config.manager import ConfigManager


@pytest.fixture
def config_dir(tmp_path):
    """Provide a temp dir and patch ConfigManager to use it."""
    d = tmp_path / "voca_test_config"
    d.mkdir()
    return d


@pytest.fixture
def make_manager(config_dir):
    """Create a ConfigManager that uses the temp directory."""
    def _make(initial_config=None):
        if initial_config:
            config_file = config_dir / "config.json"
            config_file.write_text(json.dumps(initial_config), encoding="utf-8")
        with patch.object(ConfigManager, '_get_config_dir', return_value=config_dir):
            return ConfigManager()
    return _make


class TestDefaults:
    def test_first_run_creates_config_file(self, make_manager, config_dir):
        cm = make_manager()
        config_file = config_dir / "config.json"
        assert config_file.exists()

    def test_default_transcription_mode(self, make_manager):
        cm = make_manager()
        assert cm.get_transcription_mode() == "qwen"

    def test_default_audio_settings(self, make_manager):
        cm = make_manager()
        assert cm.get_audio_sample_rate() == 16000
        assert cm.get_audio_channels() == 1

    def test_default_api_key_is_none(self, make_manager):
        cm = make_manager()
        assert cm.get_api_key() is None

    def test_default_insertion_method(self, make_manager):
        cm = make_manager()
        assert cm.get_text_insertion_method() == "auto"

    def test_first_run_flag(self, make_manager):
        cm = make_manager()
        assert cm.is_first_run() is True

    def test_default_command_mode_disabled(self, make_manager):
        cm = make_manager()
        assert cm.is_command_mode_enabled() is False


class TestPersistence:
    def test_set_setting_persists_to_disk(self, make_manager, config_dir):
        cm = make_manager()
        cm.set_setting("transcription_mode", "local")

        raw = json.loads((config_dir / "config.json").read_text())
        assert raw["transcription_mode"] == "local"

    def test_set_setting_survives_reload(self, make_manager, config_dir):
        cm = make_manager()
        cm.set_setting("transcription_mode", "api")
        cm.set_setting("api_key", "sk-test-key")

        # Re-create manager from same dir — simulates restart
        cm2 = make_manager()
        assert cm2.get_transcription_mode() == "api"
        assert cm2.get_api_key() == "sk-test-key"

    def test_set_unknown_key_returns_false(self, make_manager):
        cm = make_manager()
        assert cm.set_setting("totally_fake_key", "value") is False

    def test_set_known_key_returns_true(self, make_manager):
        cm = make_manager()
        assert cm.set_setting("transcription_mode", "local") is True

    def test_reload_picks_up_external_changes(self, make_manager, config_dir):
        cm = make_manager()
        # Simulate an external edit to the config file
        config_file = config_dir / "config.json"
        raw = json.loads(config_file.read_text())
        raw["transcription_mode"] = "api"
        raw["api_key"] = "external-edit-key"
        config_file.write_text(json.dumps(raw))

        cm._load_config()
        assert cm.get_transcription_mode() == "api"
        assert cm.get_api_key() == "external-edit-key"

    def test_corrupted_config_falls_back_to_defaults(self, make_manager, config_dir):
        config_file = config_dir / "config.json"
        config_file.write_text("NOT VALID JSON {{{{")

        cm = make_manager()
        # Should not crash, and should have defaults
        assert cm.get_transcription_mode() == "qwen"

    def test_config_file_permissions_set(self, make_manager, config_dir):
        """On Linux, config file should have 0o600 permissions."""
        import platform
        if platform.system() not in ("Linux", "Darwin"):
            pytest.skip("Permission check only on Unix")
        cm = make_manager()
        config_file = config_dir / "config.json"
        mode = config_file.stat().st_mode & 0o777
        assert mode == 0o600


class TestGetters:
    def test_get_qwen_config_returns_dict(self, make_manager):
        cm = make_manager()
        qc = cm.get_qwen_config()
        assert isinstance(qc, dict)
        assert "model_size" in qc
        assert "backend" in qc
        assert "streaming_enabled" in qc
        assert "kv_cache_mb" in qc

    def test_get_qwen_config_reflects_set_values(self, make_manager):
        cm = make_manager()
        cm.set_setting("qwen_model_size", "1.7B")
        cm.set_setting("qwen_backend", "gpu")
        qc = cm.get_qwen_config()
        assert qc["model_size"] == "1.7B"
        assert qc["backend"] == "gpu"

    def test_get_command_mode_config(self, make_manager):
        cm = make_manager()
        cmd = cm.get_command_mode_config()
        assert "enabled" in cmd
        assert "triggers" in cmd
        assert isinstance(cmd["triggers"], list)

    def test_get_command_triggers_default(self, make_manager):
        cm = make_manager()
        triggers = cm.get_command_triggers()
        assert "voca" in triggers

    def test_get_setting_with_default(self, make_manager):
        cm = make_manager()
        assert cm.get_setting("nonexistent_key", 42) == 42

    def test_get_setting_returns_stored_value(self, make_manager):
        cm = make_manager()
        assert cm.get_setting("audio_sample_rate") == 16000

    def test_get_platform_returns_string(self, make_manager):
        cm = make_manager()
        assert cm.get_platform() in ("linux", "macos", "windows", "unknown")


class TestValidation:
    def test_valid_default_config(self, make_manager):
        cm = make_manager()
        errors = cm.validate_config()
        assert errors == {}

    def test_invalid_transcription_mode(self, make_manager):
        cm = make_manager({"transcription_mode": "invalid"})
        errors = cm.validate_config()
        assert "transcription_mode" in errors

    def test_api_mode_requires_key(self, make_manager):
        cm = make_manager({"transcription_mode": "api", "api_key": ""})
        errors = cm.validate_config()
        assert "api_key" in errors

    def test_api_mode_with_key_is_valid(self, make_manager):
        cm = make_manager({"transcription_mode": "api", "api_key": "sk-test"})
        errors = cm.validate_config()
        assert "api_key" not in errors

    def test_invalid_qwen_model_size(self, make_manager):
        cm = make_manager({"transcription_mode": "qwen", "qwen_model_size": "99B"})
        errors = cm.validate_config()
        assert "qwen_model_size" in errors

    def test_invalid_sample_rate(self, make_manager):
        cm = make_manager({"audio_sample_rate": 100})
        errors = cm.validate_config()
        assert "audio_sample_rate" in errors


class TestExportImport:
    def test_export_creates_file(self, make_manager, tmp_path):
        cm = make_manager()
        export_path = str(tmp_path / "exported.json")
        assert cm.export_config(export_path) is True
        assert Path(export_path).exists()

    def test_export_import_roundtrip(self, make_manager, tmp_path):
        cm = make_manager()
        cm.set_setting("transcription_mode", "local")
        cm.set_setting("audio_sample_rate", 44100)

        export_path = str(tmp_path / "exported.json")
        cm.export_config(export_path)

        # Create fresh manager and import
        cm2 = make_manager()
        assert cm2.import_config(export_path) is True
        assert cm2.get_transcription_mode() == "local"
        assert cm2.get_audio_sample_rate() == 44100

    def test_import_nonexistent_file_returns_false(self, make_manager):
        cm = make_manager()
        assert cm.import_config("/nonexistent/path.json") is False

    def test_reset_to_defaults(self, make_manager):
        cm = make_manager()
        cm.set_setting("transcription_mode", "local")
        cm.set_setting("audio_sample_rate", 44100)
        cm.reset_to_defaults()
        assert cm.get_transcription_mode() == "qwen"
        assert cm.get_audio_sample_rate() == 16000
