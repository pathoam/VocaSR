"""Tests for the ASR backend registry."""

import pytest
from unittest.mock import MagicMock

from voca.transcription.registry import (
    register_asr,
    create_asr_service,
    available_modes,
    _registry,
)
from voca.transcription.base import TranscriptionService


class _DummyService(TranscriptionService):
    """Concrete service for testing the registry dispatch."""

    def __init__(self, config_manager, vocab_context):
        self.config_manager = config_manager
        self.vocab_context = vocab_context

    def transcribe(self, audio_file_path: str) -> str:
        return "dummy"

    def is_available(self) -> bool:
        return True


# Keep a snapshot of real registrations so we can restore after tests
_original_registry = None


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Snapshot and restore the global registry around each test."""
    global _original_registry
    _original_registry = dict(_registry)
    yield
    _registry.clear()
    _registry.update(_original_registry)


class TestRegisterASR:
    def test_decorator_registers_factory(self):
        @register_asr("test_mode")
        def factory(config_manager, vocab_context):
            return _DummyService(config_manager, vocab_context)

        assert "test_mode" in _registry
        assert _registry["test_mode"] is factory

    def test_decorator_returns_original_function(self):
        def factory(config_manager, vocab_context):
            return _DummyService(config_manager, vocab_context)

        result = register_asr("test_passthrough")(factory)
        assert result is factory

    def test_duplicate_registration_overwrites(self):
        @register_asr("dup_mode")
        def first(cm, vc):
            return _DummyService(cm, vc)

        @register_asr("dup_mode")
        def second(cm, vc):
            return _DummyService(cm, vc)

        assert _registry["dup_mode"] is second


class TestCreateASRService:
    def test_dispatches_to_registered_factory(self):
        @register_asr("dispatch_test")
        def factory(config_manager, vocab_context):
            return _DummyService(config_manager, vocab_context)

        cm = MagicMock()
        service = create_asr_service("dispatch_test", cm, "context string")
        assert isinstance(service, _DummyService)
        assert service.config_manager is cm
        assert service.vocab_context == "context string"

    def test_raises_key_error_for_unknown_mode(self):
        with pytest.raises(KeyError):
            create_asr_service("nonexistent_mode_xyz", MagicMock())

    def test_passes_none_context_by_default(self):
        @register_asr("ctx_default")
        def factory(config_manager, vocab_context):
            return _DummyService(config_manager, vocab_context)

        service = create_asr_service("ctx_default", MagicMock())
        assert service.vocab_context is None


class TestAvailableModes:
    def test_includes_builtin_backends(self):
        """The real imports in __init__.py register local, api, qwen."""
        modes = available_modes()
        assert "local" in modes
        assert "api" in modes
        assert "qwen" in modes

    def test_includes_dynamically_registered_mode(self):
        register_asr("custom_test_mode")(lambda cm, vc: _DummyService(cm, vc))
        assert "custom_test_mode" in available_modes()

    def test_returns_list(self):
        assert isinstance(available_modes(), list)
