"""Tests for CommandProcessor — backend priority, built-in commands, dispatch.

Uses a real CommandProcessor but stubs the HTTP layer since we can't hit
real OpenRouter/local LLM endpoints in unit tests. The goal is to verify:
- Backend priority ordering (local first, then OpenRouter)
- Built-in command handlers produce correct output
- Multimodal dispatch routes correctly
- Config update methods propagate to backends
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

from voca.commands.processor import CommandProcessor


@pytest.fixture
def processor():
    """A processor with both backends nominally configured."""
    return CommandProcessor(
        openrouter_api_key="sk-test-key",
        local_llm_endpoint="http://localhost:8000",
        model="test-model",
    )


@pytest.fixture
def processor_openrouter_only():
    """A processor with only OpenRouter configured."""
    return CommandProcessor(
        openrouter_api_key="sk-test-key",
        local_llm_endpoint="",
        model="test-model",
    )


@pytest.fixture
def processor_no_backends():
    """A processor with no backends configured."""
    return CommandProcessor()


class TestBuiltInCommands:
    """Built-in handlers don't touch the network — test them directly."""

    def test_time_command(self, processor):
        result = processor.process("time")
        assert result["success"] is True
        # Should contain a time string (HH:MM AM/PM)
        assert ":" in result["response"]

    def test_date_command(self, processor):
        result = processor.process("date")
        assert result["success"] is True
        now = datetime.now()
        assert str(now.year) in result["response"]

    def test_help_command(self, processor):
        result = processor.process("help")
        assert result["success"] is True
        assert "time" in result["response"].lower()
        assert "date" in result["response"].lower()
        assert "context" in result["response"].lower()

    def test_empty_command(self, processor):
        result = processor.process("")
        assert result["success"] is False
        assert "error" in result

    def test_builtin_case_insensitive(self, processor):
        result = processor.process("TIME")
        assert result["success"] is True

    def test_fix_with_no_app(self, processor):
        result = processor.process("fix")
        assert result["success"] is True


class TestBackendPriority:
    """Verify the local-first, OpenRouter-fallback ordering."""

    def test_local_backend_tried_first(self, processor):
        """When local succeeds, OpenRouter is not called."""
        processor._local.query = MagicMock(return_value={"success": True, "response": "local answer"})
        processor._openrouter.query = MagicMock(return_value={"success": True, "response": "OR answer"})

        result = processor.process("what is 2+2")
        assert result["success"] is True
        assert result["response"] == "local answer"
        processor._openrouter.query.assert_not_called()

    def test_falls_back_to_openrouter_when_local_fails(self, processor):
        processor._local.query = MagicMock(return_value={"success": False, "error": "connection refused"})
        processor._openrouter.query = MagicMock(return_value={"success": True, "response": "OR answer"})

        result = processor.process("what is 2+2")
        assert result["success"] is True
        assert result["response"] == "OR answer"

    def test_no_backends_returns_error(self, processor_no_backends):
        result = processor_no_backends.process("what is 2+2")
        assert result["success"] is False
        assert "available" in result["error"].lower() or "configured" in result["error"].lower()

    def test_openrouter_only_skips_local(self, processor_openrouter_only):
        """When local endpoint is empty, it should not be tried."""
        processor_openrouter_only._openrouter.query = MagicMock(
            return_value={"success": True, "response": "answer"})

        result = processor_openrouter_only.process("what is 2+2")
        assert result["success"] is True


class TestMultimodalDispatch:
    def test_text_clipboard_dispatched(self, processor):
        processor._local.query_multimodal = MagicMock(
            return_value={"success": True, "response": "clipboard answer"})

        clipboard = {"type": "text", "content": "some code here"}
        result = processor.process_with_clipboard("explain this", clipboard)
        assert result["success"] is True

    def test_empty_command_with_clipboard(self, processor):
        result = processor.process_with_clipboard("", {"type": "text", "content": "x"})
        assert result["success"] is False


class TestConfigUpdates:
    def test_set_openrouter_key(self, processor):
        processor.set_openrouter_key("new-key")
        assert processor._openrouter.api_key == "new-key"
        assert processor.openrouter_api_key == "new-key"

    def test_set_local_endpoint(self, processor):
        processor.set_local_endpoint("http://new:9999")
        assert processor._local.endpoint == "http://new:9999"
        assert processor.local_llm_endpoint == "http://new:9999"

    def test_set_model(self, processor):
        processor.set_model("new-model/v2")
        assert processor._openrouter.model == "new-model/v2"
        assert processor.model == "new-model/v2"


class TestVocabularyCommands:
    """Test context management built-ins that depend on vocabulary_manager."""

    def test_show_context_empty(self, processor):
        vm = MagicMock()
        vm.get_terms.return_value = []
        processor.vocabulary_manager = vm

        with patch("voca.utils.notify.notify"):
            result = processor.process("show context")
        assert result["success"] is True

    def test_show_context_with_terms(self, processor):
        vm = MagicMock()
        vm.get_terms.return_value = ["Kubernetes", "PyTorch"]
        processor.vocabulary_manager = vm

        with patch("voca.utils.notify.notify") as mock_notify:
            result = processor.process("show context")
        assert result["success"] is True
        # Verify the notification included the terms
        notify_msg = mock_notify.call_args[0][1]
        assert "Kubernetes" in notify_msg
        assert "PyTorch" in notify_msg

    def test_clear_context(self, processor):
        vm = MagicMock()
        processor.vocabulary_manager = vm

        with patch("voca.utils.notify.notify"):
            result = processor.process("clear context")
        assert result["success"] is True
        vm.clear.assert_called_once()

    def test_context_commands_without_manager(self, processor):
        processor.vocabulary_manager = None
        result = processor.process("show context")
        assert result["success"] is False
        assert "not available" in result["error"].lower()
