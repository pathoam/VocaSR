"""Tests for command detection and extraction."""

import pytest

from voca.commands.detector import CommandDetector


# ---------------------------------------------------------------------------
# is_command
# ---------------------------------------------------------------------------

class TestIsCommand:

    def test_trigger_at_start(self):
        d = CommandDetector()
        assert d.is_command("voca, do something") is True

    def test_case_insensitive(self):
        d = CommandDetector()
        assert d.is_command("VOCA, do something") is True

    def test_no_trigger(self):
        d = CommandDetector()
        assert d.is_command("hello world") is False

    def test_empty_string(self):
        d = CommandDetector()
        assert d.is_command("") is False

    def test_custom_triggers(self):
        d = CommandDetector(triggers=["hey", "yo"])
        assert d.is_command("hey, do this") is True
        assert d.is_command("yo tell me something") is True
        assert d.is_command("voca, do something") is False


# ---------------------------------------------------------------------------
# extract_command
# ---------------------------------------------------------------------------

class TestExtractCommand:

    def test_returns_trigger_and_command(self):
        d = CommandDetector()
        trigger, command = d.extract_command("voca, what time is it")
        assert trigger == "voca"
        assert command == "what time is it"

    def test_comma_separator(self):
        d = CommandDetector()
        trigger, command = d.extract_command("voca, what time")
        assert trigger == "voca"
        assert "what time" in command

    def test_colon_separator(self):
        d = CommandDetector()
        trigger, command = d.extract_command("voca: what time")
        assert trigger == "voca"
        assert "what time" in command

    def test_no_match(self):
        d = CommandDetector()
        trigger, command = d.extract_command("hello world")
        assert trigger is None
        assert command is None

    def test_empty_string(self):
        d = CommandDetector()
        trigger, command = d.extract_command("")
        assert trigger is None
        assert command is None


# ---------------------------------------------------------------------------
# extract_command_with_clipboard
# ---------------------------------------------------------------------------

class TestExtractCommandWithClipboard:

    def test_clipboard_detection(self):
        d = CommandDetector()
        trigger, command, has_clipboard = d.extract_command_with_clipboard(
            "voca, summarize the clipboard here"
        )
        assert has_clipboard is True
        assert trigger == "voca"
        # "clipboard" and context should be removed from command
        assert "clipboard" not in command.lower()

    def test_from_clipboard_pattern(self):
        d = CommandDetector()
        _, command, has_clipboard = d.extract_command_with_clipboard(
            "voca, read from clipboard"
        )
        assert has_clipboard is True
        assert "clipboard" not in command.lower()

    def test_in_the_clipboard_pattern(self):
        d = CommandDetector()
        _, command, has_clipboard = d.extract_command_with_clipboard(
            "voca, summarize text in the clipboard"
        )
        assert has_clipboard is True
        assert "clipboard" not in command.lower()

    def test_this_clipboard_pattern(self):
        d = CommandDetector()
        _, command, has_clipboard = d.extract_command_with_clipboard(
            "voca, summarize this clipboard"
        )
        assert has_clipboard is True
        assert "clipboard" not in command.lower()

    def test_thats_cleanup_after_clipboard(self):
        d = CommandDetector()
        _, command, has_clipboard = d.extract_command_with_clipboard(
            "voca, that's the clipboard summarize it"
        )
        assert has_clipboard is True
        # "that's" at the start after clipboard removal should be cleaned
        assert not command.lower().startswith("that's")

    def test_no_clipboard(self):
        d = CommandDetector()
        trigger, command, has_clipboard = d.extract_command_with_clipboard(
            "voca, what is the weather"
        )
        assert has_clipboard is False
        assert trigger == "voca"
        assert "weather" in command


# ---------------------------------------------------------------------------
# add_trigger / remove_trigger
# ---------------------------------------------------------------------------

class TestTriggerManagement:

    def test_add_trigger(self):
        d = CommandDetector(triggers=["voca"])
        d.add_trigger("jarvis")
        assert d.is_command("jarvis, do something") is True

    def test_add_duplicate_case_insensitive(self):
        d = CommandDetector(triggers=["voca"])
        d.add_trigger("VOCA")
        assert len(d.triggers) == 1

    def test_remove_trigger(self):
        d = CommandDetector(triggers=["voca", "hey"])
        d.remove_trigger("hey")
        assert d.is_command("hey, do something") is False
        assert d.is_command("voca, do something") is True

    def test_get_triggers_returns_copy(self):
        d = CommandDetector(triggers=["voca", "hey"])
        triggers = d.get_triggers()
        assert set(triggers) == {"voca", "hey"}
        # Should be a copy — mutating it shouldn't affect the detector
        triggers.append("jarvis")
        assert "jarvis" not in d.triggers

    def test_get_triggers_reflects_add(self):
        d = CommandDetector(triggers=["voca"])
        d.add_trigger("jarvis")
        assert "jarvis" in d.get_triggers()
