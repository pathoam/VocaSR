"""Tests for TranscriptionPipeline — the core transcribe→correct→detect→insert flow.

Uses real CorrectionPipeline and CommandDetector instances to verify the full
flow. Only the transcription service and text inserter are stubbed, since they
involve hardware/network I/O.
"""

import pytest
from unittest.mock import MagicMock, patch

from voca.pipeline import TranscriptionPipeline, AppState
from voca.text.corrections import CorrectionPipeline
from voca.commands.detector import CommandDetector
from voca.transcription.base import TranscriptionError


@pytest.fixture
def correction_pipeline(tmp_config_dir):
    return CorrectionPipeline(tmp_config_dir)


@pytest.fixture
def command_detector():
    return CommandDetector(["voca", "assistant"])


@pytest.fixture
def text_inserter():
    inserter = MagicMock()
    inserter.insert_text.return_value = True
    inserter.get_clipboard_type_and_content.return_value = {"type": "none"}
    return inserter


@pytest.fixture
def config_manager():
    cm = MagicMock()
    cm.get_text_insertion_method.return_value = "auto"
    return cm


@pytest.fixture
def transcription_service():
    svc = MagicMock()
    svc.transcribe.return_value = "hello world"
    svc.finish_streaming.return_value = "streamed text"
    return svc


@pytest.fixture
def make_pipeline(transcription_service, correction_pipeline, command_detector,
                  text_inserter, config_manager):
    """Factory that builds a pipeline with optional overrides."""
    def _make(**kwargs):
        defaults = dict(
            transcription_service=transcription_service,
            correction_pipeline=correction_pipeline,
            command_detector=command_detector,
            command_processor=MagicMock(),
            command_responder=MagicMock(),
            text_inserter=text_inserter,
            training_data_collector=None,
            config_manager=config_manager,
        )
        defaults.update(kwargs)
        return TranscriptionPipeline(**defaults)
    return _make


class TestNormalTextFlow:
    """Test the standard path: transcribe → correct → insert."""

    def test_transcribed_text_inserted(self, make_pipeline, text_inserter, transcription_service):
        pipe = make_pipeline()
        pipe.transcribe_and_insert("/fake/audio.wav")

        transcription_service.transcribe.assert_called_once_with("/fake/audio.wav")
        text_inserter.insert_text.assert_called_once()
        # Verify the text that was inserted
        inserted = text_inserter.insert_text.call_args[0][0]
        assert inserted == "hello world"

    def test_corrections_applied_before_insertion(self, make_pipeline, text_inserter,
                                                   transcription_service, tmp_config_dir):
        """Verify that the real correction pipeline processes text before insertion."""
        # The default corrections include "voice box" -> "Voca"
        transcription_service.transcribe.return_value = "open voice box settings"
        pipe = make_pipeline()
        pipe.transcribe_and_insert("/fake/audio.wav")

        inserted = text_inserter.insert_text.call_args[0][0]
        assert "voca" in inserted.lower()
        assert "voice box" not in inserted.lower()

    def test_last_insertion_tracked(self, make_pipeline, transcription_service):
        transcription_service.transcribe.return_value = "tracked text"
        pipe = make_pipeline()
        pipe.transcribe_and_insert("/fake/audio.wav")

        assert pipe.last_insertion is not None
        assert pipe.last_insertion["text"] == "tracked text"
        assert pipe.last_insertion["char_count"] == len("tracked text")

    def test_state_returns_to_idle_after_success(self, make_pipeline):
        pipe = make_pipeline()
        pipe.transcribe_and_insert("/fake/audio.wav")
        assert pipe.state == AppState.IDLE

    def test_no_speech_detected_skips_insertion(self, make_pipeline, text_inserter,
                                                 transcription_service):
        transcription_service.transcribe.return_value = "No speech detected"
        pipe = make_pipeline()
        pipe.transcribe_and_insert("/fake/audio.wav")

        text_inserter.insert_text.assert_not_called()
        assert pipe.last_insertion is None

    def test_empty_text_skips_insertion(self, make_pipeline, text_inserter,
                                        transcription_service):
        transcription_service.transcribe.return_value = ""
        pipe = make_pipeline()
        pipe.transcribe_and_insert("/fake/audio.wav")

        text_inserter.insert_text.assert_not_called()

    def test_streaming_finish_and_insert(self, make_pipeline, text_inserter,
                                          transcription_service):
        pipe = make_pipeline()
        pipe.finish_streaming_and_insert("/fake/audio.wav")

        transcription_service.finish_streaming.assert_called_once()
        text_inserter.insert_text.assert_called_once()
        inserted = text_inserter.insert_text.call_args[0][0]
        assert inserted == "streamed text"


class TestCommandFlow:
    """Test the command detection and routing path."""

    def test_command_detected_and_routed(self, make_pipeline, transcription_service,
                                         text_inserter):
        transcription_service.transcribe.return_value = "voca, what time is it"
        processor = MagicMock()
        processor.process.return_value = {"success": True, "response": "3:00 PM"}
        pipe = make_pipeline(command_processor=processor)
        pipe.transcribe_and_insert("/fake/audio.wav")

        processor.process.assert_called_once()
        # The command text (after trigger removal) should be passed
        cmd_arg = processor.process.call_args[0][0]
        assert "time" in cmd_arg

    def test_command_response_inserted(self, make_pipeline, transcription_service,
                                        text_inserter):
        transcription_service.transcribe.return_value = "assistant, say hello"
        processor = MagicMock()
        processor.process.return_value = {"success": True, "response": "Hello there!"}
        pipe = make_pipeline(command_processor=processor)
        pipe.transcribe_and_insert("/fake/audio.wav")

        text_inserter.insert_text.assert_called_once()
        inserted = text_inserter.insert_text.call_args[0][0]
        assert inserted == "Hello there!"

    def test_non_command_text_not_routed_to_processor(self, make_pipeline,
                                                       transcription_service):
        transcription_service.transcribe.return_value = "just regular speech"
        processor = MagicMock()
        pipe = make_pipeline(command_processor=processor)
        pipe.transcribe_and_insert("/fake/audio.wav")

        processor.process.assert_not_called()

    def test_command_with_no_detector_treated_as_text(self, make_pipeline,
                                                       transcription_service,
                                                       text_inserter):
        """With command_detector=None, everything is plain text."""
        transcription_service.transcribe.return_value = "voca, do something"
        pipe = make_pipeline(command_detector=None)
        pipe.transcribe_and_insert("/fake/audio.wav")

        text_inserter.insert_text.assert_called_once()
        inserted = text_inserter.insert_text.call_args[0][0]
        # Should be inserted as-is (after corrections), not routed to processor
        assert "voca" in inserted.lower() or "Voca" in inserted

    def test_failed_command_displays_via_responder(self, make_pipeline,
                                                    transcription_service):
        transcription_service.transcribe.return_value = "voca, fail please"
        processor = MagicMock()
        processor.process.return_value = {"success": False, "error": "LLM unavailable"}
        responder = MagicMock()
        pipe = make_pipeline(command_processor=processor, command_responder=responder)
        pipe.transcribe_and_insert("/fake/audio.wav")

        responder.display_response.assert_called_once()
        resp_arg = responder.display_response.call_args[0][0]
        assert resp_arg["success"] is False

    def test_empty_command_response_not_inserted(self, make_pipeline,
                                                   transcription_service, text_inserter):
        transcription_service.transcribe.return_value = "voca, add context"
        processor = MagicMock()
        processor.process.return_value = {"success": True, "response": ""}
        pipe = make_pipeline(command_processor=processor)
        pipe.transcribe_and_insert("/fake/audio.wav")

        text_inserter.insert_text.assert_not_called()


class TestErrorHandling:
    def test_transcription_error_sets_idle(self, make_pipeline, transcription_service):
        transcription_service.transcribe.side_effect = TranscriptionError("model failed")
        error_cb = MagicMock()
        pipe = make_pipeline(on_error=error_cb)
        pipe.transcribe_and_insert("/fake/audio.wav")

        assert pipe.state == AppState.IDLE
        error_cb.assert_called_once()
        assert "model failed" in error_cb.call_args[0][0]

    def test_unexpected_error_sets_idle(self, make_pipeline, transcription_service):
        transcription_service.transcribe.side_effect = RuntimeError("boom")
        pipe = make_pipeline()
        pipe.transcribe_and_insert("/fake/audio.wav")
        assert pipe.state == AppState.IDLE

    def test_streaming_error_sets_idle(self, make_pipeline, transcription_service):
        transcription_service.finish_streaming.side_effect = TranscriptionError("stream fail")
        pipe = make_pipeline()
        pipe.finish_streaming_and_insert("/fake/audio.wav")
        assert pipe.state == AppState.IDLE


class TestCallbacks:
    def test_on_transcription_complete_called(self, make_pipeline, transcription_service):
        transcription_service.transcribe.return_value = "callback text"
        cb = MagicMock()
        pipe = make_pipeline(on_transcription_complete=cb)
        pipe.transcribe_and_insert("/fake/audio.wav")

        cb.assert_called_once_with("callback text")

    def test_state_change_callback(self, make_pipeline, transcription_service):
        states = []
        pipe = make_pipeline(on_state_change=lambda s: states.append(s))
        pipe.transcribe_and_insert("/fake/audio.wav")

        # Should have gone through INSERTING and back to IDLE
        assert AppState.INSERTING in states
        assert states[-1] == AppState.IDLE


class TestTrainingData:
    def test_training_data_saved_on_success(self, make_pipeline, transcription_service):
        transcription_service.transcribe.return_value = "saved text"
        collector = MagicMock()
        collector.save_sample.return_value = "sample_001"
        pipe = make_pipeline(training_data_collector=collector)
        pipe.transcribe_and_insert("/fake/audio.wav")

        collector.save_sample.assert_called_once()
        call_kwargs = collector.save_sample.call_args[1]
        assert call_kwargs["audio_file"] == "/fake/audio.wav"
        assert call_kwargs["was_command"] is False

    def test_training_data_marks_commands(self, make_pipeline, transcription_service):
        transcription_service.transcribe.return_value = "voca, what time is it"
        processor = MagicMock()
        processor.process.return_value = {"success": True, "response": "3 PM"}
        collector = MagicMock()
        pipe = make_pipeline(command_processor=processor, training_data_collector=collector)
        pipe.transcribe_and_insert("/fake/audio.wav")

        collector.save_sample.assert_called_once()
        call_kwargs = collector.save_sample.call_args[1]
        assert call_kwargs["was_command"] is True
