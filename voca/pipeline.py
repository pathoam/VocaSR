"""Transcription pipeline — processes transcribed text through corrections, commands, and insertion."""

from enum import Enum
from typing import Optional, Callable

from voca.transcription.base import TranscriptionError
from voca.utils.logging import get_logger

logger = get_logger(__name__)


class AppState(Enum):
    IDLE = "idle"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"
    PROCESSING_COMMAND = "processing_command"
    INSERTING = "inserting"
    ERROR = "error"


class TranscriptionPipeline:
    """Handles the flow from transcribed text to inserted/processed output."""

    def __init__(
        self,
        transcription_service,
        correction_pipeline,
        command_detector,
        command_processor,
        command_responder,
        text_inserter,
        training_data_collector,
        config_manager,
        on_transcription_complete: Optional[Callable] = None,
        on_state_change: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
    ):
        self.transcription_service = transcription_service
        self.correction_pipeline = correction_pipeline
        self.command_detector = command_detector
        self.command_processor = command_processor
        self.command_responder = command_responder
        self.text_inserter = text_inserter
        self.training_data_collector = training_data_collector
        self.config_manager = config_manager
        self.on_transcription_complete = on_transcription_complete
        self._on_state_change = on_state_change
        self._on_error = on_error

        self._last_insertion: Optional[dict] = None
        self.state = AppState.IDLE

    def _set_state(self, state: AppState) -> None:
        self.state = state
        if self._on_state_change:
            self._on_state_change(state)

    @property
    def last_insertion(self):
        return self._last_insertion

    def transcribe_and_insert(self, audio_file: str) -> None:
        """Transcribe audio file and process the result (runs in caller's thread)."""
        try:
            transcribed_text = self.transcription_service.transcribe(audio_file)
            self._process_transcribed_text(transcribed_text, audio_file=audio_file)
        except TranscriptionError as e:
            self._report_error(f"Transcription failed: {e}", "Transcription")
        except Exception as e:
            self._report_error(f"Unexpected error: {e}", "General")
        finally:
            self._set_state(AppState.IDLE)

    def finish_streaming_and_insert(self, audio_file: str) -> None:
        """Finish streaming transcription and process the result."""
        try:
            transcribed_text = self.transcription_service.finish_streaming()
            self._process_transcribed_text(transcribed_text, audio_file=audio_file)
        except TranscriptionError as e:
            self._report_error(f"Streaming transcription failed: {e}", "Transcription")
        except Exception as e:
            self._report_error(f"Unexpected error: {e}", "General")
        finally:
            self._set_state(AppState.IDLE)

    def _report_error(self, message: str, error_type: str) -> None:
        from voca.utils.error_suggestions import get_suggestion
        suggestion = get_suggestion(Exception(message), {"operation": "transcription"})
        logger.error(f"{error_type}: {message}")
        if self._on_error:
            self._on_error(message, error_type, suggestion)

    def _process_transcribed_text(self, transcribed_text: str, audio_file: str = None) -> None:
        if not transcribed_text or transcribed_text == "No speech detected":
            print(" (no speech detected)")
            self._set_state(AppState.IDLE)
            return

        raw_text = transcribed_text
        corrections_applied = []

        # Apply corrections
        if self.correction_pipeline:
            original_text = transcribed_text
            result = self.correction_pipeline.apply(transcribed_text)
            transcribed_text = result.text
            corrections_applied = [
                {"stage": c.stage, "original": c.original, "corrected": c.corrected}
                for c in result.corrections
            ]
            if original_text != transcribed_text:
                print(f" done!\nOriginal: {original_text}")
                print(f"Fixed:    {transcribed_text}")
            else:
                print(f" done!\n{transcribed_text}")
        else:
            print(f" done!\n{transcribed_text}")

        # Check for commands
        if self.command_detector and self.command_detector.is_command(transcribed_text):
            self._set_state(AppState.PROCESSING_COMMAND)
            print("🎯 Command detected, processing...")

            trigger, command, has_clipboard = (
                self.command_detector.extract_command_with_clipboard(transcribed_text)
            )
            if command:
                if has_clipboard:
                    print("📋 Clipboard flag detected, reading clipboard...")
                    clipboard_data = self.text_inserter.get_clipboard_type_and_content()
                    result = self.command_processor.process_with_clipboard(command, clipboard_data)
                else:
                    result = self.command_processor.process(command)

                if result.get("success"):
                    llm_response = result.get("response", "").strip()
                    if llm_response:
                        self._set_state(AppState.INSERTING)
                        insertion_method = self.config_manager.get_text_insertion_method()
                        success = self.text_inserter.insert_text(llm_response, method=insertion_method)
                        if self.on_transcription_complete:
                            self.on_transcription_complete(f"[Command] {llm_response}")
                        if not success:
                            logger.error("Failed to insert LLM response")
                else:
                    error_msg = result.get("error", "Unknown error")
                    logger.error(f"Command failed: {error_msg}")
                    self.command_responder.display_response(result)

            # Save training data for commands
            if self.training_data_collector and audio_file:
                self.training_data_collector.save_sample(
                    audio_file=audio_file,
                    raw_transcript=raw_text,
                    auto_corrected=transcribed_text,
                    corrections_applied=corrections_applied,
                    was_command=True,
                )
            return

        # Normal text insertion
        if self.on_transcription_complete:
            self.on_transcription_complete(transcribed_text)

        self._set_state(AppState.INSERTING)
        insertion_method = self.config_manager.get_text_insertion_method()
        success = self.text_inserter.insert_text(transcribed_text, method=insertion_method)

        if not success:
            logger.error("Failed to insert text")
        else:
            logger.info("Text inserted successfully")
            data_id = None
            if self.training_data_collector and audio_file:
                data_id = self.training_data_collector.save_sample(
                    audio_file=audio_file,
                    raw_transcript=raw_text,
                    auto_corrected=transcribed_text,
                    corrections_applied=corrections_applied,
                    was_command=False,
                )
            self._last_insertion = {
                "text": transcribed_text,
                "char_count": len(transcribed_text),
                "audio_file": audio_file,
                "data_id": data_id,
            }
