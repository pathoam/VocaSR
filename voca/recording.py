"""Recording orchestrator — manages start/stop recording and duration limits."""

import threading
import time
from typing import Callable, Optional

from voca.audio.capture import AudioRecorder
from voca.transcription.base import StreamingTranscriptionService
from voca.config.manager import ConfigManager
from voca.utils.logging import get_logger

logger = get_logger(__name__)


class RecordingOrchestrator:
    """Manages audio recording lifecycle and duration limits."""

    def __init__(
        self,
        audio_recorder: AudioRecorder,
        transcription_service,
        config_manager: ConfigManager,
        on_audio_ready: Callable[[str, bool], None],
        on_error: Callable[[str], None],
    ):
        """
        Args:
            audio_recorder: The audio capture instance.
            transcription_service: Current transcription service.
            config_manager: Config manager for settings.
            on_audio_ready: Callback(audio_file, was_streaming) when recording stops.
            on_error: Callback(message) on error.
        """
        self.audio_recorder = audio_recorder
        self.transcription_service = transcription_service
        self.config_manager = config_manager
        self._on_audio_ready = on_audio_ready
        self._on_error = on_error

        # Recording state
        self._recording_start_time: Optional[float] = None
        self._max_recording_sec: Optional[float] = None
        self._warned_recording_limit = False
        self._auto_stopping = False
        self._streaming = False

    def is_streaming_capable(self) -> bool:
        return (
            isinstance(self.transcription_service, StreamingTranscriptionService)
            and self.transcription_service.supports_streaming()
            and self.config_manager.get_setting("qwen_streaming_enabled", True)
        )

    def start_recording(self) -> None:
        """Start audio recording, optionally with streaming."""
        self._recording_start_time = time.time()
        self._warned_recording_limit = False
        self._auto_stopping = False
        self._streaming = self.is_streaming_capable()

        # Compute max recording duration
        self._max_recording_sec = None
        if hasattr(self.transcription_service, 'get_max_recording_seconds'):
            self._max_recording_sec = self.transcription_service.get_max_recording_seconds()
            if self._max_recording_sec:
                logger.info(f"Max safe recording: {self._max_recording_sec:.0f}s")

        if self._streaming:
            logger.debug("Starting streaming transcription")
            self.transcription_service.start_streaming()
            original_feed = self.transcription_service.feed_chunk

            def _checked_feed(chunk):
                original_feed(chunk)
                self._check_recording_duration()

            self.audio_recorder.set_chunk_callback(_checked_feed)
        elif self._max_recording_sec:
            def _duration_only_callback(chunk):
                self._check_recording_duration()
            self.audio_recorder.set_chunk_callback(_duration_only_callback)

        self.audio_recorder.start_recording()

    def stop_recording(self) -> None:
        """Stop recording and invoke callback with audio file path."""
        streaming = self._streaming
        self.audio_recorder.clear_chunk_callback()
        audio_file = self.audio_recorder.stop_recording()
        self._on_audio_ready(audio_file, streaming)

    def _check_recording_duration(self) -> None:
        if not self._max_recording_sec or not self._recording_start_time:
            return
        elapsed = time.time() - self._recording_start_time

        if elapsed >= self._max_recording_sec * 0.95:
            if not self._auto_stopping:
                self._auto_stopping = True
                logger.warning(f"Recording auto-stopped at {elapsed:.0f}s (limit {self._max_recording_sec:.0f}s)")
                self._auto_stop_recording()
            return

        if not self._warned_recording_limit and elapsed >= self._max_recording_sec * 0.90:
            self._warned_recording_limit = True
            remaining = int(self._max_recording_sec - elapsed)
            logger.info(f"Recording duration warning at {elapsed:.0f}s, ~{remaining}s remaining")
            try:
                from voca.utils.notify import notify
                notify("Voca", f"~{remaining} seconds of recording remaining")
            except Exception:
                pass

    def _auto_stop_recording(self) -> None:
        t = threading.Thread(target=self._auto_stop_sync, daemon=True)
        t.start()

    def _auto_stop_sync(self) -> None:
        try:
            self.stop_recording()
        except Exception as e:
            logger.error(f"Auto-stop failed: {e}")

    def cleanup_audio_file(self, file_path: str) -> None:
        if self.config_manager.get_setting("auto_cleanup_temp_files", True):
            self.audio_recorder.cleanup_temp_file(file_path)
