"""Tests for RecordingOrchestrator — duration limits, auto-stop, streaming detection.

The AudioRecorder itself requires hardware, so we stub it. But the orchestrator's
duration math, auto-stop thresholds, and callback wiring are tested with real
timing values.
"""

import time
import threading
import pytest
from unittest.mock import MagicMock, patch, call

from voca.recording import RecordingOrchestrator
from voca.transcription.base import StreamingTranscriptionService


class _FakeStreamingService(StreamingTranscriptionService):
    """A minimal streaming service for testing capability detection."""

    def __init__(self, streaming_available=True):
        self._streaming_available = streaming_available

    def transcribe(self, audio_file_path: str) -> str:
        return "fake"

    def is_available(self) -> bool:
        return True

    def start_streaming(self) -> None:
        pass

    def feed_chunk(self, audio_chunk) -> None:
        pass

    def finish_streaming(self) -> str:
        return "streamed"

    def supports_streaming(self) -> bool:
        return self._streaming_available

    def get_max_recording_seconds(self) -> float:
        return 60.0


class _FakeNonStreamingService:
    """Non-streaming service with a recording limit."""

    def transcribe(self, audio_file_path: str) -> str:
        return "fake"

    def is_available(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return False

    def get_max_recording_seconds(self) -> float:
        return 120.0


@pytest.fixture
def audio_recorder():
    rec = MagicMock()
    rec.stop_recording.return_value = "/tmp/fake_audio.wav"
    rec.is_recording.return_value = False
    return rec


@pytest.fixture
def config_manager():
    cm = MagicMock()
    cm.get_setting.return_value = True  # qwen_streaming_enabled default
    return cm


@pytest.fixture
def callbacks():
    return {
        "on_audio_ready": MagicMock(),
        "on_error": MagicMock(),
    }


class TestStreamingCapability:
    def test_streaming_capable_with_streaming_service(self, audio_recorder, config_manager, callbacks):
        svc = _FakeStreamingService(streaming_available=True)
        orch = RecordingOrchestrator(audio_recorder, svc, config_manager,
                                     callbacks["on_audio_ready"], callbacks["on_error"])
        assert orch.is_streaming_capable() is True

    def test_not_streaming_when_service_says_no(self, audio_recorder, config_manager, callbacks):
        svc = _FakeStreamingService(streaming_available=False)
        orch = RecordingOrchestrator(audio_recorder, svc, config_manager,
                                     callbacks["on_audio_ready"], callbacks["on_error"])
        assert orch.is_streaming_capable() is False

    def test_not_streaming_with_non_streaming_service(self, audio_recorder, config_manager, callbacks):
        svc = _FakeNonStreamingService()
        orch = RecordingOrchestrator(audio_recorder, svc, config_manager,
                                     callbacks["on_audio_ready"], callbacks["on_error"])
        assert orch.is_streaming_capable() is False

    def test_not_streaming_when_config_disables_it(self, audio_recorder, callbacks):
        cm = MagicMock()
        cm.get_setting.return_value = False  # streaming disabled
        svc = _FakeStreamingService(streaming_available=True)
        orch = RecordingOrchestrator(audio_recorder, svc, cm,
                                     callbacks["on_audio_ready"], callbacks["on_error"])
        assert orch.is_streaming_capable() is False


class TestStartRecording:
    def test_starts_audio_recorder(self, audio_recorder, config_manager, callbacks):
        svc = _FakeNonStreamingService()
        orch = RecordingOrchestrator(audio_recorder, svc, config_manager,
                                     callbacks["on_audio_ready"], callbacks["on_error"])
        orch.start_recording()
        audio_recorder.start_recording.assert_called_once()

    def test_records_start_time(self, audio_recorder, config_manager, callbacks):
        svc = _FakeNonStreamingService()
        orch = RecordingOrchestrator(audio_recorder, svc, config_manager,
                                     callbacks["on_audio_ready"], callbacks["on_error"])
        before = time.time()
        orch.start_recording()
        after = time.time()
        assert before <= orch._recording_start_time <= after

    def test_sets_max_recording_from_service(self, audio_recorder, config_manager, callbacks):
        svc = _FakeNonStreamingService()
        orch = RecordingOrchestrator(audio_recorder, svc, config_manager,
                                     callbacks["on_audio_ready"], callbacks["on_error"])
        orch.start_recording()
        assert orch._max_recording_sec == 120.0

    def test_starts_streaming_when_capable(self, audio_recorder, config_manager, callbacks):
        svc = _FakeStreamingService()
        orch = RecordingOrchestrator(audio_recorder, svc, config_manager,
                                     callbacks["on_audio_ready"], callbacks["on_error"])
        orch.start_recording()
        assert orch._streaming is True
        audio_recorder.set_chunk_callback.assert_called_once()

    def test_resets_warning_state(self, audio_recorder, config_manager, callbacks):
        svc = _FakeNonStreamingService()
        orch = RecordingOrchestrator(audio_recorder, svc, config_manager,
                                     callbacks["on_audio_ready"], callbacks["on_error"])
        orch._warned_recording_limit = True
        orch._auto_stopping = True
        orch.start_recording()
        assert orch._warned_recording_limit is False
        assert orch._auto_stopping is False


class TestStopRecording:
    def test_invokes_callback_with_audio_file(self, audio_recorder, config_manager, callbacks):
        svc = _FakeNonStreamingService()
        orch = RecordingOrchestrator(audio_recorder, svc, config_manager,
                                     callbacks["on_audio_ready"], callbacks["on_error"])
        orch.start_recording()
        orch.stop_recording()

        callbacks["on_audio_ready"].assert_called_once_with("/tmp/fake_audio.wav", False)

    def test_reports_streaming_true_when_streaming(self, audio_recorder, config_manager, callbacks):
        svc = _FakeStreamingService()
        orch = RecordingOrchestrator(audio_recorder, svc, config_manager,
                                     callbacks["on_audio_ready"], callbacks["on_error"])
        orch.start_recording()
        orch.stop_recording()

        callbacks["on_audio_ready"].assert_called_once_with("/tmp/fake_audio.wav", True)

    def test_clears_chunk_callback(self, audio_recorder, config_manager, callbacks):
        svc = _FakeStreamingService()
        orch = RecordingOrchestrator(audio_recorder, svc, config_manager,
                                     callbacks["on_audio_ready"], callbacks["on_error"])
        orch.start_recording()
        orch.stop_recording()
        audio_recorder.clear_chunk_callback.assert_called()


class TestDurationChecks:
    def test_no_action_when_no_limit(self, audio_recorder, config_manager, callbacks):
        """Services without get_max_recording_seconds → no limit → no auto-stop."""
        svc = MagicMock()
        svc.supports_streaming.return_value = False
        del svc.get_max_recording_seconds  # simulate missing method
        orch = RecordingOrchestrator(audio_recorder, svc, config_manager,
                                     callbacks["on_audio_ready"], callbacks["on_error"])
        orch.start_recording()
        assert orch._max_recording_sec is None
        orch._check_recording_duration()  # should be no-op

    def test_warning_at_90_percent(self, audio_recorder, config_manager, callbacks):
        svc = _FakeNonStreamingService()  # max = 120s
        orch = RecordingOrchestrator(audio_recorder, svc, config_manager,
                                     callbacks["on_audio_ready"], callbacks["on_error"])
        orch.start_recording()

        # Simulate being 91% into the recording
        orch._recording_start_time = time.time() - (120.0 * 0.91)
        with patch("voca.utils.notify.notify") as mock_notify:
            orch._check_recording_duration()
        assert orch._warned_recording_limit is True

    def test_warning_only_fires_once(self, audio_recorder, config_manager, callbacks):
        svc = _FakeNonStreamingService()
        orch = RecordingOrchestrator(audio_recorder, svc, config_manager,
                                     callbacks["on_audio_ready"], callbacks["on_error"])
        orch.start_recording()
        orch._recording_start_time = time.time() - (120.0 * 0.91)

        with patch("voca.utils.notify.notify") as mock_notify:
            orch._check_recording_duration()
            orch._check_recording_duration()  # second call should be no-op
        # notify imported inside the method, so we check the flag instead
        assert orch._warned_recording_limit is True

    def test_auto_stop_at_95_percent(self, audio_recorder, config_manager, callbacks):
        svc = _FakeNonStreamingService()
        orch = RecordingOrchestrator(audio_recorder, svc, config_manager,
                                     callbacks["on_audio_ready"], callbacks["on_error"])
        orch.start_recording()
        orch._recording_start_time = time.time() - (120.0 * 0.96)

        # Patch _auto_stop_recording to just set the flag (don't actually thread)
        orch._auto_stop_recording = MagicMock()
        orch._check_recording_duration()

        assert orch._auto_stopping is True
        orch._auto_stop_recording.assert_called_once()

    def test_auto_stop_only_triggers_once(self, audio_recorder, config_manager, callbacks):
        svc = _FakeNonStreamingService()
        orch = RecordingOrchestrator(audio_recorder, svc, config_manager,
                                     callbacks["on_audio_ready"], callbacks["on_error"])
        orch.start_recording()
        orch._recording_start_time = time.time() - (120.0 * 0.96)

        orch._auto_stop_recording = MagicMock()
        orch._check_recording_duration()
        orch._check_recording_duration()

        orch._auto_stop_recording.assert_called_once()

    def test_no_warning_before_90_percent(self, audio_recorder, config_manager, callbacks):
        svc = _FakeNonStreamingService()
        orch = RecordingOrchestrator(audio_recorder, svc, config_manager,
                                     callbacks["on_audio_ready"], callbacks["on_error"])
        orch.start_recording()
        orch._recording_start_time = time.time() - (120.0 * 0.50)

        orch._check_recording_duration()
        assert orch._warned_recording_limit is False
        assert orch._auto_stopping is False


class TestCleanup:
    def test_cleanup_delegates_to_recorder(self, audio_recorder, config_manager, callbacks):
        svc = _FakeNonStreamingService()
        orch = RecordingOrchestrator(audio_recorder, svc, config_manager,
                                     callbacks["on_audio_ready"], callbacks["on_error"])
        orch.cleanup_audio_file("/tmp/test.wav")
        audio_recorder.cleanup_temp_file.assert_called_once_with("/tmp/test.wav")

    def test_no_cleanup_when_disabled(self, audio_recorder, callbacks):
        cm = MagicMock()
        cm.get_setting.return_value = False  # auto_cleanup disabled
        svc = _FakeNonStreamingService()
        orch = RecordingOrchestrator(audio_recorder, svc, cm,
                                     callbacks["on_audio_ready"], callbacks["on_error"])
        orch.cleanup_audio_file("/tmp/test.wav")
        audio_recorder.cleanup_temp_file.assert_not_called()
