#!/usr/bin/env python3
"""Test audio capture functionality.

These tests require audio hardware and are marked as integration tests.
Run with: uv run pytest -m integration
"""

import os
import time

import pytest

from voca.audio.capture import AudioRecorder


pytestmark = pytest.mark.integration


def test_audio_device_listing():
    """Test listing audio devices."""
    AudioRecorder.list_audio_devices()


def test_audio_recorder_init():
    """Test audio recorder initialization."""
    recorder = AudioRecorder()
    assert recorder.sample_rate > 0
    assert recorder.channels >= 1


def test_audio_recording_simulation():
    """Test audio recording start/stop (simulation)."""
    recorder = AudioRecorder()

    assert not recorder.is_recording()

    recorder.start_recording()
    assert recorder.is_recording()

    time.sleep(1)

    audio_file = recorder.stop_recording()
    assert not recorder.is_recording()
    assert os.path.exists(audio_file)

    recorder.cleanup_temp_file(audio_file)
