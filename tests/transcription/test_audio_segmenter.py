"""Tests for audio segmenter: energy computation and pause detection."""

import numpy as np
import pytest

from voca.transcription.audio_segmenter import (
    compute_energy_frames,
    find_longest_pause_near,
    find_segment_boundary,
)

SR = 16000  # standard sample rate


# ---------------------------------------------------------------------------
# compute_energy_frames
# ---------------------------------------------------------------------------

class TestComputeEnergyFrames:

    def test_silence_near_zero_energy(self):
        audio = np.zeros(SR, dtype=np.float32)  # 1 second of silence
        energies, frame_samples = compute_energy_frames(audio, SR)
        assert len(energies) > 0
        assert np.all(energies < 1e-7)

    def test_loud_signal_high_energy(self):
        audio = np.ones(SR, dtype=np.float32) * 0.8  # 1s loud constant
        energies, _ = compute_energy_frames(audio, SR)
        assert np.all(energies > 0.5)

    def test_energy_proportional_to_amplitude(self):
        quiet = np.ones(SR, dtype=np.float32) * 0.1
        loud = np.ones(SR, dtype=np.float32) * 0.5
        e_quiet, _ = compute_energy_frames(quiet, SR)
        e_loud, _ = compute_energy_frames(loud, SR)
        assert np.mean(e_loud) > np.mean(e_quiet)

    def test_short_audio_empty_energy(self):
        # Fewer samples than one frame (10ms = 160 samples at 16kHz)
        audio = np.zeros(100, dtype=np.float32)
        energies, frame_samples = compute_energy_frames(audio, SR)
        assert len(energies) == 0

    def test_frame_count(self):
        n_samples = SR * 2  # 2 seconds
        audio = np.zeros(n_samples, dtype=np.float32)
        energies, frame_samples = compute_energy_frames(audio, SR)
        expected_frames = n_samples // frame_samples
        assert len(energies) == expected_frames


# ---------------------------------------------------------------------------
# find_longest_pause_near
# ---------------------------------------------------------------------------

class TestFindLongestPauseNear:

    def _make_speech_silence_speech(self, duration_sec=6, silence_start=2.5, silence_dur=1.0):
        """Create audio with loud speech and clear silence."""
        rng = np.random.default_rng(42)
        n = int(duration_sec * SR)
        audio = rng.standard_normal(n).astype(np.float32) * 0.5
        s_start = int(silence_start * SR)
        s_end = int((silence_start + silence_dur) * SR)
        audio[s_start:s_end] = 0.0
        return audio

    def test_finds_pause_at_silence(self):
        # Use 10s audio so 1s silence is only 10%, keeping 15th percentile
        # in the noise range (above zero) rather than at exactly zero.
        audio = self._make_speech_silence_speech(
            duration_sec=10, silence_start=4.5, silence_dur=1.0
        )
        energies, frame_samples = compute_energy_frames(audio, SR)
        cut_sample, pause_ms = find_longest_pause_near(
            energies, frame_samples, SR, target_sec=5.0, search_sec=3.0
        )
        # Cut should be near the silence center (~5.0s)
        cut_sec = cut_sample / SR
        assert 3.5 < cut_sec < 6.5
        assert pause_ms > 0

    def test_no_silence_falls_back_to_quietest(self):
        # Uniform noise — no real silence
        audio = np.random.randn(SR * 4).astype(np.float32) * 0.3
        energies, frame_samples = compute_energy_frames(audio, SR)
        cut_sample, pause_ms = find_longest_pause_near(
            energies, frame_samples, SR, target_sec=2.0, search_sec=1.0
        )
        # Should still return a valid cut point with pause_ms=0 or >0
        assert 0 <= cut_sample <= len(audio)

    def test_empty_energies(self):
        cut_sample, pause_ms = find_longest_pause_near(
            np.zeros(0), 160, SR, target_sec=3.0
        )
        assert cut_sample == int(3.0 * SR)
        assert pause_ms == 0


# ---------------------------------------------------------------------------
# find_segment_boundary
# ---------------------------------------------------------------------------

class TestFindSegmentBoundary:

    def test_short_audio_returns_full_length(self):
        # Audio shorter than target - search_window (8-3=5s)
        audio = np.zeros(SR * 3, dtype=np.float32)
        cut_sample, pause_ms = find_segment_boundary(audio, SR, target_sec=8.0, search_sec=3.0)
        assert cut_sample == len(audio)
        assert pause_ms == 0

    def test_known_pause_location(self):
        # 16 seconds of audio with silence at ~8s
        n = SR * 16
        audio = np.random.randn(n).astype(np.float32) * 0.3
        # Insert 1-second silence at 7.5-8.5s
        audio[int(7.5 * SR):int(8.5 * SR)] = 0.0
        cut_sample, pause_ms = find_segment_boundary(audio, SR, target_sec=8.0, search_sec=3.0)
        cut_sec = cut_sample / SR
        # Should find boundary near the silence (5-11s range)
        assert 5.0 < cut_sec < 11.0
        assert pause_ms > 0

    def test_end_to_end_16s_silence_at_8s(self):
        """Full integration: 16s audio with clear silence at 8s."""
        n = SR * 16
        audio = np.random.randn(n).astype(np.float32) * 0.5
        # Strong 1.5s silence at 8s
        audio[int(7.5 * SR):int(9.0 * SR)] = 0.0
        cut_sample, pause_ms = find_segment_boundary(audio, SR)
        cut_sec = cut_sample / SR
        assert 6.0 < cut_sec < 10.0
