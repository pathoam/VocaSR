"""Energy-based audio segmenter for pipelined transcription.

Finds natural pause boundaries in speech audio by analyzing frame-level
RMS energy. Used to split recording into ~8s segments that can be
transcribed independently and pipelined with ongoing recording.

Algorithm validated in qwen-asr-edge experiments: target ~8s segments
with ±3s search window at longest pause produces 2.6% WER (punctuation
only) with 4-5x latency reduction.
"""

import numpy as np


def compute_energy_frames(audio: np.ndarray, sr: int, frame_ms: int = 10):
    """Compute per-frame RMS energy.

    Args:
        audio: float32 PCM audio samples
        sr: sample rate
        frame_ms: frame duration in milliseconds

    Returns:
        (energies, frame_samples) where energies is a 1D array of RMS
        values per frame, and frame_samples is the number of audio
        samples per frame.
    """
    frame_samples = int(sr * frame_ms / 1000)
    n_frames = len(audio) // frame_samples
    if n_frames == 0:
        return np.zeros(0), frame_samples
    # Reshape for vectorized RMS computation
    trimmed = audio[:n_frames * frame_samples].reshape(n_frames, frame_samples)
    energies = np.sqrt(np.mean(trimmed ** 2, axis=1))
    return energies, frame_samples


def find_longest_pause_near(
    energies: np.ndarray,
    frame_samples: int,
    sr: int,
    target_sec: float,
    search_sec: float = 3.0,
    threshold_pct: float = 15,
) -> tuple:
    """Find the longest low-energy pause within a search window around a target time.

    Searches ±search_sec around target_sec for contiguous runs of frames
    below the energy threshold (percentile of full-audio energy). Returns
    the center of the longest such run — typically a phrase or sentence
    boundary.

    Args:
        energies: per-frame RMS energy array from compute_energy_frames
        frame_samples: samples per frame
        sr: sample rate
        target_sec: target cut point in seconds
        search_sec: half-width of search window in seconds
        threshold_pct: energy percentile below which a frame is "quiet"

    Returns:
        (cut_sample, pause_ms) — the sample index to cut at, and the
        duration of the pause found (0 if no pause, just the quietest point).
    """
    if len(energies) == 0:
        return int(target_sec * sr), 0

    frame_dur = frame_samples / sr
    target_frame = int(target_sec / frame_dur)
    search_frames = int(search_sec / frame_dur)

    start_frame = max(0, target_frame - search_frames)
    end_frame = min(len(energies), target_frame + search_frames)

    if start_frame >= end_frame:
        return int(target_sec * sr), 0

    threshold = np.percentile(energies, threshold_pct)

    is_quiet = energies[start_frame:end_frame] < threshold
    runs = []
    run_start = None
    for i, quiet in enumerate(is_quiet):
        if quiet and run_start is None:
            run_start = i
        elif not quiet and run_start is not None:
            runs.append((run_start + start_frame, i - run_start))
            run_start = None
    if run_start is not None:
        runs.append((run_start + start_frame, len(is_quiet) - run_start))

    if runs:
        longest = max(runs, key=lambda r: r[1])
        center_frame = longest[0] + longest[1] // 2
        cut_sample = int(center_frame * frame_samples)
        pause_ms = longest[1] * frame_dur * 1000
        return cut_sample, pause_ms

    # No pause found — cut at the quietest point in the window
    window = energies[start_frame:end_frame]
    best_frame = start_frame + int(np.argmin(window))
    cut_sample = int(best_frame * frame_samples)
    return cut_sample, 0


def find_segment_boundary(
    audio: np.ndarray,
    sr: int,
    target_sec: float = 8.0,
    search_sec: float = 3.0,
    threshold_pct: float = 15,
) -> tuple:
    """Find the best cut point in audio near a target time.

    Convenience wrapper that computes energy and finds the pause in one call.

    Args:
        audio: float32 PCM audio (full recording so far)
        sr: sample rate (typically 16000)
        target_sec: target segment duration in seconds
        search_sec: search window half-width in seconds
        threshold_pct: energy percentile for silence threshold

    Returns:
        (cut_sample, pause_ms) — sample index to cut at and pause duration.
        Returns (len(audio), 0) if audio is shorter than target_sec - search_sec.
    """
    min_cut = (target_sec - search_sec) * sr
    if len(audio) < min_cut:
        return len(audio), 0

    energies, frame_samples = compute_energy_frames(audio, sr)
    return find_longest_pause_near(
        energies, frame_samples, sr, target_sec, search_sec, threshold_pct
    )
