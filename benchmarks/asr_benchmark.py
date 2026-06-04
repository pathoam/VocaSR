#!/usr/bin/env python3
"""ASR Benchmark: Batch vs Pipelined Latency & Text Quality.

Loads the Qwen ASR model once, runs 4 transcription tests on real recorded
audio (2 durations x 2 modes), and prints a comparison table with latency
metrics and over-punctuation analysis.

Run from voca root:
    .venv/bin/python benchmarks/asr_benchmark.py
"""

import math
import os
import sys
import time
import wave
import glob

import numpy as np

# Ensure voca package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from voca.transcription.qwen_asr import QwenASRService
from voca.text.corrections import SegmentJoinStage


# ---------------------------------------------------------------------------
# Audio selection
# ---------------------------------------------------------------------------

TRAINING_DIR = os.path.expanduser("~/.config/Voca/training_data/audio")
TARGET_DURATIONS = [10, 60]  # seconds
SEGMENT_TARGET_SEC = 8.0  # pipeline's target segment length


def find_closest_wav(target_seconds: float) -> tuple[str, float]:
    """Return (path, actual_duration) of the WAV closest to target_seconds."""
    best_path, best_dur, best_diff = None, 0.0, float("inf")
    for path in glob.glob(os.path.join(TRAINING_DIR, "*.wav")):
        with wave.open(path) as w:
            dur = w.getnframes() / w.getframerate()
        diff = abs(dur - target_seconds)
        if diff < best_diff:
            best_path, best_dur, best_diff = path, dur, diff
    if best_path is None:
        raise FileNotFoundError(f"No WAV files in {TRAINING_DIR}")
    return best_path, best_dur


def load_audio(path: str) -> np.ndarray:
    """Load a 16 kHz mono WAV as float32 numpy array."""
    with wave.open(path) as w:
        frames = w.readframes(w.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return audio


# ---------------------------------------------------------------------------
# Benchmark runs
# ---------------------------------------------------------------------------

CHUNK_SIZE = 1024  # samples per feed_chunk call


def run_batch(service: QwenASRService, wav_path: str) -> dict:
    """Batch transcription — single inference pass on the full file."""
    t0 = time.monotonic()
    text = service.transcribe(wav_path)
    elapsed = time.monotonic() - t0
    return {"text": text, "latency": elapsed}


def run_pipelined(service: QwenASRService, wav_path: str,
                  audio_duration: float) -> dict:
    """Pipelined transcription — feed all chunks instantly, then finish.

    Chunks are fed without real-time delay to measure pure compute cost.
    We then estimate the real-world post-recording wait: in actual use the
    pipeline transcribes earlier segments concurrently with recording, but
    the last segment can't begin inference until recording ends. So the user
    always waits at least one segment's worth of compute.
    """
    audio = load_audio(wav_path)

    t0 = time.monotonic()
    service.start_streaming()

    # Feed all chunks as fast as possible (no real-time delay)
    for offset in range(0, len(audio), CHUNK_SIZE):
        chunk = audio[offset:offset + CHUNK_SIZE]
        service.feed_chunk(chunk)

    t_last_chunk = time.monotonic()
    text = service.finish_streaming()
    t_done = time.monotonic()

    total_compute = t_done - t0
    num_segments = max(1, math.ceil(audio_duration / SEGMENT_TARGET_SEC))
    per_segment = total_compute / num_segments

    return {
        "text": text,
        "total_compute": total_compute,
        "feed_time": t_last_chunk - t0,
        "num_segments": num_segments,
        "per_segment": per_segment,
        # The last segment's inference is the irreducible wait after recording.
        # For short audio (1 segment), pipelined = batch (no benefit).
        # For long audio, max wait is clamped to ~1 segment regardless of
        # total duration — no more 15-20s waits.
        "estimated_wait": per_segment,
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_text(raw_text: str, joiner: SegmentJoinStage) -> dict:
    """Run SegmentJoinStage and collect period/correction stats."""
    result = joiner.apply(raw_text)
    return {
        "raw_text": raw_text,
        "corrected_text": result.text,
        "periods_raw": raw_text.count("."),
        "periods_corrected": result.text.count("."),
        "corrections": len(result.corrections),
        "correction_details": result.corrections,
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

SEP = "=" * 78


def fmt_time(seconds: float) -> str:
    return f"{seconds:.2f}s"


def print_results(duration_label: str, wav_path: str, actual_dur: float,
                  batch: dict, pipe: dict,
                  batch_analysis: dict, pipe_analysis: dict):
    print(f"\n{SEP}")
    print(f"  {duration_label} audio  ({actual_dur:.1f}s actual)")
    print(f"  {os.path.basename(wav_path)}")
    print(SEP)

    batch_wait = batch["latency"]
    pipe_wait = pipe["estimated_wait"]
    speedup = batch_wait / pipe_wait if pipe_wait > 0 else float("inf")

    print(f"\n{'Metric':<34} {'Batch':>12} {'Pipelined':>12}")
    print("-" * 60)
    print(f"{'Total compute':<34} {fmt_time(batch['latency']):>12} {fmt_time(pipe['total_compute']):>12}")
    print(f"{'Segments':<34} {'1':>12} {pipe['num_segments']:>12}")
    print(f"{'Avg per segment':<34} {fmt_time(batch['latency']):>12} {fmt_time(pipe['per_segment']):>12}")
    print(f"{'User wait after recording':<34} {fmt_time(batch_wait):>12} {fmt_time(pipe_wait):>12}")
    print(f"{'Speedup (user wait)':<34} {'baseline':>12} {f'{speedup:.1f}x':>12}")

    print(f"\n  Batch: user waits full {fmt_time(batch_wait)} after recording stops.")
    print(f"  Pipelined: {pipe['num_segments']} segments processed during recording;")
    print(f"  user waits only the last segment's inference ({fmt_time(pipe_wait)}).")

    # Text quality table
    print(f"\n{'Text Quality':<30} {'Batch':>12} {'Pipelined':>12}")
    print("-" * 56)
    print(f"{'Periods (raw)':<30} {batch_analysis['periods_raw']:>12} {pipe_analysis['periods_raw']:>12}")
    print(f"{'Periods (corrected)':<30} {batch_analysis['periods_corrected']:>12} {pipe_analysis['periods_corrected']:>12}")
    print(f"{'Corrections applied':<30} {batch_analysis['corrections']:>12} {pipe_analysis['corrections']:>12}")

    # Text comparison
    print(f"\n--- Batch text ---")
    print(batch_analysis["corrected_text"])
    print(f"\n--- Pipelined text (corrected) ---")
    print(pipe_analysis["corrected_text"])

    if batch_analysis["corrected_text"] != pipe_analysis["corrected_text"]:
        print(f"\n--- Differences ---")
        batch_words = batch_analysis["corrected_text"].split()
        pipe_words = pipe_analysis["corrected_text"].split()
        max_len = max(len(batch_words), len(pipe_words))
        diffs = []
        for i in range(max_len):
            bw = batch_words[i] if i < len(batch_words) else "<missing>"
            pw = pipe_words[i] if i < len(pipe_words) else "<missing>"
            if bw != pw:
                diffs.append(f"  word {i}: batch={bw!r}  pipe={pw!r}")
        if diffs:
            print("\n".join(diffs[:20]))
            if len(diffs) > 20:
                print(f"  ... and {len(diffs) - 20} more differences")
    else:
        print("\n  (texts are identical)")

    if pipe_analysis["correction_details"]:
        print(f"\n--- Pipelined corrections detail ---")
        for c in pipe_analysis["correction_details"]:
            print(f"  {c.original!r} -> {c.corrected!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("ASR Benchmark: Batch vs Pipelined")
    print(SEP)

    # Find test audio
    test_cases = []
    for target in TARGET_DURATIONS:
        path, actual = find_closest_wav(target)
        test_cases.append((f"~{target}s", path, actual))
        print(f"  {target}s target -> {os.path.basename(path)} ({actual:.1f}s)")

    # Load model once
    print("\nLoading model...")
    t0 = time.monotonic()
    service = QwenASRService(model_size="0.6B", backend="auto", language="auto")
    service._load_model()
    print(f"Model loaded in {time.monotonic() - t0:.1f}s")

    joiner = SegmentJoinStage()

    # Run benchmarks
    for label, wav_path, actual_dur in test_cases:
        print(f"\nRunning {label} benchmark...")

        print("  Batch transcription...")
        batch = run_batch(service, wav_path)

        print("  Pipelined transcription...")
        pipe = run_pipelined(service, wav_path, actual_dur)

        batch_analysis = analyze_text(batch["text"], joiner)
        pipe_analysis = analyze_text(pipe["text"], joiner)

        print_results(label, wav_path, actual_dur, batch, pipe,
                      batch_analysis, pipe_analysis)

    print(f"\n{SEP}")
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
