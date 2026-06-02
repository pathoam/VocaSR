# ASR Benchmark Report: Batch vs Pipelined Transcription

## Branch: `edge`

## What we built

This branch adds **pipelined segmented transcription** to VoiceBox's Qwen ASR backend, along with a **SegmentJoinStage** correction layer to fix the over-punctuation it introduces.

### Pipelined transcription (`qwen_asr.py`)

Instead of waiting until recording ends and running one large inference pass (batch mode), the pipeline:

1. Accumulates audio in a buffer during recording
2. When ~8s of audio is buffered, finds the longest pause via energy-based detection (`audio_segmenter.py`)
3. Cuts the segment at that pause and dispatches it to a background worker thread for inference
4. Repeats for each subsequent segment during recording
5. On recording end, dispatches the final (partial) segment

Because earlier segments are transcribed *during* recording, the user only waits for the **last segment's inference** after they stop talking. This bounds the post-recording wait to ~2.5s regardless of total recording length.

### SegmentJoinStage (`corrections.py`)

Segmented transcription introduces spurious sentence breaks at segment boundaries — the model sees each segment independently and tends to end/begin them as separate sentences. The 6-pass rule-based correction stage fixes:

1. Periods after non-terminal words (articles, prepositions, conjunctions)
2. Periods before continuation starters (prepositions, possessives, adverbs)
3. Periods/semicolons before coordinating conjunctions (". And" → ", and")
4. Bare pronouns before periods after commas
5. Relative pronouns after periods ("set up. That can" → "set up that can")
6. Short fragment merging (≤3 word fragments)

## Benchmark results

Test audio from `~/.config/VoiceBox/training_data/`, run on CPU (transformers backend, GTX 1070 Ti excluded by PyTorch CUDA capability check).

### 10s audio (2 segments)

| Metric | Batch | Pipelined |
|---|---|---|
| Total compute | 3.81s | 4.85s |
| Segments | 1 | 2 |
| Avg per segment | 3.81s | 2.42s |
| **User wait after recording** | **3.81s** | **2.42s** |
| **Speedup** | baseline | **1.6x** |

Short audio has only 2 segments, so the benefit is modest — one segment is processed during recording, the other after.

### 60s audio (8 segments)

| Metric | Batch | Pipelined |
|---|---|---|
| Total compute | 14.63s | 19.65s |
| Segments | 1 | 8 |
| Avg per segment | 14.63s | 2.46s |
| **User wait after recording** | **14.63s** | **2.46s** |
| **Speedup** | baseline | **6.0x** |

The pipeline's key property: **user wait is bounded to ~1 segment (~2.5s) no matter how long the recording**. Total compute is ~35% higher due to segment overlap and per-segment overhead, but that's invisible to the user since it happens during recording.

### Text quality

| Metric | Batch | Pipelined |
|---|---|---|
| Periods (raw) — 60s audio | 2 | 9 |
| Periods (corrected) | 1 | 5 |
| Corrections applied | 1 | 4 |

The pipeline produces ~4.5x more sentence breaks than batch on the same audio. SegmentJoinStage catches about half of the spurious ones (4 of ~7 extra periods removed). Remaining misses are mid-sentence splits where the continuation word is a common noun/adjective not in the known word lists (e.g., "this. Bar", "folder. Icon").

### Text comparison (60s sample)

**Batch (baseline):**
> Okay, looking at this bar, do you think we should take out the folder icon or replace it with something else? Because every team has a folder icon next to it, and I'm not sure if that's the best icon for it, but I also can't think of what I would replace it with i'm thinking, what if we just have a slightly different visual design, like a slightly different color for the agent integration.

**Pipelined (after corrections):**
> Okay, looking at this. Bar, do you think we should take out the folder. Icon or replace it with something else because every team has a folder icon next to it, and I'm not sure if that's the best icon for it, but I also can't think of what i would replace it with i'm thinking, what if we just have. A slightly different. Visual design, like a Slightly different color for the agent integration.

## Remaining work

- **SegmentJoinStage coverage**: The correction rules miss splits before common nouns/adjectives. Could add a "capitalized word after period that isn't a proper noun" heuristic, or use sentence probability scoring.
- **Segment boundary quality**: Better pause detection could reduce mid-word splits that cause transcription divergence ("courtyard" → "court grant").
