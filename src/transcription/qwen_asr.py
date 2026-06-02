import queue
import threading
import time
import uuid
from typing import Optional, Dict, List

import numpy as np
import torch

from src.transcription.audio_segmenter import find_segment_boundary
from src.transcription.base import StreamingTranscriptionService, TranscriptionError
from src.utils.logging import get_logger


logger = get_logger(__name__)

# Default session ID used by the desktop app's single-session path
_DEFAULT_SESSION = "__default__"

# Map ISO 639-1 codes to the full names qwen-asr expects
_LANG_MAP = {
    "zh": "Chinese",
    "en": "English",
    "yue": "Cantonese",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "id": "Indonesian",
    "it": "Italian",
    "ko": "Korean",
    "ru": "Russian",
    "th": "Thai",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "tr": "Turkish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "fil": "Filipino",
    "fa": "Persian",
    "el": "Greek",
    "ro": "Romanian",
    "hu": "Hungarian",
    "mk": "Macedonian",
}


def _resolve_language(lang: str) -> Optional[str]:
    """Convert a language setting to the full name qwen-asr expects, or None for auto."""
    if not lang or lang == "auto":
        return None
    # Already a full name (e.g. "English")
    if lang in _LANG_MAP.values():
        return lang
    # ISO code lookup
    resolved = _LANG_MAP.get(lang.lower())
    if resolved:
        return resolved
    logger.warning(f"Unknown language code '{lang}', falling back to auto-detect")
    return None


class _TransformersStreamingASRModel:
    """Subclass wrapper that enables streaming ASR on the transformers backend.

    The upstream Qwen3ASRModel guards its streaming methods with a vLLM-only check.
    This wrapper overrides those three methods, replacing the vLLM generate call with
    the equivalent transformers generate path. Everything else (buffer management,
    prefix rollback, parse_asr_output) is delegated to the parent.

    Based on qwen-asr 0.0.6 — if the library updates its streaming logic,
    these overrides may need updating.
    """

    @staticmethod
    def from_pretrained(model_name: str, use_gpu: bool = False):
        """Load the model via the parent's from_pretrained, then patch streaming."""
        from qwen_asr import Qwen3ASRModel
        kwargs = {}
        if use_gpu:
            kwargs["device_map"] = "auto"
            kwargs["torch_dtype"] = torch.float16
            logger.info("Loading transformers model on GPU (device_map='auto', float16)")
        else:
            logger.info("Loading transformers model on CPU")
        model = Qwen3ASRModel.from_pretrained(model_name, **kwargs)
        # Attach our override methods to this specific instance
        import types
        model.init_streaming_state = types.MethodType(
            _TransformersStreamingASRModel._init_streaming_state, model)
        model.streaming_transcribe = types.MethodType(
            _TransformersStreamingASRModel._streaming_transcribe, model)
        model.finish_streaming_transcribe = types.MethodType(
            _TransformersStreamingASRModel._finish_streaming_transcribe, model)
        model._generate_for_streaming = types.MethodType(
            _TransformersStreamingASRModel._generate_for_streaming, model)
        model._transcribe_segment = types.MethodType(
            _TransformersStreamingASRModel._transcribe_segment, model)
        return model

    @staticmethod
    def _init_streaming_state(self, **kwargs):
        """Initialize streaming state by temporarily spoofing the backend flag."""
        original_backend = self.backend
        self.backend = "vllm"
        try:
            from qwen_asr.inference.qwen3_asr import Qwen3ASRModel
            # Call the original (unbound) method from the class
            return Qwen3ASRModel.init_streaming_state(self, **kwargs)
        finally:
            self.backend = original_backend

    @staticmethod
    @torch.no_grad()
    def _generate_for_streaming(self, prompt: str, audio_accum: np.ndarray) -> str:
        """Run transformers generate for a single streaming step."""
        inputs = self.processor(
            text=[prompt], audio=[audio_accum],
            return_tensors="pt", padding=True,
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        text_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        decoded = self.processor.batch_decode(
            text_ids.sequences[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0]

    @staticmethod
    @torch.no_grad()
    def _transcribe_segment(self, audio: np.ndarray, language: str = None) -> str:
        """Batch-transcribe a single audio segment, returning clean text.

        Used by the pipelined transcription engine — each ~8s segment is
        transcribed independently with a fresh prompt.
        """
        from qwen_asr.inference.utils import parse_asr_output

        msgs = [
            {"role": "system", "content": ""},
            {"role": "user", "content": [{"type": "audio", "audio": ""}]},
        ]
        prompt = self.processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False
        )
        inputs = self.processor(
            text=[prompt], audio=[audio],
            return_tensors="pt", padding=True,
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        text_ids = self.model.generate(**inputs, max_new_tokens=512)
        decoded = self.processor.batch_decode(
            text_ids.sequences[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        raw = decoded[0]
        # Always pass user_language=None so parse_asr_output parses the
        # "language English<asr_text>..." format from the raw output.
        _lang, text = parse_asr_output(raw, user_language=None)
        return text.strip()

    @staticmethod
    def _streaming_transcribe(self, pcm16k, state):
        """streaming_transcribe with transformers backend instead of vLLM.

        Copied from qwen-asr 0.0.6 Qwen3ASRModel.streaming_transcribe with
        the vLLM guard removed and generate call replaced.
        """
        from qwen_asr.inference.utils import parse_asr_output

        if state is None:
            raise ValueError("state must not be None. Call init_streaming_state() first.")
        if pcm16k is None:
            raise ValueError("pcm16k must not be None.")

        x = np.asarray(pcm16k)
        if x.ndim != 1:
            x = x.reshape(-1)
        if x.dtype == np.int16:
            x = x.astype(np.float32) / 32768.0
        else:
            x = x.astype(np.float32, copy=False)

        if x.shape[0] > 0:
            state.buffer = np.concatenate([state.buffer, x], axis=0)

        while state.buffer.shape[0] >= state.chunk_size_samples:
            chunk = state.buffer[:state.chunk_size_samples]
            state.buffer = state.buffer[state.chunk_size_samples:]

            if state.audio_accum.shape[0] == 0:
                state.audio_accum = chunk
            else:
                state.audio_accum = np.concatenate([state.audio_accum, chunk], axis=0)

            prefix = ""
            if state.chunk_id >= state.unfixed_chunk_num:
                cur_ids = self.processor.tokenizer.encode(state._raw_decoded)
                k = int(state.unfixed_token_num)
                while True:
                    end_idx = max(0, len(cur_ids) - k)
                    prefix = self.processor.tokenizer.decode(cur_ids[:end_idx]) if end_idx > 0 else ""
                    if '\ufffd' not in prefix:
                        break
                    if end_idx == 0:
                        prefix = ""
                        break
                    k += 1

            prompt = state.prompt_raw + prefix
            gen_text = self._generate_for_streaming(prompt, state.audio_accum)

            state._raw_decoded = (prefix + gen_text) if prefix is not None else gen_text
            lang, txt = parse_asr_output(state._raw_decoded, user_language=state.force_language)
            state.language = lang
            state.text = txt
            state.chunk_id += 1

        return state

    @staticmethod
    def _finish_streaming_transcribe(self, state):
        """finish_streaming_transcribe with transformers backend instead of vLLM.

        Copied from qwen-asr 0.0.6 Qwen3ASRModel.finish_streaming_transcribe with
        the vLLM guard removed and generate call replaced.
        """
        from qwen_asr.inference.utils import parse_asr_output

        if state is None:
            raise ValueError("state must not be None.")
        if state.buffer is None or state.buffer.shape[0] == 0:
            return state

        tail = state.buffer
        state.buffer = np.zeros((0,), dtype=np.float32)

        if state.audio_accum.shape[0] == 0:
            state.audio_accum = tail
        else:
            state.audio_accum = np.concatenate([state.audio_accum, tail], axis=0)

        prefix = ""
        if state.chunk_id >= state.unfixed_chunk_num:
            cur_ids = self.processor.tokenizer.encode(state._raw_decoded)
            end_idx = max(1, len(cur_ids) - int(state.unfixed_token_num))
            prefix = self.processor.tokenizer.decode(cur_ids[:end_idx])

        prompt = state.prompt_raw + prefix
        gen_text = self._generate_for_streaming(prompt, state.audio_accum)

        state._raw_decoded = (prefix + gen_text) if prefix is not None else gen_text
        lang, txt = parse_asr_output(state._raw_decoded, user_language=state.force_language)
        state.language = lang
        state.text = txt
        state.chunk_id += 1
        return state


class _PipelinedState:
    """State for a pipelined segmented transcription session.

    Audio accumulates in audio_buffer. When it reaches ~target_sec,
    a segment is cut at the longest pause and dispatched for transcription.
    Completed segments are stored by index for ordered concatenation.
    """

    def __init__(self, language=None, target_sec=8.0, search_sec=3.0, min_seg_sec=3.0):
        self.language = language
        self.target_sec = target_sec
        self.search_sec = search_sec
        self.min_seg_sec = min_seg_sec
        self.audio_buffer = np.zeros(0, dtype=np.float32)
        self.next_seg_index = 0
        self.completed_segments: Dict[int, str] = {}
        self.partial_text = ""


class QwenGPUBackend:
    """GPU backend using the Python qwen-asr package (Qwen3ASRModel)."""

    _SENTINEL = object()  # poison pill for worker threads

    def __init__(self, model_size: str = "0.6B", language: str = "auto",
                 kv_cache_mb: int = 256, context: Optional[str] = None):
        self.model_size = model_size
        self.language = _resolve_language(language)
        self.kv_cache_mb = kv_cache_mb
        self.context = context
        self.model = None
        self._is_vllm = False
        self._is_gpu = False
        self._lock = threading.Lock()
        # Multi-session streaming state
        self._sessions: Dict[str, object] = {}
        self._session_locks: Dict[str, threading.Lock] = {}
        self._sessions_lock = threading.Lock()
        # Async inference: audio queues and worker threads per session
        self._audio_queues: Dict[str, queue.Queue] = {}
        self._workers: Dict[str, threading.Thread] = {}

    @staticmethod
    def is_available() -> bool:
        try:
            import qwen_asr  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def _probe_gpu_viable() -> tuple:
        """Check whether CUDA works on this system for any backend.

        Returns (viable: bool, reason: str). When viable is False, reason
        explains why so the caller can skip GPU entirely.
        """
        try:
            import torch as _torch
        except ImportError:
            return False, "torch is not installed"

        if not _torch.cuda.is_available():
            return False, "CUDA is not available"

        try:
            major, minor = _torch.cuda.get_device_capability(0)
        except Exception as e:
            return False, f"could not query GPU capability: {e}"

        name = _torch.cuda.get_device_properties(0).name

        # Check if the installed PyTorch actually has kernels for this GPU
        try:
            t = _torch.randn(2, 2, device="cuda", dtype=_torch.float32)
            _ = t + t
            del t
        except Exception as e:
            return False, (
                f"GPU {name} (sm_{major}{minor}) is not supported by "
                f"installed PyTorch {_torch.__version__}: {e}"
            )

        return True, f"GPU {name} (sm_{major}{minor})"

    @staticmethod
    def _probe_vllm_viable(gpu_viable: bool) -> tuple:
        """Check whether vLLM can plausibly run on this system.

        Returns (viable: bool, reason: str).
        """
        if not gpu_viable:
            return False, "GPU not viable"

        import importlib.util
        if importlib.util.find_spec("vllm") is None:
            return False, "vllm package is not installed"

        import torch as _torch
        major, minor = _torch.cuda.get_device_capability(0)
        if (major, minor) < (7, 0):
            name = _torch.cuda.get_device_properties(0).name
            return False, (
                f"GPU {name} has compute capability {major}.{minor}, "
                f"vLLM requires >= 7.0 (Volta or newer)"
            )

        return True, "ok"

    def load_model(self) -> None:
        if self.model is not None:
            return
        from qwen_asr import Qwen3ASRModel

        model_name = f"Qwen/Qwen3-ASR-{self.model_size}"

        gpu_viable, gpu_reason = self._probe_gpu_viable()
        if gpu_viable:
            logger.info(f"GPU available: {gpu_reason}")
        else:
            logger.info(f"GPU not usable: {gpu_reason}")

        vllm_viable, vllm_reason = self._probe_vllm_viable(gpu_viable)
        if not vllm_viable:
            logger.info(f"Skipping vLLM backend: {vllm_reason}")

        if vllm_viable:
            try:
                # vLLM requires max_model_len to fit in KV cache. We let vLLM
                # compute the real per-token cost (which accounts for all layers
                # including encoder-only attention) and just set a conservative
                # max_model_len. 13 tokens/s of audio, so kv_cache_mb → minutes:
                #   256MB ≈ 3 min, 2048MB ≈ 24 min
                kv_bytes_per_token = 2 * 28 * 8 * 128 * 2  # 114688 (approximate)
                max_tokens = int((self.kv_cache_mb * 1024 * 1024) // kv_bytes_per_token * 0.95)
                self.model = Qwen3ASRModel.LLM(
                    model_name,
                    skip_mm_profiling=True,
                    kv_cache_memory_bytes=self.kv_cache_mb * 1024 * 1024,
                    max_model_len=max_tokens,
                )
                self._is_vllm = True
                self._is_gpu = True
                logger.info(f"Loaded Qwen ASR with vLLM backend: {model_name}")
                return
            except Exception as e:
                logger.warning(f"vLLM backend failed ({e}), falling back to from_pretrained")

        self.model = _TransformersStreamingASRModel.from_pretrained(model_name, use_gpu=gpu_viable)
        self._is_vllm = False
        self._is_gpu = gpu_viable
        device = "GPU" if gpu_viable else "CPU"
        logger.info(f"Loaded Qwen ASR with transformers backend on {device}: {model_name}")

    def set_context(self, context: Optional[str]) -> None:
        """Update the vocabulary biasing context string."""
        self.context = context

    def _strip_context_prefix(self, text: str, context: Optional[str]) -> str:
        """Remove the context prompt if the model echoed it back in the output."""
        if not context or not text:
            return text
        # The model sometimes echoes the context string at the start of the output
        if text.startswith(context):
            text = text[len(context):].lstrip()
        # Also handle partial echoes — the context ends with a period, so
        # check for a sentence that looks like the prompt
        prefix = "The following terms may appear in the audio:"
        if text.startswith(prefix):
            # Find the end of the echoed context sentence
            dot_idx = text.find(".", len(prefix))
            if dot_idx != -1:
                text = text[dot_idx + 1:].lstrip()
        return text

    def transcribe_file(self, path: str, context: Optional[str] = None) -> str:
        self.load_model()
        ctx = context if context is not None else self.context
        kwargs = {"language": self.language}
        if ctx:
            kwargs["context"] = ctx
        results = self.model.transcribe(path, **kwargs)
        # results is a list of ASRTranscription objects
        if results:
            texts = []
            for r in results:
                text = getattr(r, 'text', None) or str(r)
                if text.strip():
                    texts.append(text.strip())
            output = " ".join(texts)
            return self._strip_context_prefix(output, ctx)
        return ""

    def supports_streaming(self) -> bool:
        # Pipelined segmented transcription works on both backends.
        # For vLLM: uses upstream streaming (partial results).
        # For transformers: uses pipelined segmentation (~8s segments
        # transcribed during recording, 4-5x latency reduction).
        return self.model is not None

    def _get_session_lock(self, session_id: str) -> threading.Lock:
        with self._sessions_lock:
            if session_id not in self._session_locks:
                self._session_locks[session_id] = threading.Lock()
            return self._session_locks[session_id]

    # ---------------------------------------------------------------
    # Pipelined segmented transcription (transformers backend)
    # ---------------------------------------------------------------

    def _start_pipelined(self, session_id: str) -> str:
        """Start a pipelined segmented transcription session."""
        state = _PipelinedState(language=self.language)
        segment_queue = queue.Queue()
        with self._sessions_lock:
            self._sessions[session_id] = state
            self._session_locks[session_id] = threading.Lock()
            self._audio_queues[session_id] = segment_queue
        worker = threading.Thread(
            target=self._pipelined_worker, args=(session_id,), daemon=True)
        self._workers[session_id] = worker
        worker.start()
        return session_id

    def _pipelined_worker(self, session_id: str) -> None:
        """Background thread that transcribes segments as they arrive."""
        _SENTINEL = self._SENTINEL
        while True:
            try:
                item = self._audio_queues[session_id].get(timeout=0.1)
            except queue.Empty:
                continue
            if item is _SENTINEL:
                break
            seg_index, audio_segment = item
            try:
                t0 = time.time()
                text = self.model._transcribe_segment(audio_segment, language=self.language)
                elapsed = time.time() - t0
                dur = len(audio_segment) / 16000
                logger.debug(
                    f"Segment {seg_index}: {dur:.1f}s audio → {elapsed:.1f}s inference → '{text[:60]}...'"
                )
            except Exception as e:
                logger.error(f"Segment {seg_index} transcription failed: {e}")
                text = ""
            with self._sessions_lock:
                state = self._sessions.get(session_id)
            if state is None:
                break
            lock = self._get_session_lock(session_id)
            with lock:
                state.completed_segments[seg_index] = text
                # Update partial result: concatenate all completed segments in order
                parts = []
                for i in range(state.next_seg_index):
                    if i in state.completed_segments:
                        parts.append(state.completed_segments[i])
                state.partial_text = " ".join(p for p in parts if p)

    def _feed_pipelined(self, chunk: np.ndarray, session_id: str) -> None:
        """Accumulate audio and dispatch segments at pause boundaries."""
        with self._sessions_lock:
            state = self._sessions.get(session_id)
        if state is None or not isinstance(state, _PipelinedState):
            return

        audio = chunk.flatten().astype(np.float32, copy=False)
        lock = self._get_session_lock(session_id)
        with lock:
            state.audio_buffer = np.concatenate([state.audio_buffer, audio])

        buffer_sec = len(state.audio_buffer) / 16000

        # Only look for a segment boundary once we have enough audio for the
        # search window to meaningfully cover the target region. Starting too
        # early (e.g. at target - search) leaves only a tiny sliver of frames
        # to search, producing cuts at non-pauses.
        if buffer_sec >= state.target_sec:
            cut_sample, pause_ms = find_segment_boundary(
                state.audio_buffer, 16000,
                target_sec=state.target_sec,
                search_sec=state.search_sec,
            )
            # Only cut if we found a point within the buffer (not at the end)
            # and the segment is at least min_seg_sec
            min_samples = int(state.min_seg_sec * 16000)
            if cut_sample < len(state.audio_buffer) and cut_sample >= min_samples:
                with lock:
                    segment = state.audio_buffer[:cut_sample].copy()
                    state.audio_buffer = state.audio_buffer[cut_sample:]
                    seg_index = state.next_seg_index
                    state.next_seg_index += 1

                seg_dur = len(segment) / 16000
                logger.debug(
                    f"Dispatching segment {seg_index}: {seg_dur:.1f}s "
                    f"(pause {pause_ms:.0f}ms)"
                )
                q = self._audio_queues.get(session_id)
                if q is not None:
                    q.put((seg_index, segment))

    def _finish_pipelined(self, session_id: str) -> str:
        """Transcribe final segment, wait for all segments, concatenate."""
        with self._sessions_lock:
            state = self._sessions.get(session_id)
        if state is None or not isinstance(state, _PipelinedState):
            return ""

        # Dispatch remaining audio as the final segment
        lock = self._get_session_lock(session_id)
        with lock:
            remaining = state.audio_buffer.copy()
            state.audio_buffer = np.zeros(0, dtype=np.float32)

        if len(remaining) > 0:
            remaining_sec = len(remaining) / 16000
            # Only transcribe if there's meaningful audio (>0.5s)
            if remaining_sec > 0.5:
                seg_index = state.next_seg_index
                with lock:
                    state.next_seg_index += 1
                logger.debug(f"Dispatching final segment {seg_index}: {remaining_sec:.1f}s")
                q = self._audio_queues.get(session_id)
                if q is not None:
                    q.put((seg_index, remaining))

        # Signal worker to finish and wait
        q = self._audio_queues.get(session_id)
        if q is not None:
            q.put(self._SENTINEL)
        worker = self._workers.pop(session_id, None)
        if worker is not None:
            worker.join()

        # Concatenate all segments in order
        with self._sessions_lock:
            state = self._sessions.get(session_id)
        if state is None:
            return ""

        parts = []
        for i in range(state.next_seg_index):
            text = state.completed_segments.get(i, "")
            if text:
                parts.append(text)
        result = " ".join(parts).strip()

        n_segs = state.next_seg_index
        logger.info(f"Pipelined transcription: {n_segs} segments → {len(result)} chars")

        # Clean up
        with self._sessions_lock:
            self._sessions.pop(session_id, None)
            self._session_locks.pop(session_id, None)
            self._audio_queues.pop(session_id, None)

        return self._strip_context_prefix(result, self.context)

    # ---------------------------------------------------------------
    # Upstream streaming (vLLM backend)
    # ---------------------------------------------------------------

    def _start_upstream_streaming(self, session_id: str, context: Optional[str] = None) -> str:
        """Start an upstream vLLM streaming session."""
        ctx = context if context is not None else self.context
        kwargs = {"language": self.language}
        if ctx:
            kwargs["context"] = ctx
        state = self.model.init_streaming_state(**kwargs)
        audio_queue = queue.Queue()
        with self._sessions_lock:
            self._sessions[session_id] = state
            self._session_locks[session_id] = threading.Lock()
            self._audio_queues[session_id] = audio_queue
        worker = threading.Thread(
            target=self._upstream_inference_worker, args=(session_id,), daemon=True)
        self._workers[session_id] = worker
        worker.start()
        return session_id

    def _upstream_inference_worker(self, session_id: str) -> None:
        """Background thread for upstream vLLM streaming inference."""
        _SENTINEL = self._SENTINEL
        while True:
            try:
                pcm = self._audio_queues[session_id].get(timeout=0.1)
            except queue.Empty:
                continue
            if pcm is _SENTINEL:
                break
            lock = self._get_session_lock(session_id)
            with lock:
                with self._sessions_lock:
                    state = self._sessions.get(session_id)
                if state is None:
                    break
                new_state = self.model.streaming_transcribe(pcm, state)
                with self._sessions_lock:
                    self._sessions[session_id] = new_state

    def _feed_upstream(self, chunk: np.ndarray, session_id: str) -> None:
        """Queue audio for upstream vLLM streaming."""
        q = self._audio_queues.get(session_id)
        if q is None:
            return
        pcm = (chunk.flatten() * 32767).astype(np.int16)
        q.put(pcm)

    def _finish_upstream(self, session_id: str) -> str:
        """Finish upstream vLLM streaming session."""
        q = self._audio_queues.get(session_id)
        if q is not None:
            q.put(self._SENTINEL)
        worker = self._workers.pop(session_id, None)
        if worker is not None:
            worker.join()
        with self._sessions_lock:
            state = self._sessions.get(session_id)
        if state is None:
            return ""
        lock = self._get_session_lock(session_id)
        with lock:
            state = self.model.finish_streaming_transcribe(state)
        result = getattr(state, 'text', '') or ''
        result = result.strip()
        with self._sessions_lock:
            self._sessions.pop(session_id, None)
            self._session_locks.pop(session_id, None)
            self._audio_queues.pop(session_id, None)
        return self._strip_context_prefix(result, self.context)

    # ---------------------------------------------------------------
    # Unified streaming API (dispatches to pipelined or upstream)
    # ---------------------------------------------------------------

    def start_streaming(self, session_id: str = _DEFAULT_SESSION, context: Optional[str] = None) -> str:
        self.load_model()
        if self._is_vllm:
            return self._start_upstream_streaming(session_id, context)
        else:
            return self._start_pipelined(session_id)

    def feed_chunk(self, chunk: np.ndarray, session_id: str = _DEFAULT_SESSION) -> None:
        """Queue audio for async inference — returns immediately, never blocks on model."""
        if self._is_vllm:
            self._feed_upstream(chunk, session_id)
        else:
            self._feed_pipelined(chunk, session_id)

    def finish_streaming(self, session_id: str = _DEFAULT_SESSION) -> str:
        if self._is_vllm:
            return self._finish_upstream(session_id)
        else:
            return self._finish_pipelined(session_id)

    def get_partial_result(self, session_id: str = _DEFAULT_SESSION) -> Optional[str]:
        with self._sessions_lock:
            state = self._sessions.get(session_id)
        if state is None:
            return None
        if isinstance(state, _PipelinedState):
            partial = state.partial_text
            if not partial:
                return None
            return self._strip_context_prefix(partial, self.context)
        # Upstream streaming state
        partial = getattr(state, 'text', '') or ''
        partial = partial.strip()
        if not partial:
            return None
        return self._strip_context_prefix(partial, self.context)

    def get_max_recording_seconds(self) -> Optional[float]:
        """Return the max safe recording duration in seconds, or None if no limit."""
        if not self._is_vllm:
            return None
        kv_bytes_per_token = 2 * 28 * 8 * 128 * 2  # 114688
        max_tokens = int((self.kv_cache_mb * 1024 * 1024) // kv_bytes_per_token * 0.95)
        tokens_per_sec = 13
        return max_tokens / tokens_per_sec * 0.95

    def cleanup_session(self, session_id: str) -> None:
        q = self._audio_queues.pop(session_id, None)
        if q is not None:
            q.put(self._SENTINEL)
        worker = self._workers.pop(session_id, None)
        if worker is not None:
            worker.join(timeout=5)
        with self._sessions_lock:
            self._sessions.pop(session_id, None)
            self._session_locks.pop(session_id, None)


class QwenASRService(StreamingTranscriptionService):
    """Qwen ASR transcription service (GPU backend)."""

    def __init__(
        self,
        model_size: str = "0.6B",
        backend: str = "auto",
        language: str = "auto",
        kv_cache_mb: int = 256,
        context: Optional[str] = None,
    ):
        self.model_size = model_size
        self.backend_preference = backend
        self.language = language
        self.kv_cache_mb = kv_cache_mb
        self.context = context
        self._backend: Optional[QwenGPUBackend] = None
        self._loaded = False

    def set_context(self, context: Optional[str]) -> None:
        """Update the vocabulary biasing context string."""
        self.context = context
        if self._backend:
            self._backend.set_context(context)

    def _load_model(self) -> None:
        """Lazy-initialize the GPU backend."""
        if self._loaded:
            return

        if not QwenGPUBackend.is_available():
            raise TranscriptionError(
                "Qwen ASR not available. Install with: pip install qwen-asr"
            )
        self._backend = QwenGPUBackend(self.model_size, self.language,
                                       kv_cache_mb=self.kv_cache_mb,
                                       context=self.context)
        self._backend.load_model()
        self._loaded = True

    def transcribe(self, audio_file_path: str) -> str:
        self._load_model()
        return self._backend.transcribe_file(audio_file_path)

    def is_available(self) -> bool:
        return QwenGPUBackend.is_available()

    # -- Backward-compatible single-session API (used by desktop app) --

    def start_streaming(self) -> None:
        self._load_model()
        self._backend.start_streaming(_DEFAULT_SESSION)

    def feed_chunk(self, audio_chunk: np.ndarray) -> None:
        if self._backend:
            self._backend.feed_chunk(audio_chunk, _DEFAULT_SESSION)

    def finish_streaming(self) -> str:
        if not self._backend:
            return ""
        return self._backend.finish_streaming(_DEFAULT_SESSION)

    def supports_streaming(self) -> bool:
        if not self._loaded:
            try:
                self._load_model()
            except TranscriptionError:
                return False
        return self._backend.supports_streaming() if self._backend else False

    def get_partial_result(self, session_id: str = None) -> Optional[str]:
        if not self._backend:
            return None
        sid = session_id if session_id else _DEFAULT_SESSION
        return self._backend.get_partial_result(sid)

    # -- Multi-session API (used by the HTTP/WS server) --

    def start_streaming_session(self, language: str = None) -> str:
        """Start a new streaming session, returning its unique session ID."""
        self._load_model()
        session_id = str(uuid.uuid4())[:8]
        if language and self._backend._is_vllm:
            # vLLM path: per-session language via init_streaming_state
            resolved = _resolve_language(language)
            state = self._backend.model.init_streaming_state(language=resolved)
            with self._backend._sessions_lock:
                self._backend._sessions[session_id] = state
                self._backend._session_locks[session_id] = threading.Lock()
        else:
            # Pipelined or default vLLM path — full setup with queue/worker
            self._backend.start_streaming(session_id)
        return session_id

    def feed_chunk_session(self, session_id: str, audio_chunk: np.ndarray) -> None:
        """Feed audio to a specific streaming session."""
        if self._backend:
            self._backend.feed_chunk(audio_chunk, session_id)

    def finish_streaming_session(self, session_id: str) -> str:
        """Finalize a specific streaming session and return text."""
        if not self._backend:
            return ""
        return self._backend.finish_streaming(session_id)

    def cleanup_session(self, session_id: str) -> None:
        """Clean up a streaming session without finalizing."""
        if self._backend:
            self._backend.cleanup_session(session_id)

    def get_max_recording_seconds(self) -> Optional[float]:
        """Return the max safe recording duration in seconds, or None if no limit."""
        if not self._backend:
            return None
        return self._backend.get_max_recording_seconds()

    def active_session_count(self) -> int:
        """Return number of active streaming sessions."""
        if not self._backend:
            return 0
        with self._backend._sessions_lock:
            return len(self._backend._sessions)
