"""Transformers-backend streaming patch for Qwen3ASRModel.

The upstream Qwen3ASRModel guards its streaming methods with a vLLM-only check.
This module patches those methods to use the transformers generate path instead,
enabling streaming ASR on CPU/GPU without vLLM.

Based on qwen-asr 0.0.6 — if the library updates its streaming logic,
these overrides may need updating.
"""

from typing import Optional

import numpy as np
import torch

from voca.utils.logging import get_logger

logger = get_logger(__name__)

# Map ISO 639-1 codes to the full names qwen-asr expects
LANG_MAP = {
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


def resolve_language(lang: str) -> Optional[str]:
    """Convert a language setting to the full name qwen-asr expects, or None for auto."""
    if not lang or lang == "auto":
        return None
    if lang in LANG_MAP.values():
        return lang
    resolved = LANG_MAP.get(lang.lower())
    if resolved:
        return resolved
    logger.warning(f"Unknown language code '{lang}', falling back to auto-detect")
    return None


class TransformersStreamingASRModel:
    """Patches a Qwen3ASRModel instance to enable streaming on transformers backend."""

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
            TransformersStreamingASRModel._init_streaming_state, model)
        model.streaming_transcribe = types.MethodType(
            TransformersStreamingASRModel._streaming_transcribe, model)
        model.finish_streaming_transcribe = types.MethodType(
            TransformersStreamingASRModel._finish_streaming_transcribe, model)
        model._generate_for_streaming = types.MethodType(
            TransformersStreamingASRModel._generate_for_streaming, model)
        model._transcribe_segment = types.MethodType(
            TransformersStreamingASRModel._transcribe_segment, model)
        return model

    @staticmethod
    def _init_streaming_state(self, **kwargs):
        """Initialize streaming state by temporarily spoofing the backend flag."""
        original_backend = self.backend
        self.backend = "vllm"
        try:
            from qwen_asr.inference.qwen3_asr import Qwen3ASRModel
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
        _lang, text = parse_asr_output(raw, user_language=None)
        return text.strip()

    @staticmethod
    def _streaming_transcribe(self, pcm16k, state):
        """streaming_transcribe with transformers backend instead of vLLM."""
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
        """finish_streaming_transcribe with transformers backend instead of vLLM."""
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
