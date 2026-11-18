"""Local Qwen3-ASR backend for Voca desktop dictation."""

import os
from pathlib import Path
import subprocess

from voca.transcription.base import TranscriptionError, TranscriptionService


LANGUAGE_NAMES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "ru": "Russian", "ja": "Japanese",
    "ko": "Korean", "zh": "Chinese", "ar": "Arabic", "hi": "Hindi",
    "tr": "Turkish", "nl": "Dutch", "sv": "Swedish", "da": "Danish",
    "fi": "Finnish", "pl": "Polish",
}


class NativeQwenService(TranscriptionService):
    def __init__(self, model_size, backend, language, context):
        root = Path(__file__).resolve().parents[3] / "qwen-asr"
        runtime = Path(os.environ.get("VOCA_NATIVE_QWEN_RUNTIME", root))
        self.binary = runtime / ("qwen_asr_cuda" if backend == "native-cuda" else "qwen_asr")
        self.model = Path(os.environ.get(
            "VOCA_NATIVE_QWEN_MODEL", runtime / f"qwen3-asr-{model_size.lower()}"
        ))
        self.language = language
        self.context = context

    def _check_files(self):
        if not self.binary.is_file() or not os.access(self.binary, os.X_OK):
            raise TranscriptionError(f"Native Qwen executable unavailable: {self.binary}")
        if not (self.model / "vocab.json").is_file() or not any(self.model.glob("*.safetensors")):
            raise TranscriptionError(f"Native Qwen model files unavailable: {self.model}")

    def is_available(self):
        try:
            self._check_files()
        except TranscriptionError:
            return False
        return True

    def transcribe(self, audio_file_path):
        self._check_files()
        command = [str(self.binary), "-d", str(self.model), "-i", audio_file_path, "--silent"]
        if self.language and self.language != "auto":
            language = LANGUAGE_NAMES.get(self.language, self.language)
            command.extend(["--language", language])
        if self.context:
            command.extend(["--prompt", self.context])
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False)
        except OSError as error:
            raise TranscriptionError(f"Could not run native Qwen: {error}") from error
        if result.returncode:
            reason = result.stderr.strip() or f"exit status {result.returncode}"
            raise TranscriptionError(f"Native Qwen transcription failed: {reason}")
        return result.stdout.strip()
