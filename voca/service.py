"""Headless Voca service entrypoint.

This module intentionally avoids importing desktop integration modules such as
hotkeys, tray UI, and text insertion. It is the API-only path used by managed
ASR containers and other service deployments.
"""

from __future__ import annotations

import os
import signal
from typing import Iterable, TypeVar

from voca.config.manager import ConfigManager
from voca.text.vocabulary import VocabularyManager
from voca.transcription.qwen_asr import QwenASRService

T = TypeVar("T")


def _setup_signal_handlers() -> None:
    """Ensure Ctrl+C always exits, even during model downloads in threads."""
    first_interrupt = [True]

    def _handler(signum, frame):
        if first_interrupt[0]:
            first_interrupt[0] = False
            print("\nInterrupted. Press Ctrl+C again to force quit.")
            raise KeyboardInterrupt

        print("\nForce quitting...")
        os._exit(1)

    signal.signal(signal.SIGINT, _handler)


def _first_env(names: Iterable[str]) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None and value != "":
            return value
    return None


def _setting(
    config_manager: ConfigManager,
    env_names: Iterable[str],
    key: str,
    default: T,
    parser,
) -> T:
    raw = _first_env(env_names)
    if raw is None:
        return config_manager.get_setting(key, default)

    try:
        return parser(raw)
    except (TypeError, ValueError):
        return default


def init_qwen_service(config_manager: ConfigManager) -> QwenASRService:
    """Initialize a QwenASRService from config for the API server."""
    qwen_config = config_manager.get_qwen_config()
    language = config_manager.get_transcription_language()
    vocab = VocabularyManager(config_manager.config_dir)
    context = vocab.get_context_string()

    service = QwenASRService(
        model_size=qwen_config["model_size"],
        backend=qwen_config["backend"],
        language=language,
        kv_cache_mb=config_manager.get_setting("vllm_kv_cache_mb", 256),
        context=context,
        quantization=qwen_config.get("quantization"),
    )
    print("Loading Qwen ASR model (may download on first run)...")
    service._load_model()
    print("Qwen ASR model loaded.")
    return service


def start_api_server_only(config_manager: ConfigManager) -> None:
    from voca.api.server import configure, run_server

    service = init_qwen_service(config_manager)
    host = _setting(
        config_manager,
        ("VOCA_API_HOST", "VOICEBOX_API_HOST"),
        "api_host",
        "127.0.0.1",
        str,
    )
    port = _setting(
        config_manager,
        ("VOCA_API_PORT", "VOICEBOX_API_PORT"),
        "api_port",
        9876,
        int,
    )
    max_streams = _setting(
        config_manager,
        ("VOCA_API_MAX_STREAMS", "VOICEBOX_API_MAX_STREAMS"),
        "api_max_streams",
        8,
        int,
    )
    configure(service, max_streams=max_streams)
    print(f"Starting API server on http://{host}:{port}")
    run_server(host=host, port=port)


def main() -> None:
    _setup_signal_handlers()
    config_manager = ConfigManager()
    start_api_server_only(config_manager)


if __name__ == "__main__":
    main()
