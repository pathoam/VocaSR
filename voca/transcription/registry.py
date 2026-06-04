"""ASR backend registry — register and create transcription services by mode name."""

from typing import Callable, Dict, List, Optional

from voca.transcription.base import TranscriptionService
from voca.config.manager import ConfigManager


ASRFactory = Callable[[ConfigManager, Optional[str]], TranscriptionService]
_registry: Dict[str, ASRFactory] = {}


def register_asr(mode: str):
    """Decorator to register an ASR factory function under the given mode name."""
    def decorator(fn: ASRFactory) -> ASRFactory:
        _registry[mode] = fn
        return fn
    return decorator


def create_asr_service(mode: str, config_manager: ConfigManager,
                       vocab_context: Optional[str] = None) -> TranscriptionService:
    """Create an ASR service by mode name.

    Raises KeyError if the mode is not registered.
    """
    factory = _registry[mode]
    return factory(config_manager, vocab_context)


def available_modes() -> List[str]:
    """Return registered ASR mode names."""
    return list(_registry.keys())
