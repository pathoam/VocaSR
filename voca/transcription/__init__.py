# Import backend modules to trigger @register_asr registration
import voca.transcription.local  # noqa: F401
import voca.transcription.api  # noqa: F401
from voca.transcription.registry import register_asr


@register_asr("qwen")
def _create_qwen_service(config_manager, vocab_context):
    qwen_config = config_manager.get_qwen_config()
    language = config_manager.get_transcription_language()
    if qwen_config["backend"] in {"native-cpu", "native-cuda"}:
        from voca.transcription.qwen_native import NativeQwenService

        return NativeQwenService(qwen_config["model_size"], qwen_config["backend"], language, vocab_context)
    from voca.transcription.qwen_asr import QwenASRService

    return QwenASRService(
        model_size=qwen_config["model_size"],
        backend=qwen_config["backend"],
        language=language,
        kv_cache_mb=config_manager.get_setting("vllm_kv_cache_mb", 256),
        context=vocab_context,
        quantization=qwen_config.get("quantization"),
    )
