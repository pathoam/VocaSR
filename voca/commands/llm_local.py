"""Local LLM backend (vLLM, Ollama, etc.)."""

import requests
from typing import Any, Dict

from voca.commands.llm_backend import LLMBackend
from voca.utils.logging import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are a helpful assistant. Give ONLY the direct answer with absolutely "
    "NO preamble, introduction, or repetition of the question. Do not say "
    "'Here is', 'Here are', 'The answer is', 'The image shows', or ANY "
    "prefacing text. Start immediately with the raw answer content only. "
    "Use only plain text with no formatting, LaTeX, markdown, asterisks, "
    "or special symbols. Be extremely brief and direct."
)


class LocalLLMBackend(LLMBackend):
    """LLM backend that queries a local OpenAI-compatible or Ollama endpoint."""

    def __init__(self, endpoint: str, timeout: int = 30):
        self.endpoint = endpoint
        self.timeout = timeout

    def is_available(self) -> bool:
        return bool(self.endpoint)

    def set_endpoint(self, endpoint: str) -> None:
        self.endpoint = endpoint

    def query(self, command: str) -> Dict[str, Any]:
        if not self.endpoint:
            return {"success": False, "error": "No local LLM endpoint configured"}
        try:
            # Try OpenAI-compatible endpoint
            resp = requests.post(
                f"{self.endpoint}/v1/chat/completions",
                json={
                    "model": "default",
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": command},
                    ],
                    "max_tokens": 200,
                    "temperature": 0.7,
                },
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                if not content or not content.strip():
                    return {"success": False, "error": "Empty response from local LLM"}
                return {"success": True, "response": content}

            # Try Ollama format
            resp = requests.post(
                f"{self.endpoint}/api/generate",
                json={"model": "llama3.2", "prompt": command, "stream": False},
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return {"success": True, "response": resp.json().get("response", "")}

        except requests.RequestException as e:
            return {"success": False, "error": f"Local LLM error: {e}"}
        return {"success": False, "error": "Failed to query local LLM"}

    def query_multimodal(self, command: str, clipboard_data: Dict[str, Any]) -> Dict[str, Any]:
        if clipboard_data["type"] == "text":
            enhanced = f"{command}\n\nClipboard content:\n{clipboard_data['content']}"
            return self.query(enhanced)
        elif clipboard_data["type"] == "image":
            return {"success": False, "error": "Local LLM doesn't support image input"}
        return self.query(command)
