"""OpenRouter LLM backend."""

import requests
from typing import Any, Dict, List, Optional

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

_NO_SYSTEM_USER_PREFIX = (
    "Give ONLY the direct answer with no preamble or repetition of the question. "
    "Do not say 'Here is', 'The answer is', or repeat any part of the question "
    "- just give the raw answer content: "
)


class OpenRouterBackend(LLMBackend):
    """LLM backend that queries the OpenRouter API."""

    def __init__(self, api_key: str, model: str = "meta-llama/llama-3.2-3b-instruct:free",
                 timeout: int = 30):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.no_system_models: set = set()

    def is_available(self) -> bool:
        return bool(self.api_key)

    def set_api_key(self, api_key: str) -> None:
        self.api_key = api_key

    def set_model(self, model: str) -> None:
        self.model = model

    # -----------------------------------------------------------------
    # Text queries
    # -----------------------------------------------------------------

    def query(self, command: str) -> Dict[str, Any]:
        if not self.api_key:
            return {"success": False, "error": "No OpenRouter API key configured"}
        if self.model in self.no_system_models:
            return self._query_no_system(command)
        return self._query_with_system(command)

    def _query_with_system(self, command: str) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": command},
        ]
        return self._send(messages, retry_no_system=True, original_command=command)

    def _query_no_system(self, command: str) -> Dict[str, Any]:
        messages = [{"role": "user", "content": _NO_SYSTEM_USER_PREFIX + command}]
        return self._send(messages)

    # -----------------------------------------------------------------
    # Multimodal queries (clipboard)
    # -----------------------------------------------------------------

    def query_multimodal(self, command: str, clipboard_data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.api_key:
            return {"success": False, "error": "No OpenRouter API key configured"}

        use_system = self.model not in self.no_system_models

        if clipboard_data["type"] == "image":
            content = self._build_multimodal_content(command, clipboard_data, use_system)
            messages = [{"role": "user", "content": content}]
        elif clipboard_data["type"] == "text":
            enhanced = f"{command}\n\nClipboard content:\n{clipboard_data['content']}"
            if use_system:
                messages = [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": enhanced},
                ]
            else:
                messages = [{"role": "user", "content": _NO_SYSTEM_USER_PREFIX + enhanced}]
        else:
            return self.query(command)

        return self._send(messages, retry_no_system=use_system, original_command=command,
                          clipboard_data=clipboard_data)

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _send(self, messages: list, *, retry_no_system: bool = False,
              original_command: str = "", clipboard_data: Dict[str, Any] = None) -> Dict[str, Any]:
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 200,
                "temperature": 0.7,
            }
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": "https://github.com/voca/voca",
                    "X-Title": "Voca",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )

            if resp.status_code == 404 and clipboard_data and clipboard_data.get("type") == "image":
                return {"success": False, "error": f"Model {self.model} doesn't support image input"}

            if resp.status_code == 200:
                return self._extract_content(resp.json())

            # Handle errors
            try:
                error_data = resp.json()
                error_msg = error_data.get("error", {}).get("message", "Unknown error")
                # Check for system-message rejection
                if retry_no_system and self._is_system_rejection(error_data):
                    self.no_system_models.add(self.model)
                    if clipboard_data:
                        return self.query_multimodal(original_command, clipboard_data)
                    return self._query_no_system(original_command)
            except Exception:
                error_msg = f"HTTP {resp.status_code}: {resp.text}"
            return {"success": False, "error": f"OpenRouter error: {error_msg}"}

        except requests.RequestException as e:
            return {"success": False, "error": f"OpenRouter request failed: {e}"}

    @staticmethod
    def _extract_content(data: dict) -> Dict[str, Any]:
        message = data.get("choices", [{}])[0].get("message", {})
        content = message.get("content", "")
        if not content or not content.strip():
            reasoning = message.get("reasoning", "")
            if reasoning and reasoning.strip():
                content = reasoning
        if not content or not content.strip():
            return {"success": False, "error": "Empty response from OpenRouter"}
        return {"success": True, "response": content}

    @staticmethod
    def _is_system_rejection(error_data: dict) -> bool:
        error_msg = error_data.get("error", {}).get("message", "").lower()
        raw_error = str(error_data.get("error", {}).get("metadata", {}).get("raw", "")).lower()
        return (
            "instruction" in error_msg or "system" in error_msg
            or "developer instruction" in raw_error
            or "instruction is not enabled" in raw_error
        )

    @staticmethod
    def _build_multimodal_content(command: str, clipboard_data: Dict[str, Any],
                                  use_system: bool) -> List[Dict[str, Any]]:
        import base64
        import io
        from PIL import Image

        img = clipboard_data["content"]

        max_size = 1024
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        if img.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
            img = background

        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=95, optimize=True)
        img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        prompt = command if use_system else (_NO_SYSTEM_USER_PREFIX + command)
        return [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
        ]
