"""Command processing with LLM integration."""

from typing import Optional, Dict, Any, List
from datetime import datetime

from voca.commands.llm_backend import LLMBackend
from voca.commands.llm_local import LocalLLMBackend
from voca.commands.llm_openrouter import OpenRouterBackend
from voca.utils.logging import get_logger


class CommandProcessor:
    """Processes commands using built-in handlers or LLM."""

    def __init__(self,
                 openrouter_api_key: Optional[str] = None,
                 local_llm_endpoint: Optional[str] = None,
                 model: str = "meta-llama/llama-3.2-3b-instruct:free"):
        self.timeout = 30
        self.logger = get_logger(__name__)

        # Build backend list (local first, then OpenRouter)
        self._local = LocalLLMBackend(local_llm_endpoint or "", timeout=self.timeout)
        self._openrouter = OpenRouterBackend(openrouter_api_key or "", model=model, timeout=self.timeout)
        self.backends: List[LLMBackend] = [self._local, self._openrouter]

        # External references set by VocaApp for vocabulary/correction commands
        self.vocabulary_manager = None
        self.text_inserter = None
        self.voca_app = None

        # Built-in command handlers
        self.built_in_commands = {
            "time": self._handle_time,
            "date": self._handle_date,
            "help": self._handle_help,
            "add context": self._handle_add_context,
            "remove context": self._handle_remove_context,
            "clear context": self._handle_clear_context,
            "show context": self._handle_show_context,
            "fix": self._handle_fix,
        }

    # -----------------------------------------------------------------
    # Public API (unchanged signatures)
    # -----------------------------------------------------------------

    def process(self, command: str) -> Dict[str, Any]:
        if not command:
            return {"success": False, "error": "No command provided"}

        # Check built-in commands
        for keyword, handler in self.built_in_commands.items():
            if command.lower().startswith(keyword):
                return handler(command)

        return self._query_backends(command)

    def process_with_clipboard(self, command: str, clipboard_data: Dict[str, Any]) -> Dict[str, Any]:
        if not command:
            return {"success": False, "error": "No command provided"}
        return self._query_backends_multimodal(command, clipboard_data)

    # -----------------------------------------------------------------
    # Config update helpers (called by reload_config)
    # -----------------------------------------------------------------

    def set_openrouter_key(self, api_key: str) -> None:
        self._openrouter.set_api_key(api_key)

    def set_local_endpoint(self, endpoint: str) -> None:
        self._local.set_endpoint(endpoint)

    def set_model(self, model: str) -> None:
        self._openrouter.set_model(model)

    # Backwards-compat properties
    @property
    def openrouter_api_key(self):
        return self._openrouter.api_key

    @property
    def local_llm_endpoint(self):
        return self._local.endpoint

    @property
    def model(self):
        return self._openrouter.model

    # -----------------------------------------------------------------
    # Backend dispatch
    # -----------------------------------------------------------------

    def _query_backends(self, command: str) -> Dict[str, Any]:
        """Try each backend in priority order."""
        for backend in self.backends:
            if backend.is_available():
                result = backend.query(command)
                if result["success"]:
                    return result
        return {"success": False, "error": "No LLM backend available or configured."}

    def _query_backends_multimodal(self, command: str, clipboard_data: Dict[str, Any]) -> Dict[str, Any]:
        """Try each backend in priority order for multimodal queries."""
        for backend in self.backends:
            if backend.is_available():
                result = backend.query_multimodal(command, clipboard_data)
                if result["success"]:
                    return result
        return {"success": False, "error": "No LLM backend available or configured."}

    # -----------------------------------------------------------------
    # Built-in command handlers
    # -----------------------------------------------------------------

    def _handle_time(self, command: str) -> Dict[str, Any]:
        current_time = datetime.now().strftime("%I:%M %p")
        return {"success": True, "response": f"The current time is {current_time}"}

    def _handle_date(self, command: str) -> Dict[str, Any]:
        current_date = datetime.now().strftime("%B %d, %Y")
        return {"success": True, "response": f"Today is {current_date}"}

    def _handle_help(self, command: str) -> Dict[str, Any]:
        help_text = """Available commands:
\u2022 time - Get current time
\u2022 date - Get current date
\u2022 add context - Add selected text to vocabulary (select text first)
\u2022 remove context - Remove selected text from vocabulary
\u2022 clear context - Clear all vocabulary terms
\u2022 show context - Show current vocabulary terms
\u2022 fix - Correct last transcription
\u2022 Any other request - Processed by AI assistant

Examples:
"voca, what's the weather?"
"voca, add context" (with text selected)
"voca, show context"
"""
        return {"success": True, "response": help_text}

    def _handle_add_context(self, command: str) -> Dict[str, Any]:
        from voca.utils.notify import notify
        if not self.vocabulary_manager:
            return {"success": False, "error": "Vocabulary manager not available"}
        try:
            import time as _time
            if self.text_inserter:
                from pynput.keyboard import Key, Controller
                kb = Controller()
                kb.press(Key.ctrl)
                kb.press('c')
                kb.release('c')
                kb.release(Key.ctrl)
                _time.sleep(0.2)

            import pyperclip
            term = pyperclip.paste().strip()
            if not term:
                notify("Voca", "No text selected to add")
                return {"success": True, "response": ""}

            if self.vocabulary_manager.add_term(term):
                notify("Voca", f"Added '{term}' to vocabulary")
            else:
                notify("Voca", f"'{term}' is already in vocabulary")
            return {"success": True, "response": ""}
        except Exception as e:
            return {"success": False, "error": f"Failed to add context: {e}"}

    def _handle_remove_context(self, command: str) -> Dict[str, Any]:
        from voca.utils.notify import notify
        if not self.vocabulary_manager:
            return {"success": False, "error": "Vocabulary manager not available"}
        try:
            import time as _time
            if self.text_inserter:
                from pynput.keyboard import Key, Controller
                kb = Controller()
                kb.press(Key.ctrl)
                kb.press('c')
                kb.release('c')
                kb.release(Key.ctrl)
                _time.sleep(0.2)

            import pyperclip
            term = pyperclip.paste().strip()
            if not term:
                notify("Voca", "No text selected to remove")
                return {"success": True, "response": ""}

            if self.vocabulary_manager.remove_term(term):
                notify("Voca", f"Removed '{term}' from vocabulary")
            else:
                notify("Voca", f"'{term}' not found in vocabulary")
            return {"success": True, "response": ""}
        except Exception as e:
            return {"success": False, "error": f"Failed to remove context: {e}"}

    def _handle_clear_context(self, command: str) -> Dict[str, Any]:
        from voca.utils.notify import notify
        if not self.vocabulary_manager:
            return {"success": False, "error": "Vocabulary manager not available"}
        self.vocabulary_manager.clear()
        notify("Voca", "Vocabulary cleared")
        return {"success": True, "response": ""}

    def _handle_show_context(self, command: str) -> Dict[str, Any]:
        from voca.utils.notify import notify
        if not self.vocabulary_manager:
            return {"success": False, "error": "Vocabulary manager not available"}
        terms = self.vocabulary_manager.get_terms()
        if terms:
            terms_str = ", ".join(terms)
            notify("Voca", f"Vocabulary ({len(terms)} terms): {terms_str}")
        else:
            notify("Voca", "Vocabulary is empty")
        return {"success": True, "response": ""}

    def _handle_fix(self, command: str) -> Dict[str, Any]:
        if self.voca_app and hasattr(self.voca_app, '_on_correction_hotkey'):
            self.voca_app._on_correction_hotkey()
            return {"success": True, "response": ""}
        return {"success": True, "response": "No recent transcription to correct"}
