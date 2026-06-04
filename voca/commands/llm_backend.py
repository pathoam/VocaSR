"""Abstract base class for LLM backends."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class LLMBackend(ABC):
    """Interface for querying an LLM (text or multimodal)."""

    @abstractmethod
    def query(self, command: str) -> Dict[str, Any]:
        """Send a text query and return {"success": bool, "response": str, "error": str?}."""

    @abstractmethod
    def query_multimodal(self, command: str, clipboard_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send a query with clipboard context (text or image)."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this backend is configured and reachable."""
