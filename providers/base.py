"""
Abstract base class for LLM providers.
Adding a new provider = subclass this and implement `complete()`.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CompletionRequest:
    prompt: str
    system_prompt: Optional[str] = None
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 512


@dataclass
class CompletionResponse:
    text: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    raw: dict = field(default_factory=dict)


class BaseProvider(ABC):
    """All LLM providers implement this interface."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Send prompt to LLM, return structured response."""
        pass

    def health_check(self) -> bool:
        """Quick ping to verify the provider is reachable."""
        try:
            resp = self.complete(CompletionRequest(prompt="Say 'ok'", max_tokens=5))
            return bool(resp.text)
        except Exception:
            return False
