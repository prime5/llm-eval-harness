"""Anthropic (Claude) provider implementation."""
import time
from config.settings import ANTHROPIC_API_KEY
from .base import BaseProvider, CompletionRequest, CompletionResponse

DEFAULT_CLAUDE_MODEL = "claude-haiku-4-5-20251001"


class AnthropicProvider(BaseProvider):

    def __init__(self):
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

    @property
    def name(self) -> str:
        return "anthropic"

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        model = request.model or DEFAULT_CLAUDE_MODEL
        kwargs = dict(
            model=model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            messages=[{"role": "user", "content": request.prompt}],
        )
        if request.system_prompt:
            kwargs["system"] = request.system_prompt

        start = time.time()
        response = self._client.messages.create(**kwargs)
        latency_ms = (time.time() - start) * 1000

        return CompletionResponse(
            text=response.content[0].text.strip(),
            model=model,
            provider=self.name,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
            raw=response.model_dump(),
        )
