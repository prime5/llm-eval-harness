"""OpenAI provider implementation."""
import time
from config.settings import OPENAI_API_KEY, DEFAULT_MODEL
from .base import BaseProvider, CompletionRequest, CompletionResponse


class OpenAIProvider(BaseProvider):

    def __init__(self):
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=OPENAI_API_KEY)
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

    @property
    def name(self) -> str:
        return "openai"

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        model = request.model or DEFAULT_MODEL
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        start = time.time()
        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        latency_ms = (time.time() - start) * 1000

        return CompletionResponse(
            text=response.choices[0].message.content.strip(),
            model=model,
            provider=self.name,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            latency_ms=latency_ms,
            raw=response.model_dump(),
        )
