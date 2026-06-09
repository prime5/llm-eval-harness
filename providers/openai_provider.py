"""OpenAI provider implementation."""
import json
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

        # Use pre-built history if provided (multi-turn), else build from prompt
        if request.messages is not None:
            messages = request.messages
        else:
            messages = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            messages.append({"role": "user", "content": request.prompt})

        kwargs = dict(
            model=model,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        if request.tools:
            kwargs["tools"] = request.tools
            kwargs["tool_choice"] = "auto"

        start = time.time()
        response = self._client.chat.completions.create(**kwargs)
        latency_ms = (time.time() - start) * 1000

        message = response.choices[0].message
        text = message.content or ""

        tool_calls = None
        if message.tool_calls:
            tool_calls = []
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, AttributeError):
                    args = {}
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": args,
                })

        return CompletionResponse(
            text=text.strip(),
            model=model,
            provider=self.name,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            latency_ms=latency_ms,
            raw=response.model_dump(),
            tool_calls=tool_calls,
        )
