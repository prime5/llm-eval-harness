"""Anthropic (Claude) provider implementation."""
import time
from config.settings import ANTHROPIC_API_KEY
from .base import BaseProvider, CompletionRequest, CompletionResponse

DEFAULT_CLAUDE_MODEL = "claude-haiku-4-5-20251001"


def _to_anthropic_tools(tools: list) -> list:
    """Translate OpenAI-style tool definitions to Anthropic format."""
    result = []
    for tool in tools:
        if tool.get("type") == "function":
            fn = tool["function"]
            result.append({
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        else:
            result.append(tool)
    return result


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

        # Use pre-built history if provided (multi-turn), else build from prompt
        if request.messages is not None:
            messages = request.messages
        else:
            messages = [{"role": "user", "content": request.prompt}]

        kwargs = dict(
            model=model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            messages=messages,
        )
        if request.system_prompt:
            kwargs["system"] = request.system_prompt
        if request.tools:
            kwargs["tools"] = _to_anthropic_tools(request.tools)

        start = time.time()
        response = self._client.messages.create(**kwargs)
        latency_ms = (time.time() - start) * 1000

        text = ""
        tool_calls = None
        tool_use_blocks = []

        for block in response.content:
            if hasattr(block, "text"):
                text += block.text
            elif getattr(block, "type", None) == "tool_use":
                tool_use_blocks.append(block)

        if tool_use_blocks:
            tool_calls = [
                {"id": b.id, "name": b.name, "arguments": b.input}
                for b in tool_use_blocks
            ]

        return CompletionResponse(
            text=text.strip(),
            model=model,
            provider=self.name,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
            raw=response.model_dump(),
            tool_calls=tool_calls,
        )
