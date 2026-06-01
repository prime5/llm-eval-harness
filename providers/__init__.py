from .base import BaseProvider
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider


def get_provider(name: str) -> BaseProvider:
    """Factory: return provider instance by name."""
    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
    }
    if name not in providers:
        raise ValueError(f"Unknown provider '{name}'. Choose from: {list(providers)}")
    return providers[name]()
