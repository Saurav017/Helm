"""
helm/core/providers/__init__.py — Model-agnostic LLM Provider gateway.
"""

from __future__ import annotations

from helm.core.providers.anthropic import AnthropicProvider, translate_tools_to_anthropic
from helm.core.providers.base import BaseLLMProvider
from helm.core.providers.gemini import GeminiProvider, translate_tools_to_gemini
from helm.core.providers.openai import OpenAIProvider, translate_tools_to_openai
from helm.exceptions import ConfigurationError


def create_provider(
    provider_name: str,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    timeout_sec: float = 30.0,
    **kwargs: object,
) -> BaseLLMProvider:
    """Factory creating a concrete BaseLLMProvider adapter by provider name."""
    norm = provider_name.strip().lower()
    if norm in ("gemini", "google"):
        return GeminiProvider(
            api_key=api_key,
            model=model,
            timeout_sec=timeout_sec,
            **kwargs,
        )
    if norm == "anthropic":
        return AnthropicProvider(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout_sec=timeout_sec,
            **kwargs,
        )
    if norm == "openai":
        return OpenAIProvider(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout_sec=timeout_sec,
            **kwargs,
        )
    raise ConfigurationError(
        f"Unsupported LLM provider: '{provider_name}'. Supported: 'gemini', 'anthropic', 'openai'."
    )


__all__ = [
    "AnthropicProvider",
    "BaseLLMProvider",
    "GeminiProvider",
    "OpenAIProvider",
    "create_provider",
    "translate_tools_to_anthropic",
    "translate_tools_to_gemini",
    "translate_tools_to_openai",
]
