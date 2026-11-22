"""
helm/core/providers/base.py — Base LLM Provider abstraction.

Defines the model-agnostic interface implemented by all concrete LLM provider
adapters (Google Gemini, Anthropic Claude, OpenAI).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseLLMProvider(ABC):
    """Abstract base class for all LLM provider adapters."""

    provider_name: str
    default_model: str

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        **kwargs: object,
    ) -> None:
        self.api_key = api_key or ""
        self.model = model or self.default_model
        self.extra_config = kwargs

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> str:
        """
        Generate a text response from the given prompt.
        """

    @abstractmethod
    async def generate_with_tools(
        self,
        prompt: str,
        system_prompt: str = "",
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[str, list[dict[str, Any]] | None]:
        """
        Generate completion with optional tool declarations.
        Returns a tuple of (response_text, list_of_tool_calls_or_None).
        Each tool call is a dictionary: {"name": str, "args": dict[str, Any]}.
        """
