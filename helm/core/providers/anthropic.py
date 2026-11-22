"""
helm/core/providers/anthropic.py — Anthropic Claude LLM Provider adapter.

Interacts with the Anthropic Messages API over HTTP (using ANTHROPIC_API_KEY).
Translates standard MCP tool schemas into Anthropic's input_schema structure.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from helm.config.auth import resolve_api_key
from helm.core.providers.base import BaseLLMProvider
from helm.exceptions import AuthenticationError, ProviderAPIError

logger = logging.getLogger(__name__)

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"


def translate_tools_to_anthropic(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate standard MCP tool schemas into Anthropic's input_schema structure."""
    anthropic_tools: list[dict[str, Any]] = []
    for tool in tools:
        schema = (
            tool.get("input_schema")
            or tool.get("inputSchema")
            or tool.get("parameters")
            or {"type": "object"}
        )
        anthropic_tools.append(
            {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "input_schema": schema,
            }
        )
    return anthropic_tools


class AnthropicProvider(BaseLLMProvider):
    """Adapter for Anthropic Claude models."""

    provider_name: str = "anthropic"
    default_model: str = "claude-3-5-sonnet-latest"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_sec: float = 30.0,
        **kwargs: object,
    ) -> None:
        resolved_key = api_key or resolve_api_key("anthropic") or ""
        super().__init__(
            api_key=resolved_key,
            model=model or self.default_model,
            **kwargs,
        )
        self.base_url = (base_url or ANTHROPIC_API_URL).rstrip("/")
        self.timeout_sec = timeout_sec

    def _get_headers(self) -> dict[str, str]:
        if not self.api_key:
            raise AuthenticationError(
                "Anthropic API key is not configured. Run 'helmx login' or set ANTHROPIC_API_KEY."
            )
        return {
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> str:
        """Execute text generation request against Anthropic Messages API."""
        headers = self._get_headers()
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 4096,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
                resp = await client.post(self.base_url, json=payload, headers=headers)
                if resp.status_code in (401, 403) or "authentication_error" in resp.text:
                    raise AuthenticationError(
                        f"Anthropic authentication failed ({resp.status_code}): {resp.text}"
                    )
                resp.raise_for_status()
                data = resp.json()
        except (AuthenticationError, ProviderAPIError):
            raise
        except httpx.HTTPStatusError as exc:
            raise ProviderAPIError(
                f"Anthropic API returned HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except Exception as exc:
            raise ProviderAPIError(f"Anthropic connection failed: {exc}") from exc

        content_blocks = data.get("content", [])
        text_segments: list[str] = [
            block.get("text", "") for block in content_blocks if block.get("type") == "text"
        ]
        return "".join(text_segments).strip()

    async def generate_with_tools(
        self,
        prompt: str,
        system_prompt: str = "",
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[str, list[dict[str, Any]] | None]:
        """Execute generation turn with tools against Anthropic Messages API."""
        headers = self._get_headers()
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 4096,
            "temperature": 0.2,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            payload["system"] = system_prompt
        if tools:
            payload["tools"] = translate_tools_to_anthropic(tools)

        try:
            async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
                resp = await client.post(self.base_url, json=payload, headers=headers)
                if resp.status_code in (401, 403) or "authentication_error" in resp.text:
                    raise AuthenticationError(
                        f"Anthropic authentication failed ({resp.status_code}): {resp.text}"
                    )
                resp.raise_for_status()
                data = resp.json()
        except (AuthenticationError, ProviderAPIError):
            raise
        except httpx.HTTPStatusError as exc:
            raise ProviderAPIError(
                f"Anthropic API returned HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except Exception as exc:
            raise ProviderAPIError(f"Anthropic connection failed: {exc}") from exc

        content_blocks = data.get("content", [])
        text_segments: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        for block in content_blocks:
            b_type = block.get("type")
            if b_type == "text":
                text_segments.append(block.get("text", "").strip())
            elif b_type == "tool_use":
                name = block.get("name", "")
                raw_input = block.get("input", {})
                tool_calls.append(
                    {
                        "name": name,
                        "args": raw_input if isinstance(raw_input, dict) else {},
                    }
                )

        result_text = " ".join(text_segments).strip()
        return result_text, tool_calls or None
