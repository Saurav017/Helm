"""
helm/core/providers/gemini.py — Google Gemini LLM Provider adapter.

Interacts with the Google Gemini API over HTTP (using GEMINI_API_KEY).
Translates standard MCP tool schemas into Gemini functionDeclarations.
"""

from __future__ import annotations

import logging
from http import HTTPStatus
from typing import Any

import httpx

from helm.config.auth import resolve_api_key
from helm.core.providers.base import BaseLLMProvider
from helm.exceptions import AuthenticationError, ProviderAPIError

logger = logging.getLogger(__name__)


def translate_tools_to_gemini(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate standard MCP tool schemas into Gemini functionDeclarations."""
    declarations: list[dict[str, Any]] = []
    for tool in tools:
        # If already formatted as functionDeclarations
        if "functionDeclarations" in tool:
            declarations.extend(tool["functionDeclarations"])
            continue

        name = tool.get("name", "")
        description = tool.get("description", "")
        parameters = (
            tool.get("parameters")
            or tool.get("inputSchema")
            or tool.get("input_schema")
            or {"type": "object"}
        )
        declarations.append(
            {
                "name": name,
                "description": description,
                "parameters": parameters,
            }
        )
    return declarations


class GeminiProvider(BaseLLMProvider):
    """Adapter for Google Gemini models."""

    provider_name: str = "gemini"
    default_model: str = "gemini-3.8-flash"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout_sec: float = 30.0,
        **kwargs: object,
    ) -> None:
        resolved_key = api_key or resolve_api_key("gemini") or ""
        super().__init__(
            api_key=resolved_key,
            model=model or self.default_model,
            **kwargs,
        )
        self.timeout_sec = timeout_sec

    def _get_api_url(self, model: str | None = None) -> str:
        target_model = model or self.model
        if not self.api_key:
            raise AuthenticationError(
                "Gemini API key is not configured. Run 'helmx login' or set GEMINI_API_KEY."
            )
        return (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{target_model}:generateContent?key={self.api_key}"
        )

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> str:
        """Execute text generation request against Gemini."""
        url = self._get_api_url()

        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature},
        }
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        try:
            async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code in (HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN) or (
                    resp.status_code == HTTPStatus.BAD_REQUEST
                    and any(
                        k in resp.text
                        for k in (
                            "API_KEY_INVALID",
                            "API key not valid",
                            "INVALID_ARGUMENT",
                        )
                    )
                ):
                    raise AuthenticationError(
                        f"Gemini authentication failed ({resp.status_code}): {resp.text}"
                    )
                resp.raise_for_status()
                data = resp.json()
        except (AuthenticationError, ProviderAPIError):
            raise
        except httpx.HTTPStatusError as exc:
            raise ProviderAPIError(
                f"Gemini API returned error HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except Exception as exc:
            raise ProviderAPIError(f"Gemini connection failed: {exc}") from exc

        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            text_segments = [p.get("text", "") for p in parts if "text" in p]
            return "".join(text_segments).strip()
        return ""

    async def generate_with_tools(
        self,
        prompt: str,
        system_prompt: str = "",
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[str, list[dict[str, Any]] | None]:
        """Execute generation turn with function declarations against Gemini."""
        url = self._get_api_url()

        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2},
        }
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        if tools:
            formatted = translate_tools_to_gemini(tools)
            payload["tools"] = [{"functionDeclarations": formatted}]
            payload["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}

        try:
            async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code in (HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN) or (
                    resp.status_code == HTTPStatus.BAD_REQUEST
                    and any(
                        k in resp.text
                        for k in (
                            "API_KEY_INVALID",
                            "API key not valid",
                            "INVALID_ARGUMENT",
                        )
                    )
                ):
                    raise AuthenticationError(
                        f"Gemini authentication failed ({resp.status_code}): {resp.text}"
                    )
                resp.raise_for_status()
                data = resp.json()
        except (AuthenticationError, ProviderAPIError):
            raise
        except httpx.HTTPStatusError as exc:
            raise ProviderAPIError(
                f"Gemini API returned error HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except Exception as exc:
            raise ProviderAPIError(f"Gemini connection failed: {exc}") from exc

        candidates = data.get("candidates", [])
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            for p in parts:
                if "text" in p:
                    text_parts.append(str(p["text"]).strip())
                if "functionCall" in p:
                    fc = p["functionCall"]
                    tool_calls.append(
                        {
                            "name": str(fc.get("name", "")),
                            "args": dict(fc.get("args", {})),
                        }
                    )

        result_text = " ".join(text_parts).strip()
        return result_text, tool_calls or None
