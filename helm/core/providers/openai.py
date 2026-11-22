"""
helm/core/providers/openai.py — OpenAI LLM Provider adapter.

Interacts with OpenAI-compatible chat completion endpoints (using OPENAI_API_KEY).
Supports custom base_url for enterprise proxies, LiteLLM, or local models (e.g. Ollama/vLLM).
Translates standard MCP tool schemas into OpenAI function calling structures.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx

from helm.config.auth import load_helm_config, resolve_api_key
from helm.core.prompts import PARAMETER_CORRECTION_INSTRUCTIONS
from helm.core.providers.base import BaseLLMProvider
from helm.exceptions import AuthenticationError, ProviderAPIError

logger = logging.getLogger(__name__)

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
_MAX_ARGS_REPR_LEN = 40


def build_tools_system_instruction(tools: list[dict[str, Any]]) -> str:
    """
    Format available tool signatures and anti-hallucination / anti-python instructions
    into a system prompt block.
    """
    categories: dict[str, list[str]] = {}
    for t in tools:
        fn = t.get("function", t) if t.get("type") == "function" else t
        t_name = fn.get("name", "")
        if not t_name:
            continue
        params = fn.get("parameters") or fn.get("inputSchema") or fn.get("input_schema") or {}
        props: list[str] = []
        if isinstance(params, dict):
            props = list(params.get("properties", {}).keys())

        args_repr = ", ".join(props) if props else ""
        if len(args_repr) > _MAX_ARGS_REPR_LEN:
            args_repr = "..."
        sig = f"`{t_name}({args_repr})`"

        # Determine category / server
        server = t.get("server_name") or fn.get("server_name")
        if not server:
            if any(k in t_name for k in ("file", "directory", "dir", "path")):
                server = "filesystem"
            elif any(k in t_name for k in ("graph", "node", "entit")):
                server = "memory"
            else:
                server = "tools"
        categories.setdefault(str(server), []).append(sig)

    lines = ["You have access to the following tools via tool-calling:"]
    for cat, sigs in categories.items():
        lines.append(f" - {cat}: {', '.join(sigs)}")
    lines.append(
        "Do NOT write Python code to execute these tools. "
        "Always trigger them via native tool/function calls."
    )
    lines.append(PARAMETER_CORRECTION_INSTRUCTIONS)
    return "\n".join(lines)


def translate_tools_to_openai(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate standard MCP tool schemas into strict OpenAI function calling format."""
    openai_tools: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") == "function" and "function" in tool:
            fn = dict(tool["function"])
            raw_params = fn.get("parameters") or fn.get("inputSchema") or fn.get("input_schema")
            if not isinstance(raw_params, dict):
                params = {"type": "object", "properties": {}}
            else:
                params = dict(raw_params)
                if "type" not in params:
                    params["type"] = "object"
                if "properties" not in params or not isinstance(params["properties"], dict):
                    params["properties"] = {}
            fn["parameters"] = params
            openai_tools.append({"type": "function", "function": fn})
            continue

        name = tool.get("name", "")
        description = tool.get("description", "")
        raw_params = tool.get("parameters") or tool.get("inputSchema") or tool.get("input_schema")
        if not isinstance(raw_params, dict):
            parameters = {"type": "object", "properties": {}}
        else:
            parameters = dict(raw_params)
            if "type" not in parameters:
                parameters["type"] = "object"
            if "properties" not in parameters or not isinstance(parameters["properties"], dict):
                parameters["properties"] = {}

        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            }
        )
    return openai_tools


class OpenAIProvider(BaseLLMProvider):
    """Adapter for OpenAI models and OpenAI-compatible proxies (LiteLLM, vLLM, etc.)."""

    provider_name: str = "openai"
    default_model: str = "gpt-4o"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_sec: float = 30.0,
        **kwargs: object,
    ) -> None:
        resolved_key = api_key or resolve_api_key("openai") or ""
        super().__init__(
            api_key=resolved_key,
            model=model or self.default_model,
            **kwargs,
        )
        # Check base_url priority: explicit param -> env var -> config.json -> default
        cfg = load_helm_config()
        resolved_base = (
            base_url
            or os.environ.get("OPENAI_BASE_URL")
            or cfg.get("openai_base_url")
            or DEFAULT_OPENAI_BASE_URL
        )
        self.base_url = str(resolved_base).rstrip("/")
        self.timeout_sec = timeout_sec

    def _get_headers(self) -> dict[str, str]:
        if not self.api_key:
            raise AuthenticationError(
                "OpenAI API key is not configured. Run 'helmx login' or set OPENAI_API_KEY."
            )
        return {
            "Authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> str:
        """Execute chat completion request against OpenAI /chat/completions."""
        headers = self._get_headers()
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        url = f"{self.base_url}/chat/completions"

        try:
            async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
                resp = await client.post(url, json=payload, headers=headers)
                if resp.status_code in (401, 403) or "invalid_api_key" in resp.text:
                    raise AuthenticationError(
                        f"OpenAI authentication failed ({resp.status_code}): {resp.text}"
                    )
                resp.raise_for_status()
                data = resp.json()
        except (AuthenticationError, ProviderAPIError):
            raise
        except httpx.HTTPStatusError as exc:
            raise ProviderAPIError(
                f"OpenAI API returned HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except Exception as exc:
            raise ProviderAPIError(f"OpenAI connection failed: {exc}") from exc

        choices = data.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            return str(msg.get("content", "") or "").strip()
        return ""

    async def generate_with_tools(
        self,
        prompt: str,
        system_prompt: str = "",
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[str, list[dict[str, Any]] | None]:
        """Execute chat completion with tools against OpenAI /chat/completions."""
        headers = self._get_headers()
        messages: list[dict[str, str]] = []
        augmented_system = system_prompt
        if tools:
            tool_instr = build_tools_system_instruction(tools)
            if augmented_system:
                augmented_system = f"{augmented_system}\n\n{tool_instr}"
            else:
                augmented_system = tool_instr

        if augmented_system:
            messages.append({"role": "system", "content": augmented_system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
        }
        if tools:
            payload["tools"] = translate_tools_to_openai(tools)
            payload["tool_choice"] = "auto"

        url = f"{self.base_url}/chat/completions"

        try:
            async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
                resp = await client.post(url, json=payload, headers=headers)
                if resp.status_code in (401, 403) or "invalid_api_key" in resp.text:
                    raise AuthenticationError(
                        f"OpenAI authentication failed ({resp.status_code}): {resp.text}"
                    )
                resp.raise_for_status()
                data = resp.json()
        except (AuthenticationError, ProviderAPIError):
            raise
        except httpx.HTTPStatusError as exc:
            raise ProviderAPIError(
                f"OpenAI API returned HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except Exception as exc:
            raise ProviderAPIError(f"OpenAI connection failed: {exc}") from exc

        choices = data.get("choices", [])
        if not choices:
            return "", None

        message = choices[0].get("message", {})
        return self._parse_tool_calls(message)

    @staticmethod
    def _parse_tool_calls(
        message: dict[str, Any],
    ) -> tuple[str, list[dict[str, Any]] | None]:
        """Parse raw message tool_calls structure into standardized tool call dictionaries."""
        result_text = str(message.get("content", "") or "").strip()
        raw_calls = message.get("tool_calls", [])
        if not raw_calls:
            return result_text, None

        tool_calls: list[dict[str, Any]] = []
        for rc in raw_calls:
            fn = rc.get("function", {})
            fn_name = fn.get("name", "")
            raw_args = fn.get("arguments", "{}")
            if isinstance(raw_args, str):
                try:
                    parsed_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    parsed_args = {"raw_arguments": raw_args}
            elif isinstance(raw_args, dict):
                parsed_args = raw_args
            else:
                parsed_args = {}

            tool_calls.append({"name": fn_name, "args": parsed_args})

        return result_text, tool_calls or None
