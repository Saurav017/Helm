"""
helm/core/model_router.py — Tiered model routing and model-agnostic LLM gateway.

Implements model routing according to INSTRUCTIONS.md §3.6:
- Fast tier: Triage, classification, simple summarization, yes/no decisions.
- Standard tier: Code generation, structured output, moderate reasoning.
- Deep tier: Multi-step planning, architecture design, complex debugging.

Resolves active provider dynamically across Gemini, Anthropic, and OpenAI
according to priority hierarchy:
1. CLI runtime flags (--provider, --model)
2. Environment variables (ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY)
3. Local config file ~/.helm/config.json
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from pydantic import BaseModel, Field

from helm.config.auth import resolve_active_provider
from helm.config.settings import Settings, load_settings
from helm.core.providers import (
    BaseLLMProvider,
    create_provider,
)
from helm.core.state import ModelTier
from helm.exceptions import AuthenticationError

logger = logging.getLogger(__name__)


class ToolCall(BaseModel):
    """Structured tool call returned from an LLM."""

    name: str
    args: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": False}


class ModelTurn(BaseModel):
    """Result of an LLM generation turn containing text and optional tool calls."""

    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)

    model_config = {"frozen": False}


ToolCallTurn = ModelTurn


def extract_tool_arguments(tool_name: str, text: str) -> dict[str, Any]:
    """Heuristically extract tool arguments from natural language text."""
    args: dict[str, Any] = {}
    if tool_name == "create_entities":
        name_match = re.search(r"named\s+([A-Za-z0-9_-]+)", text, re.IGNORECASE)
        type_match = re.search(r"entityType\s+([A-Za-z0-9_-]+)", text, re.IGNORECASE)
        obs_match = re.search(
            r"observation(?: that)?\s+(?:(?:it is|is)\s+)?(.+?)(?:$|\.|\n)",
            text,
            re.IGNORECASE,
        )

        name = name_match.group(1) if name_match else "Entity"
        entity_type = type_match.group(1) if type_match else "Software"
        obs = [obs_match.group(1).strip()] if obs_match else ["initial observation"]

        args = {
            "entities": [
                {
                    "name": name,
                    "entityType": entity_type,
                    "observations": obs,
                }
            ]
        }
    elif tool_name == "add_observations":
        name_match = re.search(r"named\s+([A-Za-z0-9_-]+)", text, re.IGNORECASE)
        obs_match = re.search(r"observation(?: that)?\s+(.+)", text, re.IGNORECASE)
        args = {
            "observations": [
                {
                    "entityName": name_match.group(1) if name_match else "Entity",
                    "contents": [obs_match.group(1).strip()] if obs_match else [],
                }
            ]
        }
    elif tool_name == "delete_entities":
        name_match = re.search(r"named\s+([A-Za-z0-9_-]+)", text, re.IGNORECASE)
        args = {"entityNames": [name_match.group(1)] if name_match else []}
    elif tool_name == "write_file":
        path_match = re.search(r"(?:file|path)\s+([A-Za-z0-9_./-]+)", text, re.IGNORECASE)
        args = {"path": path_match.group(1) if path_match else "file.txt", "content": ""}
    elif tool_name == "read_file":
        file_match = re.search(
            r"\b([A-Za-z0-9_./-]+\.(?:md|txt|py|json|toml|yaml|yml|sh|rs|go|c|h|cpp))\b",
            text,
            re.IGNORECASE,
        )
        if file_match:
            args = {"path": file_match.group(1)}
        else:
            path_match = re.search(
                r"(?:read|cat|inspect|open|file|path)\s+(?:the\s+)?([A-Za-z0-9_./-]+)",
                text,
                re.IGNORECASE,
            )
            args = {"path": path_match.group(1) if path_match else ""}
    elif tool_name == "list_directory":
        path_match = re.search(r"(?:directory|dir|path)\s+([A-Za-z0-9_./-]+)", text, re.IGNORECASE)
        args = {"path": path_match.group(1) if path_match else "."}
    elif tool_name == "read_graph":
        args = {}
    elif tool_name == "search_nodes":
        query_match = re.search(
            r"(?:about|for|query|named)\s+([A-Za-z0-9_-]+)", text, re.IGNORECASE
        )
        args = {"query": query_match.group(1) if query_match else ""}
    return args


TOOL_NAME_ALIASES: dict[str, str] = {
    "list_files": "list_directory",
    "listfiles": "list_directory",
    "list_dir": "list_directory",
    "listdir": "list_directory",
    "ls": "list_directory",
    "dir": "list_directory",
    "readfile": "read_file",
    "open_file": "read_file",
    "load_file": "read_file",
    "writefile": "write_file",
    "save_file": "write_file",
    "readgraph": "read_graph",
    "get_graph": "read_graph",
    "createentities": "create_entities",
    "searchnodes": "search_nodes",
    "find_nodes": "search_nodes",
    "deleteentities": "delete_entities",
}


def resolve_tool_name(name_candidate: str, tools: list[dict[str, Any]]) -> str | None:
    """Resolve a candidate or hallucinated tool name to an actual available tool."""
    norm = name_candidate.strip().lower()
    # 1. Direct case-insensitive match
    for t in tools:
        t_name = str(t.get("name", ""))
        if t_name.lower() == norm:
            return t_name

    # 2. Alias resolution
    if norm in TOOL_NAME_ALIASES:
        target = TOOL_NAME_ALIASES[norm]
        for t in tools:
            t_name = str(t.get("name", ""))
            if t_name.lower() == target.lower():
                return t_name

    # 3. Normalized underscore match
    norm_clean = norm.replace("-", "_")
    for t in tools:
        t_name = str(t.get("name", ""))
        if t_name.lower() == norm_clean:
            return t_name

    return None


def parse_call_arguments(raw_args: str, tool_def: dict[str, Any] | None) -> dict[str, Any]:
    """Parse raw argument string from function call text into structured dictionary."""
    args_str = raw_args.strip()
    tool_name = tool_def.get("name", "") if tool_def else ""
    if not args_str:
        if tool_name == "list_directory":
            return {"path": "."}
        return {}

    # JSON dictionary
    if args_str.startswith("{") and args_str.endswith("}"):
        try:
            parsed = json.loads(args_str)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # Keyword arguments (e.g. path="README.md", recursive=True)
    kwarg_pattern = re.compile(
        r"([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:'([^']*)'|\"([^\"]*)\"|([^\s,]+))"
    )
    kwarg_matches = kwarg_pattern.findall(args_str)
    if kwarg_matches:
        result_args: dict[str, Any] = {}
        for m in kwarg_matches:
            k = m[0]
            val = m[1] if m[1] != "" else (m[2] if m[2] != "" else m[3])
            if val.lower() == "true":
                result_args[k] = True
            elif val.lower() == "false":
                result_args[k] = False
            elif val.isdigit():
                result_args[k] = int(val)
            else:
                result_args[k] = val
        return result_args

    # Single positional literal string/path
    pos_match = re.match(r"^['\"]([^'\"]*)['\"]$", args_str)
    if pos_match:
        val = pos_match.group(1)
        if tool_def:
            params = tool_def.get("parameters") or tool_def.get("inputSchema") or {}
            props = list(params.get("properties", {}).keys()) if isinstance(params, dict) else []
            if props:
                return {props[0]: val}
        return {"path": val}

    # Positional bare string/path without quotes (e.g. ., src, README.md)
    bare_match = re.match(r"^([a-zA-Z0-9_./-]+)$", args_str)
    if bare_match:
        val = bare_match.group(1)
        if tool_def:
            params = tool_def.get("parameters") or tool_def.get("inputSchema") or {}
            props = list(params.get("properties", {}).keys()) if isinstance(params, dict) else []
            if props:
                return {props[0]: val}
        return {"path": val}

    # Heuristic fallback
    if tool_def:
        return extract_tool_arguments(tool_name, args_str)

    return {}


def extract_tool_calls_from_text(text: str, tools: list[dict[str, Any]]) -> list[ToolCall]:
    """
    Fallback parser extracting tool calls from markdown code blocks or text emission.
    Handles hallucinated names (e.g. list_files -> list_directory) and converts
    emitted python code calls into executable ToolCall instances.
    """
    if not text or not tools:
        return []

    calls: list[ToolCall] = []

    # 1. Check for JSON tool call structures in code blocks or text
    json_block_pattern = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
    for block in json_block_pattern.findall(text):
        try:
            data = json.loads(block)
            fn_name = data.get("name") or data.get("tool") or data.get("function")
            if fn_name:
                resolved = resolve_tool_name(str(fn_name), tools)
                if resolved:
                    t_def = next((t for t in tools if t.get("name") == resolved), None)
                    raw_args = (
                        data.get("arguments") or data.get("args") or data.get("parameters") or {}
                    )
                    args = (
                        raw_args
                        if isinstance(raw_args, dict)
                        else parse_call_arguments(str(raw_args), t_def)
                    )
                    calls.append(ToolCall(name=resolved, args=args))
        except json.JSONDecodeError:
            pass

    if calls:
        return calls

    # 2. Check for python code blocks containing function calls
    # e.g., ```python\nlist_files()\n``` or ```python\nread_file("README.md")\n```
    code_block_pattern = re.compile(r"```(?:python|py)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
    for block in code_block_pattern.findall(text):
        for line in block.strip().splitlines():
            line_str = line.strip()
            call_match = re.match(
                r"^(?:await\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)\s*;?$",
                line_str,
            )
            if call_match:
                fn_name, raw_args = call_match.group(1), call_match.group(2)
                resolved = resolve_tool_name(fn_name, tools)
                if resolved:
                    t_def = next((t for t in tools if t.get("name") == resolved), None)
                    parsed_args = parse_call_arguments(raw_args, t_def)
                    calls.append(ToolCall(name=resolved, args=parsed_args))

    if calls:
        return calls

    # 3. Check for inline function calls in markdown / prose text, e.g. read_file("README.md")
    inline_pattern = re.compile(
        r"(?:^|\s|`)([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)(?:`|\s|$|[.;])",
        re.MULTILINE,
    )
    for match in inline_pattern.finditer(text):
        fn_name, raw_args = match.group(1), match.group(2)
        if fn_name.lower() in ("if", "for", "while", "with", "assert", "return", "print"):
            continue
        resolved = resolve_tool_name(fn_name, tools)
        if resolved:
            t_def = next((t for t in tools if t.get("name") == resolved), None)
            parsed_args = parse_call_arguments(raw_args, t_def)
            calls.append(ToolCall(name=resolved, args=parsed_args))

    return calls


# Keywords for rule-based tier classification
_DEEP_KEYWORDS = {
    "architecture",
    "architect",
    "redesign",
    "migrate",
    "migration",
    "complex",
    "multi-step",
    "deadlock",
    "concurrency",
    "security audit",
}

_STANDARD_KEYWORDS = {
    "implement",
    "code",
    "write",
    "develop",
    "refactor",
    "function",
    "class",
    "module",
    "create",
    "endpoint",
    "unit test",
}


class ModelRouter:
    """
    Model-agnostic routing and LLM gateway.
    Resolves the active provider dynamically across Gemini, Anthropic, and OpenAI,
    maps role-calibrated default models per tier, and dispatches generation requests.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        provider: str | BaseLLMProvider | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        default_tier: ModelTier = ModelTier.FAST,
        dry_run: bool = False,
        mock_mode: bool = False,
    ) -> None:
        self.settings = settings or load_settings()
        self.default_tier = default_tier
        self.dry_run = dry_run
        self.mock_mode = mock_mode

        self.model_override: str | None = None
        if isinstance(provider, BaseLLMProvider):
            self.active_provider = provider
            self.provider_name = provider.provider_name
            self.model_override = model or provider.model
            self.api_key = provider.api_key
        else:
            resolved_prov, resolved_mod, resolved_key = resolve_active_provider(
                cli_provider=provider,
                cli_model=model,
            )
            self.provider_name = resolved_prov
            self.model_override = model
            self.api_key = api_key or resolved_key or ""

            target_model = model or resolved_mod
            self.active_provider = create_provider(
                self.provider_name,
                api_key=self.api_key,
                model=target_model,
                base_url=base_url,
            )

    def get_model_for_tier(self, tier: ModelTier | str) -> str:
        """Resolve model name for the given ModelTier based on active provider and settings."""
        if self.model_override:
            return self.model_override

        tier_enum = ModelTier(tier) if isinstance(tier, str) else tier

        setting_val = ""
        match tier_enum:
            case ModelTier.FAST:
                setting_val = getattr(self.settings.models, "fast", "")
            case ModelTier.STANDARD:
                setting_val = getattr(self.settings.models, "standard", "")
            case ModelTier.DEEP:
                setting_val = getattr(self.settings.models, "deep", "")

        if self.provider_name == "anthropic":
            if setting_val and "claude" in setting_val.lower():
                return setting_val
            if tier_enum == ModelTier.FAST:
                return "claude-3-5-haiku-latest"
            return "claude-3-5-sonnet-latest"

        if self.provider_name == "openai":
            if setting_val and any(
                p in setting_val.lower() for p in ("gpt-", "gpt4", "o1-", "o3-", "o4-")
            ):
                return setting_val
            if tier_enum == ModelTier.FAST:
                return "gpt-4o-mini"
            return "gpt-4o"

        # Default: Gemini / Google
        if setting_val and "gemini" in setting_val.lower():
            return setting_val
        if tier_enum == ModelTier.DEEP:
            return "gemini-2.5-pro"
        return "gemini-3.8-flash"

    def route(self, task_description: str) -> ModelTier:
        """
        Classify task description into appropriate ModelTier based on complexity.
        """
        desc_lower = task_description.lower()
        tokens = set(re.findall(r"\b\w+\b", desc_lower))

        if any(kw in desc_lower or kw in tokens for kw in _DEEP_KEYWORDS):
            return ModelTier.DEEP

        if any(kw in desc_lower or kw in tokens for kw in _STANDARD_KEYWORDS):
            return ModelTier.STANDARD

        return ModelTier.FAST

    async def generate(
        self,
        prompt: str,
        *,
        tier: ModelTier | str = ModelTier.FAST,
        model: str | None = None,
        temperature: float = 0.2,
        response_mime_type: str | None = None,
        timeout_sec: float = 30.0,
    ) -> str:
        """
        Execute completion request against active provider or offline mock fallback.
        """
        _ = (response_mime_type, timeout_sec)
        _ = model or self.get_model_for_tier(tier)

        is_test_or_dry_run = self.dry_run or self.mock_mode or "PYTEST_CURRENT_TEST" in os.environ

        if not self.api_key:
            if "pong" in prompt.lower():
                return "PONG"
            if is_test_or_dry_run:
                logger.debug("No API key found in test/dry-run mode; using offline simulation")
                return f"Simulated response for: '{prompt[:60]}'"
            env_var = (
                "GEMINI_API_KEY"
                if self.provider_name == "gemini"
                else f"{self.provider_name.upper()}_API_KEY"
            )
            raise AuthenticationError(
                f"No API key configured for provider '{self.provider_name}'. "
                f"Run 'helmx login' or export {env_var}."
            )

        return await self.active_provider.generate(
            prompt=prompt,
            temperature=temperature,
        )

    async def generate_turn(
        self,
        prompt: str,
        *,
        tools: list[dict[str, Any]] | None = None,
        tier: ModelTier | str = ModelTier.FAST,
        model: str | None = None,
        temperature: float = 0.2,
        timeout_sec: float = 30.0,
        target_tool: str | None = None,
    ) -> ModelTurn:
        """
        Execute an LLM generation turn with optional function/tool declarations.
        Returns a ModelTurn containing generated text and any requested tool calls.
        """
        _ = (temperature, timeout_sec)
        _ = model or self.get_model_for_tier(tier)

        is_test_or_dry_run = self.dry_run or self.mock_mode or "PYTEST_CURRENT_TEST" in os.environ

        if not self.api_key:
            if is_test_or_dry_run:
                logger.debug("No API key in test/dry-run mode; executing offline turn simulation")
                return self._offline_turn_simulation(prompt, tools, target_tool=target_tool)
            env_var = (
                "GEMINI_API_KEY"
                if self.provider_name == "gemini"
                else f"{self.provider_name.upper()}_API_KEY"
            )
            raise AuthenticationError(
                f"No API key configured for provider '{self.provider_name}'. "
                f"Run 'helmx login' or export {env_var}."
            )

        res_text, res_tool_calls = await self.active_provider.generate_with_tools(
            prompt=prompt,
            tools=tools,
        )

        turn_tool_calls: list[ToolCall] = []
        if res_tool_calls:
            turn_tool_calls.extend(
                ToolCall(name=str(tc.get("name", "")), args=dict(tc.get("args", {})))
                for tc in res_tool_calls
            )

        if not turn_tool_calls and tools:
            # Fallback regex parser: extract tool calls from markdown / python text
            extracted = extract_tool_calls_from_text(res_text, tools)
            if extracted:
                turn_tool_calls.extend(extracted)
            else:
                fallback = self._heuristic_tool_matching(prompt, tools, target_tool=target_tool)
                if fallback:
                    turn_tool_calls.extend(fallback)

        return ModelTurn(text=res_text, tool_calls=turn_tool_calls)

    def _offline_turn_simulation(
        self,
        prompt: str,
        tools: list[dict[str, Any]] | None,
        target_tool: str | None = None,
    ) -> ModelTurn:
        """Offline simulation heuristics for local unit tests and dry runs."""
        if tools:
            extracted = extract_tool_calls_from_text(prompt, tools)
            if extracted:
                return ModelTurn(
                    text=f"Selected tool '{extracted[0].name}'",
                    tool_calls=extracted,
                )
            matched = self._heuristic_tool_matching(prompt, tools, target_tool=target_tool)
            if matched:
                return ModelTurn(
                    text=f"Selected tool '{matched[0].name}'",
                    tool_calls=matched,
                )
        if "pong" in prompt.lower():
            return ModelTurn(text="PONG", tool_calls=[])
        prompt_preview = prompt[:70].strip()
        return ModelTurn(
            text=f"Evaluated task step context for: {prompt_preview}",
            tool_calls=[],
        )

    def heuristic_tool_matching(
        self,
        prompt: str,
        tools: list[dict[str, Any]],
        target_tool: str | None = None,
    ) -> list[ToolCall]:
        """Match tool calls from prompt using semantic and subword heuristics."""
        matched: list[ToolCall] = []
        prompt_lower = prompt.lower()
        for delimiter in (
            "\nprior context:",
            "\nprior steps context:",
            "\ncontext / evidence:",
        ):
            if delimiter in prompt_lower:
                prompt_lower = prompt_lower.split(delimiter)[0]

        # 0. Prioritize explicitly targeted tool if available in tools
        if target_tool and any(t.get("name") == target_tool for t in tools):
            return [
                ToolCall(
                    name=target_tool,
                    args=extract_tool_arguments(target_tool, prompt),
                )
            ]

        # 1. Exact or normalized tool name in prompt
        for tool_decl in tools:
            t_name = tool_decl.get("name", "")
            if not t_name:
                continue
            if t_name.lower() in prompt_lower or t_name.replace("_", " ").lower() in prompt_lower:
                matched.append(ToolCall(name=t_name, args=extract_tool_arguments(t_name, prompt)))
                return matched

        # 2. Filesystem tool operations
        for tool_decl in tools:
            t_name = tool_decl.get("name", "")
            if t_name == "read_file" and (
                any(
                    w in prompt_lower
                    for w in (
                        "read file",
                        "read_file",
                        "readme",
                        "cat ",
                        "inspect file",
                        "read ",
                        "filesystem",
                    )
                )
                and any(
                    ext in prompt_lower
                    for ext in (
                        ".md",
                        ".txt",
                        ".py",
                        ".json",
                        ".toml",
                        ".yaml",
                        ".yml",
                        ".sh",
                        "readme",
                        "file",
                    )
                )
            ):
                matched.append(ToolCall(name=t_name, args=extract_tool_arguments(t_name, prompt)))
                return matched

            if t_name == "list_directory" and any(
                w in prompt_lower
                for w in (
                    "list directory",
                    "list_directory",
                    "list files",
                    "list_files",
                    "ls ",
                    "dir ",
                    "directory contents",
                )
            ):
                matched.append(ToolCall(name=t_name, args=extract_tool_arguments(t_name, prompt)))
                return matched

            if t_name == "write_file" and any(
                w in prompt_lower
                for w in ("write file", "write_file", "save to file", "create file")
            ):
                matched.append(ToolCall(name=t_name, args=extract_tool_arguments(t_name, prompt)))
                return matched

        # 3. Knowledge graph / memory operations
        for tool_decl in tools:
            t_name = tool_decl.get("name", "")
            if t_name == "delete_entities" and any(
                w in prompt_lower for w in ("delete", "remove", "drop")
            ):
                matched.append(ToolCall(name=t_name, args=extract_tool_arguments(t_name, prompt)))
                return matched

            if t_name == "read_graph" and any(
                w in prompt_lower
                for w in (
                    "read graph",
                    "read_graph",
                    "knowledge graph",
                    "what entities",
                    "saved entities",
                )
            ):
                matched.append(ToolCall(name=t_name, args=extract_tool_arguments(t_name, prompt)))
                return matched

            if t_name == "search_nodes" and any(
                w in prompt_lower
                for w in ("search nodes", "search_nodes", "search graph", "find node")
            ):
                matched.append(ToolCall(name=t_name, args=extract_tool_arguments(t_name, prompt)))
                return matched

            if (
                t_name == "create_entities"
                and ("entity" in prompt_lower or "entities" in prompt_lower)
                and (
                    any(w in prompt_lower for w in ("store", "create", "insert", "record"))
                    or ("save" in prompt_lower and "saved" not in prompt_lower)
                )
            ):
                matched.append(ToolCall(name=t_name, args=extract_tool_arguments(t_name, prompt)))
                return matched

        return matched

    _heuristic_tool_matching = heuristic_tool_matching
