"""
helm/memory/pruner.py — Dynamic Tool Pruner (RAG for Tools).

RULE (INSTRUCTIONS.md §3.5):
  The system MUST NOT inject all available tool schemas into every LLM prompt.
  Before each LLM call, the Tool Pruner:
    1. Embeds the current task/query.
    2. Searches the tool description index.
    3. Returns the top-k (default: 3) most relevant tool schemas.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from helm.memory.tool_index import ToolIndex
from helm.protocols.mcp_client import MCPToolDefinition


class ToolPruner:
    """
    Retrieves and formats only the top-k relevant tools for a given task step.
    Prevents token burn and context pollution across worker/supervisor LLM calls.
    """

    def __init__(
        self,
        index: ToolIndex | None = None,
        *,
        tools: Sequence[MCPToolDefinition] | None = None,
        default_top_k: int = 3,
        threshold: float = 0.0,
    ) -> None:
        self.index = index or ToolIndex()
        self.default_top_k = default_top_k
        self.threshold = threshold
        if tools:
            self.load_tools(tools)

    def load_tools(self, tools: Sequence[MCPToolDefinition]) -> None:
        """Register tool definitions into the search index."""
        self.index.add_tools(tools)

    def prune(
        self,
        query: str,
        *,
        top_k: int | None = None,
    ) -> list[MCPToolDefinition]:
        """
        Retrieve the top-k most relevant tools for the query.

        Parameters
        ----------
        query:
            Current goal, task description, or active step.
        top_k:
            Optional override for number of tools to return.

        Returns
        -------
        list[MCPToolDefinition]:
            Pruned list of tools relevant to the active step.
        """
        k = top_k if top_k is not None else self.default_top_k

        # If total tools in index is less than or equal to k, return all
        if len(self.index) <= k:
            return self.index.get_all_tools()

        results = self.index.search(query, top_k=k, threshold=self.threshold)
        if not results and len(self.index) > 0:
            # Fallback: if similarity threshold filtered all tools, return top-k by relevance
            results = self.index.search(query, top_k=k, threshold=0.0)
        return [tool for tool, _score in results]

    def format_for_prompt(self, tools: Sequence[MCPToolDefinition]) -> str:
        """Format pruned tools into a concise text summary for LLM prompt context."""
        if not tools:
            return "No tools available."

        lines = ["Available Tools (Pruned for this step):"]
        for t in tools:
            schema_keys = list(t.input_schema.get("properties", {}).keys())
            args_str = ", ".join(schema_keys) if schema_keys else "none"
            lines.append(f"- {t.name}({args_str}): {t.description} [{t.safety.value}]")
        return "\n".join(lines)

    def format_for_llm_declarations(
        self,
        tools: Sequence[MCPToolDefinition],
    ) -> list[dict[str, Any]]:
        """
        Format pruned tools into standard JSON Schema function declarations
        compatible with Gemini and OpenAI function calling.
        """
        declarations: list[dict[str, Any]] = []
        for t in tools:
            decl: dict[str, Any] = {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema or {"type": "object", "properties": {}},
            }
            declarations.append(decl)
        return declarations
