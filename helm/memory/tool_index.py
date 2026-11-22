"""
helm/memory/tool_index.py — Vector and lexical search index for MCP tool schemas.

Enables dynamic tool retrieval (RAG for tools) so only relevant tool definitions
are included in prompt context (INSTRUCTIONS.md §3.5).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from helm.memory.embeddings import (
    VECTOR_DIM,
    batch_cosine_similarity,
    text_to_vector,
    tokenize,
)
from helm.protocols.mcp_client import MCPToolDefinition


def _build_tool_document(tool: MCPToolDefinition) -> str:
    """Build a search document string from tool name, description, and schema properties."""
    parts = [tool.name, tool.name.replace("_", " "), tool.description]

    # Extract schema properties
    schema = tool.input_schema
    if isinstance(schema, dict):
        props = schema.get("properties", {})
        if isinstance(props, dict):
            for prop_name, prop_def in props.items():
                parts.append(prop_name)
                parts.append(prop_name.replace("_", " "))
                if isinstance(prop_def, dict):
                    desc = prop_def.get("description", "")
                    if desc:
                        parts.append(desc)

    return " ".join(parts)


class ToolIndex:
    """
    Search index over normalized MCPToolDefinitions.

    Combines vector semantic similarity with lexical token overlap for robust
    ranking of tool relevance.
    """

    def __init__(self, dim: int = VECTOR_DIM) -> None:
        self.dim = dim
        self._tools: list[MCPToolDefinition] = []
        self._vectors: list[np.ndarray] = []
        self._token_sets: list[set[str]] = []
        self._matrix: np.ndarray = np.empty((0, dim), dtype=np.float32)

    def add_tool(self, tool: MCPToolDefinition) -> None:
        """Add a single MCPToolDefinition to the index."""
        # Avoid duplicate tool registrations
        if any(t.name == tool.name and t.server_name == tool.server_name for t in self._tools):
            return

        doc = _build_tool_document(tool)
        vec = text_to_vector(doc, self.dim)
        tokens = set(tokenize(doc))

        self._tools.append(tool)
        self._vectors.append(vec)
        self._token_sets.append(tokens)
        self._matrix = np.array(self._vectors, dtype=np.float32)

    def add_tools(self, tools: Sequence[MCPToolDefinition]) -> None:
        """Add multiple MCPToolDefinitions to the index."""
        for t in tools:
            self.add_tool(t)

    def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        threshold: float = 0.0,
    ) -> list[tuple[MCPToolDefinition, float]]:
        """
        Search for tools relevant to the query.

        Parameters
        ----------
        query:
            Task description or user prompt step.
        top_k:
            Maximum number of tools to return.
        threshold:
            Minimum relevance score in [0.0, 1.0].

        Returns
        -------
        list[tuple[MCPToolDefinition, float]]:
            Ranked list of (tool, score) tuples.
        """
        if not self._tools:
            return []

        query_vec = text_to_vector(query, self.dim)
        query_tokens = set(tokenize(query))

        # 1. Dense vector cosine similarity
        vector_scores = batch_cosine_similarity(query_vec, self._matrix)

        # 2. Hybrid scoring with lexical boost
        scored: list[tuple[MCPToolDefinition, float]] = []
        for i, tool in enumerate(self._tools):
            v_score = float(vector_scores[i])

            # Lexical overlap: Jaccard-like bonus
            doc_tokens = self._token_sets[i]
            overlap = len(query_tokens & doc_tokens)
            l_bonus = (overlap / (len(query_tokens) + 1e-6)) * 0.4 if query_tokens else 0.0

            # Direct tool name match bonus
            tool_name_tokens = set(tool.name.lower().split("_"))
            if tool_name_tokens & query_tokens:
                l_bonus += 0.3

            final_score = min(1.0, (v_score * 0.6) + l_bonus)

            if final_score >= threshold:
                scored.append((tool, final_score))

        # Sort descending by score
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def clear(self) -> None:
        """Clear all indexed tools."""
        self._tools.clear()
        self._vectors.clear()
        self._token_sets.clear()
        self._matrix = np.empty((0, self.dim), dtype=np.float32)

    def get_all_tools(self) -> list[MCPToolDefinition]:
        """Return a copy of all tools currently in the index."""
        return list(self._tools)

    def __len__(self) -> int:
        return len(self._tools)
