"""
tests/unit/test_tool_index.py — Unit tests for tool indexing, embeddings, and pruning.
"""

from __future__ import annotations

import numpy as np

from helm.memory.embeddings import (
    cosine_similarity,
    text_to_vector,
    tokenize,
)
from helm.memory.pruner import ToolPruner
from helm.memory.tool_index import ToolIndex
from helm.protocols.mcp_client import MCPToolDefinition
from helm.protocols.safety_gate import ActionSafety


class TestEmbeddings:
    def test_tokenize_words_and_subwords(self):
        tokens = tokenize("read_file_content")
        assert "read_file_content" in tokens
        assert "read" in tokens
        assert "file" in tokens
        assert "content" in tokens

    def test_text_to_vector_shape_and_norm(self):
        vec = text_to_vector("write data to database")
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (256,)
        # Must be unit norm
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5

    def test_empty_text_vector(self):
        vec = text_to_vector("")
        assert np.linalg.norm(vec) == 0.0

    def test_cosine_similarity(self):
        v1 = text_to_vector("git commit and push changes")
        v2 = text_to_vector("git commit code to repo")
        v3 = text_to_vector("cook spaghetti pasta recipe")

        sim_related = cosine_similarity(v1, v2)
        sim_unrelated = cosine_similarity(v1, v3)

        assert sim_related > sim_unrelated
        assert 0.0 <= sim_related <= 1.0


class TestToolIndex:
    def setup_method(self):
        self.index = ToolIndex()
        self.sample_tools = [
            MCPToolDefinition(
                name="write_file",
                description="Write content to a file on disk",
                input_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                },
                server_name="fs",
                safety=ActionSafety.MUTATING,
            ),
            MCPToolDefinition(
                name="read_file",
                description="Read content from a file on disk",
                input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
                server_name="fs",
                safety=ActionSafety.READ_ONLY,
            ),
            MCPToolDefinition(
                name="git_status",
                description="Show current git repository status and modified files",
                input_schema={"type": "object", "properties": {}},
                server_name="git",
                safety=ActionSafety.READ_ONLY,
            ),
            MCPToolDefinition(
                name="query_database",
                description="Execute a SQL SELECT query against postgres database",
                input_schema={"type": "object", "properties": {"sql": {"type": "string"}}},
                server_name="db",
                safety=ActionSafety.READ_ONLY,
            ),
        ]
        self.index.add_tools(self.sample_tools)

    def test_index_size(self):
        assert len(self.index) == 4

    def test_search_retrieves_most_relevant_tool(self):
        results = self.index.search("I want to save my notes into a file", top_k=2)
        assert len(results) == 2
        top_tool, score = results[0]
        assert top_tool.name == "write_file"
        assert score > 0.3

    def test_search_git_query(self):
        results = self.index.search("check repository working tree branch status", top_k=1)
        assert len(results) == 1
        assert results[0][0].name == "git_status"

    def test_search_database_query(self):
        results = self.index.search("run select query on users table in sql", top_k=1)
        assert len(results) == 1
        assert results[0][0].name == "query_database"

    def test_clear_index(self):
        self.index.clear()
        assert len(self.index) == 0
        assert self.index.search("write file") == []


class TestToolPruner:
    def setup_method(self):
        self.pruner = ToolPruner(default_top_k=2)
        self.sample_tools = [
            MCPToolDefinition(
                name="tool_a",
                description="Search documentation articles",
                input_schema={"properties": {"query": {}}},
                server_name="docs",
                safety=ActionSafety.READ_ONLY,
            ),
            MCPToolDefinition(
                name="tool_b",
                description="Execute shell command in terminal",
                input_schema={"properties": {"command": {}}},
                server_name="shell",
                safety=ActionSafety.MUTATING,
            ),
            MCPToolDefinition(
                name="tool_c",
                description="Read user profile settings",
                input_schema={"properties": {"user_id": {}}},
                server_name="users",
                safety=ActionSafety.READ_ONLY,
            ),
        ]
        self.pruner.load_tools(self.sample_tools)

    def test_init_with_tools(self):
        pruner = ToolPruner(tools=self.sample_tools, default_top_k=1)
        assert len(pruner.index) == len(self.sample_tools)
        selected = pruner.prune("search article docs", top_k=1)
        assert len(selected) == 1
        assert selected[0].name == "tool_a"

    def test_prune_limits_to_top_k(self):
        pruned = self.pruner.prune("search article docs", top_k=2)
        assert len(pruned) == 2
        assert pruned[0].name == "tool_a"

    def test_format_for_prompt(self):
        pruned = self.pruner.prune("shell terminal command", top_k=1)
        prompt_text = self.pruner.format_for_prompt(pruned)
        assert "tool_b" in prompt_text
        assert "command" in prompt_text
        assert "MUTATING" in prompt_text

    def test_format_for_llm_declarations(self):
        decls = self.pruner.format_for_llm_declarations(self.sample_tools)
        assert len(decls) == 3
        assert decls[0]["name"] == "tool_a"
        assert "parameters" in decls[0]
