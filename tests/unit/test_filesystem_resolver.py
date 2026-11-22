"""
tests/unit/test_filesystem_resolver.py — Unit tests for filesystem tool adapter and resolver.

Tests:
1. Robust File Resolver (exact match, case-insensitivity, extensionless match e.g. readme -> README.md)
2. list_directory graceful fallback to '.' with mild warning when path does not exist
3. Parameter correction instructions in system prompts
4. MCPClientManager integration with filesystem tools
"""

from __future__ import annotations

from pathlib import Path

import pytest

from helm.core.prompts import (
    PARAMETER_CORRECTION_INSTRUCTIONS,
    TOOL_USAGE_SYSTEM_INSTRUCTION,
)
from helm.core.providers.openai import build_tools_system_instruction
from helm.protocols.filesystem_adapter import (
    list_directory,
    list_directory_sync,
    read_file,
    read_file_sync,
    resolve_file_path,
)
from helm.protocols.mcp_client import MCPClientManager, ToolNotFoundError
from helm.protocols.safety_gate import SafetyGate
from helm.workers.tool_worker import TOOL_WORKER_PROMPT


class TestRobustFileResolver:
    """Test resolve_file_path and read_file."""

    def test_resolve_exact_file_match(self) -> None:
        """Existing file with exact path is resolved directly."""
        resolved = resolve_file_path("README.md")
        assert resolved is not None
        assert resolved.name == "README.md"
        assert resolved.is_file()

    def test_resolve_case_insensitive_match(self) -> None:
        """readme.md or Readme.MD resolves to README.md in the current directory."""
        resolved = resolve_file_path("readme.md")
        assert resolved is not None
        assert resolved.name == "README.md"

        resolved_mixed = resolve_file_path("ReadMe.Md")
        assert resolved_mixed is not None
        assert resolved_mixed.name == "README.md"

    def test_resolve_extensionless_stem_match(self) -> None:
        """'readme' resolves to 'README.md'."""
        resolved = resolve_file_path("readme")
        assert resolved is not None
        assert resolved.name == "README.md"

    def test_resolve_in_temporary_directory(self, tmp_path: Path) -> None:
        """Test resolver in a controlled directory with subdirectories and mixed cases."""
        sample_file = tmp_path / "MyConfig.JSON"
        sample_file.write_text('{"key": "value"}', encoding="utf-8")

        # Exact case
        assert resolve_file_path("MyConfig.JSON", base_dir=tmp_path) == sample_file

        # Case-insensitive
        res = resolve_file_path("myconfig.json", base_dir=tmp_path)
        assert res is not None
        assert res.name == "MyConfig.JSON"

        # Extensionless
        res_stem = resolve_file_path("myconfig", base_dir=tmp_path)
        assert res_stem is not None
        assert res_stem.name == "MyConfig.JSON"

    def test_resolve_nonexistent_returns_none(self, tmp_path: Path) -> None:
        """Non-existent path returns None."""
        assert resolve_file_path("completely_missing_file.xyz", base_dir=tmp_path) is None
        assert resolve_file_path("", base_dir=tmp_path) is None

    @pytest.mark.asyncio
    async def test_read_file_resolves_and_reads(self, tmp_path: Path) -> None:
        """read_file finds case-insensitive matches and returns content."""
        target = tmp_path / "README.md"
        target.write_text("# Test Repo\nHello world", encoding="utf-8")

        result = await read_file("readme", base_dir=tmp_path)
        assert result.get("is_error") is False
        assert result.get("content") is not None
        assert "# Test Repo" in result["content"][0]["text"]

    def test_read_file_sync_missing_returns_error(self, tmp_path: Path) -> None:
        """Missing file returns clean FileNotFoundError structure."""
        result = read_file_sync("nonexistent.txt", base_dir=tmp_path)
        assert result.get("is_error") is True
        assert "FileNotFoundError" in result.get("error", "")


class TestListDirectoryFallback:
    """Test list_directory fallback to project root '.' with mild warning."""

    def test_list_directory_valid_dir(self, tmp_path: Path) -> None:
        """Valid directory listing contains file entries and no warning."""
        f1 = tmp_path / "test1.txt"
        f1.write_text("sample", encoding="utf-8")

        res = list_directory_sync(str(tmp_path))
        assert res.get("is_error") is False
        assert "warning" not in res
        assert "test1.txt" in res["content"][0]["text"]

    def test_list_directory_invalid_path_falls_back_to_root(self, tmp_path: Path) -> None:
        """Hallucinated path 'Step' falls back to base_dir with mild warning."""
        f1 = tmp_path / "hello.txt"
        f1.write_text("content", encoding="utf-8")

        res = list_directory_sync("Step", base_dir=tmp_path)
        assert res.get("is_error") is False
        assert res.get("warning") == "Path 'Step' not found, listed root directory instead"
        assert "Path 'Step' not found, listed root directory instead" in res["content"][0]["text"]
        assert "hello.txt" in res["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_list_directory_async_fallback(self, tmp_path: Path) -> None:
        """Async list_directory falls back with mild warning."""
        res = await list_directory("InvalidSubdir123", base_dir=tmp_path)
        assert res.get("is_error") is False
        assert "Path 'InvalidSubdir123' not found, listed root directory instead" in res.get(
            "warning", ""
        )


class TestParameterCorrectionPrompt:
    """Verify system instructions reinforce exact filenames and '.' for root."""

    def test_parameter_correction_prompt_content(self) -> None:
        expected = (
            "When reading files, always provide exact filenames with extensions "
            "(e.g., 'README.md', 'pyproject.toml'). Always use '.' to list the root directory."
        )
        assert expected == PARAMETER_CORRECTION_INSTRUCTIONS
        assert PARAMETER_CORRECTION_INSTRUCTIONS in TOOL_USAGE_SYSTEM_INSTRUCTION

    def test_openai_tool_instructions_include_parameter_correction(self) -> None:
        tools = [
            {
                "type": "function",
                "function": {"name": "read_file", "parameters": {"type": "object"}},
            },
            {
                "type": "function",
                "function": {"name": "list_directory", "parameters": {"type": "object"}},
            },
        ]
        prompt = build_tools_system_instruction(tools)
        assert PARAMETER_CORRECTION_INSTRUCTIONS in prompt
        assert "When reading files, always provide exact filenames with extensions" in prompt

    def test_tool_worker_prompt_includes_parameter_correction(self) -> None:
        assert PARAMETER_CORRECTION_INSTRUCTIONS in TOOL_WORKER_PROMPT


class TestMCPClientManagerFilesystemIntegration:
    """Test MCPClientManager handling of read_file and list_directory."""

    @pytest.mark.asyncio
    async def test_mcp_client_resolves_readme_without_server(self) -> None:
        """MCPClientManager falls back to filesystem adapter when server is absent."""
        gate = SafetyGate(auto_approve=True)
        async with MCPClientManager(safety_gate=gate) as client:
            res = await client.call_tool("read_file", {"path": "readme"})
            assert res.get("is_error") is False
            assert "HELM" in res["content"][0]["text"] or "helm" in res["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_mcp_client_list_directory_step_fallback(self) -> None:
        """MCPClientManager gracefully falls back to '.' with mild warning for 'Step'."""
        gate = SafetyGate(auto_approve=True)
        async with MCPClientManager(safety_gate=gate) as client:
            res = await client.call_tool("list_directory", {"path": "Step"})
            assert res.get("is_error") is False
            assert res.get("warning") == "Path 'Step' not found, listed root directory instead"
            assert "README.md" in res["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_mcp_client_raises_for_unknown_tool(self) -> None:
        """Unknown non-filesystem tools still raise ToolNotFoundError."""
        gate = SafetyGate(auto_approve=True)
        async with MCPClientManager(safety_gate=gate) as client:
            with pytest.raises(ToolNotFoundError, match="unknown_custom_tool"):
                await client.call_tool("unknown_custom_tool", {})
