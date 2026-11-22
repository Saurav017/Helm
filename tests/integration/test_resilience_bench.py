"""
tests/integration/test_resilience_bench.py — End-to-end resilience and self-reflection benchmarks.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from helm.benchmarks.resilience import ResilienceBenchmark
from helm.core.model_router import ModelRouter, ToolCallTurn
from helm.core.state import SessionStatus
from helm.core.supervisor import Supervisor
from helm.core.task_graph import TaskGraph, TaskStep
from helm.protocols.mcp_client import MCPClientManager, MCPToolDefinition
from helm.protocols.safety_gate import ActionSafety, SafetyGate


class TestResilienceBenchmark:
    @pytest.mark.asyncio
    async def test_benchmark_1_file_recovery_scenario(self) -> None:
        """
        Benchmark 1: Initial file path fails with FileNotFoundError, agent reflects,
        dynamically inserts a directory listing step, locates the valid path,
        and completes the goal.
        """
        # Setup mock MCP client
        client = AsyncMock(spec=MCPClientManager)
        client.registry = MagicMock()
        client.registry.collisions = set()
        client.safety_gate = SafetyGate(auto_approve=True)

        tool_read = MCPToolDefinition(
            name="read_file",
            description="Read file contents from filesystem",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
            server_name="filesystem",
            safety=ActionSafety.READ_ONLY,
        )
        tool_list = MCPToolDefinition(
            name="list_directory",
            description="List directory entries",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
            server_name="filesystem",
            safety=ActionSafety.READ_ONLY,
        )
        client.list_tools.return_value = [tool_read, tool_list]

        call_counts: dict[str, int] = {"read_file": 0, "list_directory": 0}

        async def fake_call_tool(tool_name: str, args: dict[str, Any], **kwargs: Any) -> Any:
            _ = kwargs
            if tool_name == "read_file":
                call_counts["read_file"] += 1
                path = args.get("path", "")
                if "settings.json" in path or call_counts["read_file"] == 1:
                    return {
                        "is_error": True,
                        "error": (
                            "FileNotFoundError: [Errno 2] No such file or directory: "
                            "'config/settings.json'"
                        ),
                    }
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "database:\n  host: db.prod.internal\n  port: 5432\n  name: helm_db"
                            ),
                        }
                    ]
                }
            if tool_name == "list_directory":
                call_counts["list_directory"] += 1
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": '["settings.yaml", "database.env", "logging.conf"]',
                        }
                    ]
                }
            return {"content": [{"type": "text", "text": "ok"}]}

        client.call_tool = AsyncMock(side_effect=fake_call_tool)

        # Setup mock model router to drive the extraction once file contents are found
        mock_router = AsyncMock(spec=ModelRouter)
        mock_router.route.return_value = "fast"
        mock_router.embed = AsyncMock(return_value=[0.1] * 64)

        async def fake_generate(prompt: str, **kwargs: Any) -> str:
            _ = kwargs
            if "db.prod.internal" in prompt or "settings.yaml" in prompt:
                return "The database host is db.prod.internal."
            return "Extracted database configuration."

        async def fake_generate_turn(prompt: str, **kwargs: Any) -> ToolCallTurn:
            _ = kwargs
            # If list_directory has completed and we are reading settings.yaml
            if "settings.yaml" in prompt:
                from helm.core.model_router import ToolCall

                return ToolCallTurn(
                    tool_calls=[ToolCall(name="read_file", args={"path": "config/settings.yaml"})]
                )
            return ToolCallTurn(tool_calls=[])

        mock_router.generate = AsyncMock(side_effect=fake_generate)
        mock_router.generate_turn = AsyncMock(side_effect=fake_generate_turn)

        # Build initial TaskGraph with a flawed step targeting settings.json
        goal = "Read the file 'config/settings.json' and extract the database host"
        initial_graph = TaskGraph(goal=goal)
        s1 = TaskStep(
            step_id="step-1",
            title="Read file 'config/settings.json'",
            description="Read file 'config/settings.json'",
            target_tool="read_file",
            assigned_worker="ToolWorker",
            depends_on=[],
        )
        s2 = TaskStep(
            step_id="step-2",
            title="Extract database host",
            description="Extract the database host from configuration",
            assigned_worker="ResearchWorker",
            depends_on=["step-1"],
        )
        initial_graph.add_step(s1)
        initial_graph.add_step(s2)

        supervisor = Supervisor(
            model_router=mock_router,
            safety_gate=SafetyGate(auto_approve=True),
        )
        bench = ResilienceBenchmark(supervisor=supervisor)

        final_state, report = await bench.run_scenario(
            name="file_recovery_benchmark",
            goal=goal,
            initial_graph=initial_graph,
            mcp_client=client,
        )

        assert report.passed
        assert final_state.status == SessionStatus.COMPLETED
        assert report.replan_count >= 1
        # Check that remediation step was injected and executed
        final_graph = final_state.metadata["task_graph"]
        step_ids = list(final_graph["steps"].keys())
        assert any("1a" in sid for sid in step_ids)
        assert call_counts["list_directory"] >= 1

    @pytest.mark.asyncio
    async def test_benchmark_2_invalid_json_payload_repair(self) -> None:
        """
        Benchmark 2: Tool call throws parameter validation error, agent catches schema critique,
        adjusts payload on retry, and succeeds.
        """
        client = AsyncMock(spec=MCPClientManager)
        client.registry = MagicMock()
        client.registry.collisions = set()
        client.safety_gate = SafetyGate(auto_approve=True)

        tool_patch = MCPToolDefinition(
            name="patch_config",
            description="Patch configuration key-value pair",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "patch": {"type": "object"},
                },
                "required": ["path", "patch"],
            },
            server_name="config_service",
            safety=ActionSafety.MUTATING,
        )
        client.list_tools.return_value = [tool_patch]

        attempts = 0

        async def fake_call_tool(tool_name: str, args: dict[str, Any], **kwargs: Any) -> Any:
            nonlocal attempts
            attempts += 1
            _ = kwargs
            # First attempt missing required 'patch' key
            if "patch" not in args or attempts == 1:
                return {
                    "is_error": True,
                    "error": (
                        f"ValidationError: missing required argument 'patch' for tool '{tool_name}'"
                    ),
                }
            return {
                "status": "success",
                "content": [{"type": "text", "text": '{"updated": true, "key": "host"}'}],
            }

        client.call_tool = AsyncMock(side_effect=fake_call_tool)

        mock_router = AsyncMock(spec=ModelRouter)
        mock_router.route.return_value = "fast"
        mock_router.embed = AsyncMock(return_value=[0.1] * 64)
        mock_router.generate = AsyncMock(return_value="Configuration successfully patched.")

        async def fake_generate_turn(prompt: str, **kwargs: Any) -> ToolCallTurn:
            _ = kwargs
            from helm.core.model_router import ToolCall

            # If prompt contains the critique, generate the repaired arguments
            if "Critique" in prompt:
                return ToolCallTurn(
                    tool_calls=[
                        ToolCall(
                            name="patch_config",
                            args={"path": "config.json", "patch": {"host": "db.prod.internal"}},
                        )
                    ]
                )
            # Initial faulty call without 'patch'
            return ToolCallTurn(
                tool_calls=[ToolCall(name="patch_config", args={"path": "config.json"})]
            )

        mock_router.generate_turn = AsyncMock(side_effect=fake_generate_turn)

        goal = "Update configuration host"
        initial_graph = TaskGraph(goal=goal)
        s1 = TaskStep(
            step_id="step-1",
            title="Patch database configuration",
            description="Execute patch_config tool to update host",
            target_tool="patch_config",
            assigned_worker="ToolWorker",
            depends_on=[],
        )
        initial_graph.add_step(s1)

        supervisor = Supervisor(
            model_router=mock_router,
            safety_gate=SafetyGate(auto_approve=True),
        )
        bench = ResilienceBenchmark(supervisor=supervisor)

        final_state, report = await bench.run_scenario(
            name="schema_repair_benchmark",
            goal=goal,
            initial_graph=initial_graph,
            mcp_client=client,
        )

        assert report.passed
        assert final_state.status == SessionStatus.COMPLETED
        assert attempts >= 2
