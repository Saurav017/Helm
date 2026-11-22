"""
tests/unit/test_workers.py — Unit tests for BaseWorker, ToolWorker, ResearchWorker, CodeWorker.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from helm.core.model_router import ModelRouter, ModelTurn, ToolCall
from helm.core.state import TaskBrief, TaskResult, TaskStatus
from helm.protocols.mcp_client import MCPClientManager, MCPToolDefinition
from helm.protocols.safety_gate import ActionSafety, SafetyGate
from helm.workers.base import BaseWorker
from helm.workers.coder import CodeWorker
from helm.workers.devops import DevOpsWorker
from helm.workers.researcher import ResearchWorker
from helm.workers.tool_worker import ToolWorker


class DummyWorker(BaseWorker):
    """Concrete test implementation of BaseWorker."""

    async def execute(self, brief: TaskBrief) -> TaskResult:
        if "fail" in brief.description:
            raise RuntimeError("Forced execution failure")
        return TaskResult(
            task_id=brief.task_id,
            status=TaskStatus.COMPLETED,
            output=f"Executed: {brief.description}",
        )


class TestBaseWorker:
    @pytest.mark.asyncio
    async def test_lifecycle_success(self):
        worker = DummyWorker(name="TestWorker", role="Tester")
        brief = TaskBrief(task_id="task-1", description="Valid task")

        result = await worker.run(brief)

        assert result.status == TaskStatus.COMPLETED
        assert result.output == "Executed: Valid task"
        assert result.error is None

    @pytest.mark.asyncio
    async def test_lifecycle_error_handling(self):
        worker = DummyWorker(name="TestWorker", role="Tester")
        brief = TaskBrief(task_id="task-2", description="fail this task")

        result = await worker.run(brief)

        assert result.status == TaskStatus.FAILED
        assert "Forced execution failure" in (result.error or "")


class TestToolWorker:
    @pytest.mark.asyncio
    async def test_tool_worker_invokes_mcp_client(self):
        mock_router = AsyncMock(spec=ModelRouter)
        mock_router.generate_turn.return_value = ModelTurn(
            text="Calling read_graph",
            tool_calls=[ToolCall(name="read_graph", args={})],
        )

        mock_mcp = AsyncMock(spec=MCPClientManager)
        mock_mcp.call_tool.return_value = {
            "content": [{"type": "text", "text": '{"entities": ["ProjectHelm"]}'}],
        }

        safety_gate = SafetyGate(auto_approve=True)
        worker = ToolWorker(model_router=mock_router, safety_gate=safety_gate)

        tools = [
            MCPToolDefinition(
                name="read_graph",
                description="Read graph",
                server_name="mem",
                safety=ActionSafety.READ_ONLY,
            ),
        ]
        brief = TaskBrief(task_id="t-1", description="Call tool read_graph")

        result = await worker.execute(brief, mcp_client=mock_mcp, tools=tools, auto_approve=True)

        assert result.status == TaskStatus.COMPLETED
        mock_mcp.call_tool.assert_awaited_with(
            "read_graph",
            {},
            auto_approve=True,
            summary="ToolWorker executing read_graph",
        )
        assert "ProjectHelm" in result.output

    @pytest.mark.asyncio
    async def test_tool_worker_safety_denial(self):
        mock_router = AsyncMock(spec=ModelRouter)
        mock_router.generate_turn.return_value = ModelTurn(
            text="Calling delete_entities",
            tool_calls=[ToolCall(name="delete_entities", args={"names": ["Old"]})],
        )
        mock_mcp = AsyncMock(spec=MCPClientManager)

        mock_gate = AsyncMock(spec=SafetyGate)
        mock_gate.request_permission.return_value = False

        worker = ToolWorker(model_router=mock_router, safety_gate=mock_gate)
        brief = TaskBrief(task_id="t-2", description="Delete old entity")

        result = await worker.execute(brief, mcp_client=mock_mcp, auto_approve=False)

        assert result.status == TaskStatus.FAILED
        assert "denied by safety gate" in (result.error or "").lower()
        mock_mcp.call_tool.assert_not_awaited()


class TestResearchWorker:
    @pytest.mark.asyncio
    async def test_research_worker_synthesis(self):
        mock_router = AsyncMock(spec=ModelRouter)
        mock_router.generate.return_value = (
            "Entity ProjectHelm connects to Python via observations."
        )

        worker = ResearchWorker(model_router=mock_router)
        assert "Research & Synthesis Specialist" in worker.system_prompt

        brief = TaskBrief(
            task_id="r-1",
            description="Synthesize entity connections",
            relevant_context="Found entity ProjectHelm with observation written in Python.",
        )

        result = await worker.execute(brief)

        assert result.status == TaskStatus.COMPLETED
        assert "ProjectHelm connects to Python" in result.output
        mock_router.generate.assert_awaited()


class TestCodeWorker:
    @pytest.mark.asyncio
    async def test_code_worker_transformation(self):
        mock_router = AsyncMock(spec=ModelRouter)
        mock_router.generate.return_value = '{"formatted": true, "status": "valid"}'

        worker = CodeWorker(model_router=mock_router)
        assert "Code & Structured Data Specialist" in worker.system_prompt

        brief = TaskBrief(
            task_id="c-1",
            description="Format data schema",
            relevant_context="raw text input",
        )

        result = await worker.execute(brief)

        assert result.status == TaskStatus.COMPLETED
        assert "formatted" in result.output
        mock_router.generate.assert_awaited()


class TestDevOpsWorker:
    @pytest.mark.asyncio
    async def test_devops_worker(self):
        worker = DevOpsWorker()
        brief = TaskBrief(task_id="d-1", description="Check container health")
        result = await worker.execute(brief)
        assert result.status == TaskStatus.COMPLETED
        assert "DevOps execution completed" in result.output
