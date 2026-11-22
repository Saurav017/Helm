"""
tests/unit/test_m4_bdd.py — BDD feature tests for Milestone 4:
Specialized Workers, Context Compaction, and Session Recovery.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock

import pytest

from helm.core.state import SessionState, SessionStatus
from helm.core.supervisor import Supervisor
from helm.core.task_graph import StepStatus, TaskGraph, TaskStep
from helm.memory.compactor import ContextCompactor
from helm.protocols.mcp_client import MCPClientManager, MCPToolDefinition
from helm.protocols.safety_gate import ActionSafety, SafetyGate
from helm.workers.coder import CODE_WORKER_PROMPT
from helm.workers.researcher import RESEARCH_WORKER_PROMPT
from helm.workers.tool_worker import TOOL_WORKER_PROMPT


class TestMilestone4BDD:
    @pytest.mark.asyncio
    async def test_bdd_scenario_1_specialized_worker_dispatch(self):
        """
        Scenario 1: Supervisor routes distinct DAG step types to specialized workers
        """
        client = AsyncMock(spec=MCPClientManager)
        client.list_tools.return_value = [
            MCPToolDefinition(
                name="read_graph",
                description="Read knowledge graph",
                server_name="memory",
                safety=ActionSafety.READ_ONLY,
            ),
        ]
        client.call_tool.return_value = {
            "content": [{"type": "text", "text": '{"entities": [{"name": "ProjectHelm"}]}'}],
        }

        # Given a TaskGraph with step-1 and step-2
        graph = TaskGraph(goal="Explore entities and synthesize connections")
        s1 = TaskStep(
            step_id="step-1",
            title="Call MCP tool read_graph",
            description='Call MCP tool "read_graph"',
            target_tool="read_graph",
            depends_on=[],
        )
        s2 = TaskStep(
            step_id="step-2",
            title="Synthesize and summarize entity connections",
            description="Synthesize and summarize entity connections",
            depends_on=["step-1"],
        )
        graph.add_step(s1)
        graph.add_step(s2)

        supervisor = Supervisor(safety_gate=SafetyGate(auto_approve=True))
        state = SessionState(user_goal="Explore entities and synthesize connections")

        # When Supervisor executes the graph
        result = await supervisor.run(state, client, auto_approve=True, graph=graph)

        # Then step-1 assigned worker must be ToolWorker
        task_1 = result.get_task("step-1")
        assert task_1 is not None
        assert task_1.assigned_worker == "ToolWorker"
        assert task_1.worker == "ToolWorker"

        # Then step-2 assigned worker must be ResearchWorker
        task_2 = result.get_task("step-2")
        assert task_2 is not None
        assert task_2.assigned_worker == "ResearchWorker"
        assert task_2.worker == "ResearchWorker"

        # And each worker must use role-calibrated system instructions
        assert "Tool Execution Specialist" in supervisor.workers["ToolWorker"].system_prompt
        assert supervisor.workers["ToolWorker"].system_prompt == TOOL_WORKER_PROMPT
        assert (
            "Research & Synthesis Specialist" in supervisor.workers["ResearchWorker"].system_prompt
        )
        assert supervisor.workers["ResearchWorker"].system_prompt == RESEARCH_WORKER_PROMPT
        assert "Code & Structured Data Specialist" in supervisor.workers["CodeWorker"].system_prompt
        assert supervisor.workers["CodeWorker"].system_prompt == CODE_WORKER_PROMPT

    @pytest.mark.asyncio
    async def test_bdd_scenario_2_context_window_compaction(self):
        """
        Scenario 2: Intermediate outputs are compacted before reaching context limits
        """
        compactor = ContextCompactor(token_threshold=500)

        # Given a session with 4 completed steps generating verbose JSON outputs
        verbose_items = [
            {
                "entityName": f"ServiceEntity_{i}",
                "id": f"uuid-step-{i}",
                "path": f"/app/service_{i}.py",
                "error": f"50{i}",
                "payload": "very verbose intermediate json output payload data " * 20,
            }
            for i in range(1, 5)
        ]

        steps = [
            TaskStep(
                step_id=f"step-{i}",
                title=f"Fetch intermediate chunk {i}",
                status=StepStatus.COMPLETED,
                output=json.dumps(verbose_items[i - 1]),
            )
            for i in range(1, 5)
        ]

        # Total tokens exceed the compaction threshold
        total_tokens = sum(compactor.estimate_tokens(s.output or "") for s in steps)
        assert total_tokens > compactor.token_threshold

        # When compactor runs
        compacted = await compactor.compact_steps(steps)

        # Then outputs are condensed into concise Markdown summaries
        assert "### Compacted Context Summary" in compacted
        assert "[condensed]" in compacted

        # And crucial metadata (entity names, IDs, paths, error codes) is retained
        for i in range(1, 5):
            assert f"ServiceEntity_{i}" in compacted
            assert f"uuid-step-{i}" in compacted or f"step-{i}" in compacted
            assert f"/app/service_{i}.py" in compacted
            assert f"50{i}" in compacted

    @pytest.mark.asyncio
    async def test_bdd_scenario_3_session_persistence_and_resumption(self):
        """
        Scenario 3: Session progress is checkpointed to disk after every step
        """
        with TemporaryDirectory() as tmpdir:
            orig_cwd = Path.cwd()
            try:
                os.chdir(tmpdir)

                session_id = "test-session-123"
                client = AsyncMock(spec=MCPClientManager)
                client.list_tools.return_value = [
                    MCPToolDefinition(
                        name="read_graph",
                        description="Read graph",
                        server_name="memory",
                        safety=ActionSafety.READ_ONLY,
                    ),
                ]
                client.call_tool.return_value = {
                    "content": [{"type": "text", "text": '{"result": "step-1 success"}'}],
                }

                graph = TaskGraph(goal="Persist after every step")
                s1 = TaskStep(
                    step_id="step-1",
                    title="Execute first step",
                    description='Call MCP tool "read_graph"',
                    target_tool="read_graph",
                    depends_on=[],
                )
                s2 = TaskStep(
                    step_id="step-2",
                    title="Execute second step",
                    description="Synthesize findings from step-1",
                    depends_on=["step-1"],
                )
                graph.add_step(s1)
                graph.add_step(s2)

                supervisor = Supervisor(safety_gate=SafetyGate(auto_approve=True))
                state = SessionState(session_id=session_id, user_goal="Persist after every step")

                # Execute session
                result_state = await supervisor.run(
                    state,
                    client,
                    auto_approve=True,
                    graph=graph,
                )
                assert result_state.status == SessionStatus.COMPLETED

                # Then a checkpoint file ".helm/sessions/test-session-123.json" must be written to disk
                checkpoint_file = Path(".helm") / "sessions" / f"{session_id}.json"
                assert checkpoint_file.exists(), f"Expected checkpoint at {checkpoint_file}"

                # And the file must contain TaskGraph state, step statuses, and step results
                data = json.loads(checkpoint_file.read_text(encoding="utf-8"))
                assert data["session_id"] == session_id
                assert "metadata" in data
                assert "task_graph" in data["metadata"]

                saved_graph = data["metadata"]["task_graph"]
                assert "steps" in saved_graph
                assert "step-1" in saved_graph["steps"]
                assert saved_graph["steps"]["step-1"]["status"] == "completed"
                assert saved_graph["steps"]["step-1"]["output"] is not None

                # Verify resuming session from disk loads previously completed step
                loaded_state = SessionState.load(checkpoint_file)
                assert loaded_state.session_id == session_id
                assert loaded_state.status == SessionStatus.COMPLETED
            finally:
                os.chdir(orig_cwd)
