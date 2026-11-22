"""
tests/unit/test_m6_bdd.py — BDD feature tests for Milestone 6:
Autonomous Self-Reflection, Dynamic DAG Re-planning, and Governed Worker Delegation.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from helm.core.model_router import ModelRouter, ToolCallTurn
from helm.core.planner import RePlanner
from helm.core.reflection import ReflectionEngine
from helm.core.state import SessionState, SessionStatus
from helm.core.supervisor import Supervisor
from helm.core.task_graph import StepStatus, TaskGraph, TaskStep
from helm.protocols.mcp_client import MCPClientManager, MCPToolDefinition
from helm.protocols.safety_gate import ActionSafety, SafetyGate
from helm.workers.coder import SWEWorker
from helm.workers.delegator import (
    DelegationDepthExceededError,
    DelegationLimitExceededError,
    WorkerDelegator,
)
from helm.workers.tool_worker import ToolWorker


class TestMilestone6BDD:
    @pytest.mark.asyncio
    async def test_bdd_scenario_1_autonomous_error_recovery_via_dag_mutation(self) -> None:
        """
        Scenario 1: Supervisor detects a failed tool call and dynamically inserts a fix step.
        """
        goal = "Read the file 'config/settings.json' and extract the database host"
        graph = TaskGraph(goal=goal)
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
        graph.add_step(s1)
        graph.add_step(s2)

        # When step-1 executes read_file on config/settings.json and receives FileNotFoundError
        raw_error = "FileNotFoundError: [Errno 2] No such file or directory: 'config/settings.json'"
        engine = ReflectionEngine()
        reflection = engine.evaluate_step(s1, result=raw_error)

        # Then the ReflectionEngine must classify the failure as "MISSING_RESOURCE"
        assert reflection.failure_classification == "MISSING_RESOURCE"
        assert reflection.status == "FAILED_RECOVERABLE"
        assert reflection.suggested_action == "INSERT_STEPS"
        assert reflection.suggested_steps is not None
        assert len(reflection.suggested_steps) == 1

        # And the Re-planner must dynamically insert step-1a before the extract step
        replanner = RePlanner()
        new_steps = replanner.apply_recovery(
            graph=graph,
            failing_step=s1,
            reflection=reflection,
            target_step_id="step-2",
        )
        assert len(new_steps) == 1
        s1a = new_steps[0]
        assert "step-1a" in s1a.step_id or "1a" in s1a.step_id
        assert "List files in 'config/' directory" in (s1a.title or s1a.description)

        # And update the DAG dependencies so the extraction waits for step-1a
        step_2_updated = graph.get_step("step-2")
        assert step_2_updated is not None
        assert s1a.step_id in step_2_updated.depends_on
        assert "step-1" not in step_2_updated.depends_on

        # And execution must continue autonomously to locate the correct file name
        client = AsyncMock(spec=MCPClientManager)
        client.registry = MagicMock()
        client.registry.collisions = set()
        client.safety_gate = SafetyGate(auto_approve=True)

        async def fake_call(tool_name: str, args: dict[str, Any], **kwargs: Any) -> Any:
            _ = (args, kwargs)
            if tool_name == "list_directory":
                return {"content": [{"type": "text", "text": '["settings.yaml", "database.env"]'}]}
            if tool_name == "read_file":
                return {"content": [{"type": "text", "text": "host: db.prod.internal"}]}
            return {"content": [{"type": "text", "text": "ok"}]}

        client.call_tool = AsyncMock(side_effect=fake_call)

        tool_list = MCPToolDefinition(
            name="list_directory",
            description="List directory",
            input_schema={"type": "object"},
            server_name="filesystem",
            safety=ActionSafety.READ_ONLY,
        )
        tool_read = MCPToolDefinition(
            name="read_file",
            description="Read file",
            input_schema={"type": "object"},
            server_name="filesystem",
            safety=ActionSafety.READ_ONLY,
        )
        client.list_tools.return_value = [tool_list, tool_read]

        mock_router = AsyncMock(spec=ModelRouter)
        mock_router.route.return_value = "fast"
        mock_router.embed = AsyncMock(return_value=[0.1] * 64)
        mock_router.generate = AsyncMock(return_value="The database host is db.prod.internal.")
        mock_router.generate_turn = AsyncMock(
            return_value=ToolCallTurn(
                text="The database host is db.prod.internal.",
                tool_calls=[],
            )
        )

        supervisor = Supervisor(
            model_router=mock_router,
            safety_gate=SafetyGate(auto_approve=True),
        )
        state = SessionState(user_goal=goal)
        final_state = await supervisor.run(state, client, auto_approve=True, graph=graph)

        assert final_state.status == SessionStatus.COMPLETED
        assert (
            "db.prod.internal" in (final_state.final_answer or "")
            or graph.get_step("step-2").status == StepStatus.COMPLETED
        )

    @pytest.mark.asyncio
    async def test_bdd_scenario_2_reflection_quality_gate_rejection(self) -> None:
        """
        Scenario 2: Step output fails semantic verification and triggers a targeted retry.
        """
        # Given a code generation step executed by "SWEWorker"
        step = TaskStep(
            step_id="code-1",
            title="Generate database connection helper",
            description="Generate database connection helper function in python",
            assigned_worker="SWEWorker",
        )

        # When the generated code fails syntax verification
        invalid_code = "```python\ndef connect_db(:\n    pass\n```"
        engine = ReflectionEngine()

        # Then the ReflectionEngine must return critique matching the table
        critique = engine.evaluate_step(step, result=invalid_code)

        assert critique["Critique Status"] == "REJECTED"
        assert critique["Root Cause"] == "SyntaxError in Python AST"
        assert critique["Action"] == "Regenerate with corrected AST syntax"
        assert critique.status == "FAILED_RECOVERABLE"
        assert critique.suggested_action == "RETRY"

        # And the Supervisor must re-dispatch the step with the critique appended to the prompt
        attempts = 0
        mock_router = AsyncMock(spec=ModelRouter)
        mock_router.route.return_value = "standard"
        mock_router.embed = AsyncMock(return_value=[0.1] * 64)

        async def fake_generate(prompt: str, **kwargs: Any) -> str:
            nonlocal attempts
            attempts += 1
            _ = kwargs
            if "[Critique]" in prompt or attempts > 1:
                return "```python\ndef connect_db():\n    return 'connected'\n```"
            return "```python\ndef connect_db(:\n    pass\n```"

        mock_router.generate = AsyncMock(side_effect=fake_generate)
        mock_router.generate_turn = AsyncMock(
            return_value=ToolCallTurn(text="Code generated", tool_calls=[])
        )

        supervisor = Supervisor(
            model_router=mock_router,
            safety_gate=SafetyGate(auto_approve=True),
        )
        test_graph = TaskGraph(goal="Generate database connection helper")
        test_graph.add_step(step)

        client = AsyncMock(spec=MCPClientManager)
        client.list_tools.return_value = []
        client.safety_gate = SafetyGate(auto_approve=True)

        state = SessionState(user_goal="Generate database connection helper")
        final_state = await supervisor.run(state, client, auto_approve=True, graph=test_graph)

        # And must not advance to subsequent steps until verification passes
        assert final_state.status == SessionStatus.COMPLETED
        assert attempts >= 2
        assert test_graph.get_step("code-1").status == StepStatus.COMPLETED
        assert "[Critique]" in step.description

    @pytest.mark.asyncio
    async def test_bdd_scenario_3_governed_peer_worker_delegation(self) -> None:
        """
        Scenario 3: SWEWorker delegates tool inspection directly to ToolWorker.
        """
        # Given "SWEWorker" is tasked with patching a configuration file
        swe_worker = SWEWorker()
        tool_worker = ToolWorker()
        delegator = WorkerDelegator(
            workers={"SWEWorker": swe_worker, "ToolWorker": tool_worker},
            max_depth=2,
            max_delegations_per_step=3,
        )
        swe_worker.delegator = delegator

        # Check initial DAG state baseline
        main_graph = TaskGraph(goal="Patch system configuration")
        s1 = TaskStep(
            step_id="patch-1",
            title="Patch system config",
            description="Patch system config",
            assigned_worker="SWEWorker",
        )
        main_graph.add_step(s1)
        initial_steps_snapshot = list(main_graph.steps.keys())

        # When "SWEWorker" requests the exact schema of the configuration tool
        result = await swe_worker.delegate(
            target="ToolWorker",
            query="Request the exact schema of the configuration tool for patch_config",
            context={"step_id": "patch-1"},
            depth=1,
        )

        # Then the Delegator must route the sub-query to "ToolWorker"
        assert result.success
        assert result.target == "ToolWorker"
        assert result.requester in {"CodeWorker", "SWEWorker"}

        # And return the schema directly to "SWEWorker" without resetting the main DAG state
        assert isinstance(result.data, dict)
        assert result.data["name"] == "patch_config"
        assert "inputSchema" in result.data or "parameters" in result.data
        assert list(main_graph.steps.keys()) == initial_steps_snapshot

        # And enforce a maximum sub-delegation depth of 2
        # Depth 2 is allowed
        res_depth_2 = await delegator.delegate(
            requester="SWEWorker",
            target="ToolWorker",
            query="Get schema",
            depth=2,
            step_id="patch-1",
        )
        assert res_depth_2.success
        assert res_depth_2.depth == 2

        # Depth 3 is blocked with DelegationDepthExceededError
        with pytest.raises(DelegationDepthExceededError):
            await delegator.delegate(
                requester="SWEWorker",
                target="ToolWorker",
                query="Get schema beyond depth limit",
                depth=3,
                step_id="patch-1",
            )

        # Max 3 delegations per step limit
        # Currently 2 delegations have occurred on step 'patch-1'. Next one makes 3 (allowed)
        res_3 = await delegator.delegate(
            requester="SWEWorker",
            target="ToolWorker",
            query="Get schema 3rd time",
            depth=1,
            step_id="patch-1",
        )
        assert res_3.success

        # 4th delegation on the same step raises DelegationLimitExceededError
        with pytest.raises(DelegationLimitExceededError):
            await delegator.delegate(
                requester="SWEWorker",
                target="ToolWorker",
                query="Get schema 4th time",
                depth=1,
                step_id="patch-1",
            )
