"""
tests/unit/test_supervisor.py — Unit tests for Supervisor orchestration and DAG execution.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from helm.config.settings import Settings
from helm.core.circuit_breaker import CircuitBreaker
from helm.core.state import SessionState, SessionStatus, TaskStatus
from helm.core.supervisor import Supervisor
from helm.protocols.mcp_client import MCPClientManager, MCPToolDefinition
from helm.protocols.safety_gate import ActionSafety, SafetyGate


@pytest.fixture
def mock_mcp_client():
    client = AsyncMock(spec=MCPClientManager)
    client.list_tools.return_value = [
        MCPToolDefinition(
            name="read_file",
            description="Read content from file",
            server_name="fs",
            safety=ActionSafety.READ_ONLY,
        ),
        MCPToolDefinition(
            name="write_file",
            description="Write content to file",
            server_name="fs",
            safety=ActionSafety.MUTATING,
        ),
    ]
    client.call_tool.return_value = {
        "content": [{"type": "text", "text": "File written successfully"}],
        "is_error": False,
    }
    return client


class TestSupervisor:
    @pytest.mark.asyncio
    async def test_heuristic_decomposition(self):
        supervisor = Supervisor()
        graph = await supervisor.decompose_goal("Refactor auth module")
        assert len(graph.steps) >= 2
        assert graph.is_successful() is False  # not executed yet
        order = graph.get_topological_order()
        assert len(order) == len(graph.steps)

    @pytest.mark.asyncio
    async def test_supervisor_execution_loop(self, mock_mcp_client):
        supervisor = Supervisor()
        state = SessionState(user_goal="Refactor auth module")

        result_state = await supervisor.run(state, mock_mcp_client, auto_approve=True)

        assert result_state.status == SessionStatus.COMPLETED
        assert result_state.final_answer is not None
        assert len(result_state.tasks) >= 2
        # All tasks should be completed
        assert all(t.status == TaskStatus.COMPLETED for t in result_state.tasks)

    @pytest.mark.asyncio
    async def test_circuit_breaker_halts_execution(self, mock_mcp_client):
        # Configure circuit breaker with ceiling of 0 steps
        cb = CircuitBreaker()
        settings = Settings()
        settings.circuit_breaker.max_step_ceiling = 0

        supervisor = Supervisor(settings=settings, circuit_breaker=cb)
        state = SessionState(user_goal="Loop test")
        state.circuit_breaker.max_steps = 0

        result_state = await supervisor.run(state, mock_mcp_client, auto_approve=True)

        assert result_state.status in (SessionStatus.ABORTED, SessionStatus.COMPLETED)
        assert "circuit breaker" in (result_state.final_answer or "").lower()

    @pytest.mark.asyncio
    async def test_gemini_decomposition_parsing(self):
        from unittest.mock import MagicMock

        supervisor = Supervisor(api_key="fake-key")

        mock_gemini_response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": (
                                    '{"steps": ['
                                    '{"step_id": "1", "title": "Inspect code", "description": "Check auth files", "target_tool": "read_file", "depends_on": []},'
                                    '{"step_id": "2", "title": "Apply refactor", "description": "Write updated code", "target_tool": "write_file", "depends_on": ["1"]}'
                                    "]}"
                                )
                            }
                        ]
                    }
                }
            ]
        }

        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_gemini_response
        mock_resp.raise_for_status.return_value = None

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp

            graph = await supervisor.decompose_goal("Refactor auth with Gemini")
            assert len(graph.steps) == 2
            assert "1" in graph.steps
            assert "2" in graph.steps
            assert graph.steps["2"].depends_on == ["1"]

    @pytest.mark.asyncio
    async def test_worker_assignment_and_step_metrics(self, mock_mcp_client):
        supervisor = Supervisor()
        state = SessionState(user_goal="Inspect and update code")

        result_state = await supervisor.run(state, mock_mcp_client, auto_approve=True)

        assert result_state.status == SessionStatus.COMPLETED
        # Verify step metrics are tracked
        assert result_state.circuit_breaker.total_steps == len(result_state.tasks)
        for task in result_state.tasks:
            assert task.assigned_worker in (
                "ToolWorker",
                "ExecutorWorker",
                "ResearchWorker",
                "CodeWorker",
            )
            assert task.step_count >= 1

    @pytest.mark.asyncio
    async def test_tool_call_invocation_and_safety_gate(self, mock_mcp_client):
        from helm.core.model_router import ModelRouter, ModelTurn, ToolCall

        router = AsyncMock(spec=ModelRouter)
        router.generate_turn.return_value = ModelTurn(
            text="I will write to the file",
            tool_calls=[
                ToolCall(name="write_file", args={"path": "main.py", "content": "print(1)"})
            ],
        )
        router.generate.return_value = "Verified that main.py was updated"
        router.route.return_value = "standard"

        safety_gate = SafetyGate()

        supervisor = Supervisor(model_router=router, safety_gate=safety_gate)
        state = SessionState(user_goal="Write code to main.py")

        result_state = await supervisor.run(state, mock_mcp_client, auto_approve=True)

        assert result_state.status == SessionStatus.COMPLETED
        # Verify mcp_client.call_tool was executed
        mock_mcp_client.call_tool.assert_awaited()
        # Verify result contains verification
        assert any(
            "Verified that main.py" in (t.result.output if t.result else "")
            or "Verified that main.py" in str(t.description)
            or "Verified that main.py" in (result_state.final_answer or "")
            for t in result_state.tasks
        ) or "Verified that main.py" in (result_state.final_answer or "")

    @pytest.mark.asyncio
    async def test_mutating_tool_interactive_prompt_and_approval(self, mock_mcp_client):
        gate = SafetyGate()
        supervisor = Supervisor(safety_gate=gate)
        state = SessionState(user_goal="Write code to main.py")

        with (
            patch("sys.stdin.readline", return_value="y\n"),
            patch("select.select", return_value=([True], [], [])),
        ):
            result = await supervisor.run(state, mock_mcp_client, auto_approve=False)
            assert result.status == SessionStatus.COMPLETED
            mock_mcp_client.call_tool.assert_awaited()

    @pytest.mark.asyncio
    async def test_mutating_tool_interactive_prompt_and_denial(self, mock_mcp_client):
        gate = SafetyGate()
        supervisor = Supervisor(safety_gate=gate)
        state = SessionState(user_goal="Write code to main.py")

        with (
            patch("sys.stdin.readline", return_value="n\n"),
            patch("select.select", return_value=([True], [], [])),
        ):
            result = await supervisor.run(state, mock_mcp_client, auto_approve=False)
            assert result.status == SessionStatus.ABORTED

    @pytest.mark.asyncio
    async def test_live_tool_dispatch_and_worker_assignment(self, mock_mcp_client):
        from helm.core.model_router import ModelRouter, ModelTurn, ToolCall

        router = AsyncMock(spec=ModelRouter)
        router.generate_turn.return_value = ModelTurn(
            text="Calling write_file",
            tool_calls=[
                ToolCall(name="write_file", args={"path": "output.py", "content": "x = 42"})
            ],
        )
        router.generate.return_value = "Verified write_file succeeded."
        router.route.return_value = "standard"

        gate = SafetyGate()
        with patch.object(gate, "request_permission", wraps=gate.request_permission) as spy_gate:
            supervisor = Supervisor(model_router=router, safety_gate=gate)
            state = SessionState(user_goal="Write code to output.py")

            result = await supervisor.run(state, mock_mcp_client, auto_approve=True)

            assert result.status == SessionStatus.COMPLETED
            # 1. SafetyGate.request_permission was invoked
            spy_gate.assert_called()
            # 2. MCP client call_tool was executed with arguments
            mock_mcp_client.call_tool.assert_awaited_with(
                "write_file",
                {"path": "output.py", "content": "x = 42"},
                auto_approve=True,
                summary="Executing write_file for Execute resolution action",
            )
            # 3. Steps were recorded and worker assigned
            assert result.circuit_breaker.total_steps >= 2
            execution_task = next(t for t in result.tasks if t.task_id == "step-2")
            assert execution_task.worker == "ToolWorker"
            assert execution_task.assigned_worker == "ToolWorker"
            assert "Verified write_file succeeded." in (
                execution_task.result.output if execution_task.result else ""
            )

    @pytest.mark.asyncio
    async def test_bdd_scenario_1_mutating_tool_interactive_approval(self):
        """
        Scenario 1: Supervisor executes a mutating tool requiring human authorization.
        """
        client = AsyncMock(spec=MCPClientManager)
        client.list_tools.return_value = [
            MCPToolDefinition(
                name="create_entities",
                description="Create multiple new entities in the knowledge graph",
                input_schema={"type": "object", "properties": {"entities": {"type": "array"}}},
                server_name="memory",
                safety=ActionSafety.MUTATING,
            ),
            MCPToolDefinition(
                name="add_observations",
                description="Add new observations to existing entities in the knowledge graph",
                input_schema={"type": "object", "properties": {"observations": {"type": "array"}}},
                server_name="memory",
                safety=ActionSafety.MUTATING,
            ),
            MCPToolDefinition(
                name="read_graph",
                description="Read the entire knowledge graph",
                input_schema={"type": "object", "properties": {}},
                server_name="memory",
                safety=ActionSafety.READ_ONLY,
            ),
        ]
        client.call_tool.return_value = {
            "content": [{"type": "text", "text": "Created entity ProjectHelm successfully"}],
            "is_error": False,
        }

        gate = SafetyGate(auto_approve=False)
        supervisor = Supervisor(safety_gate=gate)
        goal = (
            "Store an entity named ProjectHelm with entityType Software and an "
            "observation that it is written in Python"
        )
        state = SessionState(user_goal=goal)

        # 1. Goal decomposition produces actionable step targeting create_entities or add_observations
        graph = await supervisor.decompose_goal(goal)
        actionable_steps = [
            s
            for s in graph.steps.values()
            if s.target_tool in ("create_entities", "add_observations")
        ]
        assert len(actionable_steps) >= 1

        # 2. Interactive execution pauses at HITL safety prompt and confirms with 'y'
        with (
            patch("sys.stdin.readline", return_value="y\n"),
            patch("select.select", return_value=([True], [], [])),
        ):
            result = await supervisor.run(state, client, auto_approve=False)

        # 3. Assertions matching BDD Scenario 1
        assert result.status == SessionStatus.COMPLETED
        client.call_tool.assert_awaited()
        called_tool_name = client.call_tool.await_args[0][0]
        assert called_tool_name in ("create_entities", "add_observations")
        assert result.circuit_breaker.total_steps >= 1

        for task in result.tasks:
            assert task.assigned_worker not in ("", "-", None)
            assert task.worker not in ("", "-", None)
            assert task.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_bdd_scenario_2_safety_gate_denial_clean_abort(self):
        """
        Scenario 2: Human operator denies execution of a destructive action.
        """
        client = AsyncMock(spec=MCPClientManager)
        client.list_tools.return_value = [
            MCPToolDefinition(
                name="delete_entities",
                description="Delete entities and their associated relations",
                input_schema={"type": "object", "properties": {"entityNames": {"type": "array"}}},
                server_name="memory",
                safety=ActionSafety.MUTATING,
            ),
            MCPToolDefinition(
                name="read_graph",
                description="Read graph",
                input_schema={},
                server_name="memory",
                safety=ActionSafety.READ_ONLY,
            ),
        ]

        gate = SafetyGate(auto_approve=False)
        supervisor = Supervisor(safety_gate=gate)
        goal = "Delete the entity named ObsoleteEntity from the knowledge graph"
        state = SessionState(user_goal=goal)

        # Developer inputs 'n' -> denied
        with (
            patch("sys.stdin.readline", return_value="n\n"),
            patch("select.select", return_value=([True], [], [])),
        ):
            result = await supervisor.run(state, client, auto_approve=False)

        # MCP client must NOT dispatch the tool call
        client.call_tool.assert_not_awaited()
        # Step status transitions to failed
        failed_tasks = [t for t in result.tasks if t.status == TaskStatus.FAILED]
        assert len(failed_tasks) >= 1
        # Session terminates cleanly with ABORTED status without exception
        assert result.status == SessionStatus.ABORTED

    @pytest.mark.asyncio
    async def test_bdd_scenario_3_read_only_tool_execution_bypasses_prompt(self):
        """
        Scenario 3: Supervisor autonomously executes safe read-only MCP actions.
        """
        client = AsyncMock(spec=MCPClientManager)
        client.list_tools.return_value = [
            MCPToolDefinition(
                name="create_entities",
                description="Create entities",
                input_schema={},
                server_name="memory",
                safety=ActionSafety.MUTATING,
            ),
            MCPToolDefinition(
                name="read_graph",
                description="Read the entire knowledge graph",
                input_schema={},
                server_name="memory",
                safety=ActionSafety.READ_ONLY,
            ),
            MCPToolDefinition(
                name="search_nodes",
                description="Search for nodes",
                input_schema={},
                server_name="memory",
                safety=ActionSafety.READ_ONLY,
            ),
        ]
        json_output = '{"entities": [{"name": "ProjectHelm", "entityType": "Software"}]}'
        client.call_tool.return_value = {
            "content": [{"type": "text", "text": json_output}],
            "is_error": False,
        }

        gate = SafetyGate(auto_approve=False)
        supervisor = Supervisor(safety_gate=gate)
        goal = "Read the knowledge graph and tell me what entities are currently saved"
        state = SessionState(user_goal=goal)

        # Running without stdin patch verifies that read-only execution never prompts
        result = await supervisor.run(state, client, auto_approve=False)

        assert result.status == SessionStatus.COMPLETED
        client.call_tool.assert_awaited()
        called_tool_name = client.call_tool.await_args[0][0]
        assert called_tool_name in ("read_graph", "search_nodes")
        assert result.circuit_breaker.total_steps >= 1

        combined_outputs = (
            " ".join(t.result.output for t in result.tasks if t.result)
            + " "
            + (result.final_answer or "")
        )
        assert "ProjectHelm" in combined_outputs

    @pytest.mark.asyncio
    async def test_bdd_scenario_4_anti_hallucination_and_anti_stubbing(self):
        """
        Scenario 4: Verification of actual execution vs template stubs.
        """
        from helm.cli.rendering.panels import make_task_table

        client = AsyncMock(spec=MCPClientManager)
        client.list_tools.return_value = [
            MCPToolDefinition(
                name="read_graph",
                description="Read knowledge graph",
                input_schema={},
                server_name="memory",
                safety=ActionSafety.READ_ONLY,
            ),
        ]
        client.call_tool.return_value = {
            "content": [{"type": "text", "text": '{"status": "ok"}'}],
            "is_error": False,
        }

        gate = SafetyGate(auto_approve=True)
        supervisor = Supervisor(safety_gate=gate)
        state = SessionState(user_goal="Inspect and read knowledge graph")

        result = await supervisor.run(state, client, auto_approve=True)
        assert result.status == SessionStatus.COMPLETED

        # 1. Step result must NEVER equal template "Completed analysis for '{description}'"
        for task in result.tasks:
            output = (task.result.output if task.result else "") or ""
            assert output != f"Completed analysis for '{task.description}'"
            assert not output.startswith("Completed analysis for '")

        # 2. TaskGraph UI table worker column must display assigned worker name instead of "-"
        from rich.console import Console

        from helm.cli.rendering.themes import HELM_THEME

        table = make_task_table(result)
        console = Console(theme=HELM_THEME, record=True, width=120)
        console.print(table)
        rendered = console.export_text()
        for task in result.tasks:
            assert task.assigned_worker not in ("", "-", None)
            assert task.worker not in ("", "-", None)
            assert task.assigned_worker in rendered
        # Ensure no solitary '-' column indicator in table
        assert " | - " not in rendered
        assert " - " not in rendered

        # 3. Steps used accurately counts actions
        assert result.circuit_breaker.total_steps >= 1
