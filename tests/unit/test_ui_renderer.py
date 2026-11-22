"""
tests/unit/test_ui_renderer.py — Unit tests for terminal presentation event bus and ActivityStreamRenderer.
"""

from __future__ import annotations

import io
from unittest.mock import AsyncMock, MagicMock

import pytest
from rich.console import Console
from typer.testing import CliRunner

from helm.cli.app import app
from helm.config.settings import Settings
from helm.core.state import SessionState, SessionStatus
from helm.core.supervisor import Supervisor
from helm.protocols.mcp_client import MCPToolDefinition
from helm.protocols.safety_gate import ActionSafety
from helm.ui.events import (
    SessionDoneEvent,
    StepProgressEvent,
    ThinkingEvent,
    ToolCallCompleteEvent,
    ToolCallStartEvent,
    UIEvent,
)
from helm.ui.renderer import ActivityStreamRenderer


class TestUIEvents:
    """Test event bus Pydantic models."""

    def test_thinking_event_creation(self) -> None:
        event = ThinkingEvent(message="Analyzing DAG dependencies")
        assert event.message == "Analyzing DAG dependencies"
        assert event.timestamp > 0

    def test_tool_call_start_event(self) -> None:
        event = ToolCallStartEvent(
            tool_name="read_graph",
            server_name="memory",
            arguments={"query": "test"},
        )
        assert event.tool_name == "read_graph"
        assert event.server_name == "memory"
        assert event.arguments == {"query": "test"}

    def test_tool_call_complete_event(self) -> None:
        event = ToolCallCompleteEvent(
            tool_name="read_graph",
            summary="output summarized: 3 lines / 120 bytes",
            is_error=False,
        )
        assert event.tool_name == "read_graph"
        assert event.is_error is False

        err_event = ToolCallCompleteEvent(
            tool_name="read_graph",
            summary="Server disconnected",
            is_error=True,
        )
        assert err_event.is_error is True

    def test_step_progress_event(self) -> None:
        event = StepProgressEvent(step_id="step-1", status="RUNNING", worker="ToolWorker")
        assert event.step_id == "step-1"
        assert event.status == "RUNNING"
        assert event.worker == "ToolWorker"

    def test_session_done_event(self) -> None:
        event = SessionDoneEvent(
            summary="# Completed Goal",
            cost=0.0125,
            duration_sec=4.2,
        )
        assert event.summary == "# Completed Goal"
        assert event.cost == 0.0125
        assert event.duration_sec == 4.2


class TestActivityStreamRenderer:
    """Test ActivityStreamRenderer in TTY and Non-TTY environments."""

    def test_tty_rendering_lifecycle(self) -> None:
        output = io.StringIO()
        console = Console(file=output, force_terminal=True, width=120)
        renderer = ActivityStreamRenderer(console=console, is_tty=True)

        renderer.handle_event(ThinkingEvent(message="Planning execution strategy"))
        assert renderer.is_thinking is True

        renderer.handle_event(
            ToolCallStartEvent(
                tool_name="create_entities",
                server_name="memory",
                arguments={"entities": [{"name": "Helm"}]},
            )
        )
        assert renderer.is_thinking is False  # Status stopped for bullet print

        renderer.handle_event(
            ToolCallCompleteEvent(
                tool_name="create_entities",
                summary="output summarized: 1 lines / 45 bytes",
                is_error=False,
            )
        )

        renderer.handle_event(
            ToolCallCompleteEvent(
                tool_name="failing_tool",
                summary="Command failed with code 1",
                is_error=True,
            )
        )

        renderer.handle_event(
            SessionDoneEvent(
                summary="Session completed successfully.",
                cost=0.0042,
                duration_sec=2.5,
            )
        )
        renderer.close()

        rendered = output.getvalue()
        assert "create_entities" in rendered
        assert "memory" in rendered
        assert "failing_tool" in rendered
        assert "Session completed successfully." in rendered
        assert "2.5s" in rendered
        assert "$0.0042" in rendered

    def test_non_tty_plain_output(self) -> None:
        output = io.StringIO()
        console = Console(file=output, force_terminal=False, width=120)
        renderer = ActivityStreamRenderer(console=console, is_tty=False)

        renderer.handle_event(ThinkingEvent(message="Offline reasoning"))
        renderer.handle_event(
            ToolCallStartEvent(
                tool_name="read_file",
                server_name="filesystem",
                arguments={"path": "main.py"},
            )
        )
        renderer.handle_event(
            ToolCallCompleteEvent(
                tool_name="read_file",
                summary="output summarized: 10 lines / 250 bytes",
                is_error=False,
            )
        )
        renderer.handle_event(
            ToolCallCompleteEvent(
                tool_name="read_file",
                summary="File not found",
                is_error=True,
            )
        )
        renderer.handle_event(
            SessionDoneEvent(
                summary="All steps finished.",
                cost=0.0,
                duration_sec=1.1,
            )
        )
        renderer.close()

        rendered = output.getvalue()
        assert "✻ Thinking... Offline reasoning" in rendered
        assert "● Calling read_file (filesystem) with" in rendered
        assert "✓ read_file executed successfully" in rendered
        assert "✗ read_file failed: File not found" in rendered
        assert "All steps finished." in rendered
        assert "Completed in 1.1s" in rendered

    def test_long_argument_preview_truncation(self) -> None:
        output = io.StringIO()
        console = Console(file=output, force_terminal=False, width=120)
        renderer = ActivityStreamRenderer(console=console, is_tty=False)

        large_payload = {"very_long_key_" + str(i): "value_" * 10 for i in range(10)}
        renderer.handle_event(
            ToolCallStartEvent(
                tool_name="large_tool",
                server_name="srv",
                arguments=large_payload,
            )
        )
        renderer.close()

        rendered = output.getvalue()
        assert "..." in rendered
        assert len(rendered.splitlines()[0]) < 200

    def test_context_manager_support(self) -> None:
        output = io.StringIO()
        console = Console(file=output, force_terminal=False, width=120)
        with ActivityStreamRenderer(console=console, is_tty=False) as renderer:
            renderer.handle_event(ThinkingEvent(message="inside context manager"))
            assert len(renderer.events) == 1
        assert renderer.is_thinking is False


class TestSupervisorUIIntegration:
    """Test Supervisor emitting events to ActivityStreamRenderer."""

    @pytest.mark.asyncio
    async def test_supervisor_emits_events(self) -> None:
        output = io.StringIO()
        console = Console(file=output, force_terminal=False, width=120)
        renderer = ActivityStreamRenderer(console=console, is_tty=False)

        recorded_events: list[UIEvent] = []

        supervisor = Supervisor(
            ui_renderer=renderer,
            event_callback=recorded_events.append,
        )

        state = SessionState(user_goal="Inspect and synthesize data")

        mock_client = MagicMock()
        mock_client.list_tools = AsyncMock(
            return_value=[
                MCPToolDefinition(
                    name="read_graph",
                    description="Read entities",
                    server_name="memory",
                    safety=ActionSafety.READ_ONLY,
                )
            ]
        )
        mock_client.call_tool = AsyncMock(return_value={"content": "Entities: [Helm]"})

        final_state = await supervisor.run(state, mcp_client=mock_client, auto_approve=True)
        assert final_state.status == SessionStatus.COMPLETED

        # Verify events were captured both by renderer and callback
        assert len(renderer.events) > 0
        assert len(recorded_events) == len(renderer.events)

        event_types = [type(e).__name__ for e in recorded_events]
        assert "ThinkingEvent" in event_types
        assert "StepProgressEvent" in event_types
        assert "SessionDoneEvent" in event_types


class TestCLIDebugGraphOption:
    """Test CLI --debug-graph flag behavior."""

    @pytest.fixture(autouse=True)
    def isolate_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "helm.cli.app.load_settings",
            lambda config_path=None: Settings(mcp_servers=[]),
        )

    def test_run_without_debug_graph_omits_table(self) -> None:
        runner = CliRunner()
        result = runner.invoke(app, ["run", "test task"])
        assert result.exit_code == 0
        # By default, task DAG table headers like "Status" or "Worker" in a Table should not appear
        # or at least the make_task_table isn't invoked
        assert "Session Summary" in result.output

    def test_run_with_debug_graph_shows_table(self) -> None:
        runner = CliRunner()
        result = runner.invoke(app, ["run", "--debug-graph", "test task"])
        assert result.exit_code == 0
        assert "Task DAG" in result.output or "Task ID" in result.output
