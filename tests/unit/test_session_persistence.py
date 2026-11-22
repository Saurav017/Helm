"""
tests/unit/test_session_persistence.py — Unit tests for state & graph serialization and checkpointing.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from typer.testing import CliRunner

from helm.cli.app import app
from helm.core.state import SessionState, Task, TaskResult, TaskStatus
from helm.core.task_graph import StepStatus, TaskGraph, TaskStep


class TestSessionPersistence:
    def test_task_graph_serialization_roundtrip(self):
        graph = TaskGraph(goal="Build and verify API")
        s1 = TaskStep(
            step_id="step-1", title="Step 1", status=StepStatus.COMPLETED, output="Output 1"
        )
        s2 = TaskStep(
            step_id="step-2",
            title="Step 2",
            status=StepStatus.PENDING,
            depends_on=["step-1"],
        )
        graph.add_step(s1)
        graph.add_step(s2)

        data = graph.to_dict()
        assert isinstance(data, dict)
        assert data["goal"] == "Build and verify API"
        assert len(data["steps"]) == 2

        restored = TaskGraph.from_dict(data)
        assert restored.goal == graph.goal
        assert restored.steps["step-1"].output == "Output 1"
        assert restored.steps["step-2"].status == StepStatus.PENDING
        assert restored.steps["step-2"].depends_on == ["step-1"]

    def test_session_state_save_and_load(self):
        with TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test_session.json"

            state = SessionState(session_id="sess-xyz", user_goal="Serialize state test")
            state.add_message("user", "Hello")
            state.add_message("assistant", "Hi there")

            task = Task(
                task_id="t-1",
                description="Test task",
                status=TaskStatus.COMPLETED,
                assigned_worker="ToolWorker",
                result=TaskResult(task_id="t-1", status=TaskStatus.COMPLETED, output="Task done"),
            )
            state.add_task(task)
            state.metadata["custom_key"] = "custom_value"

            saved_path = state.save(file_path)
            assert saved_path == file_path
            assert file_path.exists()

            loaded = SessionState.load(file_path)
            assert loaded.session_id == "sess-xyz"
            assert loaded.user_goal == "Serialize state test"
            assert len(loaded.messages) == 2
            assert len(loaded.tasks) == 1
            assert loaded.tasks[0].assigned_worker == "ToolWorker"
            assert loaded.tasks[0].result is not None
            assert loaded.tasks[0].result.output == "Task done"
            assert loaded.metadata.get("custom_key") == "custom_value"

    def test_session_state_default_save_path(self):
        with TemporaryDirectory() as tmpdir:
            import os

            orig_cwd = Path.cwd()
            try:
                os.chdir(tmpdir)
                state = SessionState(session_id="custom-123", user_goal="Check default path")
                target = state.save()
                assert target == Path(".helm/sessions/custom-123.json")
                assert target.exists()
                loaded = SessionState.load(target)
                assert loaded.session_id == "custom-123"
            finally:
                os.chdir(orig_cwd)

    def test_cli_resume_missing_file_fails(self):
        runner = CliRunner()
        result = runner.invoke(app, ["run", "--resume", "non_existent_session_999"])
        assert result.exit_code != 0
        assert "Session checkpoint not found" in result.output
