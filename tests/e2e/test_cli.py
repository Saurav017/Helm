"""
tests/e2e/test_cli.py — End-to-end CLI tests via Typer's test runner.

Uses ``typer.testing.CliRunner`` to exercise the full CLI surface without
spawning subprocess, allowing proper coverage collection.

Covers:
- ``helmx run "<prompt>"`` produces expected exit code and Rich output
- ``helmx tools`` output indicates no servers configured
- ``helmx config`` prints Settings fields
- ``helmx version`` exits cleanly
- ``helmx interactive`` prints not-implemented notice
- Missing prompt → non-zero exit (Typer handles this)
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from helm.cli.app import app
from helm.config.settings import Settings

runner = CliRunner()


class TestRunCommand:
    @pytest.fixture(autouse=True)
    def isolate_mcp_servers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Isolate tests from local helmx.toml MCP servers to prevent spawning external processes."""
        monkeypatch.setattr(
            "helm.cli.app.load_settings",
            lambda config_path=None: Settings(mcp_servers=[]),
        )

    def test_run_exits_zero(self):
        result = runner.invoke(app, ["run", "write a hello-world script"])
        assert result.exit_code == 0, result.output

    def test_run_prints_banner(self):
        result = runner.invoke(app, ["run", "test prompt"])
        assert "HELM" in result.output

    def test_run_shows_session_summary(self):
        result = runner.invoke(app, ["run", "test prompt"])
        assert "Session Summary" in result.output

    def test_run_shows_goal_in_panel(self):
        prompt = "refactor auth module"
        result = runner.invoke(app, ["run", prompt])
        assert prompt in result.output

    def test_run_verbose_flag_accepted(self):
        result = runner.invoke(app, ["run", "--verbose", "test"])
        assert result.exit_code == 0

    def test_run_auto_approve_flag_accepted(self):
        result = runner.invoke(app, ["run", "--yes", "test mutation"])
        assert result.exit_code == 0
        result_long = runner.invoke(app, ["run", "--auto-approve", "test mutation"])
        assert result_long.exit_code == 0

    def test_run_no_prompt_exits_nonzero(self):
        result = runner.invoke(app, ["run"])
        assert result.exit_code != 0


class TestToolsCommand:
    def test_tools_indicates_no_servers(self, monkeypatch):
        from helm.config.settings import Settings

        monkeypatch.setattr("helm.cli.app.load_settings", lambda config_path=None: Settings())
        result = runner.invoke(app, ["tools"])
        assert result.exit_code == 0
        assert "No MCP servers" in result.output or "helmx.toml" in result.output

    def test_tools_displays_table_when_tools_found(self, monkeypatch):
        from helm.config.settings import MCPServerSettings, Settings
        from helm.protocols.mcp_client import MCPToolDefinition
        from helm.protocols.safety_gate import ActionSafety

        sample_tools = [
            MCPToolDefinition(
                name="sample_read",
                description="Sample read-only tool",
                server_name="test_srv",
                safety=ActionSafety.READ_ONLY,
            ),
            MCPToolDefinition(
                name="sample_write",
                description="Sample mutating tool",
                server_name="test_srv",
                safety=ActionSafety.MUTATING,
            ),
        ]
        settings = Settings(
            mcp_servers=[MCPServerSettings(name="test_srv", transport="stdio", command="dummy")]
        )

        async def mock_fetch(_servers):
            return sample_tools

        monkeypatch.setattr("helm.cli.app.load_settings", lambda config_path=None: settings)
        monkeypatch.setattr("helm.cli.app._fetch_tools", mock_fetch)

        result = runner.invoke(app, ["tools"])
        assert result.exit_code == 0
        assert "sample_read" in result.output
        assert "sample_write" in result.output
        assert "READ_ONLY" in result.output
        assert "MUTATING" in result.output


class TestConfigCommand:
    def test_config_exits_zero(self):
        result = runner.invoke(app, ["config"])
        assert result.exit_code == 0

    def test_config_prints_settings(self):
        result = runner.invoke(app, ["config"])
        # Pydantic Pretty output should contain at least these keys
        assert "supervisor" in result.output or "max_steps" in result.output


class TestVersionCommand:
    def test_version_exits_zero(self):
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0

    def test_version_prints_helm(self):
        result = runner.invoke(app, ["version"])
        assert "HELM" in result.output


class TestInteractiveCommand:
    def test_interactive_exits_zero(self):
        result = runner.invoke(app, ["interactive"])
        assert result.exit_code == 0

    def test_interactive_prints_not_implemented(self):
        result = runner.invoke(app, ["interactive"])
        assert "not yet implemented" in result.output.lower() or "Milestone" in result.output
