"""
tests/unit/test_safety_gate.py — Unit tests for SafetyGate and classification.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from rich.console import Console

from helm.protocols.safety_gate import (
    ActionSafety,
    SafetyDecision,
    SafetyGate,
    classify_tool,
    is_mutating,
)


class TestSafetyClassification:
    def test_read_only_tools(self):
        read_only_tools = [
            "read_file",
            "fs_read",
            "list_directory",
            "directory_list",
            "get_weather",
            "search_codebase",
            "find_files",
            "cat_file",
            "grep_content",
            "git_status",
            "git_log",
            "git_diff",
            "fetch_webpage",
            "inspect_database",
        ]
        for tool in read_only_tools:
            assert classify_tool(tool) == ActionSafety.READ_ONLY, f"Expected {tool} to be READ_ONLY"
            assert not is_mutating(tool)

    def test_mutating_tools(self):
        mutating_tools = [
            "write_file",
            "fs_write",
            "create_file",
            "delete_file",
            "rm_file",
            "remove_directory",
            "edit_code",
            "patch_file",
            "update_record",
            "insert_row",
            "save_config",
            "move_file",
            "rename_symbol",
            "git_commit",
            "git_push",
            "git_reset",
            "http_post",
            "http_put",
            "drop_table",
            "kill_process",
        ]
        for tool in mutating_tools:
            assert classify_tool(tool) == ActionSafety.MUTATING, f"Expected {tool} to be MUTATING"
            assert is_mutating(tool)

    def test_shell_execution_is_mutating_by_default(self):
        assert classify_tool("bash_exec", {"command": "echo hello"}) == ActionSafety.MUTATING
        assert classify_tool("shell_exec", {"command": "pwd"}) == ActionSafety.MUTATING
        assert classify_tool("run_terminal", {"command": "ls"}) == ActionSafety.MUTATING

    def test_shell_execution_with_mutating_patterns(self):
        assert classify_tool("bash_exec", {"command": "rm -rf /tmp/data"}) == ActionSafety.MUTATING
        assert classify_tool("shell", {"cmd": "git push origin main"}) == ActionSafety.MUTATING
        assert (
            classify_tool("exec", {"command": "curl -X POST http://api.local"})
            == ActionSafety.MUTATING
        )

    def test_compound_verbs_favor_safety(self):
        # A tool that both gets and deletes must be classified as mutating
        assert classify_tool("get_and_delete_user") == ActionSafety.MUTATING


class TestSafetyGateEvaluation:
    def setup_method(self):
        self.console = MagicMock(spec=Console)
        self.gate = SafetyGate(console=self.console)

    def test_read_only_tool_auto_approved(self):
        result = self.gate.evaluate("read_file", {"path": "README.md"})
        assert result.is_approved
        assert result.decision == SafetyDecision.APPROVED
        assert "READ_ONLY" in result.reason

    def test_mutating_tool_with_auto_approve_flag(self):
        result = self.gate.evaluate("delete_file", {"path": "temp.txt"}, auto_approve=True)
        assert result.is_approved
        assert result.decision == SafetyDecision.APPROVED
        assert "Auto-approved" in result.reason

    def test_mutating_tool_with_gate_level_auto_approve(self):
        auto_gate = SafetyGate(console=self.console, auto_approve=True)
        result = auto_gate.evaluate("file_write", {"path": "main.py"})
        assert result.is_approved
        assert result.decision == SafetyDecision.APPROVED

    @patch("select.select")
    @patch("sys.stdin.readline", return_value="y\n")
    def test_mutating_tool_approved_by_user(self, _mock_readline, mock_select):
        mock_select.return_value = ([True], [], [])
        result = self.gate.evaluate(
            "file_write",
            {"path": "foo.txt", "content": "bar"},
            summary="Writing 3 bytes to foo.txt",
        )
        assert result.is_approved
        assert result.decision == SafetyDecision.APPROVED
        assert "Explicitly approved" in result.reason

    @patch("select.select")
    @patch("sys.stdin.readline", return_value="n\n")
    def test_mutating_tool_denied_by_user(self, _mock_readline, mock_select):
        mock_select.return_value = ([True], [], [])
        result = self.gate.evaluate(
            "bash_exec",
            {"command": "rm -rf .git"},
            summary="Deleting git directory",
        )
        assert not result.is_approved
        assert result.decision == SafetyDecision.REJECTED
        assert "ActionDeniedByHuman" in result.reason

    @patch("select.select")
    def test_mutating_tool_timeout_defaults_to_deny(self, mock_select):
        mock_select.return_value = ([], [], [])  # timed out
        result = self.gate.evaluate("git_push", {"branch": "main"})
        assert not result.is_approved
        assert result.decision == SafetyDecision.REJECTED

    def test_request_permission_read_only(self):
        approved = self.gate.request_permission("read_file", {"path": "foo.txt"})
        assert approved is True

    def test_request_permission_mutating_auto_approved(self):
        approved = self.gate.request_permission(
            "delete_entities", {"entityNames": ["user_profile"]}, auto_approve=True
        )
        assert approved is True

    @patch("select.select")
    @patch("sys.stdin.readline", return_value="y\n")
    def test_request_permission_mutating_interactive_approved(self, _mock_readline, mock_select):
        mock_select.return_value = ([True], [], [])
        approved = self.gate.request_permission(
            "delete_entities", {"entityNames": ["user_profile"]}
        )
        assert approved is True

    @patch("select.select")
    @patch("sys.stdin.readline", return_value="n\n")
    def test_request_permission_mutating_interactive_denied(self, _mock_readline, mock_select):
        mock_select.return_value = ([True], [], [])
        approved = self.gate.request_permission(
            "delete_entities", {"entityNames": ["user_profile"]}
        )
        assert approved is False


class TestCliSafetyGateWrapper:
    def test_confirm_mutation_delegates(self):
        from helm.cli.safety_gate import confirm_mutation

        console = MagicMock(spec=Console)
        # Read-only auto-approved
        assert confirm_mutation(console, tool_name="read_file", arguments={}) is True
        # Mutating with auto-approve
        assert (
            confirm_mutation(console, tool_name="delete_file", arguments={}, auto_approve=True)
            is True
        )
