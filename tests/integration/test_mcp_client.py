"""
tests/integration/test_mcp_client.py — Integration tests for MCPClientManager.

Spawns a mock stdio MCP server subprocess to verify:
1. Handshake and connection lifecycle
2. Schema discovery and classification
3. Tool execution (READ_ONLY vs MUTATING)
4. SafetyGate interception on mutating calls
5. Circuit breaker duplicate and step tracking integration
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from helm.core.circuit_breaker import CircuitBreaker
from helm.core.state import CircuitBreakerState
from helm.protocols.mcp_client import (
    ActionSafety,
    MCPClientManager,
    MCPConnectionError,
    ToolNotFoundError,
)
from helm.protocols.safety_gate import SafetyGate

# Python script implementing a compliant minimal stdio MCP server for tests
_MOCK_SERVER_SCRIPT = """
import sys
import json

while True:
    line = sys.stdin.readline()
    if not line:
        break
    try:
        req = json.loads(line)
    except Exception:
        continue

    method = req.get("method")
    req_id = req.get("id")

    if method == "initialize":
        resp = {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "test-mock-server", "version": "1.0.0"}
            }
        }
        sys.stdout.write(json.dumps(resp) + "\\n")
        sys.stdout.flush()

    elif method == "notifications/initialized":
        pass

    elif method == "tools/list":
        resp = {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                  {
                    "name": "get_system_time",
                    "description": "Returns current timestamp",
                    "inputSchema": {"type": "object", "properties": {}}
                  },
                  {
                    "name": "write_log_entry",
                    "description": "Appends entry to log",
                    "inputSchema": {"type": "object", "properties": {"entry": {"type": "string"}}}
                  }
                ]
            }
        }
        sys.stdout.write(json.dumps(resp) + "\\n")
        sys.stdout.flush()

    elif method == "tools/call":
        params = req.get("params", {})
        tool_name = params.get("name")
        args = params.get("arguments", {})

        if tool_name == "get_system_time":
            result = {"content": [{"type": "text", "text": "2026-11-07T12:00:00Z"}], "isError": False}
        elif tool_name == "write_log_entry":
            result = {"content": [{"type": "text", "text": f"Logged: {args.get('entry')}"}], "isError": False}
        else:
            result = {"content": [{"type": "text", "text": "Unknown"}], "isError": True}

        resp = {"jsonrpc": "2.0", "id": req_id, "result": result}
        sys.stdout.write(json.dumps(resp) + "\\n")
        sys.stdout.flush()
"""


@pytest.fixture
def mock_server_args():
    return [sys.executable, "-u", "-c", _MOCK_SERVER_SCRIPT]


class TestMCPClientIntegration:
    @pytest.mark.asyncio
    async def test_connect_and_list_tools(self, mock_server_args):
        gate = SafetyGate(auto_approve=True)
        async with MCPClientManager(safety_gate=gate) as client:
            await client.connect(
                server_name="test_server",
                command=mock_server_args[0],
                args=mock_server_args[1:],
            )

            tools = await client.list_tools()
            assert len(tools) == 2

            tool_map = {t.name: t for t in tools}
            assert "get_system_time" in tool_map
            assert "write_log_entry" in tool_map

            # Safety classifications
            assert tool_map["get_system_time"].safety == ActionSafety.READ_ONLY
            assert tool_map["write_log_entry"].safety == ActionSafety.MUTATING
            assert tool_map["get_system_time"].server_name == "test_server"

    @pytest.mark.asyncio
    async def test_call_read_only_tool_succeeds_automatically(self, mock_server_args):
        # Even with auto_approve=False, read-only tools must not prompt
        gate = SafetyGate(auto_approve=False)
        async with MCPClientManager(safety_gate=gate) as client:
            await client.connect(
                server_name="test_server",
                command=mock_server_args[0],
                args=mock_server_args[1:],
            )

            result = await client.call_tool("get_system_time", {})
            assert not result.get("is_error")
            content = result.get("content", [])
            assert len(content) > 0
            assert content[0]["text"] == "2026-11-07T12:00:00Z"

    @pytest.mark.asyncio
    async def test_call_mutating_tool_with_auto_approve(self, mock_server_args):
        gate = SafetyGate(auto_approve=True)
        async with MCPClientManager(safety_gate=gate) as client:
            await client.connect(
                server_name="test_server",
                command=mock_server_args[0],
                args=mock_server_args[1:],
            )

            result = await client.call_tool(
                "write_log_entry",
                {"entry": "test entry"},
                auto_approve=True,
            )
            assert not result.get("is_error")
            content = result.get("content", [])
            assert "Logged: test entry" in content[0]["text"]

    @pytest.mark.asyncio
    async def test_call_mutating_tool_denied_by_human(self, mock_server_args):
        # Create a mock gate that rejects mutating calls
        mock_gate = MagicMock(spec=SafetyGate)
        from helm.protocols.safety_gate import SafetyDecision, SafetyGateResult

        mock_gate.evaluate.return_value = SafetyGateResult(
            decision=SafetyDecision.REJECTED,
            reason="ActionDeniedByHuman: user clicked N",
        )

        async with MCPClientManager(safety_gate=mock_gate) as client:
            await client.connect(
                server_name="test_server",
                command=mock_server_args[0],
                args=mock_server_args[1:],
            )

            result = await client.call_tool(
                "write_log_entry",
                {"entry": "unauthorized edit"},
            )

            # Verifies denial structure returned to caller
            assert result.get("is_error") is True
            assert result.get("action_denied") is True
            assert "ActionDeniedByHuman" in result.get("error", "")

    @pytest.mark.asyncio
    async def test_call_unknown_tool_raises_error(self, mock_server_args):
        gate = SafetyGate(auto_approve=True)
        async with MCPClientManager(safety_gate=gate) as client:
            await client.connect(
                server_name="test_server",
                command=mock_server_args[0],
                args=mock_server_args[1:],
            )

            with pytest.raises(ToolNotFoundError, match="non_existent_tool"):
                await client.call_tool("non_existent_tool", {})

    @pytest.mark.asyncio
    async def test_connect_invalid_command_raises(self):
        gate = SafetyGate()
        client = MCPClientManager(safety_gate=gate)
        with pytest.raises(MCPConnectionError):
            await client.connect("invalid", "non_existent_executable_12345", [])

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, mock_server_args):
        """
        Verify that calls passing through or rejected by MCP client can be
        tracked and prevented from looping via CircuitBreaker.
        """
        cb = CircuitBreaker()
        cb_state = CircuitBreakerState(max_steps=5, duplicate_detection_enabled=True)
        gate = SafetyGate(auto_approve=True)

        async with MCPClientManager(safety_gate=gate) as client:
            await client.connect(
                server_name="test_server",
                command=mock_server_args[0],
                args=mock_server_args[1:],
            )

            # First call: evaluate circuit breaker then call tool
            args = {"entry": "entry 1"}
            cb_result = cb.check(cb_state, tool_name="write_log_entry", arguments=args)
            assert cb_result.allowed is True

            res1 = await client.call_tool("write_log_entry", args)
            assert not res1.get("is_error")

            # Duplicate call: circuit breaker immediately blocks repeated invocation
            cb_result2 = cb.check(cb_state, tool_name="write_log_entry", arguments=args)
            assert cb_result2.allowed is False
            assert cb_result2.record.was_blocked is True
            assert cb_result2.record.block_reason.value == "duplicate_call"


class TestMCPModuleReexports:
    def test_reexports(self):
        import helm.protocols.mcp as mcp_pkg
        import helm.protocols.mcp.client as mcp_client_mod
        import helm.protocols.mcp.types as mcp_types_mod

        assert mcp_pkg.MCPClientManager is MCPClientManager
        assert mcp_client_mod.MCPClientManager is MCPClientManager
        assert mcp_types_mod.ActionSafety is ActionSafety
