"""
tests/unit/test_transports.py — Unit tests for MCP transports and fleet management.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from helm.protocols.mcp_client import MCPClientManager, MCPToolDefinition
from helm.protocols.safety_gate import ActionSafety
from helm.protocols.tool_registry import ToolRegistry
from helm.protocols.transports.base import BaseTransport, RemoteTransportError
from helm.protocols.transports.sse import SSETransport
from helm.protocols.transports.stdio import StdioTransport


class TestBaseTransport:
    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError):
            BaseTransport("test")  # type: ignore[abstract]


class TestStdioTransport:
    @pytest.mark.asyncio
    async def test_stdio_lifecycle_mocked(self) -> None:
        transport = StdioTransport(server_name="stdio_srv", command="echo", args=["hello"])
        assert transport.server_name == "stdio_srv"
        assert transport.transport_type == "stdio"
        assert not transport.is_connected

        # Mock subprocess
        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdin.write = MagicMock()
        mock_proc.stdin.drain = AsyncMock()
        mock_proc.stdin.close = MagicMock()
        mock_proc.stdin.wait_closed = AsyncMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.readline = AsyncMock(
            side_effect=[
                b'{"jsonrpc": "2.0", "id": 1, "result": {"status": "ok"}}\n',
                b"",
            ]
        )
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.readline = AsyncMock(return_value=b"")
        mock_proc.returncode = None
        mock_proc.terminate = MagicMock()
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock(return_value=0)

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)):
            await transport.connect()
            assert transport.is_connected

            resp = await transport.send_request("test_method", {"key": "val"})
            assert resp["result"] == {"status": "ok"}

            await transport.disconnect()
            assert not transport.is_connected


class TestSSETransport:
    @pytest.mark.asyncio
    async def test_sse_auth_headers_and_initialization(self) -> None:
        headers = {"X-Custom": "val"}
        transport = SSETransport(
            server_name="remote_srv",
            endpoint="http://example.com/sse",
            headers=headers,
            auth_token="secret_token",  # noqa: S106
        )
        assert transport.server_name == "remote_srv"
        assert transport.transport_type == "sse"
        assert transport.headers["Authorization"] == "Bearer secret_token"
        assert transport.headers["X-Custom"] == "val"
        assert not transport.is_connected

    @pytest.mark.asyncio
    async def test_sse_connect_and_post_dispatch(self) -> None:
        transport = SSETransport(
            server_name="remote_ops",
            endpoint="http://remote.local:8080/sse",
        )

        mock_stream_lines = [
            "event: endpoint",
            "data: http://remote.local:8080/message",
            "",
        ]

        async def fake_aiter_lines() -> Any:
            for line in mock_stream_lines:
                yield line
            await asyncio.sleep(0.5)

        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__aenter__ = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = fake_aiter_lines
        mock_stream_ctx.__aenter__.return_value = mock_response
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=None)

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_stream_ctx)

        # Mock POST response for JSON-RPC call
        post_response = MagicMock()
        post_response.status_code = 200
        post_response.raise_for_status = MagicMock()
        post_response.json = MagicMock(
            return_value={"jsonrpc": "2.0", "id": 1, "result": {"output": "cloud_done"}}
        )
        mock_client.post = AsyncMock(return_value=post_response)
        mock_client.aclose = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            await transport.connect()
            await asyncio.sleep(0.05)
            assert transport.is_connected
            assert transport.post_endpoint == "http://remote.local:8080/message"

            resp = await transport.send_request("execute_remote", {"arg": 1})
            assert resp["result"] == {"output": "cloud_done"}
            assert transport.latency_ms >= 0

            await transport.disconnect()
            assert not transport.is_connected

    @pytest.mark.asyncio
    async def test_sse_reconnect_backoff_exhaustion(self) -> None:
        """Verify that 3 consecutive connection failures raise RemoteTransportError."""
        transport = SSETransport(
            server_name="failing_sse",
            endpoint="http://down.local/sse",
            max_retries=3,
            backoff_base_sec=0.01,
        )

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.stream = MagicMock(side_effect=ConnectionError("Server unreachable"))
            mock_client.aclose = AsyncMock()
            mock_client_cls.return_value = mock_client

            with pytest.raises(RemoteTransportError) as exc_info:
                await transport.connect()

            assert "after 3 attempts" in str(exc_info.value)
            assert "failing_sse" in str(exc_info.value)


class TestToolRegistry:
    def test_register_and_collision_mitigation(self) -> None:
        registry = ToolRegistry()

        t1 = MCPToolDefinition(
            name="query",
            description="Query local DB",
            server_name="local_db",
            transport="stdio",
            safety=ActionSafety.READ_ONLY,
        )
        t2 = MCPToolDefinition(
            name="query",
            description="Query remote DB",
            server_name="cloud_db",
            transport="sse",
            safety=ActionSafety.READ_ONLY,
        )

        registry.register_tool(t1)
        registry.register_tool(t2)

        # Both qualified names exist
        assert registry.get_tool("local_db.query") == t1
        assert registry.get_tool("cloud_db.query") == t2

        # Primary name points to the first registered one
        assert registry.get_tool("query") == t1

        # Check collision set
        assert "query" in registry.collisions

        # Server lookup
        assert registry.get_server_for_tool("local_db.query") == "local_db"
        assert registry.get_server_for_tool("cloud_db.query") == "cloud_db"

        # List tools
        all_tools = registry.list_tools()
        assert len(all_tools) == 2
        assert registry.list_tools("local_db") == [t1]
        assert registry.list_tools("cloud_db") == [t2]

        # Unregister server
        registry.unregister_server("local_db")
        assert registry.get_tool("local_db.query") is None
        assert registry.get_tool("cloud_db.query") == t2


class TestMCPClientManagerFleet:
    @pytest.mark.asyncio
    async def test_connect_all_and_route_call(self) -> None:
        manager = MCPClientManager()
        manager.safety_gate.auto_approve = True

        mock_stdio = MagicMock(spec=StdioTransport)
        mock_stdio.server_name = "memory"
        mock_stdio.transport_type = "stdio"
        mock_stdio.status = "connected"
        mock_stdio.latency_ms = 5.0
        mock_stdio.is_connected = True
        mock_stdio.connect = AsyncMock()
        mock_stdio.disconnect = AsyncMock()
        mock_stdio.send_request = AsyncMock(
            return_value={"result": {"tools": [{"name": "read_graph", "description": "Read"}]}}
        )

        mock_sse = MagicMock(spec=SSETransport)
        mock_sse.server_name = "cloud_ops"
        mock_sse.transport_type = "sse"
        mock_sse.status = "connected"
        mock_sse.latency_ms = 45.0
        mock_sse.is_connected = True
        mock_sse.connect = AsyncMock()
        mock_sse.disconnect = AsyncMock()
        mock_sse.send_request = AsyncMock(
            return_value={"result": {"tools": [{"name": "deploy", "description": "Deploy"}]}}
        )

        manager.register_transport(mock_stdio)
        manager.register_transport(mock_sse)

        # Discover tools
        tools = await manager.list_tools()
        assert len(tools) == 2
        tool_names = [t.name for t in tools]
        assert "read_graph" in tool_names
        assert "deploy" in tool_names

        # Status inspection
        status_memory = manager.get_server_status("memory")
        assert status_memory["transport"] == "stdio"
        assert status_memory["status"] == "connected"
        assert status_memory["latency_ms"] == 5.0

        status_cloud = manager.get_server_status("cloud_ops")
        assert status_cloud["transport"] == "sse"
        assert status_cloud["status"] == "connected"
        assert status_cloud["latency_ms"] == 45.0

        # Call tool routed to memory
        mock_stdio.send_request.return_value = {
            "result": {"content": [{"type": "text", "text": "entities_found"}]}
        }
        res1 = await manager.call_tool("read_graph", {})
        assert "entities_found" in str(res1)
        mock_stdio.send_request.assert_called()

        # Call tool routed to cloud_ops
        mock_sse.send_request.return_value = {
            "result": {"content": [{"type": "text", "text": "deployed_ok"}]}
        }
        res2 = await manager.call_tool("deploy", {})
        assert "deployed_ok" in str(res2)
        mock_sse.send_request.assert_called()

        await manager.disconnect_all()

    @pytest.mark.asyncio
    async def test_fleet_fault_isolation_when_server_offline(self) -> None:
        manager = MCPClientManager()
        manager.safety_gate.auto_approve = True

        healthy_stdio = MagicMock(spec=StdioTransport)
        healthy_stdio.server_name = "healthy_srv"
        healthy_stdio.transport_type = "stdio"
        healthy_stdio.status = "connected"
        healthy_stdio.latency_ms = 2.0
        healthy_stdio.is_connected = True
        healthy_stdio.connect = AsyncMock()
        healthy_stdio.disconnect = AsyncMock()
        healthy_stdio.send_request = AsyncMock(
            return_value={"result": {"tools": [{"name": "ping", "description": "Ping"}]}}
        )

        dead_sse = MagicMock(spec=SSETransport)
        dead_sse.server_name = "dead_remote"
        dead_sse.transport_type = "sse"
        dead_sse.status = "error"
        dead_sse.latency_ms = 0.0
        dead_sse.is_connected = False
        dead_sse.connect = AsyncMock(side_effect=RemoteTransportError("Server offline"))
        dead_sse.disconnect = AsyncMock()
        dead_sse.send_request = AsyncMock(side_effect=RemoteTransportError("Server unreachable"))

        manager.register_transport(healthy_stdio)
        manager.register_transport(dead_sse)

        tools = await manager.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "ping"

        status_healthy = manager.get_server_status("healthy_srv")
        assert status_healthy["status"] == "connected"

        status_dead = manager.get_server_status("dead_remote")
        assert status_dead["status"] == "error"

        # Calling tool on healthy server succeeds
        healthy_stdio.send_request.return_value = {
            "result": {"content": [{"type": "text", "text": "pong"}]}
        }
        res = await manager.call_tool("ping", {})
        assert "pong" in str(res)

        await manager.disconnect_all()
