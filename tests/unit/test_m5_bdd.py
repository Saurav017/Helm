"""
tests/unit/test_m5_bdd.py — BDD feature tests for Milestone 5:
Remote MCP Transports (SSE/HTTP) and Multi-Server Fleet Orchestration.

Feature: Remote MCP Transports (SSE/HTTP) and Multi-Server Fleet Orchestration
  As the Helm control plane
  I want to connect to remote MCP servers over HTTP/SSE alongside local stdio processes
  So that Helm can orchestrate distributed tool fleets, mitigate naming collisions,
  and handle network transients with exponential backoff.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from helm.config.settings import MCPServerSettings
from helm.core.circuit import CircuitBreaker
from helm.protocols.mcp_client import MCPClientManager
from helm.protocols.transports.base import RemoteTransportError
from helm.protocols.transports.sse import SSETransport
from helm.protocols.transports.stdio import StdioTransport


class TestMilestone5BDD:
    @pytest.mark.asyncio
    async def test_bdd_scenario_1_sse_connection_and_tool_discovery(self) -> None:  # noqa: PLR0915
        """
        Scenario 1: SSE transport connection and tool discovery from remote endpoint
        """
        # Given an MCP server configured with transport 'sse', endpoint, and auth token
        server_setting = MCPServerSettings(
            name="cloud_agent",
            transport="sse",
            endpoint="https://cloud.helm.run/sse",
            auth_token="bearer-token-12345",  # noqa: S106
            headers={"X-Environment": "production"},
        )

        mock_stream_lines = [
            "event: endpoint",
            "data: https://cloud.helm.run/messages",
            "",
            "event: message",
            (
                'data: {"jsonrpc": "2.0", "method": "notifications/tools/list_changed",'
                ' "params": {}}'
            ),
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

        # Mock dynamic JSON-RPC responses for initialize and tools/list
        async def fake_post(_url: str, **kwargs: Any) -> Any:
            msg = kwargs.get("json", {})
            req_id = msg.get("id", 1)
            method = msg.get("method", "")
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            resp.content = b"{}"
            if method == "initialize":
                resp.json = MagicMock(
                    return_value={
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {},
                            "serverInfo": {"name": "cloud_agent", "version": "1.0"},
                        },
                    }
                )
            elif method == "tools/list":
                resp.json = MagicMock(
                    return_value={
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "tools": [
                                {
                                    "name": "cloud_build",
                                    "description": "Trigger remote cloud build",
                                    "inputSchema": {"type": "object"},
                                }
                            ]
                        },
                    }
                )
            else:
                resp.json = MagicMock(return_value={"jsonrpc": "2.0", "id": req_id, "result": {}})
            return resp

        mock_client.post = AsyncMock(side_effect=fake_post)
        mock_client.aclose = AsyncMock()

        manager = MCPClientManager()

        # When manager connects to all configured servers
        with patch("httpx.AsyncClient", return_value=mock_client):
            await manager.connect_all([server_setting])
            await asyncio.sleep(0.05)

            # Then transport is registered as sse with proper auth headers
            transport = manager.get_transport("cloud_agent")
            assert isinstance(transport, SSETransport)
            assert transport.transport_type == "sse"
            assert transport.headers["Authorization"] == "Bearer bearer-token-12345"
            assert transport.headers["X-Environment"] == "production"
            assert transport.post_endpoint == "https://cloud.helm.run/messages"

            # And tools are discovered with server_name and transport metadata
            tools = await manager.list_tools()
            assert len(tools) == 1
            tool = tools[0]
            assert tool.name == "cloud_build"
            assert tool.server_name == "cloud_agent"
            assert tool.transport == "sse"

            # And server status reflects connection
            status = manager.get_server_status("cloud_agent")
            assert status["transport"] == "sse"
            assert status["status"] == "connected"

            await manager.disconnect_all()

    @pytest.mark.asyncio
    async def test_bdd_scenario_2_multiserver_namespacing_and_concurrent_dispatch(
        self,
    ) -> None:
        """
        Scenario 2: Multi-server fleet namespacing and concurrent dispatch
        """
        manager = MCPClientManager()
        manager.safety_gate.auto_approve = True

        # Given two servers: 'memory' (stdio) and 'remote_ops' (sse), both exposing 'search'
        mock_stdio = MagicMock(spec=StdioTransport)
        mock_stdio.server_name = "memory"
        mock_stdio.transport_type = "stdio"
        mock_stdio.status = "connected"
        mock_stdio.latency_ms = 4.2
        mock_stdio.is_connected = True
        mock_stdio.connect = AsyncMock()
        mock_stdio.disconnect = AsyncMock()
        mock_stdio.send_request = AsyncMock(
            return_value={
                "result": {
                    "tools": [
                        {
                            "name": "search",
                            "description": "Search local entities in memory",
                            "inputSchema": {},
                        }
                    ]
                }
            }
        )

        mock_sse = MagicMock(spec=SSETransport)
        mock_sse.server_name = "remote_ops"
        mock_sse.transport_type = "sse"
        mock_sse.status = "connected"
        mock_sse.latency_ms = 48.0
        mock_sse.is_connected = True
        mock_sse.connect = AsyncMock()
        mock_sse.disconnect = AsyncMock()
        mock_sse.send_request = AsyncMock(
            return_value={
                "result": {
                    "tools": [
                        {
                            "name": "search",
                            "description": "Search remote cloud index",
                            "inputSchema": {},
                        }
                    ]
                }
            }
        )

        manager.register_transport(mock_stdio)
        manager.register_transport(mock_sse)

        # When discovering tools across the fleet
        tools = await manager.list_tools()
        assert len(tools) == 2

        # Then collision detection identifies colliding tool name 'search'
        assert "search" in manager.registry.collisions

        # And both qualified names exist in the registry
        assert manager.registry.get_tool("memory.search") is not None
        assert manager.registry.get_tool("remote_ops.search") is not None

        # When concurrently dispatching calls to both servers
        mock_stdio.send_request.return_value = {
            "result": {"content": [{"type": "text", "text": "local_res"}]}
        }
        mock_sse.send_request.return_value = {
            "result": {"content": [{"type": "text", "text": "remote_res"}]}
        }

        call_local = manager.call_tool("memory.search", {"query": "helm"})
        call_remote = manager.call_tool("remote_ops.search", {"query": "helm"})

        res_local, res_remote = await asyncio.gather(call_local, call_remote)

        # Then each call routes to the correct transport without cross-talk
        assert "local_res" in str(res_local)
        assert "remote_res" in str(res_remote)
        mock_stdio.send_request.assert_called()
        mock_sse.send_request.assert_called()

        await manager.disconnect_all()

    @pytest.mark.asyncio
    async def test_bdd_scenario_3_transient_reconnect_and_backoff_exhaustion(self) -> None:
        """
        Scenario 3: Transient network failure, exponential backoff, and RemoteTransportError
        """
        # Given a remote SSE server with flaky connectivity
        transport = SSETransport(
            server_name="flaky_cloud",
            endpoint="https://flaky.helm.run/sse",
            max_retries=3,
            backoff_base_sec=0.01,
        )

        # When connect attempts fail repeatedly
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.stream = MagicMock(side_effect=ConnectionError("Temporary DNS failure"))
            mock_client.aclose = AsyncMock()
            mock_client_cls.return_value = mock_client

            # Then exponential backoff is triggered and RemoteTransportError is raised on exhaustion
            with pytest.raises(RemoteTransportError) as exc_info:
                await transport.connect()

            assert "after 3 attempts" in str(exc_info.value)
            assert transport.status == "error"
            assert not transport.is_connected

        # And CircuitBreaker records step failure cleanly without corrupting state
        from helm.core.state import SessionState

        state = SessionState(user_goal="Test resilience")
        breaker = CircuitBreaker()
        try:
            raise RemoteTransportError("Network down")
        except RemoteTransportError:
            state.record_step()

        assert state.circuit_breaker.total_steps == 1
        res = breaker.check(state.circuit_breaker, tool_name="some_tool", arguments={})
        assert res.allowed
