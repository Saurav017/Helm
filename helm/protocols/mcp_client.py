"""
helm/protocols/mcp_client.py — Native Model Context Protocol (MCP) multi-transport client.

Manages the lifecycle of external MCP servers over stdio and remote SSE/HTTP,
normalizes tool schemas, mitigates name collisions across multi-server fleets,
and routes tool calls through the SafetyGate before dispatching (INSTRUCTIONS.md §3.1).
"""

from __future__ import annotations

import asyncio
import logging
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, cast

from pydantic import BaseModel, Field

from helm.exceptions import HelmError
from helm.protocols.filesystem_adapter import (
    list_directory as fs_list_directory,
)
from helm.protocols.filesystem_adapter import (
    read_file as fs_read_file,
)
from helm.protocols.filesystem_adapter import (
    resolve_file_path,
)
from helm.protocols.safety_gate import ActionSafety, SafetyGate, classify_tool
from helm.protocols.tool_registry import ToolRegistry
from helm.protocols.transports.base import BaseTransport, MCPTransportError, RemoteTransportError
from helm.protocols.transports.sse import SSETransport
from helm.protocols.transports.stdio import StdioTransport

if TYPE_CHECKING:
    from helm.config.settings import MCPServerSettings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MCPError(HelmError):
    """Base exception for Model Context Protocol failures."""


class MCPConnectionError(MCPError):
    """Raised when failing to spawn or communicate with an MCP server."""


class MCPTimeoutError(MCPError):
    """Raised when an MCP server response exceeds the timeout."""


class ToolNotFoundError(MCPError):
    """Raised when requesting a tool not registered on any connected server."""


class ActionDeniedByHuman(MCPError):
    """Raised or returned when a human operator rejects a tool action."""


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class MCPToolDefinition(BaseModel):
    """Normalized MCP tool definition across all connected servers."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    server_name: str
    transport: str = "stdio"
    safety: ActionSafety = ActionSafety.MUTATING

    model_config = {"frozen": True}


# Compatibility alias for local stdio connection
_MCPServerConnection = StdioTransport


# ---------------------------------------------------------------------------
# MCP Client Manager
# ---------------------------------------------------------------------------


class MCPClientManager:
    """
    Manages connections to multiple MCP servers across stdio and SSE transports.

    Features:
    - Multi-server fleet orchestration: connects local processes and remote endpoints.
    - Asynchronous concurrent initialization across configured servers.
    - Tool collision mitigation via ToolRegistry.
    - SafetyGate interception on mutating and configured tool dispatches.
    """

    def __init__(
        self,
        safety_gate: SafetyGate | None = None,
        *,
        timeout_sec: float = 30.0,
    ) -> None:
        self.safety_gate = safety_gate or SafetyGate()
        self.timeout_sec = timeout_sec
        self._transports: dict[str, BaseTransport] = {}
        self.registry = ToolRegistry()
        self._cached_tools: list[MCPToolDefinition] = []

    @property
    def _connections(self) -> dict[str, Any]:
        """Backwards compatibility alias for _transports dictionary."""
        return self._transports

    @property
    def _tool_to_server(self) -> dict[str, str]:
        """Backwards compatibility alias for tool-to-server mapping."""
        return self.registry.tool_to_server

    def register_transport(self, transport: BaseTransport) -> None:
        """Register a pre-configured transport instance directly."""
        self._transports[transport.server_name] = transport

    async def connect_stdio(
        self,
        server_name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        """Connect to a local MCP server over stdio."""
        if server_name in self._transports:
            logger.info("MCP server %r already connected", server_name)
            return

        transport = StdioTransport(
            server_name=server_name,
            command=command,
            args=args,
            env=env,
            timeout_sec=self.timeout_sec,
        )
        try:
            await transport.connect()
            await self._perform_handshake(transport)
            self._transports[server_name] = transport
            logger.info("Connected to stdio MCP server %r", server_name)
        except Exception as exc:
            await transport.disconnect()
            if isinstance(exc, (MCPConnectionError, RemoteTransportError)):
                raise
            msg = f"Failed to connect to stdio MCP server {server_name!r}: {exc}"
            raise MCPConnectionError(msg) from exc

    async def connect_sse(
        self,
        server_name: str,
        endpoint: str,
        headers: dict[str, str] | None = None,
        auth_token: str = "",
    ) -> None:
        """Connect to a remote MCP server over SSE / HTTP."""
        if server_name in self._transports:
            logger.info("MCP server %r already connected", server_name)
            return

        transport = SSETransport(
            server_name=server_name,
            endpoint=endpoint,
            headers=headers,
            auth_token=auth_token,
            timeout_sec=self.timeout_sec,
        )
        try:
            await transport.connect()
            await self._perform_handshake(transport)
            self._transports[server_name] = transport
            logger.info("Connected to SSE MCP server %r at %s", server_name, endpoint)
        except Exception as exc:
            await transport.disconnect()
            if isinstance(exc, (MCPConnectionError, RemoteTransportError)):
                raise
            msg = f"Failed to connect to remote SSE MCP server {server_name!r}: {exc}"
            raise RemoteTransportError(msg) from exc

    async def connect(
        self,
        server_name: str,
        command: str = "",
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        *,
        transport: str = "stdio",
        endpoint: str = "",
        headers: dict[str, str] | None = None,
        auth_token: str = "",
    ) -> None:
        """
        Connect to an MCP server across either stdio or sse transports.
        """
        if transport == "sse" or (endpoint and not command):
            await self.connect_sse(
                server_name=server_name,
                endpoint=endpoint,
                headers=headers,
                auth_token=auth_token,
            )
        else:
            await self.connect_stdio(
                server_name=server_name,
                command=command,
                args=args,
                env=env,
            )

    # Alias for method naming convenience
    connect_server = connect

    async def connect_all(self, servers: list[MCPServerSettings]) -> None:
        """
        Connect to all configured servers concurrently.
        """
        tasks = []
        for s in servers:
            if s.transport == "sse" or s.effective_endpoint:
                tasks.append(
                    self.connect_sse(
                        server_name=s.name,
                        endpoint=s.effective_endpoint,
                        headers=s.effective_headers,
                        auth_token=s.auth_token,
                    )
                )
            elif s.command:
                tasks.append(
                    self.connect_stdio(
                        server_name=s.name,
                        command=s.command,
                        args=s.args,
                        env=s.env,
                    )
                )

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for s, res in zip(servers, results, strict=False):
                if isinstance(res, Exception):
                    logger.warning("Could not connect to MCP server %r: %s", s.name, res)

    async def _perform_handshake(self, transport: BaseTransport) -> None:
        """Execute MCP initialization handshake (initialize & notifications/initialized)."""
        init_params = {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "helmx",
                "version": "0.1.0",
            },
        }
        resp = await transport.send_request("initialize", init_params)
        if "error" in resp:
            err_msg = resp["error"].get("message", "Unknown error")
            raise MCPConnectionError(
                f"Initialization rejected by {transport.server_name!r}: {err_msg}"
            )
        await transport.send_notification("notifications/initialized")

    async def list_tools(self, *, force_refresh: bool = False) -> list[MCPToolDefinition]:
        """
        Enumerate tools across all connected transports, populate ToolRegistry,
        and return normalized definitions.
        """
        if self._cached_tools and not force_refresh:
            return list(self._cached_tools)

        self.registry.clear()
        discovered: list[MCPToolDefinition] = []

        for server_name, transport in self._transports.items():
            try:
                resp = await transport.send_request("tools/list")
                result = resp.get("result", {})
                raw_tools = result.get("tools", [])

                for item in raw_tools:
                    name = item.get("name", "")
                    if not name:
                        continue

                    desc = item.get("description", "")
                    schema = item.get("inputSchema", {})
                    safety = classify_tool(name)

                    tool_def = MCPToolDefinition(
                        name=name,
                        description=desc,
                        input_schema=schema,
                        server_name=server_name,
                        transport=transport.transport_type,
                        safety=safety,
                    )
                    discovered.append(tool_def)
                    self.registry.register_tool(tool_def)
            except Exception as exc:
                logger.warning("Failed to list tools from MCP server %r: %s", server_name, exc)

        self._cached_tools = list(discovered)
        return list(discovered)

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        auto_approve: bool | None = None,
        summary: str = "",
    ) -> dict[str, Any]:
        """
        Invoke an MCP tool with SafetyGate interception and fleet routing.
        """
        # Ensure tools are discovered
        target_server = self.registry.get_server_for_tool(tool_name)
        if not target_server:
            await self.list_tools(force_refresh=True)
            target_server = self.registry.get_server_for_tool(tool_name)

        if not target_server or target_server not in self._transports:
            base_tool = tool_name.rsplit(".", maxsplit=1)[-1]
            if base_tool in ("read_file", "list_directory"):
                if base_tool == "read_file":
                    return await fs_read_file(arguments.get("path", ""))
                return await fs_list_directory(arguments.get("path", "."))
            msg = f"Tool {tool_name!r} not found on any connected MCP server"
            raise ToolNotFoundError(msg)

        # 2. Intercept with SafetyGate
        gate_result = self.safety_gate.evaluate(
            tool_name,
            arguments,
            auto_approve=auto_approve,
            summary=summary,
        )

        if not gate_result.is_approved:
            logger.warning(
                "Tool call denied by safety gate",
                extra={
                    "event": "tool_call_denied_by_human",
                    "component": "MCPClientManager",
                    "tool_name": tool_name,
                    "reason": gate_result.reason,
                },
            )
            return {
                "is_error": True,
                "error": "ActionDeniedByHuman: Execution denied by user in safety gate",
                "action_denied": True,
                "tool_name": tool_name,
            }

        # 3. Resolve base tool name if namespaced (server_name.tool_name)
        actual_name = tool_name
        if "." in tool_name and tool_name.startswith(f"{target_server}."):
            actual_name = tool_name[len(f"{target_server}.") :]
        elif "." in tool_name:
            actual_name = tool_name.rsplit(".", maxsplit=1)[-1]

        raw_args = gate_result.modified_arguments or arguments
        effective_args, warning_msg = self._preprocess_filesystem_arguments(actual_name, raw_args)

        transport = self._transports[target_server]
        params = {
            "name": actual_name,
            "arguments": effective_args,
        }

        try:
            resp = await transport.send_request("tools/call", params)
        except RemoteTransportError:
            raise
        except MCPTransportError as exc:
            msg = f"Transport error invoking {tool_name!r} on {target_server!r}: {exc}"
            raise MCPConnectionError(msg) from exc

        return await self._handle_tool_call_response(resp, actual_name, effective_args, warning_msg)

    async def _handle_tool_call_response(
        self,
        resp: dict[str, Any],
        actual_name: str,
        effective_args: dict[str, Any],
        warning_msg: str | None,
    ) -> dict[str, Any]:
        """Process MCP server response with filesystem fallback and warnings."""
        if "error" in resp:
            # Fallback to local filesystem adapter for read_file and list_directory
            if actual_name == "read_file":
                local_res = await fs_read_file(effective_args.get("path", ""))
                if not local_res.get("is_error"):
                    return local_res
            elif actual_name == "list_directory":
                local_res = await fs_list_directory(effective_args.get("path", "."))
                if not local_res.get("is_error"):
                    if warning_msg and "warning" not in local_res:
                        local_res["warning"] = warning_msg
                    return local_res

            err = resp["error"]
            return {
                "is_error": True,
                "error": err.get("message", "MCP server execution error"),
                "code": err.get("code"),
            }

        result = resp.get("result", {})
        res_dict = cast(
            "dict[str, Any]", result if isinstance(result, dict) else {"result": result}
        )

        # Fall back if the MCP server returned isError inside result
        if res_dict.get("isError") and actual_name == "read_file":
            local_res = await fs_read_file(effective_args.get("path", ""))
            if not local_res.get("is_error"):
                return local_res

        if warning_msg:
            res_dict["warning"] = warning_msg
            if "content" in res_dict and isinstance(res_dict["content"], list):
                for item in res_dict["content"]:
                    if isinstance(item, dict) and item.get("type") == "text":
                        item["text"] = f"{warning_msg}\n\n{item.get('text', '')}"
                        break
        return res_dict

    @staticmethod
    def _preprocess_filesystem_arguments(
        actual_name: str,
        arguments: dict[str, Any],
    ) -> tuple[dict[str, Any], str | None]:
        """Apply robust file resolving to read_file and graceful fallback to list_directory."""
        effective = dict(arguments)
        warning_msg: str | None = None

        if actual_name == "read_file":
            path_arg = effective.get("path")
            if isinstance(path_arg, str) and path_arg:
                resolved = resolve_file_path(path_arg)
                if resolved:
                    effective["path"] = str(resolved)
        elif actual_name == "list_directory":
            path_arg = effective.get("path", ".")
            is_valid_dir = False
            try:
                if isinstance(path_arg, str) and path_arg and Path(path_arg).is_dir():
                    is_valid_dir = True
            except OSError:
                is_valid_dir = False
            if not is_valid_dir:
                warning_msg = f"Path '{path_arg}' not found, listed root directory instead"
                effective["path"] = "."

        return effective, warning_msg

    def get_transport(self, server_name: str) -> BaseTransport | None:
        """Return the transport instance for a given server name."""
        return self._transports.get(server_name)

    def get_server_status(self, server_name: str) -> dict[str, Any]:
        """Return live health and latency metrics for a connected server."""
        transport = self._transports.get(server_name)
        if not transport:
            return {
                "server_name": server_name,
                "status": "offline",
                "transport": "unknown",
                "latency_ms": 0.0,
            }
        return {
            "server_name": server_name,
            "status": transport.status,
            "transport": transport.transport_type,
            "latency_ms": round(transport.latency_ms, 1),
        }

    async def disconnect(self, server_name: str) -> None:
        """Disconnect a specific server transport."""
        transport = self._transports.pop(server_name, None)
        if transport:
            await transport.disconnect()
            self.registry.unregister_server(server_name)
            self._cached_tools = [t for t in self._cached_tools if t.server_name != server_name]

    async def disconnect_all(self) -> None:
        """Gracefully disconnect all server transports."""
        for transport in list(self._transports.values()):
            await transport.disconnect()
        self._transports.clear()
        self.registry.clear()
        self._cached_tools = []

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        await self.disconnect_all()
