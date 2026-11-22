"""
helm/protocols/tool_registry.py — Multi-server MCP tool registry and collision mitigation.

Indexes discovered MCP tool definitions by server origin and namespaces colliding
tool names (server_name.tool_name) across multi-server fleets.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from helm.protocols.mcp_client import MCPToolDefinition

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Fleet-wide tool registry mapping tool names to definitions and originating servers.

    Features:
    - Collision mitigation: if two servers expose the same tool name, both namespaced
      aliases ('<server>.<tool>') and primary names are indexed.
    - Server lookup: provides O(1) resolution from tool name to target server.
    """

    def __init__(self) -> None:
        self._tools_by_name: dict[str, MCPToolDefinition] = {}
        self._tool_to_server: dict[str, str] = {}
        self._server_tools: dict[str, list[MCPToolDefinition]] = {}
        self._collisions: set[str] = set()

    @property
    def collisions(self) -> set[str]:
        """Set of tool names that collide across multiple servers."""
        return set(self._collisions)

    @property
    def tool_to_server(self) -> dict[str, str]:
        """Mapping from tool name to originating server name."""
        return dict(self._tool_to_server)

    def register_tool(self, tool_def: MCPToolDefinition) -> None:
        """Register a single tool and detect potential fleet name collisions."""
        name = tool_def.name
        server = tool_def.server_name
        qualified_name = f"{server}.{name}"

        # Always register the qualified name
        self._tools_by_name[qualified_name] = tool_def
        self._tool_to_server[qualified_name] = server

        if name in self._tools_by_name and self._tool_to_server.get(name) != server:
            # Collision detected
            existing = self._tools_by_name[name]
            existing_qualified = f"{existing.server_name}.{name}"
            self._collisions.add(name)
            logger.warning(
                "Tool collision detected for '%s' between server '%s' and '%s'. "
                "Namespaced access via '%s' and '%s' is enabled.",
                name,
                existing.server_name,
                server,
                existing_qualified,
                qualified_name,
            )
            # Default lookup retains first registered or qualified
        else:
            self._tools_by_name[name] = tool_def
            self._tool_to_server[name] = server

        if server not in self._server_tools:
            self._server_tools[server] = []
        if tool_def not in self._server_tools[server]:
            self._server_tools[server].append(tool_def)

    def register_tools(self, tools: list[MCPToolDefinition]) -> None:
        """Register multiple tools across servers."""
        for t in tools:
            self.register_tool(t)

    def get_tool(self, name: str) -> MCPToolDefinition | None:
        """Retrieve tool definition by simple or namespaced name."""
        return self._tools_by_name.get(name)

    def get_server_for_tool(self, name: str) -> str | None:
        """Return the originating server name for a given tool."""
        return self._tool_to_server.get(name)

    def list_tools(self, server_name: str | None = None) -> list[MCPToolDefinition]:
        """Return all registered tools, optionally filtered by server."""
        if server_name is not None:
            return list(self._server_tools.get(server_name, []))

        # Return unique tool definitions (ignoring qualified duplicates)
        seen: set[tuple[str, str]] = set()
        unique: list[MCPToolDefinition] = []
        for t in self._tools_by_name.values():
            key = (t.server_name, t.name)
            if key not in seen:
                seen.add(key)
                unique.append(t)
        return unique

    def unregister_server(self, server_name: str) -> None:
        """Remove all tools belonging to a disconnected server."""
        self._server_tools.pop(server_name, None)
        self._tools_by_name = {
            k: v for k, v in self._tools_by_name.items() if v.server_name != server_name
        }
        self._tool_to_server = {k: v for k, v in self._tool_to_server.items() if v != server_name}

    def clear(self) -> None:
        """Clear all registered tools."""
        self._tools_by_name.clear()
        self._tool_to_server.clear()
        self._server_tools.clear()
        self._collisions.clear()
