"""
helm/protocols — MCP protocol client, tool registry, and safety gate.
"""

from __future__ import annotations

from helm.protocols.filesystem_adapter import (
    format_directory_entries,
    list_directory_sync,
    read_file_sync,
    resolve_file_path,
)
from helm.protocols.filesystem_adapter import (
    list_directory as fs_list_directory,
)
from helm.protocols.filesystem_adapter import (
    read_file as fs_read_file,
)
from helm.protocols.mcp_client import (
    ActionDeniedByHuman,
    MCPClientManager,
    MCPConnectionError,
    MCPError,
    MCPTimeoutError,
    MCPToolDefinition,
    ToolNotFoundError,
)
from helm.protocols.safety_gate import (
    ActionSafety,
    SafetyDecision,
    SafetyGate,
    SafetyGateResult,
    classify_tool,
    is_mutating,
)
from helm.protocols.tool_registry import ToolRegistry
from helm.protocols.transports.base import RemoteTransportError

__all__ = [
    "ActionDeniedByHuman",
    "ActionSafety",
    "MCPClientManager",
    "MCPConnectionError",
    "MCPError",
    "MCPTimeoutError",
    "MCPToolDefinition",
    "RemoteTransportError",
    "SafetyDecision",
    "SafetyGate",
    "SafetyGateResult",
    "ToolNotFoundError",
    "ToolRegistry",
    "classify_tool",
    "format_directory_entries",
    "fs_list_directory",
    "fs_read_file",
    "is_mutating",
    "list_directory_sync",
    "read_file_sync",
    "resolve_file_path",
]
