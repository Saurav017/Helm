"""
helm/protocols/transports/base.py — Abstract base class for MCP transports.

Defines the pluggable transport contract (Architecture Invariant 3).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from helm.exceptions import TransportError

logger = logging.getLogger(__name__)


class MCPTransportError(TransportError):
    """Base exception for transport layer failures."""


class RemoteTransportError(MCPTransportError):
    """Raised when a remote network transport fails or exhausts reconnect retries."""


class BaseTransport(ABC):
    """
    Abstract base transport interface for MCP communications.

    Subclasses implement stdio subprocesses, HTTP-SSE streaming, or WebSockets.
    """

    def __init__(
        self,
        server_name: str,
        timeout_sec: float = 30.0,
    ) -> None:
        self.server_name = server_name
        self.timeout_sec = timeout_sec
        self.latency_ms: float = 0.0
        self.status: str = "disconnected"

    @property
    @abstractmethod
    def transport_type(self) -> str:
        """Return the transport identifier (e.g. 'stdio', 'sse')."""
        raise NotImplementedError

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Return True if the transport connection is currently active."""
        raise NotImplementedError

    @abstractmethod
    async def connect(self) -> None:
        """Establish transport connection and begin message ingestion."""
        raise NotImplementedError

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection and release underlying resources."""
        raise NotImplementedError

    @abstractmethod
    async def send_message(self, message: dict[str, Any]) -> None:
        """Transmit raw JSON-RPC message dictionary over transport."""
        raise NotImplementedError

    @abstractmethod
    def receive_messages(self) -> AsyncIterator[dict[str, Any]]:
        """Yield incoming JSON-RPC message dictionaries received from the server."""
        raise NotImplementedError

    @abstractmethod
    async def send_request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send JSON-RPC request and wait for the corresponding response."""
        raise NotImplementedError

    @abstractmethod
    async def send_notification(
        self,
        method: str,
        params: dict[str, Any] | None = None,
    ) -> None:
        """Send JSON-RPC notification (no response expected)."""
        raise NotImplementedError
