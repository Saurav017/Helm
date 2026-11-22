"""
helm/protocols/transports/stdio.py — Subprocess Stdio MCP Transport.

Spawns a local process and streams JSON-RPC 2.0 lines across stdin/stdout.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from typing import Any

from helm.protocols.transports.base import BaseTransport, MCPTransportError

logger = logging.getLogger(__name__)


class StdioTransport(BaseTransport):
    """MCP transport executing a local child process over stdio."""

    def __init__(
        self,
        server_name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: float = 30.0,
    ) -> None:
        super().__init__(server_name=server_name, timeout_sec=timeout_sec)
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.process: asyncio.subprocess.Process | None = None
        self._request_id = 0
        self._pending_requests: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._incoming_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._reader_task: asyncio.Task[None] | None = None
        self._closed = False

    @property
    def transport_type(self) -> str:
        return "stdio"

    @property
    def is_connected(self) -> bool:
        return not self._closed and self.process is not None and self.process.returncode is None

    async def connect(self) -> None:
        """Spawn the child process and start reading stdout."""
        if self.is_connected:
            return

        merged_env = os.environ.copy()
        if self.env:
            merged_env.update(self.env)

        t0 = time.monotonic()
        try:
            self.process = await asyncio.create_subprocess_exec(
                self.command,
                *self.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=merged_env,
            )
            self._closed = False
            self.status = "connected"
            self.latency_ms = (time.monotonic() - t0) * 1000.0
            self._reader_task = asyncio.create_task(
                self._read_loop(),
                name=f"stdio-reader-{self.server_name}",
            )
        except (FileNotFoundError, PermissionError) as exc:
            self.status = "error"
            msg = f"Failed to spawn stdio server {self.server_name!r}: {exc}"
            raise MCPTransportError(msg) from exc

    async def disconnect(self) -> None:
        """Terminate child process and cancel reader loop."""
        self._closed = True
        self.status = "disconnected"

        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()

        for fut in self._pending_requests.values():
            if not fut.done():
                fut.cancel()
        self._pending_requests.clear()

        if self.process and self.process.returncode is None:
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=2.0)
            except (TimeoutError, ProcessLookupError):
                with contextlib.suppress(ProcessLookupError):
                    self.process.kill()
        self.process = None

    async def send_message(self, message: dict[str, Any]) -> None:
        """Serialize and write JSON line to process stdin."""
        if not self.is_connected or not self.process or not self.process.stdin:
            msg = f"Cannot send to disconnected stdio server {self.server_name!r}"
            raise MCPTransportError(msg)

        line = json.dumps(message) + "\n"
        self.process.stdin.write(line.encode("utf-8"))
        await self.process.stdin.drain()

    async def receive_messages(self) -> AsyncIterator[dict[str, Any]]:
        """Async iterator yielding messages received from stdout."""
        while self.is_connected:
            try:
                msg = await self._incoming_queue.get()
                yield msg
            except asyncio.CancelledError:
                break

    async def send_notification(self, method: str, params: dict[str, Any] | None = None) -> None:
        """Send a JSON-RPC notification."""
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        await self.send_message(payload)

    async def send_request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send a JSON-RPC request and wait for the response."""
        self._request_id += 1
        req_id = self._request_id
        fut: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending_requests[req_id] = fut

        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
        }
        if params is not None:
            payload["params"] = params

        t0 = time.monotonic()
        try:
            await self.send_message(payload)
            async with asyncio.timeout(self.timeout_sec):
                res = await fut
                self.latency_ms = (time.monotonic() - t0) * 1000.0
                result = res.get("result")
                if isinstance(result, dict):
                    merged = dict(result)
                    merged.update(res)
                    return merged
                return res
        except TimeoutError as err:
            self._pending_requests.pop(req_id, None)
            raise MCPTransportError(
                f"Stdio server {self.server_name!r} timed out on method {method!r}"
            ) from err
        except Exception:
            self._pending_requests.pop(req_id, None)
            raise

    async def _read_loop(self) -> None:
        """Read stdout lines and dispatch to queue and pending futures."""
        if not self.process or not self.process.stdout:
            return

        while not self._closed:
            try:
                line_bytes = await self.process.stdout.readline()
                if not line_bytes:
                    break  # EOF

                line = line_bytes.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug("Non-JSON line from stdio %r: %s", self.server_name, line)
                    continue

                req_id = msg.get("id")
                if req_id is not None and req_id in self._pending_requests:
                    fut = self._pending_requests.pop(req_id)
                    if not fut.done():
                        fut.set_result(msg)

                await self._incoming_queue.put(msg)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.debug("Exception in stdio read loop for %r: %s", self.server_name, exc)
                break
