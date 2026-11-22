"""
helm/protocols/transports/sse.py — Remote Server-Sent Events (SSE) MCP Transport.

Consumes text/event-stream SSE endpoints for server messages and issues HTTP POST
requests for client JSON-RPC dispatches with exponential backoff reconnection.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urljoin

import httpx

from helm.protocols.transports.base import BaseTransport, RemoteTransportError

logger = logging.getLogger(__name__)


class SSETransport(BaseTransport):
    """
    Remote MCP transport over Server-Sent Events (SSE) and HTTP POST.

    Specification:
    1. Connects to GET /sse endpoint and establishes a persistent stream.
    2. Server may emit an 'endpoint' event specifying the URI for client POSTs.
    3. Client sends JSON-RPC requests via HTTP POST to the resolved endpoint.
    4. Incoming JSON-RPC responses arrive either via SSE stream or immediate POST response.
    5. Implements exponential backoff on stream disconnection.
    """

    def __init__(
        self,
        server_name: str,
        endpoint: str,
        *,
        headers: dict[str, str] | None = None,
        auth_token: str = "",
        timeout_sec: float = 30.0,
        max_retries: int = 3,
        backoff_base_sec: float = 0.1,
    ) -> None:
        super().__init__(server_name=server_name, timeout_sec=timeout_sec)
        self.endpoint = endpoint
        self.headers = dict(headers or {})
        if auth_token and "Authorization" not in self.headers:
            self.headers["Authorization"] = f"Bearer {auth_token}"

        self.max_retries = max_retries
        self.backoff_base_sec = backoff_base_sec
        self.post_endpoint: str | None = None
        self._endpoint_event: asyncio.Event = asyncio.Event()
        self._client: httpx.AsyncClient | None = None
        self._request_id = 0
        self._pending_requests: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._incoming_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._stream_task: asyncio.Task[None] | None = None
        self._closed = False
        self._connected = False

    @property
    def transport_type(self) -> str:
        return "sse"

    @property
    def is_connected(self) -> bool:
        return self._connected and not self._closed

    async def connect(self) -> None:
        """Connect to SSE endpoint with exponential backoff on transient failures."""
        if self.is_connected:
            return

        self._closed = False
        self._client = httpx.AsyncClient(
            headers=self.headers,
            timeout=httpx.Timeout(self.timeout_sec, read=None),
        )

        connected = await self._connect_with_retry()
        if not connected:
            self.status = "error"
            raise RemoteTransportError(
                f"Failed to connect to SSE server '{self.server_name}' "
                f"at {self.endpoint} after {self.max_retries} attempts"
            )

    async def _connect_with_retry(self) -> bool:
        """Attempt SSE connection with exponential backoff."""
        for attempt in range(self.max_retries):
            t0 = time.monotonic()
            try:
                # Test initial reachability or start stream
                self.status = "connecting"
                stream_ready: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
                self._stream_task = asyncio.create_task(
                    self._sse_stream_loop(stream_ready),
                    name=f"sse-stream-{self.server_name}",
                )

                async with asyncio.timeout(min(self.timeout_sec, 10.0)):
                    is_ready = await stream_ready

                if is_ready:
                    # Give server brief window to announce POST endpoint via 'endpoint' event
                    with contextlib.suppress(TimeoutError):
                        async with asyncio.timeout(min(self.timeout_sec, 2.0)):
                            await self._endpoint_event.wait()

                    self._connected = True
                    self.status = "connected"
                    self.latency_ms = (time.monotonic() - t0) * 1000.0
                    logger.info("Connected to remote SSE MCP server '%s'", self.server_name)
                    return True
            except Exception as exc:
                backoff = self.backoff_base_sec * (2**attempt)
                logger.warning(
                    "SSE connection attempt %d/%d to '%s' failed (%s). Retrying in %.2fs",
                    attempt + 1,
                    self.max_retries,
                    self.server_name,
                    exc,
                    backoff,
                )
                if self._stream_task and not self._stream_task.done():
                    self._stream_task.cancel()
                await asyncio.sleep(backoff)

        return False

    async def disconnect(self) -> None:
        """Close SSE stream and HTTP client."""
        self._closed = True
        self._connected = False
        self.status = "disconnected"

        if self._stream_task and not self._stream_task.done():
            self._stream_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._stream_task
            self._stream_task = None

        for fut in self._pending_requests.values():
            if not fut.done():
                fut.cancel()
        self._pending_requests.clear()

        if self._client:
            await self._client.aclose()
            self._client = None

    async def send_message(self, message: dict[str, Any]) -> None:
        """Transmit JSON-RPC request/notification via HTTP POST."""
        if not self._client:
            raise RemoteTransportError(f"Cannot send to closed SSE server {self.server_name!r}")

        if not self._endpoint_event.is_set():
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(min(self.timeout_sec, 2.0)):
                    await self._endpoint_event.wait()

        target_url = self.post_endpoint or self.endpoint
        try:
            resp = await self._client.post(target_url, json=message)
            resp.raise_for_status()

            # Handle direct JSON response if returned on POST
            if resp.content:
                with contextlib.suppress(Exception):
                    body = resp.json()
                    req_id = body.get("id")
                    if req_id is not None and req_id in self._pending_requests:
                        fut = self._pending_requests.pop(req_id)
                        if not fut.done():
                            fut.set_result(body)
                    await self._incoming_queue.put(body)
        except Exception as exc:
            raise RemoteTransportError(
                f"HTTP POST failed for SSE server {self.server_name!r}: {exc}"
            ) from exc

    async def receive_messages(self) -> AsyncIterator[dict[str, Any]]:
        """Yield parsed incoming messages received from SSE stream."""
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
        """Send a JSON-RPC request and await response."""
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
            raise RemoteTransportError(
                f"SSE server {self.server_name!r} timed out on method {method!r}"
            ) from err
        except Exception:
            self._pending_requests.pop(req_id, None)
            raise

    async def _sse_stream_loop(self, ready_fut: asyncio.Future[bool]) -> None:
        """Consume SSE stream and parse incoming events."""
        if not self._client:
            return

        reconnect_count = 0
        while not self._closed:
            try:
                headers = {"Accept": "text/event-stream"}
                async with self._client.stream("GET", self.endpoint, headers=headers) as response:
                    response.raise_for_status()
                    if not ready_fut.done():
                        ready_fut.set_result(True)

                    current_event = "message"
                    data_lines: list[str] = []

                    async for raw_line in response.aiter_lines():
                        if self._closed:
                            break

                        line = raw_line.strip()
                        if not line:
                            # Event boundary dispatch
                            if data_lines:
                                full_data = "\n".join(data_lines)
                                await self._handle_sse_event(current_event, full_data)
                                data_lines = []
                                current_event = "message"
                            continue

                        if line.startswith("event:"):
                            current_event = line[len("event:") :].strip()
                        elif line.startswith("data:"):
                            data_lines.append(line[len("data:") :].strip())

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning(
                    "SSE stream connection dropped for '%s': %s",
                    self.server_name,
                    exc,
                )
                self._connected = False
                if not ready_fut.done():
                    ready_fut.set_result(False)

                if self._closed:
                    break

                # Transient reconnect attempt
                reconnect_count += 1
                if reconnect_count > self.max_retries:
                    self.status = "error"
                    logger.exception("SSE stream reconnect exhausted for '%s'", self.server_name)
                    break

                backoff = self.backoff_base_sec * (2**reconnect_count)
                await asyncio.sleep(backoff)

    async def _handle_sse_event(self, event_type: str, data: str) -> None:
        """Process dispatched SSE events."""
        if event_type == "endpoint":
            # Target endpoint for client POST requests
            resolved = urljoin(self.endpoint, data.strip())
            self.post_endpoint = resolved
            self._endpoint_event.set()
            logger.debug("SSE server '%s' announced POST endpoint: %s", self.server_name, resolved)
            return

        try:
            msg = json.loads(data)
            req_id = msg.get("id")
            if req_id is not None and req_id in self._pending_requests:
                fut = self._pending_requests.pop(req_id)
                if not fut.done():
                    fut.set_result(msg)

            await self._incoming_queue.put(msg)
        except json.JSONDecodeError:
            logger.debug("Non-JSON SSE payload from '%s': %s", self.server_name, data)
