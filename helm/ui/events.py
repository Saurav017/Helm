"""
helm/ui/events.py — Terminal presentation event bus models.

Defines the core presentation events emitted by the Supervisor and execution
pipeline to decouple presentation from internal orchestration logic.
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field


class UIEvent(BaseModel):
    """Base class for all terminal UI presentation events."""

    timestamp: float = Field(default_factory=time.time)

    model_config = {"frozen": False}


class ThinkingEvent(UIEvent):
    """Emitted when the supervisor or a worker enters a reasoning state."""

    message: str


class ToolCallStartEvent(UIEvent):
    """Emitted when an MCP tool begins execution."""

    tool_name: str
    server_name: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolCallCompleteEvent(UIEvent):
    """Emitted when an MCP tool completes execution."""

    tool_name: str
    summary: str = ""
    is_error: bool = False


class StepProgressEvent(UIEvent):
    """Emitted when a DAG step updates its status or worker assignment."""

    step_id: str
    status: str
    worker: str = ""


class SessionDoneEvent(UIEvent):
    """Emitted when the execution session completes or aborts."""

    summary: str
    cost: float = 0.0
    duration_sec: float = 0.0
