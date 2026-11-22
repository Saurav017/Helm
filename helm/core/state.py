"""
helm/core/state.py — Strongly-typed session state schema.

All runtime state lives here. No component holds shadow state; every
read/write passes through these Pydantic models (Architecture Invariant 4).

Design notes
------------
- Pure Pydantic v2.  No langgraph / langchain dependency — Helm IS the
  framework (INSTRUCTIONS.md §1.2).
- All fields are immutable-by-default; mutations produce new model instances
  via ``model.model_copy(update={...})``.
- ``SessionState`` is the top-level object handed to and returned from every
  core component (Supervisor, Circuit Breaker, Scheduler, …).
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class TaskStatus(StrEnum):
    """Lifecycle status of a single decomposed sub-task."""

    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"  # circuit-breaker triggered


class SessionStatus(StrEnum):
    """Top-level status of the entire Helm session."""

    IDLE = "idle"
    RUNNING = "running"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


class ModelTier(StrEnum):
    """Tiered model routing classification (INSTRUCTIONS.md §3.6)."""

    FAST = "fast"
    STANDARD = "standard"
    DEEP = "deep"


class CircuitBlockReason(StrEnum):
    """Reason a circuit-breaker blocked a tool call."""

    DUPLICATE_CALL = "duplicate_call"
    STEP_CEILING = "step_ceiling"
    BUDGET_CAP = "budget_cap"


# ---------------------------------------------------------------------------
# Leaf models
# ---------------------------------------------------------------------------


class ToolCallRecord(BaseModel):
    """Immutable record of a single tool invocation attempt."""

    call_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str
    arguments: dict[str, Any]
    call_hash: str  # SHA-256 of (tool_name, canonical_arguments)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    result: str | None = None
    error: str | None = None
    was_blocked: bool = False
    block_reason: CircuitBlockReason | None = None

    model_config = {"frozen": True}

    @field_validator("call_hash")
    @classmethod
    def validate_hash_format(cls, v: str) -> str:
        """Ensure hash looks like a 64-char hex SHA-256 digest."""
        _SHA256_HEX_LEN = 64  # noqa: N806 — constant inside method for locality
        if len(v) != _SHA256_HEX_LEN or not all(c in "0123456789abcdef" for c in v):
            msg = "call_hash must be a 64-character lowercase hex SHA-256 digest"
            raise ValueError(msg)
        return v


class TaskBrief(BaseModel):
    """
    Isolated context package delivered to a worker sub-agent.

    RULE: Workers MUST NEVER receive raw full-chat context (INSTRUCTIONS.md §3.3).
    This model enforces that only the minimum required information is forwarded.
    """

    task_id: str
    description: str
    relevant_context: str = ""  # Distilled excerpt — NOT the full conversation
    available_tool_names: list[str] = Field(default_factory=list)
    max_steps: int = 25
    budget_usd: float = 0.50

    model_config = {"frozen": True}


class TaskResult(BaseModel):
    """Typed output returned by a worker upon task completion or failure."""

    task_id: str
    status: TaskStatus
    output: str = ""
    error: str | None = None
    steps_used: int = 0
    tokens_used: int = 0
    cost_usd: float = 0.0
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    model_config = {"frozen": True}


class Task(BaseModel):
    """A single decomposed sub-task in the execution DAG."""

    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str
    status: TaskStatus = TaskStatus.PENDING
    assigned_worker: str | None = None
    tier: ModelTier = ModelTier.STANDARD
    depends_on: list[str] = Field(default_factory=list)  # task_ids
    brief: TaskBrief | None = None
    result: TaskResult | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    step_count: int = 0
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)

    model_config = {"frozen": False}  # mutable — updated in-place by Supervisor

    @property
    def worker(self) -> str | None:
        return self.assigned_worker

    @worker.setter
    def worker(self, value: str | None) -> None:
        self.assigned_worker = value

    @model_validator(mode="after")
    def validate_dependency_self_reference(self) -> Task:
        """A task must not depend on itself."""
        if self.task_id in self.depends_on:
            msg = f"Task {self.task_id!r} cannot depend on itself"
            raise ValueError(msg)
        return self


class CircuitBreakerState(BaseModel):
    """
    Live state consumed by the Circuit Breaker on every tool invocation.

    Invariant: this object is embedded in ``SessionState`` and is the ONLY
    place circuit-breaker counters live.
    """

    seen_call_hashes: set[str] = Field(default_factory=set)
    total_steps: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0

    # Limits (populated from Settings at session start)
    max_steps: int = 25
    budget_usd: float = 2.00
    duplicate_detection_enabled: bool = True

    model_config = {"frozen": False}


class Message(BaseModel):
    """A single entry in the conversation history."""

    role: str  # "user" | "assistant" | "system" | "tool"
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        allowed = {"user", "assistant", "system", "tool"}
        if v not in allowed:
            msg = f"Message role must be one of {allowed!r}, got {v!r}"
            raise ValueError(msg)
        return v


# ---------------------------------------------------------------------------
# Top-level session state — single source of truth
# ---------------------------------------------------------------------------


class SessionState(BaseModel):
    """
    Top-level session state.  The single source of truth for all runtime data
    (Architecture Invariant 4).

    Every core component receives a reference to this object and returns it
    (possibly mutated) after completing its operation.  No shadow copies.
    """

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: SessionStatus = SessionStatus.IDLE
    user_goal: str
    messages: list[Message] = Field(default_factory=list)
    tasks: list[Task] = Field(default_factory=list)
    circuit_breaker: CircuitBreakerState = Field(default_factory=CircuitBreakerState)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    final_answer: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": False}

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_task(self, task_id: str) -> Task | None:
        """Return the Task with the given ID, or None."""
        return next((t for t in self.tasks if t.task_id == task_id), None)

    def pending_tasks(self) -> list[Task]:
        """Return all tasks in PENDING status."""
        return [t for t in self.tasks if t.status == TaskStatus.PENDING]

    def runnable_tasks(self) -> list[Task]:
        """
        Return tasks whose dependencies are all COMPLETED and that are
        themselves still PENDING — i.e., ready to be assigned to a worker.
        """
        completed_ids = {t.task_id for t in self.tasks if t.status == TaskStatus.COMPLETED}
        return [
            t
            for t in self.tasks
            if t.status == TaskStatus.PENDING and all(dep in completed_ids for dep in t.depends_on)
        ]

    def add_message(self, role: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        """Append a message to the conversation history and update timestamp."""
        self.messages.append(Message(role=role, content=content, metadata=metadata or {}))
        self.updated_at = datetime.now(UTC)

    def add_task(self, task: Task) -> None:
        """Append a task to the task list and update timestamp."""
        self.tasks.append(task)
        self.updated_at = datetime.now(UTC)

    def transition(self, new_status: SessionStatus) -> None:
        """Update session status and timestamp — the canonical transition point."""
        self.status = new_status
        self.updated_at = datetime.now(UTC)

    def record_step(self, count: int = 1) -> int:
        """Increment total steps used in session circuit breaker state."""
        self.circuit_breaker.total_steps += count
        self.updated_at = datetime.now(UTC)
        return self.circuit_breaker.total_steps

    def to_dict(self) -> dict[str, Any]:
        """Serialize SessionState to JSON-serializable dictionary."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionState:
        """Construct SessionState from dictionary."""
        return cls.model_validate(data)

    def save(self, path: Path | str | None = None) -> Path:
        """
        Flush SessionState to disk as JSON.
        Defaults to .helm/sessions/<session_id>.json.
        """
        if path is not None:
            target_path = Path(path)
        else:
            target_path = Path(".helm") / "sessions" / f"{self.session_id}.json"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target_path

    @classmethod
    def load(cls, path: Path | str) -> SessionState:
        """Load SessionState from a checkpoint JSON file."""
        source_path = Path(path)
        if not source_path.exists():
            raise FileNotFoundError(f"Session checkpoint file not found: {source_path}")
        content = source_path.read_text(encoding="utf-8")
        data = json.loads(content)
        return cls.from_dict(data)


# Alias for backwards/milestone compatibility
AgentTaskState = SessionState

# Re-exports for Milestone 6 convenience
from helm.core.task_graph import CyclicDependencyError, StepStatus, TaskGraph, TaskStep  # noqa: E402

__all__ = [
    "AgentTaskState",
    "CircuitBlockReason",
    "CircuitBreakerState",
    "CyclicDependencyError",
    "Message",
    "ModelTier",
    "SessionState",
    "SessionStatus",
    "StepStatus",
    "Task",
    "TaskBrief",
    "TaskGraph",
    "TaskResult",
    "TaskStatus",
    "TaskStep",
    "ToolCallRecord",
]
