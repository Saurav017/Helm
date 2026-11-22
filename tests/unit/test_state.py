"""
tests/unit/test_state.py — Unit tests for helm.core.state

Covers:
- All enum values exist and are strings
- ToolCallRecord hash validation
- Task self-dependency validation
- Task.depends_on resolution (runnable_tasks, pending_tasks)
- SessionState.add_message / add_task / transition
- Message role validation
- TaskBrief frozen immutability
"""

from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError

from helm.core.state import (
    CircuitBlockReason,
    CircuitBreakerState,
    Message,
    ModelTier,
    SessionState,
    SessionStatus,
    Task,
    TaskBrief,
    TaskStatus,
    ToolCallRecord,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_HASH = "a" * 64  # 64 lowercase hex chars


def make_state(goal: str = "test goal") -> SessionState:
    return SessionState(user_goal=goal)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class TestEnums:
    def test_task_status_values(self):
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.COMPLETED == "completed"
        assert TaskStatus.BLOCKED == "blocked"

    def test_session_status_values(self):
        assert SessionStatus.IDLE == "idle"
        assert SessionStatus.AWAITING_CONFIRMATION == "awaiting_confirmation"

    def test_model_tier_values(self):
        assert ModelTier.FAST == "fast"
        assert ModelTier.STANDARD == "standard"
        assert ModelTier.DEEP == "deep"

    def test_circuit_block_reason_values(self):
        assert CircuitBlockReason.DUPLICATE_CALL == "duplicate_call"
        assert CircuitBlockReason.STEP_CEILING == "step_ceiling"
        assert CircuitBlockReason.BUDGET_CAP == "budget_cap"


# ---------------------------------------------------------------------------
# ToolCallRecord
# ---------------------------------------------------------------------------


class TestToolCallRecord:
    def test_valid_hash_accepted(self):
        rec = ToolCallRecord(tool_name="fs_read", arguments={}, call_hash=VALID_HASH)
        assert rec.call_hash == VALID_HASH

    def test_short_hash_rejected(self):
        with pytest.raises(ValidationError, match="SHA-256"):
            ToolCallRecord(tool_name="fs_read", arguments={}, call_hash="abc123")

    def test_non_hex_hash_rejected(self):
        with pytest.raises(ValidationError, match="SHA-256"):
            ToolCallRecord(tool_name="fs_read", arguments={}, call_hash="Z" * 64)

    def test_record_is_frozen(self):
        rec = ToolCallRecord(tool_name="fs_read", arguments={}, call_hash=VALID_HASH)
        with pytest.raises(ValidationError):
            rec.tool_name = "other"  # type: ignore[misc]

    def test_blocked_record_fields(self):
        rec = ToolCallRecord(
            tool_name="fs_write",
            arguments={"path": "/workspace/file.txt"},
            call_hash=VALID_HASH,
            was_blocked=True,
            block_reason=CircuitBlockReason.DUPLICATE_CALL,
        )
        assert rec.was_blocked is True
        assert rec.block_reason == CircuitBlockReason.DUPLICATE_CALL

    def test_default_call_id_is_uuid(self):
        rec = ToolCallRecord(tool_name="fs_read", arguments={}, call_hash=VALID_HASH)
        uuid.UUID(rec.call_id)  # raises if not valid UUID


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------


class TestMessage:
    def test_valid_roles(self):
        for role in ("user", "assistant", "system", "tool"):
            m = Message(role=role, content="hello")
            assert m.role == role

    def test_invalid_role_rejected(self):
        with pytest.raises(ValidationError, match="role"):
            Message(role="admin", content="hello")

    def test_message_is_frozen(self):
        m = Message(role="user", content="hi")
        with pytest.raises(ValidationError):
            m.content = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


class TestTask:
    def test_default_status_is_pending(self):
        t = Task(description="do something")
        assert t.status == TaskStatus.PENDING

    def test_self_dependency_rejected(self):
        tid = str(uuid.uuid4())
        with pytest.raises(ValueError, match="cannot depend on itself"):
            Task(task_id=tid, description="x", depends_on=[tid])

    def test_dependencies_are_stored(self):
        dep_id = str(uuid.uuid4())
        t = Task(description="x", depends_on=[dep_id])
        assert dep_id in t.depends_on


# ---------------------------------------------------------------------------
# TaskBrief
# ---------------------------------------------------------------------------


class TestTaskBrief:
    def test_frozen(self):
        brief = TaskBrief(task_id="t1", description="do x")
        with pytest.raises(ValidationError):
            brief.description = "changed"  # type: ignore[misc]

    def test_defaults(self):
        brief = TaskBrief(task_id="t1", description="do x")
        assert brief.max_steps == 25
        assert brief.budget_usd == 0.50


# ---------------------------------------------------------------------------
# CircuitBreakerState
# ---------------------------------------------------------------------------


class TestCircuitBreakerState:
    def test_defaults(self):
        cb = CircuitBreakerState()
        assert cb.total_steps == 0
        assert cb.total_cost_usd == 0.0
        assert cb.duplicate_detection_enabled is True
        assert len(cb.seen_call_hashes) == 0

    def test_mutation(self):
        cb = CircuitBreakerState()
        cb.total_steps += 1
        assert cb.total_steps == 1


# ---------------------------------------------------------------------------
# SessionState
# ---------------------------------------------------------------------------


class TestSessionState:
    def test_initial_status_is_idle(self):
        s = make_state()
        assert s.status == SessionStatus.IDLE

    def test_session_id_is_uuid(self):
        s = make_state()
        uuid.UUID(s.session_id)

    def test_add_message(self):
        s = make_state()
        s.add_message("user", "hello")
        assert len(s.messages) == 1
        assert s.messages[0].content == "hello"

    def test_add_task(self):
        s = make_state()
        t = Task(description="task A")
        s.add_task(t)
        assert len(s.tasks) == 1

    def test_transition_updates_status_and_timestamp(self):
        s = make_state()
        before = s.updated_at
        s.transition(SessionStatus.RUNNING)
        assert s.status == SessionStatus.RUNNING
        assert s.updated_at >= before

    def test_get_task_found(self):
        s = make_state()
        t = Task(description="x")
        s.add_task(t)
        assert s.get_task(t.task_id) is t

    def test_get_task_not_found(self):
        s = make_state()
        assert s.get_task("nonexistent") is None

    def test_pending_tasks(self):
        s = make_state()
        t1 = Task(description="a", status=TaskStatus.PENDING)
        t2 = Task(description="b", status=TaskStatus.COMPLETED)
        s.add_task(t1)
        s.add_task(t2)
        assert s.pending_tasks() == [t1]

    def test_runnable_tasks_no_deps(self):
        """Tasks with no dependencies are immediately runnable."""
        s = make_state()
        t = Task(description="a")
        s.add_task(t)
        assert t in s.runnable_tasks()

    def test_runnable_tasks_with_incomplete_dep(self):
        """Tasks whose dependency is still PENDING should NOT be runnable."""
        s = make_state()
        t1 = Task(description="dep")
        t2 = Task(description="needs dep", depends_on=[t1.task_id])
        s.add_task(t1)
        s.add_task(t2)
        # t1 is PENDING, so t2 is NOT runnable
        assert t2 not in s.runnable_tasks()

    def test_runnable_tasks_with_completed_dep(self):
        """Tasks whose dependency is COMPLETED should be runnable."""
        s = make_state()
        t1 = Task(description="dep", status=TaskStatus.COMPLETED)
        t2 = Task(description="needs dep", depends_on=[t1.task_id])
        s.add_task(t1)
        s.add_task(t2)
        assert t2 in s.runnable_tasks()
