"""
tests/unit/test_circuit_breaker.py — Unit tests for helm.core.circuit_breaker

Covers:
- Hash computation is deterministic and order-independent
- Duplicate call detection blocks second identical invocation
- Step ceiling blocks calls once max_steps is reached
- Budget cap blocks calls once budget is exhausted
- Allowed calls update state correctly
- record_cost accumulates correctly
- CircuitBreakerTripped carries the right fields
"""

from __future__ import annotations

from helm.core import circuit
from helm.core.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerTripped,
    HelmError,
    _compute_call_hash,
)
from helm.core.state import CircuitBlockReason, CircuitBreakerState

# Stable test path: not using /tmp (S108)
_PATH_A = "/workspace/file_a.txt"
_PATH_B = "/workspace/file_b.txt"


# ---------------------------------------------------------------------------
# Hash helper
# ---------------------------------------------------------------------------


class TestComputeCallHash:
    def test_deterministic(self):
        h1 = _compute_call_hash("fs_read", {"path": _PATH_A})
        h2 = _compute_call_hash("fs_read", {"path": _PATH_A})
        assert h1 == h2

    def test_argument_order_independent(self):
        h1 = _compute_call_hash("fs_read", {"b": 2, "a": 1})
        h2 = _compute_call_hash("fs_read", {"a": 1, "b": 2})
        assert h1 == h2

    def test_different_tools_differ(self):
        h1 = _compute_call_hash("fs_read", {"path": _PATH_A})
        h2 = _compute_call_hash("fs_write", {"path": _PATH_A})
        assert h1 != h2

    def test_different_args_differ(self):
        h1 = _compute_call_hash("fs_read", {"path": _PATH_A})
        h2 = _compute_call_hash("fs_read", {"path": _PATH_B})
        assert h1 != h2

    def test_returns_64_char_hex(self):
        h = _compute_call_hash("tool", {})
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# CircuitBreaker.check
# ---------------------------------------------------------------------------


class TestCircuitBreakerCheck:
    def setup_method(self):
        self.cb = CircuitBreaker()

    def _fresh_state(self, **overrides: object) -> CircuitBreakerState:
        return CircuitBreakerState(**overrides)

    # -- Allow path ----------------------------------------------------------

    def test_first_call_is_allowed(self):
        state = self._fresh_state()
        result = self.cb.check(state, tool_name="fs_read", arguments={"path": _PATH_A})
        assert result.allowed is True
        assert result.record.was_blocked is False

    def test_allowed_call_increments_steps(self):
        state = self._fresh_state()
        self.cb.check(state, tool_name="fs_read", arguments={})
        assert state.total_steps == 1

    def test_allowed_call_adds_hash_to_seen(self):
        state = self._fresh_state()
        result = self.cb.check(state, tool_name="fs_read", arguments={"path": _PATH_A})
        assert result.record.call_hash in state.seen_call_hashes

    # -- Duplicate detection -------------------------------------------------

    def test_duplicate_call_is_blocked(self):
        state = self._fresh_state()
        self.cb.check(state, tool_name="fs_read", arguments={"path": _PATH_A})
        result = self.cb.check(state, tool_name="fs_read", arguments={"path": _PATH_A})
        assert result.allowed is False
        assert result.record.block_reason == CircuitBlockReason.DUPLICATE_CALL

    def test_duplicate_detection_disabled(self):
        state = self._fresh_state(duplicate_detection_enabled=False)
        self.cb.check(state, tool_name="fs_read", arguments={"path": _PATH_A})
        result = self.cb.check(state, tool_name="fs_read", arguments={"path": _PATH_A})
        # Duplicate detection off → second call allowed
        assert result.allowed is True

    def test_different_args_not_duplicate(self):
        state = self._fresh_state()
        self.cb.check(state, tool_name="fs_read", arguments={"path": _PATH_A})
        result = self.cb.check(state, tool_name="fs_read", arguments={"path": _PATH_B})
        assert result.allowed is True

    # -- Step ceiling --------------------------------------------------------

    def test_step_ceiling_blocks_at_limit(self):
        state = self._fresh_state(max_steps=2)
        self.cb.check(state, tool_name="t1", arguments={"i": 1})
        self.cb.check(state, tool_name="t2", arguments={"i": 2})
        # Steps == max_steps == 2 → next call blocked
        result = self.cb.check(state, tool_name="t3", arguments={"i": 3})
        assert result.allowed is False
        assert result.record.block_reason == CircuitBlockReason.STEP_CEILING

    def test_step_ceiling_just_below_limit_is_allowed(self):
        state = self._fresh_state(max_steps=3)
        self.cb.check(state, tool_name="t1", arguments={"i": 1})
        self.cb.check(state, tool_name="t2", arguments={"i": 2})
        # total_steps == 2 < 3 → allowed
        result = self.cb.check(state, tool_name="t3", arguments={"i": 3})
        assert result.allowed is True

    # -- Budget cap ----------------------------------------------------------

    def test_budget_cap_blocks_when_exceeded(self):
        state = self._fresh_state(budget_usd=0.10, total_cost_usd=0.11)
        result = self.cb.check(state, tool_name="expensive", arguments={})
        assert result.allowed is False
        assert result.record.block_reason == CircuitBlockReason.BUDGET_CAP

    def test_budget_cap_exactly_at_limit_blocks(self):
        state = self._fresh_state(budget_usd=1.00, total_cost_usd=1.00)
        result = self.cb.check(state, tool_name="expensive", arguments={})
        assert result.allowed is False

    def test_budget_just_under_limit_is_allowed(self):
        state = self._fresh_state(budget_usd=1.00, total_cost_usd=0.99)
        result = self.cb.check(state, tool_name="cheap", arguments={})
        assert result.allowed is True

    # -- Priority order: duplicates before step ceiling ----------------------

    def test_duplicate_checked_before_step_ceiling(self):
        """Duplicate detection fires before step ceiling to give clearest signal."""
        state = self._fresh_state(max_steps=1)
        # First call fills the hash set AND hits the step ceiling
        self.cb.check(state, tool_name="t1", arguments={})
        # Second identical call → should report DUPLICATE not STEP_CEILING
        result = self.cb.check(state, tool_name="t1", arguments={})
        assert result.record.block_reason == CircuitBlockReason.DUPLICATE_CALL


# ---------------------------------------------------------------------------
# CircuitBreaker.record_cost
# ---------------------------------------------------------------------------


class TestRecordCost:
    def test_record_cost_accumulates(self):
        cb = CircuitBreaker()
        state = CircuitBreakerState()
        cb.record_cost(state, tokens=100, cost_usd=0.002)
        cb.record_cost(state, tokens=50, cost_usd=0.001)
        assert state.total_tokens == 150
        assert abs(state.total_cost_usd - 0.003) < 1e-9


# ---------------------------------------------------------------------------
# Module alias coverage
# ---------------------------------------------------------------------------


class TestCircuitAlias:
    def test_reexports(self):
        assert circuit.CircuitBreaker is CircuitBreaker
        assert circuit.CircuitBreakerTripped is CircuitBreakerTripped


class TestExceptions:
    def test_circuit_breaker_tripped_is_helm_error(self):
        exc = CircuitBreakerTripped(
            reason=CircuitBlockReason.DUPLICATE_CALL,
            tool_name="fs_read",
            call_hash="a" * 64,
        )
        assert isinstance(exc, HelmError)

    def test_exception_message_contains_reason(self):
        exc = CircuitBreakerTripped(
            reason=CircuitBlockReason.STEP_CEILING,
            tool_name="bash_exec",
            call_hash="b" * 64,
        )
        assert "step_ceiling" in str(exc)
        assert "bash_exec" in str(exc)
