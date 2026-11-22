"""
helm/core/circuit_breaker.py — Deterministic safety guardrails.

The Circuit Breaker sits between the Supervisor and the MCP layer and CANNOT
be bypassed by workers or the CLI (Architecture Invariant 5).

It enforces three hard limits on every tool invocation:
1. **Duplicate detection** — SHA-256 hash of (tool_name, canonical_arguments).
   Identical re-invocations are blocked and logged.
2. **Step ceiling** — Global step counter across the session.  Exceeding
   ``max_steps`` force-terminates the call.
3. **Budget cap** — Cumulative USD cost tally.  Exceeding ``budget_usd``
   blocks further calls.

Every blocked call is logged with its reason (INSTRUCTIONS.md §3.4).
Silent failures are forbidden (INSTRUCTIONS.md §2.3).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any

from helm.core.state import CircuitBlockReason, CircuitBreakerState, ToolCallRecord
from helm.exceptions import CircuitBreakerError, HelmError

logger = logging.getLogger(__name__)

__all__ = [
    "CircuitBreaker",
    "CircuitBreakerError",
    "CircuitBreakerResult",
    "CircuitBreakerTripped",
    "HelmError",
    "_compute_call_hash",
]


# ---------------------------------------------------------------------------
# Exceptions (all inherit HelmError — INSTRUCTIONS.md §2.3)
# ---------------------------------------------------------------------------


class CircuitBreakerTripped(CircuitBreakerError):
    """
    Raised when the Circuit Breaker blocks a tool call.

    Attributes
    ----------
    reason:
        Machine-readable reason code.
    tool_name:
        The tool that was about to be invoked.
    call_hash:
        SHA-256 digest of the blocked call.
    """

    def __init__(
        self,
        reason: CircuitBlockReason,
        tool_name: str,
        call_hash: str,
        detail: str = "",
    ) -> None:
        self.reason = reason
        self.tool_name = tool_name
        self.call_hash = call_hash
        self.detail = detail
        super().__init__(
            f"[CircuitBreaker] {reason.value} — tool={tool_name!r} hash={call_hash[:12]}… {detail}"
        )


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CircuitBreakerResult:
    """
    Returned by :func:`check` for every tool call attempt.

    ``allowed`` is True when the call may proceed; False when it was blocked.
    ``record`` is the :class:`~helm.core.state.ToolCallRecord` created for
    this attempt (blocked calls have ``was_blocked=True``).
    """

    allowed: bool
    record: ToolCallRecord


# ---------------------------------------------------------------------------
# Hash helper
# ---------------------------------------------------------------------------


def _compute_call_hash(tool_name: str, arguments: dict[str, Any]) -> str:
    """
    Return a 64-char lowercase SHA-256 hex digest of the canonical call.

    ``arguments`` is JSON-serialised with sorted keys so that
    ``{"b": 2, "a": 1}`` and ``{"a": 1, "b": 2}`` produce the same hash.
    """
    canonical = json.dumps({"tool": tool_name, "args": arguments}, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------


class CircuitBreaker:
    """
    Stateless callable that evaluates ``CircuitBreakerState`` and returns
    a :class:`CircuitBreakerResult`.

    Usage
    -----
    ::

        cb = CircuitBreaker()
        result = cb.check(state.circuit_breaker, tool_name="fs_read", arguments={...})
        if not result.allowed:
            # call was blocked; result.record explains why
            raise CircuitBreakerTripped(...)

    The caller is responsible for appending ``result.record`` to the
    appropriate :class:`~helm.core.state.Task`.
    """

    def check(
        self,
        cb_state: CircuitBreakerState,
        *,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> CircuitBreakerResult:
        """
        Evaluate all three circuit-breaker conditions.

        Parameters
        ----------
        cb_state:
            Mutable circuit-breaker state embedded in ``SessionState``.
            Updated in-place when a call is allowed.
        tool_name:
            Name of the MCP tool being invoked.
        arguments:
            Raw argument dict for the call.

        Returns
        -------
        CircuitBreakerResult
            ``allowed=True`` and an updated record when the call may proceed;
            ``allowed=False`` and a blocked record otherwise.
        """
        call_hash = _compute_call_hash(tool_name, arguments)

        # ── 1. Duplicate detection ───────────────────────────────────────
        if cb_state.duplicate_detection_enabled and call_hash in cb_state.seen_call_hashes:
            record = ToolCallRecord(
                tool_name=tool_name,
                arguments=arguments,
                call_hash=call_hash,
                was_blocked=True,
                block_reason=CircuitBlockReason.DUPLICATE_CALL,
            )
            logger.warning(
                "Circuit breaker blocked duplicate call",
                extra={
                    "event": "circuit_breaker_blocked",
                    "component": "CircuitBreaker",
                    "reason": CircuitBlockReason.DUPLICATE_CALL,
                    "tool_name": tool_name,
                    "call_hash": call_hash,
                },
            )
            return CircuitBreakerResult(allowed=False, record=record)

        # ── 2. Step ceiling ──────────────────────────────────────────────
        if cb_state.total_steps >= cb_state.max_steps:
            record = ToolCallRecord(
                tool_name=tool_name,
                arguments=arguments,
                call_hash=call_hash,
                was_blocked=True,
                block_reason=CircuitBlockReason.STEP_CEILING,
            )
            logger.warning(
                "Circuit breaker blocked call: step ceiling reached",
                extra={
                    "event": "circuit_breaker_blocked",
                    "component": "CircuitBreaker",
                    "reason": CircuitBlockReason.STEP_CEILING,
                    "tool_name": tool_name,
                    "total_steps": cb_state.total_steps,
                    "max_steps": cb_state.max_steps,
                },
            )
            return CircuitBreakerResult(allowed=False, record=record)

        # ── 3. Budget cap ────────────────────────────────────────────────
        if cb_state.total_cost_usd >= cb_state.budget_usd:
            record = ToolCallRecord(
                tool_name=tool_name,
                arguments=arguments,
                call_hash=call_hash,
                was_blocked=True,
                block_reason=CircuitBlockReason.BUDGET_CAP,
            )
            logger.warning(
                "Circuit breaker blocked call: budget cap reached",
                extra={
                    "event": "circuit_breaker_blocked",
                    "component": "CircuitBreaker",
                    "reason": CircuitBlockReason.BUDGET_CAP,
                    "tool_name": tool_name,
                    "total_cost_usd": cb_state.total_cost_usd,
                    "budget_usd": cb_state.budget_usd,
                },
            )
            return CircuitBreakerResult(allowed=False, record=record)

        # ── All checks passed — allow the call ───────────────────────────
        cb_state.seen_call_hashes.add(call_hash)
        cb_state.total_steps += 1

        record = ToolCallRecord(
            tool_name=tool_name,
            arguments=arguments,
            call_hash=call_hash,
            was_blocked=False,
        )

        logger.debug(
            "Circuit breaker allowed call",
            extra={
                "event": "circuit_breaker_allowed",
                "component": "CircuitBreaker",
                "tool_name": tool_name,
                "call_hash": call_hash,
                "total_steps": cb_state.total_steps,
            },
        )
        return CircuitBreakerResult(allowed=True, record=record)

    def record_cost(
        self,
        cb_state: CircuitBreakerState,
        *,
        tokens: int,
        cost_usd: float,
    ) -> None:
        """
        Accumulate token and cost counters after a successful LLM call.

        Must be called by the Supervisor after every LLM response.
        """
        cb_state.total_tokens += tokens
        cb_state.total_cost_usd += cost_usd
        logger.debug(
            "Circuit breaker cost updated",
            extra={
                "event": "circuit_breaker_cost_updated",
                "component": "CircuitBreaker",
                "tokens_added": tokens,
                "cost_usd_added": cost_usd,
                "total_tokens": cb_state.total_tokens,
                "total_cost_usd": cb_state.total_cost_usd,
            },
        )
