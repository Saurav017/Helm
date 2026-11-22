"""
helm/cli/safety_gate.py — Terminal Human-in-the-Loop (HITL) confirmation gate.

Re-exports and wraps ``helm.protocols.safety_gate`` for CLI presentation.
"""

from __future__ import annotations

from typing import Any

from rich.console import Console

from helm.protocols.safety_gate import (
    ActionSafety,
    SafetyDecision,
    SafetyGate,
    SafetyGateResult,
    classify_tool,
    is_mutating,
)

__all__ = [
    "ActionSafety",
    "SafetyDecision",
    "SafetyGate",
    "SafetyGateResult",
    "classify_tool",
    "confirm_mutation",
    "is_mutating",
]


def confirm_mutation(
    console: Console,
    *,
    tool_name: str,
    arguments: dict[str, Any],
    summary: str = "",
    timeout_sec: int = 60,
    auto_approve: bool = False,
) -> bool:
    """
    Convenience wrapper presenting a terminal HITL confirmation prompt.

    Delegates directly to ``SafetyGate``.
    """
    gate = SafetyGate(console=console, auto_approve=auto_approve, timeout_sec=timeout_sec)
    result = gate.evaluate(tool_name, arguments, summary=summary)
    return result.is_approved
