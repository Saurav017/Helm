"""
helm/core/circuit.py — Circuit breaker module alias.

Re-exports core circuit breaker classes and functions from
``helm.core.circuit_breaker`` for milestone compatibility.
"""

from __future__ import annotations

from helm.core.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerResult,
    CircuitBreakerTripped,
    HelmError,
    _compute_call_hash,
)

__all__ = [
    "CircuitBreaker",
    "CircuitBreakerResult",
    "CircuitBreakerTripped",
    "HelmError",
    "_compute_call_hash",
]
