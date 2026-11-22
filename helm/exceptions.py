"""
helm/exceptions.py — Top-level Helm exception hierarchy.

All custom exceptions across Helm inherit from HelmError (INSTRUCTIONS.md §2.3).
Placing the base exception here avoids layer violation and circular imports.
"""

from __future__ import annotations


class HelmError(Exception):
    """Base exception for all Helm runtime errors."""


class AuthenticationError(HelmError):
    """Raised when authentication credentials for an LLM provider are missing or invalid."""


class ConfigurationError(HelmError):
    """Raised when Helm runtime configuration is missing or invalid."""


class ProviderAPIError(HelmError):
    """Raised when an external LLM provider API call fails."""


class TransportError(HelmError):
    """Raised when MCP or network transport communication fails."""


class CircuitBreakerError(HelmError):
    """Raised when a circuit breaker blocks execution or trips."""
