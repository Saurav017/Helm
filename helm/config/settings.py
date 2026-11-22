"""
helm/config/settings.py — Application settings loaded from TOML / env vars.

Configuration flows one way:
  config files → Settings model → components
Components never write back (Architecture Invariant 7).

Precedence (highest → lowest):
  1. Environment variables  ``HELMX_*``
  2. Project-local          ``./helmx.toml``
  3. User-global            ``~/.config/helmx/config.toml``
  4. Shipped defaults       ``configs/default.toml``
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Sub-settings groups
# ---------------------------------------------------------------------------


class SupervisorSettings(BaseModel):
    max_steps: int = Field(default=25, ge=1, le=1000)
    budget_usd: float = Field(default=2.00, ge=0.0)
    default_tier: str = "standard"


class ModelSettings(BaseModel):
    fast: str = "gemini-3.8-flash"
    standard: str = "gemini-3.8-flash"
    deep: str = "gemini-2.5-pro"


class SafetySettings(BaseModel):
    hitl_mutations: bool = True
    auto_approve_reads: bool = True
    confirmation_timeout_sec: int = Field(default=60, ge=1)


class CircuitBreakerSettings(BaseModel):
    duplicate_detection: bool = True
    max_step_ceiling: int = Field(default=25, ge=1, le=1000)
    budget_cap_enabled: bool = True


class ToolPruningSettings(BaseModel):
    enabled: bool = True
    top_k: int = Field(default=5, ge=1, le=50)
    similarity_threshold: float = Field(default=0.3, ge=0.0, le=1.0)


class LoggingSettings(BaseModel):
    level: str = "INFO"
    structured: bool = True
    file: str = ""  # empty = stderr only

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            msg = f"Log level must be one of {allowed!r}, got {v!r}"
            raise ValueError(msg)
        return upper


class MCPServerSettings(BaseModel):
    name: str
    transport: str = "stdio"
    command: str = ""
    args: list[str] = Field(default_factory=list)
    url: str = ""
    endpoint: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    auth_token: str = ""
    env: dict[str, str] = Field(default_factory=dict)
    description: str = ""

    @property
    def effective_endpoint(self) -> str:
        return self.endpoint or self.url

    @property
    def effective_headers(self) -> dict[str, str]:
        h = dict(self.headers)
        if self.auth_token and "Authorization" not in h:
            h["Authorization"] = f"Bearer {self.auth_token}"
        return h


# ---------------------------------------------------------------------------
# Root settings model
# ---------------------------------------------------------------------------


class Settings(BaseModel):
    """
    Root settings object.  Instantiate via :func:`load_settings` rather than
    directly so that TOML files and env-var overrides are applied correctly.
    """

    supervisor: SupervisorSettings = Field(default_factory=SupervisorSettings)
    models: ModelSettings = Field(default_factory=ModelSettings)
    safety: SafetySettings = Field(default_factory=SafetySettings)
    circuit_breaker: CircuitBreakerSettings = Field(default_factory=CircuitBreakerSettings)
    tool_pruning: ToolPruningSettings = Field(default_factory=ToolPruningSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    mcp_servers: list[MCPServerSettings] = Field(default_factory=list)

    model_config = {"frozen": False}


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _load_toml(path: Path) -> dict[str, Any]:
    """Read a TOML file, returning an empty dict if the file does not exist."""
    if not path.exists():
        return {}
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Apply ``HELMX_*`` environment variables on top of the TOML dict.

    Mapping (examples):
      ``HELMX_SUPERVISOR__MAX_STEPS=50`` → ``raw["supervisor"]["max_steps"] = 50``
      ``HELMX_SAFETY__HITL_MUTATIONS=false`` → ``raw["safety"]["hitl_mutations"] = False``
    """
    prefix = "HELMX_"
    for key, val in os.environ.items():
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix) :].lower().split("__")
        node: dict[str, Any] = raw
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        leaf = parts[-1]
        # Coerce to bool / int / float where possible
        if val.lower() in {"true", "false"}:
            node[leaf] = val.lower() == "true"
        else:
            try:
                node[leaf] = int(val)
            except ValueError:
                try:
                    node[leaf] = float(val)
                except ValueError:
                    node[leaf] = val
    return raw


def load_settings(config_path: Path | None = None) -> Settings:
    """
    Build a :class:`Settings` instance by merging (lowest → highest priority):

    1. Package defaults
    2. User-global ``~/.config/helmx/config.toml``
    3. Project-local ``./helmx.toml`` (or ``config_path`` if supplied)
    4. ``HELMX_*`` environment variables

    Returns
    -------
    Settings
        Fully validated settings object ready for injection into components.
    """
    raw: dict[str, Any] = {}

    # 2. User-global
    user_config = Path.home() / ".config" / "helmx" / "config.toml"
    raw.update(_load_toml(user_config))

    # 3. Project-local
    project_config = config_path or Path("helmx.toml")
    raw.update(_load_toml(project_config))

    # 4. Env-var overrides
    raw = _apply_env_overrides(raw)

    return Settings.model_validate(raw)
