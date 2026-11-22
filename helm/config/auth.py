"""
helm/config/auth.py — Persistent configuration and credential management.

Handles reading and writing ~/.helm/config.json and restricted credentials
at ~/.helm/credentials (chmod 600) according to INSTRUCTIONS.md security standards.
"""

from __future__ import annotations

import json
import logging
import os
import stat
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MODELS: dict[str, str] = {
    "gemini": "gemini-3.8-flash",
    "anthropic": "claude-3-5-sonnet-latest",
    "openai": "gpt-4o",
}

PROVIDER_ENV_VARS: dict[str, list[str]] = {
    "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
}


def get_helm_home_dir(create: bool = False) -> Path:
    """Return ~/.helm directory path, ensuring it exists with 0700 permissions when created."""
    if "HELM_HOME" in os.environ:
        d = Path(os.environ["HELM_HOME"])
        if create and not d.exists():
            d.mkdir(parents=True, exist_ok=True)
        return d

    home_helm = Path.home() / ".helm"
    if home_helm.exists():
        return home_helm

    if create:
        try:
            home_helm.mkdir(parents=True, exist_ok=True)
            try:
                home_helm.chmod(stat.S_IRWXU)  # 0700
            except OSError:
                logger.debug("Could not chmod 0700 on %s", home_helm)
        except OSError:
            # Fallback to local workspace directory
            local_helm = Path(".helm")
            local_helm.mkdir(parents=True, exist_ok=True)
            return local_helm
        else:
            return home_helm

    # For read operations when ~/.helm does not exist, check local workspace .helm
    local_helm = Path(".helm")
    if local_helm.exists():
        return local_helm
    return home_helm


def get_config_path(create_parent: bool = False) -> Path:
    """Return path to ~/.helm/config.json."""
    return get_helm_home_dir(create=create_parent) / "config.json"


def get_credentials_path(create_parent: bool = False) -> Path:
    """Return path to ~/.helm/credentials."""
    return get_helm_home_dir(create=create_parent) / "credentials"


def load_helm_config() -> dict[str, Any]:
    """Load configuration from ~/.helm/config.json."""
    config_file = get_config_path(create_parent=False)
    if not config_file.exists():
        return {}
    try:
        with config_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.warning("Failed to parse %s: %s", config_file, exc)
        return {}


def save_helm_config(config: dict[str, Any]) -> None:
    """Save configuration to ~/.helm/config.json."""
    config_file = get_config_path(create_parent=True)
    tmp_file = config_file.with_suffix(".tmp")
    with tmp_file.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    tmp_file.replace(config_file)


def load_credentials() -> dict[str, str]:
    """Load stored credentials from ~/.helm/credentials."""
    creds_file = get_credentials_path(create_parent=False)
    if not creds_file.exists():
        return {}
    try:
        with creds_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return {str(k).lower(): str(v) for k, v in data.items()}
    except Exception as exc:
        logger.warning("Failed to parse %s: %s", creds_file, exc)
    return {}


def save_credentials(credentials: dict[str, str]) -> None:
    """
    Save credentials to ~/.helm/credentials with restricted permissions (chmod 600).
    """
    creds_file = get_credentials_path(create_parent=True)
    # Normalize keys to lowercase
    normalized = {str(k).lower(): str(v) for k, v in credentials.items()}

    # Merge with existing credentials
    existing = load_credentials()
    existing.update(normalized)

    tmp_file = creds_file.with_suffix(".tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    mode = stat.S_IRUSR | stat.S_IWUSR  # 0600
    fd = os.open(tmp_file, flags, mode)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)

    try:
        tmp_file.chmod(mode)
    except OSError:
        logger.debug("chmod 600 failed on temporary credentials file")

    tmp_file.replace(creds_file)
    try:
        creds_file.chmod(mode)
    except OSError:
        logger.debug("chmod 600 failed on credentials file")


def resolve_api_key(provider: str) -> str | None:
    """
    Resolve API key for the specified provider prioritizing shell environment
    variables, falling back to ~/.helm/credentials.
    """
    prov_norm = provider.lower()
    if prov_norm == "google":
        prov_norm = "gemini"

    # 1. Environment variables
    env_keys = PROVIDER_ENV_VARS.get(prov_norm, [])
    for env_key in env_keys:
        val = os.environ.get(env_key)
        if val:
            return val.strip()

    # 2. Stored credentials
    creds = load_credentials()
    if prov_norm in creds:
        return creds[prov_norm]
    # Check alternate keys like "gemini_api_key"
    alt_key = f"{prov_norm}_api_key"
    if alt_key in creds:
        return creds[alt_key]

    return None


def resolve_active_provider(
    cli_provider: str | None = None,
    cli_model: str | None = None,
) -> tuple[str, str, str | None]:
    """
    Resolve active provider, model name, and API key using hierarchy:
      1. CLI runtime flag (--provider, --model)
      2. Environment variables (ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY)
      3. Local config file ~/.helm/config.json
    """
    cfg = load_helm_config()
    default_from_cfg = str(cfg.get("default_provider", "")).lower()

    resolved_provider: str = ""
    # 1. CLI flag
    if cli_provider:
        resolved_provider = cli_provider.lower()
    # 2. Environment variables
    elif os.environ.get("ANTHROPIC_API_KEY"):
        resolved_provider = "anthropic"
    elif os.environ.get("OPENAI_API_KEY"):
        resolved_provider = "openai"
    elif os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        resolved_provider = "gemini"
    # 3. Config file
    elif default_from_cfg in DEFAULT_MODELS:
        resolved_provider = default_from_cfg
    else:
        resolved_provider = "gemini"

    if resolved_provider == "google":
        resolved_provider = "gemini"

    # Resolve model
    resolved_model: str = ""
    if cli_model:
        resolved_model = cli_model
    elif default_from_cfg == resolved_provider and cfg.get("default_model"):
        resolved_model = str(cfg["default_model"])
    else:
        resolved_model = DEFAULT_MODELS.get(resolved_provider, "gemini-3.8-flash")

    api_key = resolve_api_key(resolved_provider)
    return resolved_provider, resolved_model, api_key
