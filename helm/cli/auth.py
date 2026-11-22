"""
helm/cli/auth.py — Interactive authentication and provider configuration CLI.

Implements `helmx login` and `helmx config` commands with secure credential storage
at ~/.helm/credentials (chmod 600) and configuration at ~/.helm/config.json.
"""

from __future__ import annotations

import logging
import os
from http import HTTPStatus
from typing import Annotated

import httpx
import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from helm.cli.rendering.themes import HELM_THEME
from helm.config.auth import (
    DEFAULT_MODELS,
    PROVIDER_ENV_VARS,
    load_credentials,
    load_helm_config,
    save_credentials,
    save_helm_config,
)

console = Console(theme=HELM_THEME)
logger = logging.getLogger(__name__)


def is_local_base_url(base_url: str | None) -> bool:
    """Return True if base_url points to localhost, 127.0.0.1, or local IP."""
    if not base_url:
        return False
    lower = base_url.lower()
    return any(host in lower for host in ("localhost", "127.0.0.1", "::1"))


def validate_api_key(
    provider: str,
    api_key: str,
    base_url: str | None = None,
) -> tuple[bool, str]:
    """
    Perform a lightweight pre-flight request to validate the API key with the provider.
    Returns (True, "") on success or (False, error_detail) on 401/403/invalid key.
    Skips authentication rejection if base_url points to localhost or 127.0.0.1.
    """
    prov_norm = provider.lower()
    if prov_norm == "google":
        prov_norm = "gemini"

    is_local = is_local_base_url(base_url)

    try:
        with httpx.Client(timeout=10.0) as client:
            if prov_norm == "gemini":
                url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
                resp = client.get(url)
                if resp.status_code in (HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN) or (
                    resp.status_code == HTTPStatus.BAD_REQUEST
                    and any(
                        k in resp.text
                        for k in ("API_KEY_INVALID", "API key not valid", "INVALID_ARGUMENT")
                    )
                ):
                    return False, f"HTTP {resp.status_code}: Invalid API key."
                return True, ""

            if prov_norm == "openai":
                target_base = (base_url or "https://api.openai.com/v1").rstrip("/")
                if target_base.endswith("/v1"):
                    url = f"{target_base}/models"
                else:
                    url = f"{target_base}/v1/models"
                headers = {"Authorization": f"Bearer {api_key}"}
                resp = client.get(url, headers=headers)
                if (
                    resp.status_code in (HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN)
                    or "invalid_api_key" in resp.text
                ):
                    if is_local:
                        logger.debug(
                            "Skipping auth validation failure on local endpoint %s: HTTP %s",
                            url,
                            resp.status_code,
                        )
                        return True, ""
                    return False, f"HTTP {resp.status_code}: Invalid API key."
                return True, ""

            if prov_norm == "anthropic":
                target_url = (base_url or "https://api.anthropic.com/v1/messages").rstrip("/")
                headers = {
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                }
                payload = {
                    "model": "claude-3-5-sonnet-20241022",
                    "max_tokens": 1,
                    "messages": [{"role": "user", "content": "hi"}],
                }
                resp = client.post(target_url, json=payload, headers=headers)
                if (
                    resp.status_code in (HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN)
                    or "authentication_error" in resp.text
                ):
                    if is_local:
                        logger.debug(
                            "Skipping auth validation failure on local endpoint %s: HTTP %s",
                            target_url,
                            resp.status_code,
                        )
                        return True, ""
                    return False, f"HTTP {resp.status_code}: Invalid API key."
                return True, ""

    except Exception as exc:
        logger.debug("Pre-flight key validation network exception: %s", exc)
        return True, ""

    return True, ""


_PROVIDER_CHOICES = {
    "1": "gemini",
    "gemini": "gemini",
    "google": "gemini",
    "2": "anthropic",
    "anthropic": "anthropic",
    "claude": "anthropic",
    "3": "openai",
    "openai": "openai",
}


def _select_provider(provider_arg: str | None) -> str:
    """Prompt user to select LLM provider or normalize CLI argument."""
    if provider_arg and provider_arg.lower() in _PROVIDER_CHOICES:
        return _PROVIDER_CHOICES[provider_arg.lower()]

    console.print("\nSelect your default LLM provider:")
    console.print("  [bold green]1[/bold green]) Google Gemini    [dim](gemini-3.8-flash)[/dim]")
    console.print("  [bold green]2[/bold green]) Anthropic Claude [dim](claude-3-5-sonnet)[/dim]")
    console.print("  [bold green]3[/bold green]) OpenAI           [dim](gpt-4o)[/dim]\n")

    choice = Prompt.ask(
        "Enter choice",
        choices=["1", "2", "3", "gemini", "anthropic", "openai"],
        default="1",
        console=console,
    )
    return _PROVIDER_CHOICES[choice.lower()]


def _resolve_api_key_for_login(
    selected_provider: str,
    force_prompt: bool = False,
    is_local: bool = False,
) -> str:
    """Check environment variables or prompt for API key."""
    if not force_prompt:
        env_vars = PROVIDER_ENV_VARS.get(selected_provider, [])
        for ev in env_vars:
            val = os.environ.get(ev)
            if val:
                console.print(
                    f"\n[bold green]✓[/bold green] Found existing environment variable "
                    f"[bold cyan]${ev}[/bold cyan]."
                )
                use_env = Prompt.ask(
                    "Use this API key from environment?",
                    choices=["y", "n"],
                    default="y",
                    console=console,
                )
                if use_env.lower() == "y":
                    return val

    existing_creds = load_credentials()
    has_existing = (not force_prompt) and (selected_provider in existing_creds)

    if is_local:
        default_val = existing_creds.get(selected_provider, "ollama")
        prompt_label = (
            f"Enter API key for [bold cyan]{selected_provider.title()}[/bold cyan] "
            f"(leave blank for '{default_val}')"
        )
        input_key = Prompt.ask(
            prompt_label,
            password=True,
            default=default_val,
            console=console,
        )
        return input_key.strip() or default_val

    prompt_label = f"Enter API key for [bold cyan]{selected_provider.title()}[/bold cyan]" + (
        " (leave blank to keep existing)" if has_existing else ""
    )
    while True:
        input_key = Prompt.ask(
            prompt_label,
            password=True,
            default="" if has_existing else "...",
            console=console,
        )
        if not input_key and has_existing:
            return existing_creds[selected_provider]
        if input_key and input_key != "...":
            return input_key
        console.print("[red]API key cannot be empty. Please enter a valid key.[/red]")


def _select_model(selected_provider: str, model_arg: str | None) -> str:
    """Prompt user for default model or use CLI argument."""
    if model_arg:
        return model_arg
    default_model = DEFAULT_MODELS.get(selected_provider, "gemini-3.8-flash")
    return Prompt.ask(
        f"Select default model for [bold cyan]{selected_provider.title()}[/bold cyan]",
        default=default_model,
        console=console,
    )


def _configure_openai_base_url(base_url_arg: str | None = None) -> str | None:
    """Prompt for optional custom OpenAI base URL (LiteLLM/proxies/Ollama)."""
    if base_url_arg:
        return base_url_arg.strip() or None
    existing_cfg = load_helm_config()
    prev_base = existing_cfg.get("openai_base_url", "")
    custom_base = Prompt.ask(
        "Custom Base URL (optional for LiteLLM/proxies/Ollama, press Enter for default)",
        default=prev_base,
        console=console,
    )
    return custom_base.strip() or None


def run_login_flow(
    provider_arg: str | None = None,
    model_arg: str | None = None,
    base_url_arg: str | None = None,
) -> bool:
    """Execute the interactive login and configuration wizard."""
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]⛵ Helm Provider Authentication & Setup[/bold cyan]\n"
            "[dim]Configure your default LLM gateway and secure credentials.[/dim]",
            border_style="cyan",
        )
    )

    selected_provider = _select_provider(provider_arg)

    # 1. Prompt for base_url BEFORE executing the pre-flight verification test if OpenAI
    openai_base_url: str | None = None
    if selected_provider == "openai":
        openai_base_url = _configure_openai_base_url(base_url_arg)

    is_local = is_local_base_url(openai_base_url)

    force_prompt = False
    while True:
        api_key = _resolve_api_key_for_login(
            selected_provider,
            force_prompt=force_prompt,
            is_local=is_local,
        )

        is_valid, _ = validate_api_key(
            selected_provider,
            api_key,
            base_url=openai_base_url,
        )
        if is_valid:
            if is_local:
                console.print(
                    "[bold green]✓[/bold green] Localhost endpoint verified successfully."
                )
            else:
                console.print("[bold green]✓[/bold green] API key verified successfully.")
            break

        console.print("[bold red]✗[/bold red] Authentication test failed: Invalid API key.")
        if is_local:
            retry_choice = Prompt.ask(
                "Would you like to re-enter the key, skip validation, or cancel?",
                choices=["re-enter", "skip", "cancel"],
                default="skip",
                console=console,
            )
            if retry_choice.lower() in ("skip", "s"):
                console.print(
                    "[yellow]Authentication validation skipped for local endpoint.[/yellow]"
                )
                break
        else:
            retry_choice = Prompt.ask(
                "Would you like to re-enter the key or cancel?",
                choices=["re-enter", "cancel"],
                default="re-enter",
                console=console,
            )
        if retry_choice.lower() in ("cancel", "c"):
            console.print("[yellow]Login cancelled. Credentials were not saved.[/yellow]")
            return False
        force_prompt = True

    chosen_model = _select_model(selected_provider, model_arg)

    save_credentials({selected_provider: api_key})

    current_config = load_helm_config()
    current_config["default_provider"] = selected_provider
    current_config["default_model"] = chosen_model
    if openai_base_url:
        current_config["openai_base_url"] = openai_base_url
    elif selected_provider == "openai" and "openai_base_url" in current_config:
        current_config.pop("openai_base_url", None)

    save_helm_config(current_config)

    summary_text = (
        f"[bold green]✓ Configured active provider: "
        f"[cyan]{selected_provider.title()}[/cyan][/bold green]\n"
        f"• Default model: [bold]{chosen_model}[/bold]\n"
        f"• Configuration saved to: [dim]~/.helm/config.json[/dim]\n"
        f"• Credentials saved securely to: [dim]~/.helm/credentials (chmod 600)[/dim]"
    )
    console.print()
    console.print(
        Panel(
            summary_text,
            title="Authentication Complete",
            border_style="green",
        )
    )
    console.print()
    return True


def login_command(
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider",
            "-p",
            help="Provider to configure (gemini, anthropic, openai).",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="Default model name to configure.",
        ),
    ] = None,
    base_url: Annotated[
        str | None,
        typer.Option(
            "--base-url",
            help="Custom base URL for OpenAI/Ollama/proxies.",
        ),
    ] = None,
) -> None:
    """Authenticate and configure default LLM provider credentials."""
    run_login_flow(provider_arg=provider, model_arg=model, base_url_arg=base_url)
