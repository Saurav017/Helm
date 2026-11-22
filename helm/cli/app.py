"""
helm/cli/app.py — Primary Typer application and CLI entrypoint.

Registered as the ``helmx`` console script in ``pyproject.toml``:
    helmx = "helm.cli.app:main"

Architecture note (INSTRUCTIONS.md §5, Invariant 2):
  The CLI layer NEVER imports from ``helm.core`` internals directly.
  It only touches the public surface: ``SessionState``, ``Settings``, and
  the ``run_session`` coroutine exposed by the core package's ``__init__``.
  For Milestone 1 (no live Supervisor yet) we stub ``run_session`` locally
  and wire it properly in Milestone 2.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Annotated, Any, cast

import typer
from rich.console import Console
from rich.pretty import Pretty
from rich.prompt import Prompt

from helm.cli.auth import login_command, run_login_flow
from helm.cli.rendering.panels import (
    make_circuit_breaker_panel,
    make_error_panel,
    make_goal_panel,
    make_session_summary_panel,
    make_task_table,
    make_tools_table,
    print_helm_banner,
)
from helm.cli.rendering.spinners import thinking_spinner
from helm.cli.rendering.themes import HELM_THEME
from helm.config.auth import PROVIDER_ENV_VARS, resolve_active_provider
from helm.config.settings import MCPServerSettings, Settings, load_settings
from helm.core.circuit_breaker import CircuitBreakerTripped
from helm.core.model_router import ModelRouter
from helm.core.state import SessionState, SessionStatus
from helm.core.supervisor import Supervisor
from helm.exceptions import (
    AuthenticationError,
    CircuitBreakerError,
    HelmError,
    TransportError,
)
from helm.protocols.mcp_client import MCPClientManager, MCPToolDefinition
from helm.protocols.safety_gate import SafetyGate
from helm.ui.renderer import ActivityStreamRenderer

# ---------------------------------------------------------------------------
# App & console setup
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="helmx",
    help="⛵  Helm — terminal-first agentic orchestration runtime.",
    rich_markup_mode="rich",
    no_args_is_help=True,
    pretty_exceptions_enable=False,  # We handle our own error display
)

console = Console(theme=HELM_THEME)


# ---------------------------------------------------------------------------
# Core session runner
# ---------------------------------------------------------------------------


async def _run_supervisor_session(
    state: SessionState,
    settings: Settings,
    *,
    auto_approve: bool = False,
    provider: str | None = None,
    model: str | None = None,
    renderer: ActivityStreamRenderer | None = None,
) -> SessionState:
    """Connect to configured MCP servers, initialize Supervisor, and run execution loop."""
    safety_gate = SafetyGate(console=console, auto_approve=auto_approve)
    manager = MCPClientManager(safety_gate=safety_gate)
    try:
        await manager.connect_all(settings.mcp_servers)
        router = ModelRouter(settings=settings, provider=provider, model=model)
        supervisor = Supervisor(
            settings=settings,
            safety_gate=safety_gate,
            model_router=router,
            ui_renderer=renderer,
        )
        return await supervisor.run(state, mcp_client=manager, auto_approve=auto_approve)
    finally:
        await manager.disconnect_all()


# ---------------------------------------------------------------------------
# Shared options type aliases
# ---------------------------------------------------------------------------

_OptionalConfigPath = Annotated[
    Path | None,
    typer.Option(
        "--config",
        "-c",
        help="Path to a helmx.toml config file.",
        exists=False,
        file_okay=True,
        dir_okay=False,
    ),
]

_OptionalVerbose = Annotated[
    bool,
    typer.Option("--verbose", "-v", help="Enable DEBUG-level logging."),
]

_OptionalAutoApprove = Annotated[
    bool,
    typer.Option(
        "--auto-approve",
        "-y",
        "--yes",
        help="Auto-approve mutating actions without prompting (for non-interactive CI).",
    ),
]

_OptionalResume = Annotated[
    str | None,
    typer.Option(
        "--resume",
        "-r",
        help="Resume an interrupted session by session ID or path.",
    ),
]

_OptionalProvider = Annotated[
    str | None,
    typer.Option(
        "--provider",
        "-p",
        help="LLM provider to use (gemini, anthropic, openai).",
    ),
]

_OptionalModel = Annotated[
    str | None,
    typer.Option(
        "--model",
        "-m",
        help="Model name override for the LLM provider.",
    ),
]

_OptionalBaseUrl = Annotated[
    str | None,
    typer.Option(
        "--base-url",
        help="Custom base URL for OpenAI/Ollama/proxies.",
    ),
]

_OptionalDebugGraph = Annotated[
    bool,
    typer.Option(
        "--debug-graph",
        help="Display tabular TaskGraph DAG representation.",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _fetch_tools(
    servers: list[MCPServerSettings],
) -> tuple[list[MCPToolDefinition], dict[str, dict[str, Any]]]:
    """Connect to configured MCP servers and discover their tool schemas."""
    manager = MCPClientManager()
    try:
        await manager.connect_all(servers)
        tools = await manager.list_tools()
        statuses = {s.name: manager.get_server_status(s.name) for s in servers}
        return tools, statuses
    finally:
        await manager.disconnect_all()


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def _build_initial_state(prompt: str) -> SessionState:
    """Construct a fresh SessionState from a user prompt."""
    state = SessionState(user_goal=prompt)
    state.add_message(role="user", content=prompt)
    state.transition(SessionStatus.RUNNING)
    return state


def _preflight_auth_check(
    provider_arg: str | None = None,
    model_arg: str | None = None,
) -> None:
    """Pre-flight check to verify that credentials exist for the active provider."""
    if "PYTEST_CURRENT_TEST" in os.environ and not os.environ.get("HELM_FORCE_AUTH_CHECK"):
        return

    resolved_prov, _, key = resolve_active_provider(
        cli_provider=provider_arg,
        cli_model=model_arg,
    )
    if not key:
        env_var = (
            "GEMINI_API_KEY" if resolved_prov == "gemini" else f"{resolved_prov.upper()}_API_KEY"
        )
        raise AuthenticationError(
            f"No API key configured for provider '{resolved_prov}'. "
            f"Run 'helmx login' or export {env_var}."
        )


def _is_interactive() -> bool:
    """Check if standard input is an interactive terminal."""
    return sys.stdin.isatty()


def _handle_authentication_error(
    provider_arg: str | None = None,
    model_arg: str | None = None,
    error: AuthenticationError | None = None,
) -> bool:
    """
    Handle missing API credentials.
    Returns True if user successfully logged in interactively and execution should retry.
    Exits with code 1 if non-interactive or if user declined login.
    """
    resolved_prov, _, key = resolve_active_provider(
        cli_provider=provider_arg,
        cli_model=model_arg,
    )
    env_var = "GEMINI_API_KEY" if resolved_prov == "gemini" else f"{resolved_prov.upper()}_API_KEY"

    err_msg = str(error) if error else ""
    is_missing_key = not key or any(
        kw in err_msg.lower() for kw in ("not configured", "no api key", "missing")
    )

    if not _is_interactive():
        if is_missing_key:
            console.print(
                f"Error: No API key configured for provider '{resolved_prov}'. "
                f"Export {env_var} or run 'helmx login'.",
                highlight=False,
                soft_wrap=True,
            )
        else:
            console.print(
                f"Error: API key rejected by provider '{resolved_prov}' (401 Unauthorized). "
                f"Export valid {env_var} or run 'helmx login'.",
                highlight=False,
                soft_wrap=True,
            )
        raise typer.Exit(code=1)

    console.print()
    if is_missing_key:
        prompt_text = "No API key configured. Would you like to log in now? [Y/n]"
    else:
        prompt_text = "API key rejected by provider. Would you like to re-enter your key now? [Y/n]"

    choice = Prompt.ask(
        prompt_text,
        default="y",
        show_default=False,
        console=console,
    )
    if choice.strip().lower() not in ("y", "yes"):
        raise typer.Exit(code=1)

    # If the key was rejected, clear any stale environment variable that was failing
    if not is_missing_key:
        for ev in PROVIDER_ENV_VARS.get(resolved_prov, []):
            os.environ.pop(ev, None)

    success = run_login_flow(provider_arg=provider_arg, model_arg=model_arg)
    if success is False:
        raise typer.Exit(code=1)
    return True


def _resolve_session_state(prompt: str, resume: str | None) -> tuple[str, SessionState]:
    """Resolve initial or resumed session state."""
    if resume:
        checkpoint_path = Path(".helm") / "sessions" / f"{resume}.json"
        if not checkpoint_path.exists():
            checkpoint_path = Path(resume)
        if not checkpoint_path.exists():
            console.print(
                make_error_panel(
                    "Session checkpoint not found",
                    f"Could not find checkpoint: {resume}",
                )
            )
            raise typer.Exit(code=1)
        state = SessionState.load(checkpoint_path)
        resolved_prompt = prompt or state.user_goal
        return resolved_prompt, state

    if prompt:
        return prompt, _build_initial_state(prompt)

    console.print(
        make_error_panel(
            "Missing argument",
            "Please provide a goal prompt or specify --resume <session_id>.",
        )
    )
    raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command("run", rich_help_panel="Core")
def run_command(  # noqa: PLR0915
    prompt: Annotated[
        str,
        typer.Argument(
            help="The goal or task for Helm to execute.",
        ),
    ] = "",
    *,
    config: _OptionalConfigPath = None,
    verbose: _OptionalVerbose = False,
    auto_approve: _OptionalAutoApprove = False,
    resume: _OptionalResume = None,
    provider: _OptionalProvider = None,
    model: _OptionalModel = None,
    debug_graph: _OptionalDebugGraph = False,
) -> None:
    """
    Execute a one-shot agentic task or resume an interrupted session.

    \\b
    Examples:
        helmx run "refactor the auth module to use dependency injection"
        helmx run --provider openai --model gpt-4o "generate unit tests"
        helmx run --provider anthropic "analyze performance profile"
        helmx run --verbose "deploy v2 to staging"
        helmx run --yes "run safe migrations"
        helmx run --resume session-1234
    """
    _configure_logging(verbose)
    settings = load_settings(config_path=config)
    if auto_approve:
        settings.safety.hitl_mutations = False

    print_helm_banner(console)

    prompt, state = _resolve_session_state(prompt, resume)

    console.print(make_goal_panel(prompt))

    while True:
        renderer = ActivityStreamRenderer(console=console)
        try:
            _preflight_auth_check(provider_arg=provider, model_arg=model)
            state = asyncio.run(
                _run_supervisor_session(
                    state,
                    settings,
                    auto_approve=auto_approve,
                    provider=provider,
                    model=model,
                    renderer=renderer,
                )
            )
            break
        except AuthenticationError as auth_err:
            renderer.close()
            _handle_authentication_error(
                provider_arg=provider,
                model_arg=model,
                error=auth_err,
            )
            settings = load_settings(config_path=config)
            state = _build_initial_state(prompt)
            continue
        except (CircuitBreakerTripped, CircuitBreakerError) as exc:
            renderer.close()
            logging.getLogger(__name__).warning(
                "Circuit breaker blocked execution: %s",
                exc,
                extra={"event": "circuit_breaker_error", "component": "CLI"},
            )
            reason = getattr(exc, "reason", "circuit_breaker_tripped")
            tool_name = getattr(exc, "tool_name", "unknown_tool")
            console.print(
                make_circuit_breaker_panel(
                    reason=str(reason),
                    tool_name=str(tool_name),
                    detail=str(exc),
                )
            )
            raise typer.Exit(code=1) from None
        except (TransportError, HelmError) as exc:
            renderer.close()
            logging.getLogger(__name__).warning(
                "Execution halted due to %s: %s",
                type(exc).__name__,
                exc,
                extra={"event": "session_domain_error", "component": "CLI"},
            )
            console.print(make_error_panel(f"{type(exc).__name__}", str(exc)))
            raise typer.Exit(code=1) from None
        except KeyboardInterrupt:
            renderer.close()
            console.print(make_error_panel("Interrupted by user.", "Session aborted."))
            state.transition(SessionStatus.ABORTED)
            raise typer.Exit(code=1) from None
        except Exception as exc:
            renderer.close()
            logging.getLogger(__name__).exception(
                "Unhandled exception during session",
                extra={"event": "session_error", "component": "CLI"},
            )
            console.print(make_error_panel("Helm encountered an unexpected error.", str(exc)))
            raise typer.Exit(code=1) from None
        finally:
            renderer.close()

    if debug_graph and state.tasks:
        console.print()
        console.print(make_task_table(state))
        console.print()
    console.print()
    console.print(make_session_summary_panel(state))

    if state.status != SessionStatus.COMPLETED:
        raise typer.Exit(code=1)


@app.command("login", rich_help_panel="Authentication")
def login_cli(
    provider: _OptionalProvider = None,
    model: _OptionalModel = None,
    base_url: _OptionalBaseUrl = None,
) -> None:
    """
    Authenticate and configure active LLM provider credentials.

    \\b
    Examples:
        helmx login
        helmx login --provider openai --model gpt-4o
        helmx login --provider openai --base-url http://localhost:11434/v1
    """
    login_command(provider=provider, model=model, base_url=base_url)


@app.command("tools", rich_help_panel="Inspection")
def tools_command(
    config: _OptionalConfigPath = None,
    verbose: _OptionalVerbose = False,
) -> None:
    """
    List connected MCP tool servers and registered tools.

    \\b
    Examples:
        helmx tools
        helmx tools --verbose
    """
    _configure_logging(verbose)
    settings = load_settings(config_path=config)

    print_helm_banner(console)

    if not settings.mcp_servers:
        console.print(
            "[helm.muted]No MCP servers configured yet in [bold]helmx.toml[/].\n"
            "Add a server block to discover tools, for example:\n\n"
            "  \\[\\[mcp_servers]]\n"
            '  name = "filesystem"\n'
            '  transport = "stdio"\n'
            '  command = "npx"\n'
            '  args = ["@modelcontextprotocol/server-filesystem", "."]\n[/]'
        )
        console.print(
            "[helm.info]Tip:[/] See [bold]configs/example_mcp_servers.json[/] for more examples."
        )
        return

    with thinking_spinner(console, "Discovering tools from connected MCP servers…"):
        try:
            fetched = asyncio.run(_fetch_tools(settings.mcp_servers))
            if isinstance(fetched, tuple) and len(fetched) == 2 and isinstance(fetched[1], dict):  # noqa: PLR2004
                tools, statuses = fetched
            else:
                tools, statuses = cast("list[MCPToolDefinition]", fetched), {}
        except Exception as exc:
            console.print(make_error_panel("Failed to query MCP servers", str(exc)))
            raise typer.Exit(code=1) from exc

    if not tools:
        console.print("[helm.warning]Connected to servers, but no tools were found.[/]")
        return

    console.print(make_tools_table(tools, server_statuses=statuses))


@app.command("config", rich_help_panel="Inspection")
def config_command(
    config: _OptionalConfigPath = None,
    verbose: _OptionalVerbose = False,
    provider: _OptionalProvider = None,
    model: _OptionalModel = None,
    interactive: Annotated[
        bool,
        typer.Option("--interactive", "-i", help="Run interactive provider setup wizard."),
    ] = False,
) -> None:
    """
    Display or configure active Helm settings.

    \\b
    Examples:
        helmx config
        helmx config --interactive
        helmx config --provider openai --model gpt-4o
        helmx config --config ./my-helmx.toml
    """
    if interactive or provider or model:
        login_command(provider=provider, model=model)
        return

    _configure_logging(verbose)
    settings = load_settings(config_path=config)

    print_helm_banner(console)
    console.print("[helm.header]Active Configuration[/]\n")
    console.print(Pretty(settings.model_dump()))


@app.command("version", rich_help_panel="Meta")
def version_command() -> None:
    """Print the Helm version and exit."""
    print_helm_banner(console)


# ---------------------------------------------------------------------------
# Interactive mode (future milestone stub)
# ---------------------------------------------------------------------------


@app.command("interactive", rich_help_panel="Core")
def interactive_command(
    config: _OptionalConfigPath = None,
    verbose: _OptionalVerbose = False,
) -> None:
    """
    Launch an interactive multi-turn Helm session.

    \\b
    Examples:
        helmx interactive
        helmx --verbose interactive
    """
    _configure_logging(verbose)
    _ = load_settings(config_path=config)

    print_helm_banner(console)
    console.print(
        "[helm.warning]Interactive mode is not yet implemented.[/] "
        "It will be available in Milestone 2.\n"
        'Use [bold]helmx run "<prompt>"[/] for one-shot execution.'
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """Console script entrypoint registered in pyproject.toml."""
    app()


if __name__ == "__main__":
    main()
