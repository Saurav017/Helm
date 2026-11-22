"""
helm/cli/rendering/panels.py — Rich panel/table rendering helpers.

All terminal output construction (except raw print) goes through these
helpers so that formatting is consistent and testable independently of
the CLI command layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from helm.core.state import SessionState

if TYPE_CHECKING:
    from rich.console import Console

# Max description chars to show in task table before truncating.
_TASK_DESC_MAX_CHARS = 60


def make_goal_panel(user_goal: str) -> Panel:
    """Render the user's goal in a prominent panel."""
    return Panel(
        Text(user_goal, style="bold white"),
        title="[helm.header]⛵ Helm — Goal[/]",
        border_style="helm.border",
        padding=(1, 2),
    )


def make_session_summary_panel(state: SessionState) -> Panel:
    """Render a compact summary of the completed session."""
    table = Table.grid(padding=(0, 2))
    table.add_column(style="helm.muted", no_wrap=True)
    table.add_column()

    table.add_row("Session ID", state.session_id)
    table.add_row("Status", f"[helm.status.{state.status.value}]{state.status.value}[/]")
    table.add_row("Tasks", str(len(state.tasks)))
    table.add_row("Steps used", str(state.circuit_breaker.total_steps))
    table.add_row("Cost (USD)", f"${state.circuit_breaker.total_cost_usd:.4f}")

    if state.final_answer:
        table.add_row("Answer", state.final_answer)

    return Panel(
        table,
        title="[helm.header]Session Summary[/]",
        border_style="helm.border",
    )


def make_task_table(state: SessionState) -> Table:
    """Render a table of all tasks and their current statuses."""
    table = Table(
        "Task ID",
        "Description",
        "Status",
        "Worker",
        "Steps",
        title="[helm.header]Task Graph[/]",
        border_style="helm.border",
        show_lines=True,
    )
    for task in state.tasks:
        status_style = f"helm.status.{task.status.value}"
        desc = task.description
        truncated = desc[:_TASK_DESC_MAX_CHARS] + ("…" if len(desc) > _TASK_DESC_MAX_CHARS else "")
        table.add_row(
            task.task_id[:8] + "…",
            truncated,
            f"[{status_style}]{task.status.value}[/]",
            task.assigned_worker or "ExecutorWorker",
            str(task.step_count),
        )
    return table


def make_error_panel(message: str, detail: str = "") -> Panel:
    """Render a prominent error panel."""
    body = Text(message, style="helm.error")
    if detail:
        body.append(f"\n\n{detail}", style="helm.muted")
    return Panel(body, title="[helm.error]✗ Error[/]", border_style="red")


def make_circuit_breaker_panel(reason: str, tool_name: str, detail: str = "") -> Panel:
    """Render a circuit-breaker trip notification."""
    body = Text()
    body.append("🔴 Circuit breaker tripped\n\n", style="bold yellow")
    body.append("Reason:  ", style="helm.muted")
    body.append(f"{reason}\n", style="bold yellow")
    body.append("Tool:    ", style="helm.muted")
    body.append(f"{tool_name}\n", style="bold white")
    if detail:
        body.append(f"\n{detail}", style="helm.muted")
    return Panel(body, title="[bold yellow]⚡ Safety Guardrail[/]", border_style="yellow")


def make_tools_table(
    tools: list[Any],
    server_statuses: dict[str, dict[str, Any]] | None = None,
) -> Table:
    """Render a table of registered tools from connected MCP servers."""
    table = Table(
        title="[helm.header]Available MCP Tools[/]",
        border_style="helm.border",
        show_lines=True,
    )
    table.add_column("Tool Name", no_wrap=True)
    table.add_column("Description", ratio=1)
    table.add_column("Server", no_wrap=True)
    table.add_column("Transport", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Latency", no_wrap=True)
    table.add_column("Safety", no_wrap=True)
    statuses = server_statuses or {}
    for tool in tools:
        safety_str = getattr(tool, "safety", "")
        safety_val = safety_str.value if hasattr(safety_str, "value") else str(safety_str)
        style = "bold green" if safety_val == "READ_ONLY" else "bold yellow"

        server_name = getattr(tool, "server_name", "local")
        status_info = statuses.get(server_name, {})
        transport_val = getattr(tool, "transport", "") or status_info.get("transport", "stdio")
        raw_status = status_info.get("status", "connected")
        status_styled = (
            f"[bold green]{raw_status}[/]"
            if raw_status == "connected"
            else f"[bold red]{raw_status}[/]"
        )
        latency_val = f"{status_info['latency_ms']}ms" if "latency_ms" in status_info else "—"

        table.add_row(
            getattr(tool, "name", str(tool)),
            getattr(tool, "description", "") or "—",
            server_name,
            transport_val,
            status_styled,
            latency_val,
            f"[{style}]{safety_val}[/]",
        )
    return table


def print_helm_banner(console: Console) -> None:
    """Print the Helm ASCII banner."""
    banner = Text()
    banner.append("⛵ ", style="bright_blue")
    banner.append("HELM", style="bold bright_blue")
    banner.append("  v0.1.0", style="dim")
    banner.append("  —  terminal-first agentic orchestration", style="dim")
    console.print(banner)
    console.print()
