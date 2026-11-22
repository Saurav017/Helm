"""
helm/cli/rendering/spinners.py — Rich Live / spinner context managers.

Provides thin wrappers around ``rich.status.Status`` so that the CLI layer
can show progress indicators without coupling to Rich internals.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

from rich.console import Console
from rich.status import Status


@contextmanager
def thinking_spinner(console: Console, message: str = "Thinking…") -> Generator[Status, None, None]:
    """Context manager that shows a spinner while Helm is processing."""
    with console.status(f"[helm.info]{message}[/]", spinner="dots") as status:
        yield status


@contextmanager
def tool_spinner(console: Console, tool_name: str) -> Generator[Status, None, None]:
    """Context manager that shows a spinner while a tool call executes."""
    with console.status(
        f"[helm.muted]Invoking tool:[/] [bold]{tool_name}[/]", spinner="dots2"
    ) as status:
        yield status
