"""
helm/ui/renderer.py — Dynamic Claude Code-style activity feed terminal renderer.

Renders real-time thinking states, streaming tool intent bullets, collapsed result
checkpoints, and formatted markdown final synthesis according to INSTRUCTIONS.md §1.2.
"""

from __future__ import annotations

import json
import logging
import sys
import types
from typing import Any, Self

from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.status import Status

from helm.core.synthesis import clean_synthesizer_output
from helm.ui.events import (
    SessionDoneEvent,
    StepProgressEvent,
    ThinkingEvent,
    ToolCallCompleteEvent,
    ToolCallStartEvent,
    UIEvent,
)

logger = logging.getLogger(__name__)

_MAX_ARGS_PREVIEW_LEN = 120


class ActivityStreamRenderer:
    """
    Claude Code-style dynamic activity feed renderer.

    Presents transient thinking spinners, action bullets for tool calls,
    compact single-line completion checkpoints, and formatted markdown final answers.
    Automatically detects non-TTY environments (CI/redirects) to output clean plain text.
    """

    def __init__(
        self,
        console: Console | None = None,
        *,
        is_tty: bool | None = None,
    ) -> None:
        self.console = console or Console()
        if is_tty is not None:
            self.is_tty = is_tty
        else:
            self.is_tty = sys.stdout.isatty() and getattr(self.console, "is_terminal", True)

        self._active_status: Status | None = None
        self._events: list[UIEvent] = []

    @property
    def events(self) -> list[UIEvent]:
        """Return history of recorded UI events."""
        return list(self._events)

    @property
    def is_thinking(self) -> bool:
        """Return True if a transient thinking status spinner is currently active."""
        return self._active_status is not None

    def handle_event(self, event: UIEvent) -> None:
        """Process and render incoming UI presentation event."""
        self._events.append(event)
        match event:
            case ThinkingEvent():
                self._handle_thinking(event)
            case ToolCallStartEvent():
                self._handle_tool_start(event)
            case ToolCallCompleteEvent():
                self._handle_tool_complete(event)
            case StepProgressEvent():
                self._handle_step_progress(event)
            case SessionDoneEvent():
                self._handle_session_done(event)
            case _:
                logger.debug("Unhandled UI event type: %s", type(event).__name__)

    # ---------------------------------------------------------------------------
    # Event Handlers
    # ---------------------------------------------------------------------------

    def _handle_thinking(self, event: ThinkingEvent) -> None:
        """Display a transient spinner with dimmed reasoning text."""
        clean_msg = escape(event.message.strip())
        if self.is_tty:
            display_text = f"[dim]✻ Thinking... {clean_msg}[/dim]"
            if self._active_status is None:
                self._active_status = self.console.status(display_text, spinner="dots")
                self._active_status.start()
            else:
                self._active_status.update(display_text)
        else:
            self._stop_status()
            self.console.print(f"✻ Thinking... {event.message.strip()}", highlight=False)

    def _handle_tool_start(self, event: ToolCallStartEvent) -> None:
        """Print an action intent bullet before executing an MCP tool."""
        self._stop_status()

        args_str = self._format_arguments(event.arguments)
        server_str = f" ({event.server_name})" if event.server_name else ""

        if self.is_tty:
            self.console.print(
                f"[bold cyan]●[/bold cyan] Calling [bold]{escape(event.tool_name)}[/bold]"
                f"[dim]{escape(server_str)}[/dim] with [dim]{escape(args_str)}[/dim]"
            )
        else:
            self.console.print(
                f"● Calling {event.tool_name}{server_str} with {args_str}",
                highlight=False,
            )

    def _handle_tool_complete(self, event: ToolCallCompleteEvent) -> None:
        """Render a compact single-line execution checkpoint."""
        self._stop_status()
        summary_str = event.summary or "Done"

        if not event.is_error:
            if self.is_tty:
                self.console.print(
                    f"[bold green]✓[/bold green] [bold]{escape(event.tool_name)}[/bold] "
                    f"executed successfully [dim]({escape(summary_str)})[/dim]"
                )
            else:
                self.console.print(
                    f"✓ {event.tool_name} executed successfully ({summary_str})",
                    highlight=False,
                )
        elif self.is_tty:
            self.console.print(
                f"[bold red]✗[/bold red] [bold]{escape(event.tool_name)}[/bold] "
                f"failed: [red]{escape(summary_str)}[/red]"
            )
        else:
            self.console.print(
                f"✗ {event.tool_name} failed: {summary_str}",
                highlight=False,
            )

    def _handle_step_progress(self, event: StepProgressEvent) -> None:
        """Handle step lifecycle transitions."""
        if event.status == "FAILED":
            self._stop_status()
            if self.is_tty:
                self.console.print(f"[bold red]✗ Step '{escape(event.step_id)}' failed[/bold red]")
            else:
                self.console.print(f"✗ Step '{event.step_id}' failed", highlight=False)

    def _handle_session_done(self, event: SessionDoneEvent) -> None:
        """Render final synthesizer output in Markdown and footer metrics."""
        self._stop_status()
        self.console.print()

        clean_text = clean_synthesizer_output(event.summary)
        if clean_text:
            if self.is_tty:
                self.console.print(Markdown(clean_text))
            else:
                self.console.print(clean_text, highlight=False)

        if event.duration_sec > 0 or event.cost > 0:
            self.console.print()
            footer = f"[dim]Completed in {event.duration_sec:.1f}s | Cost: ${event.cost:.4f}[/dim]"
            if self.is_tty:
                self.console.print(footer)
            else:
                self.console.print(
                    f"Completed in {event.duration_sec:.1f}s | Cost: ${event.cost:.4f}",
                    highlight=False,
                )

    # ---------------------------------------------------------------------------
    # Helpers & Lifecycle
    # ---------------------------------------------------------------------------

    def _stop_status(self) -> None:
        """Stop and clear the transient spinner if active."""
        if self._active_status is not None:
            try:
                self._active_status.stop()
            except Exception:
                logger.debug("Failed to stop transient status", exc_info=True)
            self._active_status = None

    @staticmethod
    def _format_arguments(args: dict[str, Any]) -> str:
        """Format argument dictionary into a clean compact single-line string."""
        if not args:
            return "{}"
        try:
            compact = json.dumps(args, ensure_ascii=False, separators=(", ", ": "))
        except Exception:
            raw = str(args)
            if len(raw) > _MAX_ARGS_PREVIEW_LEN:
                return f"{raw[: _MAX_ARGS_PREVIEW_LEN - 3]}..."
            return raw
        else:
            if len(compact) > _MAX_ARGS_PREVIEW_LEN:
                return f"{compact[: _MAX_ARGS_PREVIEW_LEN - 3]}..."
            return compact

    def close(self) -> None:
        """Clean up active statuses and flush console."""
        self._stop_status()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        self.close()
