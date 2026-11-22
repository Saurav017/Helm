"""
helm/protocols/safety_gate.py — Terminal Human-in-the-Loop (HITL) safety gate.

RULE (INSTRUCTIONS.md §3.2):
  Every mutating action — file writes, file deletions, bash/shell execution,
  git commits, git pushes, API calls with side effects — MUST be gated through
  the interactive terminal confirmation mechanism.
  Bypass is forbidden in code — the only path is explicit user confirmation
  or explicitly configured auto-approval flags.
"""

from __future__ import annotations

import io
import logging
import select
import sys
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

logger = logging.getLogger(__name__)

# Max chars for argument value display in confirmation panel
_ARG_VALUE_MAX_CHARS = 80
_ARG_VALUE_TRUNC_TO = 77
_ARG_DISPLAY_LIMIT = 5


class ActionSafety(StrEnum):
    """Classification of tool action safety."""

    READ_ONLY = "READ_ONLY"
    MUTATING = "MUTATING"


class SafetyDecision(StrEnum):
    """Decision returned by the safety gate."""

    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    MODIFIED = "MODIFIED"


@dataclass(frozen=True, slots=True)
class SafetyGateResult:
    """Structured result returned by the SafetyGate."""

    decision: SafetyDecision
    reason: str = ""
    modified_arguments: dict[str, Any] | None = None

    @property
    def is_approved(self) -> bool:
        """Return True if the action is approved to proceed."""
        return self.decision in (SafetyDecision.APPROVED, SafetyDecision.MODIFIED)

    def __bool__(self) -> bool:
        """Allow boolean evaluation of gate result."""
        return self.is_approved


# ---------------------------------------------------------------------------
# Rule-based safety classification
# ---------------------------------------------------------------------------

# Verbs/tokens strongly associated with mutations
_MUTATING_VERBS: frozenset[str] = frozenset(
    {
        "write",
        "create",
        "delete",
        "remove",
        "unlink",
        "destroy",
        "rm",
        "edit",
        "patch",
        "update",
        "insert",
        "upsert",
        "save",
        "append",
        "mkdir",
        "move",
        "rename",
        "mv",
        "cp",
        "copy",
        "commit",
        "push",
        "merge",
        "rebase",
        "revert",
        "reset",
        "checkout_new",
        "branch_create",
        "post",
        "put",
        "drop",
        "truncate",
        "alter",
        "migrate",
        "kill",
        "terminate",
        "install",
        "uninstall",
    }
)

# Execution tools that execute arbitrary commands
_EXECUTION_TOOLS: frozenset[str] = frozenset(
    {
        "bash",
        "bash_exec",
        "shell",
        "shell_exec",
        "exec",
        "execute",
        "run",
        "terminal",
        "cmd",
        "system",
        "process",
    }
)

# Read-only verbs/prefixes
_READ_ONLY_PREFIXES: tuple[str, ...] = (
    "read",
    "get",
    "list",
    "search",
    "find",
    "cat",
    "grep",
    "status",
    "log",
    "diff",
    "show",
    "describe",
    "fetch",
    "inspect",
    "check",
    "head",
    "tail",
    "info",
    "query",
)

# Specific read-only tool names and suffixes
_READ_ONLY_SUFFIXES: tuple[str, ...] = (
    "_read",
    "_list",
    "_info",
    "_status",
    "_log",
    "_diff",
    "_search",
    "_cat",
    "_grep",
    "_find",
    "_get",
)

_GIT_READ_ONLY_COMMANDS: frozenset[str] = frozenset(
    {"status", "log", "diff", "show", "branch", "tag", "remote"}
)

# Commands within shell_exec that mutate state
_MUTATING_SHELL_PATTERNS: tuple[str, ...] = (
    "rm ",
    "rmdir ",
    "mv ",
    "cp ",
    "mkdir ",
    "chmod ",
    "chown ",
    "touch ",
    "dd ",
    "mkfs",
    "kill ",
    "pkill ",
    "git commit",
    "git push",
    "git checkout -b",
    "git reset",
    "git rebase",
    "git merge",
    "curl -X POST",
    "curl -X PUT",
    "curl -X DELETE",
    "wget ",
    ">",
    ">>",
    "| tee",
    "sed -i",
    "pip install",
    "npm install",
    "apt ",
    "brew ",
)


def classify_tool(tool_name: str, arguments: dict[str, Any] | None = None) -> ActionSafety:
    """
    Classify a tool call as READ_ONLY or MUTATING based on tool name and arguments.

    Parameters
    ----------
    tool_name:
        The identifier of the tool (e.g., 'file_write', 'bash_exec', 'fs_read').
    arguments:
        Optional payload passed to the tool.

    Returns
    -------
    ActionSafety:
        READ_ONLY or MUTATING.
    """
    normalized = tool_name.lower().replace("-", "_")

    # 1. Check for execution tools (bash, shell, etc.)
    if normalized in _EXECUTION_TOOLS or any(
        normalized.startswith(f"{t}_") for t in _EXECUTION_TOOLS
    ):
        # Inspect command arguments if present
        if arguments:
            cmd = arguments.get("command") or arguments.get("cmd") or arguments.get("script") or ""
            if isinstance(cmd, str):
                cmd_lower = cmd.lower()
                # Check for mutating shell patterns
                if any(pat in cmd_lower for pat in _MUTATING_SHELL_PATTERNS):
                    return ActionSafety.MUTATING

        # By default, arbitrary command execution is mutating (unbounded side effects)
        return ActionSafety.MUTATING

    # 2. Check git read-only commands
    if normalized.startswith("git_"):
        subcmd = normalized[len("git_") :]
        if subcmd in _GIT_READ_ONLY_COMMANDS:
            return ActionSafety.READ_ONLY

    # 3. Check explicitly recognized read-only prefixes
    if any(normalized.startswith(prefix) for prefix in _READ_ONLY_PREFIXES):
        tokens = set(normalized.split("_"))
        if not (tokens & _MUTATING_VERBS):
            return ActionSafety.READ_ONLY

    # 4. Check read-only suffixes (e.g. fs_read, directory_list)
    if any(normalized.endswith(suffix) for suffix in _READ_ONLY_SUFFIXES):
        tokens = set(normalized.split("_"))
        if not (tokens & _MUTATING_VERBS):
            return ActionSafety.READ_ONLY

    # 5. Check for mutating tokens or verbs
    tokens = set(normalized.split("_"))
    if tokens & _MUTATING_VERBS:
        return ActionSafety.MUTATING

    for verb in _MUTATING_VERBS:
        if verb in normalized:
            return ActionSafety.MUTATING

    # Conservative default: assume mutating for safety
    return ActionSafety.MUTATING


def is_mutating(tool_name: str, arguments: dict[str, Any] | None = None) -> bool:
    """Return True if the tool action is classified as MUTATING."""
    return classify_tool(tool_name, arguments) == ActionSafety.MUTATING


# ---------------------------------------------------------------------------
# Interactive Safety Gate
# ---------------------------------------------------------------------------


class SafetyGate:
    """
    Terminal Human-in-the-Loop (HITL) confirmation gate.

    Intercepts mutating tool calls, renders an approval card, and prompts
    the user for interactive approval [y/N].
    """

    def __init__(
        self,
        console: Console | None = None,
        *,
        auto_approve: bool = False,
        timeout_sec: int = 60,
    ) -> None:
        self.console = console or Console()
        self.auto_approve = auto_approve
        self.timeout_sec = timeout_sec

    def evaluate(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        auto_approve: bool | None = None,
        summary: str = "",
    ) -> SafetyGateResult:
        """
        Evaluate a tool invocation request.

        - If READ_ONLY: automatically approved.
        - If MUTATING: pauses for terminal approval or checks auto-approve flags.

        Parameters
        ----------
        tool_name:
            Target tool name.
        arguments:
            Arguments passed to the tool.
        auto_approve:
            Override flag for this invocation.
        summary:
            Human-readable description of what will be affected.

        Returns
        -------
        SafetyGateResult:
            Approval decision and rationale.
        """
        classification = classify_tool(tool_name, arguments)

        # Read-only actions pass through automatically
        if classification == ActionSafety.READ_ONLY:
            return SafetyGateResult(
                decision=SafetyDecision.APPROVED,
                reason="Tool action is classified as READ_ONLY",
            )

        # Mutating action: check auto-approval
        effective_auto_approve = auto_approve if auto_approve is not None else self.auto_approve
        if effective_auto_approve:
            logger.info(
                "Safety gate auto-approved mutating action",
                extra={
                    "event": "safety_gate_auto_approved",
                    "component": "SafetyGate",
                    "tool_name": tool_name,
                },
            )
            return SafetyGateResult(
                decision=SafetyDecision.APPROVED,
                reason="Auto-approved via configuration/CLI flag",
            )

        # Prompt user in terminal
        approved = self._prompt_terminal(tool_name, arguments, summary=summary)
        if approved:
            return SafetyGateResult(
                decision=SafetyDecision.APPROVED,
                reason="Explicitly approved by human operator in terminal",
            )
        return SafetyGateResult(
            decision=SafetyDecision.REJECTED,
            reason="ActionDeniedByHuman: Execution denied by user in safety gate",
        )

    def request_permission(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        *,
        auto_approve: bool | None = None,
        summary: str = "",
    ) -> bool:
        """
        Request permission to execute a tool.

        Parameters
        ----------
        tool_name:
            Target tool name.
        arguments:
            Arguments passed to the tool.
        auto_approve:
            Override flag for this invocation.
        summary:
            Human-readable description of what will be affected.

        Returns
        -------
        bool:
            True if approved (or read-only), False if denied or timed out.
        """
        result = self.evaluate(
            tool_name,
            arguments or {},
            auto_approve=auto_approve,
            summary=summary,
        )
        return result.is_approved

    def _prompt_terminal(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        summary: str,
    ) -> bool:
        """Render interactive confirmation card and read [y/N] input."""
        body = Text()
        body.append("⚠️  MUTATING ACTION REQUESTED\n\n", style="bold yellow")
        body.append("Tool:     ", style="dim")
        body.append(f"{tool_name}\n", style="bold white")

        if summary:
            body.append("Action:   ", style="dim")
            body.append(f"{summary}\n", style="white")

        # Arguments preview
        if arguments:
            body.append("\nArguments:\n", style="dim")
            for k, v in list(arguments.items())[:_ARG_DISPLAY_LIMIT]:
                v_str = str(v)
                if len(v_str) > _ARG_VALUE_MAX_CHARS:
                    v_str = v_str[:_ARG_VALUE_TRUNC_TO] + "…"
                body.append(f"  {k}: ", style="dim")
                body.append(f"{v_str}\n", style="helm.muted")

        body.append(f"\n[timeout: {self.timeout_sec}s — default is N]\n", style="dim")

        self.console.print(
            Panel(
                body,
                title="[helm.gate.prompt]🔐 Safety Gate[/]",
                border_style="magenta",
            )
        )
        self.console.print("[helm.gate.prompt]Approve execution? [y/N]:[/] ", end="")

        try:
            if sys.platform != "win32":
                try:
                    ready, _, _ = select.select([sys.stdin], [], [], float(self.timeout_sec))
                    if not ready:
                        self.console.print("\n[helm.gate.deny]✗ Timed out — action denied.[/]")
                        return False
                except (io.UnsupportedOperation, OSError):
                    # Stdin does not support select (e.g. redirected pseudofile)
                    pass
            answer = sys.stdin.readline().strip().lower()
        except (EOFError, KeyboardInterrupt, OSError):
            self.console.print(
                "\n[helm.gate.deny]✗ Interrupted or unreadable stdin — action denied.[/]"
            )
            return False

        allowed = answer in {"y", "yes"}
        if allowed:
            self.console.print("[helm.gate.allow]✓ Confirmed — proceeding.[/]")
        else:
            self.console.print("[helm.gate.deny]✗ Denied — action cancelled.[/]")

        return allowed
