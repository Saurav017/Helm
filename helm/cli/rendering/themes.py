"""
helm/cli/rendering/themes.py — Centralised Rich colour/style definitions.

All Rich markup styles are defined here so that the visual language of the
terminal UI is consistent and changeable from one place.
"""

from __future__ import annotations

from rich.theme import Theme

HELM_THEME = Theme(
    {
        # Structural
        "helm.border": "bright_blue",
        "helm.header": "bold bright_blue",
        "helm.subheader": "dim blue",
        # Semantic
        "helm.success": "bold green",
        "helm.warning": "bold yellow",
        "helm.error": "bold red",
        "helm.info": "cyan",
        "helm.muted": "dim",
        # Task statuses
        "helm.status.pending": "dim white",
        "helm.status.in_progress": "bold cyan",
        "helm.status.completed": "bold green",
        "helm.status.failed": "bold red",
        "helm.status.blocked": "bold yellow",
        # Safety gate
        "helm.gate.prompt": "bold magenta",
        "helm.gate.deny": "bold red",
        "helm.gate.allow": "bold green",
    }
)
