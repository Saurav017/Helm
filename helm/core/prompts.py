"""
helm/core/prompts.py — System prompts, tool instructions, and parameter guidelines.
"""

from __future__ import annotations

# Instruction reinforcing precise parameter generation for local and standard LLM providers
PARAMETER_CORRECTION_INSTRUCTIONS = (
    "When reading files, always provide exact filenames with extensions (e.g., 'README.md', "
    "'pyproject.toml'). Always use '.' to list the root directory."
)

TOOL_USAGE_SYSTEM_INSTRUCTION = (
    "Do NOT write Python code to execute these tools. "
    "Always trigger them via native tool/function calls.\n"
    f"{PARAMETER_CORRECTION_INSTRUCTIONS}"
)
