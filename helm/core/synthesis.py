"""
helm/core/synthesis.py — Output synthesis cleaner and reasoning filter.

Ensures that final synthesized answers presented to the user contain only
the synthesized result, filtering out internal planning steps, reasoning chatter,
scratchpad tokens, and step checklist headers.
"""

from __future__ import annotations

import re


def clean_synthesizer_output(text: str) -> str:
    """
    Filter out internal planning steps, reasoning chatter, and scratchpad traces
    from synthesizer output, returning only the synthesized answer.
    """
    if not text:
        return ""

    raw = text.strip()

    # 1. Remove thinking / scratchpad blocks: <think>...</think> or ```thinking...```
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    cleaned = re.sub(r"```(?:thinking|thought|scratchpad)\b.*?```", "", cleaned, flags=re.DOTALL)

    # Remove tool invocation scratchpad blocks
    cleaned = re.sub(
        r"(?:\*\*|#+\s*)?Tool Invocation:?\s*(?:```(?:python|py)?\s*.*?```)?",
        "",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    cleaned = re.sub(
        r"```(?:python|py)?\s*(?:#[^\n]*\n\s*)*[a-zA-Z_][a-zA-Z0-9_]*\s*\(.*?\)\s*```",
        "",
        cleaned,
        flags=re.DOTALL,
    )

    # 2. If there's an explicit "Final Answer:" or "Answer:" or "Synthesis:" section, extract it
    answer_match = re.search(
        r"(?:^|\n)(?:#+\s*)?(?:Final Answer|Answer|Synthesis|Result|Summary):\s*\n*(.+)",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if answer_match:
        candidate = answer_match.group(1).strip()
        if candidate:
            cleaned = candidate

    # 3. Filter lines matching internal planning chatter
    step_planning_pattern = re.compile(
        r"^(?:[-*]\s*)?(?:Let's proceed with step \d+|Step \d+)[:.]?\s*(.*)$",
        re.IGNORECASE,
    )
    chatter_pattern = re.compile(
        r"^(?:"
        r"Thinking Process:?|"
        r"(?:[-*]\s*)?(?:First|Next|Then),\s*let's proceed.*|"
        r"\d+\.\s*(?:Understand the goal|Examine the context|"
        r"Formulate the answer|Analyze the evidence):?.*"
        r")$",
        re.IGNORECASE,
    )

    planning_verbs = (
        "inspect",
        "analyze",
        "check",
        "review",
        "formulate",
        "proceed",
        "let's",
        "we will",
        "i will",
        "provide",
        "first",
        "next",
        "then",
    )

    filtered_lines: list[str] = []
    lines = cleaned.splitlines()
    for line in lines:
        stripped = line.strip()
        if not stripped:
            filtered_lines.append("")
            continue
        if chatter_pattern.match(stripped):
            continue
        step_match = step_planning_pattern.match(stripped)
        if step_match:
            rest = step_match.group(1).strip()
            rest_lower = rest.lower()
            if not rest or any(rest_lower.startswith(v) for v in planning_verbs):
                continue
            filtered_lines.append(rest)
            continue

        filtered_lines.append(line)

    result = "\n".join(filtered_lines).strip()

    # 4. Filter inline sentence planning prefixes if at start of string
    inline_prefix_pattern = re.compile(
        r"^(?:(?:Let's proceed with step \d+|Step \d+)[:.]?\s*)+",
        re.IGNORECASE,
    )
    result = inline_prefix_pattern.sub("", result).strip()

    return result or raw
