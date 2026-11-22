"""
helm/memory/compactor.py — Context window compaction engine.

Summarizes prior execution steps and verbose outputs when cumulative context
crosses token/character limits, preserving critical IDs, paths, and entity names
(INSTRUCTIONS.md §3.3).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from helm.core.model_router import ModelRouter
from helm.core.state import ModelTier

logger = logging.getLogger(__name__)

DEFAULT_COMPACTION_TOKEN_THRESHOLD = 2000
_MAX_PREVIEW_LEN = 200
_CONDENSED_PREVIEW_LEN = 180


class ContextCompactor:
    """
    Context compactor that condenses verbose step outputs into compact Markdown
    while strictly preserving entities, identifiers, file paths, and error codes.
    """

    def __init__(
        self,
        token_threshold: int = DEFAULT_COMPACTION_TOKEN_THRESHOLD,
        model_router: ModelRouter | None = None,
    ) -> None:
        self.token_threshold = token_threshold
        self.model_router = model_router or ModelRouter()

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count (approx. 4 characters per token)."""
        if not text:
            return 0
        return max(1, len(text) // 4)

    def needs_compaction(self, items: list[Any] | str) -> bool:
        """Check if total tokens in text or step outputs exceed compaction threshold."""
        total_tokens = 0
        if isinstance(items, str):
            total_tokens = self.estimate_tokens(items)
        elif isinstance(items, list):
            for item in items:
                if isinstance(item, str):
                    total_tokens += self.estimate_tokens(item)
                elif hasattr(item, "output") and item.output:
                    total_tokens += self.estimate_tokens(str(item.output))
                elif hasattr(item, "result") and item.result:
                    if hasattr(item.result, "output") and item.result.output:
                        total_tokens += self.estimate_tokens(str(item.result.output))
                    elif isinstance(item.result, str):
                        total_tokens += self.estimate_tokens(item.result)
        return total_tokens >= self.token_threshold

    @staticmethod
    def extract_critical_metadata(text: str) -> dict[str, list[str]]:
        """
        Extract critical identifiers, entity names, paths, and error codes
        to ensure they are never lost during compaction.
        """
        metadata: dict[str, list[str]] = {
            "entity_names": [],
            "ids": [],
            "paths": [],
            "error_codes": [],
        }

        # 1. Parse JSON objects if present
        json_matches = re.findall(r"\{.*?\}|\[.*?\]", text, re.DOTALL)
        for jm in json_matches:
            try:
                parsed = json.loads(jm)
                if isinstance(parsed, dict):
                    for k, v in parsed.items():
                        kl = k.lower()
                        if kl in ("name", "entity", "entityname") and isinstance(v, str):
                            metadata["entity_names"].append(v)
                        elif kl in ("id", "task_id", "step_id", "call_id") and isinstance(v, str):
                            metadata["ids"].append(v)
                        elif kl in ("path", "file", "filepath") and isinstance(v, str):
                            metadata["paths"].append(v)
                        elif kl in ("error", "code", "status_code") and isinstance(v, (str, int)):
                            metadata["error_codes"].append(str(v))
                elif isinstance(parsed, list):
                    for el in parsed:
                        if isinstance(el, dict):
                            for k, v in el.items():
                                kl = k.lower()
                                if kl in ("name", "entity", "entityname") and isinstance(v, str):
                                    metadata["entity_names"].append(v)
                                elif kl in ("id", "task_id", "step_id") and isinstance(v, str):
                                    metadata["ids"].append(v)
            except json.JSONDecodeError:
                logger.debug("Skipping non-JSON match in metadata extraction")

        # 2. Regex fallbacks for IDs, entity names, paths, and error codes
        id_pattern = (
            r"\b(?:step-[0-9a-zA-Z_-]+|task-[0-9a-zA-Z_-]+|"
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4})\b"
        )
        metadata["ids"].extend(re.findall(id_pattern, text))

        path_pattern = r"(?:/[\w.-]+)+|\b[\w.-]+\.(?:py|json|toml|txt|md|yaml|yml)\b"
        metadata["paths"].extend(re.findall(path_pattern, text))

        error_pattern = (
            r"\b(?:40[0-9]|50[0-9]|DUPLICATE_CALL|STEP_CEILING|BUDGET_CAP|FAILED|BLOCKED)\b"
        )
        metadata["error_codes"].extend(re.findall(error_pattern, text))

        # Deduplicate
        for k, values in metadata.items():
            metadata[k] = list(dict.fromkeys(values))

        return metadata

    async def compact_steps(
        self,
        steps: list[Any],
        *,
        force: bool = False,
    ) -> str:
        """
        Condense a list of completed steps into a structured Markdown representation
        if token threshold is exceeded (or force is True).
        """
        if not steps:
            return ""

        if not force and not self.needs_compaction(steps):
            # No compaction needed; build standard summary
            entries: list[str] = []
            for s in steps:
                s_id = getattr(s, "step_id", getattr(s, "task_id", "step"))
                s_out = getattr(s, "output", "")
                if hasattr(s, "result") and s.result:
                    s_out = getattr(s.result, "output", str(s.result))
                entries.append(f"Step {s_id}: {s_out}")
            return "\n\n".join(entries)

        logger.info(
            "Context compaction triggered for %d steps (threshold: %d tokens)",
            len(steps),
            self.token_threshold,
            extra={"event": "context_compaction_triggered", "step_count": len(steps)},
        )

        compacted_lines: list[str] = ["### Compacted Context Summary"]
        for s in steps:
            s_id = getattr(s, "step_id", getattr(s, "task_id", "step"))
            s_title = getattr(s, "title", getattr(s, "description", ""))
            s_status = getattr(s, "status", "completed")
            s_out = getattr(s, "output", "")
            if hasattr(s, "result") and s.result:
                s_out = getattr(s.result, "output", str(s.result)) or s_out
            s_err = getattr(s, "error", "")

            meta = self.extract_critical_metadata(f"{s_title} {s_out} {s_err}")

            condensed_out = s_out
            if len(s_out) > _MAX_PREVIEW_LEN:
                condensed_out = s_out[:_CONDENSED_PREVIEW_LEN].strip() + " … [condensed]"

            line = f"- **Step {s_id}** (`{s_title}`): Status: `{s_status}`"
            compacted_lines.append(line)
            if meta["entity_names"]:
                compacted_lines.append(f"  - Entities: {', '.join(meta['entity_names'])}")
            if meta["ids"]:
                compacted_lines.append(f"  - IDs: {', '.join(meta['ids'])}")
            if meta["paths"]:
                compacted_lines.append(f"  - Paths: {', '.join(meta['paths'])}")
            if meta["error_codes"]:
                compacted_lines.append(f"  - Codes: {', '.join(meta['error_codes'])}")
            compacted_lines.append(f"  - Outcome: {condensed_out}")

        return "\n".join(compacted_lines)

    async def compact_context(self, context: str, *, force: bool = False) -> str:
        """
        Condense a raw context string if it exceeds the threshold.
        """
        if not force and not self.needs_compaction(context):
            return context

        meta = self.extract_critical_metadata(context)
        if self.model_router.api_key:
            prompt = (
                "Summarize the following execution context concisely in Markdown. "
                "You MUST preserve all entity names, IDs, paths, and status codes:\n\n"
                f"{context[:4000]}"
            )
            try:
                summary = await self.model_router.generate(prompt, tier=ModelTier.FAST)
                if summary.strip() and not summary.startswith("Simulated response for:"):
                    return summary.strip()
            except Exception as exc:
                logger.warning("Compactor LLM generation failed (%s); using fallback summary", exc)

        # Fallback deterministic markdown summary
        parts: list[str] = ["### Compacted Context"]
        if meta["entity_names"]:
            parts.append(f"- **Entities**: {', '.join(meta['entity_names'])}")
        if meta["ids"]:
            parts.append(f"- **IDs**: {', '.join(meta['ids'])}")
        if meta["paths"]:
            parts.append(f"- **Paths**: {', '.join(meta['paths'])}")
        if meta["error_codes"]:
            parts.append(f"- **Error Codes**: {', '.join(meta['error_codes'])}")
        parts.append(f"- **Details**: {context[:300].strip()} … [condensed]")
        return "\n".join(parts)
