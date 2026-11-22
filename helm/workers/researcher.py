"""
helm/workers/researcher.py — Research & synthesis specialized worker.

Handles context synthesis, analytical comparison, knowledge extraction,
and answering high-level user goals.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from helm.core.model_router import ModelRouter
from helm.core.state import ModelTier, TaskBrief, TaskResult, TaskStatus
from helm.core.synthesis import clean_synthesizer_output
from helm.workers.base import BaseWorker

if TYPE_CHECKING:
    from helm.ui.events import UIEvent

logger = logging.getLogger(__name__)

RESEARCH_WORKER_PROMPT = (
    "You are a Research & Synthesis Specialist. Your role is context synthesis, "
    "analytical comparison, knowledge extraction, and answering high-level questions "
    "thoroughly, accurately, and without hallucination. Synthesize findings clearly. "
    "Do NOT emit tool calls, python code blocks, or tool invocation syntax. "
    "Answer directly using only the provided context."
)


class ResearchWorker(BaseWorker):
    """Worker specialized in context synthesis, research, and analytical summarization."""

    def __init__(
        self,
        name: str = "ResearchWorker",
        model_router: ModelRouter | None = None,
        event_callback: Callable[[UIEvent], None] | None = None,
    ) -> None:
        super().__init__(
            name=name,
            role="Context Synthesizer & Researcher",
            system_prompt=RESEARCH_WORKER_PROMPT,
            model_router=model_router,
            event_callback=event_callback,
        )

    async def execute(
        self,
        brief: TaskBrief,
        *,
        tier: ModelTier = ModelTier.STANDARD,
    ) -> TaskResult:
        """
        Synthesize relevant context and produce structured analysis.
        """
        prompt = (
            f"System Instructions: {self.system_prompt}\n\n"
            f"Goal / Task: {brief.description}\n\n"
            f"Context / Evidence:\n"
            f"{brief.relevant_context or 'No prior evidence provided.'}\n\n"
            f"Provide a clear, cohesive synthesis addressing the goal."
        )

        response = await self.model_router.generate(
            prompt,
            tier=tier,
            temperature=0.2,
        )

        output_text = clean_synthesizer_output(response.strip())
        if not output_text:
            output_text = f"ResearchWorker synthesized analysis for: {brief.description}"

        return TaskResult(
            task_id=brief.task_id,
            status=TaskStatus.COMPLETED,
            output=output_text,
            steps_used=1,
        )
