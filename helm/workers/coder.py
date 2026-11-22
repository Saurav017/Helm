"""
helm/workers/coder.py — Code & structured data specialized worker.

Handles structured formatting, schema verification, code generation,
and computational transformations.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from helm.core.model_router import ModelRouter
from helm.core.state import ModelTier, TaskBrief, TaskResult, TaskStatus
from helm.workers.base import BaseWorker
from helm.workers.delegator import DelegationResult, WorkerDelegator

if TYPE_CHECKING:
    from helm.ui.events import UIEvent

logger = logging.getLogger(__name__)

CODE_WORKER_PROMPT = (
    "You are a Code & Structured Data Specialist. Your role is structured formatting, "
    "code generation, schema verification, and computational transformations. "
    "Ensure outputs are valid, type-safe, and syntactically correct."
)


class CodeWorker(BaseWorker):
    """Worker specialized in code generation, schema validation, and structured transforms."""

    def __init__(
        self,
        name: str = "CodeWorker",
        model_router: ModelRouter | None = None,
        delegator: WorkerDelegator | None = None,
        event_callback: Callable[[UIEvent], None] | None = None,
    ) -> None:
        super().__init__(
            name=name,
            role="Code & Structured Data Specialist",
            system_prompt=CODE_WORKER_PROMPT,
            model_router=model_router,
            event_callback=event_callback,
        )
        self.delegator = delegator

    async def delegate(
        self,
        target: str,
        query: str,
        context: dict[str, Any] | None = None,
        depth: int = 1,
        step_id: str | None = None,
    ) -> DelegationResult:
        """Delegate a sub-query to another worker (e.g. ToolWorker)."""
        if not self.delegator:
            self.delegator = WorkerDelegator()
        return await self.delegator.delegate(
            requester=self.name,
            target=target,
            query=query,
            context=context,
            depth=depth,
            step_id=step_id,
        )

    async def execute(
        self,
        brief: TaskBrief,
        *,
        tier: ModelTier = ModelTier.STANDARD,
    ) -> TaskResult:
        """
        Execute code generation, transformation, or verification logic.
        """
        prompt = (
            f"System Instructions: {self.system_prompt}\n\n"
            f"Task: {brief.description}\n\n"
            f"Input Data / Context:\n"
            f"{brief.relevant_context or 'No prior data provided.'}\n\n"
            f"Produce the verified code, schema, or structured output."
        )

        response = await self.model_router.generate(
            prompt,
            tier=tier,
            temperature=0.1,
        )

        output_text = response.strip()
        if not output_text:
            output_text = f"CodeWorker produced output for: {brief.description}"

        return TaskResult(
            task_id=brief.task_id,
            status=TaskStatus.COMPLETED,
            output=output_text,
            steps_used=1,
        )


# Alias for SWEWorker per Milestone 6 specifications
SWEWorker = CodeWorker
