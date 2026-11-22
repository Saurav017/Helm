"""
helm/workers/base.py — Base worker agent abstraction.

Provides the common lifecycle interface (prepare, execute, handle_error)
for all specialized Helm worker sub-agents (Architecture Invariant 1).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

from helm.core.model_router import ModelRouter
from helm.core.state import TaskBrief, TaskResult, TaskStatus

if TYPE_CHECKING:
    from helm.ui.events import UIEvent

logger = logging.getLogger(__name__)


class BaseWorker(ABC):
    """
    Abstract base class for all specialized Helm workers.

    Lifecycle:
        1. prepare()      — Validate brief, set up worker-specific context.
        2. execute()      — Perform LLM reasoning, code transforms, or tool execution.
        3. handle_error() — Catch exceptions and produce structured TaskResult failure.
    """

    def __init__(
        self,
        name: str = "BaseWorker",
        role: str = "General Worker",
        system_prompt: str = "",
        model_router: ModelRouter | None = None,
        event_callback: Callable[[UIEvent], None] | None = None,
    ) -> None:
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.model_router = model_router or ModelRouter()
        self.event_callback = event_callback

    def emit_event(self, event: UIEvent) -> None:
        """Emit a UI event through the registered event callback."""
        if self.event_callback is not None:
            try:
                self.event_callback(event)
            except Exception as exc:
                logger.debug("Worker '%s' event callback error: %s", self.name, exc)

    async def prepare(self, brief: TaskBrief) -> None:
        """
        Lifecycle step 1: Validate context and prepare execution state.
        Default implementation logs task preparation.
        """
        logger.info(
            "Worker '%s' preparing task %s",
            self.name,
            brief.task_id,
            extra={"worker": self.name, "task_id": brief.task_id, "event": "worker_prepare"},
        )

    @abstractmethod
    async def execute(self, brief: TaskBrief) -> TaskResult:
        """
        Lifecycle step 2: Execute the core task logic.
        Must be implemented by concrete specialized workers.
        """
        raise NotImplementedError

    async def handle_error(self, brief: TaskBrief, error: Exception) -> TaskResult:
        """
        Lifecycle step 3 (fallback): Handle failures and return a typed TaskResult.
        """
        error_msg = str(error)
        logger.error(
            "Worker '%s' failed executing task %s: %s",
            self.name,
            brief.task_id,
            error_msg,
            extra={"worker": self.name, "task_id": brief.task_id, "event": "worker_error"},
        )
        return TaskResult(
            task_id=brief.task_id,
            status=TaskStatus.FAILED,
            output="",
            error=error_msg,
            steps_used=1,
        )

    async def run(self, brief: TaskBrief) -> TaskResult:
        """
        Execute the full worker lifecycle: prepare -> execute -> handle_error.
        """
        try:
            await self.prepare(brief)
            return await self.execute(brief)
        except Exception as exc:
            return await self.handle_error(brief, exc)
