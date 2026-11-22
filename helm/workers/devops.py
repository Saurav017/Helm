"""
helm/workers/devops.py — DevOps & infrastructure worker.
"""

from __future__ import annotations

from helm.core.state import TaskBrief, TaskResult, TaskStatus
from helm.workers.base import BaseWorker


class DevOpsWorker(BaseWorker):
    """Worker specialized in infrastructure, deployment, and environment commands."""

    def __init__(self, name: str = "DevOpsWorker") -> None:
        super().__init__(
            name=name,
            role="DevOps & Infrastructure Engineer",
            system_prompt="You are a DevOps Specialist handling infrastructure and commands.",
        )

    async def execute(self, brief: TaskBrief) -> TaskResult:
        return TaskResult(
            task_id=brief.task_id,
            status=TaskStatus.COMPLETED,
            output=f"DevOps execution completed for: {brief.description}",
            steps_used=1,
        )
