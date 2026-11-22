"""
helm/core/planner.py — Dynamic DAG Re-planning and remediation engine.

Generates remediation task steps based on reflection critiques and mutates
the active TaskGraph at runtime (INSTRUCTIONS.md §3, Milestone 6).
"""

from __future__ import annotations

import logging
from typing import Any

from helm.core.reflection import ReflectionResult
from helm.core.task_graph import TaskGraph, TaskStep

logger = logging.getLogger(__name__)


class RePlanner:
    """
    Dynamic DAG Re-planner that mutates active task graphs when steps fail
    or require remediation.
    """

    def __init__(self) -> None:
        pass

    def apply_recovery(
        self,
        graph: TaskGraph,
        failing_step: TaskStep,
        reflection: ReflectionResult,
        context: dict[str, Any] | None = None,
        target_step_id: str | None = None,
    ) -> list[TaskStep]:
        """
        Apply recovery actions from reflection to the active TaskGraph.
        Returns the list of newly injected steps.
        """
        _ = context
        if reflection.suggested_action == "INSERT_STEPS" and reflection.suggested_steps:
            new_steps = reflection.suggested_steps
            target = target_step_id or failing_step.step_id

            logger.info(
                "Dynamic replanning: inserting %d recovery step(s) before %s",
                len(new_steps),
                target,
                extra={
                    "event": "dag_replan_insert",
                    "target_step": target,
                    "new_steps": [s.step_id for s in new_steps],
                },
            )

            graph.insert_before(target, new_steps)
            if target_step_id and target_step_id != failing_step.step_id:
                graph.reroute_dependencies(failing_step.step_id, new_steps[-1].step_id)

            return new_steps

        return []
