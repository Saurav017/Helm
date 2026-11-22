"""
helm/core/task_graph.py — Directed Acyclic Graph (DAG) Task Engine.

Models structured multi-step execution plans with dependency resolution,
cycle validation, and step progression.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from helm.exceptions import HelmError


class CyclicDependencyError(HelmError):
    """Raised when the task graph contains a cycle or invalid dependency."""


class StepStatus(StrEnum):
    """Execution status for an individual step in the DAG."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TaskStep(BaseModel):
    """A discrete unit of work with defined dependencies in the execution graph."""

    step_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    description: str = ""
    target_tool: str | None = None
    assigned_worker: str = "general"
    status: StepStatus = StepStatus.PENDING
    depends_on: list[str] = Field(default_factory=list)
    output: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None

    model_config = {"frozen": False, "populate_by_name": True}

    @property
    def worker(self) -> str:
        return self.assigned_worker

    @worker.setter
    def worker(self, value: str) -> None:
        self.assigned_worker = value

    @model_validator(mode="before")
    @classmethod
    def _normalize_step_inputs(cls, data: Any) -> Any:  # noqa: ANN401
        if isinstance(data, dict):
            # Support 'id' as alias for 'step_id'
            if "id" in data and "step_id" not in data:
                data["step_id"] = data["id"]
            # Support 'dependencies' as alias for 'depends_on'
            if "dependencies" in data and "depends_on" not in data:
                data["depends_on"] = data["dependencies"]
            # Default description to title or vice versa if one is missing
            if not data.get("description") and data.get("title"):
                data["description"] = data["title"]
            elif not data.get("title") and data.get("description"):
                data["title"] = data["description"]
        return data

    @model_validator(mode="after")
    def validate_no_self_dependency(self) -> TaskStep:
        if self.step_id in self.depends_on:
            msg = f"TaskStep {self.step_id!r} cannot depend on itself"
            raise ValueError(msg)
        return self

    @property
    def id(self) -> str:
        """Alias for step_id."""
        return self.step_id

    @property
    def dependencies(self) -> list[str]:
        """Alias for depends_on."""
        return self.depends_on

    @property
    def result(self) -> str | None:
        """Alias for output."""
        return self.output


class TaskGraph(BaseModel):
    """
    Directed Acyclic Graph (DAG) managing a multi-step execution plan.
    """

    graph_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    goal: str = ""
    steps: dict[str, TaskStep] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    model_config = {"frozen": False}

    def add_step(self, step: TaskStep) -> None:
        """Add a step to the graph and validate acyclic invariants."""
        if step.step_id in self.steps:
            msg = f"Step with ID {step.step_id!r} already exists in graph"
            raise ValueError(msg)

        self.steps[step.step_id] = step
        self.validate_dag()

    def insert_before(self, target_step_id: str, new_steps: list[TaskStep]) -> None:
        """
        Dynamically insert one or more steps immediately before target_step_id in the DAG.

        The first inserted step inherits the original dependencies of target_step_id.
        Subsequent inserted steps chain to the preceding one.
        Finally, target_step_id is updated to depend on the last inserted step.
        """
        if not new_steps:
            return
        target = self.steps.get(target_step_id)
        if not target:
            raise KeyError(f"Target step {target_step_id!r} not found in graph")

        original_deps = list(target.depends_on)
        for i, step in enumerate(new_steps):
            if i == 0:
                step.depends_on = list(original_deps)
            else:
                step.depends_on = [new_steps[i - 1].step_id]

            if step.step_id not in self.steps:
                self.steps[step.step_id] = step
            else:
                self.steps[step.step_id].depends_on = step.depends_on

        # Target now depends on the last inserted step
        target.depends_on = [new_steps[-1].step_id]
        if target.status in (StepStatus.FAILED, StepStatus.RUNNING):
            target.status = StepStatus.PENDING
            target.error = None

        self.validate_dag()

    def insert_after(self, target_step_id: str, new_steps: list[TaskStep]) -> None:
        """
        Dynamically insert one or more steps immediately after target_step_id in the DAG.

        The first inserted step depends on target_step_id.
        Subsequent inserted steps chain to the preceding one.
        Any existing steps that previously depended on target_step_id are rerouted
        to depend on the last inserted step.
        """
        if not new_steps:
            return
        target = self.steps.get(target_step_id)
        if not target:
            raise KeyError(f"Target step {target_step_id!r} not found in graph")

        for i, step in enumerate(new_steps):
            if i == 0:
                step.depends_on = [target_step_id]
            else:
                step.depends_on = [new_steps[i - 1].step_id]

            if step.step_id not in self.steps:
                self.steps[step.step_id] = step
            else:
                self.steps[step.step_id].depends_on = step.depends_on

        last_new_id = new_steps[-1].step_id
        for step in self.steps.values():
            if step.step_id in [s.step_id for s in new_steps]:
                continue
            if target_step_id in step.depends_on:
                step.depends_on = [
                    last_new_id if d == target_step_id else d for d in step.depends_on
                ]

        self.validate_dag()

    def reroute_dependencies(self, old_dep: str, new_dep: str) -> None:
        """
        Reroute all steps depending on old_dep to depend on new_dep instead.
        """
        for step in self.steps.values():
            if old_dep in step.depends_on:
                step.depends_on = [new_dep if d == old_dep else d for d in step.depends_on]
                seen: set[str] = set()
                deduped: list[str] = []
                for d in step.depends_on:
                    if d not in seen and d != step.step_id:
                        seen.add(d)
                        deduped.append(d)
                step.depends_on = deduped
        self.validate_dag()

    def get_step(self, step_id: str) -> TaskStep | None:
        """Lookup a step by ID."""
        return self.steps.get(step_id)

    def validate_dag(self) -> None:
        """
        Validate that all dependencies exist and that the graph has no cycles.

        Raises
        ------
        CyclicDependencyError:
            If a dependency cycle or missing node is detected.
        """
        # 1. Check all referenced dependencies exist
        for step_id, step in self.steps.items():
            for dep_id in step.depends_on:
                if dep_id not in self.steps:
                    msg = f"Step {step_id!r} depends on non-existent step {dep_id!r}"
                    raise CyclicDependencyError(msg)

        # 2. Cycle detection via DFS (3-color tracking: 0=unvisited, 1=visiting, 2=visited)
        visited: dict[str, int] = dict.fromkeys(self.steps, 0)

        def dfs(node: str) -> None:
            visited[node] = 1  # visiting
            for dep in self.steps[node].depends_on:
                if visited[dep] == 1:
                    msg = f"Cycle detected in task graph involving step {dep!r} and {node!r}"
                    raise CyclicDependencyError(msg)
                if visited[dep] == 0:
                    dfs(dep)
            visited[node] = 2  # visited

        for node in self.steps:
            if visited[node] == 0:
                dfs(node)

    def get_ready_steps(self) -> list[TaskStep]:
        """
        Return steps that are ready for execution:
        - Status is PENDING
        - All steps in depends_on are COMPLETED
        """
        ready: list[TaskStep] = []
        for step in self.steps.values():
            if step.status != StepStatus.PENDING:
                continue

            all_deps_completed = True
            for dep_id in step.depends_on:
                dep_step = self.steps.get(dep_id)
                if not dep_step or dep_step.status != StepStatus.COMPLETED:
                    all_deps_completed = False
                    break

            if all_deps_completed:
                ready.append(step)

        return ready

    def mark_running(self, step_id: str) -> None:
        """Transition step to RUNNING status."""
        step = self.steps.get(step_id)
        if not step:
            raise KeyError(f"Step {step_id!r} not found")
        step.status = StepStatus.RUNNING

    def mark_completed(
        self,
        step_id: str,
        output: str = "",
        *,
        result: str | None = None,
    ) -> None:
        """Transition step to COMPLETED status with result output."""
        step = self.steps.get(step_id)
        if not step:
            raise KeyError(f"Step {step_id!r} not found")
        step.status = StepStatus.COMPLETED
        step.output = result if result is not None else output
        step.completed_at = datetime.now(UTC)

    def mark_failed(
        self,
        step_id: str,
        error: str = "",
        *,
        reason: str | None = None,
    ) -> None:
        """Transition step to FAILED status with error details."""
        step = self.steps.get(step_id)
        if not step:
            raise KeyError(f"Step {step_id!r} not found")
        step.status = StepStatus.FAILED
        step.error = reason if reason is not None else error
        step.completed_at = datetime.now(UTC)

    def is_finished(self) -> bool:
        """Return True if all steps are in a terminal state (COMPLETED, FAILED, or SKIPPED)."""
        terminal = {StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.SKIPPED}
        return len(self.steps) > 0 and all(s.status in terminal for s in self.steps.values())

    def is_successful(self) -> bool:
        """Return True if all steps in the graph completed successfully."""
        return len(self.steps) > 0 and all(
            s.status == StepStatus.COMPLETED for s in self.steps.values()
        )

    def get_topological_order(self) -> list[TaskStep]:
        """Return steps sorted in valid dependency execution order."""
        self.validate_dag()
        in_degree: dict[str, int] = {sid: len(s.depends_on) for sid, s in self.steps.items()}
        # Adjacency: dep -> list of steps that depend on dep
        adj: dict[str, list[str]] = {sid: [] for sid in self.steps}
        for sid, step in self.steps.items():
            for dep in step.depends_on:
                adj[dep].append(sid)

        queue = [sid for sid, deg in in_degree.items() if deg == 0]
        order: list[TaskStep] = []

        while queue:
            curr = queue.pop(0)
            order.append(self.steps[curr])
            for dependent in adj[curr]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        return order

    def to_summary_dict(self) -> dict[str, Any]:
        """Return compact summary dict for state tracking."""
        return {
            "graph_id": self.graph_id,
            "goal": self.goal,
            "total_steps": len(self.steps),
            "completed": sum(1 for s in self.steps.values() if s.status == StepStatus.COMPLETED),
            "failed": sum(1 for s in self.steps.values() if s.status == StepStatus.FAILED),
            "pending": sum(1 for s in self.steps.values() if s.status == StepStatus.PENDING),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize TaskGraph to JSON-serializable dictionary."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskGraph:
        """Construct TaskGraph from dictionary."""
        return cls.model_validate(data)

    def save(self, path: Path | str) -> Path:
        """Save TaskGraph to a JSON file."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target_path

    @classmethod
    def load(cls, path: Path | str) -> TaskGraph:
        """Load TaskGraph from a JSON file."""
        source_path = Path(path)
        if not source_path.exists():
            raise FileNotFoundError(f"TaskGraph file not found: {source_path}")
        content = source_path.read_text(encoding="utf-8")
        data = json.loads(content)
        return cls.from_dict(data)
