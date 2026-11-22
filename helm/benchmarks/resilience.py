"""
helm/benchmarks/resilience.py — Telemetry and resilience benchmark harness.

Benchmarks autonomous error recovery, dynamic DAG mutation, and schema repair.
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field

from helm.core.state import SessionState, SessionStatus
from helm.core.supervisor import Supervisor
from helm.core.task_graph import TaskGraph
from helm.protocols.mcp_client import MCPClientManager


class BenchmarkReport(BaseModel):
    """Telemetry report recording self-reflection recovery metrics."""

    name: str
    passed: bool
    iterations_used: int
    replan_count: int = 0
    duration_ms: float = 0.0
    details: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": False}


class ResilienceBenchmark:
    """Harness executing recovery benchmarks and measuring resilience metrics."""

    def __init__(self, supervisor: Supervisor) -> None:
        self.supervisor = supervisor

    async def run_scenario(
        self,
        name: str,
        goal: str,
        initial_graph: TaskGraph,
        mcp_client: MCPClientManager,
        *,
        auto_approve: bool = True,
    ) -> tuple[SessionState, BenchmarkReport]:
        """Execute a resilience benchmark scenario and produce a telemetry report."""
        state = SessionState(user_goal=goal)
        initial_step_count = len(initial_graph.steps)
        t0 = time.monotonic()

        final_state = await self.supervisor.run(
            state=state,
            mcp_client=mcp_client,
            auto_approve=auto_approve,
            graph=initial_graph,
        )

        duration = (time.monotonic() - t0) * 1000.0
        final_graph_dict = final_state.metadata.get("task_graph", {})
        final_step_count = len(final_graph_dict.get("steps", {}))
        replan_count = max(0, final_step_count - initial_step_count)
        passed = final_state.status == SessionStatus.COMPLETED

        report = BenchmarkReport(
            name=name,
            passed=passed,
            iterations_used=final_state.circuit_breaker.total_steps,
            replan_count=replan_count,
            duration_ms=round(duration, 2),
            details={
                "session_id": final_state.session_id,
                "final_answer": final_state.final_answer or "",
                "total_steps": final_step_count,
            },
        )

        return final_state, report
