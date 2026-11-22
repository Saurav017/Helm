"""
tests/unit/test_task_graph.py — Unit tests for TaskGraph and DAG engine.
"""

from __future__ import annotations

import pytest

from helm.core.task_graph import (
    CyclicDependencyError,
    StepStatus,
    TaskGraph,
    TaskStep,
)


class TestTaskGraph:
    def test_task_step_aliases_and_defaults(self):
        graph = TaskGraph()
        s1 = TaskStep(id="step-1", title="Inspect directory")
        s2 = TaskStep(id="step-2", title="Analyze code", dependencies=["step-1"])
        assert s1.id == "step-1"
        assert s1.step_id == "step-1"
        assert s1.description == "Inspect directory"
        assert s2.dependencies == ["step-1"]
        assert s2.depends_on == ["step-1"]

        graph.add_step(s1)
        graph.add_step(s2)

        ready = graph.get_ready_steps()
        assert ready == [s1]

        graph.mark_completed("step-1", result="Found 3 files")
        assert s1.result == "Found 3 files"
        assert s1.output == "Found 3 files"
        ready_next = graph.get_ready_steps()
        assert ready_next == [s2]

    def test_linear_dag_execution(self):
        graph = TaskGraph(goal="Build and deploy")
        s1 = TaskStep(step_id="step-1", description="Build artifact")
        s2 = TaskStep(step_id="step-2", description="Deploy artifact", depends_on=["step-1"])
        graph.add_step(s1)
        graph.add_step(s2)

        # Initially, only step-1 has satisfied dependencies
        ready = graph.get_ready_steps()
        assert len(ready) == 1
        assert ready[0].step_id == "step-1"

        # Complete step-1
        graph.mark_running("step-1")
        assert graph.get_step("step-1").status == StepStatus.RUNNING
        graph.mark_completed("step-1", "artifact.tar.gz")

        # Now step-2 is ready
        ready2 = graph.get_ready_steps()
        assert len(ready2) == 1
        assert ready2[0].step_id == "step-2"

        # Complete step-2
        graph.mark_completed("step-2", "deployed to v1.0")

        assert graph.is_finished()
        assert graph.is_successful()

    def test_branched_dag_dependencies(self):
        graph = TaskGraph(goal="Parallel pipeline")
        s1 = TaskStep(step_id="fetch-a", description="Fetch data A")
        s2 = TaskStep(step_id="fetch-b", description="Fetch data B")
        s3 = TaskStep(
            step_id="merge",
            description="Merge A and B",
            depends_on=["fetch-a", "fetch-b"],
        )

        graph.add_step(s1)
        graph.add_step(s2)
        graph.add_step(s3)

        # Both s1 and s2 should be ready concurrently
        ready = graph.get_ready_steps()
        assert {s.step_id for s in ready} == {"fetch-a", "fetch-b"}

        # Complete only s1 -> s3 should still NOT be ready
        graph.mark_completed("fetch-a", "data A")
        ready_after_a = graph.get_ready_steps()
        assert {s.step_id for s in ready_after_a} == {"fetch-b"}

        # Complete s2 -> now s3 is ready
        graph.mark_completed("fetch-b", "data B")
        ready_after_b = graph.get_ready_steps()
        assert {s.step_id for s in ready_after_b} == {"merge"}

    def test_cycle_detection_raises_error(self):
        graph = TaskGraph(goal="Cycle test")
        s1 = TaskStep(step_id="a", description="Step A", depends_on=["b"])
        s2 = TaskStep(step_id="b", description="Step B", depends_on=["a"])

        graph.steps["a"] = s1
        graph.steps["b"] = s2

        with pytest.raises(CyclicDependencyError, match="Cycle detected"):
            graph.validate_dag()

    def test_self_dependency_rejected(self):
        with pytest.raises(ValueError, match="cannot depend on itself"):
            TaskStep(step_id="self-dep", description="oops", depends_on=["self-dep"])

    def test_missing_dependency_raises_error(self):
        graph = TaskGraph(goal="Missing dep test")
        s1 = TaskStep(step_id="a", description="Step A", depends_on=["non_existent"])
        with pytest.raises(CyclicDependencyError, match="non-existent"):
            graph.add_step(s1)

    def test_topological_ordering(self):
        graph = TaskGraph(goal="Order test")
        s1 = TaskStep(step_id="1", description="First")
        s2 = TaskStep(step_id="2", description="Second", depends_on=["1"])
        s3 = TaskStep(step_id="3", description="Third", depends_on=["2"])

        graph.add_step(s1)
        graph.add_step(s2)
        graph.add_step(s3)

        order = graph.get_topological_order()
        assert [s.step_id for s in order] == ["1", "2", "3"]

    def test_mark_failed_marks_unfinished_and_unsuccessful(self):
        graph = TaskGraph(goal="Failure test")
        s1 = TaskStep(step_id="1", description="Fail step")
        graph.add_step(s1)
        graph.mark_failed("1", "network timeout")

        assert graph.get_step("1").status == StepStatus.FAILED
        assert graph.is_finished()
        assert not graph.is_successful()
