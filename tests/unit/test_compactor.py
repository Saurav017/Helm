"""
tests/unit/test_compactor.py — Unit tests for ContextCompactor.
"""

from __future__ import annotations

import pytest

from helm.core.task_graph import StepStatus, TaskStep
from helm.memory.compactor import ContextCompactor


class TestContextCompactor:
    def test_token_estimation(self):
        compactor = ContextCompactor()
        assert compactor.estimate_tokens("") == 0
        assert compactor.estimate_tokens("abcd") == 1
        assert compactor.estimate_tokens("a" * 400) == 100

    def test_needs_compaction(self):
        compactor = ContextCompactor(token_threshold=100)
        # 200 chars ~ 50 tokens < 100
        assert not compactor.needs_compaction("a" * 200)
        # 500 chars ~ 125 tokens > 100
        assert compactor.needs_compaction("a" * 500)

    def test_extract_critical_metadata(self):
        text = (
            '{"name": "ProjectHelm", "id": "step-101", "path": "helm/memory/compactor.py", '
            '"error": "500", "code": "DUPLICATE_CALL"} '
            "Task step-202 failed with code 404 on /var/data/test.json"
        )
        meta = ContextCompactor.extract_critical_metadata(text)

        assert "ProjectHelm" in meta["entity_names"]
        assert "step-101" in meta["ids"]
        assert "step-202" in meta["ids"]
        assert "helm/memory/compactor.py" in meta["paths"]
        assert "/var/data/test.json" in meta["paths"]
        assert "500" in meta["error_codes"]
        assert "404" in meta["error_codes"]
        assert "DUPLICATE_CALL" in meta["error_codes"]

    @pytest.mark.asyncio
    async def test_compact_steps_below_threshold(self):
        compactor = ContextCompactor(token_threshold=500)
        steps = [
            TaskStep(step_id="step-1", title="Step 1", output="Short output 1"),
            TaskStep(step_id="step-2", title="Step 2", output="Short output 2"),
        ]

        summary = await compactor.compact_steps(steps)
        assert "Step step-1: Short output 1" in summary
        assert "Step step-2: Short output 2" in summary
        # Should not be in condensed markdown form
        assert "### Compacted Context Summary" not in summary

    @pytest.mark.asyncio
    async def test_compact_steps_above_threshold_preserves_metadata(self):
        compactor = ContextCompactor(token_threshold=50)  # low threshold to trigger compaction

        verbose_payload = (
            '{"entityName": "AlphaService", "id": "task-uuid-1", "path": "/src/service.py", '
            '"status": 500, "details": "' + ("verbose log data " * 50) + '"}'
        )

        steps = [
            TaskStep(
                step_id="step-1",
                title="Query server",
                status=StepStatus.COMPLETED,
                output=verbose_payload,
            ),
        ]

        summary = await compactor.compact_steps(steps)

        assert "### Compacted Context Summary" in summary
        assert "AlphaService" in summary
        assert "task-uuid-1" in summary or "step-1" in summary
        assert "/src/service.py" in summary
        assert "500" in summary
        assert "[condensed]" in summary

    @pytest.mark.asyncio
    async def test_compact_context_text(self):
        compactor = ContextCompactor(token_threshold=20)
        raw_text = (
            "Encountered error DUPLICATE_CALL on entity OmegaNode at path /var/run/app.json " * 5
        )

        compacted = await compactor.compact_context(raw_text)
        assert "### Compacted Context" in compacted
        assert "OmegaNode" in compacted or "DUPLICATE_CALL" in compacted
        assert "[condensed]" in compacted
