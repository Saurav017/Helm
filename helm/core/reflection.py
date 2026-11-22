"""
helm/core/reflection.py — Autonomous Self-Reflection Engine.

Evaluates intermediate step outputs against expectations, classifies failures,
and generates structured critiques and repair actions (INSTRUCTIONS.md §3, Milestone 6).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from helm.core.task_graph import StepStatus, TaskStep


class ReflectionResult(BaseModel):
    """Structured assessment of a TaskStep execution outcome."""

    status: Literal["PASSED", "FAILED_RECOVERABLE", "FAILED_FATAL"]
    critique: str = ""
    suggested_action: Literal["RETRY", "INSERT_STEPS", "ABORT", "CONTINUE"]
    suggested_steps: list[TaskStep] | None = None
    critique_status: Literal["ACCEPTED", "REJECTED"] = "ACCEPTED"
    root_cause: str | None = None
    failure_classification: str | None = None

    model_config = {"frozen": False}

    @property
    def action(self) -> str:
        """Human-readable action string for critique reports."""
        if self.root_cause == "SyntaxError in Python AST":
            return "Regenerate with corrected AST syntax"
        if self.failure_classification == "MISSING_RESOURCE":
            return "List parent directory and locate valid resource path"
        return self.suggested_action

    def __getitem__(self, item: str) -> object:
        """Support table-based lookups for BDD and reporting."""
        key = item.strip().lower().replace(" ", "_")
        if key in ("critique_status", "status"):
            return self.critique_status
        if key == "root_cause":
            return self.root_cause or ""
        if key == "action":
            return self.action
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(item)


class ReflectionEngine:
    """
    Self-reflection engine that audits step results for semantic, syntactic,
    and runtime validity.
    """

    def __init__(self) -> None:
        pass

    def evaluate_step(
        self,
        step: TaskStep,
        result: object,
        context: dict[str, Any] | None = None,
    ) -> ReflectionResult:
        """
        Evaluate step output against expectations, tools, and code invariants.
        """
        ctx = context or {}
        raw_text = self._extract_result_text(result, step)

        # 1. Fatal circuit / security guardrail failure check
        fatal_res = self._check_fatal(step, raw_text)
        if fatal_res:
            return fatal_res

        # 2. Missing resource / FileNotFoundError check
        resource_res = self._check_missing_resource(step, raw_text, ctx)
        if resource_res:
            return resource_res

        # 3. Python AST / Syntax Error check
        syntax_res = self._check_syntax_error(step, raw_text)
        if syntax_res:
            return syntax_res

        # 4. Schema / Parameter validation check
        schema_res = self._check_schema_validation(step, raw_text)
        if schema_res:
            return schema_res

        # 5. Check if step is explicitly failed
        if step.status == StepStatus.FAILED:
            err = step.error or raw_text or "Step execution failed"
            return ReflectionResult(
                status="FAILED_RECOVERABLE",
                critique=f"Step failed with error: {err}",
                suggested_action="RETRY",
                critique_status="REJECTED",
                root_cause=err,
                failure_classification="EXECUTION_ERROR",
            )

        # 6. Default: Output verified and accepted
        return ReflectionResult(
            status="PASSED",
            critique="Step output meets execution quality criteria.",
            suggested_action="CONTINUE",
            critique_status="ACCEPTED",
            failure_classification=None,
        )

    def _extract_result_text(self, result: object, step: TaskStep) -> str:
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            if "error" in result:
                return str(result["error"])
            if "content" in result:
                return str(result["content"])
            return str(result)
        if step.error:
            return step.error
        if step.output:
            return step.output
        return str(result or "")

    def _check_fatal(self, step: TaskStep, text: str) -> ReflectionResult | None:
        _ = step
        lower = text.lower()
        if "action denied by operator" in lower or "denied in safety gate" in lower:
            return ReflectionResult(
                status="FAILED_FATAL",
                critique="Mutating action explicitly rejected by human operator in SafetyGate.",
                suggested_action="ABORT",
                critique_status="REJECTED",
                root_cause="SecurityPermissionDenied",
                failure_classification="PERMISSION_DENIED",
            )
        if "circuit breaker blocked" in lower or "step ceiling reached" in lower:
            return ReflectionResult(
                status="FAILED_FATAL",
                critique="Circuit breaker limit reached. Terminating session to avoid loop.",
                suggested_action="ABORT",
                critique_status="REJECTED",
                root_cause="CircuitBreakerTripped",
                failure_classification="CIRCUIT_BREAKER",
            )
        return None

    def _check_missing_resource(
        self,
        step: TaskStep,
        text: str,
        context: dict[str, Any],
    ) -> ReflectionResult | None:
        lower = text.lower()
        missing_indicators = [
            "filenotfounderror",
            "no such file or directory",
            "file not found",
            "does not exist",
            "path not found",
        ]
        if not any(ind in lower for ind in missing_indicators):
            return None

        # Extract file path mentioned in text, step description, or context
        path_str = self._extract_path(step.description or step.title, text, context)
        parent_dir = "config"
        if path_str:
            parent = Path(path_str).parent
            parent_dir = str(parent) if str(parent) != "." else "config"

        remediation_step = TaskStep(
            step_id=f"{step.step_id}a",
            title=f"List files in '{parent_dir}/' directory",
            description=f"List files in '{parent_dir}/' directory",
            target_tool="list_directory",
            assigned_worker="ToolWorker",
            status=StepStatus.PENDING,
            depends_on=list(step.depends_on),
        )

        return ReflectionResult(
            status="FAILED_RECOVERABLE",
            critique=(
                f"Target file '{path_str or 'unknown'}' was not found. "
                f"Directory exploration in '{parent_dir}/' is required to discover the real path."
            ),
            suggested_action="INSERT_STEPS",
            suggested_steps=[remediation_step],
            critique_status="REJECTED",
            root_cause="FileNotFoundError",
            failure_classification="MISSING_RESOURCE",
        )

    def _check_syntax_error(self, step: TaskStep, text: str) -> ReflectionResult | None:
        lower = text.lower()

        # Check explicit error mentions
        if "syntaxerror" in lower or "invalid syntax" in lower:
            return ReflectionResult(
                status="FAILED_RECOVERABLE",
                critique="SyntaxError in Python AST: invalid syntax detected in generated code.",
                suggested_action="RETRY",
                critique_status="REJECTED",
                root_cause="SyntaxError in Python AST",
                failure_classification="SYNTAX_ERROR",
            )

        # If CodeWorker/SWEWorker generated Python code, perform AST validation
        worker_name = (step.assigned_worker or "").lower()
        if worker_name in ("codeworker", "sweworker", "coder"):
            code_snippets = self._extract_python_code(text)
            for snippet in code_snippets:
                try:
                    ast.parse(snippet)
                except SyntaxError as exc:
                    return ReflectionResult(
                        status="FAILED_RECOVERABLE",
                        critique=(
                            f"SyntaxError in Python AST: {exc.msg} at line {exc.lineno}. "
                            "Please regenerate with valid Python syntax."
                        ),
                        suggested_action="RETRY",
                        critique_status="REJECTED",
                        root_cause="SyntaxError in Python AST",
                        failure_classification="SYNTAX_ERROR",
                    )

        return None

    def _check_schema_validation(self, step: TaskStep, text: str) -> ReflectionResult | None:
        _ = step
        lower = text.lower()
        schema_indicators = [
            "validationerror",
            "invalid parameter",
            "missing required argument",
            "schema validation failed",
            "argument validation error",
            "invalid json payload",
        ]
        if any(ind in lower for ind in schema_indicators):
            return ReflectionResult(
                status="FAILED_RECOVERABLE",
                critique=(
                    f"Parameter schema validation failed: {text}. "
                    "Adjust arguments to match tool inputSchema and retry."
                ),
                suggested_action="RETRY",
                critique_status="REJECTED",
                root_cause="ParameterValidationError",
                failure_classification="SCHEMA_VALIDATION",
            )
        return None

    def _extract_path(self, *sources: object) -> str:
        for src in sources:
            src_str = str(src) if isinstance(src, dict) else src
            if not isinstance(src_str, str):
                continue
            # Look for quoted paths or words with slashes
            match = re.search(r"['\"]([a-zA-Z0-9_\-\./\\]+\.[a-zA-Z0-9]+)['\"]", src_str)
            if match:
                return match.group(1)
            match_slash = re.search(r"\b([a-zA-Z0-9_\-]+/[a-zA-Z0-9_\-\./]+)\b", src_str)
            if match_slash:
                return match_slash.group(1)
        return "config/settings.json"

    def _extract_python_code(self, text: str) -> list[str]:
        # Extract markdown code blocks
        blocks = re.findall(r"```(?:python)?\s*(.*?)\s*```", text, re.DOTALL)
        if blocks:
            return blocks
        # If output looks like raw python code
        if any(keyword in text for keyword in ("def ", "class ", "import ", "return ")):
            return [text]
        return []
