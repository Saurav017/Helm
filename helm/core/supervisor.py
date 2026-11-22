"""
helm/core/supervisor.py — Supervisor Node and Execution Coordinator.

Implements the top-level orchestration loop:
  1. Decomposes goals into a validated TaskGraph via Gemini API or heuristic fallback.
  2. Prunes available tools dynamically per active step.
  3. Enforces CircuitBreaker checks and routes tool calls through SafetyGate via MCPClientManager.
  4. Synthesizes task results and completes the session.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable
from datetime import UTC, datetime

import httpx

from helm.config.settings import Settings
from helm.core.circuit_breaker import CircuitBreaker
from helm.core.model_router import ModelRouter, ToolCall, extract_tool_arguments
from helm.core.planner import RePlanner
from helm.core.reflection import ReflectionEngine
from helm.core.state import (
    ModelTier,
    SessionState,
    SessionStatus,
    Task,
    TaskBrief,
    TaskResult,
    TaskStatus,
)
from helm.core.synthesis import clean_synthesizer_output
from helm.core.task_graph import StepStatus, TaskGraph, TaskStep
from helm.memory.compactor import ContextCompactor
from helm.memory.pruner import ToolPruner
from helm.protocols.mcp_client import MCPClientManager
from helm.protocols.safety_gate import SafetyGate
from helm.ui.events import (
    SessionDoneEvent,
    StepProgressEvent,
    ThinkingEvent,
    ToolCallCompleteEvent,
    ToolCallStartEvent,
    UIEvent,
)
from helm.ui.renderer import ActivityStreamRenderer
from helm.workers.base import BaseWorker
from helm.workers.coder import CodeWorker
from helm.workers.delegator import WorkerDelegator
from helm.workers.researcher import ResearchWorker
from helm.workers.tool_worker import ToolWorker

logger = logging.getLogger(__name__)

# Default model target
DEFAULT_PLANNING_MODEL = "gemini-3.8-flash"
_MAX_REFLECTION_RETRIES = 2


class Supervisor:
    """
    Central orchestration supervisor.
    Coordinates goal decomposition, task graph execution, tool pruning, and safety enforcement.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        tool_pruner: ToolPruner | None = None,
        model_router: ModelRouter | None = None,
        safety_gate: SafetyGate | None = None,
        *,
        compactor: ContextCompactor | None = None,
        reflection_engine: ReflectionEngine | None = None,
        replanner: RePlanner | None = None,
        delegator: WorkerDelegator | None = None,
        api_key: str | None = None,
        model_name: str = DEFAULT_PLANNING_MODEL,
        ui_renderer: ActivityStreamRenderer | None = None,
        event_callback: Callable[[UIEvent], None] | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.tool_pruner = tool_pruner or ToolPruner(
            default_top_k=self.settings.tool_pruning.top_k,
            threshold=self.settings.tool_pruning.similarity_threshold,
        )
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
        self.model_name = model_name
        self.model_router = model_router or ModelRouter(
            settings=self.settings,
            api_key=self.api_key,
        )
        self.safety_gate = safety_gate or SafetyGate()
        self.compactor = compactor or ContextCompactor(model_router=self.model_router)
        self.reflection_engine = reflection_engine or ReflectionEngine()
        self.replanner = replanner or RePlanner()
        self.ui_renderer = ui_renderer
        self.event_callback = event_callback

        code_worker = CodeWorker(
            model_router=self.model_router,
            event_callback=self.emit_event,
        )
        self.workers: dict[str, BaseWorker] = {
            "ToolWorker": ToolWorker(
                model_router=self.model_router,
                safety_gate=self.safety_gate,
                event_callback=self.emit_event,
            ),
            "ResearchWorker": ResearchWorker(
                model_router=self.model_router,
                event_callback=self.emit_event,
            ),
            "CodeWorker": code_worker,
            "SWEWorker": code_worker,
        }
        self.delegator = delegator or WorkerDelegator(workers=self.workers)
        code_worker.delegator = self.delegator

    def emit_event(self, event: UIEvent) -> None:
        """Emit presentation event to active UI renderer and optional callback."""
        if self.ui_renderer is not None:
            try:
                self.ui_renderer.handle_event(event)
            except Exception as exc:
                logger.debug("UI renderer handle_event failed: %s", exc)
        if self.event_callback is not None:
            try:
                self.event_callback(event)
            except Exception as exc:
                logger.debug("Supervisor event_callback failed: %s", exc)

    def resolve_worker_name(self, step: TaskStep, *, has_tools: bool = False) -> str:
        """Route step to appropriate specialized worker based on step intent."""
        if step.assigned_worker in ("ToolWorker", "ResearchWorker", "CodeWorker", "SWEWorker"):
            return step.assigned_worker

        title_lower = (step.title or "").lower()
        desc_lower = (step.description or "").lower()
        combined = f"{title_lower} {desc_lower}"

        # 1. MCP tool invocation intent
        if (
            step.target_tool
            or has_tools
            or any(
                w in combined
                for w in (
                    "call mcp tool",
                    "call tool",
                    "invoke tool",
                    "tool call",
                    "mcp tool",
                    "read_graph",
                    "create_entities",
                    "delete_entities",
                    "add_observations",
                    "write_file",
                    "read_file",
                )
            )
        ):
            return "ToolWorker"

        # 2. Code, schema, transformation intent
        if any(
            w in combined
            for w in (
                "code",
                "format",
                "schema",
                "transform",
                "pydantic",
                "script",
                "refactor",
                "generate code",
                "json format",
            )
        ):
            return "CodeWorker"

        # 3. Research, analysis, synthesis intent
        return "ResearchWorker"

    async def decompose_goal(self, goal: str, available_tools_summary: str = "") -> TaskGraph:
        """
        Decompose a high-level user goal into a validated TaskGraph.

        Attempts remote decomposition via Gemini API if api_key is present;
        otherwise falls back to structured heuristic decomposition.
        """
        if self.api_key:
            try:
                return await self._decompose_with_gemini(goal, available_tools_summary)
            except Exception as exc:
                logger.warning("Gemini goal decomposition failed (%s); using fallback planner", exc)

        return self._heuristic_decompose(goal)

    async def _decompose_with_gemini(self, goal: str, available_tools_summary: str) -> TaskGraph:
        """Query Gemini REST API to decompose user goal into a structured DAG."""
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_name}:generateContent?key={self.api_key}"
        )

        prompt = (
            "You are an expert AI agent architect. Decompose the following user goal into "
            "2 to 4 discrete, dependent task steps forming a directed acyclic graph (DAG).\n\n"
            f"User Goal: {goal}\n\n"
            f"{available_tools_summary}\n\n"
            "Output ONLY a raw JSON object with this exact structure:\n"
            '{"steps": [\n'
            '  {"step_id": "1", "title": "Step title", '
            '"description": "What to do", "target_tool": "tool_name_or_null", "depends_on": []},\n'
            '  {"step_id": "2", "title": "Next step", '
            '"description": "What to do next", "target_tool": null, "depends_on": ["1"]}\n'
            "]}\n"
        )

        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        text = ""
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                text = parts[0].get("text", "")

        # Parse JSON from response
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\n?", "", text)
            text = re.sub(r"\n?```$", "", text)

        parsed = json.loads(text)
        steps_raw = parsed.get("steps", [])

        graph = TaskGraph(goal=goal)
        for s in steps_raw:
            s_target = s.get("target_tool")
            step = TaskStep(
                step_id=str(s.get("step_id", "")),
                title=s.get("title", ""),
                description=s.get("description", ""),
                target_tool=s_target,
                assigned_worker="ToolWorker" if s_target else "ExecutorWorker",
                depends_on=[str(d) for d in s.get("depends_on", [])],
            )
            graph.add_step(step)

        # Fallback: if no step specified target_tool, infer from goal intent
        if not any(step.target_tool for step in graph.steps.values()):
            goal_lower = goal.lower()
            inferred_tool: str | None = None
            if "entity" in goal_lower or "entities" in goal_lower:
                if any(w in goal_lower for w in ("delete", "remove", "drop")):
                    inferred_tool = "delete_entities"
                elif any(
                    w in goal_lower
                    for w in ("read", "get", "what", "list", "show", "inspect", "query", "fetch")
                ):
                    inferred_tool = "read_graph"
                elif any(w in goal_lower for w in ("store", "create", "record", "insert")) or (
                    "save" in goal_lower and "saved" not in goal_lower
                ):
                    inferred_tool = "create_entities"
                elif "observation" in goal_lower:
                    inferred_tool = "add_observations"
                elif "saved" in goal_lower:
                    inferred_tool = "read_graph"
            elif any(
                w in goal_lower
                for w in ("knowledge graph", "read graph", "what entities", "saved entities")
            ):
                inferred_tool = "read_graph"

            if inferred_tool:
                for step in graph.steps.values():
                    if step.depends_on:
                        step.target_tool = inferred_tool
                        step.assigned_worker = "ToolWorker"
                        break

        graph.validate_dag()
        return graph

    def _heuristic_decompose(self, goal: str) -> TaskGraph:
        """Fast, deterministic fallback goal decomposition for offline execution."""
        graph = TaskGraph(goal=goal)
        goal_lower = goal.lower()

        target_tool: str | None = None
        if "entity" in goal_lower or "entities" in goal_lower:
            if any(w in goal_lower for w in ("delete", "remove", "drop")):
                target_tool = "delete_entities"
            elif any(
                w in goal_lower
                for w in ("read", "get", "what", "list", "show", "inspect", "query", "fetch")
            ):
                target_tool = "read_graph"
            elif any(w in goal_lower for w in ("store", "create", "record", "insert")) or (
                "save" in goal_lower and "saved" not in goal_lower
            ):
                target_tool = "create_entities"
            elif "observation" in goal_lower:
                target_tool = "add_observations"
            elif "saved" in goal_lower:
                target_tool = "read_graph"
        elif any(
            w in goal_lower
            for w in ("knowledge graph", "read graph", "what entities", "saved entities")
        ):
            target_tool = "read_graph"
        elif "write" in goal_lower or "file" in goal_lower:
            if any(w in goal_lower for w in ("read", "cat", "inspect", "show")):
                target_tool = "read_file"
            else:
                target_tool = "write_file"

        # Step 1: Analysis & Discovery
        s1 = TaskStep(
            step_id="step-1",
            title="Inspect environment & tools",
            description=f"Analyze active tools and context relevant to goal: '{goal}'",
            assigned_worker="ResearchWorker",
            depends_on=[],
        )
        graph.add_step(s1)

        # Step 2: Primary execution
        s2 = TaskStep(
            step_id="step-2",
            title="Execute resolution action",
            description=f"Execute required actions or tool calls for: '{goal}'",
            target_tool=target_tool,
            assigned_worker="ToolWorker" if target_tool else "CodeWorker",
            depends_on=["step-1"],
        )
        graph.add_step(s2)

        # Step 3: Synthesis & Verification
        s3 = TaskStep(
            step_id="step-3",
            title="Synthesize final output",
            description=f"Verify results and synthesize final response for: '{goal}'",
            assigned_worker="ResearchWorker",
            depends_on=["step-2"],
        )
        graph.add_step(s3)

        return graph

    async def run(  # noqa: PLR0915
        self,
        state: SessionState,
        mcp_client: MCPClientManager,
        *,
        auto_approve: bool = False,
        graph: TaskGraph | None = None,
    ) -> SessionState:
        """
        Execute the supervisor planning and task resolution loop.

        Parameters
        ----------
        state:
            Active session state (mutated in place and returned).
        mcp_client:
            Connected MCP manager providing tool execution.
        auto_approve:
            Override flag bypassing terminal safety gate confirmation.
        graph:
            Optional pre-configured or resumed TaskGraph.
        """
        state.transition(SessionStatus.RUNNING)
        self.emit_event(
            ThinkingEvent(message=f"Decomposing goal into execution DAG: {state.user_goal}")
        )

        # 1. Discover all tools from MCP servers and seed the ToolPruner
        all_tools = await mcp_client.list_tools()
        if all_tools:
            self.tool_pruner.load_tools(all_tools)

        tools_summary = self.tool_pruner.format_for_prompt(all_tools[:10])

        # 2. Decompose user goal into TaskGraph (or restore provided/persisted)
        if graph is not None:
            active_graph = graph
        elif (
            "task_graph" in state.metadata
            and isinstance(state.metadata["task_graph"], dict)
            and "steps" in state.metadata["task_graph"]
        ):
            active_graph = TaskGraph.from_dict(state.metadata["task_graph"])
        else:
            active_graph = await self.decompose_goal(state.user_goal, tools_summary)

        graph = active_graph
        state.metadata["task_graph"] = graph.to_dict()

        # Mirror graph steps as core tasks in state
        for step in graph.steps.values():
            if not state.get_task(step.step_id):
                initial_worker = self.resolve_worker_name(step)
                new_task = Task(
                    task_id=step.step_id,
                    description=step.description,
                    status=TaskStatus.COMPLETED
                    if step.status == StepStatus.COMPLETED
                    else TaskStatus.PENDING,
                    assigned_worker=initial_worker,
                    depends_on=step.depends_on,
                )
                state.add_task(new_task)

        # 3. Execution loop
        max_loop_iterations = self.settings.circuit_breaker.max_step_ceiling
        loop_iteration = 0
        step_retry_counts: dict[str, int] = {}

        while not graph.is_finished() and loop_iteration < max_loop_iterations:
            loop_iteration += 1

            # Check circuit breaker global step limit
            if state.circuit_breaker.total_steps >= self.settings.circuit_breaker.max_step_ceiling:
                logger.warning("Circuit breaker global step ceiling reached")
                state.transition(SessionStatus.ABORTED)
                state.final_answer = (
                    "Execution halted: circuit breaker maximum step ceiling reached."
                )
                self.emit_event(
                    SessionDoneEvent(
                        summary=state.final_answer,
                        cost=state.circuit_breaker.total_cost_usd,
                        duration_sec=max(
                            0.0, (datetime.now(UTC) - state.created_at).total_seconds()
                        ),
                    )
                )
                return state

            ready_steps = graph.get_ready_steps()
            if not ready_steps:
                # Graph deadlocked or waiting
                break

            for step in ready_steps:
                # Check circuit breaker global step limit before step dispatch
                if (
                    state.circuit_breaker.total_steps
                    >= self.settings.circuit_breaker.max_step_ceiling
                ):
                    logger.warning("Circuit breaker global step ceiling reached during dispatch")
                    state.transition(SessionStatus.ABORTED)
                    state.final_answer = (
                        "Execution halted: circuit breaker maximum step ceiling reached."
                    )
                    self.emit_event(
                        SessionDoneEvent(
                            summary=state.final_answer,
                            cost=state.circuit_breaker.total_cost_usd,
                            duration_sec=max(
                                0.0, (datetime.now(UTC) - state.created_at).total_seconds()
                            ),
                        )
                    )
                    return state

                graph.mark_running(step.step_id)
                matched_task = state.get_task(step.step_id)
                if matched_task:
                    matched_task.status = TaskStatus.IN_PROGRESS
                    matched_task.step_count += 1

                # Dynamic Tool Pruning (RAG for tools)
                pruning_query = f"{state.user_goal} {step.title} {step.description}"
                pruned_tools = self.tool_pruner.prune(
                    pruning_query,
                    top_k=self.settings.tool_pruning.top_k,
                )
                if not pruned_tools and all_tools:
                    pruned_tools = list(all_tools[: self.settings.tool_pruning.top_k])
                if step.target_tool and any(t.name == step.target_tool for t in all_tools):
                    target_tool_obj = next(t for t in all_tools if t.name == step.target_tool)
                    if target_tool_obj not in pruned_tools:
                        pruned_tools.insert(0, target_tool_obj)

                # Context compaction across completed steps
                completed_steps = [
                    s for s in graph.steps.values() if s.status == StepStatus.COMPLETED
                ]
                compacted_context = await self.compactor.compact_steps(completed_steps)

                # 1. Worker Assignment & Step Classification
                title_lower = (step.title or "").lower()
                desc_lower = (step.description or "").lower()
                non_action_titles = (
                    "inspect",
                    "analysis",
                    "discovery",
                    "synthesize",
                    "synthesis",
                    "verify",
                    "verification",
                    "summary",
                )
                action_kws = (
                    "execute",
                    "action",
                    "create",
                    "store",
                    "write",
                    "delete",
                    "run",
                    "apply",
                    "call",
                    "perform",
                    "resolve",
                    "add",
                    "update",
                    "insert",
                    "save",
                    "record",
                    "modify",
                    "edit",
                    "patch",
                    "remove",
                )
                is_inspection_or_synthesis = any(
                    kw in title_lower for kw in non_action_titles
                ) or any(kw in desc_lower for kw in non_action_titles)
                is_execution_step = not is_inspection_or_synthesis and bool(
                    step.target_tool
                    or any(kw in title_lower for kw in action_kws)
                    or any(kw in desc_lower for kw in action_kws)
                )

                worker_name = self.resolve_worker_name(
                    step,
                    has_tools=bool(pruned_tools and is_execution_step),
                )
                is_tool_action = worker_name == "ToolWorker"
                worker = self.workers.get(worker_name, self.workers["ResearchWorker"])
                step.assigned_worker = worker_name
                step.worker = worker_name
                if matched_task:
                    matched_task.assigned_worker = worker_name
                    matched_task.worker = worker_name

                self.emit_event(
                    StepProgressEvent(
                        step_id=step.step_id,
                        status="RUNNING",
                        worker=worker_name,
                    )
                )
                self.emit_event(
                    ThinkingEvent(message=step.description or step.title or step.step_id)
                )

                # 2. Real Tool Invocation via ModelRouter with pruned tool schemas
                tool_declarations = (
                    self.tool_pruner.format_for_llm_declarations(pruned_tools)
                    if is_tool_action
                    else None
                )
                step_prompt = (
                    f"Role Instructions: {worker.system_prompt}\n\n"
                    f"User Goal: {state.user_goal}\n"
                    f"Current Step: {step.title or step.step_id}\n"
                    f"Step Description: {step.description}\n"
                )
                if compacted_context:
                    step_prompt += f"\nPrior Steps Context:\n{compacted_context}\n"
                step_prompt += (
                    "\nInvoke the appropriate tool function call or reason to execute this step."
                )
                turn = await self.model_router.generate_turn(
                    prompt=step_prompt,
                    tools=tool_declarations,
                    tier=self.model_router.route(step.description),
                    target_tool=step.target_tool,
                )

                active_tool_calls = list(turn.tool_calls) if is_tool_action else []
                if not active_tool_calls and is_tool_action:
                    if step.target_tool and any(t.name == step.target_tool for t in all_tools):
                        extracted_args = extract_tool_arguments(step.target_tool, state.user_goal)
                        active_tool_calls.append(
                            ToolCall(name=step.target_tool, args=extracted_args)
                        )
                    elif pruned_tools:
                        goal_lower = state.user_goal.lower()
                        for pt in pruned_tools:
                            if any(t.name == pt.name for t in all_tools):
                                if (
                                    pt.name == "create_entities"
                                    and ("entity" in goal_lower or "entities" in goal_lower)
                                    and any(
                                        w in goal_lower
                                        for w in ("store", "create", "add", "record", "save")
                                    )
                                ):
                                    extracted_args = extract_tool_arguments(
                                        "create_entities", state.user_goal
                                    )
                                    active_tool_calls.append(
                                        ToolCall(name="create_entities", args=extracted_args)
                                    )
                                    break
                                if pt.name == "write_file" and (
                                    "write" in goal_lower or "file" in goal_lower
                                ):
                                    extracted_args = extract_tool_arguments(
                                        "write_file", state.user_goal
                                    )
                                    active_tool_calls.append(
                                        ToolCall(name="write_file", args=extracted_args)
                                    )
                                    break

                step_output = ""
                if active_tool_calls:
                    for tc in active_tool_calls:
                        tool_to_call = tc.name
                        tool_args = tc.args

                        if not any(t.name == tool_to_call for t in all_tools):
                            continue

                        matched_tool_def = next(
                            (t for t in all_tools if t.name == tool_to_call), None
                        )
                        server_name = matched_tool_def.server_name if matched_tool_def else ""

                        self.emit_event(
                            ToolCallStartEvent(
                                tool_name=tool_to_call,
                                server_name=server_name,
                                arguments=tool_args,
                            )
                        )

                        # Check circuit breaker
                        cb_check = self.circuit_breaker.check(
                            state.circuit_breaker,
                            tool_name=tool_to_call,
                            arguments=tool_args,
                        )
                        if not cb_check.allowed:
                            graph.mark_failed(
                                step.step_id,
                                reason=(
                                    f"Circuit breaker blocked call: {cb_check.record.block_reason}"
                                ),
                            )
                            if matched_task:
                                matched_task.status = TaskStatus.BLOCKED
                                matched_task.tool_calls.append(cb_check.record)
                            self.emit_event(
                                ToolCallCompleteEvent(
                                    tool_name=tool_to_call,
                                    summary=(
                                        f"Circuit breaker blocked: {cb_check.record.block_reason}"
                                    ),
                                    is_error=True,
                                )
                            )
                            break

                        # Intercept with SafetyGate
                        tool_desc = step.title or step.description
                        approved = self.safety_gate.request_permission(
                            tool_name=tool_to_call,
                            arguments=tool_args,
                            auto_approve=auto_approve or not self.settings.safety.hitl_mutations,
                            summary=f"Executing {tool_to_call} for {tool_desc}",
                        )
                        if not approved:
                            graph.mark_failed(
                                step.step_id,
                                reason="Action denied by operator in safety gate",
                            )
                            step.error = "Action denied by operator in safety gate"
                            if matched_task:
                                matched_task.status = TaskStatus.FAILED
                                matched_task.tool_calls.append(cb_check.record)
                            self.emit_event(
                                ToolCallCompleteEvent(
                                    tool_name=tool_to_call,
                                    summary="Action denied by operator in safety gate",
                                    is_error=True,
                                )
                            )
                            break

                        call_result = await mcp_client.call_tool(
                            tool_to_call,
                            tool_args,
                            auto_approve=True,
                            summary=f"Executing {tool_to_call} for {tool_desc}",
                        )

                        if call_result.get("action_denied"):
                            graph.mark_failed(
                                step.step_id,
                                reason="Action denied by operator in safety gate",
                            )
                            step.error = "Action denied by operator in safety gate"
                            if matched_task:
                                matched_task.status = TaskStatus.FAILED
                                matched_task.tool_calls.append(cb_check.record)
                            self.emit_event(
                                ToolCallCompleteEvent(
                                    tool_name=tool_to_call,
                                    summary="Action denied by operator in safety gate",
                                    is_error=True,
                                )
                            )
                            break

                        # Increment step counter and assign worker
                        state.record_step()
                        step.worker = worker_name
                        if matched_task:
                            matched_task.worker = worker_name

                        raw_result_str = json.dumps(call_result.get("content", call_result))
                        args_str = json.dumps(tool_args)

                        # Detect tool error return payloads
                        is_tool_err = False
                        if isinstance(call_result, dict) and (
                            call_result.get("is_error") or "error" in call_result
                        ):
                            is_tool_err = True
                            step.error = str(call_result.get("error", raw_result_str))
                            step_output = step.error
                            self.emit_event(
                                ToolCallCompleteEvent(
                                    tool_name=tool_to_call,
                                    summary=step.error,
                                    is_error=True,
                                )
                            )
                        elif isinstance(call_result, str) and (
                            "error" in call_result.lower() or "exception" in call_result.lower()
                        ):
                            is_tool_err = True
                            step.error = call_result
                            step_output = call_result
                            self.emit_event(
                                ToolCallCompleteEvent(
                                    tool_name=tool_to_call,
                                    summary=step.error,
                                    is_error=True,
                                )
                            )
                        else:
                            line_count = raw_result_str.count("\n") + 1
                            byte_count = len(raw_result_str.encode("utf-8"))
                            self.emit_event(
                                ToolCallCompleteEvent(
                                    tool_name=tool_to_call,
                                    summary=(
                                        f"output summarized: {line_count} lines / "
                                        f"{byte_count} bytes"
                                    ),
                                    is_error=False,
                                )
                            )

                        if not is_tool_err:
                            verify_prompt = (
                                f"Step: {tool_desc}\n"
                                f"Tool '{tool_to_call}' executed with args {args_str}.\n"
                                f"Result: {raw_result_str}\n\n"
                                f"Verify this action and provide a concise summary of the outcome."
                            )
                            verification = await self.model_router.generate(
                                verify_prompt,
                                tier=ModelTier.FAST,
                            )
                            step_output = verification or raw_result_str
                            read_keywords = ("read", "search", "list", "get", "query")
                            if any(k in tool_to_call for k in read_keywords):
                                step_output = (
                                    f"{verification}\n\n{raw_result_str}"
                                    if verification
                                    else raw_result_str
                                )
                        if matched_task:
                            matched_task.tool_calls.append(cb_check.record)

                if not step_output and step.status != StepStatus.FAILED:
                    # Informational reasoning step via specialized worker
                    state.record_step()
                    worker = self.workers.get(worker_name, self.workers["ResearchWorker"])
                    brief = TaskBrief(
                        task_id=step.step_id,
                        description=step.description or step.title,
                        relevant_context=compacted_context,
                    )
                    worker_res = await worker.run(brief)
                    raw_text = (
                        worker_res.output
                        or turn.text
                        or (f"Evaluated and completed step '{step.title or step.step_id}'")
                    )
                    if "Completed analysis for" in raw_text:
                        raw_text = f"Evaluated and completed step '{step.title or step.step_id}'"
                    step_output = raw_text
                    if matched_task:
                        matched_task.worker = worker_name
                elif not step_output:
                    step_output = step.error or "Step failed"

                # 3. Autonomous Self-Reflection & Dynamic Re-planning (Milestone 6)
                self.emit_event(
                    ThinkingEvent(
                        message=f"Reflecting on step '{step.title or step.step_id}' results..."
                    )
                )
                reflection = self.reflection_engine.evaluate_step(
                    step=step,
                    result=step_output,
                    context={
                        "user_goal": state.user_goal,
                        "tools": all_tools,
                        "step_id": step.step_id,
                    },
                )

                if reflection.status == "FAILED_RECOVERABLE":
                    if reflection.suggested_action == "INSERT_STEPS" and reflection.suggested_steps:
                        new_steps = self.replanner.apply_recovery(graph, step, reflection)
                        self.emit_event(
                            ThinkingEvent(
                                message=(
                                    f"Dynamic DAG replanning: inserting {len(new_steps)} "
                                    f"recovery step(s)"
                                )
                            )
                        )
                        for ns in new_steps:
                            if not state.get_task(ns.step_id):
                                state.tasks.append(
                                    Task(
                                        task_id=ns.step_id,
                                        description=ns.description or ns.title,
                                        status=TaskStatus.PENDING,
                                        depends_on=list(ns.depends_on),
                                        assigned_worker=ns.assigned_worker,
                                    )
                                )
                        step.status = StepStatus.PENDING
                        step.error = None
                        if matched_task:
                            matched_task.status = TaskStatus.PENDING
                            matched_task.depends_on = list(step.depends_on)
                        logger.info(
                            "Dynamic DAG replanning: inserted %d recovery step(s) before '%s'",
                            len(new_steps),
                            step.step_id,
                            extra={"event": "dag_mutation", "target": step.step_id},
                        )
                        state.metadata["task_graph"] = graph.to_dict()
                        state.save()
                        break

                    if reflection.suggested_action == "RETRY":
                        curr_retries = step_retry_counts.get(step.step_id, 0)
                        if curr_retries < _MAX_REFLECTION_RETRIES:
                            self.emit_event(
                                ThinkingEvent(
                                    message=(
                                        f"Retrying step '{step.step_id}' "
                                        f"({curr_retries + 1}/{_MAX_REFLECTION_RETRIES}): "
                                        f"{reflection.critique}"
                                    )
                                )
                            )
                            step_retry_counts[step.step_id] = curr_retries + 1
                            critique_text = f"\n[Critique]: {reflection.critique}"
                            if critique_text not in step.description:
                                step.description = f"{step.description}{critique_text}"
                            step.status = StepStatus.PENDING
                            step.error = reflection.critique
                            step.output = None
                            if matched_task:
                                matched_task.status = TaskStatus.PENDING
                                matched_task.description = step.description
                            logger.info(
                                "Step %s failed reflection check. Retrying (%d/%d): %s",
                                step.step_id,
                                curr_retries + 1,
                                _MAX_REFLECTION_RETRIES,
                                reflection.critique,
                                extra={"event": "reflection_retry", "step_id": step.step_id},
                            )
                            state.metadata["task_graph"] = graph.to_dict()
                            state.save()
                            break

                        graph.mark_failed(step.step_id, reason=reflection.critique)
                        if matched_task:
                            matched_task.status = TaskStatus.FAILED
                            matched_task.result = TaskResult(
                                task_id=step.step_id,
                                status=TaskStatus.FAILED,
                                error=reflection.critique,
                                steps_used=matched_task.step_count,
                            )
                        self.emit_event(
                            StepProgressEvent(
                                step_id=step.step_id,
                                status="FAILED",
                                worker=worker_name,
                            )
                        )

                elif reflection.status == "FAILED_FATAL":
                    graph.mark_failed(step.step_id, reason=reflection.critique)
                    if matched_task:
                        matched_task.status = TaskStatus.FAILED
                        matched_task.result = TaskResult(
                            task_id=step.step_id,
                            status=TaskStatus.FAILED,
                            error=reflection.critique,
                            steps_used=matched_task.step_count,
                        )
                    self.emit_event(
                        StepProgressEvent(
                            step_id=step.step_id,
                            status="FAILED",
                            worker=worker_name,
                        )
                    )
                    state.metadata["task_graph"] = graph.to_dict()
                    state.save()
                    break

                # If reflection passed, finalize step completion
                if step.status != StepStatus.FAILED:
                    graph.mark_completed(step.step_id, result=step_output)
                    if matched_task:
                        matched_task.status = TaskStatus.COMPLETED
                        matched_task.result = TaskResult(
                            task_id=step.step_id,
                            status=TaskStatus.COMPLETED,
                            output=step_output,
                            steps_used=matched_task.step_count,
                        )
                    self.emit_event(
                        StepProgressEvent(
                            step_id=step.step_id,
                            status="COMPLETED",
                            worker=worker_name,
                        )
                    )
                else:
                    self.emit_event(
                        StepProgressEvent(
                            step_id=step.step_id,
                            status="FAILED",
                            worker=worker_name,
                        )
                    )

                # Update conversation messages
                step_name = step.title or step.step_id
                state.add_message(
                    role="assistant",
                    content=f"Step '{step_name}' completed by {worker_name}: {step_output}",
                )

                # Checkpoint session state and task graph to disk after step completion
                state.metadata["task_graph"] = graph.to_dict()
                state.save()

        # 4. Final synthesis
        any_denied = any(
            s.status == StepStatus.FAILED and "denied" in (s.error or "").lower()
            for s in graph.steps.values()
        )
        if any_denied or not graph.is_finished():
            state.transition(SessionStatus.ABORTED)
            state.final_answer = (
                "Session aborted: Action denied by operator in safety gate."
                if any_denied
                else "Execution halted: circuit breaker maximum step ceiling reached."
            )
        elif graph.is_successful():
            state.transition(SessionStatus.COMPLETED)
            synthesis_step = next(
                (
                    s
                    for s in reversed(graph.get_topological_order())
                    if s.assigned_worker == "ResearchWorker"
                    or "synthesize" in (s.title or "").lower()
                ),
                None,
            )
            if synthesis_step and synthesis_step.output:
                state.final_answer = clean_synthesizer_output(synthesis_step.output)
            else:
                summary_parts = [f"Successfully executed task graph for goal: '{state.user_goal}'."]
                for step in graph.get_topological_order():
                    step_text = clean_synthesizer_output(step.output) if step.output else "Done"
                    summary_parts.append(f"- {step.title or step.step_id}: {step_text}")
                state.final_answer = "\n".join(summary_parts)
        elif any(s.status == StepStatus.FAILED for s in graph.steps.values()):
            state.transition(SessionStatus.FAILED)
            failed_steps = [
                s.step_id for s in graph.steps.values() if s.status == StepStatus.FAILED
            ]
            state.final_answer = f"Task execution failed at steps: {', '.join(failed_steps)}."
        else:
            state.transition(SessionStatus.COMPLETED)
            state.final_answer = f"Finished processing goal: '{state.user_goal}'."

        # Emit session done event
        duration_sec = max(0.0, (datetime.now(UTC) - state.created_at).total_seconds())
        cost = state.circuit_breaker.total_cost_usd
        self.emit_event(
            SessionDoneEvent(
                summary=state.final_answer or "",
                cost=cost,
                duration_sec=duration_sec,
            )
        )

        # Final checkpoint to disk
        state.metadata["task_graph"] = graph.to_dict()
        state.save()

        return state
