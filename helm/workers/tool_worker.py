"""
helm/workers/tool_worker.py — Deterministic MCP tool execution worker.

Specialized in tool parameter binding, schema conformance, and safe tool invocations.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from helm.core.model_router import ModelRouter, ToolCall
from helm.core.prompts import PARAMETER_CORRECTION_INSTRUCTIONS
from helm.core.state import ModelTier, TaskBrief, TaskResult, TaskStatus
from helm.protocols.mcp_client import MCPClientManager, MCPToolDefinition
from helm.protocols.safety_gate import SafetyGate
from helm.workers.base import BaseWorker

if TYPE_CHECKING:
    from helm.ui.events import UIEvent

logger = logging.getLogger(__name__)

TOOL_WORKER_PROMPT = (
    "You are a Tool Execution Specialist. Your role is deterministic MCP tool invocation, "
    "strict argument schema conformance, and parameter verification. "
    "Execute operations accurately and adhere to safety constraints. "
    f"{PARAMETER_CORRECTION_INSTRUCTIONS}"
)


class ToolWorker(BaseWorker):
    """Worker specialized in executing MCP tool calls."""

    def __init__(
        self,
        name: str = "ToolWorker",
        model_router: ModelRouter | None = None,
        safety_gate: SafetyGate | None = None,
        event_callback: Callable[[UIEvent], None] | None = None,
    ) -> None:
        super().__init__(
            name=name,
            role="Deterministic MCP Tool Executor",
            system_prompt=TOOL_WORKER_PROMPT,
            model_router=model_router,
            event_callback=event_callback,
        )
        self.safety_gate = safety_gate or SafetyGate()

    async def execute(  # noqa: C901, PLR0912
        self,
        brief: TaskBrief,
        *,
        mcp_client: MCPClientManager | None = None,
        tools: list[MCPToolDefinition] | None = None,
        auto_approve: bool = False,
    ) -> TaskResult:
        """
        Execute a tool task by generating tool call and invoking through MCP client.
        """
        tool_defs = tools or []
        tools_payload = [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            }
            for t in tool_defs
        ]

        prompt = (
            f"Role: {self.system_prompt}\n\n"
            f"Task: {brief.description}\n"
            f"Context:\n{brief.relevant_context or 'No prior context.'}\n\n"
            f"Determine and output the necessary tool call."
        )

        turn = await self.model_router.generate_turn(
            prompt,
            tools=tools_payload,
            tier=ModelTier.FAST,
        )

        active_calls = list(turn.tool_calls)
        if not active_calls and tool_defs:
            # Fallback heuristic if turn didn't parse a call
            desc_lower = brief.description.lower()
            for t in tool_defs:
                if t.name in desc_lower or t.name.replace("_", " ") in desc_lower:
                    active_calls.append(ToolCall(name=t.name, args={}))
                    break

        if not active_calls or not mcp_client:
            output_text = turn.text or f"ToolWorker evaluated: {brief.description}"
            return TaskResult(
                task_id=brief.task_id,
                status=TaskStatus.COMPLETED,
                output=output_text,
                steps_used=1,
            )

        executed_outputs: list[str] = []
        for call in active_calls:
            # Check safety gate
            approved = self.safety_gate.request_permission(
                tool_name=call.name,
                arguments=call.args,
                auto_approve=auto_approve,
                summary=f"ToolWorker executing {call.name} for {brief.description}",
            )
            if not approved:
                return TaskResult(
                    task_id=brief.task_id,
                    status=TaskStatus.FAILED,
                    error=f"Action '{call.name}' denied by safety gate",
                    steps_used=1,
                )

            res = await mcp_client.call_tool(
                call.name,
                call.args,
                auto_approve=True,
                summary=f"ToolWorker executing {call.name}",
            )
            raw_str = json.dumps(res.get("content", res))
            executed_outputs.append(f"Tool '{call.name}' output: {raw_str}")

        return TaskResult(
            task_id=brief.task_id,
            status=TaskStatus.COMPLETED,
            output="\n\n".join(executed_outputs),
            steps_used=len(active_calls),
        )
