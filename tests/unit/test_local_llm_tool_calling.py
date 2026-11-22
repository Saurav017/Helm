"""
tests/unit/test_local_llm_tool_calling.py — Tests for local LLM tool calling fixes:
1. Tool Schema Formatting & System Prompt Injection (OpenAI/Ollama).
2. Tool-Call Fallback Parser (Python code block / markdown text extraction & alias resolution).
3. Clean Synthesizer Output Presentation (filtering internal planning steps).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from helm.core.model_router import (
    ModelRouter,
    extract_tool_calls_from_text,
    resolve_tool_name,
)
from helm.core.providers.openai import (
    OpenAIProvider,
    build_tools_system_instruction,
    translate_tools_to_openai,
)
from helm.core.state import ModelTier, TaskBrief
from helm.core.synthesis import clean_synthesizer_output
from helm.workers.researcher import ResearchWorker

MOCK_TOOLS = [
    {
        "name": "read_file",
        "description": "Read file contents from disk.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "list_directory",
        "description": "List contents of a directory.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "read_graph",
        "description": "Read the entire knowledge graph.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "create_entities",
        "description": "Create multiple new entities in the knowledge graph.",
        "parameters": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {"type": "object"},
                }
            },
            "required": ["entities"],
        },
    },
]


# ===========================================================================
# 1. Tool Schema Formatting & System Prompt Injection Tests
# ===========================================================================


class TestToolSchemaAndPromptInjection:
    def test_translate_tools_to_openai_strict_format(self) -> None:
        """Tools without parameters or with raw inputSchema conform to OpenAI function spec."""
        raw_tools = [
            {"name": "no_params", "description": "No params tool"},
            {
                "name": "custom_schema",
                "description": "Custom schema",
                "inputSchema": {"properties": {"query": {"type": "string"}}},
            },
        ]
        translated = translate_tools_to_openai(raw_tools)
        assert len(translated) == 2

        assert translated[0]["type"] == "function"
        fn0 = translated[0]["function"]
        assert fn0["name"] == "no_params"
        assert fn0["parameters"]["type"] == "object"
        assert fn0["parameters"]["properties"] == {}

        fn1 = translated[1]["function"]
        assert fn1["name"] == "custom_schema"
        assert fn1["parameters"]["type"] == "object"
        assert "query" in fn1["parameters"]["properties"]

    def test_build_tools_system_instruction(self) -> None:
        """System prompt injection contains tool signatures and anti-python instruction."""
        prompt_block = build_tools_system_instruction(MOCK_TOOLS)
        assert "You have access to the following tools via tool-calling:" in prompt_block
        assert (
            "Do NOT write Python code to execute these tools. Always trigger them via native tool/function calls."
            in prompt_block
        )
        assert "filesystem:" in prompt_block
        assert "`read_file(path)`" in prompt_block
        assert "`list_directory(path)`" in prompt_block
        assert "memory:" in prompt_block
        assert "`read_graph()`" in prompt_block
        assert "`create_entities(entities)`" in prompt_block

    @pytest.mark.asyncio
    async def test_openai_generate_with_tools_injects_system_prompt(self) -> None:
        """OpenAIProvider.generate_with_tools injects tool instructions into system prompt."""
        provider = OpenAIProvider(api_key="test-key")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "Done"}}]
        }
        mock_resp.raise_for_status.return_value = None

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            await provider.generate_with_tools(
                prompt="List files in repo",
                system_prompt="You are a helpful assistant.",
                tools=MOCK_TOOLS,
            )

            call_payload = mock_post.call_args[1]["json"]
            messages = call_payload["messages"]
            system_msg = next((m for m in messages if m["role"] == "system"), None)
            assert system_msg is not None
            assert "You are a helpful assistant." in system_msg["content"]
            assert (
                "You have access to the following tools via tool-calling:" in system_msg["content"]
            )
            assert "filesystem:" in system_msg["content"]
            assert "`list_directory(path)`" in system_msg["content"]
            assert "Do NOT write Python code to execute these tools." in system_msg["content"]


# ===========================================================================
# 2. Tool-Call Fallback Parser Tests
# ===========================================================================


class TestToolCallFallbackParser:
    def test_resolve_tool_name_aliases(self) -> None:
        """Alias resolution maps common model hallucinations to actual MCP tools."""
        assert resolve_tool_name("list_files", MOCK_TOOLS) == "list_directory"
        assert resolve_tool_name("listfiles", MOCK_TOOLS) == "list_directory"
        assert resolve_tool_name("ls", MOCK_TOOLS) == "list_directory"
        assert resolve_tool_name("dir", MOCK_TOOLS) == "list_directory"
        assert resolve_tool_name("readfile", MOCK_TOOLS) == "read_file"
        assert resolve_tool_name("read_file", MOCK_TOOLS) == "read_file"
        assert resolve_tool_name("readgraph", MOCK_TOOLS) == "read_graph"
        assert resolve_tool_name("unknown_tool", MOCK_TOOLS) is None

    def test_extract_python_code_block_list_files(self) -> None:
        """Qwen 2.5 Coder emission ````python list_files() ```` resolves to list_directory."""
        raw_text = "Here is the code to check the directory:\n```python\nlist_files()\n```"
        calls = extract_tool_calls_from_text(raw_text, MOCK_TOOLS)
        assert len(calls) == 1
        assert calls[0].name == "list_directory"
        assert calls[0].args == {"path": "."}

    def test_extract_python_code_block_read_file(self) -> None:
        """Markdown code block with positional path argument is extracted correctly."""
        raw_text = "```python\nread_file('README.md')\n```"
        calls = extract_tool_calls_from_text(raw_text, MOCK_TOOLS)
        assert len(calls) == 1
        assert calls[0].name == "read_file"
        assert calls[0].args == {"path": "README.md"}

    def test_extract_python_code_block_keyword_args(self) -> None:
        """Markdown code block with keyword arguments is extracted correctly."""
        raw_text = '```python\nread_file(path="src/main.py")\n```'
        calls = extract_tool_calls_from_text(raw_text, MOCK_TOOLS)
        assert len(calls) == 1
        assert calls[0].name == "read_file"
        assert calls[0].args == {"path": "src/main.py"}

    def test_extract_json_code_block(self) -> None:
        """JSON code block tool call is extracted and hallucinated name resolved."""
        raw_text = '```json\n{"name": "list_files", "arguments": {"path": "/workspace"}}\n```'
        calls = extract_tool_calls_from_text(raw_text, MOCK_TOOLS)
        assert len(calls) == 1
        assert calls[0].name == "list_directory"
        assert calls[0].args == {"path": "/workspace"}

    def test_extract_inline_tool_call(self) -> None:
        """Inline function call syntax in text is parsed as a tool call."""
        raw_text = 'I will now run read_file("pyproject.toml") to inspect the configuration.'
        calls = extract_tool_calls_from_text(raw_text, MOCK_TOOLS)
        assert len(calls) == 1
        assert calls[0].name == "read_file"
        assert calls[0].args == {"path": "pyproject.toml"}

    @pytest.mark.asyncio
    async def test_model_router_fallback_on_text_emission(self) -> None:
        """When provider returns python code text, ModelRouter extracts it into ToolCall."""
        from helm.core.providers.base import BaseLLMProvider

        mock_provider = MagicMock(spec=BaseLLMProvider)
        mock_provider.provider_name = "openai"
        mock_provider.api_key = "test-key"
        mock_provider.model = "gpt-4o"
        # Simulate local LLM emitting python markdown without structured tool_calls
        mock_provider.generate_with_tools = AsyncMock(
            return_value=("```python\nlist_files()\n```", None)
        )

        router = ModelRouter(provider=mock_provider)
        turn = await router.generate_turn("Inspect directory", tools=MOCK_TOOLS)

        assert len(turn.tool_calls) == 1
        assert turn.tool_calls[0].name == "list_directory"
        assert turn.tool_calls[0].args == {"path": "."}


# ===========================================================================
# 3. Clean Output Presentation Tests
# ===========================================================================


class TestCleanOutputPresentation:
    def test_filter_internal_planning_lines(self) -> None:
        """Internal planning step lines like 'Let's proceed with step 1:' are stripped."""
        raw = (
            "Let's proceed with step 1: Inspect the workspace directory.\n"
            "Let's proceed with step 2: Analyze found files.\n"
            "The repository contains the following files:\n"
            "- README.md\n"
            "- pyproject.toml"
        )
        cleaned = clean_synthesizer_output(raw)
        assert "Let's proceed with step 1" not in cleaned
        assert "Let's proceed with step 2" not in cleaned
        assert "The repository contains the following files:" in cleaned
        assert "- README.md" in cleaned

    def test_filter_inline_planning_prefix(self) -> None:
        """Inline sentence prefix 'Let's proceed with step 1:' is stripped."""
        raw = "Let's proceed with step 1: The current version of Helm is 0.1.0."
        cleaned = clean_synthesizer_output(raw)
        assert cleaned == "The current version of Helm is 0.1.0."

    def test_filter_thinking_tags_and_explicit_answer(self) -> None:
        """Thinking scratchpad tags and explicit final answer blocks are handled."""
        raw = (
            "<think>Analyzing user query and comparing results...</think>\n"
            "Thinking Process:\n"
            "- Step 1: Check evidence\n\n"
            "Final Answer:\n"
            "All services are healthy and operational."
        )
        cleaned = clean_synthesizer_output(raw)
        assert "<think>" not in cleaned
        assert "Thinking Process" not in cleaned
        assert cleaned == "All services are healthy and operational."

    def test_clean_output_preserves_clean_answers(self) -> None:
        """Answers that have no planning chatter remain completely untouched."""
        clean_text = "Successfully updated configuration and verified syntax."
        assert clean_synthesizer_output(clean_text) == clean_text

    @pytest.mark.asyncio
    async def test_research_worker_cleans_synthesizer_output(self) -> None:
        """ResearchWorker execute filters planning steps before returning TaskResult."""
        mock_router = AsyncMock()
        mock_router.generate.return_value = (
            "Let's proceed with step 1: Formulate the synthesis.\n"
            "Step 2: Provide clear answer.\n"
            "The database connection has been established successfully."
        )

        worker = ResearchWorker(model_router=mock_router)
        brief = TaskBrief(task_id="task-1", description="Synthesize status")
        result = await worker.execute(brief, tier=ModelTier.STANDARD)

        assert "Let's proceed with step 1" not in result.output
        assert "Step 2" not in result.output
        assert "The database connection has been established successfully." in result.output

    def test_clean_output_filters_tool_invocation_block(self) -> None:
        """Emitted tool invocation blocks are stripped from synthesizer answers."""
        raw = (
            "Based on the analysis, here is the summary of the repository.\n\n"
            "Tool Invocation:\n"
            "```python\n"
            "# Function call to list all files in the current directory\n"
            "list_files()\n"
            "```\n"
            "Helm is an open-source terminal-first agentic orchestration runtime."
        )
        cleaned = clean_synthesizer_output(raw)
        assert "Tool Invocation:" not in cleaned
        assert "list_files()" not in cleaned
        assert "Helm is an open-source terminal-first agentic orchestration runtime." in cleaned

    def test_heuristic_tool_matching_prioritizes_read_file_over_read_graph(self) -> None:
        """Prompt to read a file matches read_file, never erroneously matching read_graph."""
        router = ModelRouter(mock_mode=True)
        tools = [
            {"name": "read_graph", "description": "Read knowledge graph"},
            {"name": "read_file", "description": "Read file"},
            {"name": "list_directory", "description": "List directory"},
        ]
        prompt = "Use the filesystem tool to read README.md and summarize what this repository is"
        calls = router.heuristic_tool_matching(prompt, tools)
        assert len(calls) == 1
        assert calls[0].name == "read_file"
        assert calls[0].args == {"path": "README.md"}

    def test_heuristic_tool_matching_respects_target_tool(self) -> None:
        """Explicit target_tool is prioritized over generic keyword heuristics."""
        router = ModelRouter(mock_mode=True)
        tools = [
            {"name": "read_graph", "description": "Read knowledge graph"},
            {"name": "read_file", "description": "Read file"},
        ]
        prompt = "Process the goal"
        calls = router.heuristic_tool_matching(prompt, tools, target_tool="read_file")
        assert len(calls) == 1
        assert calls[0].name == "read_file"
