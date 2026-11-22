"""
tests/unit/test_providers.py — Unit tests for model-agnostic LLM provider adapters,
tool schema translation, authentication hierarchy, and interactive configuration.
"""

from __future__ import annotations

import stat
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from helm.cli.app import app
from helm.cli.auth import run_login_flow
from helm.config.auth import (
    load_credentials,
    load_helm_config,
    resolve_active_provider,
    save_credentials,
    save_helm_config,
)
from helm.core.model_router import ModelRouter, ModelTier
from helm.core.providers import (
    AnthropicProvider,
    GeminiProvider,
    OpenAIProvider,
    create_provider,
    translate_tools_to_anthropic,
    translate_tools_to_gemini,
    translate_tools_to_openai,
)
from helm.exceptions import AuthenticationError, ConfigurationError

runner = CliRunner()

SAMPLE_MCP_TOOLS: list[dict[str, Any]] = [
    {
        "name": "read_file",
        "description": "Read file from disk",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "create_record",
        "description": "Create a database record",
        "parameters": {
            "type": "object",
            "properties": {"table": {"type": "string"}, "data": {"type": "object"}},
            "required": ["table"],
        },
    },
]


# ===========================================================================
# 1. Schema Translation Tests
# ===========================================================================


class TestSchemaTranslations:
    def test_gemini_translation(self) -> None:
        gemini_tools = translate_tools_to_gemini(SAMPLE_MCP_TOOLS)
        assert len(gemini_tools) == 2
        assert gemini_tools[0]["name"] == "read_file"
        assert gemini_tools[0]["description"] == "Read file from disk"
        assert "properties" in gemini_tools[0]["parameters"]
        assert gemini_tools[0]["parameters"]["required"] == ["path"]
        assert gemini_tools[1]["name"] == "create_record"

    def test_anthropic_translation(self) -> None:
        anthropic_tools = translate_tools_to_anthropic(SAMPLE_MCP_TOOLS)
        assert len(anthropic_tools) == 2
        assert anthropic_tools[0]["name"] == "read_file"
        assert anthropic_tools[0]["description"] == "Read file from disk"
        assert "input_schema" in anthropic_tools[0]
        assert anthropic_tools[0]["input_schema"]["type"] == "object"
        assert "path" in anthropic_tools[0]["input_schema"]["properties"]
        assert anthropic_tools[1]["input_schema"]["properties"]["table"]["type"] == "string"

    def test_openai_translation(self) -> None:
        openai_tools = translate_tools_to_openai(SAMPLE_MCP_TOOLS)
        assert len(openai_tools) == 2
        assert openai_tools[0]["type"] == "function"
        fn = openai_tools[0]["function"]
        assert fn["name"] == "read_file"
        assert fn["description"] == "Read file from disk"
        assert "properties" in fn["parameters"]
        assert fn["parameters"]["required"] == ["path"]


# ===========================================================================
# 2. GeminiProvider Adapter Tests
# ===========================================================================


class TestGeminiProvider:
    @pytest.mark.asyncio
    async def test_gemini_generate_success(self) -> None:
        provider = GeminiProvider(api_key="fake-gemini-key", model="gemini-3.8-flash")
        mock_payload = {"candidates": [{"content": {"parts": [{"text": "Hello from Gemini"}]}}]}
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_payload
        mock_resp.raise_for_status.return_value = None

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            result = await provider.generate("Hello")
            assert result == "Hello from Gemini"
            mock_post.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_gemini_generate_with_tools_success(self) -> None:
        provider = GeminiProvider(api_key="fake-gemini-key", model="gemini-3.8-flash")
        mock_payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Calling tool..."},
                            {
                                "functionCall": {
                                    "name": "read_file",
                                    "args": {"path": "config.json"},
                                }
                            },
                        ]
                    }
                }
            ]
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_payload
        mock_resp.raise_for_status.return_value = None

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            text, tool_calls = await provider.generate_with_tools(
                prompt="Read config",
                tools=SAMPLE_MCP_TOOLS,
            )
            assert text == "Calling tool..."
            assert tool_calls is not None
            assert len(tool_calls) == 1
            assert tool_calls[0]["name"] == "read_file"
            assert tool_calls[0]["args"] == {"path": "config.json"}

    @pytest.mark.asyncio
    async def test_gemini_auth_error(self) -> None:
        provider = GeminiProvider(api_key="invalid-key")
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "API_KEY_INVALID"

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            with pytest.raises(AuthenticationError, match="Gemini authentication failed"):
                await provider.generate("Test")


# ===========================================================================
# 3. AnthropicProvider Adapter Tests
# ===========================================================================


class TestAnthropicProvider:
    @pytest.mark.asyncio
    async def test_anthropic_generate_success(self) -> None:
        provider = AnthropicProvider(api_key="fake-anthropic-key")
        mock_payload = {
            "content": [{"type": "text", "text": "Claude here."}],
            "role": "assistant",
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_payload
        mock_resp.raise_for_status.return_value = None

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            result = await provider.generate("Who are you?")
            assert result == "Claude here."
            headers = mock_post.call_args[1]["headers"]
            assert headers["x-api-key"] == "fake-anthropic-key"
            assert headers["anthropic-version"] == "2023-06-01"

    @pytest.mark.asyncio
    async def test_anthropic_generate_with_tools(self) -> None:
        provider = AnthropicProvider(api_key="fake-anthropic-key")
        mock_payload = {
            "content": [
                {"type": "text", "text": "Using read_file tool."},
                {
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "read_file",
                    "input": {"path": "main.py"},
                },
            ]
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_payload
        mock_resp.raise_for_status.return_value = None

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            text, tool_calls = await provider.generate_with_tools(
                prompt="Inspect main.py",
                tools=SAMPLE_MCP_TOOLS,
            )
            assert text == "Using read_file tool."
            assert tool_calls is not None
            assert len(tool_calls) == 1
            assert tool_calls[0]["name"] == "read_file"
            assert tool_calls[0]["args"] == {"path": "main.py"}

    @pytest.mark.asyncio
    async def test_anthropic_auth_failure(self) -> None:
        provider = AnthropicProvider(api_key="bad-key")
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "invalid_api_key"

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            with pytest.raises(AuthenticationError, match="Anthropic authentication failed"):
                await provider.generate("Test")


# ===========================================================================
# 4. OpenAIProvider Adapter Tests
# ===========================================================================


class TestOpenAIProvider:
    @pytest.mark.asyncio
    async def test_openai_generate_success(self) -> None:
        provider = OpenAIProvider(api_key="fake-openai-key", model="gpt-4o")
        mock_payload = {
            "choices": [{"message": {"role": "assistant", "content": "GPT-4o response."}}]
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_payload
        mock_resp.raise_for_status.return_value = None

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            result = await provider.generate("Hello GPT")
            assert result == "GPT-4o response."
            url = mock_post.call_args[0][0]
            assert url == "https://api.openai.com/v1/chat/completions"
            headers = mock_post.call_args[1]["headers"]
            assert headers["Authorization"] == "Bearer fake-openai-key"

    @pytest.mark.asyncio
    async def test_openai_custom_base_url(self) -> None:
        provider = OpenAIProvider(
            api_key="proxy-key",
            base_url="http://litellm-proxy.internal:8000/v1",
        )
        mock_payload = {"choices": [{"message": {"role": "assistant", "content": "Proxy answer"}}]}
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_payload
        mock_resp.raise_for_status.return_value = None

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            res = await provider.generate("Ping")
            assert res == "Proxy answer"
            url = mock_post.call_args[0][0]
            assert url == "http://litellm-proxy.internal:8000/v1/chat/completions"

    @pytest.mark.asyncio
    async def test_openai_generate_with_tools(self) -> None:
        provider = OpenAIProvider(api_key="fake-openai-key")
        mock_payload = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Calling function",
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": "create_record",
                                    "arguments": '{"table": "users", "data": {"name": "Alice"}}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_payload
        mock_resp.raise_for_status.return_value = None

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            text, tool_calls = await provider.generate_with_tools(
                prompt="Add user Alice",
                tools=SAMPLE_MCP_TOOLS,
            )
            assert text == "Calling function"
            assert tool_calls is not None
            assert len(tool_calls) == 1
            assert tool_calls[0]["name"] == "create_record"
            assert tool_calls[0]["args"]["table"] == "users"
            assert tool_calls[0]["args"]["data"]["name"] == "Alice"


# ===========================================================================
# 5. Factory & Provider Resolution Hierarchy Tests
# ===========================================================================


class TestProviderResolution:
    def test_create_provider_unsupported(self) -> None:
        with pytest.raises(ConfigurationError, match="Unsupported LLM provider"):
            create_provider("unsupported_ai")

    def test_hierarchy_cli_flag_priority(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-anthropic-key")
        monkeypatch.setenv("OPENAI_API_KEY", "env-openai-key")

        # CLI flag overrides env vars
        prov, model, key = resolve_active_provider(cli_provider="openai", cli_model="gpt-4o-mini")
        assert prov == "openai"
        assert model == "gpt-4o-mini"
        assert key == "env-openai-key"

    def test_hierarchy_env_var_priority(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "claude-key")

        prov, model, key = resolve_active_provider()
        assert prov == "anthropic"
        assert model == "claude-3-5-sonnet-latest"
        assert key == "claude-key"

    def test_hierarchy_config_file_fallback(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv("HELM_HOME", str(tmp_path))

        save_helm_config({"default_provider": "openai", "default_model": "o3-mini"})
        save_credentials({"openai": "stored-openai-key"})

        prov, model, key = resolve_active_provider()
        assert prov == "openai"
        assert model == "o3-mini"
        assert key == "stored-openai-key"

    def test_model_router_calibrated_tier_models(self) -> None:
        # Anthropic tier models
        router_claude = ModelRouter(provider="anthropic", api_key="test-key")
        assert router_claude.get_model_for_tier(ModelTier.FAST) == "claude-3-5-haiku-latest"
        assert router_claude.get_model_for_tier(ModelTier.STANDARD) == "claude-3-5-sonnet-latest"
        assert router_claude.get_model_for_tier(ModelTier.DEEP) == "claude-3-5-sonnet-latest"

        # OpenAI tier models
        router_openai = ModelRouter(provider="openai", api_key="test-key")
        assert router_openai.get_model_for_tier(ModelTier.FAST) == "gpt-4o-mini"
        assert router_openai.get_model_for_tier(ModelTier.STANDARD) == "gpt-4o"
        assert router_openai.get_model_for_tier(ModelTier.DEEP) == "gpt-4o"

        # Gemini tier models
        router_gemini = ModelRouter(provider="gemini", api_key="test-key")
        assert router_gemini.get_model_for_tier(ModelTier.FAST) == "gemini-3.8-flash"
        assert router_gemini.get_model_for_tier(ModelTier.DEEP) == "gemini-2.5-pro"


# ===========================================================================
# 6. Credentials File Security & Login CLI Tests
# ===========================================================================


class TestAuthCLI:
    def test_credentials_permissions(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("HELM_HOME", str(tmp_path))
        save_credentials({"gemini": "secret-gemini-token"})

        creds = load_credentials()
        assert creds["gemini"] == "secret-gemini-token"

        creds_file = tmp_path / "credentials"
        assert creds_file.exists()
        mode = stat.S_IMODE(creds_file.stat().st_mode)
        # Verify chmod 0600 (owner read/write only)
        assert mode == 0o600

    def test_login_flow_interactive(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("HELM_HOME", str(tmp_path))
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        # Mock interactive answers: provider=3 (openai), base_url="", api_key=sk-test-123, model=gpt-4o
        answers = iter(["3", "", "sk-test-123", "gpt-4o"])
        with patch("rich.prompt.Prompt.ask", side_effect=lambda *args, **kwargs: next(answers)):
            run_login_flow()

        cfg = load_helm_config()
        assert cfg["default_provider"] == "openai"
        assert cfg["default_model"] == "gpt-4o"

        creds = load_credentials()
        assert creds["openai"] == "sk-test-123"

    def test_cli_help_shows_login_and_config(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "login" in result.output
        assert "config" in result.output
        assert "--provider" in runner.invoke(app, ["run", "--help"]).output
