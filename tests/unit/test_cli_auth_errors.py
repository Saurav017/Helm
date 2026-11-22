"""
tests/unit/test_cli_auth_errors.py — Tests for dynamic auth prompting and domain error handling.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from helm.cli.app import _preflight_auth_check, app
from helm.config.settings import Settings
from helm.core.circuit_breaker import CircuitBreakerTripped
from helm.core.state import CircuitBlockReason, SessionStatus
from helm.exceptions import AuthenticationError
from helm.protocols.transports.base import RemoteTransportError

runner = CliRunner()


class TestDynamicAuthPrompting:
    """Test dynamic authentication prompting and non-interactive handling on missing keys."""

    @pytest.fixture(autouse=True)
    def isolate_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Clear provider API keys and isolate settings."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(
            "helm.cli.app.load_settings",
            lambda config_path=None: Settings(mcp_servers=[]),
        )
        monkeypatch.setattr(
            "helm.config.auth.load_credentials",
            dict,
        )

    def test_non_interactive_missing_key_gemini_exits_cleanly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """In non-interactive environments, missing keys output single-line error and exit 1."""
        monkeypatch.setenv("HELM_FORCE_AUTH_CHECK", "1")
        monkeypatch.setattr("helm.cli.app._is_interactive", lambda: False)

        result = runner.invoke(app, ["run", "test task"])
        assert result.exit_code == 1
        expected_line = (
            "Error: No API key configured for provider 'gemini'. "
            "Export GEMINI_API_KEY or run 'helmx login'."
        )
        assert expected_line in result.output
        assert "Traceback (most recent call last)" not in result.output

    def test_non_interactive_missing_key_anthropic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """In non-interactive environments with --provider anthropic, points to ANTHROPIC_API_KEY."""
        monkeypatch.setenv("HELM_FORCE_AUTH_CHECK", "1")
        monkeypatch.setattr("helm.cli.app._is_interactive", lambda: False)

        result = runner.invoke(app, ["run", "--provider", "anthropic", "test task"])
        assert result.exit_code == 1
        expected_line = (
            "Error: No API key configured for provider 'anthropic'. "
            "Export ANTHROPIC_API_KEY or run 'helmx login'."
        )
        assert expected_line in result.output
        assert "Traceback (most recent call last)" not in result.output

    def test_non_interactive_missing_key_openai(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """In non-interactive environments with --provider openai, points to OPENAI_API_KEY."""
        monkeypatch.setenv("HELM_FORCE_AUTH_CHECK", "1")
        monkeypatch.setattr("helm.cli.app._is_interactive", lambda: False)

        result = runner.invoke(app, ["run", "--provider", "openai", "test task"])
        assert result.exit_code == 1
        expected_line = (
            "Error: No API key configured for provider 'openai'. "
            "Export OPENAI_API_KEY or run 'helmx login'."
        )
        assert expected_line in result.output
        assert "Traceback (most recent call last)" not in result.output

    def test_interactive_declined_login_exits_code_1_without_traceback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Interactive terminal prompts user to log in; declining cleanly exits with code 1."""
        monkeypatch.setenv("HELM_FORCE_AUTH_CHECK", "1")
        monkeypatch.setattr("helm.cli.app._is_interactive", lambda: True)

        # User enters 'n'
        result = runner.invoke(app, ["run", "test task"], input="n\n")
        assert result.exit_code == 1
        assert "No API key configured. Would you like to log in now? [Y/n]" in result.output
        assert "Traceback (most recent call last)" not in result.output

    def test_interactive_accepted_login_proceeds_to_goal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Interactive terminal prompts user to log in; confirming runs login flow and retries."""
        monkeypatch.setenv("HELM_FORCE_AUTH_CHECK", "1")
        monkeypatch.setattr("helm.cli.app._is_interactive", lambda: True)

        login_flow_called = False

        def mock_login_flow(*_args: object, **_kwargs: object) -> None:
            nonlocal login_flow_called
            login_flow_called = True
            # Simulate key being configured
            monkeypatch.setattr(
                "helm.cli.app.resolve_active_provider",
                lambda **_kw: ("gemini", "gemini-3.8-flash", "test-key"),
            )
            monkeypatch.delenv("HELM_FORCE_AUTH_CHECK", raising=False)

        monkeypatch.setattr("helm.cli.app.run_login_flow", mock_login_flow)

        # Mock supervisor session completion
        async def mock_supervisor(state: object, _settings: object, **_kwargs: object) -> object:
            if hasattr(state, "transition"):
                state.transition(SessionStatus.COMPLETED)
                state.final_answer = "Goal completed successfully!"
            return state

        monkeypatch.setattr("helm.cli.app._run_supervisor_session", mock_supervisor)

        # User presses Enter ('\n') which defaults to 'y'
        result = runner.invoke(app, ["run", "test task"], input="\n")
        assert result.exit_code == 0
        assert login_flow_called is True
        assert "Goal completed successfully!" in result.output
        assert "Traceback (most recent call last)" not in result.output

    def test_runtime_401_interactive_prompt_and_decline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On Runtime 401, interactive terminal asks to re-enter key; declining exits cleanly."""
        monkeypatch.setattr("helm.cli.app._is_interactive", lambda: True)
        monkeypatch.setattr(
            "helm.cli.app.resolve_active_provider",
            lambda **_kw: ("gemini", "gemini-3.8-flash", "bad-key"),
        )

        async def fail_with_401(*_args: object, **_kwargs: object) -> None:
            raise AuthenticationError("Gemini authentication failed (401): Invalid API key")

        monkeypatch.setattr("helm.cli.app._run_supervisor_session", fail_with_401)

        result = runner.invoke(app, ["run", "test task"], input="n\n")
        assert result.exit_code == 1
        assert (
            "API key rejected by provider. Would you like to re-enter your key now? [Y/n]"
            in result.output
        )
        assert "Traceback (most recent call last)" not in result.output

    def test_runtime_401_interactive_prompt_and_retry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On Runtime 401, confirming prompts for new key and proceeds with execution."""
        monkeypatch.setattr("helm.cli.app._is_interactive", lambda: True)
        monkeypatch.setattr(
            "helm.cli.app.resolve_active_provider",
            lambda **_kw: ("gemini", "gemini-3.8-flash", "bad-key"),
        )

        login_called = False

        def mock_login(*_args: object, **_kwargs: object) -> bool:
            nonlocal login_called
            login_called = True
            # Update provider to now return good key
            monkeypatch.setattr(
                "helm.cli.app.resolve_active_provider",
                lambda **_kw: ("gemini", "gemini-3.8-flash", "good-key"),
            )
            return True

        monkeypatch.setattr("helm.cli.app.run_login_flow", mock_login)

        calls = 0

        async def mock_supervisor(state: object, _settings: object, **_kwargs: object) -> object:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise AuthenticationError("Gemini authentication failed (401): Invalid API key")
            if hasattr(state, "transition"):
                state.transition(SessionStatus.COMPLETED)
                state.final_answer = "Resumed and completed after re-login!"
            return state

        monkeypatch.setattr("helm.cli.app._run_supervisor_session", mock_supervisor)

        result = runner.invoke(app, ["run", "test task"], input="y\n")
        assert result.exit_code == 0
        assert login_called is True
        assert "Resumed and completed after re-login!" in result.output


class TestEliminateRawTracebacks:
    """Ensure known domain errors are rendered in Rich panels without raw tracebacks."""

    def test_transport_error_renders_panel_without_traceback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "helm.cli.app.load_settings",
            lambda config_path=None: Settings(mcp_servers=[]),
        )

        async def fail_with_transport(*_args: object, **_kwargs: object) -> None:
            raise RemoteTransportError("Connection refused by remote SSE server")

        monkeypatch.setattr("helm.cli.app._run_supervisor_session", fail_with_transport)

        result = runner.invoke(app, ["run", "test task"])
        assert result.exit_code == 1
        assert "RemoteTransportError" in result.output
        assert "Connection refused by remote SSE server" in result.output
        assert "Traceback (most recent call last)" not in result.output

    def test_circuit_breaker_tripped_renders_panel_without_traceback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "helm.cli.app.load_settings",
            lambda config_path=None: Settings(mcp_servers=[]),
        )

        async def fail_with_cb(*_args: object, **_kwargs: object) -> None:
            raise CircuitBreakerTripped(
                reason=CircuitBlockReason.STEP_CEILING,
                tool_name="delete_entities",
                call_hash="a" * 64,
                detail="Circuit breaker step ceiling exceeded",
            )

        monkeypatch.setattr("helm.cli.app._run_supervisor_session", fail_with_cb)

        result = runner.invoke(app, ["run", "test task"])
        assert result.exit_code == 1
        assert "Circuit breaker tripped" in result.output
        assert "delete_entities" in result.output
        assert "step_ceiling" in result.output
        assert "Traceback (most recent call last)" not in result.output

    def test_preflight_auth_check_directly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HELM_FORCE_AUTH_CHECK", raising=False)
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "active")
        # In test mode without force check, preflight passes
        _preflight_auth_check()

        # With force check and no key, preflight raises AuthenticationError
        monkeypatch.setenv("HELM_FORCE_AUTH_CHECK", "1")
        monkeypatch.setattr(
            "helm.cli.app.resolve_active_provider",
            lambda **_kw: ("openai", "gpt-4o", None),
        )
        with pytest.raises(AuthenticationError) as exc_info:
            _preflight_auth_check(provider_arg="openai")
        assert "No API key configured for provider 'openai'" in str(exc_info.value)
        assert "OPENAI_API_KEY" in str(exc_info.value)


class TestPreflightValidationInLogin:
    """Test pre-flight key validation during login flow."""

    def test_validate_api_key_gemini(self) -> None:
        from helm.cli.auth import validate_api_key

        with patch("httpx.Client.get") as mock_get:
            # 200 Success
            mock_get.return_value = MagicMock(status_code=200)
            valid, _ = validate_api_key("gemini", "good-key")
            assert valid is True

            # 400 API_KEY_INVALID
            mock_get.return_value = MagicMock(status_code=400, text="API_KEY_INVALID")
            valid, err = validate_api_key("gemini", "bad-key")
            assert valid is False
            assert "Invalid API key" in err

            # 401
            mock_get.return_value = MagicMock(status_code=401, text="Unauthorized")
            valid, err = validate_api_key("gemini", "bad-key")
            assert valid is False

    def test_validate_api_key_openai(self) -> None:
        from helm.cli.auth import validate_api_key

        with patch("httpx.Client.get") as mock_get:
            # 200 Success
            mock_get.return_value = MagicMock(status_code=200)
            valid, _ = validate_api_key("openai", "sk-good")
            assert valid is True

            # 401 Invalid Key
            mock_get.return_value = MagicMock(status_code=401, text="invalid_api_key")
            valid, err = validate_api_key("openai", "sk-bad")
            assert valid is False
            assert "Invalid API key" in err

    def test_validate_api_key_openai_localhost_and_127_0_0_1(self) -> None:
        from helm.cli.auth import validate_api_key

        with patch("httpx.Client.get") as mock_get:
            # 200 Success against localhost probe
            mock_get.return_value = MagicMock(status_code=200)
            valid, _ = validate_api_key("openai", "ollama", base_url="http://localhost:11434/v1")
            assert valid is True
            mock_get.assert_called_with(
                "http://localhost:11434/v1/models",
                headers={"Authorization": "Bearer ollama"},
            )

            # 401 against localhost is skipped (treated as valid)
            mock_get.return_value = MagicMock(status_code=401, text="unauthorized")
            valid, _ = validate_api_key("openai", "ollama", base_url="http://localhost:11434/v1")
            assert valid is True

            # 401 against 127.0.0.1 is skipped (treated as valid)
            mock_get.return_value = MagicMock(status_code=401, text="unauthorized")
            valid, _ = validate_api_key(
                "openai", "placeholder", base_url="http://127.0.0.1:8000/v1"
            )
            assert valid is True

            # 401 against public openai is NOT skipped
            mock_get.return_value = MagicMock(status_code=401, text="invalid_api_key")
            valid, err = validate_api_key("openai", "bad-key", base_url="https://api.openai.com/v1")
            assert valid is False
            assert "Invalid API key" in err

    def test_validate_api_key_anthropic(self) -> None:
        from helm.cli.auth import validate_api_key

        with patch("httpx.Client.post") as mock_post:
            # 200 Success
            mock_post.return_value = MagicMock(status_code=200)
            valid, _ = validate_api_key("anthropic", "ant-good")
            assert valid is True

            # 401 Unauthorized
            mock_post.return_value = MagicMock(status_code=401, text="authentication_error")
            valid, err = validate_api_key("anthropic", "ant-bad")
            assert valid is False
            assert "Invalid API key" in err

    def test_login_flow_validation_failure_cancel(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from helm.cli.auth import run_login_flow
        from helm.config.auth import load_credentials

        monkeypatch.setenv("HELM_HOME", str(tmp_path))
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        monkeypatch.setattr(
            "helm.cli.auth.validate_api_key",
            lambda *args, **kwargs: (False, "HTTP 401: Invalid API key."),
        )

        answers = iter(["1", "bad-key", "cancel"])
        with patch("rich.prompt.Prompt.ask", side_effect=lambda *args, **kwargs: next(answers)):
            success = run_login_flow()
            assert success is False

        # Credentials must NOT be saved
        creds = load_credentials()
        assert "gemini" not in creds

    def test_login_flow_validation_failure_re_enter_success(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from helm.cli.auth import run_login_flow
        from helm.config.auth import load_credentials

        monkeypatch.setenv("HELM_HOME", str(tmp_path))
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        validation_attempts = 0

        def mock_validate(*_args: object, **_kwargs: object) -> tuple[bool, str]:
            nonlocal validation_attempts
            validation_attempts += 1
            if validation_attempts == 1:
                return False, "HTTP 401: Invalid API key."
            return True, ""

        monkeypatch.setattr("helm.cli.auth.validate_api_key", mock_validate)

        # Answers: provider=1 (gemini), api_key=bad-key, retry=re-enter, api_key=good-key, model=gemini-3.8-flash
        answers = iter(["1", "bad-key", "re-enter", "good-key", "gemini-3.8-flash"])
        with patch("rich.prompt.Prompt.ask", side_effect=lambda *args, **kwargs: next(answers)):
            success = run_login_flow()
            assert success is True

        # Credentials must have the good key
        creds = load_credentials()
        assert creds["gemini"] == "good-key"

    def test_login_flow_openai_ollama_base_url_first(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When OpenAI is selected, base_url is prompted first and passed to probe/credentials."""
        from helm.cli.auth import run_login_flow
        from helm.config.auth import load_credentials, load_helm_config

        monkeypatch.setenv("HELM_HOME", str(tmp_path))
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        probed_urls: list[str] = []

        def mock_validate(
            _provider: str, _api_key: str, base_url: str | None = None
        ) -> tuple[bool, str]:
            if base_url:
                probed_urls.append(base_url)
            return True, ""

        monkeypatch.setattr("helm.cli.auth.validate_api_key", mock_validate)

        # Prompt order: 1. provider=3 (openai), 2. base_url="http://localhost:11434/v1",
        #               3. api_key="ollama", 4. model="llama3.2"
        answers = iter(["3", "http://localhost:11434/v1", "ollama", "llama3.2"])
        with patch("rich.prompt.Prompt.ask", side_effect=lambda *args, **kwargs: next(answers)):
            success = run_login_flow()
            assert success is True

        assert "http://localhost:11434/v1" in probed_urls

        cfg = load_helm_config()
        assert cfg["default_provider"] == "openai"
        assert cfg["default_model"] == "llama3.2"
        assert cfg["openai_base_url"] == "http://localhost:11434/v1"

        creds = load_credentials()
        assert creds["openai"] == "ollama"

    def test_login_flow_openai_localhost_skip_validation(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """On localhost, if validation fails, user can skip auth validation and proceed."""
        from helm.cli.auth import run_login_flow
        from helm.config.auth import load_credentials, load_helm_config

        monkeypatch.setenv("HELM_HOME", str(tmp_path))
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        # Mock validate to fail
        monkeypatch.setattr(
            "helm.cli.auth.validate_api_key",
            lambda *args, **kwargs: (False, "HTTP 401: Unauthorized"),
        )

        # Prompt order: 1. provider=3, 2. base_url="http://localhost:11434/v1",
        #               3. api_key="my-local-key", 4. retry_choice="skip", 5. model="mistral"
        answers = iter(["3", "http://localhost:11434/v1", "my-local-key", "skip", "mistral"])
        with patch("rich.prompt.Prompt.ask", side_effect=lambda *args, **kwargs: next(answers)):
            success = run_login_flow()
            assert success is True

        cfg = load_helm_config()
        assert cfg["default_model"] == "mistral"
        assert cfg["openai_base_url"] == "http://localhost:11434/v1"

        creds = load_credentials()
        assert creds["openai"] == "my-local-key"
