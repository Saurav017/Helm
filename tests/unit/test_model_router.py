"""
tests/unit/test_model_router.py — Unit tests for Tiered Model Router.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from helm.config.settings import Settings
from helm.core.model_router import ModelRouter
from helm.core.state import ModelTier


class TestModelRouter:
    def test_routing_classification(self):
        router = ModelRouter()

        assert (
            router.route("Design overall system architecture and migration plan") == ModelTier.DEEP
        )
        assert router.route("Implement JWT authentication endpoint") == ModelTier.STANDARD
        assert router.route("Check if user exists and return yes or no") == ModelTier.FAST

    def test_get_model_for_tier(self):
        settings = Settings()
        settings.models.fast = "gemini-2.0-flash"
        settings.models.deep = "gemini-2.5-pro"

        router = ModelRouter(settings=settings)
        assert router.get_model_for_tier(ModelTier.FAST) == "gemini-2.0-flash"
        assert router.get_model_for_tier(ModelTier.DEEP) == "gemini-2.5-pro"
        assert router.get_model_for_tier("fast") == "gemini-2.0-flash"

    @pytest.mark.asyncio
    async def test_generate_offline_pong(self):
        router = ModelRouter(api_key="")
        res = await router.generate("Respond with PONG")
        assert res == "PONG"

    @pytest.mark.asyncio
    async def test_generate_with_gemini_api_mock(self):
        router = ModelRouter(api_key="mock-key")

        mock_payload = {"candidates": [{"content": {"parts": [{"text": "PONG"}]}}]}

        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_payload
        mock_resp.raise_for_status.return_value = None

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            res = await router.generate("Say PONG")
            assert res == "PONG"
            mock_post.assert_awaited_once()
