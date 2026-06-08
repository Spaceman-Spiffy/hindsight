"""
Nous Portal LLM provider for Hindsight.

Thin wrapper over :class:`OpenAICompatibleLLM`. The Nous Portal speaks the
OpenAI chat-completions wire format, so all request/response handling is
inherited unchanged. The ONLY thing Nous needs on top is a rotating
``inference:invoke`` bearer token (there is no static API key), refreshed via
:class:`NousAuthManager` before each outbound call and reactively on a 401.

Configure with::

    llm_provider = "nous"
    llm_base_url = "https://inference-api.nousresearch.com/v1"   # or omit; resolver supplies it
    llm_model    = "deepseek/deepseek-v4-flash"                  # any Nous-hosted slug

No API key is required in config — the token is resolved dynamically from the
shared Hermes auth store.
"""

from __future__ import annotations

import logging
from typing import Any

from openai import APIStatusError, AsyncOpenAI

from hindsight_api.engine.providers.nous_auth import NousAuthManager
from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM

logger = logging.getLogger(__name__)


class NousLLM(OpenAICompatibleLLM):
    """OpenAI-compatible provider for the Nous Portal with rotating-JWT auth."""

    def __init__(
        self,
        provider: str,
        api_key: str,
        base_url: str,
        model: str,
        reasoning_effort: str = "low",
        **kwargs: Any,
    ):
        self._auth = NousAuthManager()
        # Obtain an initial token so the parent's client builds with valid auth.
        token = self._auth.ensure_fresh_token()
        resolved_base = base_url or self._auth.base_url
        # Parent validates provider against a fixed list; present as "openai"
        # (identical wire format) while we retain the true identity for logs.
        super().__init__(
            provider="openai",
            api_key=token,
            base_url=resolved_base,
            model=model,
            reasoning_effort=reasoning_effort,
            **kwargs,
        )
        self._nous_provider_name = provider
        logger.info(
            "Nous LLM initialized: model=%s base_url=%s (rotating inference:invoke JWT)",
            self.model, self.base_url,
        )

    def _apply_fresh_token(self) -> None:
        """Refresh the token if near expiry and rebuild the SDK client if it changed."""
        token = self._auth.ensure_fresh_token()
        if token != self.api_key:
            self.api_key = token
            self._client = AsyncOpenAI(
                api_key=token, base_url=self.base_url,
                max_retries=0, timeout=self.timeout,
            )

    def _force_token(self) -> None:
        """Hard refresh after a 401 and rebuild the client."""
        token = self._auth.refresh(force=True)
        self.api_key = token
        self._client = AsyncOpenAI(
            api_key=token, base_url=self.base_url,
            max_retries=0, timeout=self.timeout,
        )

    async def verify_connection(self) -> None:
        self._apply_fresh_token()
        return await super().verify_connection()

    async def call(self, *args: Any, **kwargs: Any):
        self._apply_fresh_token()
        try:
            return await super().call(*args, **kwargs)
        except APIStatusError as e:
            if getattr(e, "status_code", None) == 401:
                logger.warning("Nous 401 — forcing token refresh and retrying once.")
                self._force_token()
                return await super().call(*args, **kwargs)
            raise

    async def call_with_tools(self, *args: Any, **kwargs: Any):
        self._apply_fresh_token()
        try:
            return await super().call_with_tools(*args, **kwargs)
        except APIStatusError as e:
            if getattr(e, "status_code", None) == 401:
                logger.warning("Nous 401 (tools) — forcing token refresh and retrying once.")
                self._force_token()
                return await super().call_with_tools(*args, **kwargs)
            raise
