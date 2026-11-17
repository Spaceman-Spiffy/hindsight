"""
Nous token refresh delegated to Hermes (fork-local, not upstream).

Upstream's :class:`NousAuthManager` refreshes natively: it redeems the single-use
``refresh_token`` itself and patches ``providers.nous`` in ``~/.hermes/auth.json``.
That is correct when Hindsight is the only process redeeming the grant. On a host
where a Hermes gateway uses the same Nous login, two behaviours combine badly:

* the native manager redeems the refresh token whenever *its own* in-memory access
  token is stale, even when a peer has just rotated and a fresh pair is on disk; and
* Hermes' credential pool copies its pair back into ``providers.nous`` after its
  refresh transaction has released ``auth.lock``, with no freshness check.

A second rotation that lands in that gap is overwritten by the pool's older pair.
The next refresh then replays a redeemed token and the Portal revokes the session,
which only an interactive ``hermes auth add nous`` repairs.

This manager never talks to the Portal and never writes ``auth.json``. Every
refresh goes through ``hermes_cli.auth.resolve_nous_runtime_credentials``, so Hermes
stays the only redeemer and keeps ``providers.nous``, its pool copy and the shared
store in step. A pair a peer already rotated is adopted, never re-spent.

It also starts without a usable login: the token is resolved on first use, so a
re-login is picked up without restarting the server.

Requires ``hermes_cli`` to be importable (run the server from the Hermes venv).
Set ``HINDSIGHT_API_NOUS_REFRESH=native`` to use upstream's manager instead.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from pathlib import Path
from typing import Any

from .nous_auth import (
    _NOUS_TOKEN_REFRESH_SKEW_SECONDS,
    NousAuthManager,
    NousNotLoggedInError,
    NousRefreshExpiredError,
    _default_auth_file,
)

logger = logging.getLogger(__name__)

ENV_NOUS_REFRESH = "HINDSIGHT_API_NOUS_REFRESH"

#: Bearer held until Hermes resolves a real one. Never valid upstream; always stale here.
PENDING_TOKEN = "nous-token-pending-hermes-resolve"


def nous_auth_manager_class() -> type[NousAuthManager]:
    """The manager ``NousLLM`` should build: delegating unless explicitly set to native."""
    if os.environ.get(ENV_NOUS_REFRESH, "").strip().lower() == "native":
        return NousAuthManager
    return HermesDelegatingNousAuthManager


def _hermes_resolver() -> Any:
    """Return Hermes' ``resolve_nous_runtime_credentials``.

    Imported lazily, per call, so an import that fails once (Hermes mid-update)
    is retried on the next refresh rather than cached as a permanent failure.
    """
    try:
        from hermes_cli.auth import resolve_nous_runtime_credentials
    except Exception as e:  # environment dependent: any import-time failure
        raise RuntimeError(
            "Nous token refresh is delegated to Hermes, but hermes_cli.auth could not be "
            f"imported ({type(e).__name__}). Run the server from the Hermes virtualenv, or set "
            f"{ENV_NOUS_REFRESH}=native to refresh without Hermes."
        ) from e
    return resolve_nous_runtime_credentials


class HermesDelegatingNousAuthManager(NousAuthManager):
    """A :class:`NousAuthManager` whose refresh is Hermes' resolver, not a Portal request."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Held in the worker thread, never across an ``await``. Not ``oauth_store_lock``:
        # that takes the flock on ``auth.lock``, which Hermes' resolver takes itself through
        # another file description, and the two would wait on each other until timeout.
        self._delegate_lock = threading.Lock()
        logger.info("Nous token refresh is delegated to the Hermes resolver")

    @classmethod
    def from_file(cls, auth_file: Path | None = None) -> "HermesDelegatingNousAuthManager":
        """Like upstream's, but a missing login is deferred to first use instead of raised."""
        try:
            return super().from_file(auth_file)  # type: ignore[return-value]
        except NousNotLoggedInError as e:
            logger.warning(f"No usable Nous login at startup ({e}). Resolving through Hermes on first use.")
            return cls(
                access_token=PENDING_TOKEN,
                refresh_token=None,
                auth_file=auth_file if auth_file is not None else _default_auth_file(),
            )

    def _token_is_stale(self, skew_seconds: int = _NOUS_TOKEN_REFRESH_SKEW_SECONDS) -> bool:
        if not self.access_token or self.access_token == PENDING_TOKEN:
            return True
        return super()._token_is_stale(skew_seconds)

    async def refresh_tokens(self, reason: str = "", *, force: bool = False) -> None:
        """Resolve the bearer through Hermes. Single-flight per manager.

        Raises
        ------
        NousRefreshExpiredError:
            Hermes reports that the login itself must be redone.
        RuntimeError:
            Any other failure (Hermes not importable, network, lock timeout).
        """
        await asyncio.to_thread(self._resolve_through_hermes, reason, force, self.access_token)

    def _resolve_through_hermes(self, reason: str, force: bool, token_before: str) -> None:
        with self._delegate_lock:
            if force:
                if self.access_token != token_before:
                    return  # another caller already resolved while we waited
            elif not self._token_is_stale():
                return

            resolve = _hermes_resolver()
            kwargs: dict[str, Any] = {"force_refresh": force}
            if force and token_before and token_before != PENDING_TOKEN:
                # The bearer that just failed. Hermes skips the Portal request when the
                # store already holds a different usable one (a peer rotated first).
                kwargs["stale_access_token"] = token_before

            log_reason = f" ({reason})" if reason else ""
            logger.info(f"Resolving Nous Portal access_token through Hermes{log_reason}")
            try:
                creds = resolve(**kwargs)
            except Exception as e:
                if getattr(e, "relogin_required", False):
                    raise NousRefreshExpiredError(
                        f"Hermes reports the Nous login is no longer valid ({e}). "
                        "Run 'hermes auth add nous' to re-authenticate."
                    ) from e
                raise RuntimeError(f"Nous credential resolution through Hermes failed: {type(e).__name__}: {e}") from e

            token = creds.get("api_key") if isinstance(creds, dict) else None
            if not isinstance(token, str) or not token:
                raise RuntimeError("Hermes resolved Nous credentials without an inference token")
            self.access_token = token
