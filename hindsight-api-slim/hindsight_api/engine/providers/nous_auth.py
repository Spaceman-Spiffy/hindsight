"""
Nous Portal authentication manager for Hindsight.

The Nous Portal inference endpoint (https://inference-api.nousresearch.com/v1)
authenticates with a short-lived ``inference:invoke`` JWT that must be refreshed
periodically — a static API key is not available. This mirrors the pattern in
``codex_auth.py`` (proactive refresh before expiry, reactive refresh on 401),
but instead of re-implementing the OAuth dance it delegates to Hermes' own
``resolve_nous_runtime_credentials()``, which already handles the refresh,
file-locking against the shared auth store, and atomic persistence.

This keeps a single source of truth for Nous credentials: whatever the Hermes
agent uses, the Hindsight daemon uses too, because both call the same resolver
against ``~/.hermes/auth.json``.

Usage::

    mgr = NousAuthManager()
    token = mgr.ensure_fresh_token()          # proactive; refreshes if near expiry
    ...                                         # use token as Bearer
    token = mgr.refresh(force=True)            # reactive, on a 401
"""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)

# Refresh this many seconds before the JWT's expiry to avoid edge races where a
# request leaves the client just as the token lapses.
_REFRESH_SKEW_SECONDS = 120

# Default Nous inference base URL — overridable via the resolver's return value.
DEFAULT_NOUS_BASE_URL = "https://inference-api.nousresearch.com/v1"


class NousAuthError(RuntimeError):
    """Raised when a fresh Nous credential cannot be resolved."""


class NousAuthManager:
    """Resolves and caches a fresh Nous inference JWT via Hermes' resolver.

    Thread-safe: a single lock guards refreshes so concurrent consolidation
    batches don't stampede the resolver (which itself takes an auth-store file
    lock, but in-process serialization avoids redundant work).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._token: str | None = None
        self._base_url: str = DEFAULT_NOUS_BASE_URL
        self._expires_at: float = 0.0

    def _resolve(self, *, force_refresh: bool) -> dict:
        """Call Hermes' resolver. Imported lazily so the daemon only needs the
        Hermes path importable when the Nous provider is actually used."""
        try:
            from hermes_cli.auth import resolve_nous_runtime_credentials
        except Exception as exc:  # pragma: no cover - environment dependent
            raise NousAuthError(
                "Cannot import hermes_cli.auth.resolve_nous_runtime_credentials; "
                "the Nous provider requires the Hermes package on PYTHONPATH."
            ) from exc
        try:
            return resolve_nous_runtime_credentials(force_refresh=force_refresh)
        except Exception as exc:
            raise NousAuthError(f"Nous credential resolution failed: {exc}") from exc

    def _store(self, creds: dict) -> str:
        token = creds.get("api_key")
        if not token:
            raise NousAuthError("Resolver returned no api_key for Nous.")
        self._token = token
        base = creds.get("base_url")
        if base:
            self._base_url = base.rstrip("/")
        # expires_at may be an epoch float or absent; normalize defensively.
        exp = creds.get("expires_at")
        try:
            self._expires_at = float(exp) if exp is not None else time.time() + 600
        except (TypeError, ValueError):
            self._expires_at = time.time() + 600
        return token

    @property
    def base_url(self) -> str:
        return self._base_url

    def ensure_fresh_token(self) -> str:
        """Return a valid token, refreshing proactively if near/at expiry."""
        with self._lock:
            now = time.time()
            if self._token and now < (self._expires_at - _REFRESH_SKEW_SECONDS):
                return self._token
            creds = self._resolve(force_refresh=False)
            return self._store(creds)

    def refresh(self, *, force: bool = True) -> str:
        """Force a refresh (e.g. after a reactive 401) and return the new token."""
        with self._lock:
            creds = self._resolve(force_refresh=force)
            token = self._store(creds)
            logger.info("Nous inference token refreshed (reactive/forced).")
            return token
