"""
Tests for the fork-local delegating Nous auth manager (``nous_hermes_delegate``).

The manager must never redeem the refresh token itself and never write the Hermes
auth store: every refresh is one call to Hermes' resolver. A stand-in
``hermes_cli.auth`` module records those calls.

Self-contained (no database, no network):

    pytest --noconftest -o asyncio_mode=auto tests/test_nous_hermes_delegate.py
"""

import asyncio
import base64
import fcntl
import json
import sys
import time
import types
from pathlib import Path
from typing import Any

import aiohttp
import pytest

from hindsight_api.engine.providers import nous_hermes_delegate as delegate
from hindsight_api.engine.providers.nous_auth import NousAuthManager, NousRefreshExpiredError
from hindsight_api.engine.providers.nous_hermes_delegate import (
    PENDING_TOKEN,
    HermesDelegatingNousAuthManager,
    nous_auth_manager_class,
)


def _jwt_with_exp(exp_unixtime: int, marker: str = "") -> str:
    def b64(obj: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).decode().rstrip("=")

    return f"{b64({'alg': 'none'})}.{b64({'exp': exp_unixtime, 'm': marker})}.sig"


def _fresh(marker: str = "") -> str:
    return _jwt_with_exp(int(time.time()) + 3600, marker)


def _stale(marker: str = "") -> str:
    return _jwt_with_exp(int(time.time()) - 10, marker)


def _write_store(path: Path, access_token: str | None, refresh_token: str = "rt-on-disk") -> None:
    state: dict[str, Any] = {"refresh_token": refresh_token, "client_id": "hermes-cli"}
    if access_token is not None:
        state["access_token"] = access_token
    path.write_text(json.dumps({"version": 1, "providers": {"nous": state}, "credential_pool": {"nous": []}}))


class _Resolver:
    """Stand-in for ``hermes_cli.auth.resolve_nous_runtime_credentials``."""

    def __init__(self, reply: Any, *, lock_path: Path | None = None, delay: float = 0.0) -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []
        self.lock_path = lock_path
        self.lock_was_free: list[bool] = []
        self.delay = delay

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.lock_path is not None:
            # Hermes takes this flock itself. It must be free while the resolver runs.
            with open(self.lock_path, "a+") as fh:
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self.lock_was_free.append(True)
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                except OSError:
                    self.lock_was_free.append(False)
        if self.delay:
            time.sleep(self.delay)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


@pytest.fixture
def hermes(monkeypatch: pytest.MonkeyPatch):
    """Install a fake ``hermes_cli.auth`` and return a setter for its resolver."""

    def install(resolver: Any) -> Any:
        pkg = types.ModuleType("hermes_cli")
        mod = types.ModuleType("hermes_cli.auth")
        mod.resolve_nous_runtime_credentials = resolver  # type: ignore[attr-defined]
        pkg.auth = mod  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "hermes_cli", pkg)
        monkeypatch.setitem(sys.modules, "hermes_cli.auth", mod)
        return resolver

    return install


@pytest.fixture(autouse=True)
def no_portal(monkeypatch: pytest.MonkeyPatch):
    """Any HTTP request from the manager is a failure: it must not reach the Portal."""

    def refuse(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("the delegating manager opened an HTTP request")

    monkeypatch.setattr(aiohttp.ClientSession, "_request", refuse)


def test_default_is_delegating_and_native_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(delegate.ENV_NOUS_REFRESH, raising=False)
    assert nous_auth_manager_class() is HermesDelegatingNousAuthManager
    monkeypatch.setenv(delegate.ENV_NOUS_REFRESH, "hermes")
    assert nous_auth_manager_class() is HermesDelegatingNousAuthManager
    monkeypatch.setenv(delegate.ENV_NOUS_REFRESH, "Native")
    assert nous_auth_manager_class() is NousAuthManager


async def test_fresh_token_never_calls_hermes(tmp_path: Path, hermes) -> None:
    auth = tmp_path / "auth.json"
    token = _fresh("a")
    _write_store(auth, token)
    resolver = hermes(_Resolver({"api_key": "unused"}))
    mgr = HermesDelegatingNousAuthManager.from_file(auth)
    assert await mgr.ensure_fresh_token() == token
    assert resolver.calls == []


async def test_stale_token_resolves_through_hermes_without_forcing(tmp_path: Path, hermes) -> None:
    auth = tmp_path / "auth.json"
    _write_store(auth, _stale("old"))
    before = auth.read_bytes()
    new = _fresh("new")
    resolver = hermes(_Resolver({"api_key": new, "base_url": "https://x/v1"}, lock_path=tmp_path / "auth.lock"))
    mgr = HermesDelegatingNousAuthManager.from_file(auth)

    assert await mgr.ensure_fresh_token() == new
    assert resolver.calls == [{"force_refresh": False}]
    assert resolver.lock_was_free == [True], "auth.lock was held across the Hermes call"
    assert auth.read_bytes() == before, "the manager wrote the Hermes auth store"
    assert not list(tmp_path.glob(".auth.*")), "a temp store file was left behind"


async def test_reactive_refresh_passes_the_failed_bearer(tmp_path: Path, hermes) -> None:
    auth = tmp_path / "auth.json"
    old = _fresh("old")
    _write_store(auth, old)
    new = _fresh("new")
    resolver = hermes(_Resolver({"api_key": new}))
    mgr = HermesDelegatingNousAuthManager.from_file(auth)

    await mgr.refresh_tokens("reactive (HTTP 401 on call)", force=True)
    assert resolver.calls == [{"force_refresh": True, "stale_access_token": old}]
    assert mgr.access_token == new


async def test_concurrent_refreshes_make_one_hermes_call(tmp_path: Path, hermes) -> None:
    auth = tmp_path / "auth.json"
    _write_store(auth, _stale("old"))
    new = _fresh("new")
    resolver = hermes(_Resolver({"api_key": new}, delay=0.2))
    mgr = HermesDelegatingNousAuthManager.from_file(auth)

    await asyncio.gather(*(mgr.refresh_tokens("proactive") for _ in range(8)))
    assert len(resolver.calls) == 1
    assert mgr.access_token == new


async def test_concurrent_forced_refreshes_make_one_hermes_call(tmp_path: Path, hermes) -> None:
    auth = tmp_path / "auth.json"
    _write_store(auth, _fresh("old"))
    resolver = hermes(_Resolver({"api_key": _fresh("new")}, delay=0.2))
    mgr = HermesDelegatingNousAuthManager.from_file(auth)

    await asyncio.gather(*(mgr.refresh_tokens("reactive", force=True) for _ in range(8)))
    assert len(resolver.calls) == 1


async def test_relogin_error_is_terminal(tmp_path: Path, hermes) -> None:
    class AuthError(RuntimeError):
        relogin_required = True

    auth = tmp_path / "auth.json"
    _write_store(auth, _stale())
    hermes(_Resolver(AuthError("Invalid refresh token")))
    mgr = HermesDelegatingNousAuthManager.from_file(auth)
    with pytest.raises(NousRefreshExpiredError, match="hermes auth add nous"):
        await mgr.ensure_fresh_token()


async def test_other_errors_are_runtime_errors_and_keep_the_token(tmp_path: Path, hermes) -> None:
    auth = tmp_path / "auth.json"
    old = _stale("old")
    _write_store(auth, old)
    hermes(_Resolver(TimeoutError("Timed out waiting for auth store lock")))
    mgr = HermesDelegatingNousAuthManager.from_file(auth)
    with pytest.raises(RuntimeError, match="through Hermes failed: TimeoutError"):
        await mgr.ensure_fresh_token()
    assert mgr.access_token == old


async def test_reply_without_a_token_is_an_error(tmp_path: Path, hermes) -> None:
    auth = tmp_path / "auth.json"
    _write_store(auth, _stale())
    hermes(_Resolver({"api_key": ""}))
    mgr = HermesDelegatingNousAuthManager.from_file(auth)
    with pytest.raises(RuntimeError, match="without an inference token"):
        await mgr.ensure_fresh_token()


async def test_missing_hermes_is_a_clear_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    auth = tmp_path / "auth.json"
    _write_store(auth, _stale())
    monkeypatch.setitem(sys.modules, "hermes_cli", None)
    monkeypatch.setitem(sys.modules, "hermes_cli.auth", None)
    mgr = HermesDelegatingNousAuthManager.from_file(auth)
    with pytest.raises(RuntimeError, match="hermes_cli.auth could not be imported"):
        await mgr.ensure_fresh_token()


@pytest.mark.parametrize("store", ["missing", "no-access-token"])
async def test_logged_out_at_startup_resolves_on_first_use(tmp_path: Path, hermes, store: str) -> None:
    auth = tmp_path / "auth.json"
    if store == "no-access-token":
        _write_store(auth, None)
    new = _fresh("after-login")
    resolver = hermes(_Resolver({"api_key": new}))

    mgr = HermesDelegatingNousAuthManager.from_file(auth)
    assert mgr.access_token == PENDING_TOKEN
    assert mgr._token_is_stale()
    assert await mgr.ensure_fresh_token() == new
    assert resolver.calls == [{"force_refresh": False}]

    # A 401 on the placeholder must not be reported to Hermes as a failed bearer.
    mgr.access_token = PENDING_TOKEN
    await mgr.refresh_tokens("reactive", force=True)
    assert resolver.calls[-1] == {"force_refresh": True}
