"""Tests for Voca API authentication helpers."""

import pytest

from voca.api import auth


@pytest.mark.asyncio
async def test_static_auth_token_accepts_matching_token(monkeypatch):
    monkeypatch.setattr(auth, "AUTH_ENABLED", True)
    monkeypatch.setattr(auth, "AUTH_MODE", "auto")
    monkeypatch.setattr(auth, "AUTH_TOKEN", "secret-token")
    monkeypatch.setattr(auth, "AUTH_TOKENS", "")

    result = await auth.validate_token("secret-token")

    assert result.ok is True
    assert result.token_type == "token"


@pytest.mark.asyncio
async def test_static_auth_token_rejects_non_matching_token(monkeypatch):
    monkeypatch.setattr(auth, "AUTH_ENABLED", True)
    monkeypatch.setattr(auth, "AUTH_MODE", "token")
    monkeypatch.setattr(auth, "AUTH_TOKEN", "secret-token")
    monkeypatch.setattr(auth, "AUTH_TOKENS", "")

    result = await auth.validate_token("wrong-token")

    assert result.ok is False
    assert result.error == "TOKEN_INVALID"


@pytest.mark.asyncio
async def test_static_auth_tokens_accepts_comma_separated_tokens(monkeypatch):
    monkeypatch.setattr(auth, "AUTH_ENABLED", True)
    monkeypatch.setattr(auth, "AUTH_MODE", "static")
    monkeypatch.setattr(auth, "AUTH_TOKEN", "")
    monkeypatch.setattr(auth, "AUTH_TOKENS", "first-token, second-token")

    result = await auth.validate_token("second-token")

    assert result.ok is True
    assert result.token_type == "token"


@pytest.mark.asyncio
async def test_ws_config_token_uses_static_auth(monkeypatch):
    monkeypatch.setattr(auth, "AUTH_ENABLED", True)
    monkeypatch.setattr(auth, "AUTH_MODE", "token")
    monkeypatch.setattr(auth, "AUTH_TOKEN", "stream-token")
    monkeypatch.setattr(auth, "AUTH_TOKENS", "")

    result = await auth.validate_ws_config_token({"token": "stream-token"})

    assert result.ok is True
    assert result.token_type == "token"

