"""Tests for Okta AI Agent (v2) identity forwarding (proposal §13.7).

Proves the adoption contract for this wrapper:

- Okta AI Agent identity configuration reaches the base SDK's
  ``OktaAgentIdentity``/``OpenBoxConfig`` UNCHANGED — this module never
  re-validates the private key, re-implements the 2048-bit floor, or
  re-derives the algorithm allowlist.
- ``GovernanceClient``/``GovernanceActivities`` route Okta-configured calls to
  the v2 endpoints with an RS256 assertion header, never a v1 DID header.
- 401/403 in Okta mode fails closed unconditionally (proposal §13.6),
  translated into THIS package's own ``OpenBoxAuthError``/``OpenBoxSigningError``.
- DID / unsigned mode is completely unaffected by the new okta_identity wiring.

No test here builds a JWT/RSA signature itself — every assertion is produced
by the base SDK (``openbox_core.identity_okta``), imported and called exactly
as production code does.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cryptography.hazmat.primitives import serialization as crypto_serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from openbox.activities import GovernanceActivities
from openbox.client import GovernanceClient
from openbox.config import (
    _bootstrap_okta_identity,
    _load_okta_identity,
    get_global_config,
    initialize,
    resolve_signing_defaults,
)
from openbox.core_adapter import create_core_runtime
from openbox.errors import OpenBoxAuthError, OpenBoxConfigError, OpenBoxSigningError
from openbox.governance_state import TemporalGovernanceState
from openbox.request_signing import (
    HEADER_ASSERTION,
    HEADER_DID,
    prepare_okta_signed_request,
)

API_URL = "https://core.openbox.test"
API_KEY = "obx_test_valid"


def _generate_pkcs8_pem(key_size: int = 2048) -> str:
    key = rsa.generate_private_key(public_exponent=65537, key_size=key_size)
    return key.private_bytes(
        encoding=crypto_serialization.Encoding.PEM,
        format=crypto_serialization.PrivateFormat.PKCS8,
        encryption_algorithm=crypto_serialization.NoEncryption(),
    ).decode("ascii")


_PRIVATE_KEY_PEM = _generate_pkcs8_pem()

_OKTA_KWARGS = {
    "openbox_agent_id": "agent-1",
    "organization_id": "org-1",
    "deployment_id": "dep-1",
    "okta_agent_id": "wlp-1",
    "okta_agent_key_id": "kid-1",
    "okta_agent_private_key": _PRIVATE_KEY_PEM,
    "agent_proof_audience": "urn:openbox:dep-1:core",
}


def _okta_identity():
    return _load_okta_identity(**_OKTA_KWARGS)


# ═══════════════════════════════════════════════════════════════════
# request_signing.prepare_okta_signed_request
# ═══════════════════════════════════════════════════════════════════


def test_prepare_okta_signed_request_no_v1_headers():
    """v2 request carries only the assertion header — never a v1 DID header."""
    identity = _okta_identity()
    headers, body = prepare_okta_signed_request(
        "POST",
        "/api/v2/governance/evaluate",
        {"event_type": "WorkflowStarted"},
        api_key=API_KEY,
        okta_identity=identity,
    )
    assert HEADER_ASSERTION in headers
    assert HEADER_DID not in headers
    assert body == b'{"event_type":"WorkflowStarted"}'


def test_prepare_okta_signed_request_empty_body_get():
    """GET (empty-body) requests hash EMPTY_BODY_SHA256 — used by validate."""
    identity = _okta_identity()
    headers, body = prepare_okta_signed_request(
        "GET", "/api/v2/auth/validate", None, api_key=API_KEY, okta_identity=identity
    )
    assert body == b""
    assert HEADER_ASSERTION in headers


# ═══════════════════════════════════════════════════════════════════
# config: _load_okta_identity / initialize / resolve_signing_defaults
# ═══════════════════════════════════════════════════════════════════


def test_load_okta_identity_builds_real_signer():
    identity = _okta_identity()
    assert identity.external_agent_id == "wlp-1"
    assert identity.key_id == "kid-1"
    assert identity.algorithm == "RS256"


def test_load_okta_identity_rejects_undersized_key():
    small_pem = _generate_pkcs8_pem(key_size=1024)
    with pytest.raises(OpenBoxConfigError):
        _load_okta_identity(**{**_OKTA_KWARGS, "okta_agent_private_key": small_pem})


def test_initialize_configures_okta_identity(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "openbox.config._validate_api_key_with_server", lambda *a, **k: None
    )
    initialize(API_URL, API_KEY, **_OKTA_KWARGS)

    gc = get_global_config()
    assert gc.has_okta_identity()
    identity = gc.get_okta_identity()
    assert identity.external_agent_id == "wlp-1"
    assert gc.agent_did is None  # v1 stays unset — mutually exclusive


def test_initialize_bootstraps_okta_identity_from_private_key_only(
    monkeypatch: pytest.MonkeyPatch,
):
    identity = _okta_identity()
    bootstrap = MagicMock(return_value=identity)
    monkeypatch.setattr("openbox.config._bootstrap_okta_identity", bootstrap)

    initialize(
        API_URL,
        API_KEY,
        okta_agent_private_key=_PRIVATE_KEY_PEM,
    )

    bootstrap.assert_called_once_with(
        api_url=API_URL,
        api_key=API_KEY,
        timeout=30.0,
        okta_agent_private_key=_PRIVATE_KEY_PEM,
    )
    gc = get_global_config()
    assert gc.get_okta_identity() is identity
    assert gc.agent_did is None


def test_bootstrap_okta_identity_delegates_to_base_client(
    monkeypatch: pytest.MonkeyPatch,
):
    document = SimpleNamespace(
        openbox_agent_id="agent-1",
        organization_id="org-1",
        deployment_id="dep-1",
        assertion_audience="urn:openbox:dep-1:core",
        okta=SimpleNamespace(external_agent_id="wlp-1", credential_kid="kid-1"),
    )
    client = MagicMock()
    client.identity_metadata.return_value = document
    client_type = MagicMock(return_value=client)
    monkeypatch.setattr("openbox_core.client.EvaluationClient", client_type)

    identity = _bootstrap_okta_identity(
        api_url=API_URL,
        api_key=API_KEY,
        timeout=12.0,
        okta_agent_private_key=_PRIVATE_KEY_PEM,
    )

    client_type.assert_called_once()
    call = client_type.call_args
    assert call.args == (API_URL, API_KEY)
    assert call.kwargs["timeout_seconds"] == 12.0
    assert call.kwargs["okta_bootstrap_private_key"] == _PRIVATE_KEY_PEM
    client.validate_api_key.assert_called_once_with()
    client.close.assert_called_once_with()
    assert identity.external_agent_id == "wlp-1"
    assert identity.key_id == "kid-1"


def test_initialize_rejects_bootstrap_without_server_validation():
    with pytest.raises(OpenBoxConfigError, match="requires server validation"):
        initialize(
            API_URL,
            API_KEY,
            validate=False,
            okta_agent_private_key=_PRIVATE_KEY_PEM,
        )


def test_initialize_rejects_bootstrap_with_unsupported_algorithm():
    with pytest.raises(OpenBoxConfigError, match="only 'RS256'"):
        initialize(
            API_URL,
            API_KEY,
            okta_agent_private_key=_PRIVATE_KEY_PEM,
            okta_agent_algorithm="RS512",
        )


def test_initialize_rejects_did_and_okta_together(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "openbox.config._validate_api_key_with_server", lambda *a, **k: None
    )
    with pytest.raises(OpenBoxConfigError, match="mutually exclusive"):
        initialize(
            API_URL,
            API_KEY,
            agent_did="did:aip:12345678-1234-1234-1234-1234567890ab",
            agent_private_key="x",
            **_OKTA_KWARGS,
        )


def test_initialize_rejects_partial_okta_config(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "openbox.config._validate_api_key_with_server", lambda *a, **k: None
    )
    partial = dict(_OKTA_KWARGS)
    del partial["organization_id"]
    with pytest.raises(OpenBoxConfigError, match="Partial Okta"):
        initialize(API_URL, API_KEY, **partial)


def test_initialize_validates_against_v2_when_okta_configured(
    monkeypatch: pytest.MonkeyPatch,
):
    """The startup validation ping signs+targets /api/v2/auth/validate, never v1."""
    calls: list[tuple] = []

    def _fake_validate(
        api_url, api_key, timeout, *, agent_did=None, signer=None, okta_identity=None
    ):
        calls.append((agent_did, signer, okta_identity))

    monkeypatch.setattr("openbox.config._validate_api_key_with_server", _fake_validate)
    initialize(API_URL, API_KEY, **_OKTA_KWARGS)

    assert len(calls) == 1
    agent_did, signer, okta_identity = calls[0]
    assert agent_did is None
    assert signer is None
    assert okta_identity is not None
    assert okta_identity.external_agent_id == "wlp-1"


def test_resolve_signing_defaults_falls_back_to_global_okta_identity(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "openbox.config._validate_api_key_with_server", lambda *a, **k: None
    )
    initialize(API_URL, API_KEY, **_OKTA_KWARGS)

    agent_did, signer, okta_identity = resolve_signing_defaults(None, None, None)
    assert agent_did is None
    assert signer is None
    assert okta_identity is not None
    assert okta_identity.external_agent_id == "wlp-1"


def test_resolve_signing_defaults_explicit_values_are_not_overridden():
    """Worker/plugin pass explicit values — the global fallback must be a no-op."""
    explicit_identity = _okta_identity()
    agent_did, signer, okta_identity = resolve_signing_defaults(
        None, None, explicit_identity
    )
    assert okta_identity is explicit_identity


# ═══════════════════════════════════════════════════════════════════
# core_adapter.create_core_runtime
# ═══════════════════════════════════════════════════════════════════


def test_create_core_runtime_preserves_okta_identity():
    runtime = create_core_runtime(
        api_url=API_URL,
        api_key=API_KEY,
        state=TemporalGovernanceState(),
        **_OKTA_KWARGS,
    )
    try:
        assert runtime.config.identity_method == "okta_ai_agent"
        identity = runtime.config.load_okta_identity()
        assert identity is not None
        assert identity.external_agent_id == "wlp-1"
        assert runtime.config.load_identity() is None
    finally:
        runtime.close()


def test_create_core_runtime_uses_prebootstrapped_okta_identity():
    identity = _okta_identity()
    runtime = create_core_runtime(
        api_url=API_URL,
        api_key=API_KEY,
        state=TemporalGovernanceState(),
        okta_agent_private_key=_PRIVATE_KEY_PEM,
        resolved_okta_identity=identity,
    )
    try:
        assert runtime.config.okta_config_mode() == "bootstrap"
        assert runtime.client._identity is identity
    finally:
        runtime.close()


# ═══════════════════════════════════════════════════════════════════
# GovernanceClient: v2 dispatch + auth-failure classification
# ═══════════════════════════════════════════════════════════════════


def _mock_httpx_response(status_code: int, json_body: dict, content: bytes = b"{}"):
    response = MagicMock()
    response.status_code = status_code
    response.json = MagicMock(return_value=json_body)
    response.content = content
    return response


@pytest.mark.asyncio
async def test_governance_client_evaluate_routes_to_v2_with_okta():
    identity = _okta_identity()
    client = GovernanceClient(api_url=API_URL, api_key=API_KEY, okta_identity=identity)

    mock_response = _mock_httpx_response(200, {"verdict": "allow"})
    mock_async_client = AsyncMock()
    mock_async_client.post = AsyncMock(return_value=mock_response)
    mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=mock_async_client):
        result = await client.evaluate_event({"event_type": "WorkflowStarted"})

    assert result is not None
    call = mock_async_client.post.call_args
    assert call.args[0] == f"{API_URL}/api/v2/governance/evaluate"
    headers = call.kwargs["headers"]
    assert HEADER_ASSERTION in headers
    assert HEADER_DID not in headers


@pytest.mark.asyncio
async def test_governance_client_poll_approval_routes_to_v2_with_okta():
    identity = _okta_identity()
    client = GovernanceClient(api_url=API_URL, api_key=API_KEY, okta_identity=identity)

    mock_response = _mock_httpx_response(200, {"verdict": "allow"})
    mock_async_client = AsyncMock()
    mock_async_client.post = AsyncMock(return_value=mock_response)
    mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=mock_async_client):
        result = await client.poll_approval("wf-1", "run-1", "act-1")

    assert result is not None
    call = mock_async_client.post.call_args
    assert call.args[0] == f"{API_URL}/api/v2/governance/approval"


@pytest.mark.asyncio
async def test_governance_client_evaluate_401_raises_in_okta_mode():
    """401 fails closed unconditionally (proposal §13.6) — never fallback ALLOW."""
    identity = _okta_identity()
    client = GovernanceClient(
        api_url=API_URL,
        api_key=API_KEY,
        okta_identity=identity,
        on_api_error="fail_open",
    )

    mock_response = _mock_httpx_response(
        401, {}, content=b'{"reason_code": "assertion_signature_invalid"}'
    )
    mock_async_client = AsyncMock()
    mock_async_client.post = AsyncMock(return_value=mock_response)
    mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=mock_async_client):
        with pytest.raises((OpenBoxAuthError, OpenBoxSigningError)):
            await client.evaluate_event({"event_type": "WorkflowStarted"})


@pytest.mark.asyncio
async def test_governance_client_poll_approval_401_raises_never_none():
    """401 during approval polling must never be read as 'still pending'."""
    identity = _okta_identity()
    client = GovernanceClient(api_url=API_URL, api_key=API_KEY, okta_identity=identity)

    mock_response = _mock_httpx_response(401, {}, content=b"{}")
    mock_async_client = AsyncMock()
    mock_async_client.post = AsyncMock(return_value=mock_response)
    mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=mock_async_client):
        with pytest.raises(OpenBoxAuthError):
            await client.poll_approval("wf-1", "run-1", "act-1")


@pytest.mark.asyncio
async def test_governance_client_did_mode_unaffected_by_okta_wiring():
    """A DID-configured (or unsigned) client must still hit v1 — zero regression."""
    client = GovernanceClient(api_url=API_URL, api_key=API_KEY)  # no identity at all

    mock_response = _mock_httpx_response(200, {"verdict": "allow"})
    mock_async_client = AsyncMock()
    mock_async_client.post = AsyncMock(return_value=mock_response)
    mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=mock_async_client):
        await client.evaluate_event({"event_type": "WorkflowStarted"})

    call = mock_async_client.post.call_args
    assert call.args[0] == f"{API_URL}/api/v1/governance/evaluate"


# ═══════════════════════════════════════════════════════════════════
# GovernanceActivities: v2 dispatch (activity-context signing)
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_governance_activities_send_event_routes_to_v2_with_okta():
    identity = _okta_identity()
    activities = GovernanceActivities(API_URL, API_KEY, okta_identity=identity)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json = MagicMock(return_value={"verdict": "allow"})

    mock_async_client = AsyncMock()
    mock_async_client.post = AsyncMock(return_value=mock_response)
    mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client.__aexit__ = AsyncMock(return_value=None)

    with patch("openbox.activities.httpx.AsyncClient", return_value=mock_async_client):
        await activities.send_governance_event(
            {"payload": {"event_type": "WorkflowStarted"}}
        )

    call = mock_async_client.post.call_args
    assert call.args[0] == f"{API_URL}/api/v2/governance/evaluate"
    headers = call.kwargs["headers"]
    assert HEADER_ASSERTION in headers
    assert HEADER_DID not in headers
