"""AIP DID + Ed25519 signed-request construction (v1), plus Okta AI Agent (v2)
RS256 assertion construction (proposal §13.7).

This module keeps the Temporal SDK's public surface while delegating ALL
canonical-string/signature construction to ``openbox_core``::

    v1: UPPER(METHOD)\\nPATH\\nTIMESTAMP\\nNONCE\\nBODY_SHA256_HEX  (Ed25519)
    v2: compact RS256 JWT assertion, contract §§2-4                (Okta AI Agent)

SANDBOX SAFETY: ``cryptography`` work happens only through a pre-loaded signer
object (v1) or a pre-loaded ``OktaAgentIdentity`` (v2) passed by the caller,
``httpx`` is imported lazily inside the transports, and this module must not
be imported from the Temporal workflow sandbox path.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timezone
from typing import Any, Optional, Tuple

from openbox_core.identity import (  # noqa: F401  (re-exported public names)
    EMPTY_BODY_SHA256,
    HEADER_BODY_SHA256,
    HEADER_DID,
    HEADER_NONCE,
    HEADER_SIGNATURE,
    HEADER_TIMESTAMP,
    AgentIdentity,
)
from openbox_core.identity import prepare_signed_request as _core_prepare_signed_request
from openbox_core.identity_okta import HEADER_ASSERTION  # noqa: F401  (re-export)
from openbox_core.identity_okta import (
    prepare_okta_signed_request as _core_prepare_okta_signed_request,
)
from openbox_core.serialization import serialize_body  # noqa: F401  (re-export)


def _sdk_identifier() -> str:
    from . import __version__

    return f"openbox-temporal-python-v{__version__.removeprefix('v')}"


def prepare_signed_request(
    method: str,
    path: str,
    payload: Optional[dict],
    *,
    api_key: str,
    agent_did: Optional[str],
    signer,
) -> Tuple[dict, bytes]:
    """Build request headers + exact body bytes — the single source of truth.

    Args:
        method: HTTP method (case-insensitive; upper-cased into the canonical string).
        path: URL path only, no host/query (e.g. ``/api/v1/governance/evaluate``).
        payload: JSON-serializable body, or ``None`` for empty-body (GET) requests.
        api_key: Bearer API key for the base auth headers.
        agent_did: Agent DID asserted in ``X-OpenBox-Agent-DID`` (or ``None``).
        signer: Loaded ``Ed25519PrivateKey`` object, or ``None`` for unsigned mode.

    Returns:
        ``(headers, body_bytes)``. Callers MUST send ``content=body_bytes`` —
        never ``json=`` — so the transmitted bytes match the hashed bytes.
    """
    from . import __version__

    sdk_identifier = f"openbox-temporal-python-v{__version__.removeprefix('v')}"

    if signer is not None and agent_did:
        identity = AgentIdentity(agent_did=agent_did, signer=signer)
        timestamp = datetime.now(timezone.utc).isoformat()
        nonce = secrets.token_urlsafe(24)
        return _core_prepare_signed_request(
            method,
            path,
            payload,
            api_key=api_key,
            identity=identity,
            sdk_version=sdk_identifier,
            _timestamp=timestamp,
            _nonce=nonce,
        )
    return _core_prepare_signed_request(
        method,
        path,
        payload,
        api_key=api_key,
        identity=None,
        sdk_version=sdk_identifier,
    )


def prepare_okta_signed_request(
    method: str,
    path: str,
    payload: Optional[dict],
    *,
    api_key: str,
    okta_identity: Any,
) -> Tuple[dict, bytes]:
    """Build v2 (Okta AI Agent) request headers + exact body bytes.

    Entirely delegates RS256/JWT assertion construction to the base SDK's
    ``openbox_core.identity_okta.prepare_okta_signed_request`` (proposal
    §13.7) — this module never builds a JWT, hashes a key, or selects an
    algorithm itself.

    Args:
        method: HTTP method (case-insensitive; upper-cased into ``htm``).
        path: URL path only, no host/query — INCLUDES the ``/api/v2`` prefix
            (e.g. ``/api/v2/governance/evaluate``).
        payload: JSON-serializable body, or ``None`` for empty-body (GET)
            requests.
        api_key: Bearer API key for the base auth headers and
            ``obx_api_key_sha256``.
        okta_identity: A loaded ``openbox_core.identity_okta.OktaAgentIdentity``.

    Returns:
        ``(headers, body_bytes)``. Callers MUST send ``content=body_bytes`` —
        never ``json=`` — so the transmitted bytes match the hashed bytes.
    """
    return _core_prepare_okta_signed_request(
        method,
        path,
        payload,
        api_key=api_key,
        identity=okta_identity,
        sdk_version=_sdk_identifier(),
    )


def send_sync(client, url: str, headers: dict, body_bytes: bytes):
    """POST prepared bytes via a sync ``httpx.Client``. Sends ``content=`` — never
    ``json=`` — so the transmitted bytes match the signed/hashed bytes."""
    return client.post(url, headers=headers, content=body_bytes)


async def send_async(client, url: str, headers: dict, body_bytes: bytes):
    """POST prepared bytes via an async ``httpx.AsyncClient``. Sends ``content=`` —
    never ``json=`` — so the transmitted bytes match the signed/hashed bytes."""
    return await client.post(url, headers=headers, content=body_bytes)
