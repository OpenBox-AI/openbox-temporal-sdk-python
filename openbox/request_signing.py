# openbox/request_signing.py
"""AIP DID + Ed25519 signed-request construction — thin shim over the base SDK.

The canonical signing contract now lives in ``openbox_core.identity`` (the
OpenBox base SDK); this module keeps the Temporal SDK's public surface and
delegates the canonical-string/signature construction, byte-for-byte
(gated by tests/test_base_sdk_signing_parity.py against the golden fixture)::

    UPPER(METHOD)\\nPATH\\nTIMESTAMP\\nNONCE\\nBODY_SHA256_HEX

SANDBOX SAFETY: unchanged — ``cryptography`` work happens only through a
pre-loaded signer object passed in by the caller, ``httpx`` is imported lazily
inside the transports, and this module must NEVER be imported on the Temporal
workflow sandbox path. Timestamp/nonce generation stays in THIS module's
namespace (``datetime``/``secrets``) so existing deterministic-signing tests
keep patching the same seams.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timezone
from typing import Optional, Tuple

# Canonical constants come from the base SDK — single source of truth.
from openbox_core.identity import (  # noqa: F401  (re-exported public names)
    EMPTY_BODY_SHA256,
    HEADER_BODY_SHA256,
    HEADER_DID,
    HEADER_NONCE,
    HEADER_SIGNATURE,
    HEADER_TIMESTAMP,
    AgentIdentity,
)
from openbox_core.identity import (
    prepare_signed_request as _core_prepare_signed_request,
)
from openbox_core.serialization import serialize_body  # noqa: F401  (re-export)


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
    # Base auth headers (Authorization / User-Agent / SDK-Version) stay
    # Temporal-branded — only the 5 AIP headers come from the base SDK.
    from .hook_governance import build_auth_headers

    body_bytes = serialize_body(payload)
    headers = build_auth_headers(api_key)

    if signer is not None and agent_did:
        # Generate timestamp/nonce HERE (patchable seams preserved); the base
        # SDK builds the canonical string, signature, and AIP headers from
        # these exact inputs.
        timestamp = datetime.now(timezone.utc).isoformat()
        nonce = secrets.token_urlsafe(24)
        identity = AgentIdentity(agent_did=agent_did, signer=signer)
        signed_headers, _ = _core_prepare_signed_request(
            method,
            path,
            payload,
            api_key=api_key,
            identity=identity,
            _timestamp=timestamp,
            _nonce=nonce,
        )
        for header_name in (
            HEADER_DID,
            HEADER_TIMESTAMP,
            HEADER_NONCE,
            HEADER_SIGNATURE,
            HEADER_BODY_SHA256,
        ):
            headers[header_name] = signed_headers[header_name]

    return headers, body_bytes


def send_sync(client, url: str, headers: dict, body_bytes: bytes):
    """POST prepared bytes via a sync ``httpx.Client``. Sends ``content=`` — never
    ``json=`` — so the transmitted bytes match the signed/hashed bytes."""
    return client.post(url, headers=headers, content=body_bytes)


async def send_async(client, url: str, headers: dict, body_bytes: bytes):
    """POST prepared bytes via an async ``httpx.AsyncClient``. Sends ``content=`` —
    never ``json=`` — so the transmitted bytes match the signed/hashed bytes."""
    return await client.post(url, headers=headers, content=body_bytes)
