"""Temporal-compatible wrappers around the shared exact-body signing contract.

The shared SDK owns JSON byte serialization, canonical request construction,
and Ed25519 signing.  This module keeps the historical Temporal call signature,
raw Temporal SDK headers, and timestamp/nonce patch seams used by callers and
tests.  It is intentionally absent from deterministic workflow import paths.
"""

from __future__ import annotations

import secrets
from datetime import UTC, datetime

from openbox_core.identity import (
    EMPTY_BODY_SHA256,
    HEADER_BODY_SHA256,
    HEADER_DID,
    HEADER_NONCE,
    HEADER_SIGNATURE,
    HEADER_TIMESTAMP,
    AgentIdentity,
)
from openbox_core.identity import (
    prepare_signed_request as _prepare_signed_request,
)
from openbox_core.serialization import serialize_body

__all__ = [
    "EMPTY_BODY_SHA256",
    "HEADER_DID",
    "HEADER_TIMESTAMP",
    "HEADER_NONCE",
    "HEADER_SIGNATURE",
    "HEADER_BODY_SHA256",
    "serialize_body",
    "prepare_signed_request",
    "send_sync",
    "send_async",
]


def prepare_signed_request(
    method: str,
    path: str,
    payload: dict | None,
    *,
    api_key: str,
    agent_did: str | None,
    signer,
) -> tuple[dict, bytes]:
    """Build Temporal auth headers and shared exact signed body bytes.

    Callers must transmit the returned body with ``content=``.  A partial
    DID/signer pair intentionally preserves the historical unsigned behavior.
    """
    # Keep Temporal's established header values rather than replacing them
    # with the base SDK's framework identifier.

    identity = (
        AgentIdentity(agent_did=agent_did, signer=signer)
        if agent_did and signer is not None
        else None
    )
    from openbox import __version__

    return _prepare_signed_request(
        method,
        path,
        payload,
        api_key=api_key,
        identity=identity,
        sdk_engine="temporal",
        sdk_version=__version__,
        # Preserve the module-level deterministic seams from the legacy helper.
        _timestamp=(datetime.now(UTC).isoformat() if identity else None),
        _nonce=(secrets.token_urlsafe(24) if identity else None),
    )


def send_sync(client, url: str, headers: dict, body_bytes: bytes):
    """Send prepared bytes through a sync client (never ``json=``)."""
    return client.post(url, headers=headers, content=body_bytes)


async def send_async(client, url: str, headers: dict, body_bytes: bytes):
    """Send prepared bytes through an async client (never ``json=``)."""
    return await client.post(url, headers=headers, content=body_bytes)
