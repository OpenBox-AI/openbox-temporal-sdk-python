# openbox/request_signing.py
"""Single source of truth for AIP DID + Ed25519 signed-request construction.

SANDBOX SAFETY: this module imports ``cryptography`` work only through a
pre-loaded signer object passed in by the caller, and imports ``httpx`` lazily
inside the transports. It must NEVER be imported on the Temporal workflow
sandbox path and is deliberately omitted from eager ``openbox/__init__.py``
exports. Signing happens only in activities / client / config — never in
workflow code.

Canonical string contract (must match Core ``agent.go:93``)::

    UPPER(METHOD)\\nPATH\\nTIMESTAMP\\nNONCE\\nBODY_SHA256_HEX

The five AIP headers (``agent.go:26-30``) are attached only when both a signer
and an agent DID are configured.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
from datetime import datetime, timezone
from typing import Optional, Tuple

# SHA-256 of empty bytes — body hash for GET / empty-body requests.
EMPTY_BODY_SHA256 = (
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
)

# AIP signed-request header names (Core agent.go:26-30).
HEADER_DID = "X-OpenBox-Agent-DID"
HEADER_TIMESTAMP = "X-OpenBox-Agent-Timestamp"
HEADER_NONCE = "X-OpenBox-Agent-Nonce"
HEADER_SIGNATURE = "X-OpenBox-Agent-Signature"
HEADER_BODY_SHA256 = "X-OpenBox-Body-SHA256"


def serialize_body(payload: Optional[dict]) -> bytes:
    """Serialize a payload to the EXACT bytes that will be transmitted.

    - ``None`` -> ``b""`` (body hash becomes :data:`EMPTY_BODY_SHA256`).
    - Compact separators (no spaces) and a single serialization pass so the
      bytes we hash are identical to the bytes we send. Re-serializing with
      ``json=`` elsewhere would break Core's body-hash verification.
    """
    if payload is None:
        return b""
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


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
    # Base auth headers (Authorization / User-Agent / SDK-Version) — DRY reuse.
    from .hook_governance import build_auth_headers

    body_bytes = serialize_body(payload)
    headers = build_auth_headers(api_key)

    if signer is not None and agent_did:
        body_sha256 = hashlib.sha256(body_bytes).hexdigest()
        timestamp = datetime.now(timezone.utc).isoformat()
        nonce = secrets.token_urlsafe(24)
        canonical = "\n".join([method.upper(), path, timestamp, nonce, body_sha256])
        signature = base64.b64encode(
            signer.sign(canonical.encode("utf-8"))
        ).decode("ascii")
        headers[HEADER_DID] = agent_did
        headers[HEADER_TIMESTAMP] = timestamp
        headers[HEADER_NONCE] = nonce
        headers[HEADER_SIGNATURE] = signature
        headers[HEADER_BODY_SHA256] = body_sha256

    return headers, body_bytes


def send_sync(client, url: str, headers: dict, body_bytes: bytes):
    """POST prepared bytes via a sync ``httpx.Client``. Sends ``content=`` — never
    ``json=`` — so the transmitted bytes match the signed/hashed bytes."""
    return client.post(url, headers=headers, content=body_bytes)


async def send_async(client, url: str, headers: dict, body_bytes: bytes):
    """POST prepared bytes via an async ``httpx.AsyncClient``. Sends ``content=`` —
    never ``json=`` — so the transmitted bytes match the signed/hashed bytes."""
    return await client.post(url, headers=headers, content=body_bytes)
