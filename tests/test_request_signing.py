"""Unit + local-verify tests for AIP DID + Ed25519 request signing.

Body-hash determinism is the #1 failure mode against Core, so these tests assert
the exact transmitted bytes are what gets hashed/signed, and replicate Core's
verification (verify sig over METHOD\\nPATH\\nTS\\nNONCE\\nSHA + recompute body sha).
"""

import base64
import hashlib
import json
import re

import pytest

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from openbox.request_signing import (
    EMPTY_BODY_SHA256,
    HEADER_BODY_SHA256,
    HEADER_DID,
    HEADER_NONCE,
    HEADER_SIGNATURE,
    HEADER_TIMESTAMP,
    prepare_signed_request,
    send_async,
    send_sync,
    serialize_body,
)

TEST_DID = "did:aip:12345678-1234-1234-1234-1234567890ab"


@pytest.fixture
def signer() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.generate()


def _verify_like_core(headers: dict, method: str, path: str, body: bytes,
                      public_key: Ed25519PublicKey) -> None:
    """Replicate Core's verification. Raises if the request would be rejected."""
    # 1. recompute body sha and compare to the header
    recomputed = hashlib.sha256(body).hexdigest()
    assert headers[HEADER_BODY_SHA256] == recomputed
    # 2. rebuild canonical string from the headers and verify the signature
    canonical = "\n".join(
        [
            method.upper(),
            path,
            headers[HEADER_TIMESTAMP],
            headers[HEADER_NONCE],
            headers[HEADER_BODY_SHA256],
        ]
    )
    sig = base64.b64decode(headers[HEADER_SIGNATURE])
    public_key.verify(sig, canonical.encode("utf-8"))  # raises on mismatch


# serialize_body / empty-body hash


def test_empty_body_is_empty_bytes():
    assert serialize_body(None) == b""


def test_empty_body_sha256_constant_matches_sha256_of_empty():
    assert hashlib.sha256(b"").hexdigest() == EMPTY_BODY_SHA256


def test_serialize_body_is_compact_no_spaces():
    body = serialize_body({"a": 1, "b": 2})
    assert body == b'{"a":1,"b":2}'
    assert b", " not in body and b": " not in body


# Signed round-trip (local verify harness)


def test_signed_request_verifies_with_public_key(signer):
    payload = {"event_type": "WorkflowStarted", "workflow_id": "wf-1"}
    headers, body = prepare_signed_request(
        "POST", "/api/v1/governance/evaluate", payload,
        api_key="obx_test_k", agent_did=TEST_DID, signer=signer,
    )
    _verify_like_core(headers, "POST", "/api/v1/governance/evaluate", body,
                      signer.public_key())


def test_tampered_body_fails_verification(signer):
    headers, body = prepare_signed_request(
        "POST", "/api/v1/governance/evaluate", {"x": 1},
        api_key="k", agent_did=TEST_DID, signer=signer,
    )
    tampered = body + b" "  # any change to the bytes
    with pytest.raises(AssertionError):
        # body sha header no longer matches the tampered bytes
        _verify_like_core(headers, "POST", "/api/v1/governance/evaluate",
                          tampered, signer.public_key())


def test_hashed_bytes_equal_transmitted_bytes(signer):
    """The header body-sha must be the sha of exactly the returned body bytes."""
    payload = {"nested": {"b": 2, "a": [1, 2, 3]}, "k": "v"}
    headers, body = prepare_signed_request(
        "POST", "/api/v1/governance/evaluate", payload,
        api_key="k", agent_did=TEST_DID, signer=signer,
    )
    assert headers[HEADER_BODY_SHA256] == hashlib.sha256(body).hexdigest()
    # and body is parseable back to the same logical payload
    assert json.loads(body) == payload


def test_empty_body_get_signed_uses_empty_hash(signer):
    headers, body = prepare_signed_request(
        "GET", "/api/v1/auth/validate", None,
        api_key="k", agent_did=TEST_DID, signer=signer,
    )
    assert body == b""
    assert headers[HEADER_BODY_SHA256] == EMPTY_BODY_SHA256
    _verify_like_core(headers, "GET", "/api/v1/auth/validate", body,
                      signer.public_key())


def test_all_five_sites_verify(signer):
    """Local-verify harness for every signed transport site."""
    sites = [
        ("POST", "/api/v1/governance/evaluate", {"event_type": "WorkflowStarted"}),
        ("POST", "/api/v1/governance/evaluate", {"event_type": "ActivityStarted"}),
        ("POST", "/api/v1/governance/approval",
         {"workflow_id": "w", "run_id": "r", "activity_id": "a"}),
        ("GET", "/api/v1/auth/validate", None),
    ]
    for method, path, payload in sites:
        headers, body = prepare_signed_request(
            method, path, payload, api_key="k", agent_did=TEST_DID, signer=signer,
        )
        _verify_like_core(headers, method, path, body, signer.public_key())


# Header format / canonical correctness


def test_five_aip_headers_present_when_signing(signer):
    headers, _ = prepare_signed_request(
        "POST", "/x", {"a": 1}, api_key="k", agent_did=TEST_DID, signer=signer,
    )
    for h in (HEADER_DID, HEADER_TIMESTAMP, HEADER_NONCE, HEADER_SIGNATURE,
              HEADER_BODY_SHA256):
        assert h in headers
    assert headers[HEADER_DID] == TEST_DID


def test_base_auth_headers_always_present(signer):
    headers, _ = prepare_signed_request(
        "POST", "/x", {"a": 1}, api_key="obx_test_k", agent_did=TEST_DID, signer=signer,
    )
    assert headers["Authorization"] == "Bearer obx_test_k"
    assert "User-Agent" in headers and "X-OpenBox-SDK-Version" in headers
    assert headers["X-OpenBox-SDK-Version"].startswith("openbox-temporal-python-v")


def test_body_sha_is_lowercase_hex(signer):
    headers, _ = prepare_signed_request(
        "POST", "/x", {"a": 1}, api_key="k", agent_did=TEST_DID, signer=signer,
    )
    assert re.fullmatch(r"[0-9a-f]{64}", headers[HEADER_BODY_SHA256])


def test_signature_is_base64(signer):
    headers, _ = prepare_signed_request(
        "POST", "/x", {"a": 1}, api_key="k", agent_did=TEST_DID, signer=signer,
    )
    # round-trips through base64 and is a 64-byte Ed25519 signature
    raw = base64.b64decode(headers[HEADER_SIGNATURE])
    assert len(raw) == 64


def test_timestamp_is_rfc3339_utc(signer):
    headers, _ = prepare_signed_request(
        "POST", "/x", {"a": 1}, api_key="k", agent_did=TEST_DID, signer=signer,
    )
    ts = headers[HEADER_TIMESTAMP]
    # ISO 8601 with UTC offset; Python isoformat() emits +00:00
    assert ts.endswith("+00:00")
    from datetime import datetime

    datetime.fromisoformat(ts)  # parseable, raises otherwise


def test_method_is_upper_in_canonical(signer):
    """Lowercase method input still signs an UPPER(METHOD) canonical string."""
    headers, body = prepare_signed_request(
        "post", "/api/v1/governance/evaluate", {"a": 1},
        api_key="k", agent_did=TEST_DID, signer=signer,
    )
    # verifying with 'POST' must succeed
    _verify_like_core(headers, "POST", "/api/v1/governance/evaluate", body,
                      signer.public_key())


# Nonce uniqueness


def test_nonce_unique_across_calls(signer):
    nonces = set()
    for _ in range(500):
        headers, _ = prepare_signed_request(
            "POST", "/x", {"a": 1}, api_key="k", agent_did=TEST_DID, signer=signer,
        )
        nonces.add(headers[HEADER_NONCE])
    assert len(nonces) == 500


# Unsigned mode


def test_unsigned_mode_no_aip_headers():
    headers, body = prepare_signed_request(
        "POST", "/x", {"a": 1}, api_key="obx_test_k", agent_did=None, signer=None,
    )
    for h in (HEADER_DID, HEADER_TIMESTAMP, HEADER_NONCE, HEADER_SIGNATURE,
              HEADER_BODY_SHA256):
        assert h not in headers
    # base auth headers still present, body still serialized
    assert headers["Authorization"] == "Bearer obx_test_k"
    assert body == b'{"a":1}'


def test_did_without_signer_stays_unsigned():
    headers, _ = prepare_signed_request(
        "POST", "/x", {"a": 1}, api_key="k", agent_did=TEST_DID, signer=None,
    )
    assert HEADER_SIGNATURE not in headers


# Transports send content= (never json=)


def test_send_sync_posts_content_bytes():
    from unittest.mock import MagicMock

    client = MagicMock()
    send_sync(client, "http://h/x", {"Authorization": "Bearer k"}, b'{"a":1}')
    client.post.assert_called_once()
    assert client.post.call_args.kwargs["content"] == b'{"a":1}'
    assert "json" not in client.post.call_args.kwargs


@pytest.mark.asyncio
async def test_send_async_posts_content_bytes():
    from unittest.mock import AsyncMock

    client = AsyncMock()
    await send_async(client, "http://h/x", {"Authorization": "Bearer k"}, b'{"a":1}')
    client.post.assert_awaited_once()
    assert client.post.call_args.kwargs["content"] == b'{"a":1}'
    assert "json" not in client.post.call_args.kwargs


# Sandbox import isolation — cryptography/request_signing must NOT load eagerly


def test_workflow_import_path_excludes_crypto_and_signing():
    import subprocess
    import sys

    code = (
        "import sys;"
        "import openbox, openbox.workflow_interceptor, openbox.types;"
        "m=set(sys.modules);"
        "cryptol=[x for x in m if x=='cryptography' or x.startswith('cryptography.')];"
        "assert not cryptol, cryptol;"
        "assert 'openbox.request_signing' not in m;"
        "print('ok')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout
