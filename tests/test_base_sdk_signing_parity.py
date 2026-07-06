"""Byte-parity gate: openbox.request_signing vs the base-SDK golden fixture.

The fixture lives at tests/signing/golden_temporal_signed_request.json in
openbox-sdk-python and pins the exact bytes for a fixed timestamp and nonce.
"""

from __future__ import annotations

import base64
import hashlib
import json
import pathlib
import types
from unittest import mock

import pytest

FIXTURE_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "openbox-sdk-python"
    / "tests"
    / "signing"
    / "golden_temporal_signed_request.json"
)

pytestmark = pytest.mark.skipif(
    not FIXTURE_PATH.exists(), reason="base-SDK golden fixture not present"
)


@pytest.fixture(scope="module")
def fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


@pytest.fixture(scope="module")
def signer(fixture):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    return Ed25519PrivateKey.from_private_bytes(base64.b64decode(fixture["seed_b64"]))


def _sign_with_local_module(fixture, signer):
    """Run openbox.request_signing with the fixture's fixed timestamp+nonce."""
    from openbox import request_signing

    class FixedDatetime:
        @staticmethod
        def now(tz=None):
            import datetime as _dt

            return _dt.datetime.fromisoformat(fixture["timestamp"])

    fixed_secrets = types.SimpleNamespace(token_urlsafe=lambda n=32: fixture["nonce"])

    with mock.patch.object(request_signing, "datetime", FixedDatetime), mock.patch.object(
        request_signing, "secrets", fixed_secrets
    ):
        return request_signing.prepare_signed_request(
            fixture["method"],
            fixture["path"],
            fixture["payload"],
            api_key=fixture["api_key"],
            agent_did=fixture["agent_did"],
            signer=signer,
        )


class TestSignedBytesParity:
    def test_body_bytes_exact(self, fixture, signer):
        _, body = _sign_with_local_module(fixture, signer)
        assert body == base64.b64decode(fixture["body_b64"])

    def test_signed_headers_exact(self, fixture, signer):
        headers, body = _sign_with_local_module(fixture, signer)
        for name, expected in fixture["signed_headers"].items():
            assert headers[name] == expected, f"{name} diverged from golden fixture"
        assert hashlib.sha256(body).hexdigest() == fixture["body_sha256"]

    def test_signing_timestamp_keeps_offset(self, fixture, signer):
        headers, _ = _sign_with_local_module(fixture, signer)
        assert headers["X-OpenBox-Agent-Timestamp"].endswith("+00:00")

    def test_module_constants_match_base_sdk(self):
        from openbox import request_signing
        from openbox_core import identity as core_identity

        assert request_signing.EMPTY_BODY_SHA256 == core_identity.EMPTY_BODY_SHA256
        assert request_signing.HEADER_DID == core_identity.HEADER_DID
        assert request_signing.HEADER_TIMESTAMP == core_identity.HEADER_TIMESTAMP
        assert request_signing.HEADER_NONCE == core_identity.HEADER_NONCE
        assert request_signing.HEADER_SIGNATURE == core_identity.HEADER_SIGNATURE
        assert request_signing.HEADER_BODY_SHA256 == core_identity.HEADER_BODY_SHA256

    def test_serialize_body_matches_base_sdk(self, fixture):
        from openbox import request_signing
        from openbox_core.serialization import serialize_body

        assert request_signing.serialize_body(fixture["payload"]) == serialize_body(
            fixture["payload"]
        )
        assert request_signing.serialize_body(None) == b""
