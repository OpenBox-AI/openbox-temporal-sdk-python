"""Config-level DID/key validation and signing error-mapping tests."""

import base64
import os

import pytest

from openbox.config import (
    _GlobalConfig,
    _validate_did,
    initialize,
)
from openbox_core.identity import load_ed25519_seed as _load_ed25519_seed
from openbox.errors import (
    OpenBoxConfigError,
    OpenBoxSigningError,
    map_signing_error,
)
VALID_DID = "did:aip:12345678-1234-1234-1234-1234567890ab"


def _valid_seed_b64() -> str:
    return base64.b64encode(os.urandom(32)).decode("ascii")


# DID validation


def test_validate_did_accepts_valid():
    _validate_did(VALID_DID)  # no raise


@pytest.mark.parametrize(
    "bad",
    [
        "did:web:example.com",
        "aip:12345678-1234-1234-1234-1234567890ab",
        "did:aip:short",
        "did:aip:ZZZZZZZZ-1234-1234-1234-1234567890ab",  # non-hex
        "did:aip:12345678-1234-1234-1234-1234567890ag",  # trailing non-hex
        "",
    ],
)
def test_validate_did_rejects_bad(bad):
    with pytest.raises(OpenBoxConfigError):
        _validate_did(bad)


# Ed25519 seed loading


def test_load_seed_valid_32_bytes():
    key = _load_ed25519_seed(_valid_seed_b64())
    # loaded object can sign
    sig = key.sign(b"hello")
    key.public_key().verify(sig, b"hello")


def test_load_seed_rejects_bad_base64():
    with pytest.raises(OpenBoxConfigError) as ei:
        _load_ed25519_seed("not valid base64 !!!")
    assert "base64" in str(ei.value)


def test_load_seed_rejects_wrong_length():
    short = base64.b64encode(os.urandom(16)).decode("ascii")
    with pytest.raises(OpenBoxConfigError) as ei:
        _load_ed25519_seed(short)
    assert "32-byte" in str(ei.value)


def test_seed_errors_never_echo_key_bytes():
    secret = base64.b64encode(os.urandom(16)).decode("ascii")  # wrong length
    with pytest.raises(OpenBoxConfigError) as ei:
        _load_ed25519_seed(secret)
    assert secret not in str(ei.value)


# initialize() both-or-neither + repr safety


def test_initialize_requires_both_did_and_key():
    with pytest.raises(OpenBoxConfigError) as ei:
        initialize(
            api_url="http://localhost:9999",
            api_key="obx_test_abc",
            validate=False,
            agent_did=VALID_DID,  # key missing
        )
    assert "together" in str(ei.value)


def test_initialize_loads_signer(monkeypatch):
    initialize(
        api_url="http://localhost:9999",
        api_key="obx_test_abc",
        validate=False,
        agent_did=VALID_DID,
        agent_private_key=_valid_seed_b64(),
    )
    from openbox.config import get_global_config

    cfg = get_global_config()
    assert cfg.has_signing() is True
    assert cfg.get_signer() is not None
    assert cfg.agent_did == VALID_DID


def test_global_config_repr_excludes_key_and_signer():
    cfg = _GlobalConfig()
    seed_b64 = _valid_seed_b64()
    signer = _load_ed25519_seed(seed_b64)
    cfg.configure(
        api_url="http://h",
        api_key="obx_live_secretkey1234",
        agent_did=VALID_DID,
        signer=signer,
    )
    r = repr(cfg)
    # neither the raw seed nor the API key body appears
    assert seed_b64 not in r
    assert "secretkey1234" not in r
    assert "signing=enabled" in r


def test_has_signing_false_without_credentials():
    cfg = _GlobalConfig()
    cfg.configure(api_url="http://h", api_key="obx_test_k")
    assert cfg.has_signing() is False
    assert cfg.get_signer() is None


# Signing error mapping


@pytest.mark.parametrize(
    "code",
    [
        "signature_invalid",
        "nonce_replayed",
        "did_agent_mismatch",
        "verifier_not_configured",
        "timestamp_skew",
    ],
)
def test_map_signing_error_known_codes(code):
    err = map_signing_error(code)
    assert isinstance(err, OpenBoxSigningError)
    assert err.reason_code == code
    assert code in str(err)  # message references the code


def test_map_signing_error_unknown_code_falls_back():
    err = map_signing_error("some_new_code")
    assert isinstance(err, OpenBoxSigningError)
    assert err.reason_code == "some_new_code"


def test_map_signing_error_none():
    err = map_signing_error(None)
    assert isinstance(err, OpenBoxSigningError)


# P1: manual-path constructors inherit the global signer (no unsigned leak)


def test_governance_client_inherits_global_signer():
    """A manually-built GovernanceClient must sign when initialize() enabled signing."""
    from openbox.client import GovernanceClient

    initialize(
        api_url="http://localhost:9999",
        api_key="obx_test_abc",
        validate=False,
        agent_did=VALID_DID,
        agent_private_key=_valid_seed_b64(),
    )
    client = GovernanceClient(api_url="http://localhost:9999", api_key="obx_test_abc")
    assert client._signer is not None
    assert client._agent_did == VALID_DID


def test_governance_client_unsigned_when_no_global_signing():
    from openbox.client import GovernanceClient

    initialize(
        api_url="http://localhost:9999",
        api_key="obx_test_abc",
        validate=False,
    )
    client = GovernanceClient(api_url="http://localhost:9999", api_key="obx_test_abc")
    assert client._signer is None and client._agent_did is None


def test_build_activities_inherits_global_signer():
    """Manual build_governance_activities must inherit the global signer/DID."""
    from openbox.activities import build_governance_activities

    initialize(
        api_url="http://localhost:9999",
        api_key="obx_test_abc",
        validate=False,
        agent_did=VALID_DID,
        agent_private_key=_valid_seed_b64(),
    )
    acts = build_governance_activities(
        api_url="http://localhost:9999", api_key="obx_test_abc"
    )
    assert acts._signer is not None and acts._agent_did == VALID_DID


def test_explicit_signer_overrides_global():
    from openbox.client import GovernanceClient

    initialize(
        api_url="http://localhost:9999",
        api_key="obx_test_abc",
        validate=False,
        agent_did=VALID_DID,
        agent_private_key=_valid_seed_b64(),
    )
    other_did = "did:aip:abcdef01-2345-6789-abcd-ef0123456789"
    other_signer = _load_ed25519_seed(_valid_seed_b64())
    client = GovernanceClient(
        api_url="http://h", api_key="k", agent_did=other_did, signer=other_signer
    )
    assert client._agent_did == other_did
    assert client._signer is other_signer


# P3: OpenBoxSigningError is public


def test_signing_error_exported_from_package_root():
    import openbox

    assert hasattr(openbox, "OpenBoxSigningError")
    assert openbox.OpenBoxSigningError is OpenBoxSigningError
    # subclass of OpenBoxAuthError → existing catchers still work
    from openbox import OpenBoxAuthError

    assert issubclass(openbox.OpenBoxSigningError, OpenBoxAuthError)
