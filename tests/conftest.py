"""Shared test fixtures for governance tests.

These fixtures cover Temporal framework integration: global-config lifecycle and
transport-agnostic payload decoding.
"""

import json as _json

import pytest


def posted_payload(call) -> dict:
    """Decode the governance payload from a mocked ``.post`` call.

    The signed transport sends ``content=<compact json bytes>`` instead of the
    legacy ``json=<dict>``. This helper transparently reads whichever form the
    call used so payload assertions stay transport-agnostic.
    """
    kw = getattr(call, "kwargs", None) or (call[1] if len(call) > 1 else {})
    if kw.get("json") is not None:
        return kw["json"]
    body = kw.get("content")
    if body is None:
        return {}
    if isinstance(body, (bytes, bytearray)):
        body = body.decode("utf-8")
    return _json.loads(body)


@pytest.fixture(autouse=True)
def reset_global_config():
    """Reset the global _GlobalConfig singleton around every test.

    initialize() mutates a module-level singleton that worker.py reads via
    get_global_config().get_signer(). Without this, a test that enables signing
    leaks the signer/DID into later tests (e.g. worker tests that mock
    validate_api_key and expect signer=None).

    ``_okta_identity`` (v2, proposal §13.7) is included in the same save/
    restore tuple — otherwise a test that configures Okta identity would leak
    it into later tests exactly like the pre-existing signer/DID leak this
    fixture already guards against.
    """
    import openbox.config as _cfg

    cfg = _cfg.get_global_config()
    saved = (
        cfg.api_url,
        cfg.api_key,
        cfg.governance_timeout,
        cfg.agent_did,
        cfg._signer,
        cfg._okta_identity,
    )
    yield
    (
        cfg.api_url,
        cfg.api_key,
        cfg.governance_timeout,
        cfg.agent_did,
        cfg._signer,
        cfg._okta_identity,
    ) = saved
