"""Activity-side AIP Ed25519 signer for dispatcher Core preflight."""

from __future__ import annotations

import base64
import hashlib
import secrets
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from openbox.errors import OpenBoxConfigError


class AipEd25519RequestSigner:
    """Sign exact dispatcher request bytes using the current Core AIP contract.

    The object retains a loaded private-key object, never the raw seed string.
    It structurally implements ``openbox_sandbox.dispatcher.GovernanceRequestSigner``
    without creating a dependency from the standalone dispatcher back to this SDK.
    """

    __slots__ = ("_agent_did", "_clock", "_nonce", "_private_key")

    def __init__(
        self,
        agent_did: str,
        private_key: Any,
        *,
        clock: Callable[[], datetime] | None = None,
        nonce: Callable[[], str] | None = None,
    ) -> None:
        if not isinstance(agent_did, str) or not agent_did.startswith("did:aip:"):
            raise OpenBoxConfigError("AIP request signer configuration rejected")
        try:
            parsed = uuid.UUID(agent_did[len("did:aip:") :])
        except ValueError as error:
            raise OpenBoxConfigError(
                "AIP request signer configuration rejected"
            ) from error
        if str(parsed) != agent_did[len("did:aip:") :] or not callable(
            getattr(private_key, "sign", None)
        ):
            raise OpenBoxConfigError("AIP request signer configuration rejected")
        self._agent_did = agent_did
        self._private_key = private_key
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._nonce = nonce or (lambda: secrets.token_urlsafe(24))

    @classmethod
    def from_base64_seed(
        cls, agent_did: str, agent_private_key: str
    ) -> "AipEd25519RequestSigner":
        """Load a base64 raw 32-byte Ed25519 seed without retaining the string."""
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        try:
            seed = base64.b64decode(agent_private_key, validate=True)
        except Exception as error:
            raise OpenBoxConfigError(
                "AIP request signer configuration rejected"
            ) from error
        if len(seed) != 32:
            raise OpenBoxConfigError("AIP request signer configuration rejected")
        try:
            private_key = Ed25519PrivateKey.from_private_bytes(seed)
        except Exception as error:
            raise OpenBoxConfigError(
                "AIP request signer configuration rejected"
            ) from error
        return cls(agent_did, private_key)

    @property
    def agent_did(self) -> str:
        return self._agent_did

    def sign_headers(self, method: str, path: str, body: bytes) -> dict[str, str]:
        if (
            not isinstance(method, str)
            or not method
            or not isinstance(path, str)
            or not path.startswith("/")
            or not isinstance(body, bytes)
        ):
            raise OpenBoxConfigError("AIP signing input rejected")
        current = self._clock()
        if not isinstance(current, datetime) or current.tzinfo is None:
            raise OpenBoxConfigError("AIP signing clock rejected")
        timestamp = current.astimezone(timezone.utc).isoformat()
        nonce = self._nonce()
        if not isinstance(nonce, str) or not nonce:
            raise OpenBoxConfigError("AIP signing nonce rejected")
        body_sha256 = hashlib.sha256(body).hexdigest()
        canonical = "\n".join(
            [method.upper(), path, timestamp, nonce, body_sha256]
        ).encode()
        signature = base64.b64encode(self._private_key.sign(canonical)).decode()
        return {
            "X-OpenBox-Agent-DID": self._agent_did,
            "X-OpenBox-Agent-Timestamp": timestamp,
            "X-OpenBox-Agent-Nonce": nonce,
            "X-OpenBox-Agent-Signature": signature,
            "X-OpenBox-Body-SHA256": body_sha256,
        }

    def __repr__(self) -> str:
        return (
            f"AipEd25519RequestSigner(agent_did={self._agent_did!r}, "
            "private_key=<redacted>)"
        )
