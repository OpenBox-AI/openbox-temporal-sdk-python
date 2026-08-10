"""Governed-command receipt verification.

The signed verifier remains fail closed. The separately named local verifier is
explicitly insecure and exists only for local/offline testing.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .types import GovernedCommandReceipt, GovernedCommandRequest

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ED25519_SIGNATURE = re.compile(r"[0-9a-f]{128}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_WORKFLOW_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}\Z")
_MAX_RECEIPT_LIFETIME = timedelta(minutes=10)


class GovernedCommandReceiptError(ValueError):
    """Raised when a command is not authorized by its receipt."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise GovernedCommandReceiptError(
            "governed command receipt rejected"
        ) from error


def _timestamp(value: object) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise GovernedCommandReceiptError("governed command receipt rejected")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise GovernedCommandReceiptError(
            "governed command receipt rejected"
        ) from error
    if parsed.tzinfo is None:
        raise GovernedCommandReceiptError("governed command receipt rejected")
    return parsed.astimezone(UTC)


def request_arguments_sha256(request: GovernedCommandRequest) -> str:
    """Hash the exact typed operation and ordered safe input snapshot."""
    if not isinstance(request, GovernedCommandRequest):
        raise GovernedCommandReceiptError("governed command receipt rejected")
    return hashlib.sha256(
        _canonical(
            {
                "profile_id": request.profile_id,
                "arguments": [
                    {"name": item.name, "value": item.value}
                    for item in request.arguments
                ],
            }
        )
    ).hexdigest()


def command_sha256(argv: Sequence[str]) -> str:
    """Hash derived argv without placing argv itself in a receipt or history."""
    if (
        isinstance(argv, (str, bytes))
        or not isinstance(argv, Sequence)
        or not argv
        or not all(isinstance(value, str) for value in argv)
    ):
        raise GovernedCommandReceiptError("governed command receipt rejected")
    return hashlib.sha256(_canonical(list(argv))).hexdigest()


def asset_bundle_sha256(asset_bundle: object) -> str:
    """Hash the trusted runtime asset identity without exposing policy bodies."""
    to_wire = getattr(asset_bundle, "to_wire", None)
    value = to_wire() if callable(to_wire) else asset_bundle
    if not isinstance(value, Mapping):
        raise GovernedCommandReceiptError("governed command receipt rejected")
    return hashlib.sha256(_canonical(dict(value))).hexdigest()


def receipt_binding(
    request: GovernedCommandRequest,
    *,
    command_argv: Sequence[str],
    asset_bundle: object,
    profile_fingerprint: str,
) -> dict[str, str]:
    """Derive the safe binding fields independently used by agent and Worker."""
    if (
        not isinstance(profile_fingerprint, str)
        or _SHA256.fullmatch(profile_fingerprint) is None
    ):
        raise GovernedCommandReceiptError("governed command receipt rejected")
    return {
        "arguments_sha256": request_arguments_sha256(request),
        "command_sha256": command_sha256(command_argv),
        "asset_bundle_sha256": asset_bundle_sha256(asset_bundle),
        "profile_fingerprint": profile_fingerprint,
    }


def receipt_payload(receipt: GovernedCommandReceipt) -> dict[str, Any]:
    """Return the canonical signed fields, excluding the signature itself."""
    return {
        "schema_version": receipt.schema_version,
        "receipt_id": receipt.receipt_id,
        "nonce": receipt.nonce,
        "workflow_id": receipt.workflow_id,
        "verdict": receipt.verdict,
        "profile_id": receipt.profile_id,
        "arguments_sha256": receipt.arguments_sha256,
        "command_sha256": receipt.command_sha256,
        "asset_bundle_sha256": receipt.asset_bundle_sha256,
        "profile_fingerprint": receipt.profile_fingerprint,
        "issued_at": receipt.issued_at,
        "expires_at": receipt.expires_at,
        "key_id": receipt.key_id,
    }


@dataclass(frozen=True, slots=True)
class _VerificationErrors:
    required: str
    rejected: str
    verifier_rejected: str
    consumed: str


_SIGNED_ERRORS = _VerificationErrors(
    required="governed command receipt required",
    rejected="governed command receipt rejected",
    verifier_rejected="receipt verifier rejected",
    consumed="governed command receipt already consumed",
)
_INSECURE_LOCAL_ERRORS = _VerificationErrors(
    required="INSECURE LOCAL unsigned governed command receipt required",
    rejected="INSECURE LOCAL unsigned governed command receipt rejected",
    verifier_rejected="INSECURE LOCAL receipt verifier rejected",
    consumed="INSECURE LOCAL unsigned governed command receipt already consumed",
)


def _validate_clock(clock: Callable[[], datetime], errors: _VerificationErrors) -> None:
    if not callable(clock):
        raise GovernedCommandReceiptError(errors.verifier_rejected)


def _validate_common_receipt(
    request: GovernedCommandRequest,
    *,
    expected_workflow_id: str,
    command_argv: Sequence[str],
    asset_bundle: object,
    profile_fingerprint: str,
    errors: _VerificationErrors,
) -> GovernedCommandReceipt:
    """Validate every receipt property unrelated to key authentication."""
    if not isinstance(request, GovernedCommandRequest):
        raise GovernedCommandReceiptError(errors.rejected)
    receipt = request.receipt
    if receipt is None:
        raise GovernedCommandReceiptError(errors.required)
    if not isinstance(receipt, GovernedCommandReceipt):
        raise GovernedCommandReceiptError(errors.rejected)
    try:
        binding = receipt_binding(
            request,
            command_argv=command_argv,
            asset_bundle=asset_bundle,
            profile_fingerprint=profile_fingerprint,
        )
    except GovernedCommandReceiptError as error:
        if str(error) == errors.rejected:
            raise
        raise GovernedCommandReceiptError(errors.rejected) from error

    common_strings = (
        receipt.receipt_id,
        receipt.nonce,
        receipt.workflow_id,
        receipt.profile_id,
        receipt.arguments_sha256,
        receipt.command_sha256,
        receipt.asset_bundle_sha256,
        receipt.profile_fingerprint,
        receipt.issued_at,
        receipt.expires_at,
    )
    if (
        type(receipt.schema_version) is not int
        or receipt.schema_version != 1
        or receipt.verdict != "constrain"
        or not all(isinstance(value, str) and value for value in common_strings)
        or _IDENTIFIER.fullmatch(receipt.receipt_id) is None
        or _IDENTIFIER.fullmatch(receipt.nonce) is None
        or _WORKFLOW_ID.fullmatch(receipt.workflow_id) is None
        or not isinstance(expected_workflow_id, str)
        or _WORKFLOW_ID.fullmatch(expected_workflow_id) is None
        or receipt.workflow_id != expected_workflow_id
        or _IDENTIFIER.fullmatch(receipt.profile_id) is None
        or receipt.profile_id != request.profile_id
        or _SHA256.fullmatch(receipt.arguments_sha256) is None
        or _SHA256.fullmatch(receipt.command_sha256) is None
        or _SHA256.fullmatch(receipt.asset_bundle_sha256) is None
        or _SHA256.fullmatch(receipt.profile_fingerprint) is None
        or receipt.arguments_sha256 != binding["arguments_sha256"]
        or receipt.command_sha256 != binding["command_sha256"]
        or receipt.asset_bundle_sha256 != binding["asset_bundle_sha256"]
        or receipt.profile_fingerprint != binding["profile_fingerprint"]
    ):
        raise GovernedCommandReceiptError(errors.rejected)

    return receipt


def _validate_common_time_window(
    receipt: GovernedCommandReceipt,
    *,
    clock: Callable[[], datetime],
    errors: _VerificationErrors,
) -> None:
    """Validate the shared issued/expiry window against a verifier clock."""
    try:
        issued_at = _timestamp(receipt.issued_at)
        expires_at = _timestamp(receipt.expires_at)
    except GovernedCommandReceiptError as error:
        if str(error) == errors.rejected:
            raise
        raise GovernedCommandReceiptError(errors.rejected) from error
    now = clock()
    if not isinstance(now, datetime) or now.tzinfo is None:
        raise GovernedCommandReceiptError(errors.verifier_rejected)
    now = now.astimezone(UTC)
    lifetime = expires_at - issued_at
    if (
        issued_at > now
        or expires_at <= now
        or issued_at >= expires_at
        or lifetime > _MAX_RECEIPT_LIFETIME
    ):
        raise GovernedCommandReceiptError(errors.rejected)


def _consume_receipt(
    receipt: GovernedCommandReceipt,
    *,
    consumed_receipt_ids: set[str],
    consumed_nonces: set[str],
    lock: threading.Lock,
    error_message: str,
) -> str:
    """Atomically consume one validated receipt ID and nonce in this process."""
    with lock:
        if (
            receipt.receipt_id in consumed_receipt_ids
            or receipt.nonce in consumed_nonces
        ):
            raise GovernedCommandReceiptError(error_message)
        consumed_receipt_ids.add(receipt.receipt_id)
        consumed_nonces.add(receipt.nonce)
    return receipt.receipt_id


@dataclass(repr=False)
class GovernedCommandReceiptVerifier:
    """Ed25519 verifier with atomic in-process one-time receipt consumption."""

    key_id: str
    public_key: bytes
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    _consumed_receipt_ids: set[str] = field(default_factory=set, init=False, repr=False)
    _consumed_nonces: set[str] = field(default_factory=set, init=False, repr=False)
    _consumption_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.key_id, str)
            or _IDENTIFIER.fullmatch(self.key_id) is None
            or not isinstance(self.public_key, bytes)
            or len(self.public_key) != 32
        ):
            raise GovernedCommandReceiptError(_SIGNED_ERRORS.verifier_rejected)
        _validate_clock(self.clock, _SIGNED_ERRORS)
        try:
            Ed25519PublicKey.from_public_bytes(self.public_key)
        except ValueError as error:
            raise GovernedCommandReceiptError(
                _SIGNED_ERRORS.verifier_rejected
            ) from error

    def verify(
        self,
        request: GovernedCommandRequest,
        *,
        expected_workflow_id: str,
        command_argv: Sequence[str],
        asset_bundle: object,
        profile_fingerprint: str,
    ) -> str:
        receipt = _validate_common_receipt(
            request,
            expected_workflow_id=expected_workflow_id,
            command_argv=command_argv,
            asset_bundle=asset_bundle,
            profile_fingerprint=profile_fingerprint,
            errors=_SIGNED_ERRORS,
        )
        if (
            not isinstance(receipt.key_id, str)
            or not receipt.key_id
            or receipt.key_id != self.key_id
            or not isinstance(receipt.signature, str)
            or not receipt.signature
            or _ED25519_SIGNATURE.fullmatch(receipt.signature) is None
        ):
            raise GovernedCommandReceiptError(_SIGNED_ERRORS.rejected)
        _validate_common_time_window(receipt, clock=self.clock, errors=_SIGNED_ERRORS)
        try:
            Ed25519PublicKey.from_public_bytes(self.public_key).verify(
                bytes.fromhex(receipt.signature), _canonical(receipt_payload(receipt))
            )
        except (InvalidSignature, ValueError) as error:
            raise GovernedCommandReceiptError(_SIGNED_ERRORS.rejected) from error

        # Consumption happens only after every structural, binding, temporal, and
        # cryptographic check passes. The lock makes concurrent duplicate use
        # fail closed before any caller can begin sandbox creation.
        return _consume_receipt(
            receipt,
            consumed_receipt_ids=self._consumed_receipt_ids,
            consumed_nonces=self._consumed_nonces,
            lock=self._consumption_lock,
            error_message=_SIGNED_ERRORS.consumed,
        )

    def __repr__(self) -> str:
        return (
            f"GovernedCommandReceiptVerifier(key_id={self.key_id!r}, "
            "public_key=<redacted>, replay_protection=in_process)"
        )


@dataclass(repr=False)
class InsecureLocalReceiptVerifier:
    """INSECURE local/testing-only verifier that ignores receipt signatures.

    This verifier is never selected automatically and must be constructed
    explicitly. It performs the same schema, binding, time-window, lifetime,
    and atomic in-process replay checks as the signed verifier, but deliberately
    does not require a Core key. ``key_id`` is syntactically required but its
    value is not authenticated, and any string ``signature`` (including an
    empty unsigned value) is ignored. Never use this verifier in production.
    """

    clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    _consumed_receipt_ids: set[str] = field(default_factory=set, init=False, repr=False)
    _consumed_nonces: set[str] = field(default_factory=set, init=False, repr=False)
    _consumption_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def __post_init__(self) -> None:
        _validate_clock(self.clock, _INSECURE_LOCAL_ERRORS)

    def verify(
        self,
        request: GovernedCommandRequest,
        *,
        expected_workflow_id: str,
        command_argv: Sequence[str],
        asset_bundle: object,
        profile_fingerprint: str,
    ) -> str:
        """Validate and consume one unsigned local/testing receipt."""
        receipt = _validate_common_receipt(
            request,
            expected_workflow_id=expected_workflow_id,
            command_argv=command_argv,
            asset_bundle=asset_bundle,
            profile_fingerprint=profile_fingerprint,
            errors=_INSECURE_LOCAL_ERRORS,
        )
        if (
            not isinstance(receipt.key_id, str)
            or not receipt.key_id
            or _IDENTIFIER.fullmatch(receipt.key_id) is None
            or not isinstance(receipt.signature, str)
        ):
            raise GovernedCommandReceiptError(_INSECURE_LOCAL_ERRORS.rejected)

        _validate_common_time_window(
            receipt, clock=self.clock, errors=_INSECURE_LOCAL_ERRORS
        )

        # The caller invokes verification before dispatcher sandbox creation.
        # Consume only after all retained checks pass, and atomically reject
        # concurrent duplicate receipt IDs or nonces in this process.
        return _consume_receipt(
            receipt,
            consumed_receipt_ids=self._consumed_receipt_ids,
            consumed_nonces=self._consumed_nonces,
            lock=self._consumption_lock,
            error_message=_INSECURE_LOCAL_ERRORS.consumed,
        )

    def __repr__(self) -> str:
        return (
            "InsecureLocalReceiptVerifier("
            "mode=INSECURE_LOCAL_UNSIGNED_TESTING_ONLY, "
            "signature_verification=disabled, "
            "replay_protection=in_process)"
        )
