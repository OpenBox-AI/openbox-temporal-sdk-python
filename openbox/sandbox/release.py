"""Approved sandbox release identity owned by the SDK distribution.

The SDK — not the local agent and not customer configuration — decides which
sandbox runtime identity and OpenShell policy are acceptable. A release build
of the SDK must install exactly one :class:`ApprovedSandboxRelease` through
:func:`install_approved_sandbox_release` (normally from a private packaging
module at import time). Until that happens, resolving the release fails
closed with the constant public error; the SDK never silently accepts an
agent-advertised template, adapter, or policy.
"""

from __future__ import annotations

import hashlib
import re
import threading
from dataclasses import dataclass

from .errors import GovernedCommandDeploymentError

_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IMAGE = re.compile(r"[^\s]+@sha256:[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_MAX_POLICY_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True, repr=False)
class ApprovedSandboxRelease:
    """One exact, immutable sandbox runtime + policy identity."""

    runtime_contract_version: int
    adapter_build_sha256: str
    template: str
    policy_id: str
    policy_version: int
    policy_media_type: str
    policy_body: bytes
    compatibility_id: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.runtime_contract_version, bool)
            or not isinstance(self.runtime_contract_version, int)
            or self.runtime_contract_version < 1
            or not isinstance(self.adapter_build_sha256, str)
            or _HEX_SHA256.fullmatch(self.adapter_build_sha256) is None
            or not isinstance(self.template, str)
            or _IMAGE.fullmatch(self.template) is None
            or not isinstance(self.policy_id, str)
            or _IDENTIFIER.fullmatch(self.policy_id) is None
            or isinstance(self.policy_version, bool)
            or not isinstance(self.policy_version, int)
            or self.policy_version < 1
            or not isinstance(self.policy_media_type, str)
            or not 0 < len(self.policy_media_type.encode()) <= 128
            or not isinstance(self.policy_body, bytes)
            or not 0 < len(self.policy_body) <= _MAX_POLICY_BYTES
            or not isinstance(self.compatibility_id, str)
            or _IDENTIFIER.fullmatch(self.compatibility_id) is None
        ):
            raise GovernedCommandDeploymentError()

    @property
    def policy_sha256(self) -> str:
        return hashlib.sha256(self.policy_body).hexdigest()

    def __repr__(self) -> str:
        return (
            "ApprovedSandboxRelease("
            f"runtime_contract_version={self.runtime_contract_version}, "
            f"template=<digest-pinned>, policy_id={self.policy_id!r}, "
            f"policy_version={self.policy_version}, "
            f"compatibility_id={self.compatibility_id!r}, policy_body=<redacted>)"
        )


_lock = threading.Lock()
_installed: ApprovedSandboxRelease | None = None


def install_approved_sandbox_release(release: ApprovedSandboxRelease) -> None:
    """Install the one process-wide approved release (packaging/test seam).

    Reinstalling the identical release is a no-op; installing a different
    release after one is already active fails closed.
    """
    if not isinstance(release, ApprovedSandboxRelease):
        raise GovernedCommandDeploymentError()
    global _installed
    with _lock:
        if _installed is not None and _installed != release:
            raise GovernedCommandDeploymentError()
        _installed = release


def _clear_approved_sandbox_release_for_testing() -> None:
    global _installed
    with _lock:
        _installed = None


def approved_sandbox_release() -> ApprovedSandboxRelease:
    """Return the installed approved release, failing closed when absent."""
    with _lock:
        if _installed is None:
            raise GovernedCommandDeploymentError()
        return _installed
