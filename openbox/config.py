"""
OpenBox SDK - Configuration for workflow-boundary governance (SPEC-003).

GovernanceConfig: Configuration for interceptors
Global config singleton with initialize() function

IMPORTANT: No module-level logging import! Python's logging module uses
linecache -> os.stat which triggers Temporal sandbox restrictions.
"""

import re
from dataclasses import dataclass, field
from typing import Set, Optional

# NOTE: urllib and logging imports are lazy to avoid Temporal sandbox restrictions.
# Both use os.stat internally which triggers sandbox errors.


def _get_logger():
    """Lazy logger to avoid sandbox restrictions."""
    import logging

    return logging.getLogger(__name__)


# API key format pattern (obx_live_... or obx_test_...)
API_KEY_PATTERN = re.compile(r"^obx_(live|test)_\w+$")

# Agent DID prefix; the suffix must be a parseable UUID (validated via uuid.UUID,
# matching Core's UUID parser rather than a loose regex).
AGENT_DID_PREFIX = "did:aip:"


# Re-export from errors.py for backward compatibility
from .errors import (  # noqa: F401
    OpenBoxConfigError,
    OpenBoxAuthError,
    OpenBoxNetworkError,
    OpenBoxInsecureURLError,
)

# GovernanceConfig - Configuration for interceptors


@dataclass
class GovernanceConfig:
    """
    Configuration for governance interceptors.

    Used by both GovernanceInterceptor (workflow-level) and
    ActivityGovernanceInterceptor (activity-level).

    Attributes:
        skip_workflow_types: Workflow types to skip governance for
        skip_signals: Signal names to skip governance for
        enforce_task_queues: Task queues to enforce governance on (None = all)
        on_api_error: Behavior when OpenBox API is unreachable
            - "fail_open" (default) = allow workflow to continue
            - "fail_closed" = deny/stop workflow execution
        api_timeout: Timeout for governance API calls (seconds)
        max_body_size: Maximum body size to capture (None = no limit)
        send_start_event: Send WorkflowStarted event (can disable for performance)
        send_activity_start_event: Send ActivityStarted event (can disable for performance)
        skip_activity_types: Activity types to skip governance for
        hitl_enabled: Enable approval polling for require-approval verdicts (default: True)
        skip_hitl_activity_types: Activity types to skip approval checks (avoids infinite loops)
    """

    # Workflow types to skip governance for
    skip_workflow_types: Set[str] = field(default_factory=set)

    # Signal names to skip governance for
    skip_signals: Set[str] = field(default_factory=set)

    # Task queues to enforce governance on (None = all)
    enforce_task_queues: Optional[Set[str]] = None

    # Behavior when OpenBox API is unreachable
    # "fail_open" (default) = allow workflow to continue
    # "fail_closed" = deny/stop workflow execution
    on_api_error: str = "fail_open"

    # Timeout for governance API calls (seconds)
    api_timeout: float = 30.0

    # Maximum body size to capture in chars (default: 64KB)
    max_body_size: int = 65536

    # Send WorkflowStarted event (can disable for performance)
    send_start_event: bool = True

    # Send ActivityStarted event before each activity (can disable for performance)
    send_activity_start_event: bool = True

    # Activity types to skip governance for
    # By default, skip the governance event activity to avoid infinite loops
    skip_activity_types: Set[str] = field(
        default_factory=lambda: {"send_governance_event"}
    )

    # Approval polling configuration
    # Enable approval polling for require-approval verdicts
    hitl_enabled: bool = True

    # Activity types to skip approval checks (to avoid infinite loops)
    # By default, skip the governance event activity
    skip_hitl_activity_types: Set[str] = field(
        default_factory=lambda: {"send_governance_event"}
    )

    # Reserved for future non-retry polling interval (ms).
    # Temporal currently uses its native retry backoff for HITL polling.
    hitl_poll_interval_ms: int = 5000


# Global Configuration Singleton


def _validate_api_key_format(api_key: str) -> bool:
    """Validate API key format (obx_live_... or obx_test_...)."""
    return bool(API_KEY_PATTERN.match(api_key))


def _validate_did(agent_did: str) -> None:
    """Validate agent DID format (did:aip:<uuid>).

    Parses the suffix with uuid.UUID so malformed UUID layouts fail locally at
    init (matching Core's parser) rather than slipping through to a Core 4xx.

    Raises OpenBoxConfigError on mismatch.
    """
    import uuid

    if not isinstance(agent_did, str) or not agent_did.startswith(AGENT_DID_PREFIX):
        raise OpenBoxConfigError(
            f"Invalid agent DID format. Expected 'did:aip:<uuid>', "
            f"got: '{str(agent_did)[:24]}...' (showing first 24 chars)"
        )
    suffix = agent_did[len(AGENT_DID_PREFIX) :]
    try:
        uuid.UUID(suffix)
    except (ValueError, AttributeError):
        raise OpenBoxConfigError(
            f"Invalid agent DID: '{agent_did[:24]}...' — the part after "
            f"'{AGENT_DID_PREFIX}' is not a valid UUID."
        )


def resolve_signing_defaults(agent_did, signer):
    """Fall back to the global config's loaded DID + signer when both are omitted.

    Ensures a MANUAL SDK setup that called initialize(..., agent_did=, agent_private_key=)
    also signs governance calls — not just the /auth/validate GET. Worker/plugin pass
    explicit values so this is a no-op for them. Falls back only when BOTH are None
    (DID + key are both-or-neither), so it never produces a partial signing state.
    """
    if agent_did is None and signer is None:
        gc = get_global_config()
        return gc.agent_did, gc.get_signer()
    return agent_did, signer


def _load_ed25519_seed(agent_private_key: str):
    """Decode a base64 raw 32-byte Ed25519 seed and load a private key object.

    The provisioned key is a raw 32-byte seed (base64), NOT PKCS8. Returns a
    cryptography Ed25519PrivateKey. Never echoes key bytes in error messages —
    the seed is non-repudiation material.

    Raises OpenBoxConfigError on any failure (bad base64, wrong length, load error).
    """
    import base64

    # cryptography imported lazily — keeps it off any eager import path.
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    try:
        seed = base64.b64decode(agent_private_key, validate=True)
    except Exception:
        raise OpenBoxConfigError(
            "Invalid agent private key: not valid base64 (key bytes not shown)."
        )

    if len(seed) != 32:
        raise OpenBoxConfigError(
            f"Invalid agent private key: expected a 32-byte Ed25519 seed, "
            f"got {len(seed)} bytes (key bytes not shown)."
        )

    try:
        return Ed25519PrivateKey.from_private_bytes(seed)
    except Exception:
        raise OpenBoxConfigError(
            "Invalid agent private key: could not load Ed25519 key (key bytes not shown)."
        )


def _validate_url_security(api_url: str) -> None:
    """
    Validate that non-localhost URLs use HTTPS.

    Raises:
        OpenBoxInsecureURLError: If HTTP is used for non-localhost URLs.
    """
    from urllib.parse import urlparse

    parsed = urlparse(api_url)

    # Allow HTTP only for localhost/127.0.0.1
    is_localhost = parsed.hostname in ("localhost", "127.0.0.1", "::1")

    if parsed.scheme == "http" and not is_localhost:
        raise OpenBoxInsecureURLError(
            f"Insecure HTTP URL detected: {api_url}. "
            "Use HTTPS for non-localhost URLs to protect API keys in transit."
        )


def _extract_reason_code(http_error) -> Optional[str]:
    """Parse Core's JSON error body for a machine reason code, if present.

    Returns the value of "reason_code", "reason", or "code" — whichever exists.
    Safe on non-JSON / unreadable bodies (returns None).
    """
    import json as _json

    try:
        body = http_error.read()
        if not body:
            return None
        data = _json.loads(body.decode("utf-8", errors="replace"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    code = data.get("reason_code") or data.get("code") or data.get("reason")
    return code if isinstance(code, str) else None


def _validate_api_key_with_server(
    api_url: str,
    api_key: str,
    timeout: float,
    *,
    agent_did: Optional[str] = None,
    signer=None,
) -> None:
    """
    Validate API key by calling /v1/auth/validate endpoint.
    Raises OpenBoxAuthError for invalid key, OpenBoxNetworkError for connectivity issues.

    When agent_did + signer are provided, the GET is signed (AIP DID headers) so a
    signing_required=true agent passes verification. The GET carries no body, so the
    body hash is SHA-256(b"").

    NOTE: urllib imports are lazy to avoid Temporal sandbox restrictions.
    urllib.request uses os.stat internally which triggers sandbox errors.
    """
    # Lazy imports to avoid sandbox restrictions
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError, URLError

    # Build signed (or plain) headers via the single source of truth.
    from .request_signing import prepare_signed_request

    headers, _body = prepare_signed_request(
        "GET",
        "/api/v1/auth/validate",
        None,
        api_key=api_key,
        agent_did=agent_did,
        signer=signer,
    )

    try:
        req = Request(
            f"{api_url}/api/v1/auth/validate",
            headers={**headers, "Content-Type": "application/json"},
            method="GET",
        )

        with urlopen(req, timeout=timeout) as response:
            status_code = response.getcode()
            if status_code != 200:
                raise OpenBoxAuthError(
                    "Invalid API key. Check your API key at dashboard.openbox.ai"
                )
            _get_logger().info("OpenBox API key validated successfully")

    except HTTPError as e:
        if e.code == 401 or e.code == 403:
            # When signing is enabled, surface Core's machine reason code as an
            # actionable signing error (signature_invalid, nonce_replayed, ...).
            reason_code = _extract_reason_code(e) if signer is not None else None
            if reason_code:
                from .errors import map_signing_error

                raise map_signing_error(reason_code)
            raise OpenBoxAuthError(
                "Invalid API key. Check your API key at dashboard.openbox.ai"
            )
        raise OpenBoxNetworkError(
            f"Cannot reach OpenBox Core at {api_url}: HTTP {e.code}"
        )

    except URLError as e:
        raise OpenBoxNetworkError(f"Cannot reach OpenBox Core at {api_url}: {e.reason}")

    except (OpenBoxAuthError, OpenBoxNetworkError):
        raise

    except Exception as e:
        raise OpenBoxNetworkError(f"Cannot reach OpenBox Core at {api_url}: {e}")


class _GlobalConfig:
    """Global OpenBox configuration singleton."""

    def __init__(self):
        self.api_url: str = ""
        self.api_key: str = ""
        self.governance_timeout: float = 30.0
        self.agent_did: Optional[str] = None
        # Loaded Ed25519PrivateKey object — NEVER the raw seed bytes/string.
        self._signer = None

    def configure(
        self,
        api_url: str,
        api_key: str,
        governance_timeout: float = 30.0,
        *,
        agent_did: Optional[str] = None,
        signer=None,
    ):
        """Configure OpenBox settings.

        signer is a pre-loaded Ed25519PrivateKey object (loaded by initialize()),
        never the raw seed. agent_did is the public identity asserted in signed headers.
        """
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.governance_timeout = governance_timeout
        self.agent_did = agent_did
        self._signer = signer

    def is_configured(self) -> bool:
        """Check if OpenBox is configured."""
        return bool(self.api_url and self.api_key)

    def has_signing(self) -> bool:
        """True when a DID + Ed25519 signer are loaded (signed requests enabled)."""
        return bool(self.agent_did and self._signer is not None)

    def get_signer(self):
        """Return the loaded Ed25519PrivateKey signer (or None)."""
        return self._signer

    def __repr__(self) -> str:
        """Return string representation with masked API key. NEVER includes the key."""
        if self.api_key and len(self.api_key) > 8:
            masked_key = f"obx_****{self.api_key[-4:]}"
        elif self.api_key:
            masked_key = "****"
        else:
            masked_key = ""
        return (
            f"_GlobalConfig(api_url={self.api_url!r}, "
            f"api_key={masked_key!r}, "
            f"governance_timeout={self.governance_timeout}, "
            f"agent_did={self.agent_did!r}, "
            f"signing={'enabled' if self.has_signing() else 'disabled'})"
        )


# Global singleton
_config = _GlobalConfig()


def get_global_config() -> _GlobalConfig:
    """Get global config singleton."""
    return _config


def initialize(
    api_url: str,
    api_key: str,
    governance_timeout: float = 30.0,
    validate: bool = True,
    *,
    agent_did: Optional[str] = None,
    agent_private_key: Optional[str] = None,
) -> None:
    """
    Initialize OpenBox SDK.

    Args:
        api_url: OpenBox Core API endpoint URL
        api_key: API key (format: obx_live_... or obx_test_...)
        governance_timeout: Timeout for governance requests in seconds (default: 30.0)
        validate: Validate API key with server on initialization (default: True)
        agent_did: Agent DID (format: did:aip:<uuid>). Asserted in signed request
            headers. Pair with agent_private_key (both-or-neither).
        agent_private_key: Base64 raw 32-byte Ed25519 seed. Signs every Core request
            locally. Non-repudiation material — never logged or stored as raw bytes.

    Raises:
        OpenBoxAuthError: Invalid API key
        OpenBoxConfigError: Invalid DID / private key, or only one of the pair set
        OpenBoxNetworkError: Cannot reach OpenBox Core

    Note:
        Most users do not call ``initialize()`` directly — use
        ``create_openbox_worker(...)`` or ``OpenBoxPlugin(...)``, which validate
        the key, build and own the base ``OpenBoxRuntime`` (installing all
        HTTP/DB/file/function hook instrumentation), and wire the interceptors.

    Example:
        from openbox import create_openbox_worker

        worker = create_openbox_worker(
            client=client,
            task_queue="my-queue",
            workflows=[MyWorkflow],
            activities=[my_activity],
            openbox_url="https://api.openbox.ai",
            openbox_api_key="obx_live_...",
        )
    """
    # Validate URL security (HTTPS required for non-localhost)
    _validate_url_security(api_url)

    # Validate API key format
    if not _validate_api_key_format(api_key):
        raise OpenBoxAuthError(
            f"Invalid API key format. Expected 'obx_live_*' or 'obx_test_*', "
            f"got: '{api_key[:15]}...' (showing first 15 chars)"
        )

    # DID + private key: both-or-neither, then validate + load the signer once.
    if bool(agent_did) != bool(agent_private_key):
        raise OpenBoxConfigError(
            "agent_did and agent_private_key must be provided together "
            "(got only one). Provide both to enable signed requests, or neither."
        )

    signer = None
    if agent_did and agent_private_key:
        _validate_did(agent_did)
        signer = _load_ed25519_seed(agent_private_key)

    _config.configure(
        api_url=api_url,
        api_key=api_key,
        governance_timeout=governance_timeout,
        agent_did=agent_did,
        signer=signer,
    )

    # Validate API key with server (signed when signing is configured).
    if validate:
        _validate_api_key_with_server(
            api_url.rstrip("/"),
            api_key,
            governance_timeout,
            agent_did=agent_did,
            signer=signer,
        )

    _get_logger().info(
        f"OpenBox SDK initialized with API URL: {api_url} "
        f"(signing={'enabled' if signer else 'disabled'})"
    )
