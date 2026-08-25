"""
OpenBox SDK - Configuration for workflow-boundary governance (SPEC-003).

GovernanceConfig: Configuration for interceptors
Global config singleton with initialize() function

IMPORTANT: No module-level logging import! Python's logging module uses
linecache -> os.stat which triggers Temporal sandbox restrictions.
"""

import re
from dataclasses import dataclass, field
from typing import Optional, Set

from openbox_core.errors import OpenBoxConfigError as CoreOpenBoxConfigError

# NOTE: urllib, logging, and openbox_core.identity imports are lazy to avoid
# Temporal sandbox restrictions. urllib/logging use os.stat internally; identity
# pulls cryptography. None may load on a workflow-sandbox import path
# (openbox/__init__ → worker → config), so identity is imported inside the
# functions that need it and via module __getattr__ for AGENT_DID_PREFIX.
# Guarded by tests/test_workflow_sandbox_import_safety.py.


def _get_logger():
    """Lazy logger to avoid sandbox restrictions."""
    import logging

    return logging.getLogger(__name__)


# API key format pattern (obx_live_... or obx_test_...)
API_KEY_PATTERN = re.compile(r"^obx_(live|test)_\w+$")


# Backward-compatible public constant, owned by the base SDK. Resolved lazily via
# module __getattr__ (PEP 562) so a workflow-sandbox import of openbox.config never
# eagerly loads openbox_core.identity (and its cryptography dependency).
def __getattr__(name: str):
    if name == "AGENT_DID_PREFIX":
        from openbox_core.identity import AGENT_DID_PREFIX as _prefix

        return _prefix
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Re-export from errors.py for backward compatibility
from .errors import (  # noqa: F401
    OpenBoxAuthError,
    OpenBoxConfigError,
    OpenBoxInsecureURLError,
    OpenBoxNetworkError,
    OpenBoxSigningError,
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

    # Maximum Continue-As-New restarts a BLOCK with patch may trigger across the
    # whole workflow chain (bounded input-remediation loop). Applies uniformly to
    # every event origin; must be >= 1.
    max_patch_restarts: int = 3

    def __post_init__(self) -> None:
        # GovernanceConfig is public and can be handed directly to an interceptor,
        # bypassing the factory/plugin — so validation lives on the dataclass (the
        # single source of truth), not only at the call sites. Pure + sandbox-safe:
        # raises the already-imported OpenBoxConfigError, no logging/IO.
        if self.max_patch_restarts < 1:
            raise OpenBoxConfigError(
                "max_patch_restarts must be >= 1 " f"(got {self.max_patch_restarts})"
            )


# Global Configuration Singleton


def _validate_api_key_format(api_key: str) -> bool:
    """Validate API key format (obx_live_... or obx_test_...)."""
    return bool(API_KEY_PATTERN.match(api_key))


def _validate_did(agent_did: str) -> None:
    """Validate agent DID format through the shared base SDK."""
    from openbox_core.identity import validate_agent_did as validate_core_agent_did

    try:
        validate_core_agent_did(agent_did)
    except CoreOpenBoxConfigError as exc:
        raise OpenBoxConfigError(str(exc)) from exc


def resolve_signing_defaults(agent_did, signer, okta_identity=None):
    """Fall back to the global config's loaded identity when ALL THREE identity
    args are omitted.

    Ensures a MANUAL SDK setup that called initialize(..., agent_did=,
    agent_private_key=) or initialize(..., okta_agent_id=, ...) also signs
    governance calls — not just the /auth/validate GET. Worker/plugin pass
    explicit values so this is a no-op for them. Falls back only when
    agent_did, signer, AND okta_identity are all None, so it never produces a
    partial/mixed signing state.

    Returns:
        ``(agent_did, signer, okta_identity)`` — exactly one of
        ``(agent_did, signer)`` or ``okta_identity`` is non-None (or all None
        for legacy_unsigned), never both (proposal §13.1 mutual exclusion).
    """
    if agent_did is None and signer is None and okta_identity is None:
        gc = get_global_config()
        return gc.agent_did, gc.get_signer(), gc.get_okta_identity()
    return agent_did, signer, okta_identity


def _load_ed25519_seed(agent_private_key: str):
    """Decode and load an Ed25519 signer through the shared base SDK."""
    from openbox_core.identity import load_ed25519_seed as load_core_ed25519_seed

    try:
        return load_core_ed25519_seed(agent_private_key)
    except CoreOpenBoxConfigError as exc:
        raise OpenBoxConfigError(str(exc)) from exc


def _load_okta_identity(
    *,
    openbox_agent_id: str,
    organization_id: str,
    deployment_id: str,
    okta_agent_id: str,
    okta_agent_key_id: str,
    okta_agent_private_key: str,
    agent_proof_audience: str,
    okta_agent_algorithm: str = "RS256",
):
    """Build + load an Okta AI Agent (v2) identity through the shared base SDK.

    Entirely delegates RSA key loading, the 2048-bit floor, and algorithm
    allowlisting to ``openbox_core.identity_okta.OktaAgentIdentity`` (proposal
    §13.7) — never re-implemented here.
    """
    from openbox_core.identity_okta import OktaAgentIdentity
    from openbox_core.identity_types import OktaAiAgentIdentityConfig

    try:
        return OktaAgentIdentity.from_config(
            OktaAiAgentIdentityConfig(
                openbox_agent_id=openbox_agent_id,
                organization_id=organization_id,
                deployment_id=deployment_id,
                external_agent_id=okta_agent_id,
                key_id=okta_agent_key_id,
                audience=agent_proof_audience,
                private_key=okta_agent_private_key,
                algorithm=okta_agent_algorithm,  # type: ignore[arg-type]
            )
        )
    except CoreOpenBoxConfigError as exc:
        raise OpenBoxConfigError(str(exc)) from exc


def _bootstrap_okta_identity(
    *,
    api_url: str,
    api_key: str,
    timeout: float,
    okta_agent_private_key: str,
):
    """Resolve an Okta identity from Core using only the local private key.

    The base SDK owns bootstrap response validation, key parsing, thumbprint
    matching, endpoint selection, and request signing. Temporal keeps only the
    resulting loaded identity object; the raw PEM is never copied into the
    global configuration singleton.
    """
    from openbox_core.client import EvaluationClient
    from openbox_core.errors import OpenBoxAuthError as CoreOpenBoxAuthError
    from openbox_core.errors import OpenBoxConfigError as CoreOpenBoxConfigError
    from openbox_core.errors import OpenBoxNetworkError as CoreOpenBoxNetworkError
    from openbox_core.errors import OpenBoxSigningError as CoreOpenBoxSigningError

    from .request_signing import _sdk_identifier

    client = EvaluationClient(
        api_url,
        api_key,
        timeout_seconds=timeout,
        okta_bootstrap_private_key=okta_agent_private_key,
        sdk_version=_sdk_identifier(),
    )
    try:
        client.validate_api_key()
        document = client.identity_metadata()
        if document is None:
            raise OpenBoxConfigError(
                "Okta identity bootstrap completed without identity metadata."
            )
        return _load_okta_identity(
            openbox_agent_id=document.openbox_agent_id,
            organization_id=document.organization_id,
            deployment_id=document.deployment_id,
            okta_agent_id=document.okta.external_agent_id,
            okta_agent_key_id=document.okta.credential_kid,
            okta_agent_private_key=okta_agent_private_key,
            agent_proof_audience=document.assertion_audience,
            okta_agent_algorithm="RS256",
        )
    except CoreOpenBoxSigningError as exc:
        raise OpenBoxSigningError(str(exc), getattr(exc, "reason_code", None)) from exc
    except CoreOpenBoxAuthError as exc:
        raise OpenBoxAuthError(str(exc)) from exc
    except CoreOpenBoxNetworkError as exc:
        raise OpenBoxNetworkError(str(exc)) from exc
    except CoreOpenBoxConfigError as exc:
        raise OpenBoxConfigError(str(exc)) from exc
    finally:
        client.close()


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


def _validate_workload_identity_with_server(
    api_url: str,
    api_key: str,
    timeout: float,
    *,
    workload_private_key: str,
) -> None:
    """Validate the API key and active service account through Core v3."""

    from openbox_core.client import EvaluationClient
    from openbox_core.errors import OpenBoxAuthError as CoreOpenBoxAuthError
    from openbox_core.errors import OpenBoxConfigError as CoreOpenBoxConfigError
    from openbox_core.errors import OpenBoxNetworkError as CoreOpenBoxNetworkError
    from openbox_core.errors import OpenBoxSigningError as CoreOpenBoxSigningError

    from .request_signing import _sdk_identifier

    client = EvaluationClient(
        api_url,
        api_key,
        timeout_seconds=timeout,
        workload_private_key=workload_private_key,
        sdk_version=_sdk_identifier(),
    )
    try:
        client.validate_api_key()
    except CoreOpenBoxSigningError as exc:
        raise OpenBoxSigningError(str(exc), getattr(exc, "reason_code", None)) from exc
    except CoreOpenBoxAuthError as exc:
        raise OpenBoxAuthError(str(exc)) from exc
    except CoreOpenBoxNetworkError as exc:
        raise OpenBoxNetworkError(str(exc)) from exc
    except CoreOpenBoxConfigError as exc:
        raise OpenBoxConfigError(str(exc)) from exc
    finally:
        client.close()


def _validate_api_key_with_server(
    api_url: str,
    api_key: str,
    timeout: float,
    *,
    agent_did: Optional[str] = None,
    signer=None,
    okta_identity=None,
) -> None:
    """
    Validate API key by calling the version-appropriate auth-validate endpoint.
    Raises OpenBoxAuthError for invalid key, OpenBoxNetworkError for connectivity issues.

    When agent_did + signer are provided, the GET is signed (AIP DID headers) so a
    signing_required=true agent passes verification. The GET carries no body, so the
    body hash is SHA-256(b"").

    When okta_identity is provided (v2, proposal §13.7), the GET is signed with an
    RS256 assertion and sent to ``/api/v2/auth/validate`` instead — Core rejects a
    v1 call from an okta_ai_agent-method agent (no-downgrade rule), so an
    Okta-configured agent must never validate against v1.

    NOTE: urllib imports are lazy to avoid Temporal sandbox restrictions.
    urllib.request uses os.stat internally which triggers sandbox errors.
    """
    # Lazy imports to avoid sandbox restrictions
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen

    # Build signed (or plain) headers via the single source of truth.
    if okta_identity is not None:
        from .request_signing import prepare_okta_signed_request

        headers, _body = prepare_okta_signed_request(
            "GET",
            "/api/v2/auth/validate",
            None,
            api_key=api_key,
            okta_identity=okta_identity,
        )
        pathname = "/api/v2/auth/validate"
    else:
        from .request_signing import prepare_signed_request

        headers, _body = prepare_signed_request(
            "GET",
            "/api/v1/auth/validate",
            None,
            api_key=api_key,
            agent_did=agent_did,
            signer=signer,
        )
        pathname = "/api/v1/auth/validate"

    try:
        req = Request(
            f"{api_url}{pathname}",
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
            # v2 (Okta) reason codes (contract §7) are a disjoint, larger set
            # than v1's — delegate to the BASE SDK's own mapper for those
            # rather than duplicating its reason-code table here.
            reason_code = (
                _extract_reason_code(e)
                if (signer is not None or okta_identity is not None)
                else None
            )
            if reason_code:
                if okta_identity is not None:
                    # Reuse the base SDK's v2 reason-code CLASSIFICATION (all 12
                    # contract §7 codes) rather than duplicating that table here
                    # — but always raise THIS package's own exception type, so
                    # callers catching `openbox.errors.OpenBoxSigningError`
                    # work identically for v1 and v2.
                    from openbox_core.errors import (
                        map_signing_error as core_map_signing_error,
                    )

                    core_exc = core_map_signing_error(reason_code)
                    raise OpenBoxSigningError(str(core_exc), reason_code) from core_exc
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
        # Loaded OktaAgentIdentity object (v2, proposal §13.7) — NEVER the raw
        # PEM string. Mutually exclusive with agent_did/_signer.
        self._okta_identity = None

    def configure(
        self,
        api_url: str,
        api_key: str,
        governance_timeout: float = 30.0,
        *,
        agent_did: Optional[str] = None,
        signer=None,
        okta_identity=None,
    ):
        """Configure OpenBox settings.

        signer is a pre-loaded Ed25519PrivateKey object (loaded by initialize()),
        never the raw seed. agent_did is the public identity asserted in signed headers.
        okta_identity is a pre-loaded OktaAgentIdentity object (v2), mutually
        exclusive with agent_did/signer.
        """
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.governance_timeout = governance_timeout
        self.agent_did = agent_did
        self._signer = signer
        self._okta_identity = okta_identity

    def is_configured(self) -> bool:
        """Check if OpenBox is configured."""
        return bool(self.api_url and self.api_key)

    def has_signing(self) -> bool:
        """True when a DID + Ed25519 signer are loaded (signed requests enabled)."""
        return bool(self.agent_did and self._signer is not None)

    def get_signer(self):
        """Return the loaded Ed25519PrivateKey signer (or None)."""
        return self._signer

    def has_okta_identity(self) -> bool:
        """True when an Okta AI Agent (v2) identity is loaded."""
        return self._okta_identity is not None

    def get_okta_identity(self):
        """Return the loaded OktaAgentIdentity (or None)."""
        return self._okta_identity

    def __repr__(self) -> str:
        """Return string representation with masked API key. NEVER includes the key."""
        if self.api_key and len(self.api_key) > 8:
            masked_key = f"obx_****{self.api_key[-4:]}"
        elif self.api_key:
            masked_key = "****"
        else:
            masked_key = ""
        okta_agent_id = getattr(self._okta_identity, "external_agent_id", None)
        return (
            f"_GlobalConfig(api_url={self.api_url!r}, "
            f"api_key={masked_key!r}, "
            f"governance_timeout={self.governance_timeout}, "
            f"agent_did={self.agent_did!r}, "
            f"signing={'enabled' if self.has_signing() else 'disabled'}, "
            f"okta_agent_id={okta_agent_id!r})"
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
    workload_private_key: Optional[str] = None,
    openbox_agent_id: Optional[str] = None,
    organization_id: Optional[str] = None,
    deployment_id: Optional[str] = None,
    okta_agent_id: Optional[str] = None,
    okta_agent_key_id: Optional[str] = None,
    okta_agent_private_key: Optional[str] = None,
    okta_agent_algorithm: Optional[str] = None,
    agent_proof_audience: Optional[str] = None,
) -> None:
    """
    Initialize OpenBox SDK.

    Args:
        api_url: OpenBox Core API endpoint URL
        api_key: API key (format: obx_live_... or obx_test_...)
        governance_timeout: Timeout for governance requests in seconds (default: 30.0)
        validate: Validate API key with server on initialization (default: True)
        agent_did: Agent DID (format: did:aip:<uuid>, v1). Asserted in signed request
            headers. Pair with agent_private_key (both-or-neither).
        agent_private_key: Base64 raw 32-byte Ed25519 seed (v1). Signs every Core
            request locally. Non-repudiation material — never logged or stored as
            raw bytes.
        workload_private_key: PKCS8 PEM RSA private key for the active Keycloak
            service account (v3). The base SDK exchanges a short-lived token and
            composes it with the agent API key on every governed request.
        openbox_agent_id: OpenBox agent UUID (v2, Okta AI Agent).
        organization_id: OpenBox organization ID (v2).
        deployment_id: OpenBox deployment ID (v2).
        okta_agent_id: External Okta AI Agent ID (v2).
        okta_agent_key_id: Okta credential ``kid`` (v2).
        okta_agent_private_key: PKCS8 PEM RSA private key (v2). By itself, selects
            bootstrap mode: Core supplies the non-secret identity metadata. With
            all explicit Okta metadata fields, preserves the legacy explicit mode.
            Signs every Core request locally; never logged or stored raw.
        okta_agent_algorithm: Signing algorithm, only ``"RS256"`` is supported at
            launch (default ``"RS256"``).
        agent_proof_audience: Deployment-specific assertion audience (v2).

    Okta v2 supports either private-key-only bootstrap or the complete legacy
    explicit field set. A partial explicit set is rejected. Both modes are
    mutually exclusive with agent_did/agent_private_key (proposal §13.1).

    Raises:
        OpenBoxAuthError: Invalid API key
        OpenBoxConfigError: Invalid DID / private key / Okta fields, both
            identity methods configured together, or a partial pair/set
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

    okta_metadata_fields = {
        "openbox_agent_id": openbox_agent_id,
        "organization_id": organization_id,
        "deployment_id": deployment_id,
        "okta_agent_id": okta_agent_id,
        "okta_agent_key_id": okta_agent_key_id,
        "agent_proof_audience": agent_proof_audience,
    }
    okta_metadata_present = [
        name for name, value in okta_metadata_fields.items() if value
    ]
    okta_bootstrap = bool(okta_agent_private_key) and not okta_metadata_present
    okta_any_present = bool(okta_agent_private_key) or bool(okta_metadata_present)

    explicit_okta_fields = {
        **okta_metadata_fields,
        "okta_agent_private_key": okta_agent_private_key,
    }
    okta_explicit_missing = [
        name for name, value in explicit_okta_fields.items() if not value
    ]
    if okta_metadata_present and okta_explicit_missing:
        raise OpenBoxConfigError(
            "Partial Okta AI Agent identity configuration: "
            f"{', '.join(okta_metadata_present)} given but missing "
            f"{', '.join(okta_explicit_missing)}. Provide only "
            "okta_agent_private_key for bootstrap mode, or provide the complete "
            "explicit Okta identity field set."
        )

    if agent_did and okta_any_present:
        raise OpenBoxConfigError(
            "agent_did/agent_private_key (v1) and the Okta AI Agent fields (v2) "
            "are mutually exclusive. Configure exactly one identity verification "
            "method."
        )

    if okta_bootstrap and okta_agent_algorithm not in (None, "RS256"):
        raise OpenBoxConfigError(
            f"Unsupported okta_agent_algorithm {okta_agent_algorithm!r}; "
            "only 'RS256' is supported at launch."
        )

    signer = None
    if agent_did and agent_private_key:
        _validate_did(agent_did)
        signer = _load_ed25519_seed(agent_private_key)

    okta_identity = None
    if okta_metadata_present:
        # `okta_explicit_missing` above is empty here, so every field is non-None —
        # asserted (not just relied on) so mypy narrows str | None -> str,
        # matching _load_okta_identity's signature.
        assert openbox_agent_id is not None
        assert organization_id is not None
        assert deployment_id is not None
        assert okta_agent_id is not None
        assert okta_agent_key_id is not None
        assert okta_agent_private_key is not None
        assert agent_proof_audience is not None
        okta_identity = _load_okta_identity(
            openbox_agent_id=openbox_agent_id,
            organization_id=organization_id,
            deployment_id=deployment_id,
            okta_agent_id=okta_agent_id,
            okta_agent_key_id=okta_agent_key_id,
            okta_agent_private_key=okta_agent_private_key,
            agent_proof_audience=agent_proof_audience,
            okta_agent_algorithm=okta_agent_algorithm or "RS256",
        )
    elif okta_bootstrap:
        if not validate:
            raise OpenBoxConfigError(
                "Okta private-key-only bootstrap requires server validation; "
                "initialize(..., validate=False) cannot resolve the identity metadata."
            )
        assert okta_agent_private_key is not None
        okta_identity = _bootstrap_okta_identity(
            api_url=api_url.rstrip("/"),
            api_key=api_key,
            timeout=governance_timeout,
            okta_agent_private_key=okta_agent_private_key,
        )

    _config.configure(
        api_url=api_url,
        api_key=api_key,
        governance_timeout=governance_timeout,
        agent_did=agent_did,
        signer=signer,
        okta_identity=okta_identity,
    )

    # Validate API key with server (signed when signing is configured).
    if validate and workload_private_key:
        _validate_workload_identity_with_server(
            api_url.rstrip("/"),
            api_key,
            governance_timeout,
            workload_private_key=workload_private_key,
        )
    elif validate and not okta_bootstrap:
        _validate_api_key_with_server(
            api_url.rstrip("/"),
            api_key,
            governance_timeout,
            agent_did=agent_did,
            signer=signer,
            okta_identity=okta_identity,
        )

    _get_logger().info(
        f"OpenBox SDK initialized with API URL: {api_url} "
        f"(signing={'workload' if workload_private_key else ('enabled' if signer else ('okta' if okta_identity else 'disabled'))})"
    )
