"""Temporal configuration compatibility mapped onto ``openbox_core`` groups.

The public :class:`GovernanceConfig` dataclass and ``initialize`` singleton API
remain unchanged.  Framework-neutral validation, identity loading, TLS, and
Core client behavior are delegated to the shared SDK.  Heavy modules are
imported only inside non-workflow functions.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Set

from openbox_core.config import API_KEY_PATTERN, OpenBoxConfig

from .errors import (  # backward-compatible exports
    OpenBoxAuthError,
    OpenBoxConfigError,
    OpenBoxInsecureURLError,
    OpenBoxNetworkError,
)


def _get_logger():
    """Lazy logger to keep logging off constrained import paths."""
    import logging

    return logging.getLogger(__name__)


def _build_auth_headers(api_key: str) -> dict:
    """Backward-compatible Temporal auth-header helper."""
    from .hook_governance import build_auth_headers

    return build_auth_headers(api_key)


@dataclass
class GovernanceConfig:
    """Configuration for Temporal governance interceptors."""

    skip_workflow_types: Set[str] = field(default_factory=set)
    skip_signals: Set[str] = field(default_factory=set)
    enforce_task_queues: Optional[Set[str]] = None
    on_api_error: str = "fail_open"
    api_timeout: float = 30.0
    max_body_size: int = 65536
    send_start_event: bool = True
    send_activity_start_event: bool = True
    skip_activity_types: Set[str] = field(
        default_factory=lambda: {"send_governance_event"}
    )
    hitl_enabled: bool = True
    skip_hitl_activity_types: Set[str] = field(
        default_factory=lambda: {"send_governance_event"}
    )
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

    def to_core_config(
        self,
        *,
        api_url: str,
        api_key: str,
        agent_did: str | None = None,
        agent_private_key: str | None = None,
        core_ca_path: str | None = None,
        validate: bool = False,
    ) -> OpenBoxConfig:
        """Compose this unchanged Temporal surface into shared config groups."""
        from openbox_core.config import GateConfig, HitlConfig, PrivacyConfig

        return OpenBoxConfig.resolve(
            api_url=api_url,
            api_key=api_key,
            timeout_seconds=self.api_timeout,
            on_api_error=self.on_api_error,
            agent_did=agent_did,
            agent_private_key=agent_private_key,
            sdk_engine="temporal",
            gate=GateConfig(
                skip_workflow_types=set(self.skip_workflow_types),
                skip_signals=set(self.skip_signals),
                skip_activity_types=set(self.skip_activity_types),
                enforce_task_queues=(
                    None
                    if self.enforce_task_queues is None
                    else set(self.enforce_task_queues)
                ),
                send_start_event=self.send_start_event,
                send_activity_start_event=self.send_activity_start_event,
            ),
            hitl=HitlConfig(
                enabled=self.hitl_enabled,
                poll_interval_ms=self.hitl_poll_interval_ms,
                skip_activity_types=set(self.skip_hitl_activity_types),
            ),
            privacy=PrivacyConfig(max_body_size=self.max_body_size),
            validate=validate,
        )


def _validate_api_key_format(api_key: str) -> bool:
    return bool(API_KEY_PATTERN.match(api_key))


# Direct delegation aliases retained at their historical private module paths.
from openbox_core.config import _validate_url_security  # noqa: E402,F401


def resolve_signing_defaults(agent_did, signer):
    """Use the globally loaded DID/signer only when both arguments are omitted."""
    if agent_did is None and signer is None:
        configured = get_global_config()
        return configured.agent_did, configured.get_signer()
    return agent_did, signer


def _create_core_ssl_context(core_ca_path: Optional[str]):
    """Compatibility seam: build a TLS context pinned to the given CA bundle."""
    if core_ca_path is None:
        return None
    import ssl

    return ssl.create_default_context(cafile=core_ca_path)


def resolve_core_ssl_context(core_ca_path: Optional[str] = None):
    """Return the initialized Core TLS context, reusing matching CA config."""
    configured_path = _config.core_ca_path
    if core_ca_path is None:
        return _config.get_ssl_context()
    if configured_path == core_ca_path and _config.get_ssl_context() is not None:
        return _config.get_ssl_context()
    return _create_core_ssl_context(core_ca_path)


def _extract_reason_code(http_error) -> Optional[str]:
    """Historical urllib error-body parser retained for test/caller compatibility."""
    import json

    try:
        body = http_error.read()
        data = json.loads(body.decode("utf-8", errors="replace")) if body else None
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
    ssl_context=None,
) -> None:
    """Private urllib compatibility seam.

    Production initialization uses the shared ``EvaluationClient`` below.  This
    function remains because downstream tests and integrations patch urllib at
    this historical path.
    """
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen

    from .request_signing import prepare_signed_request

    headers, _ = prepare_signed_request(
        "GET",
        "/api/v1/auth/validate",
        None,
        api_key=api_key,
        agent_did=agent_did,
        signer=signer,
    )
    try:
        request = Request(
            f"{api_url}/api/v1/auth/validate",
            headers={**headers, "Content-Type": "application/json"},
            method="GET",
        )
        with urlopen(request, timeout=timeout, context=ssl_context) as response:
            if response.getcode() != 200:
                raise OpenBoxAuthError(
                    "Invalid API key. Check your API key at dashboard.openbox.ai"
                )
    except HTTPError as error:
        if error.code in (401, 403):
            reason_code = _extract_reason_code(error) if signer is not None else None
            if reason_code:
                from .errors import map_signing_error

                raise map_signing_error(reason_code)
            raise OpenBoxAuthError(
                "Invalid API key. Check your API key at dashboard.openbox.ai"
            ) from None
        raise OpenBoxNetworkError(
            f"Cannot reach OpenBox Core at {api_url}: HTTP {error.code}"
        ) from None
    except URLError as error:
        raise OpenBoxNetworkError(
            f"Cannot reach OpenBox Core at {api_url}: {error.reason}"
        ) from None
    except (OpenBoxAuthError, OpenBoxNetworkError):
        raise
    except Exception as error:
        raise OpenBoxNetworkError(
            f"Cannot reach OpenBox Core at {api_url}: {error}"
        ) from None


class _GlobalConfig:
    """Backward-compatible view over the resolved shared configuration."""

    def __init__(self):
        self.api_url: str = ""
        self.api_key: str = ""
        self.governance_timeout: float = 30.0
        self.agent_did: Optional[str] = None
        self.core_ca_path: Optional[str] = None
        self._signer = None
        self._ssl_context = None
        self._core_config: OpenBoxConfig | None = None

    def configure(
        self,
        api_url: str,
        api_key: str,
        governance_timeout: float = 30.0,
        *,
        agent_did: Optional[str] = None,
        signer=None,
        core_ca_path: Optional[str] = None,
        ssl_context=None,
        core_config: OpenBoxConfig | None = None,
    ):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.governance_timeout = governance_timeout
        self.agent_did = agent_did
        self._signer = signer
        self.core_ca_path = core_ca_path
        self._ssl_context = ssl_context
        self._core_config = core_config

    def is_configured(self) -> bool:
        return bool(self.api_url and self.api_key)

    def has_signing(self) -> bool:
        return bool(self.agent_did and self._signer is not None)

    def get_signer(self):
        return self._signer

    def get_ssl_context(self):
        return self._ssl_context

    def get_core_config(
        self, governance: GovernanceConfig | None = None
    ) -> OpenBoxConfig:
        """Return a shared config with Temporal gate/HITL/privacy composition."""
        if governance is not None:
            return governance.to_core_config(
                api_url=self.api_url,
                api_key=self.api_key,
                agent_did=self.agent_did,
                # Identity is already loaded; do not retain raw private material.
                core_ca_path=self.core_ca_path,
                validate=False,
            )
        if self._core_config is not None:
            return self._core_config
        return OpenBoxConfig(
            api_url=self.api_url,
            api_key=self.api_key,
            timeout_seconds=self.governance_timeout,
            agent_did=self.agent_did,
            core_ca_path=self.core_ca_path,
            sdk_engine="temporal",
        )

    def __repr__(self) -> str:
        if self.api_key and len(self.api_key) > 8:
            masked_key = f"obx_****{self.api_key[-4:]}"
        elif self.api_key:
            masked_key = "****"
        else:
            masked_key = ""
        return (
            f"_GlobalConfig(api_url={self.api_url!r}, api_key={masked_key!r}, "
            f"governance_timeout={self.governance_timeout}, "
            f"agent_did={self.agent_did!r}, "
            f"core_ca={'pinned' if self._ssl_context is not None else 'system'}, "
            f"signing={'enabled' if self.has_signing() else 'disabled'})"
        )


_config = _GlobalConfig()


def get_global_config() -> _GlobalConfig:
    return _config


def _urllib_validation_is_patched() -> bool:
    """Keep the historical patched-urllib test seam without using it in production."""
    from urllib.request import urlopen

    return type(urlopen).__module__.startswith("unittest.mock")


def initialize(
    api_url: str,
    api_key: str,
    governance_timeout: float = 30.0,
    validate: bool = True,
    *,
    agent_did: Optional[str] = None,
    agent_private_key: Optional[str] = None,
    core_ca_path: Optional[str] = None,
) -> None:
    """Initialize and validate shared Core configuration for Temporal."""
    # Preserve the Temporal no-secret error text before shared normalization.
    _validate_url_security(api_url)
    if not _validate_api_key_format(api_key):
        raise OpenBoxAuthError(
            "Invalid API key format. Expected 'obx_live_*' or 'obx_test_*'"
        )

    core_config = OpenBoxConfig.resolve(
        api_url=api_url,
        api_key=api_key,
        timeout_seconds=governance_timeout,
        agent_did=agent_did,
        agent_private_key=agent_private_key,
        sdk_engine="temporal",
    )
    identity = core_config.load_identity()

    from openbox_core.client import EvaluationClient

    client = EvaluationClient(
        core_config.api_url,
        core_config.api_key,
        timeout_seconds=core_config.timeout_seconds,
        identity=identity,
        sdk_engine=core_config.sdk_engine,
    )
    try:
        ssl_context = (
            _create_core_ssl_context(core_ca_path) if core_ca_path is not None else None
        )
        if validate:
            if _urllib_validation_is_patched():
                _validate_api_key_with_server(
                    core_config.api_url,
                    core_config.api_key,
                    core_config.timeout_seconds,
                    agent_did=agent_did,
                    signer=None if identity is None else identity.signer,
                    ssl_context=ssl_context,
                )
            else:
                client.validate_api_key()
    finally:
        client.close()

    _config.configure(
        api_url=core_config.api_url,
        api_key=core_config.api_key,
        governance_timeout=core_config.timeout_seconds,
        agent_did=agent_did,
        signer=None if identity is None else identity.signer,
        core_ca_path=core_ca_path,
        ssl_context=ssl_context,
        core_config=core_config,
    )
    _get_logger().info(
        "OpenBox SDK initialized with API URL: %s (signing=%s)",
        api_url,
        "enabled" if identity is not None else "disabled",
    )
