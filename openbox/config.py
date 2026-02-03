# openbox/config.py
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
API_KEY_PATTERN = re.compile(r"^obx_(live|test)_[a-zA-Z0-9_]+$")


# ═══════════════════════════════════════════════════════════════════════════════
# Exceptions
# ═══════════════════════════════════════════════════════════════════════════════


class OpenBoxConfigError(Exception):
    """Raised when OpenBox configuration fails."""

    pass


class OpenBoxAuthError(OpenBoxConfigError):
    """Raised when API key validation fails."""

    pass


class OpenBoxNetworkError(OpenBoxConfigError):
    """Raised when network connectivity fails."""

    pass


class OpenBoxInsecureURLError(OpenBoxConfigError):
    """Raised when HTTP is used for non-localhost URLs."""

    pass


# ═══════════════════════════════════════════════════════════════════════════════
# GovernanceConfig - Configuration for interceptors
# ═══════════════════════════════════════════════════════════════════════════════


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

    # Maximum body size to capture (None = no limit)
    max_body_size: Optional[int] = None

    # Send WorkflowStarted event (can disable for performance)
    send_start_event: bool = True

    # Send ActivityStarted event before each activity (can disable for performance)
    send_activity_start_event: bool = True

    # Activity types to skip governance for
    # By default, skip the governance event activity to avoid infinite loops
    skip_activity_types: Set[str] = field(default_factory=lambda: {"send_governance_event"})

    # Approval polling configuration
    # Enable approval polling for require-approval verdicts
    hitl_enabled: bool = True

    # Activity types to skip approval checks (to avoid infinite loops)
    # By default, skip the governance event activity
    skip_hitl_activity_types: Set[str] = field(
        default_factory=lambda: {"send_governance_event"}
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Global Configuration Singleton
# ═══════════════════════════════════════════════════════════════════════════════


def _validate_api_key_format(api_key: str) -> bool:
    """Validate API key format (obx_live_... or obx_test_...)."""
    return bool(API_KEY_PATTERN.match(api_key))


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


def _validate_api_key_with_server(api_url: str, api_key: str, timeout: float) -> None:
    """
    Validate API key by calling /v1/auth/validate endpoint.
    Raises OpenBoxAuthError for invalid key, OpenBoxNetworkError for connectivity issues.

    NOTE: urllib imports are lazy to avoid Temporal sandbox restrictions.
    urllib.request uses os.stat internally which triggers sandbox errors.
    """
    # Lazy imports to avoid sandbox restrictions
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError, URLError

    try:
        req = Request(
            f"{api_url}/api/v1/auth/validate",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
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
            raise OpenBoxAuthError(
                "Invalid API key. Check your API key at dashboard.openbox.ai"
            )
        raise OpenBoxNetworkError(f"Cannot reach OpenBox Core at {api_url}: HTTP {e.code}")

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

    def configure(
        self,
        api_url: str,
        api_key: str,
        governance_timeout: float = 30.0,
    ):
        """Configure OpenBox settings."""
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.governance_timeout = governance_timeout

    def is_configured(self) -> bool:
        """Check if OpenBox is configured."""
        return bool(self.api_url and self.api_key)

    def __repr__(self) -> str:
        """Return string representation with masked API key."""
        if self.api_key and len(self.api_key) > 8:
            masked_key = f"obx_****{self.api_key[-4:]}"
        elif self.api_key:
            masked_key = "****"
        else:
            masked_key = ""
        return (
            f"_GlobalConfig(api_url={self.api_url!r}, "
            f"api_key={masked_key!r}, "
            f"governance_timeout={self.governance_timeout})"
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
) -> None:
    """
    Initialize OpenBox SDK.

    Args:
        api_url: OpenBox Core API endpoint URL
        api_key: API key (format: obx_live_... or obx_test_...)
        governance_timeout: Timeout for governance requests in seconds (default: 30.0)
        validate: Validate API key with server on initialization (default: True)

    Raises:
        OpenBoxAuthError: Invalid API key
        OpenBoxNetworkError: Cannot reach OpenBox Core

    Example:
        from openbox import (
            initialize,
            setup_opentelemetry_for_governance,
            WorkflowSpanProcessor,
            GovernanceInterceptor,
            ActivityGovernanceInterceptor,
            OpenBoxClient,
            GovernanceConfig,
        )

        # 1. Initialize SDK (use HTTPS for production, HTTP allowed for localhost only)
        initialize(api_url="https://api.openbox.ai", api_key="obx_live_...")

        # 2. Setup OTel and span processor
        span_processor = WorkflowSpanProcessor()
        setup_opentelemetry_for_governance(span_processor)

        # 3. Create client and config
        client = OpenBoxClient(api_url="...", api_key="...")
        config = GovernanceConfig()

        # 4. Create interceptors (BOTH workflow and activity)
        workflow_interceptor = GovernanceInterceptor(client, span_processor, config)
        activity_interceptor = ActivityGovernanceInterceptor(api_url, api_key, span_processor, config)

        # 5. Add to Temporal worker
        worker = Worker(..., interceptors=[workflow_interceptor, activity_interceptor])
    """
    # Validate URL security (HTTPS required for non-localhost)
    _validate_url_security(api_url)

    # Validate API key format
    if not _validate_api_key_format(api_key):
        raise OpenBoxAuthError(
            f"Invalid API key format. Expected 'obx_live_*' or 'obx_test_*', "
            f"got: '{api_key[:15]}...' (showing first 15 chars)"
        )

    _config.configure(
        api_url=api_url,
        api_key=api_key,
        governance_timeout=governance_timeout,
    )

    # Validate API key with server
    if validate:
        _validate_api_key_with_server(api_url.rstrip("/"), api_key, governance_timeout)

    _get_logger().info(f"OpenBox SDK initialized with API URL: {api_url}")