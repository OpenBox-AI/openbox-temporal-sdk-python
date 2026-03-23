"""OpenBox SDK - Workflow-Boundary Governance with OpenTelemetry"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("openbox-temporal-sdk-python")
except PackageNotFoundError:
    __version__ = "0.0.0"  # Fallback for editable installs without metadata

# ═══════════════════════════════════════════════════════════════════════════════
# Simple Factories (recommended)
# ═══════════════════════════════════════════════════════════════════════════════

from .worker import create_openbox_worker

# ═══════════════════════════════════════════════════════════════════════════════
# Core Configuration
# ═══════════════════════════════════════════════════════════════════════════════

from .config import (
    initialize,
    get_global_config,
    GovernanceConfig,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Errors (unified hierarchy in errors.py)
# ═══════════════════════════════════════════════════════════════════════════════

from .errors import (
    OpenBoxError,
    OpenBoxConfigError,
    OpenBoxAuthError,
    OpenBoxNetworkError,
    OpenBoxInsecureURLError,
    GovernanceBlockedError,
    GovernanceHaltError,
    GovernanceAPIError,
    GuardrailsValidationError,
    ApprovalExpiredError,
    ApprovalRejectedError,
    ApprovalTimeoutError,
    extract_governance_error,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Types (sandbox-safe - can be imported in workflow code)
# ═══════════════════════════════════════════════════════════════════════════════

from .types import (
    Verdict,
    WorkflowEventType,
    WorkflowSpanBuffer,
    GovernanceVerdictResponse,
    GuardrailsCheckResult,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Span Processor
# ═══════════════════════════════════════════════════════════════════════════════

from .span_processor import WorkflowSpanProcessor

# ═══════════════════════════════════════════════════════════════════════════════
# Interceptors
# ═══════════════════════════════════════════════════════════════════════════════

from .workflow_interceptor import GovernanceInterceptor

# ═══════════════════════════════════════════════════════════════════════════════
# Verdict Handler
# ═══════════════════════════════════════════════════════════════════════════════

from .verdict_handler import enforce_verdict, VerdictEnforcementResult

# ═══════════════════════════════════════════════════════════════════════════════
# HITL — Human-in-the-Loop approval helpers
# ═══════════════════════════════════════════════════════════════════════════════

from .hitl import handle_approval_response, raise_approval_pending, should_skip_hitl

# ═══════════════════════════════════════════════════════════════════════════════
# Governance HTTP Client (sandbox-safe — httpx imported lazily inside methods)
# ═══════════════════════════════════════════════════════════════════════════════

from .client import GovernanceClient

# NOTE: ActivityGovernanceInterceptor is NOT imported here because it imports
# OpenTelemetry which uses importlib_metadata -> os.stat, causing sandbox issues.
# Users must import directly: from openbox.activity_interceptor import ActivityGovernanceInterceptor

# ═══════════════════════════════════════════════════════════════════════════════
# Activities - DO NOT import here!
# ═══════════════════════════════════════════════════════════════════════════════
#
# IMPORTANT: Do NOT import activities from openbox/__init__.py!
# activities.py imports httpx which uses os.stat internally. If we re-export them
# here, it triggers Temporal sandbox restrictions ("Cannot access os.stat.__call__").
#
# Users must import directly:
#   from openbox.activities import send_governance_event

# ═══════════════════════════════════════════════════════════════════════════════
# OTel Setup - NOT imported here to avoid sandbox issues!
# ═══════════════════════════════════════════════════════════════════════════════
#
# IMPORTANT: Do NOT import otel_setup here!
# otel_setup imports OpenTelemetry which uses importlib_metadata -> os.stat
# This triggers Temporal sandbox restrictions.
#
# Users must import directly: from openbox.otel_setup import setup_opentelemetry_for_governance

# ═══════════════════════════════════════════════════════════════════════════════
# Tracing Decorators - NOT imported here to avoid sandbox issues!
# ═══════════════════════════════════════════════════════════════════════════════
#
# Use the @traced decorator to capture internal function calls as spans.
# Import directly: from openbox.tracing import traced, create_span
#
# Example:
#     from openbox.tracing import traced
#
#     @traced
#     def my_function(data):
#         return process(data)


__all__ = [
    # Simple Worker Factory (recommended)
    "create_openbox_worker",
    # Configuration
    "initialize",
    "get_global_config",
    "GovernanceConfig",
    # Errors (unified hierarchy)
    "OpenBoxError",
    "OpenBoxConfigError",
    "OpenBoxAuthError",
    "OpenBoxNetworkError",
    "OpenBoxInsecureURLError",
    "GovernanceBlockedError",
    "GovernanceHaltError",
    "GovernanceAPIError",
    "GuardrailsValidationError",
    "ApprovalExpiredError",
    "ApprovalRejectedError",
    "ApprovalTimeoutError",
    "extract_governance_error",
    # Types
    "Verdict",
    "WorkflowEventType",
    "WorkflowSpanBuffer",
    "GovernanceVerdictResponse",
    "GuardrailsCheckResult",
    # Span Processor
    "WorkflowSpanProcessor",
    # Interceptors
    "GovernanceInterceptor",
    # Verdict handler
    "enforce_verdict",
    "VerdictEnforcementResult",
    # HITL helpers
    "handle_approval_response",
    "raise_approval_pending",
    "should_skip_hitl",
    # Governance HTTP client
    "GovernanceClient",
]
