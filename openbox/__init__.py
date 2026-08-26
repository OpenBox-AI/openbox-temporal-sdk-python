"""OpenBox SDK - Workflow-Boundary Governance with OpenTelemetry"""

from typing import Any

# STATIC on purpose — never read via importlib.metadata. A metadata lookup
# OPENS A FILE; with file instrumentation active (traced_open patches
# builtins.open + io.open) that read re-enters governance: eagerly at import
# it crashes the workflow sandbox as a circular import, lazily it recurses
# unboundedly from build_auth_headers on every evaluate. Keep in sync with
# pyproject.toml on release.
__version__ = "1.4.0"

from .config import GovernanceConfig, get_global_config, initialize
from .errors import (
    ApprovalExpiredError,
    ApprovalRejectedError,
    ApprovalTimeoutError,
    GovernanceAPIError,
    GovernanceBlockedError,
    GovernanceHaltError,
    GuardrailsValidationError,
    OpenBoxAuthError,
    OpenBoxConfigError,
    OpenBoxError,
    OpenBoxInsecureURLError,
    OpenBoxNetworkError,
    OpenBoxSigningError,
    extract_governance_error,
    map_signing_error,
)

# Multi-agent primitives (sandbox-safe — only imports temporalio.workflow eagerly;
# signing/HTTP routed lazily through the governance activity).
from .multi_agent import emit_handoff

# BLOCK-with-patch restart envelope (sandbox-safe: pure stdlib + base contracts).
from .patch import GOVERNANCE_PATCH_SCHEMA_VERSION, PatchRequest
from .types import (
    GovernanceVerdictResponse,
    GuardrailsCheckResult,
    Verdict,
    WorkflowEventType,
)
from .workflow_interceptor import GovernanceInterceptor

try:
    from temporalio.plugin import SimplePlugin  # noqa: F401 — probe only

    from .plugin import OpenBoxPlugin
except ImportError:
    pass  # temporalio < 1.23.0, plugin not available

from .client import GovernanceClient
from .hitl import handle_approval_response, raise_approval_pending, should_skip_hitl
from .verdict_handler import VerdictEnforcementResult, enforce_verdict

# NOTE: ActivityGovernanceInterceptor is NOT imported here because it imports
# OpenTelemetry which uses importlib_metadata -> os.stat, causing sandbox issues.
# Users must import directly: from openbox.activity_interceptor import ActivityGovernanceInterceptor

# Activities - DO NOT import here!
#
# IMPORTANT: Do NOT import activities from openbox/__init__.py!
# activities.py imports httpx which uses os.stat internally. If we re-export them
# here, it triggers Temporal sandbox restrictions ("Cannot access os.stat.__call__").
#
# Users must import directly:
#   from openbox.activities import send_governance_event

# Instrumentation — owned by the openbox_core base runtime
#
# HTTP/DB/file/function hook instrumentation is installed by the base runtime
# There is no Temporal-local OpenTelemetry setup to import.
#
# Tracing Decorators - NOT imported here to avoid sandbox issues!
#
# @traced wraps the base SDK's governed() decorator. Import directly:
#     from openbox.tracing import traced, create_span
#
#     @traced
#     def my_function(data):
#         return process(data)


__all__ = [
    "OpenBoxPlugin",
    "initialize",
    "get_global_config",
    "GovernanceConfig",
    "OpenBoxError",
    "OpenBoxConfigError",
    "OpenBoxAuthError",
    "OpenBoxNetworkError",
    "OpenBoxInsecureURLError",
    "OpenBoxSigningError",
    "GovernanceBlockedError",
    "GovernanceHaltError",
    "GovernanceAPIError",
    "GuardrailsValidationError",
    "ApprovalExpiredError",
    "ApprovalRejectedError",
    "ApprovalTimeoutError",
    "extract_governance_error",
    "map_signing_error",
    "Verdict",
    "WorkflowEventType",
    "GovernanceVerdictResponse",
    "GuardrailsCheckResult",
    "PatchRequest",
    "GOVERNANCE_PATCH_SCHEMA_VERSION",
    "emit_handoff",
    "GovernanceInterceptor",
    "enforce_verdict",
    "VerdictEnforcementResult",
    "handle_approval_response",
    "raise_approval_pending",
    "should_skip_hitl",
    "GovernanceClient",
]


# Sandbox surface (local extension): resolved lazily so importing any
# sandbox-safe module never pulls the signing/network stack.
_SANDBOX_NAMES = (
    "AipEd25519RequestSigner",
    "GovernedCommandActivityResult",
    "GovernedCommandInputError",
    "GovernedCommandRequest",
    "GovernedCommandResultValue",
    "GovernedCommandTypedResult",
    "StructuredCommandArgument",
    "TemporalCommandProfileBundle",
    "TemporalHeartbeatSink",
    "TemporalSandboxConfig",
)


def __getattr__(name: str) -> Any:
    if name in _SANDBOX_NAMES:
        import openbox.sandbox as _sandbox_pkg

        value = getattr(_sandbox_pkg, name)
        globals()[name] = value
        return value
    raise AttributeError(name)
