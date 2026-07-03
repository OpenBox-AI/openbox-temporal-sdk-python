"""Public API compatibility gate for the base-SDK migration branch.

Pins the post-migration public surface: every export, the
``create_openbox_worker``/``OpenBoxPlugin`` kwargs, and the shared shim types'
(Verdict / EvaluationResult / WorkflowEventType) behavioral identity. Any
accidental rename/removal fails here first.

Note: ``WorkflowSpanProcessor`` and ``WorkflowSpanBuffer`` were INTENTIONALLY
removed — hook instrumentation now lives entirely in the base SDK
(``openbox_core``) and the worker builds+owns an ``OpenBoxRuntime`` instead of a
Temporal-local span processor. Their absence is asserted below so they are not
silently re-added.
"""

from __future__ import annotations

import inspect

import openbox

# Post-migration export list. WorkflowSpanProcessor / WorkflowSpanBuffer are
# deliberately absent (see module docstring); every other pre-migration name
# is preserved for backward compatibility.
EXPECTED_EXPORTS = {
    "ApprovalExpiredError",
    "ApprovalRejectedError",
    "ApprovalTimeoutError",
    "GovernanceAPIError",
    "GovernanceBlockedError",
    "GovernanceClient",
    "GovernanceConfig",
    "GovernanceHaltError",
    "GovernanceInterceptor",
    "GovernanceVerdictResponse",
    "GuardrailsCheckResult",
    "GuardrailsValidationError",
    "OpenBoxAuthError",
    "OpenBoxConfigError",
    "OpenBoxError",
    "OpenBoxInsecureURLError",
    "OpenBoxNetworkError",
    "OpenBoxPlugin",
    "OpenBoxSigningError",
    "Verdict",
    "VerdictEnforcementResult",
    "WorkflowEventType",
    "create_openbox_worker",
    "enforce_verdict",
    "extract_governance_error",
    "get_global_config",
    "handle_approval_response",
    "initialize",
    "map_signing_error",
    "raise_approval_pending",
    "should_skip_hitl",
    # emit_handoff — multi-agent primitive kept in the public surface.
    "emit_handoff",
}

# Symbols removed by the migration; instrumentation is base-SDK-owned now.
REMOVED_EXPORTS = {
    "WorkflowSpanProcessor",
    "WorkflowSpanBuffer",
}

WORKER_KWARGS = {
    "client",
    "task_queue",
    "workflows",
    "activities",
    "openbox_url",
    "openbox_api_key",
    "agent_did",
    "agent_private_key",
    "governance_timeout",
    "governance_policy",
    "send_start_event",
    "send_activity_start_event",
    "skip_workflow_types",
    "skip_activity_types",
}

PLUGIN_KWARGS = {
    "openbox_url",
    "openbox_api_key",
    "agent_did",
    "agent_private_key",
    "governance_timeout",
    "governance_policy",
    "hitl_enabled",
    "send_start_event",
    "send_activity_start_event",
    "skip_workflow_types",
    "skip_activity_types",
    "skip_signals",
    "instrument_databases",
    "db_libraries",
    "instrument_file_io",
    "sqlalchemy_engine",
    "enable_trace_propagation",
}


class TestExports:
    def test_all_expected_exports_present(self):
        missing = EXPECTED_EXPORTS - set(openbox.__all__)
        assert not missing, f"public exports removed: {sorted(missing)}"

    def test_exports_resolve(self):
        for name in EXPECTED_EXPORTS:
            assert getattr(openbox, name, None) is not None, name

    def test_public_surface_is_exactly_expected(self):
        # __all__ is the whole public surface — nothing beyond the expected set.
        assert set(openbox.__all__) == EXPECTED_EXPORTS
        assert len(openbox.__all__) == 32

    def test_removed_exports_absent(self):
        # Instrumentation shims are base-SDK-owned now; they must NOT reappear.
        for name in REMOVED_EXPORTS:
            assert name not in openbox.__all__, f"{name} unexpectedly re-exported"
            assert not hasattr(openbox, name), f"{name} unexpectedly present on module"


class TestEntrypointSignatures:
    def test_create_openbox_worker_kwargs_preserved(self):
        parameters = set(inspect.signature(openbox.create_openbox_worker).parameters)
        missing = WORKER_KWARGS - parameters
        assert not missing, f"create_openbox_worker lost kwargs: {sorted(missing)}"

    def test_plugin_kwargs_preserved(self):
        parameters = set(inspect.signature(openbox.OpenBoxPlugin.__init__).parameters)
        missing = PLUGIN_KWARGS - parameters
        assert not missing, f"OpenBoxPlugin lost kwargs: {sorted(missing)}"


class TestShimBehavioralIdentity:
    def test_verdict_is_shared_enum_with_full_surface(self):
        from openbox_core.contracts.results import Verdict as CoreVerdict

        assert openbox.Verdict is CoreVerdict
        assert openbox.Verdict.from_string("continue") is openbox.Verdict.ALLOW
        assert openbox.Verdict.HALT.priority == 5
        assert openbox.Verdict.BLOCK.should_stop()

    def test_governance_verdict_response_is_shared_result(self):
        from openbox_core.contracts.results import EvaluationResult

        response = openbox.GovernanceVerdictResponse.from_dict(
            {"verdict": "block", "reason": "x", "guardrails_result": {"input_type": "activity_input"}}
        )
        assert isinstance(response, EvaluationResult)
        assert response.guardrails_result is response.guardrails  # same object
        assert response.action == "block"
        # New shared fields present with safe defaults:
        assert response.fallback_used is False
        assert response.raw["reason"] == "x"

    def test_workflow_event_type_values_unchanged(self):
        assert openbox.WorkflowEventType.ACTIVITY_STARTED.value == "ActivityStarted"
        assert openbox.WorkflowEventType.HANDOFF.value == "Handoff"
        assert len(openbox.WorkflowEventType) == 7
