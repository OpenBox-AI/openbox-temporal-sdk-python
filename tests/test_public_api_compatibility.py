"""Public API compatibility gate.

Pins every export, the ``create_openbox_worker``/``OpenBoxPlugin`` kwargs, and
the shared shim types' behavioral identity. Any accidental rename/removal fails
here first.

``WorkflowSpanProcessor`` and ``WorkflowSpanBuffer`` are intentionally absent.
Hook instrumentation lives in ``openbox_core`` and the worker owns an
``OpenBoxRuntime``.
"""

from __future__ import annotations

import inspect

import openbox

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
    # Retryable-BLOCK restart envelope + schema version.
    "RetryableBlockRequest",
    "GOVERNANCE_RETRYABLE_BLOCK_SCHEMA_VERSION",
}

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
        assert len(openbox.__all__) == 34

    def test_removed_exports_absent(self):
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
            {
                "verdict": "block",
                "reason": "x",
                "guardrails_result": {"input_type": "activity_input"},
            }
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
