# tests/test_types.py
"""Comprehensive tests for the OpenBox SDK types module."""

import pytest
from openbox.types import (
    WorkflowEventType,
    Verdict,
    WorkflowSpanBuffer,
    GuardrailsCheckResult,
    GovernanceVerdictResponse,
)


class TestWorkflowEventType:
    """Tests for WorkflowEventType enum."""

    def test_workflow_started_value(self):
        """Test WORKFLOW_STARTED enum value."""
        assert WorkflowEventType.WORKFLOW_STARTED == "WorkflowStarted"
        assert WorkflowEventType.WORKFLOW_STARTED.value == "WorkflowStarted"

    def test_workflow_completed_value(self):
        """Test WORKFLOW_COMPLETED enum value."""
        assert WorkflowEventType.WORKFLOW_COMPLETED == "WorkflowCompleted"
        assert WorkflowEventType.WORKFLOW_COMPLETED.value == "WorkflowCompleted"

    def test_workflow_failed_value(self):
        """Test WORKFLOW_FAILED enum value."""
        assert WorkflowEventType.WORKFLOW_FAILED == "WorkflowFailed"
        assert WorkflowEventType.WORKFLOW_FAILED.value == "WorkflowFailed"

    def test_signal_received_value(self):
        """Test SIGNAL_RECEIVED enum value."""
        assert WorkflowEventType.SIGNAL_RECEIVED == "SignalReceived"
        assert WorkflowEventType.SIGNAL_RECEIVED.value == "SignalReceived"

    def test_activity_started_value(self):
        """Test ACTIVITY_STARTED enum value."""
        assert WorkflowEventType.ACTIVITY_STARTED == "ActivityStarted"
        assert WorkflowEventType.ACTIVITY_STARTED.value == "ActivityStarted"

    def test_activity_completed_value(self):
        """Test ACTIVITY_COMPLETED enum value."""
        assert WorkflowEventType.ACTIVITY_COMPLETED == "ActivityCompleted"
        assert WorkflowEventType.ACTIVITY_COMPLETED.value == "ActivityCompleted"

    def test_all_enum_values_exist(self):
        """Test all expected enum values exist."""
        expected_values = {
            "WORKFLOW_STARTED": "WorkflowStarted",
            "WORKFLOW_COMPLETED": "WorkflowCompleted",
            "WORKFLOW_FAILED": "WorkflowFailed",
            "SIGNAL_RECEIVED": "SignalReceived",
            "ACTIVITY_STARTED": "ActivityStarted",
            "ACTIVITY_COMPLETED": "ActivityCompleted",
        }
        for name, value in expected_values.items():
            assert hasattr(WorkflowEventType, name)
            assert getattr(WorkflowEventType, name).value == value

    def test_enum_is_string_subclass(self):
        """Test that WorkflowEventType is a string enum."""
        assert isinstance(WorkflowEventType.WORKFLOW_STARTED, str)
        # Can use directly as string
        event_str: str = WorkflowEventType.WORKFLOW_STARTED
        assert event_str == "WorkflowStarted"


class TestVerdict:
    """Tests for Verdict enum."""

    # Test enum values
    def test_allow_value(self):
        """Test ALLOW enum value."""
        assert Verdict.ALLOW == "allow"
        assert Verdict.ALLOW.value == "allow"

    def test_constrain_value(self):
        """Test CONSTRAIN enum value."""
        assert Verdict.CONSTRAIN == "constrain"
        assert Verdict.CONSTRAIN.value == "constrain"

    def test_require_approval_value(self):
        """Test REQUIRE_APPROVAL enum value."""
        assert Verdict.REQUIRE_APPROVAL == "require_approval"
        assert Verdict.REQUIRE_APPROVAL.value == "require_approval"

    def test_block_value(self):
        """Test BLOCK enum value."""
        assert Verdict.BLOCK == "block"
        assert Verdict.BLOCK.value == "block"

    def test_halt_value(self):
        """Test HALT enum value."""
        assert Verdict.HALT == "halt"
        assert Verdict.HALT.value == "halt"

    # Test from_string() - v1.0 compat mappings
    def test_from_string_continue_maps_to_allow(self):
        """Test v1.0 compat: 'continue' maps to ALLOW."""
        assert Verdict.from_string("continue") == Verdict.ALLOW

    def test_from_string_continue_case_insensitive(self):
        """Test v1.0 compat: 'CONTINUE' (uppercase) maps to ALLOW."""
        assert Verdict.from_string("CONTINUE") == Verdict.ALLOW
        assert Verdict.from_string("Continue") == Verdict.ALLOW

    def test_from_string_stop_maps_to_halt(self):
        """Test v1.0 compat: 'stop' maps to HALT."""
        assert Verdict.from_string("stop") == Verdict.HALT

    def test_from_string_stop_case_insensitive(self):
        """Test v1.0 compat: 'STOP' (uppercase) maps to HALT."""
        assert Verdict.from_string("STOP") == Verdict.HALT
        assert Verdict.from_string("Stop") == Verdict.HALT

    def test_from_string_require_approval_with_hyphen(self):
        """Test v1.0 compat: 'require-approval' (hyphenated) maps to REQUIRE_APPROVAL."""
        assert Verdict.from_string("require-approval") == Verdict.REQUIRE_APPROVAL

    def test_from_string_request_approval_variant(self):
        """Test v1.0 compat: 'request-approval' variant maps to REQUIRE_APPROVAL."""
        assert Verdict.from_string("request-approval") == Verdict.REQUIRE_APPROVAL
        assert Verdict.from_string("request_approval") == Verdict.REQUIRE_APPROVAL

    # Test from_string() - v1.1 values
    def test_from_string_allow(self):
        """Test v1.1: 'allow' parses correctly."""
        assert Verdict.from_string("allow") == Verdict.ALLOW
        assert Verdict.from_string("ALLOW") == Verdict.ALLOW

    def test_from_string_constrain(self):
        """Test v1.1: 'constrain' parses correctly."""
        assert Verdict.from_string("constrain") == Verdict.CONSTRAIN
        assert Verdict.from_string("CONSTRAIN") == Verdict.CONSTRAIN

    def test_from_string_require_approval_underscore(self):
        """Test v1.1: 'require_approval' parses correctly."""
        assert Verdict.from_string("require_approval") == Verdict.REQUIRE_APPROVAL
        assert Verdict.from_string("REQUIRE_APPROVAL") == Verdict.REQUIRE_APPROVAL

    def test_from_string_block(self):
        """Test v1.1: 'block' parses correctly."""
        assert Verdict.from_string("block") == Verdict.BLOCK
        assert Verdict.from_string("BLOCK") == Verdict.BLOCK

    def test_from_string_halt(self):
        """Test v1.1: 'halt' parses correctly."""
        assert Verdict.from_string("halt") == Verdict.HALT
        assert Verdict.from_string("HALT") == Verdict.HALT

    # Test from_string() - None and invalid values
    def test_from_string_none_returns_allow(self):
        """Test that None returns ALLOW."""
        assert Verdict.from_string(None) == Verdict.ALLOW

    def test_from_string_invalid_returns_allow(self):
        """Test that invalid values return ALLOW."""
        assert Verdict.from_string("invalid") == Verdict.ALLOW
        assert Verdict.from_string("unknown") == Verdict.ALLOW
        assert Verdict.from_string("") == Verdict.ALLOW
        assert Verdict.from_string("random_string") == Verdict.ALLOW

    # Test priority property
    def test_priority_halt_is_5(self):
        """Test HALT priority is 5 (highest)."""
        assert Verdict.HALT.priority == 5

    def test_priority_block_is_4(self):
        """Test BLOCK priority is 4."""
        assert Verdict.BLOCK.priority == 4

    def test_priority_require_approval_is_3(self):
        """Test REQUIRE_APPROVAL priority is 3."""
        assert Verdict.REQUIRE_APPROVAL.priority == 3

    def test_priority_constrain_is_2(self):
        """Test CONSTRAIN priority is 2."""
        assert Verdict.CONSTRAIN.priority == 2

    def test_priority_allow_is_1(self):
        """Test ALLOW priority is 1 (lowest)."""
        assert Verdict.ALLOW.priority == 1

    def test_priority_ordering(self):
        """Test that priorities are correctly ordered."""
        assert Verdict.HALT.priority > Verdict.BLOCK.priority
        assert Verdict.BLOCK.priority > Verdict.REQUIRE_APPROVAL.priority
        assert Verdict.REQUIRE_APPROVAL.priority > Verdict.CONSTRAIN.priority
        assert Verdict.CONSTRAIN.priority > Verdict.ALLOW.priority

    # Test highest_priority()
    def test_highest_priority_returns_halt(self):
        """Test highest_priority returns HALT from mixed list."""
        verdicts = [Verdict.ALLOW, Verdict.HALT, Verdict.CONSTRAIN]
        assert Verdict.highest_priority(verdicts) == Verdict.HALT

    def test_highest_priority_returns_block(self):
        """Test highest_priority returns BLOCK when no HALT."""
        verdicts = [Verdict.ALLOW, Verdict.BLOCK, Verdict.CONSTRAIN]
        assert Verdict.highest_priority(verdicts) == Verdict.BLOCK

    def test_highest_priority_returns_require_approval(self):
        """Test highest_priority returns REQUIRE_APPROVAL when highest."""
        verdicts = [Verdict.ALLOW, Verdict.REQUIRE_APPROVAL, Verdict.CONSTRAIN]
        assert Verdict.highest_priority(verdicts) == Verdict.REQUIRE_APPROVAL

    def test_highest_priority_returns_constrain(self):
        """Test highest_priority returns CONSTRAIN when highest."""
        verdicts = [Verdict.ALLOW, Verdict.CONSTRAIN]
        assert Verdict.highest_priority(verdicts) == Verdict.CONSTRAIN

    def test_highest_priority_single_element(self):
        """Test highest_priority with single element list."""
        assert Verdict.highest_priority([Verdict.ALLOW]) == Verdict.ALLOW
        assert Verdict.highest_priority([Verdict.HALT]) == Verdict.HALT

    def test_highest_priority_empty_list_returns_allow(self):
        """Test highest_priority returns ALLOW for empty list."""
        assert Verdict.highest_priority([]) == Verdict.ALLOW

    def test_highest_priority_all_same(self):
        """Test highest_priority with all same verdicts."""
        verdicts = [Verdict.CONSTRAIN, Verdict.CONSTRAIN, Verdict.CONSTRAIN]
        assert Verdict.highest_priority(verdicts) == Verdict.CONSTRAIN

    def test_highest_priority_all_verdicts(self):
        """Test highest_priority with all verdict types."""
        all_verdicts = [
            Verdict.ALLOW,
            Verdict.CONSTRAIN,
            Verdict.REQUIRE_APPROVAL,
            Verdict.BLOCK,
            Verdict.HALT,
        ]
        assert Verdict.highest_priority(all_verdicts) == Verdict.HALT

    # Test should_stop()
    def test_should_stop_halt_returns_true(self):
        """Test HALT should_stop returns True."""
        assert Verdict.HALT.should_stop() is True

    def test_should_stop_block_returns_true(self):
        """Test BLOCK should_stop returns True."""
        assert Verdict.BLOCK.should_stop() is True

    def test_should_stop_require_approval_returns_false(self):
        """Test REQUIRE_APPROVAL should_stop returns False."""
        assert Verdict.REQUIRE_APPROVAL.should_stop() is False

    def test_should_stop_constrain_returns_false(self):
        """Test CONSTRAIN should_stop returns False."""
        assert Verdict.CONSTRAIN.should_stop() is False

    def test_should_stop_allow_returns_false(self):
        """Test ALLOW should_stop returns False."""
        assert Verdict.ALLOW.should_stop() is False

    # Test requires_approval()
    def test_requires_approval_require_approval_returns_true(self):
        """Test REQUIRE_APPROVAL requires_approval returns True."""
        assert Verdict.REQUIRE_APPROVAL.requires_approval() is True

    def test_requires_approval_halt_returns_false(self):
        """Test HALT requires_approval returns False."""
        assert Verdict.HALT.requires_approval() is False

    def test_requires_approval_block_returns_false(self):
        """Test BLOCK requires_approval returns False."""
        assert Verdict.BLOCK.requires_approval() is False

    def test_requires_approval_constrain_returns_false(self):
        """Test CONSTRAIN requires_approval returns False."""
        assert Verdict.CONSTRAIN.requires_approval() is False

    def test_requires_approval_allow_returns_false(self):
        """Test ALLOW requires_approval returns False."""
        assert Verdict.ALLOW.requires_approval() is False


class TestWorkflowSpanBuffer:
    """Tests for WorkflowSpanBuffer dataclass."""

    def test_creation_with_required_fields(self):
        """Test creation with only required fields."""
        buffer = WorkflowSpanBuffer(
            workflow_id="wf-123",
            run_id="run-456",
            workflow_type="TestWorkflow",
            task_queue="test-queue",
        )
        assert buffer.workflow_id == "wf-123"
        assert buffer.run_id == "run-456"
        assert buffer.workflow_type == "TestWorkflow"
        assert buffer.task_queue == "test-queue"

    def test_default_values(self):
        """Test that optional fields have correct default values."""
        buffer = WorkflowSpanBuffer(
            workflow_id="wf-123",
            run_id="run-456",
            workflow_type="TestWorkflow",
            task_queue="test-queue",
        )
        assert buffer.parent_workflow_id is None
        assert buffer.spans == []
        assert buffer.status is None
        assert buffer.error is None
        assert buffer.verdict is None
        assert buffer.verdict_reason is None
        assert buffer.pending_approval is False

    def test_creation_with_all_fields(self):
        """Test creation with all fields specified."""
        error_data = {"type": "TestError", "message": "Something failed"}
        spans_data = [{"name": "span1"}, {"name": "span2"}]

        buffer = WorkflowSpanBuffer(
            workflow_id="wf-123",
            run_id="run-456",
            workflow_type="TestWorkflow",
            task_queue="test-queue",
            parent_workflow_id="parent-wf-789",
            spans=spans_data,
            status="completed",
            error=error_data,
            verdict=Verdict.BLOCK,
            verdict_reason="Policy violation",
            pending_approval=True,
        )

        assert buffer.workflow_id == "wf-123"
        assert buffer.run_id == "run-456"
        assert buffer.workflow_type == "TestWorkflow"
        assert buffer.task_queue == "test-queue"
        assert buffer.parent_workflow_id == "parent-wf-789"
        assert buffer.spans == spans_data
        assert buffer.status == "completed"
        assert buffer.error == error_data
        assert buffer.verdict == Verdict.BLOCK
        assert buffer.verdict_reason == "Policy violation"
        assert buffer.pending_approval is True

    def test_spans_default_factory_isolation(self):
        """Test that spans default factory creates separate lists."""
        buffer1 = WorkflowSpanBuffer(
            workflow_id="wf-1",
            run_id="run-1",
            workflow_type="TestWorkflow",
            task_queue="test-queue",
        )
        buffer2 = WorkflowSpanBuffer(
            workflow_id="wf-2",
            run_id="run-2",
            workflow_type="TestWorkflow",
            task_queue="test-queue",
        )

        buffer1.spans.append({"name": "span1"})
        assert len(buffer1.spans) == 1
        assert len(buffer2.spans) == 0  # Should not be affected

    def test_status_values(self):
        """Test various status values."""
        statuses = ["completed", "failed", "cancelled", "terminated"]
        for status in statuses:
            buffer = WorkflowSpanBuffer(
                workflow_id="wf-123",
                run_id="run-456",
                workflow_type="TestWorkflow",
                task_queue="test-queue",
                status=status,
            )
            assert buffer.status == status


class TestGuardrailsCheckResult:
    """Tests for GuardrailsCheckResult dataclass."""

    def test_creation_with_required_fields(self):
        """Test creation with required fields."""
        result = GuardrailsCheckResult(
            redacted_input={"key": "[REDACTED]"},
            input_type="activity_input",
        )
        assert result.redacted_input == {"key": "[REDACTED]"}
        assert result.input_type == "activity_input"

    def test_default_values(self):
        """Test default values for optional fields."""
        result = GuardrailsCheckResult(
            redacted_input={"key": "value"},
            input_type="activity_input",
        )
        assert result.raw_logs is None
        assert result.validation_passed is True
        assert result.reasons == []

    def test_creation_with_all_fields(self):
        """Test creation with all fields."""
        raw_logs = {"log_id": "123", "timestamp": "2024-01-01"}
        reasons = [
            {"type": "pii", "field": "email", "reason": "Contains PII"},
            {"type": "validation", "field": "amount", "reason": "Exceeds limit"},
        ]

        result = GuardrailsCheckResult(
            redacted_input={"email": "[REDACTED]", "amount": 1000},
            input_type="activity_output",
            raw_logs=raw_logs,
            validation_passed=False,
            reasons=reasons,
        )

        assert result.redacted_input == {"email": "[REDACTED]", "amount": 1000}
        assert result.input_type == "activity_output"
        assert result.raw_logs == raw_logs
        assert result.validation_passed is False
        assert result.reasons == reasons

    def test_get_reason_strings_extracts_reasons(self):
        """Test get_reason_strings extracts reason field from reasons list."""
        reasons = [
            {"type": "pii", "field": "email", "reason": "Contains PII"},
            {"type": "validation", "field": "amount", "reason": "Exceeds limit"},
        ]
        result = GuardrailsCheckResult(
            redacted_input={},
            input_type="activity_input",
            reasons=reasons,
        )

        reason_strings = result.get_reason_strings()
        assert reason_strings == ["Contains PII", "Exceeds limit"]

    def test_get_reason_strings_empty_reasons(self):
        """Test get_reason_strings with empty reasons list."""
        result = GuardrailsCheckResult(
            redacted_input={},
            input_type="activity_input",
            reasons=[],
        )
        assert result.get_reason_strings() == []

    def test_get_reason_strings_missing_reason_field(self):
        """Test get_reason_strings skips entries without reason field."""
        reasons = [
            {"type": "pii", "field": "email"},  # No reason field
            {"type": "validation", "field": "amount", "reason": "Exceeds limit"},
            {"reason": ""},  # Empty reason
        ]
        result = GuardrailsCheckResult(
            redacted_input={},
            input_type="activity_input",
            reasons=reasons,
        )

        reason_strings = result.get_reason_strings()
        assert reason_strings == ["Exceeds limit"]

    def test_get_reason_strings_only_reason_field(self):
        """Test get_reason_strings with reasons containing only reason field."""
        reasons = [
            {"reason": "First reason"},
            {"reason": "Second reason"},
        ]
        result = GuardrailsCheckResult(
            redacted_input={},
            input_type="activity_input",
            reasons=reasons,
        )

        reason_strings = result.get_reason_strings()
        assert reason_strings == ["First reason", "Second reason"]

    def test_reasons_default_factory_isolation(self):
        """Test that reasons default factory creates separate lists."""
        result1 = GuardrailsCheckResult(
            redacted_input={},
            input_type="activity_input",
        )
        result2 = GuardrailsCheckResult(
            redacted_input={},
            input_type="activity_input",
        )

        result1.reasons.append({"reason": "test"})
        assert len(result1.reasons) == 1
        assert len(result2.reasons) == 0


class TestGovernanceVerdictResponse:
    """Tests for GovernanceVerdictResponse dataclass."""

    def test_creation_with_required_fields(self):
        """Test creation with only required verdict field."""
        response = GovernanceVerdictResponse(verdict=Verdict.ALLOW)
        assert response.verdict == Verdict.ALLOW

    def test_default_values(self):
        """Test default values for optional fields."""
        response = GovernanceVerdictResponse(verdict=Verdict.ALLOW)
        assert response.reason is None
        assert response.policy_id is None
        assert response.risk_score == 0.0
        assert response.metadata is None
        assert response.governance_event_id is None
        assert response.guardrails_result is None
        assert response.trust_tier is None
        assert response.behavioral_violations is None
        assert response.alignment_score is None
        assert response.approval_id is None
        assert response.constraints is None

    def test_creation_with_all_fields(self):
        """Test creation with all fields specified."""
        guardrails = GuardrailsCheckResult(
            redacted_input={},
            input_type="activity_input",
        )
        constraints = [{"type": "rate_limit", "value": 100}]
        metadata = {"source": "test"}

        response = GovernanceVerdictResponse(
            verdict=Verdict.CONSTRAIN,
            reason="Rate limited",
            policy_id="policy-123",
            risk_score=0.75,
            metadata=metadata,
            governance_event_id="event-456",
            guardrails_result=guardrails,
            trust_tier="standard",
            behavioral_violations=["violation1", "violation2"],
            alignment_score=0.85,
            approval_id="approval-789",
            constraints=constraints,
        )

        assert response.verdict == Verdict.CONSTRAIN
        assert response.reason == "Rate limited"
        assert response.policy_id == "policy-123"
        assert response.risk_score == 0.75
        assert response.metadata == metadata
        assert response.governance_event_id == "event-456"
        assert response.guardrails_result == guardrails
        assert response.trust_tier == "standard"
        assert response.behavioral_violations == ["violation1", "violation2"]
        assert response.alignment_score == 0.85
        assert response.approval_id == "approval-789"
        assert response.constraints == constraints

    # Test action property - backward compat
    def test_action_property_allow_returns_continue(self):
        """Test action property returns 'continue' for ALLOW verdict."""
        response = GovernanceVerdictResponse(verdict=Verdict.ALLOW)
        assert response.action == "continue"

    def test_action_property_halt_returns_stop(self):
        """Test action property returns 'stop' for HALT verdict."""
        response = GovernanceVerdictResponse(verdict=Verdict.HALT)
        assert response.action == "stop"

    def test_action_property_require_approval_returns_hyphenated(self):
        """Test action property returns 'require-approval' for REQUIRE_APPROVAL verdict."""
        response = GovernanceVerdictResponse(verdict=Verdict.REQUIRE_APPROVAL)
        assert response.action == "require-approval"

    def test_action_property_block_returns_value(self):
        """Test action property returns verdict value for BLOCK."""
        response = GovernanceVerdictResponse(verdict=Verdict.BLOCK)
        assert response.action == "block"

    def test_action_property_constrain_returns_value(self):
        """Test action property returns verdict value for CONSTRAIN."""
        response = GovernanceVerdictResponse(verdict=Verdict.CONSTRAIN)
        assert response.action == "constrain"

    # Test from_dict() - v1.0 responses
    def test_from_dict_v10_continue_action(self):
        """Test parsing v1.0 response with 'continue' action."""
        data = {"action": "continue", "reason": "Allowed by policy"}
        response = GovernanceVerdictResponse.from_dict(data)

        assert response.verdict == Verdict.ALLOW
        assert response.reason == "Allowed by policy"

    def test_from_dict_v10_stop_action(self):
        """Test parsing v1.0 response with 'stop' action."""
        data = {"action": "stop", "reason": "Blocked by policy"}
        response = GovernanceVerdictResponse.from_dict(data)

        assert response.verdict == Verdict.HALT
        assert response.reason == "Blocked by policy"

    def test_from_dict_v10_require_approval_action(self):
        """Test parsing v1.0 response with 'require-approval' action."""
        data = {"action": "require-approval", "reason": "Needs review"}
        response = GovernanceVerdictResponse.from_dict(data)

        assert response.verdict == Verdict.REQUIRE_APPROVAL
        assert response.reason == "Needs review"

    def test_from_dict_v10_full_response(self):
        """Test parsing full v1.0 response with all fields."""
        data = {
            "action": "continue",
            "reason": "Allowed",
            "policy_id": "policy-123",
            "risk_score": 0.25,
            "metadata": {"key": "value"},
            "governance_event_id": "event-456",
        }
        response = GovernanceVerdictResponse.from_dict(data)

        assert response.verdict == Verdict.ALLOW
        assert response.reason == "Allowed"
        assert response.policy_id == "policy-123"
        assert response.risk_score == 0.25
        assert response.metadata == {"key": "value"}
        assert response.governance_event_id == "event-456"

    # Test from_dict() - v1.1 responses
    def test_from_dict_v11_allow_verdict(self):
        """Test parsing v1.1 response with 'allow' verdict."""
        data = {"verdict": "allow", "reason": "Allowed"}
        response = GovernanceVerdictResponse.from_dict(data)

        assert response.verdict == Verdict.ALLOW
        assert response.reason == "Allowed"

    def test_from_dict_v11_constrain_verdict(self):
        """Test parsing v1.1 response with 'constrain' verdict."""
        data = {"verdict": "constrain", "reason": "Rate limited"}
        response = GovernanceVerdictResponse.from_dict(data)

        assert response.verdict == Verdict.CONSTRAIN
        assert response.reason == "Rate limited"

    def test_from_dict_v11_require_approval_verdict(self):
        """Test parsing v1.1 response with 'require_approval' verdict."""
        data = {"verdict": "require_approval", "reason": "Needs approval"}
        response = GovernanceVerdictResponse.from_dict(data)

        assert response.verdict == Verdict.REQUIRE_APPROVAL
        assert response.reason == "Needs approval"

    def test_from_dict_v11_block_verdict(self):
        """Test parsing v1.1 response with 'block' verdict."""
        data = {"verdict": "block", "reason": "Blocked"}
        response = GovernanceVerdictResponse.from_dict(data)

        assert response.verdict == Verdict.BLOCK
        assert response.reason == "Blocked"

    def test_from_dict_v11_halt_verdict(self):
        """Test parsing v1.1 response with 'halt' verdict."""
        data = {"verdict": "halt", "reason": "Halted"}
        response = GovernanceVerdictResponse.from_dict(data)

        assert response.verdict == Verdict.HALT
        assert response.reason == "Halted"

    def test_from_dict_v11_full_response(self):
        """Test parsing full v1.1 response with all new fields."""
        data = {
            "verdict": "constrain",
            "reason": "Rate limited",
            "trust_tier": "elevated",
            "behavioral_violations": ["violation1"],
            "alignment_score": 0.92,
            "approval_id": "approval-123",
            "constraints": [{"type": "rate_limit", "value": 50}],
        }
        response = GovernanceVerdictResponse.from_dict(data)

        assert response.verdict == Verdict.CONSTRAIN
        assert response.reason == "Rate limited"
        assert response.trust_tier == "elevated"
        assert response.behavioral_violations == ["violation1"]
        assert response.alignment_score == 0.92
        assert response.approval_id == "approval-123"
        assert response.constraints == [{"type": "rate_limit", "value": 50}]

    def test_from_dict_verdict_takes_precedence_over_action(self):
        """Test that verdict field takes precedence over action field."""
        data = {
            "verdict": "block",
            "action": "continue",  # Should be ignored
        }
        response = GovernanceVerdictResponse.from_dict(data)

        assert response.verdict == Verdict.BLOCK

    # Test from_dict() - with guardrails_result
    def test_from_dict_with_guardrails_result(self):
        """Test parsing response with guardrails_result."""
        data = {
            "verdict": "allow",
            "guardrails_result": {
                "redacted_input": {"email": "[REDACTED]"},
                "input_type": "activity_input",
                "raw_logs": {"log_id": "123"},
                "validation_passed": True,
                "reasons": [{"reason": "PII redacted"}],
            },
        }
        response = GovernanceVerdictResponse.from_dict(data)

        assert response.verdict == Verdict.ALLOW
        assert response.guardrails_result is not None
        assert response.guardrails_result.redacted_input == {"email": "[REDACTED]"}
        assert response.guardrails_result.input_type == "activity_input"
        assert response.guardrails_result.raw_logs == {"log_id": "123"}
        assert response.guardrails_result.validation_passed is True
        assert response.guardrails_result.reasons == [{"reason": "PII redacted"}]

    def test_from_dict_with_guardrails_result_validation_failed(self):
        """Test parsing response with guardrails_result that failed validation."""
        data = {
            "verdict": "block",
            "guardrails_result": {
                "redacted_input": {"amount": 10000},
                "input_type": "activity_output",
                "validation_passed": False,
                "reasons": [
                    {"type": "validation", "field": "amount", "reason": "Exceeds limit"}
                ],
            },
        }
        response = GovernanceVerdictResponse.from_dict(data)

        assert response.verdict == Verdict.BLOCK
        assert response.guardrails_result is not None
        assert response.guardrails_result.validation_passed is False
        assert len(response.guardrails_result.reasons) == 1
        assert response.guardrails_result.get_reason_strings() == ["Exceeds limit"]

    def test_from_dict_guardrails_result_minimal(self):
        """Test parsing response with minimal guardrails_result."""
        data = {
            "verdict": "allow",
            "guardrails_result": {
                "redacted_input": None,
                "input_type": "",
            },
        }
        response = GovernanceVerdictResponse.from_dict(data)

        assert response.guardrails_result is not None
        assert response.guardrails_result.redacted_input is None
        assert response.guardrails_result.input_type == ""
        assert response.guardrails_result.raw_logs is None
        assert response.guardrails_result.validation_passed is True
        assert response.guardrails_result.reasons == []

    def test_from_dict_no_guardrails_result(self):
        """Test parsing response without guardrails_result."""
        data = {"verdict": "allow"}
        response = GovernanceVerdictResponse.from_dict(data)

        assert response.guardrails_result is None

    def test_from_dict_empty_guardrails_result(self):
        """Test parsing response with empty/falsy guardrails_result."""
        data = {"verdict": "allow", "guardrails_result": None}
        response = GovernanceVerdictResponse.from_dict(data)

        assert response.guardrails_result is None

        data2 = {"verdict": "allow", "guardrails_result": {}}
        response2 = GovernanceVerdictResponse.from_dict(data2)

        assert response2.guardrails_result is None

    # Edge cases
    def test_from_dict_empty_dict(self):
        """Test parsing empty dict defaults to ALLOW."""
        data = {}
        response = GovernanceVerdictResponse.from_dict(data)

        assert response.verdict == Verdict.ALLOW

    def test_from_dict_missing_risk_score_defaults_to_zero(self):
        """Test that missing risk_score defaults to 0.0."""
        data = {"verdict": "allow"}
        response = GovernanceVerdictResponse.from_dict(data)

        assert response.risk_score == 0.0

    def test_from_dict_null_reasons_becomes_empty_list(self):
        """Test that null reasons in guardrails_result becomes empty list."""
        data = {
            "verdict": "allow",
            "guardrails_result": {
                "redacted_input": {},
                "input_type": "activity_input",
                "reasons": None,
            },
        }
        response = GovernanceVerdictResponse.from_dict(data)

        assert response.guardrails_result.reasons == []
