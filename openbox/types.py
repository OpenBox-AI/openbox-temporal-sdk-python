# openbox/types.py
"""
Data types for workflow-boundary governance.

WorkflowEventType: Enum of workflow/activity lifecycle events
WorkflowSpanBuffer: Buffer for spans during workflow execution
GovernanceVerdictResponse: Response from governance API
GuardrailsCheckResult: Guardrails redaction result
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from enum import Enum


class WorkflowEventType(str, Enum):
    """Workflow lifecycle events for governance."""

    WORKFLOW_STARTED = "WorkflowStarted"
    WORKFLOW_COMPLETED = "WorkflowCompleted"
    WORKFLOW_FAILED = "WorkflowFailed"
    SIGNAL_RECEIVED = "SignalReceived"
    ACTIVITY_STARTED = "ActivityStarted"
    ACTIVITY_COMPLETED = "ActivityCompleted"


@dataclass
class WorkflowSpanBuffer:
    """
    Buffer for spans generated during workflow execution.

    NOTE: No timestamps stored here. The workflow interceptor passes workflow
    metadata to the activity, and the activity adds timestamps when it executes.
    This maintains workflow determinism (no time.time() calls in workflow code).
    """

    workflow_id: str
    run_id: str
    workflow_type: str
    task_queue: str
    parent_workflow_id: Optional[str] = None
    spans: List[dict] = field(default_factory=list)
    status: Optional[str] = None  # "completed", "failed", "cancelled", "terminated"
    error: Optional[Dict[str, Any]] = None

    # Governance verdict (set by workflow interceptor, checked by activity interceptor)
    # This allows signal handlers to block subsequent activities without sandbox issues
    verdict: Optional[str] = None  # "continue" or "stop"
    verdict_reason: Optional[str] = None

    # Pending approval: governance_event_id from /evaluate response when action is "request-approval"
    pending_approval_governance_event_id: Optional[str] = None


@dataclass
class GuardrailsCheckResult:
    """
    Guardrails check result from governance API.

    Contains redacted input/output that should replace the original activity data,
    plus validation results that can block execution.
    """

    redacted_input: Any  # Redacted activity_input or activity_output (JSON-decoded)
    input_type: str  # "activity_input" or "activity_output"
    raw_logs: Optional[Dict[str, Any]] = None  # Raw logs from guardrails evaluation
    validation_passed: bool = True  # If False, workflow should be stopped
    reasons: List[Dict[str, str]] = field(default_factory=list)  # [{type, field, reason}, ...]

    def get_reason_strings(self) -> List[str]:
        """Extract just the 'reason' field from each reason object."""
        return [r.get("reason", "") for r in self.reasons if r.get("reason")]


@dataclass
class GovernanceVerdictResponse:
    """
    Response from governance API evaluation.

    action: "continue" to allow, "stop" to terminate, "require-approval" for HITL
    governance_event_id: ID returned by /evaluate API for tracking this governance event
    guardrails_result: Optional redacted input from guardrails
    """

    action: str  # "continue" or "stop" or "require-approval"
    reason: Optional[str] = None
    policy_id: Optional[str] = None
    risk_score: float = 0.0
    metadata: Optional[Dict[str, Any]] = None
    governance_event_id: Optional[str] = None  # ID from /evaluate response
    guardrails_result: Optional[GuardrailsCheckResult] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GovernanceVerdictResponse":
        """Parse governance response from JSON dict."""
        guardrails_result = None
        if data.get("guardrails_result"):
            gr = data["guardrails_result"]
            guardrails_result = GuardrailsCheckResult(
                redacted_input=gr.get("redacted_input"),
                input_type=gr.get("input_type", ""),
                raw_logs=gr.get("raw_logs"),
                validation_passed=gr.get("validation_passed", True),
                reasons=gr.get("reasons") or [],
            )

        return cls(
            action=data.get("action", "continue"),
            reason=data.get("reason"),
            policy_id=data.get("policy_id"),
            risk_score=data.get("risk_score", 0.0),
            metadata=data.get("metadata"),
            governance_event_id=data.get("governance_event_id"),
            guardrails_result=guardrails_result,
        )
