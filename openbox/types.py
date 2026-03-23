# openbox/types.py
"""Data types and shared utilities for workflow-boundary governance."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Union
from enum import Enum


def rfc3339_now() -> str:
    """Return current UTC time in RFC3339 format (e.g. '2026-03-08T12:00:00.000Z')."""
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'


class GovernanceBlockedError(Exception):
    """Raised by OTel hooks when governance blocks an operation."""

    def __init__(self, verdict: Union[str, "Verdict"], reason: str, url: str = ""):
        # Normalize to Verdict enum for consistent downstream comparisons
        if isinstance(verdict, str):
            from_string = Verdict.from_string  # avoid circular at class level
            self.verdict = from_string(verdict)
        else:
            self.verdict = verdict
        self.reason = reason
        self.url = url
        super().__init__(f"Governance {self.verdict.value}: {reason}")


class WorkflowEventType(str, Enum):
    """Workflow lifecycle events for governance."""

    WORKFLOW_STARTED = "WorkflowStarted"
    WORKFLOW_COMPLETED = "WorkflowCompleted"
    WORKFLOW_FAILED = "WorkflowFailed"
    SIGNAL_RECEIVED = "SignalReceived"
    ACTIVITY_STARTED = "ActivityStarted"
    ACTIVITY_COMPLETED = "ActivityCompleted"


class Verdict(str, Enum):
    """5-tier graduated response. Priority: HALT > BLOCK > REQUIRE_APPROVAL > CONSTRAIN > ALLOW"""

    ALLOW = "allow"
    CONSTRAIN = "constrain"
    REQUIRE_APPROVAL = "require_approval"
    BLOCK = "block"
    HALT = "halt"

    @classmethod
    def from_string(cls, value: str) -> "Verdict":
        """Parse with v1.0 compat: 'continue'→ALLOW, 'stop'→HALT, 'require-approval'→REQUIRE_APPROVAL"""
        if value is None:
            return cls.ALLOW
        normalized = value.lower().replace("-", "_")
        if normalized == "continue":
            return cls.ALLOW
        if normalized == "stop":
            return cls.HALT
        if normalized in ("require_approval", "request_approval"):
            return cls.REQUIRE_APPROVAL
        try:
            return cls(normalized)
        except ValueError:
            return cls.ALLOW

    @property
    def priority(self) -> int:
        """Priority for aggregation: HALT=5, BLOCK=4, REQUIRE_APPROVAL=3, CONSTRAIN=2, ALLOW=1"""
        return {Verdict.ALLOW: 1, Verdict.CONSTRAIN: 2, Verdict.REQUIRE_APPROVAL: 3, Verdict.BLOCK: 4, Verdict.HALT: 5}[self]

    @classmethod
    def highest_priority(cls, verdicts: List["Verdict"]) -> "Verdict":
        """Get highest priority verdict from list. Returns ALLOW if empty."""
        return max(verdicts, key=lambda v: v.priority) if verdicts else cls.ALLOW

    def should_stop(self) -> bool:
        """True if BLOCK or HALT."""
        return self in (Verdict.BLOCK, Verdict.HALT)

    def requires_approval(self) -> bool:
        """True if REQUIRE_APPROVAL."""
        return self == Verdict.REQUIRE_APPROVAL


@dataclass
class WorkflowSpanBuffer:
    """Buffer for workflow governance state (verdicts, approvals, abort flags)."""

    workflow_id: str
    run_id: str
    workflow_type: str
    task_queue: str
    parent_workflow_id: Optional[str] = None
    spans: List[dict] = field(default_factory=list)  # kept for backward compat, always empty
    status: Optional[str] = None  # "completed", "failed", "cancelled", "terminated"
    error: Optional[Dict[str, Any]] = None

    # Governance verdict (set by workflow interceptor, checked by activity interceptor)
    verdict: Optional[Verdict] = None
    verdict_reason: Optional[str] = None

    # Pending approval: True when activity is waiting for human approval
    pending_approval: bool = False


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
    """Response from governance API evaluation."""

    verdict: Verdict  # v1.1: 5-tier verdict
    reason: Optional[str] = None
    # v1.0 fields (kept for compatibility)
    policy_id: Optional[str] = None
    risk_score: float = 0.0
    metadata: Optional[Dict[str, Any]] = None
    governance_event_id: Optional[str] = None
    guardrails_result: Optional[GuardrailsCheckResult] = None
    # v1.1 fields
    trust_tier: Optional[str] = None
    behavioral_violations: Optional[List[str]] = None
    alignment_score: Optional[float] = None
    approval_id: Optional[str] = None
    constraints: Optional[List[Dict[str, Any]]] = None

    @property
    def action(self) -> str:
        """Backward compat: return action string from verdict."""
        if self.verdict == Verdict.ALLOW:
            return "continue"
        if self.verdict == Verdict.HALT:
            return "stop"
        if self.verdict == Verdict.REQUIRE_APPROVAL:
            return "require-approval"
        return self.verdict.value

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GovernanceVerdictResponse":
        """Parse governance response from JSON dict (v1.0 and v1.1 compatible)."""
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

        # Parse verdict (v1.1) or action (v1.0)
        verdict = Verdict.from_string(data.get("verdict") or data.get("action", "continue"))

        return cls(
            verdict=verdict,
            reason=data.get("reason"),
            policy_id=data.get("policy_id"),
            risk_score=data.get("risk_score", 0.0),
            metadata=data.get("metadata"),
            governance_event_id=data.get("governance_event_id"),
            guardrails_result=guardrails_result,
            trust_tier=data.get("trust_tier"),
            behavioral_violations=data.get("behavioral_violations"),
            alignment_score=data.get("alignment_score"),
            approval_id=data.get("approval_id"),
            constraints=data.get("constraints"),
        )
