# openbox/types.py
"""Data types for workflow-boundary governance — shims over the OpenBox base SDK.

``Verdict``, ``GuardrailsCheckResult`` (base name: ``GuardrailsResult``),
``WorkflowEventType`` (base name: ``EventType``), and the evaluation-result
parsing now come from ``openbox_core.contracts`` — the shared contracts every
OpenBox framework SDK consumes. Public names, signatures, and behavior are
preserved; ``GovernanceVerdictResponse`` additionally gains the base-SDK
fields Temporal never parsed before (``fallback_used``, ``diagnostics``,
``raw``).

SANDBOX SAFETY: ``openbox_core.contracts`` modules are pure (no network,
crypto, OTel, or wall-clock at import) — verified by the base SDK's
import-safety harness — so this module remains safe exactly where it was
safe before.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Shared contracts — the single source of truth for these shapes.
from openbox_core.contracts.events import EventType as WorkflowEventType  # noqa: F401
from openbox_core.contracts.results import (  # noqa: F401
    EvaluationResult,
    Verdict,
)
from openbox_core.contracts.results import (
    GuardrailsResult as GuardrailsCheckResult,  # noqa: F401  (Temporal-parity name)
)

# Re-export from errors.py for backward compatibility
from .errors import GovernanceBlockedError  # noqa: F401


def rfc3339_now() -> str:
    """Return current UTC time in RFC3339 format (e.g. '2026-03-08T12:00:00.000Z')."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


@dataclass
class WorkflowSpanBuffer:
    """Buffer for workflow governance state (verdicts, approvals, abort flags)."""

    workflow_id: str
    run_id: str
    workflow_type: str
    task_queue: str
    parent_workflow_id: Optional[str] = None
    spans: List[dict] = field(
        default_factory=list
    )  # kept for backward compat, always empty
    status: Optional[str] = None  # "completed", "failed", "cancelled", "terminated"
    error: Optional[Dict[str, Any]] = None

    # Governance verdict (set by workflow interceptor, checked by activity interceptor)
    verdict: Optional[Verdict] = None
    verdict_reason: Optional[str] = None

    # Pending approval: True when activity is waiting for human approval
    pending_approval: bool = False


class GovernanceVerdictResponse(EvaluationResult):
    """Response from governance API evaluation.

    Now a thin subclass of the shared ``openbox_core`` ``EvaluationResult`` —
    same public surface as before (``verdict``, ``reason``, ``policy_id``,
    ``risk_score``, ``metadata``, ``governance_event_id``,
    ``guardrails_result``, v1.1 fields, the ``action`` property, and
    ``from_dict``) plus the shared fields (``fallback_used``, ``diagnostics``,
    ``raw``, ``approval_expiration_time``).
    """

    def __init__(
        self,
        verdict: Verdict,
        reason: Optional[str] = None,
        policy_id: Optional[str] = None,
        risk_score: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
        governance_event_id: Optional[str] = None,
        guardrails_result: Optional[GuardrailsCheckResult] = None,
        trust_tier: Optional[str] = None,
        behavioral_violations: Optional[List[str]] = None,
        alignment_score: Optional[float] = None,
        approval_id: Optional[str] = None,
        constraints: Optional[List[Dict[str, Any]]] = None,
        **shared_fields: Any,
    ):
        super().__init__(
            verdict=verdict,
            reason=reason,
            policy_id=policy_id,
            risk_score=risk_score,
            metadata=metadata,
            governance_event_id=governance_event_id,
            guardrails=guardrails_result,
            trust_tier=trust_tier,
            behavioral_violations=behavioral_violations,
            alignment_score=alignment_score,
            approval_id=approval_id,
            constraints=constraints,
            **shared_fields,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GovernanceVerdictResponse":
        """Parse a governance response (v1.0 and v1.1 compatible) via the
        SHARED base-SDK parser — no hand-parsing of common fields remains."""
        base = EvaluationResult.from_dict(data)
        instance = cls.__new__(cls)
        instance.__dict__.update(base.__dict__)
        return instance
