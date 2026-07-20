"""Data types for workflow-boundary governance.

``Verdict``, ``GuardrailsCheckResult`` (base name: ``GuardrailsResult``),
``WorkflowEventType`` (base name: ``EventType``), and evaluation-result parsing
come from ``openbox_core.contracts``. Public names, signatures, and behavior are
preserved.

SANDBOX SAFETY: ``openbox_core.contracts`` modules are pure (no network,
crypto, OTel, or wall-clock at import).
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from openbox_core.contracts.events import EventType as WorkflowEventType  # noqa: F401
from openbox_core.contracts.results import (  # noqa: F401
    EvaluationResult,
    Verdict,
)
from openbox_core.contracts.results import (
    GuardrailsResult as GuardrailsCheckResult,  # noqa: F401  (Temporal-parity name)
)

from .errors import GovernanceBlockedError  # noqa: F401


def rfc3339_now() -> str:
    """Return current UTC time in RFC3339 format (e.g. '2026-03-08T12:00:00.000Z')."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class GovernanceVerdictResponse(EvaluationResult):
    """Response from governance API evaluation.

    Thin subclass of the shared ``openbox_core`` ``EvaluationResult`` that keeps
    the Temporal SDK's public class name.
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
        shared parser."""
        base = EvaluationResult.from_dict(data)
        instance = cls.__new__(cls)
        instance.__dict__.update(base.__dict__)
        return instance
