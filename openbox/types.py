"""Data types for workflow-boundary governance.

``Verdict``, ``GuardrailsCheckResult`` (base name: ``GuardrailsResult``),
``WorkflowEventType`` (base name: ``EventType``), and evaluation-result parsing
come from ``openbox_core.contracts``. Public names, signatures, and behavior are
preserved.

SANDBOX SAFETY: ``openbox_core.contracts`` modules are pure (no network,
crypto, OTel, or wall-clock at import).
"""

from datetime import UTC, datetime
from typing import Any

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
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class GovernanceVerdictResponse(EvaluationResult):
    """Response from governance API evaluation.

    Thin subclass of the shared ``openbox_core`` ``EvaluationResult`` that keeps
    the Temporal SDK's public class name.
    """

    def __init__(
        self,
        verdict: Verdict,
        reason: str | None = None,
        policy_id: str | None = None,
        risk_score: float = 0.0,
        metadata: dict[str, Any] | None = None,
        governance_event_id: str | None = None,
        guardrails_result: GuardrailsCheckResult | None = None,
        trust_tier: str | None = None,
        behavioral_violations: list[str] | None = None,
        alignment_score: float | None = None,
        approval_id: str | None = None,
        constraints: list[dict[str, Any]] | None = None,
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
    def from_dict(cls, data: dict[str, Any]) -> "GovernanceVerdictResponse":
        """Parse a governance response (v1.0 and v1.1 compatible) via the
        shared parser."""
        base = EvaluationResult.from_dict(data)
        instance = cls.__new__(cls)
        instance.__dict__.update(base.__dict__)
        return instance
