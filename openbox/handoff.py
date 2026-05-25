# openbox/handoff.py
"""Public multi-agent handoff API.

``emit_handoff`` posts a ``Handoff`` governance event identifying the sending
agent (``from_agent_did``) and the shared ``multi_agent_session_id``. The
receiving agent (``to_agent``) is derived server-side from the authenticated
signed identity and is NEVER sent.

SANDBOX SAFETY: this module is import-safe in the workflow sandbox — it only
imports ``temporalio.workflow`` and routes through the existing
``send_governance_event`` activity, so the actual signing/HTTP happens in
activity context (never in the workflow sandbox). It is therefore safe to
export eagerly from ``openbox/__init__.py``.

Core contract: Handoff short-circuits governance (skips OPA/Guardrails/AGE),
writes one ``session_handoffs`` row, and returns ALLOW. ``from_agent_did`` is
resolved tenant-scoped (cross-org blocked, intra-org allowed).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional


def _to_rfc3339(ts: datetime) -> str:
    """Format a datetime as RFC3339 (UTC, millisecond precision, trailing Z)."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def build_handoff_payload(
    from_agent_did: str,
    multi_agent_session_id: str,
    ts: Optional[datetime] = None,
) -> dict:
    """Build + validate a Handoff event payload.

    Mirrors Core's ``ValidateHandoffPayload``: both ``from_agent_did`` and
    ``multi_agent_session_id`` are required and non-empty. Raises ``ValueError``
    before any network call so misconfiguration fails fast and locally.

    Note: ``agent_did`` is deliberately NOT included — Core's payload contract
    carries only ``multi_agent_session_id`` + ``from_agent_did``.
    """
    if not from_agent_did or not from_agent_did.strip():
        raise ValueError("emit_handoff: from_agent_did is required and must be non-empty")
    if not multi_agent_session_id or not multi_agent_session_id.strip():
        raise ValueError(
            "emit_handoff: multi_agent_session_id is required and must be non-empty"
        )

    payload = {
        "source": "workflow-telemetry",
        "event_type": "Handoff",
        "from_agent_did": from_agent_did,
        "multi_agent_session_id": multi_agent_session_id,
    }
    if ts is not None:
        # Activity uses setdefault, so an explicit timestamp is preserved.
        payload["timestamp"] = _to_rfc3339(ts)
    return payload


async def emit_handoff(
    from_agent_did: str,
    multi_agent_session_id: str,
    ts: Optional[datetime] = None,
    *,
    timeout: float = 30.0,
    on_api_error: str = "fail_open",
) -> Optional[dict]:
    """Emit a multi-agent Handoff event from within a Temporal workflow.

    Routes through the ``send_governance_event`` activity (which signs + sends),
    keeping all crypto/HTTP off the workflow sandbox path. Call from workflow code.

    Args:
        from_agent_did: DID of the agent handing off (the sender).
        multi_agent_session_id: Shared session id grouping the agents.
        ts: Optional event timestamp; if omitted the activity stamps it.
        timeout: Activity start-to-close budget (seconds).
        on_api_error: "fail_open" or "fail_closed".

    Returns:
        The activity result dict (ALLOW verdict) or None on fail_open error.

    Raises:
        ValueError: if required fields are missing/empty.
    """
    payload = build_handoff_payload(from_agent_did, multi_agent_session_id, ts)

    # Reuse the workflow-sandbox-safe activity dispatcher (no signing here).
    from .workflow_interceptor import _send_governance_event

    return await _send_governance_event(payload, timeout, on_api_error)
