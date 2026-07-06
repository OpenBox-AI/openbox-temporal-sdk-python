"""Multi-agent primitives: handoff events + session-context propagation.

The SDK does NOT own routing, session-id minting, or an agent registry — those
are the application's responsibility. The app supplies the shared
``multi_agent_session_id`` (via a workflow memo) and the SDK only propagates it
onto the governance events it already emits, plus exposes ``emit_handoff`` for
explicit agent-to-agent handoffs.

Supply mechanism
----------------
- App sets ``memo={MEMO_KEY: session_id}`` at ``start_workflow(...)``.
- The workflow interceptor reads it with ``workflow.memo_value`` (deterministic,
  replay-safe) and tags workflow events with it.
- The workflow outbound interceptor stamps it onto a Temporal ``HEADER_KEY``
  header on every scheduled activity; the activity interceptor reads that header
  and tags activity (and hook) events.

SANDBOX SAFETY: this module is import-safe in the workflow sandbox — at module
level it only imports ``temporalio.workflow``, and routes ``emit_handoff``
through the existing ``send_governance_event`` activity (signing/HTTP happen in
activity context, never in the workflow sandbox). Safe to export eagerly from
``openbox/__init__.py``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Optional

# App-facing memo key (start_workflow(memo={...})) → SDK reads on the workflow side.
MEMO_KEY = "openbox_multi_agent_session_id"
# Workflow → activity propagation header. Namespaced to avoid clashing with
# Temporal's bundled TracingInterceptor headers.
HEADER_KEY = "openbox-multi-agent-session-id"


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

    Both ``from_agent_did`` and ``multi_agent_session_id`` are required and
    non-empty. Raises ``ValueError`` before any network call so misconfiguration
    fails fast and locally.

    Note: ``agent_did`` is deliberately NOT included — the receiver is derived
    server-side from the authenticated signed identity, never sent.
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
        # The activity uses setdefault, so an explicit timestamp is preserved.
        payload["timestamp"] = _to_rfc3339(ts)
    return payload


async def emit_handoff(
    multi_agent_session_id: str,
    from_agent_did: str,
    ts: Optional[datetime] = None,
    *,
    timeout: float = 30.0,
    on_api_error: str = "fail_open",
) -> Optional[dict]:
    """Emit a multi-agent Handoff event from within a Temporal workflow.

    Routes through the ``send_governance_event`` activity (which signs + sends),
    keeping all crypto/HTTP off the workflow sandbox path. Call from workflow code.

    Args:
        multi_agent_session_id: Shared session id grouping the agents.
        from_agent_did: DID of the agent handing off (the sender).
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




def read_session_from_memo() -> Optional[str]:
    """Session id from the workflow memo, or None. Deterministic / replay-safe."""
    from temporalio import workflow

    return workflow.memo_value(MEMO_KEY, None)


def inject_session_header(
    headers: Mapping[str, Any], session_id: str, payload_converter: Any
) -> dict:
    """Return ``headers`` plus the session-id header (pure; caller passes converter)."""
    return {**headers, HEADER_KEY: payload_converter.to_payload(session_id)}


def read_session_from_header(
    headers: Optional[Mapping[str, Any]], payload_converter: Any
) -> Optional[str]:
    """Session id from a Temporal headers mapping, or None."""
    payload = headers.get(HEADER_KEY) if headers else None
    if payload is None:
        return None
    return payload_converter.from_payload(payload, str)
