# openbox/hook_governance.py
"""Hook-level governance evaluation for HTTP requests and file operations.

Sends per-operation governance evaluations to OpenBox Core during activity
execution. Used by OTel hooks in otel_setup.py to evaluate each HTTP request
and file operation at two stages: 'started' (before) and 'completed' (after).

Architecture:
    1. otel_setup.py hooks detect an operation (HTTP request, file open, etc.)
    2. Hook builds a span_data dict and hook_trigger dict
    3. Hook calls evaluate_sync() or evaluate_async()
    4. This module: looks up activity context, assembles payload, sends to API
    5. If verdict is BLOCK/HALT → raises GovernanceBlockedError

Shared by all operation types (HTTP, file I/O, future: database queries).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

import httpx

if TYPE_CHECKING:
    from .span_processor import WorkflowSpanProcessor

logger = logging.getLogger(__name__)

# Error policy constants
FAIL_OPEN = "fail_open"
FAIL_CLOSED = "fail_closed"

# Module-level config (set once by configure())
_api_url: str = ""
_api_key: str = ""
_api_timeout: float = 30.0
_on_api_error: str = FAIL_OPEN
_span_processor: Optional["WorkflowSpanProcessor"] = None


def configure(
    api_url: str,
    api_key: str,
    span_processor: "WorkflowSpanProcessor",
    *,
    api_timeout: float = 30.0,
    on_api_error: str = "fail_open",
) -> None:
    """Set governance config. Called once by setup_opentelemetry_for_governance().

    Args:
        api_url: OpenBox Core API URL
        api_key: API key for authentication
        span_processor: WorkflowSpanProcessor for activity context lookup
        api_timeout: Timeout for governance API calls (seconds)
        on_api_error: Error policy — "fail_open" or "fail_closed"
    """
    global _api_url, _api_key, _api_timeout, _on_api_error, _span_processor
    _api_url = api_url.rstrip("/")
    _api_key = api_key
    _api_timeout = api_timeout
    _on_api_error = on_api_error
    _span_processor = span_processor
    logger.info("Hook-level governance configured")


def is_configured() -> bool:
    """Check if hook-level governance is active."""
    return bool(_api_url and _span_processor is not None)


def _auth_headers() -> dict:
    """Build standard auth headers for governance API calls."""
    return {
        "Authorization": f"Bearer {_api_key}",
        "User-Agent": "OpenBox-SDK/1.0",
    }


def _build_payload(
    span: Any,
    hook_trigger: Dict[str, Any],
    span_data: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Build governance evaluation payload from activity context + hook trigger.

    Returns None if no activity context found (not inside a governed activity).

    Args:
        span: OTel span for the current operation
        hook_trigger: Dict describing what triggered governance (type, stage, etc.)
        span_data: Optional span data dict to store in the buffer
    """
    if _span_processor is None:
        return None

    # Look up activity context by trace_id
    # Use get_span_context() for compatibility with both _Span and NonRecordingSpan
    span_context = span.get_span_context() if hasattr(span, 'get_span_context') else span.context
    activity_context = _span_processor.get_activity_context_by_trace(span_context.trace_id)
    if activity_context is None:
        return None

    workflow_id = activity_context.get("workflow_id")
    activity_id = activity_context.get("activity_id")

    # Store span data in buffer if provided, tagged with activity_id for retrieval
    buffer = _span_processor.get_buffer(workflow_id) if workflow_id else None
    if buffer and span_data:
        if activity_id and "activity_id" not in span_data:
            span_data["activity_id"] = activity_id
        buffer.spans.append(span_data)

    # Collect all activity spans for the payload
    all_activity_spans = []
    if buffer and activity_id:
        all_activity_spans = [
            s for s in buffer.spans
            if (s.get("activity_id") == activity_id
                or s.get("attributes", {}).get("temporal.activity_id") == activity_id)
        ]

    # Assemble payload
    payload = dict(activity_context)
    payload["spans"] = all_activity_spans
    payload["span_count"] = len(all_activity_spans)
    payload["hook_trigger"] = hook_trigger
    from .activities import _rfc3339_now
    payload["timestamp"] = _rfc3339_now()
    return payload


def _handle_verdict(data: Dict[str, Any], identifier: str) -> None:
    """Check API response verdict and raise GovernanceBlockedError if blocked.

    Args:
        data: Parsed JSON response from governance API
        identifier: Resource identifier for error context (URL or file path)
    """
    from .types import GovernanceBlockedError

    verdict_str = (data.get("verdict") or data.get("action", "continue")).lower().replace("-", "_")
    if verdict_str in ("stop", "block", "halt"):
        raise GovernanceBlockedError(
            verdict_str, data.get("reason", "Blocked by governance"), identifier
        )
    if verdict_str in ("require_approval", "request_approval"):
        raise GovernanceBlockedError(
            verdict_str, data.get("reason", "Approval required - blocked at hook level"), identifier
        )


def _send_and_handle(response: Any, identifier: str) -> None:
    """Handle governance API response (shared between sync/async).

    Args:
        response: httpx Response object
        identifier: Resource identifier for error context
    """
    from .types import GovernanceBlockedError

    if response.status_code == 200:
        _handle_verdict(response.json(), identifier)
    elif response.status_code >= 400:
        logger.warning(f"Hook governance API error: HTTP {response.status_code}")
        if _on_api_error == FAIL_CLOSED:
            raise GovernanceBlockedError(
                "halt", f"Governance API error: HTTP {response.status_code}", identifier
            )


def evaluate_sync(
    span: Any,
    hook_trigger: Dict[str, Any],
    identifier: str,
    span_data: Optional[Dict[str, Any]] = None,
) -> None:
    """Synchronous governance evaluation. Blocks until verdict is received.

    Raises GovernanceBlockedError if verdict is BLOCK, HALT, or REQUIRE_APPROVAL.

    Args:
        span: OTel span for the current operation
        hook_trigger: Dict describing what triggered governance
        identifier: Resource identifier (URL or file path) for error context
        span_data: Optional span data dict to store in the buffer
    """
    if not is_configured():
        return

    payload = _build_payload(span, hook_trigger, span_data)
    if payload is None:
        return

    from .types import GovernanceBlockedError

    try:
        with httpx.Client(timeout=_api_timeout) as client:
            response = client.post(
                f"{_api_url}/api/v1/governance/evaluate",
                json=payload,
                headers=_auth_headers(),
            )
        _send_and_handle(response, identifier)

    except GovernanceBlockedError:
        raise
    except Exception as e:
        logger.warning(f"Hook governance evaluation failed: {e}")
        if _on_api_error == FAIL_CLOSED:
            raise GovernanceBlockedError("halt", f"Governance evaluation error: {e}", identifier)


async def evaluate_async(
    span: Any,
    hook_trigger: Dict[str, Any],
    identifier: str,
    span_data: Optional[Dict[str, Any]] = None,
) -> None:
    """Async governance evaluation. Awaits until verdict is received.

    Raises GovernanceBlockedError if verdict is BLOCK, HALT, or REQUIRE_APPROVAL.

    Args:
        span: OTel span for the current operation
        hook_trigger: Dict describing what triggered governance
        identifier: Resource identifier (URL or file path) for error context
        span_data: Optional span data dict to store in the buffer
    """
    if not is_configured():
        return

    payload = _build_payload(span, hook_trigger, span_data)
    if payload is None:
        return

    from .types import GovernanceBlockedError

    try:
        async with httpx.AsyncClient(timeout=_api_timeout) as client:
            response = await client.post(
                f"{_api_url}/api/v1/governance/evaluate",
                json=payload,
                headers=_auth_headers(),
            )
        _send_and_handle(response, identifier)

    except GovernanceBlockedError:
        raise
    except Exception as e:
        logger.warning(f"Hook governance evaluation failed: {e}")
        if _on_api_error == FAIL_CLOSED:
            raise GovernanceBlockedError("halt", f"Governance evaluation error: {e}", identifier)
