# openbox/hook_governance.py
"""Hook-level governance evaluation for all operation types.

Sends per-operation governance evaluations to OpenBox Core during activity
execution. Used by OTel hooks to evaluate each operation (HTTP, file I/O,
database, traced functions) at two stages: 'started' and 'completed'.

Architecture:
    1. Hook modules detect an operation and build a span_data dict
    2. Hook calls evaluate_sync() or evaluate_async()
    3. This module: looks up activity context, assembles payload, sends to API
    4. If verdict is BLOCK/HALT → raises GovernanceBlockedError
"""

from __future__ import annotations

import json
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
_cached_auth_headers: Optional[dict] = None

# Persistent HTTP clients (lazy-init, thread-safe for requests)
_sync_client: Optional[httpx.Client] = None
_async_client: Optional[httpx.AsyncClient] = None


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
    global _api_url, _api_key, _api_timeout, _on_api_error, _span_processor, _sync_client, _async_client, _cached_auth_headers
    _api_url = api_url.rstrip("/")
    _api_key = api_key
    _api_timeout = api_timeout
    _on_api_error = on_api_error
    _span_processor = span_processor
    # Cache auth headers (immutable after configure)
    _cached_auth_headers = build_auth_headers(api_key)
    # Reset persistent clients so they pick up new timeout/config
    _sync_client = None
    _async_client = None
    logger.info("Hook-level governance configured")


def _get_sync_client() -> httpx.Client:
    """Get or create persistent sync HTTP client."""
    global _sync_client
    if _sync_client is None or _sync_client.is_closed:
        _sync_client = httpx.Client(timeout=_api_timeout)
    return _sync_client


def _get_async_client() -> httpx.AsyncClient:
    """Get or create persistent async HTTP client."""
    global _async_client
    if _async_client is None or _async_client.is_closed:
        _async_client = httpx.AsyncClient(timeout=_api_timeout)
    return _async_client


def is_configured() -> bool:
    """Check if hook-level governance is active."""
    return bool(_api_url and _span_processor is not None)


def get_span_processor() -> "WorkflowSpanProcessor | None":
    """Return the configured span processor (or None)."""
    return _span_processor


def extract_span_context(span) -> tuple:
    """Extract (span_id_hex, trace_id_hex, parent_span_id_hex) from a span.

    Handles NonRecordingSpan, MagicMock, and missing attributes safely.
    Returns 16-char hex span_id, 32-char hex trace_id, and parent_span_id (or None).
    """
    span_ctx = span.get_span_context() if hasattr(span, 'get_span_context') else getattr(span, 'context', None)
    try:
        span_id = format(span_ctx.span_id, "016x") if span_ctx and isinstance(span_ctx.span_id, int) else "0" * 16
    except (AttributeError, TypeError):
        span_id = "0" * 16
    try:
        trace_id = format(span_ctx.trace_id, "032x") if span_ctx and isinstance(span_ctx.trace_id, int) else "0" * 32
    except (AttributeError, TypeError):
        trace_id = "0" * 32

    parent_span_id = None
    parent = getattr(span, 'parent', None)
    if parent and hasattr(parent, 'span_id') and isinstance(getattr(parent, 'span_id', None), int):
        parent_span_id = format(parent.span_id, "016x")

    return span_id, trace_id, parent_span_id


def _auth_headers() -> dict:
    """Return cached auth headers (built once in configure())."""
    return _cached_auth_headers or build_auth_headers(_api_key)


def build_auth_headers(api_key: str) -> dict:
    """Build standard auth headers for governance API calls.

    Shared by all modules that call OpenBox Core API.
    """
    from . import __version__
    return {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": f"OpenBox-SDK/{__version__}",
        "X-OpenBox-SDK-Version": __version__,
    }


def _build_payload(
    span: Any,
    span_data: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Build governance evaluation payload from activity context + span data.

    Returns None if no activity context found (not inside a governed activity).

    Args:
        span: OTel span for the current operation
        span_data: Span data dict with hook_type, stage, and type-specific fields at root
    """
    if _span_processor is None:
        logger.debug("[GOV] _build_payload: span_processor is None — skipping")
        return None

    # Look up activity context by trace_id
    # Use get_span_context() for compatibility with both _Span and NonRecordingSpan
    span_context = span.get_span_context() if hasattr(span, 'get_span_context') else span.context
    trace_id = span_context.trace_id
    logger.debug(f"[GOV] _build_payload: looking up trace_id={trace_id}")
    activity_context = _span_processor.get_activity_context_by_trace(trace_id)
    if activity_context is None:
        logger.debug(f"[GOV] _build_payload: NO activity context for trace_id={trace_id} — skipping")
        return None
    logger.debug(f"[GOV] _build_payload: found activity context wf={activity_context.get('workflow_id')}")

    workflow_id = activity_context.get("workflow_id")
    activity_id = activity_context.get("activity_id")

    # Tag span_data with activity_id for server-side correlation
    if span_data and activity_id and "activity_id" not in span_data:
        span_data["activity_id"] = activity_id

    # Assemble payload — send only the current span (server processes each individually)
    payload = dict(activity_context)
    payload["spans"] = [span_data] if span_data else []
    payload["span_count"] = 1 if span_data else 0
    payload["hook_trigger"] = True
    from .types import rfc3339_now
    payload["timestamp"] = rfc3339_now()

    # Ensure JSON-serializable (Temporal Payload objects slip through from activity_context)
    try:
        json.dumps(payload)
    except (TypeError, ValueError):
        payload = json.loads(json.dumps(payload, default=str))

    return payload


def _resolve_activity_ids(span) -> Optional[tuple]:
    """Resolve span → (workflow_id, activity_id) via trace_id lookup.

    Returns (workflow_id, activity_id) tuple or None if resolution fails.
    Handles NonRecordingSpan, MagicMock, and missing attributes safely.
    """
    if _span_processor is None:
        return None
    span_context = span.get_span_context() if hasattr(span, 'get_span_context') else getattr(span, 'context', None)
    if not span_context or not isinstance(getattr(span_context, 'trace_id', None), int):
        return None
    activity_ctx = _span_processor.get_activity_context_by_trace(span_context.trace_id)
    if not activity_ctx or not isinstance(activity_ctx, dict):
        return None
    return activity_ctx.get("workflow_id", ""), activity_ctx.get("activity_id", "")


def _check_activity_abort(span) -> Optional[str]:
    """Check if the activity owning this span has been aborted.

    Returns abort reason if aborted, None otherwise.
    """
    # Skip if span_processor lacks abort method (MagicMock/old processors)
    if not hasattr(_span_processor, 'get_activity_abort') or not callable(getattr(_span_processor, 'get_activity_abort', None)):
        return None
    ids = _resolve_activity_ids(span)
    if not ids:
        return None
    result = _span_processor.get_activity_abort(ids[0], ids[1])
    # Ensure result is actually a string (not a MagicMock or other truthy object)
    return result if isinstance(result, str) else None


def _set_activity_abort(span, reason: str) -> None:
    """Set abort flag for the activity owning this span."""
    ids = _resolve_activity_ids(span)
    if not ids:
        return
    _span_processor.set_activity_abort(ids[0], ids[1], reason)


def _handle_verdict(data: Dict[str, Any], identifier: str, span: Any = None) -> None:
    """Check API response verdict and raise GovernanceBlockedError if blocked.

    Args:
        data: Parsed JSON response from governance API
        identifier: Resource identifier for error context (URL or file path)
        span: OTel span (used to set abort flag on require_approval)
    """
    from .types import GovernanceBlockedError

    verdict_str = (data.get("verdict") or data.get("action", "continue")).lower().replace("-", "_")
    if verdict_str in ("stop", "block", "halt"):
        if span:
            reason = data.get("reason", "Blocked by governance")
            ids = _resolve_activity_ids(span)
            if ids:
                _span_processor.set_activity_abort(ids[0], ids[1], reason)
                # For HALT, also set halt-requested flag so activity interceptor calls terminate()
                if verdict_str in ("halt", "stop") and hasattr(_span_processor, 'set_halt_requested'):
                    _span_processor.set_halt_requested(ids[0], ids[1], reason)
        raise GovernanceBlockedError(
            verdict_str, data.get("reason", "Blocked by governance"), identifier
        )
    if verdict_str in ("require_approval", "request_approval"):
        if span:
            _set_activity_abort(span, data.get("reason", "Approval required"))
        raise GovernanceBlockedError(
            verdict_str, data.get("reason", "Approval required - blocked at hook level"), identifier
        )


def _send_and_handle(response: Any, identifier: str, span: Any = None) -> None:
    """Handle governance API response (shared between sync/async).

    Args:
        response: httpx Response object
        identifier: Resource identifier for error context
        span: OTel span (passed to _handle_verdict for abort flag)
    """
    from .types import GovernanceBlockedError

    if response.status_code == 200:
        _handle_verdict(response.json(), identifier, span=span)
    elif response.status_code >= 400:
        logger.warning(f"Hook governance API error: HTTP {response.status_code}")
        if _on_api_error == FAIL_CLOSED:
            raise GovernanceBlockedError(
                "halt", f"Governance API error: HTTP {response.status_code}", identifier
            )


def evaluate_sync(
    span: Any,
    identifier: str,
    span_data: Optional[Dict[str, Any]] = None,
) -> None:
    """Synchronous governance evaluation. Blocks until verdict is received.

    Raises GovernanceBlockedError if verdict is BLOCK, HALT, or REQUIRE_APPROVAL.
    Short-circuits immediately if the activity has been aborted by a prior hook.

    Args:
        span: OTel span for the current operation
        identifier: Resource identifier (URL or file path) for error context
        span_data: Span data dict with hook_type and type-specific fields at root
    """
    if not is_configured():
        return

    from .types import GovernanceBlockedError

    # Short-circuit if activity already aborted by a prior hook verdict
    abort_reason = _check_activity_abort(span)
    if abort_reason:
        raise GovernanceBlockedError("require_approval", abort_reason, identifier)

    payload = _build_payload(span, span_data)
    if payload is None:
        return

    try:
        client = _get_sync_client()
        response = client.post(
            f"{_api_url}/api/v1/governance/evaluate",
            json=payload,
            headers=_auth_headers(),
        )
        _send_and_handle(response, identifier, span=span)

    except GovernanceBlockedError:
        raise
    except Exception as e:
        logger.warning(f"Hook governance evaluation failed: {e}")
        if _on_api_error == FAIL_CLOSED:
            raise GovernanceBlockedError("halt", f"Governance evaluation error: {e}", identifier)


async def evaluate_async(
    span: Any,
    identifier: str,
    span_data: Optional[Dict[str, Any]] = None,
) -> None:
    """Async governance evaluation. Awaits until verdict is received.

    Raises GovernanceBlockedError if verdict is BLOCK, HALT, or REQUIRE_APPROVAL.
    Short-circuits immediately if the activity has been aborted by a prior hook.

    Args:
        span: OTel span for the current operation
        identifier: Resource identifier (URL or file path) for error context
        span_data: Span data dict with hook_type and type-specific fields at root
    """
    if not is_configured():
        return

    from .types import GovernanceBlockedError

    # Short-circuit if activity already aborted by a prior hook verdict
    abort_reason = _check_activity_abort(span)
    if abort_reason:
        raise GovernanceBlockedError("require_approval", abort_reason, identifier)

    payload = _build_payload(span, span_data)
    if payload is None:
        return

    try:
        client = _get_async_client()
        response = await client.post(
            f"{_api_url}/api/v1/governance/evaluate",
            json=payload,
            headers=_auth_headers(),
        )
        _send_and_handle(response, identifier, span=span)

    except GovernanceBlockedError:
        raise
    except Exception as e:
        logger.warning(f"Hook governance evaluation failed: {e}")
        if _on_api_error == FAIL_CLOSED:
            raise GovernanceBlockedError("halt", f"Governance evaluation error: {e}", identifier)
