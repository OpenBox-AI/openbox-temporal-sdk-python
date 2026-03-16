# openbox/http_governance_hooks.py
"""HTTP governance hooks for requests, httpx, urllib3, and urllib.

Captures request/response bodies and sends hook-level governance
evaluations (started/completed) for every HTTP operation during
activity execution.

Each library's OTel instrumentor calls these hooks. The hooks:
1. Extract request/response metadata
2. Build a span_data dict via _build_http_span_data()
3. Call hook_governance.evaluate_sync/async() for governance evaluation
"""

from __future__ import annotations

import contextvars
import logging
from typing import Dict, Optional

from . import hook_governance as _hook_gov
# Late import: otel_setup imports us, so we get a partially-loaded module ref.
# That's fine — we only access _otel._span_processor and _otel._ignored_url_prefixes
# at function call time, when both modules are fully loaded.
from . import otel_setup as _otel

logger = logging.getLogger(__name__)

# ContextVar to pass HTTP child span from OTel request hooks to _patched_send.
# Request hooks receive the correct HTTP span; we store it here so _patched_send
# can use it after _original_send() returns (when the HTTP span has ended and
# trace.get_current_span() would return the parent activity span).
_httpx_http_span: contextvars.ContextVar = contextvars.ContextVar(
    '_httpx_http_span', default=None
)

# Timing for HTTP hooks: span_id → perf_counter start time
# Used by request_hook (started) to pass timing to response_hook (completed)
_http_hook_timings: Dict[int, float] = {}
_HTTP_HOOK_TIMINGS_MAX = 1000

# Text content types that are safe to capture as body
_TEXT_CONTENT_TYPES = (
    "text/",
    "application/json",
    "application/xml",
    "application/javascript",
    "application/x-www-form-urlencoded",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Shared HTTP utilities
# ═══════════════════════════════════════════════════════════════════════════════


def _should_ignore_url(url: str) -> bool:
    """Check if URL should be ignored (e.g., OpenBox Core API)."""
    if not url:
        return False
    for prefix in _otel._ignored_url_prefixes:
        if url.startswith(prefix):
            return True
    return False


def _is_text_content_type(content_type: Optional[str]) -> bool:
    """Check if content type indicates text content (safe to decode)."""
    if not content_type:
        return True  # Assume text if no content-type
    content_type = content_type.lower().split(";")[0].strip()
    return any(content_type.startswith(t) for t in _TEXT_CONTENT_TYPES)


def _build_http_span_data(
    span,
    http_method: str,
    http_url: str,
    stage: str,
    request_body: Optional[str] = None,
    request_headers: Optional[dict] = None,
    response_body: Optional[str] = None,
    response_headers: Optional[dict] = None,
    http_status_code: Optional[int] = None,
    duration_ms: Optional[float] = None,
) -> dict:
    """Build span data dict for an HTTP request (used by governance hooks).

    attributes: OTel-original only. All custom data at root level.
    """
    import time as _time

    span_id_hex, trace_id_hex, parent_span_id = _hook_gov.extract_span_context(span)
    attrs = dict(span.attributes) if hasattr(span, 'attributes') and span.attributes else {}

    now_ns = _time.time_ns()
    duration_ns = int(duration_ms * 1_000_000) if duration_ms else None
    end_time = now_ns if stage == "completed" else None
    start_time = (now_ns - duration_ns) if duration_ns else now_ns

    error = None
    if http_status_code is not None and http_status_code >= 400:
        error = f"HTTP {http_status_code}"

    return {
        "span_id": span_id_hex,
        "trace_id": trace_id_hex,
        "parent_span_id": parent_span_id,
        "name": span.name if hasattr(span, 'name') and span.name else f"HTTP {http_method}",
        "kind": "CLIENT",
        "stage": stage,
        "start_time": start_time,
        "end_time": end_time,
        "duration_ns": duration_ns,
        "attributes": attrs,
        "status": {"code": "ERROR" if error else "UNSET", "description": error},
        "events": [],
        # Hook type identification
        "hook_type": "http_request",
        # HTTP-specific root fields
        "http_method": http_method,
        "http_url": http_url,
        "request_body": request_body,
        "request_headers": request_headers,
        "response_body": response_body,
        "response_headers": response_headers,
        "http_status_code": http_status_code,
        "error": error,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# requests hooks
# ═══════════════════════════════════════════════════════════════════════════════


def _requests_request_hook(span, request) -> None:
    """Hook called before requests library sends a request.

    Args:
        span: OTel span
        request: requests.PreparedRequest
    """
    if _otel._span_processor is None:
        return

    body = None
    try:
        if request.body:
            body = request.body
            if isinstance(body, bytes):
                body = body.decode("utf-8", errors="ignore")
    except Exception:
        pass

    # Hook-level governance evaluation
    if _hook_gov.is_configured():
        import time as _time
        url = str(request.url) if hasattr(request, 'url') else None
        if url and not _should_ignore_url(url):
            # Record start time for duration calculation in response hook
            if hasattr(span, 'context') and hasattr(span.context, 'span_id'):
                if len(_http_hook_timings) >= _HTTP_HOOK_TIMINGS_MAX:
                    _http_hook_timings.clear()
                _http_hook_timings[span.context.span_id] = _time.perf_counter()
            headers = dict(request.headers) if hasattr(request, 'headers') and request.headers else None
            method = request.method or "UNKNOWN"
            span_data = _build_http_span_data(span, method, url, "started", request_body=body, request_headers=headers)
            _hook_gov.evaluate_sync(
                span,
                identifier=url,
                span_data=span_data,
            )


def _requests_response_hook(span, request, response) -> None:
    """Hook called after requests library receives a response.

    Args:
        span: OTel span
        request: requests.PreparedRequest
        response: requests.Response
    """
    if _otel._span_processor is None:
        return

    resp_body = None
    resp_headers = None
    try:
        resp_headers = dict(response.headers) if hasattr(response, 'headers') and response.headers else None
        content_type = response.headers.get("content-type", "")
        if _is_text_content_type(content_type):
            resp_body = response.text
    except Exception:
        pass

    # Hook-level governance evaluation (response stage)
    if _hook_gov.is_configured():
        import time as _time
        url = str(request.url) if hasattr(request, 'url') else None
        if url and not _should_ignore_url(url):
            # Compute duration from started hook timing
            _dur_ms = None
            if hasattr(span, 'context') and hasattr(span.context, 'span_id'):
                _start = _http_hook_timings.pop(span.context.span_id, None)
                if _start:
                    _dur_ms = (_time.perf_counter() - _start) * 1000
            req_headers = dict(request.headers) if hasattr(request, 'headers') and request.headers else None
            req_body = None
            try:
                if request.body:
                    req_body = request.body
                    if isinstance(req_body, bytes):
                        req_body = req_body.decode("utf-8", errors="ignore")
            except Exception:
                pass
            method = request.method or "UNKNOWN"
            span_data = _build_http_span_data(
                span, method, url, "completed",
                request_body=req_body, request_headers=req_headers,
                response_body=resp_body, response_headers=resp_headers,
                http_status_code=getattr(response, 'status_code', None),
                duration_ms=_dur_ms,
            )
            _hook_gov.evaluate_sync(
                span,
                identifier=url,
                span_data=span_data,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# httpx hooks
# ═══════════════════════════════════════════════════════════════════════════════


def _httpx_request_hook(span, request) -> None:
    """Hook called before httpx sends a request.

    Args:
        span: OTel span
        request: RequestInfo namedtuple with (method, url, headers, stream, extensions)
    """
    if _otel._span_processor is None:
        return

    # Check if URL should be ignored
    url = str(request.url) if hasattr(request, 'url') else None
    if url and _should_ignore_url(url):
        return

    body = None
    request_headers = None
    try:
        # Capture request headers from RequestInfo namedtuple
        if hasattr(request, 'headers') and request.headers:
            request_headers = dict(request.headers)

        # Try to get request body - RequestInfo has a 'stream' attribute
        # httpx ByteStream stores body in _stream (not body or _body)
        if hasattr(request, 'stream'):
            stream = request.stream
            if hasattr(stream, '_stream') and isinstance(stream._stream, bytes):
                body = stream._stream
            elif hasattr(stream, 'body'):
                body = stream.body
            elif hasattr(stream, '_body'):
                body = stream._body
            elif isinstance(stream, bytes):
                body = stream

        # Fallback: Direct content attribute (for httpx.Request objects)
        if not body and hasattr(request, '_content') and request._content:
            body = request._content

        if not body and hasattr(request, 'content'):
            try:
                content = request.content
                if content:
                    body = content
            except Exception:
                pass

        if body:
            if isinstance(body, bytes):
                body = body.decode("utf-8", errors="ignore")
            elif not isinstance(body, str):
                body = str(body)

    except Exception:
        pass  # Best effort

    # Store HTTP child span so _patched_send can use it for governance
    _httpx_http_span.set(span)

    # Hook-level governance evaluation
    if _hook_gov.is_configured() and url:
        method = str(request.method) if hasattr(request, 'method') else "UNKNOWN"
        req_body = body if isinstance(body, str) else None
        span_data = _build_http_span_data(span, method, url, "started", request_body=req_body, request_headers=request_headers)
        _hook_gov.evaluate_sync(
            span,
            identifier=url,
            span_data=span_data,
        )


def _httpx_response_hook(span, request, response) -> None:
    """Hook called after httpx receives a response.

    NOTE: At this point the response may not have been fully read yet.
    We try to read it here, but body capture may need to happen via
    the patched send method instead.

    Args:
        span: OTel span
        request: httpx.Request
        response: httpx.Response
    """
    if _otel._span_processor is None:
        return

    # Check if URL should be ignored
    url = str(request.url) if hasattr(request, 'url') else None
    if url and _should_ignore_url(url):
        return

    resp_body = None
    resp_headers = None
    try:
        # Capture response headers first (always available even for streaming)
        if hasattr(response, 'headers') and response.headers:
            resp_headers = dict(response.headers)

        content_type = response.headers.get("content-type", "")
        if _is_text_content_type(content_type):
            body = None

            # Check if response has already been read (has _content)
            if hasattr(response, '_content') and response._content:
                body = response._content
            # Try .content property
            elif hasattr(response, 'content'):
                try:
                    body = response.content
                except Exception:
                    pass

            if body:
                if isinstance(body, bytes):
                    body = body.decode("utf-8", errors="ignore")
                resp_body = body
    except Exception:
        pass  # Best effort

    # NOTE: "completed" governance evaluation is handled in _patched_send
    # where request/response bodies are guaranteed to be available.


async def _httpx_async_request_hook(span, request) -> None:
    """Async version of request hook with async governance evaluation."""
    if _otel._span_processor is None:
        return

    # Check if URL should be ignored
    url = str(request.url) if hasattr(request, 'url') else None
    if url and _should_ignore_url(url):
        return

    body = None
    request_headers = None
    try:
        # Capture request headers
        if hasattr(request, 'headers') and request.headers:
            request_headers = dict(request.headers)

        # Try to get request body
        if hasattr(request, 'stream'):
            stream = request.stream
            if hasattr(stream, 'body'):
                body = stream.body
            elif hasattr(stream, '_body'):
                body = stream._body
            elif isinstance(stream, bytes):
                body = stream

        if not body and hasattr(request, '_content') and request._content:
            body = request._content

        if not body and hasattr(request, 'content'):
            try:
                content = request.content
                if content:
                    body = content
            except Exception:
                pass

        if body:
            if isinstance(body, bytes):
                body = body.decode("utf-8", errors="ignore")
            elif not isinstance(body, str):
                body = str(body)

    except Exception:
        pass  # Best effort

    # Store HTTP child span so _patched_async_send can use it for governance
    _httpx_http_span.set(span)

    # Async hook-level governance evaluation
    if _hook_gov.is_configured() and url:
        method = str(request.method) if hasattr(request, 'method') else "UNKNOWN"
        req_body = body if isinstance(body, str) else None
        span_data = _build_http_span_data(span, method, url, "started", request_body=req_body, request_headers=request_headers)
        await _hook_gov.evaluate_async(
            span,
            identifier=url,
            span_data=span_data,
        )


async def _httpx_async_response_hook(span, request, response) -> None:
    """Async version of response hook."""
    if _otel._span_processor is None:
        return

    # Check if URL should be ignored
    url = str(request.url) if hasattr(request, 'url') else None
    if url and _should_ignore_url(url):
        return

    resp_body = None
    resp_headers = None
    try:
        # Capture response headers
        if hasattr(response, 'headers') and response.headers:
            resp_headers = dict(response.headers)

        content_type = response.headers.get("content-type", "")
        if _is_text_content_type(content_type):
            body = None

            # Check if response has already been read
            if hasattr(response, '_content') and response._content:
                body = response._content
                if isinstance(body, bytes):
                    body = body.decode("utf-8", errors="ignore")
            # For async, try to read the response - THIS WILL CONSUME IT
            # but httpx caches it in _content after first read
            elif hasattr(response, 'aread'):
                try:
                    await response.aread()
                    if hasattr(response, '_content') and response._content:
                        body = response._content
                        if isinstance(body, bytes):
                            body = body.decode("utf-8", errors="ignore")
                except Exception:
                    pass

            if body:
                resp_body = body

        # Also try to get request body from the stream
        request_body = None
        if hasattr(request, 'stream'):
            stream = request.stream
            if hasattr(stream, 'body'):
                request_body = stream.body
            elif hasattr(stream, '_body'):
                request_body = stream._body

        if request_body:
            if isinstance(request_body, bytes):
                request_body = request_body.decode("utf-8", errors="ignore")

    except Exception:
        pass  # Best effort

    # NOTE: "completed" governance evaluation is handled in _patched_async_send
    # where request/response bodies are guaranteed to be available.


# ═══════════════════════════════════════════════════════════════════════════════
# httpx body capture (patches Client.send)
# ═══════════════════════════════════════════════════════════════════════════════


def _capture_httpx_request_data(request) -> tuple:
    """Extract request body and headers from an httpx request.

    Returns:
        (request_body, request_headers) tuple. Either may be None.
    """
    request_body = None
    request_headers = None
    try:
        if hasattr(request, '_content') and request._content:
            request_body = request._content
            if isinstance(request_body, bytes):
                request_body = request_body.decode("utf-8", errors="ignore")
        elif hasattr(request, 'content') and request.content:
            request_body = request.content
            if isinstance(request_body, bytes):
                request_body = request_body.decode("utf-8", errors="ignore")
        if hasattr(request, 'headers') and request.headers:
            request_headers = dict(request.headers)
    except Exception as e:
        logger.debug(f"Failed to capture request body/headers: {e}")
    return request_body, request_headers


def _capture_httpx_response_data(response) -> tuple:
    """Extract response body and headers from an httpx response.

    Returns:
        (response_body, response_headers) tuple. Either may be None.
    """
    response_body = None
    response_headers = None
    content_type = response.headers.get("content-type", "")
    try:
        if hasattr(response, 'headers') and response.headers:
            response_headers = dict(response.headers)
        if _is_text_content_type(content_type):
            response_body = response.text
    except (UnicodeDecodeError, Exception) as e:
        logger.debug(f"Failed to capture response body: {e}")
    return response_body, response_headers


def _get_httpx_http_span():
    """Retrieve and reset the HTTP span stored by request hooks.

    Falls back to the current OTel span if no stored span is found.
    """
    http_span = _httpx_http_span.get(None)
    _httpx_http_span.set(None)
    if http_span is None:
        from opentelemetry import trace
        http_span = trace.get_current_span()
    return http_span


def _prepare_completed_governance(http_span, request, url, request_body, request_headers,
                                  response_body, response_headers, status_code, duration_ms=None):
    """Build 'completed' governance args. Returns tuple or None if not applicable."""
    if not (_hook_gov.is_configured() and url and http_span):
        return None
    method = str(request.method) if hasattr(request, 'method') else "UNKNOWN"
    span_data = _build_http_span_data(
        http_span, method, url, "completed",
        request_body=request_body, request_headers=request_headers,
        response_body=response_body, response_headers=response_headers,
        http_status_code=status_code, duration_ms=duration_ms,
    )
    return http_span, url, span_data


def setup_httpx_body_capture(span_processor: "WorkflowSpanProcessor") -> None:
    """Setup httpx body capture using Client.send patching.

    This is separate from OTel instrumentation because OTel hooks
    receive streams that cannot be safely consumed.
    """
    try:
        import httpx

        _original_send = httpx.Client.send
        _original_async_send = httpx.AsyncClient.send

        def _patched_send(self, request, *args, **kwargs):
            import time as _time
            url = str(request.url) if hasattr(request, 'url') else None
            if url and _should_ignore_url(url):
                return _original_send(self, request, *args, **kwargs)

            request_body, request_headers = _capture_httpx_request_data(request)
            _start = _time.perf_counter()
            response = _original_send(self, request, *args, **kwargs)
            _dur_ms = (_time.perf_counter() - _start) * 1000
            http_span = _get_httpx_http_span()
            response_body, response_headers = _capture_httpx_response_data(response)

            gov_args = _prepare_completed_governance(
                http_span, request, url, request_body, request_headers,
                response_body, response_headers, getattr(response, 'status_code', None),
                duration_ms=_dur_ms,
            )
            if gov_args:
                _hook_gov.evaluate_sync(gov_args[0], identifier=gov_args[1], span_data=gov_args[2])
            return response

        async def _patched_async_send(self, request, *args, **kwargs):
            import time as _time
            url = str(request.url) if hasattr(request, 'url') else None
            if url and _should_ignore_url(url):
                return await _original_async_send(self, request, *args, **kwargs)

            request_body, request_headers = _capture_httpx_request_data(request)
            _start = _time.perf_counter()
            response = await _original_async_send(self, request, *args, **kwargs)
            _dur_ms = (_time.perf_counter() - _start) * 1000
            http_span = _get_httpx_http_span()
            response_body, response_headers = _capture_httpx_response_data(response)

            gov_args = _prepare_completed_governance(
                http_span, request, url, request_body, request_headers,
                response_body, response_headers, getattr(response, 'status_code', None),
                duration_ms=_dur_ms,
            )
            if gov_args:
                await _hook_gov.evaluate_async(gov_args[0], identifier=gov_args[1], span_data=gov_args[2])
            return response

        httpx.Client.send = _patched_send
        httpx.AsyncClient.send = _patched_async_send
        logger.info("Patched httpx for body capture")

    except ImportError:
        logger.debug("httpx not available for body capture")


# ═══════════════════════════════════════════════════════════════════════════════
# urllib3 hooks
# ═══════════════════════════════════════════════════════════════════════════════


def _urllib3_request_hook(span, pool, request_info) -> None:
    """Hook called before urllib3 sends a request.

    Args:
        span: OTel span
        pool: urllib3.HTTPConnectionPool
        request_info: RequestInfo namedtuple
    """
    if _otel._span_processor is None:
        return

    body = None
    try:
        if hasattr(request_info, "body") and request_info.body:
            body = request_info.body
            if isinstance(body, bytes):
                body = body.decode("utf-8", errors="ignore")
    except Exception:
        pass

    # Hook-level governance evaluation
    if _hook_gov.is_configured():
        # Reconstruct URL from pool and request_info
        scheme = getattr(pool, 'scheme', 'http')
        host = getattr(pool, 'host', 'unknown')
        port = getattr(pool, 'port', None)
        url_path = getattr(request_info, 'url', getattr(request_info, 'request_url', '/'))
        if port and port not in (80, 443):
            url = f"{scheme}://{host}:{port}{url_path}"
        else:
            url = f"{scheme}://{host}{url_path}"

        if not _should_ignore_url(url):
            import time as _time
            if hasattr(span, 'context') and hasattr(span.context, 'span_id'):
                if len(_http_hook_timings) >= _HTTP_HOOK_TIMINGS_MAX:
                    _http_hook_timings.clear()
                _http_hook_timings[span.context.span_id] = _time.perf_counter()
            method = getattr(request_info, 'method', 'UNKNOWN')
            headers = dict(request_info.headers) if hasattr(request_info, 'headers') and request_info.headers else None
            req_body = body if isinstance(body, str) else None
            span_data = _build_http_span_data(span, method, url, "started", request_body=req_body, request_headers=headers)
            _hook_gov.evaluate_sync(
                span,
                identifier=url,
                span_data=span_data,
            )


def _urllib3_response_hook(span, pool, response) -> None:
    """Hook called after urllib3 receives a response.

    Args:
        span: OTel span
        pool: urllib3.HTTPConnectionPool
        response: urllib3.HTTPResponse
    """
    if _otel._span_processor is None:
        return

    resp_body = None
    resp_headers = None
    try:
        resp_headers = dict(response.headers) if hasattr(response, 'headers') and response.headers else None
        content_type = response.headers.get("content-type", "")
        if _is_text_content_type(content_type):
            body = response.data
            if isinstance(body, bytes):
                body = body.decode("utf-8", errors="ignore")
            if body:
                resp_body = body
    except Exception:
        pass

    # Hook-level governance evaluation (response stage)
    if _hook_gov.is_configured():
        # Reconstruct URL from pool
        scheme = getattr(pool, 'scheme', 'http')
        host = getattr(pool, 'host', 'unknown')
        port = getattr(pool, 'port', None)
        if port and port not in (80, 443):
            url = f"{scheme}://{host}:{port}/"
        else:
            url = f"{scheme}://{host}/"

        if not _should_ignore_url(url):
            import time as _time
            _dur_ms = None
            if hasattr(span, 'context') and hasattr(span.context, 'span_id'):
                _start = _http_hook_timings.pop(span.context.span_id, None)
                if _start:
                    _dur_ms = (_time.perf_counter() - _start) * 1000
            status_code = getattr(response, 'status', None)
            span_data = _build_http_span_data(
                span, "UNKNOWN", url, "completed",
                response_body=resp_body, response_headers=resp_headers,
                http_status_code=status_code, duration_ms=_dur_ms,
            )
            _hook_gov.evaluate_sync(
                span,
                identifier=url,
                span_data=span_data,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# urllib hooks (standard library)
# NOTE: Response body capture is NOT supported - read() consumes the socket stream
# ═══════════════════════════════════════════════════════════════════════════════


def _urllib_request_hook(span, request) -> None:
    """Hook called before urllib sends a request."""
    if _otel._span_processor is None:
        return

    try:
        if request.data:
            body = request.data
            if isinstance(body, bytes):
                body = body.decode("utf-8", errors="ignore")
    except Exception:
        pass
