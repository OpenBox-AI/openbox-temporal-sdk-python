# openbox/tracing.py
"""
OpenBox Tracing Decorators for capturing internal function calls.

Use the @traced decorator to capture function calls as OpenTelemetry spans.
These spans will be automatically captured by WorkflowSpanProcessor and
included in governance events.

Usage:
    from openbox.tracing import traced

    @traced
    def my_function(arg1, arg2):
        return do_something(arg1, arg2)

    @traced(name="custom-span-name", capture_args=True, capture_result=True)
    async def my_async_function(data):
        return await process(data)
"""

import json
import logging
from functools import wraps
from typing import Any, Callable, Optional, TypeVar, Union

from opentelemetry import trace

from . import hook_governance as _hook_gov

logger = logging.getLogger(__name__)


def _build_traced_span_data(
    span,
    func_name: str,
    module: str,
    stage: str,
    error: Optional[str] = None,
    duration_ms: Optional[float] = None,
    args: Any = None,
    result: Any = None,
) -> dict:
    """Build span data dict for a @traced function call.

    attributes: OTel-original only. All custom data at root level.
    """
    import time as _time

    span_id_hex, trace_id_hex, parent_span_id = _hook_gov.extract_span_context(span)
    raw_attrs = getattr(span, "attributes", None)
    attrs = dict(raw_attrs) if raw_attrs and isinstance(raw_attrs, dict) else {}

    now_ns = _time.time_ns()
    duration_ns = int(duration_ms * 1_000_000) if duration_ms else None
    end_time = now_ns if stage == "completed" else None
    start_time = (now_ns - duration_ns) if duration_ns else now_ns

    return {
        "span_id": span_id_hex,
        "trace_id": trace_id_hex,
        "parent_span_id": parent_span_id,
        "name": getattr(span, "name", None) or func_name,
        "kind": "INTERNAL",
        "stage": stage,
        "start_time": start_time,
        "end_time": end_time,
        "duration_ns": duration_ns,
        "attributes": attrs,
        "status": {"code": "ERROR" if error else "UNSET", "description": error},
        "events": [],
        # Hook type identification
        "hook_type": "function_call",
        # Function-specific root fields
        "function": func_name,
        "module": module,
        "args": args,
        "result": result,
        "error": error,
    }


# Get tracer for internal function tracing
_tracer: Optional[trace.Tracer] = None


def _get_tracer() -> trace.Tracer:
    """Lazy tracer initialization."""
    global _tracer
    if _tracer is None:
        _tracer = trace.get_tracer("openbox.traced")
    return _tracer


def _safe_serialize(value: Any, max_length: int = 2000) -> str:
    """Safely serialize a value to string for span attributes."""
    try:
        if value is None:
            return "null"
        if isinstance(value, (str, int, float, bool)):
            result = str(value)
        elif isinstance(value, (list, dict)):
            result = json.dumps(value, default=str)
        else:
            result = str(value)

        # Truncate if too long
        if len(result) > max_length:
            return result[:max_length] + "...[truncated]"
        return result
    except Exception:
        return "<unserializable>"


F = TypeVar("F", bound=Callable[..., Any])


def _setup_span(span, func, capture_args, args, kwargs, max_arg_length):
    """Set span metadata and capture arguments."""
    span.set_attribute("code.function", func.__name__)
    span.set_attribute("code.namespace", func.__module__)
    if capture_args:
        _set_args_attributes(span, args, kwargs, max_arg_length)


def _build_started_data(span, func, capture_args, args, kwargs, max_arg_length):
    """Build started governance span data."""
    args_data = (
        _safe_serialize({"args": args, "kwargs": kwargs}, max_arg_length)
        if capture_args else None
    )
    return _build_traced_span_data(
        span, func.__name__, func.__module__, "started", args=args_data,
    )


def _build_completed_data(span, func, duration_ms, capture_result, result, max_arg_length):
    """Build completed governance span data."""
    result_data = _safe_serialize(result, max_arg_length) if capture_result else None
    return _build_traced_span_data(
        span, func.__name__, func.__module__, "completed",
        duration_ms=duration_ms, result=result_data,
    )


def _build_error_data(span, func, error):
    """Build error governance span data."""
    return _build_traced_span_data(
        span, func.__name__, func.__module__, "completed", error=str(error),
    )


def _capture_error_attrs(span, e, capture_exception):
    """Set error attributes on span."""
    if capture_exception:
        span.set_attribute("error", True)
        span.set_attribute("error.type", type(e).__name__)
        span.set_attribute("error.message", str(e))


def traced(
    _func: Optional[F] = None,
    *,
    name: Optional[str] = None,
    capture_args: bool = True,
    capture_result: bool = True,
    capture_exception: bool = True,
    max_arg_length: int = 2000,
) -> Union[F, Callable[[F], F]]:
    """Decorator to trace function calls as OpenTelemetry spans.

    Spans are captured by WorkflowSpanProcessor and included in governance events.
    """

    def decorator(func: F) -> F:
        span_name = name or func.__name__

        if _is_async_function(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                import time as _time
                tracer = _get_tracer()
                with tracer.start_as_current_span(span_name) as span:
                    _setup_span(span, func, capture_args, args, kwargs, max_arg_length)

                    if _hook_gov.is_configured():
                        sd = _build_started_data(span, func, capture_args, args, kwargs, max_arg_length)
                        await _hook_gov.evaluate_async(span, identifier=func.__name__, span_data=sd)

                    _start = _time.perf_counter()
                    try:
                        result = await func(*args, **kwargs)
                        _dur_ms = (_time.perf_counter() - _start) * 1000
                        if capture_result:
                            span.set_attribute("function.result", _safe_serialize(result, max_arg_length))
                        if _hook_gov.is_configured():
                            sd = _build_completed_data(span, func, _dur_ms, capture_result, result, max_arg_length)
                            await _hook_gov.evaluate_async(span, identifier=func.__name__, span_data=sd)
                        return result
                    except Exception as e:
                        _capture_error_attrs(span, e, capture_exception)
                        if _hook_gov.is_configured():
                            sd = _build_error_data(span, func, e)
                            await _hook_gov.evaluate_async(span, identifier=func.__name__, span_data=sd)
                        raise

            return async_wrapper  # type: ignore

        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                import time as _time
                tracer = _get_tracer()
                with tracer.start_as_current_span(span_name) as span:
                    _setup_span(span, func, capture_args, args, kwargs, max_arg_length)

                    if _hook_gov.is_configured():
                        sd = _build_started_data(span, func, capture_args, args, kwargs, max_arg_length)
                        _hook_gov.evaluate_sync(span, identifier=func.__name__, span_data=sd)

                    _start = _time.perf_counter()
                    try:
                        result = func(*args, **kwargs)
                        _dur_ms = (_time.perf_counter() - _start) * 1000
                        if capture_result:
                            span.set_attribute("function.result", _safe_serialize(result, max_arg_length))
                        if _hook_gov.is_configured():
                            sd = _build_completed_data(span, func, _dur_ms, capture_result, result, max_arg_length)
                            _hook_gov.evaluate_sync(span, identifier=func.__name__, span_data=sd)
                        return result
                    except Exception as e:
                        _capture_error_attrs(span, e, capture_exception)
                        if _hook_gov.is_configured():
                            sd = _build_error_data(span, func, e)
                            _hook_gov.evaluate_sync(span, identifier=func.__name__, span_data=sd)
                        raise

            return sync_wrapper  # type: ignore

    if _func is not None:
        return decorator(_func)
    return decorator


def _is_async_function(func: Callable) -> bool:
    """Check if function is async."""
    import asyncio

    return asyncio.iscoroutinefunction(func)


def _set_args_attributes(
    span: trace.Span, args: tuple, kwargs: dict, max_length: int
) -> None:
    """Set function arguments as span attributes."""
    if args:
        for i, arg in enumerate(args):
            span.set_attribute(f"function.arg.{i}", _safe_serialize(arg, max_length))

    if kwargs:
        for key, value in kwargs.items():
            span.set_attribute(
                f"function.kwarg.{key}", _safe_serialize(value, max_length)
            )


# Convenience function to create a span context manager
def create_span(
    name: str,
    attributes: Optional[dict] = None,
) -> trace.Span:
    """
    Create a span context manager for manual tracing.

    Usage:
        from openbox.tracing import create_span

        with create_span("my-operation", {"input": data}) as span:
            result = do_something()
            span.set_attribute("output", result)

    Args:
        name: Span name
        attributes: Initial attributes to set on the span

    Returns:
        Span context manager
    """
    tracer = _get_tracer()
    span = tracer.start_span(name)

    if attributes:
        for key, value in attributes.items():
            span.set_attribute(key, _safe_serialize(value))

    return span
