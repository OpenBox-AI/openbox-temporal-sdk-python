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

logger = logging.getLogger(__name__)

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


def traced(
    _func: Optional[F] = None,
    *,
    name: Optional[str] = None,
    capture_args: bool = True,
    capture_result: bool = True,
    capture_exception: bool = True,
    max_arg_length: int = 2000,
) -> Union[F, Callable[[F], F]]:
    """
    Decorator to trace function calls as OpenTelemetry spans.

    The spans will be captured by WorkflowSpanProcessor and included
    in ActivityCompleted governance events.

    Args:
        name: Custom span name. Defaults to function name.
        capture_args: Capture function arguments as span attributes.
        capture_result: Capture return value as span attribute.
        capture_exception: Capture exception details on error.
        max_arg_length: Maximum length for serialized arguments.

    Examples:
        # Basic usage
        @traced
        def process_data(input_data):
            return transform(input_data)

        # With options
        @traced(name="data-processing", capture_result=False)
        def process_sensitive_data(data):
            return handle(data)

        # Async functions
        @traced
        async def fetch_data(url):
            return await http_get(url)
    """

    def decorator(func: F) -> F:
        span_name = name or func.__name__
        is_async = _is_async_function(func)

        if is_async:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                tracer = _get_tracer()
                with tracer.start_as_current_span(span_name) as span:
                    # Set function metadata
                    span.set_attribute("code.function", func.__name__)
                    span.set_attribute("code.namespace", func.__module__)

                    # Capture arguments
                    if capture_args:
                        _set_args_attributes(span, args, kwargs, max_arg_length)

                    try:
                        result = await func(*args, **kwargs)

                        # Capture result
                        if capture_result:
                            span.set_attribute(
                                "function.result", _safe_serialize(result, max_arg_length)
                            )

                        return result

                    except Exception as e:
                        if capture_exception:
                            span.set_attribute("error", True)
                            span.set_attribute("error.type", type(e).__name__)
                            span.set_attribute("error.message", str(e))
                        raise

            return async_wrapper  # type: ignore

        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                tracer = _get_tracer()
                with tracer.start_as_current_span(span_name) as span:
                    # Set function metadata
                    span.set_attribute("code.function", func.__name__)
                    span.set_attribute("code.namespace", func.__module__)

                    # Capture arguments
                    if capture_args:
                        _set_args_attributes(span, args, kwargs, max_arg_length)

                    try:
                        result = func(*args, **kwargs)

                        # Capture result
                        if capture_result:
                            span.set_attribute(
                                "function.result", _safe_serialize(result, max_arg_length)
                            )

                        return result

                    except Exception as e:
                        if capture_exception:
                            span.set_attribute("error", True)
                            span.set_attribute("error.type", type(e).__name__)
                            span.set_attribute("error.message", str(e))
                        raise

            return sync_wrapper  # type: ignore

    # Handle both @traced and @traced() syntax
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
            span.set_attribute(f"function.kwarg.{key}", _safe_serialize(value, max_length))


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
