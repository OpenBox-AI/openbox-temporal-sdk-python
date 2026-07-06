"""Function tracing decorator — a compatibility wrapper over the base SDK.

``@traced`` delegates to ``openbox_core.instrumentation.function.governed``:
when a worker has installed the base runtime, a decorated function emits
started/completed FUNCTION_CALL hook events and is subject to governance;
without an installed runtime it is a transparent passthrough. ``create_span``
remains a plain OpenTelemetry span helper with no governance.

Governance payload building, evaluation, and enforcement live entirely in the
base SDK — this module holds no hook logic of its own.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, TypeVar, Union

from opentelemetry import trace

from openbox_core.instrumentation.function import governed

__all__ = ["traced", "create_span"]

F = TypeVar("F", bound=Callable[..., Any])


def traced(
    _func: Optional[F] = None,
    *,
    name: Optional[str] = None,
    capture_args: bool = True,
    capture_result: bool = True,
    capture_exception: bool = True,  # accepted for compat; base always records errors
    max_arg_length: int = 2000,  # accepted for compat; base truncation is centralized
) -> Union[F, Callable[[F], F]]:
    """Trace + govern a function via the base SDK's ``governed`` decorator.

    ``capture_exception`` / ``max_arg_length`` are accepted for backward
    compatibility but no longer tuned here — the base SDK records the
    completed/error stages and applies its own privacy truncation.
    """
    return governed(
        _func,
        name=name,
        capture_args=capture_args,
        capture_result=capture_result,
    )


def create_span(name: str, attributes: Optional[dict] = None) -> Any:
    """Plain OpenTelemetry span context manager (no governance).

    Usage::

        with create_span("my-operation", {"input": data}) as span:
            result = do_something()
            span.set_attribute("output", result)
    """
    tracer = trace.get_tracer("openbox.tracing")
    return tracer.start_as_current_span(name, attributes=attributes or {})
