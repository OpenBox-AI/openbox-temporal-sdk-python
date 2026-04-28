# openbox/workflow_interceptor.py
"""
Temporal workflow interceptor for workflow-boundary governance.

Sends workflow lifecycle events via activity (for determinism).

Events:
- WorkflowStarted
- WorkflowCompleted
- WorkflowFailed
- SignalReceived

IMPORTANT: No logging inside workflow code! Python's logging module uses
linecache -> os.stat which triggers Temporal sandbox restrictions.
"""

import json
from dataclasses import asdict, is_dataclass
from datetime import timedelta
from typing import Any, Optional, Type

from temporalio import workflow
from temporalio.exceptions import ActivityError, ApplicationError
from temporalio.worker import (
    Interceptor,
    WorkflowInboundInterceptor,
    WorkflowInterceptorClassInput,
    ExecuteWorkflowInput,
    HandleSignalInput,
)

from .errors import (
    GOVERNANCE_API_ERROR_TYPE,
    GOVERNANCE_BLOCK_ERROR_TYPE,
    GOVERNANCE_HALT_ERROR_TYPE,
    GOVERNANCE_STOP_ERROR_TYPE,
)
from .types import Verdict


def _application_error_type(exc: BaseException) -> Optional[str]:
    """Walk exception chain and return the ApplicationError.type if present.

    Temporal wraps activity failures as ActivityError(cause=ApplicationError).
    We walk cause/__cause__/__context__ to find the first ApplicationError and
    return its `type` field. Matching on this field is stable across message
    reformatting, locale changes, and nested wrapping.
    """
    seen: set[int] = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ApplicationError):
            return getattr(current, "type", None)
        next_exc = (
            getattr(current, "cause", None)
            or getattr(current, "__cause__", None)
            or getattr(current, "__context__", None)
        )
        current = next_exc
    return None


def _safe_error_type(exc) -> Optional[str]:
    """Extract error type string from an exception, sanitized for JSON."""
    t = getattr(exc, "type", None)
    if isinstance(t, str) and len(t) < 200:
        return t
    return None


def _extract_cause_info(exc) -> Optional[dict]:
    """Extract cause info dict from an exception's cause chain."""
    cause = getattr(exc, "cause", None) or getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
    if not cause:
        return None

    info = {"type": type(cause).__name__, "message": str(cause)[:500]}
    cause_type = _safe_error_type(cause)
    if cause_type:
        info["error_type"] = cause_type
    if hasattr(cause, "non_retryable"):
        info["non_retryable"] = cause.non_retryable
    return info


def _extract_root_cause_info(exc) -> Optional[dict]:
    """Extract root cause info from an exception's deeper cause chain."""
    cause = getattr(exc, "cause", None) or getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
    if not cause:
        return None
    deeper = getattr(cause, "cause", None) or getattr(cause, "__cause__", None)
    if not deeper:
        return None

    info = {"type": type(deeper).__name__, "message": str(deeper)[:500]}
    dc_type = _safe_error_type(deeper)
    if dc_type:
        info["error_type"] = dc_type
    return info


def _build_error_dict(exc: Exception) -> dict:
    """Build error dict with cause chain for WorkflowFailed payload."""
    error = {"type": type(exc).__name__, "message": str(exc)}
    cause_info = _extract_cause_info(exc)
    if cause_info:
        error["cause"] = cause_info
    root_info = _extract_root_cause_info(exc)
    if root_info:
        error["root_cause"] = root_info
    return error


def _serialize_value(value: Any) -> Any:
    """Convert a value to JSON-serializable format for workflow result.

    NOTE: Intentionally duplicated from activity_interceptor._serialize_value.
    Workflow interceptor runs inside Temporal sandbox — cannot import from
    activity_interceptor (which has non-sandbox-safe imports like httpx).
    """
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            import base64

            return base64.b64encode(value).decode("ascii")
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    try:
        return json.loads(json.dumps(value, default=str))
    except Exception:
        return str(value)


# Re-export from errors.py for backward compatibility
from .errors import GovernanceHaltError  # noqa: F401


async def _send_governance_event(
    payload: dict,
    timeout: float,
    on_api_error: str = "fail_open",
) -> Optional[dict]:
    """
    Send governance event via activity.

    Args:
        on_api_error: "fail_open" (default) = continue on error
                      "fail_closed" = halt workflow if governance API fails

    Credentials (api_url, api_key) are held by the activity instance itself —
    never passed through activity inputs, so they never land in workflow
    history. The on_api_error policy is passed to the activity, which handles
    logging (safe outside sandbox) and raises GovernanceAPIError if
    fail_closed. This interceptor catches that and re-raises as
    GovernanceHaltError.
    """
    try:
        result = await workflow.execute_activity(
            "send_governance_event",
            args=[
                {
                    "payload": payload,
                    "timeout": timeout,
                    "on_api_error": on_api_error,
                }
            ],
            start_to_close_timeout=timedelta(seconds=timeout + 5),
        )
        return result
    except Exception as e:
        app_error_type = _application_error_type(e)

        # GovernanceHalt / legacy GovernanceStop: workflow should terminate
        # (client.terminate() already called by the activity, but re-raise to
        # ensure the workflow code path stops).
        if app_error_type in (
            GOVERNANCE_HALT_ERROR_TYPE,
            GOVERNANCE_STOP_ERROR_TYPE,
        ):
            raise GovernanceHaltError(str(e))

        # GovernanceBlock: the current activity is blocked; the workflow continues.
        if app_error_type == GOVERNANCE_BLOCK_ERROR_TYPE:
            return None

        # Activity raised GovernanceAPIError (fail_closed + API unreachable).
        if app_error_type == GOVERNANCE_API_ERROR_TYPE:
            raise GovernanceHaltError(str(e))

        # Other errors with fail_open: silently continue.
        return None


class GovernanceInterceptor(Interceptor):
    """Factory for workflow interceptor. Events sent via activity for determinism."""

    def __init__(
        self,
        api_url: str = "",
        api_key: str = "",
        span_processor=None,  # Shared with activity interceptor for HTTP spans
        config=None,  # Optional GovernanceConfig
    ):
        # api_url / api_key are accepted for backward compatibility but no longer
        # flow to the governance activity — credentials live on GovernanceActivities.
        # Kept on self in case downstream callers introspect.
        self.api_url = api_url.rstrip("/") if api_url else ""
        self.api_key = api_key
        self.span_processor = span_processor
        self.api_timeout = getattr(config, "api_timeout", 30.0) if config else 30.0
        self.on_api_error = (
            getattr(config, "on_api_error", "fail_open") if config else "fail_open"
        )
        self.send_start_event = (
            getattr(config, "send_start_event", True) if config else True
        )
        self.skip_workflow_types = (
            getattr(config, "skip_workflow_types", set()) if config else set()
        )
        self.skip_signals = getattr(config, "skip_signals", set()) if config else set()

    def workflow_interceptor_class(
        self, input: WorkflowInterceptorClassInput
    ) -> Optional[Type[WorkflowInboundInterceptor]]:
        # Capture via closure
        span_processor = self.span_processor
        timeout = self.api_timeout
        on_error = self.on_api_error
        send_start = self.send_start_event
        skip_types = self.skip_workflow_types
        skip_sigs = self.skip_signals

        class _Inbound(WorkflowInboundInterceptor):
            async def execute_workflow(self, input: ExecuteWorkflowInput) -> Any:
                info = workflow.info()

                # Skip if configured
                if info.workflow_type in skip_types:
                    return await super().execute_workflow(input)

                # WorkflowStarted event
                if send_start and workflow.patched("openbox-v2-start"):
                    await _send_governance_event(
                        {
                            "source": "workflow-telemetry",
                            "event_type": "WorkflowStarted",
                            "workflow_id": info.workflow_id,
                            "run_id": info.run_id,
                            "workflow_type": info.workflow_type,
                            "task_queue": info.task_queue,
                        },
                        timeout,
                        on_error,
                    )

                # Execute workflow
                error = None
                try:
                    result = await super().execute_workflow(input)

                    # WorkflowCompleted event (success)
                    if workflow.patched("openbox-v2-complete"):
                        # Serialize workflow result for governance
                        workflow_output = None
                        try:
                            workflow_output = _serialize_value(result)
                        except Exception:
                            workflow_output = (
                                str(result) if result is not None else None
                            )

                        await _send_governance_event(
                            {
                                "source": "workflow-telemetry",
                                "event_type": "WorkflowCompleted",
                                "workflow_id": info.workflow_id,
                                "run_id": info.run_id,
                                "workflow_type": info.workflow_type,
                                "workflow_output": workflow_output,
                            },
                            timeout,
                            on_error,
                        )

                    return result
                except Exception as e:
                    error = _build_error_dict(e)

                    if workflow.patched("openbox-v2-failed"):
                        # Swallow failures from the failure-reporting activity itself
                        # (fail_closed + governance API down would otherwise raise
                        # GovernanceHaltError and shadow the real workflow exception).
                        try:
                            await _send_governance_event(
                                {
                                    "source": "workflow-telemetry",
                                    "event_type": "WorkflowFailed",
                                    "workflow_id": info.workflow_id,
                                    "run_id": info.run_id,
                                    "workflow_type": info.workflow_type,
                                    "error": error,
                                },
                                timeout,
                                on_error,
                            )
                        except Exception:
                            pass

                    raise

            async def handle_signal(self, input: HandleSignalInput) -> None:
                info = workflow.info()

                # Skip if configured
                if input.signal in skip_sigs or info.workflow_type in skip_types:
                    return await super().handle_signal(input)

                # SignalReceived event - check verdict and store if "stop"
                if workflow.patched("openbox-v2-signal"):
                    result = await _send_governance_event(
                        {
                            "source": "workflow-telemetry",
                            "event_type": "SignalReceived",
                            "workflow_id": info.workflow_id,
                            "run_id": info.run_id,
                            "workflow_type": info.workflow_type,
                            "task_queue": info.task_queue,
                            "signal_name": input.signal,
                            "signal_args": input.args,
                        },
                        timeout,
                        on_error,
                    )

                    # If governance returned BLOCK/HALT, store verdict for activity interceptor
                    # The next activity will check this and fail with GovernanceStop
                    verdict = (
                        Verdict.from_string(
                            result.get("verdict") or result.get("action")
                        )
                        if result
                        else Verdict.ALLOW
                    )
                    if verdict.should_stop() and span_processor:
                        span_processor.set_verdict(
                            info.workflow_id,
                            verdict,
                            result.get("reason"),
                            info.run_id,  # Include run_id to detect stale verdicts
                        )

                await super().handle_signal(input)

        return _Inbound
