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
from temporalio.worker import (
    Interceptor,
    WorkflowInboundInterceptor,
    WorkflowInterceptorClassInput,
    ExecuteWorkflowInput,
    HandleSignalInput,
)

from .types import Verdict


def _serialize_value(value: Any) -> Any:
    """Convert a value to JSON-serializable format for workflow result."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode('utf-8')
        except Exception:
            import base64
            return base64.b64encode(value).decode('ascii')
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


class GovernanceHaltError(Exception):
    """Raised when governance halts workflow execution."""
    def __init__(self, message: str):
        super().__init__(message)


async def _send_governance_event(
    api_url: str,
    api_key: str,
    payload: dict,
    timeout: float,
    on_api_error: str = "fail_open",
) -> Optional[dict]:
    """
    Send governance event via activity.

    Args:
        on_api_error: "fail_open" (default) = continue on error
                      "fail_closed" = halt workflow if governance API fails

    The on_api_error policy is passed to the activity, which handles logging
    (safe outside sandbox) and raises GovernanceAPIError if fail_closed.
    This interceptor catches that and re-raises as GovernanceHaltError.
    """
    try:
        result = await workflow.execute_activity(
            "send_governance_event",
            args=[{
                "api_url": api_url,
                "api_key": api_key,
                "payload": payload,
                "timeout": timeout,
                "on_api_error": on_api_error,
            }],
            start_to_close_timeout=timedelta(seconds=timeout + 5),
        )
        return result
    except Exception as e:
        error_str = str(e)
        error_type = type(e).__name__

        # ApplicationError with type="GovernanceStop" (non-retryable, terminates workflow)
        # This is raised when governance API returns action='stop'
        if "ApplicationError" in error_type or "Governance blocked:" in error_str:
            raise GovernanceHaltError(error_str)

        # Activity raised GovernanceAPIError (fail_closed and API unreachable)
        if "GovernanceAPIError" in error_type or "GovernanceAPIError" in error_str:
            raise GovernanceHaltError(error_str)

        # Other errors with fail_open: silently continue
        return None


class GovernanceInterceptor(Interceptor):
    """Factory for workflow interceptor. Events sent via activity for determinism."""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        span_processor=None,  # Shared with activity interceptor for HTTP spans
        config=None,  # Optional GovernanceConfig
    ):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.span_processor = span_processor
        self.api_timeout = getattr(config, 'api_timeout', 30.0) if config else 30.0
        self.on_api_error = getattr(config, 'on_api_error', 'fail_open') if config else 'fail_open'
        self.send_start_event = getattr(config, 'send_start_event', True) if config else True
        self.skip_workflow_types = getattr(config, 'skip_workflow_types', set()) if config else set()
        self.skip_signals = getattr(config, 'skip_signals', set()) if config else set()

    def workflow_interceptor_class(
        self, input: WorkflowInterceptorClassInput
    ) -> Optional[Type[WorkflowInboundInterceptor]]:
        # Capture via closure
        api_url = self.api_url
        api_key = self.api_key
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
                    await _send_governance_event(api_url, api_key, {
                        "source": "workflow-telemetry",
                        "event_type": "WorkflowStarted",
                        "workflow_id": info.workflow_id,
                        "run_id": info.run_id,
                        "workflow_type": info.workflow_type,
                        "task_queue": info.task_queue,
                    }, timeout, on_error)

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
                            workflow_output = str(result) if result is not None else None

                        await _send_governance_event(api_url, api_key, {
                            "source": "workflow-telemetry",
                            "event_type": "WorkflowCompleted",
                            "workflow_id": info.workflow_id,
                            "run_id": info.run_id,
                            "workflow_type": info.workflow_type,
                            "workflow_output": workflow_output,
                        }, timeout, on_error)

                    return result
                except Exception as e:
                    # Extract error details, including nested cause for ActivityError
                    error = {
                        "type": type(e).__name__,
                        "message": str(e),
                    }

                    # Get cause using Temporal's .cause property or Python's __cause__/__context__
                    cause = getattr(e, 'cause', None) or e.__cause__ or e.__context__

                    if cause:
                        error["cause"] = {
                            "type": type(cause).__name__,
                            "message": str(cause),
                        }
                        # Check for ApplicationError details (e.g., GovernanceStop)
                        if hasattr(cause, 'type') and cause.type:
                            error["cause"]["error_type"] = cause.type
                        if hasattr(cause, 'non_retryable'):
                            error["cause"]["non_retryable"] = cause.non_retryable

                        # Go deeper if there's another cause
                        deeper_cause = getattr(cause, 'cause', None) or getattr(cause, '__cause__', None)
                        if deeper_cause:
                            error["root_cause"] = {
                                "type": type(deeper_cause).__name__,
                                "message": str(deeper_cause),
                            }
                            if hasattr(deeper_cause, 'type') and deeper_cause.type:
                                error["root_cause"]["error_type"] = deeper_cause.type

                    # WorkflowFailed event
                    if workflow.patched("openbox-v2-failed"):
                        await _send_governance_event(api_url, api_key, {
                            "source": "workflow-telemetry",
                            "event_type": "WorkflowFailed",
                            "workflow_id": info.workflow_id,
                            "run_id": info.run_id,
                            "workflow_type": info.workflow_type,
                            "error": error,
                        }, timeout, on_error)

                    raise

            async def handle_signal(self, input: HandleSignalInput) -> None:
                info = workflow.info()

                # Skip if configured
                if input.signal in skip_sigs or info.workflow_type in skip_types:
                    return await super().handle_signal(input)

                # SignalReceived event - check verdict and store if "stop"
                if workflow.patched("openbox-v2-signal"):
                    result = await _send_governance_event(api_url, api_key, {
                        "source": "workflow-telemetry",
                        "event_type": "SignalReceived",
                        "workflow_id": info.workflow_id,
                        "run_id": info.run_id,
                        "workflow_type": info.workflow_type,
                        "task_queue": info.task_queue,
                        "signal_name": input.signal,
                        "signal_args": input.args,
                    }, timeout, on_error)

                    # If governance returned BLOCK/HALT, store verdict for activity interceptor
                    # The next activity will check this and fail with GovernanceStop
                    verdict = Verdict.from_string(result.get("verdict") or result.get("action")) if result else Verdict.ALLOW
                    if verdict.should_stop() and span_processor:
                        span_processor.set_verdict(
                            info.workflow_id,
                            verdict,
                            result.get("reason"),
                            info.run_id,  # Include run_id to detect stale verdicts
                        )

                await super().handle_signal(input)

        return _Inbound