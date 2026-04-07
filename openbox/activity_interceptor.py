# openbox/activity_interceptor.py
# Handles: ActivityStarted, ActivityCompleted (direct HTTP, WITH spans)
"""
Temporal activity interceptor for activity-boundary governance.

ActivityGovernanceInterceptor: Factory that creates ActivityInboundInterceptor

Captures 2 activity-level events:
4. ActivityStarted (execute_activity entry)
5. ActivityCompleted (execute_activity exit)

NOTE: Workflow events (WorkflowStarted, WorkflowCompleted, SignalReceived) are
handled by GovernanceInterceptor in workflow_interceptor.py

IMPORTANT: Activities CAN use datetime/time and make HTTP calls directly.
This is different from workflow interceptors which must maintain determinism.
"""

from typing import Optional, Any, List
import dataclasses
from dataclasses import asdict, is_dataclass, fields
import time
import json


from .types import rfc3339_now as _rfc3339_now  # shared utility


def _deep_update_dataclass(obj: Any, data: dict, _logger=None) -> None:
    """Recursively update a dataclass object's fields from a dict."""
    if not is_dataclass(obj) or isinstance(obj, type):
        return

    for field in fields(obj):
        if field.name not in data:
            continue

        new_value = data[field.name]
        current_value = getattr(obj, field.name)

        if _should_recurse_dataclass(current_value, new_value):
            _deep_update_dataclass(current_value, new_value, _logger)
        elif isinstance(current_value, list) and isinstance(new_value, list):
            _update_list_items(current_value, new_value, _logger)
        else:
            if _logger:
                _logger.info(
                    f"_deep_update: Setting {type(obj).__name__}.{field.name} = {new_value}"
                )
            setattr(obj, field.name, new_value)


def _should_recurse_dataclass(current: Any, new_value: Any) -> bool:
    """Check if current value is a dataclass that should be recursively updated."""
    return (
        is_dataclass(current)
        and not isinstance(current, type)
        and isinstance(new_value, dict)
    )


def _update_list_items(current_list: list, new_list: list, _logger=None) -> None:
    """Update list items, recursing into dataclass items."""
    for i, (curr_item, new_item) in enumerate(zip(current_list, new_list)):
        if _should_recurse_dataclass(curr_item, new_item):
            _deep_update_dataclass(curr_item, new_item, _logger)
        elif i < len(current_list):
            current_list[i] = new_item


from temporalio import activity
from temporalio.worker import (
    Interceptor,
    ActivityInboundInterceptor,
    ExecuteActivityInput,
)
from opentelemetry import trace

from .span_processor import WorkflowSpanProcessor
from .config import GovernanceConfig
from .types import (
    WorkflowEventType,
    WorkflowSpanBuffer,
    GovernanceVerdictResponse,
    Verdict,
    GovernanceBlockedError,
)
from .hook_governance import build_auth_headers
from .activities import _terminate_workflow_for_halt
from .verdict_handler import enforce_verdict
from .errors import GovernanceHaltError, GuardrailsValidationError
from .client import GovernanceClient


def _serialize_value(value: Any) -> Any:
    """Convert a value to JSON-serializable format."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return _serialize_bytes(value)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    if hasattr(value, "data") and hasattr(value, "metadata"):
        return _serialize_temporal_payload(value)
    return _serialize_fallback(value)


def _serialize_bytes(value: bytes) -> str:
    """Serialize bytes to UTF-8 string or base64."""
    try:
        return value.decode("utf-8")
    except Exception:
        import base64
        return base64.b64encode(value).decode("ascii")


def _serialize_temporal_payload(value: Any) -> Any:
    """Serialize a Temporal Payload object."""
    try:
        payload_data = value.data
        if isinstance(payload_data, bytes):
            return json.loads(payload_data.decode("utf-8"))
        return str(payload_data)
    except Exception:
        return f"<Payload: {len(value.data) if hasattr(value, 'data') else '?'} bytes>"


def _serialize_fallback(value: Any) -> Any:
    """Last-resort serialization via json.dumps(default=str)."""
    try:
        return json.loads(json.dumps(value, default=str))
    except Exception:
        return str(value)


class ActivityGovernanceInterceptor(Interceptor):
    """Factory for activity interceptor. Events sent directly (activities can do HTTP)."""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        span_processor: WorkflowSpanProcessor,
        config: Optional[GovernanceConfig] = None,
        client: Optional[GovernanceClient] = None,
    ):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.span_processor = span_processor
        self.config = config or GovernanceConfig()
        self._client = client or GovernanceClient(
            api_url=api_url,
            api_key=api_key,
            timeout=self.config.api_timeout,
            on_api_error=self.config.on_api_error,
        )

    def intercept_activity(
        self, next_interceptor: ActivityInboundInterceptor
    ) -> ActivityInboundInterceptor:
        return _ActivityInterceptor(
            next_interceptor,
            self.api_url,
            self.api_key,
            self.span_processor,
            self.config,
            self._client,
        )


class _ActivityInterceptor(ActivityInboundInterceptor):
    def __init__(
        self,
        next_interceptor: ActivityInboundInterceptor,
        api_url: str,
        api_key: str,
        span_processor: WorkflowSpanProcessor,
        config: GovernanceConfig,
        client: Optional[GovernanceClient] = None,
    ):
        super().__init__(next_interceptor)
        self._api_url = api_url
        self._api_key = api_key
        self._span_processor = span_processor
        self._config = config
        self._client = client or GovernanceClient(
            api_url=api_url,
            api_key=api_key,
            timeout=config.api_timeout,
            on_api_error=config.on_api_error,
        )

    async def execute_activity(self, input: ExecuteActivityInput) -> Any:
        info = activity.info()
        start_time = time.time()

        # Skip if configured (e.g., send_governance_event to avoid loops)
        if info.activity_type in self._config.skip_activity_types:
            return await self.next.execute_activity(input)

        # Check for blocking verdicts from prior governance (signal or buffer)
        await self._check_pending_verdicts(info)

        # Check for pending approval on retry (HITL polling)
        await self._check_pending_approval(info)

        # Clear stale state and register fresh buffer
        self._span_processor.clear_activity_abort(info.workflow_id, info.activity_id)
        buffer = WorkflowSpanBuffer(
            workflow_id=info.workflow_id,
            run_id=info.workflow_run_id,
            workflow_type=info.workflow_type,
            task_queue=info.task_queue,
        )
        self._span_processor.register_workflow(info.workflow_id, buffer)

        # Serialize activity input
        activity_input = self._serialize_input(input, info)

        # Send ActivityStarted event (optional)
        governance_verdict: Optional[GovernanceVerdictResponse] = None
        if self._config.send_activity_start_event:
            governance_verdict = await self._send_activity_event(
                info,
                WorkflowEventType.ACTIVITY_STARTED.value,
                activity_input=activity_input,
            )

        # Buffer activity context for hook-level governance
        self._span_processor.set_activity_context(
            info.workflow_id,
            info.activity_id,
            {
                "source": "workflow-telemetry",
                "event_type": WorkflowEventType.ACTIVITY_STARTED.value,
                "workflow_id": info.workflow_id,
                "run_id": info.workflow_run_id,
                "workflow_type": info.workflow_type,
                "activity_id": info.activity_id,
                "activity_type": info.activity_type,
                "task_queue": info.task_queue,
                "attempt": info.attempt,
                "activity_input": activity_input,
                "activity_output": None,
            },
        )

        # Enforce ActivityStarted verdict (HITL, BLOCK, HALT, guardrails)
        if governance_verdict:
            await self._enforce_verdict(governance_verdict, info, "activity_start")

        # Apply guardrails input redaction
        activity_input = self._apply_input_redaction(
            governance_verdict, input, activity_input
        )

        # Execute the actual activity
        result, status, error, activity_output, end_time = await self._run_activity(
            input, info
        )

        # Send ActivityCompleted + enforce verdict + apply output redaction
        result = await self._handle_completion(
            info, status, error, start_time, end_time,
            activity_input, activity_output, result,
        )

        return result

    # ─── Verdict checks ───────────────────────────────────────────────────

    async def _check_pending_verdicts(self, info) -> None:
        """Check for blocking verdicts from prior signal governance or buffer."""
        buffer = self._span_processor.get_buffer(info.workflow_id)

        # Clear stale buffer from previous workflow run
        if buffer and buffer.run_id != info.workflow_run_id:
            activity.logger.info(
                f"Clearing stale buffer for workflow {info.workflow_id}"
            )
            self._span_processor.unregister_workflow(info.workflow_id)
            buffer = None

        # Check pending verdict (stored by workflow interceptor for SignalReceived stop)
        pending_verdict = self._span_processor.get_verdict(info.workflow_id)
        if pending_verdict and pending_verdict.get("run_id") != info.workflow_run_id:
            self._span_processor.clear_verdict(info.workflow_id)
            pending_verdict = None

        activity.logger.info(
            f"Checking verdict for workflow {info.workflow_id}: "
            f"buffer={buffer is not None}, "
            f"buffer.verdict={buffer.verdict if buffer else None}, "
            f"pending_verdict={pending_verdict}"
        )

        # Enforce pending verdict from signal governance
        if pending_verdict:
            verdict_str = pending_verdict.get("verdict")
            if verdict_str and Verdict.from_string(verdict_str).should_stop():
                await self._enforce_stop_verdict(
                    Verdict.from_string(verdict_str),
                    pending_verdict.get("reason") or "Workflow blocked by governance",
                    info.workflow_id,
                )

        # Enforce buffer verdict
        if buffer and buffer.verdict and buffer.verdict.should_stop():
            await self._enforce_stop_verdict(
                buffer.verdict,
                buffer.verdict_reason or "Workflow blocked by governance",
                info.workflow_id,
            )

    async def _enforce_stop_verdict(
        self, verdict: Verdict, reason: str, workflow_id: str
    ) -> None:
        """Enforce a BLOCK or HALT verdict."""
        activity.logger.info(f"Activity blocked by prior governance verdict: {reason}")
        if verdict == Verdict.HALT:
            await _terminate_workflow_for_halt(workflow_id, reason)
        else:
            from temporalio.exceptions import ApplicationError
            raise ApplicationError(
                f"Governance blocked: {reason}",
                type="GovernanceBlock",
                non_retryable=True,
            )

    # ─── HITL approval ────────────────────────────────────────────────────

    async def _check_pending_approval(self, info) -> bool:
        """Poll for pending HITL approval on retry. Returns True if approved."""
        from .hitl import handle_approval_response, should_skip_hitl

        if should_skip_hitl(
            info.activity_type,
            hitl_enabled=self._config.hitl_enabled,
            skip_types=self._config.skip_hitl_activity_types,
        ):
            return False

        buffer = self._span_processor.get_buffer(info.workflow_id)
        if not (buffer and buffer.pending_approval):
            return False

        activity.logger.info(
            f"Polling approval status for workflow_id={info.workflow_id}, "
            f"activity_id={info.activity_id}"
        )
        approval_response = await self._client.poll_approval(
            info.workflow_id, info.workflow_run_id, info.activity_id
        )
        approved = handle_approval_response(
            approval_response,
            info.activity_type,
            info.workflow_id,
            info.workflow_run_id,
            info.activity_id,
        )
        if approved:
            activity.logger.info(
                f"Approval granted for workflow_id={info.workflow_id}"
            )
            buffer.pending_approval = False
            return True
        return False

    # ─── Input serialization ──────────────────────────────────────────────

    def _serialize_input(self, input: ExecuteActivityInput, info) -> list:
        """Serialize activity input arguments."""
        try:
            args_list = list(input.args) if input.args is not None else []
            if args_list:
                result = _serialize_value(args_list)
            else:
                result = []
            activity.logger.info(
                f"Activity {info.activity_type} input: {len(args_list)} args, "
                f"types: {[type(a).__name__ for a in args_list]}"
            )
            return result
        except Exception as e:
            activity.logger.warning(f"Failed to serialize activity input: {e}")
            try:
                return [str(arg) for arg in input.args] if input.args else []
            except Exception:
                return []

    # ─── Verdict enforcement ──────────────────────────────────────────────

    async def _enforce_verdict(
        self, verdict_response: GovernanceVerdictResponse, info, context: str
    ) -> None:
        """Enforce a governance verdict (HITL, BLOCK, HALT, guardrails)."""
        from .hitl import should_skip_hitl, raise_approval_pending

        try:
            verdict_result = enforce_verdict(verdict_response, context)
            if verdict_result.requires_hitl and not should_skip_hitl(
                info.activity_type,
                hitl_enabled=self._config.hitl_enabled,
                skip_types=self._config.skip_hitl_activity_types,
            ):
                buffer = self._span_processor.get_buffer(info.workflow_id)
                if buffer:
                    buffer.pending_approval = True
                    activity.logger.info(
                        f"Pending approval stored: workflow_id={info.workflow_id}"
                    )
                raise_approval_pending(
                    f"Approval required: {verdict_response.reason or 'Activity requires human approval'}"
                )
        except GovernanceHaltError as e:
            await _terminate_workflow_for_halt(info.workflow_id, str(e))
        except GovernanceBlockedError as e:
            from temporalio.exceptions import ApplicationError
            raise ApplicationError(
                f"Governance blocked: {e.reason}",
                type="GovernanceBlock",
                non_retryable=True,
            )
        except GuardrailsValidationError as e:
            from temporalio.exceptions import ApplicationError
            activity.logger.info(f"Guardrails validation failed: {e}")
            raise ApplicationError(
                f"Guardrails validation failed: {e}",
                type="GuardrailsValidationFailed",
                non_retryable=True,
            )

    # ─── Guardrails redaction ─────────────────────────────────────────────

    def _apply_input_redaction(
        self,
        verdict: Optional[GovernanceVerdictResponse],
        input: ExecuteActivityInput,
        activity_input: list,
    ) -> list:
        """Apply guardrails input redaction if present. Returns updated activity_input."""
        if not (
            verdict
            and verdict.guardrails_result
            and verdict.guardrails_result.input_type == "activity_input"
        ):
            return activity_input

        redacted = verdict.guardrails_result.redacted_input
        activity.logger.info("Applying guardrails redaction to activity input")

        # Normalize to list to match args structure
        if isinstance(redacted, dict):
            redacted = [redacted]

        if not isinstance(redacted, list):
            activity.logger.warning(
                f"Unexpected redacted_input type: {type(redacted).__name__}"
            )
            return activity_input

        original_args = list(input.args) if input.args else []
        for i, redacted_item in enumerate(redacted):
            if i < len(original_args) and isinstance(redacted_item, dict):
                original_arg = original_args[i]
                if is_dataclass(original_arg) and not isinstance(original_arg, type):
                    _deep_update_dataclass(original_arg, redacted_item, activity.logger)
                else:
                    original_args[i] = redacted_item

        activity.logger.info("Updated activity_input for completed event")
        return _serialize_value(original_args)

    def _apply_output_redaction(
        self, verdict: Optional[GovernanceVerdictResponse], result: Any
    ) -> Any:
        """Apply guardrails output redaction if present."""
        if not (
            verdict
            and verdict.guardrails_result
            and verdict.guardrails_result.input_type == "activity_output"
        ):
            return result

        redacted_output = verdict.guardrails_result.redacted_input
        activity.logger.info("Applying guardrails redaction to activity output")

        if redacted_output is None:
            return result

        if (
            is_dataclass(result)
            and not isinstance(result, type)
            and isinstance(redacted_output, dict)
        ):
            _deep_update_dataclass(result, redacted_output)
            return result

        return redacted_output

    # ─── Activity execution ───────────────────────────────────────────────

    async def _run_activity(self, input: ExecuteActivityInput, info):
        """Execute the activity and handle hook-level governance errors."""
        from .hitl import should_skip_hitl, raise_approval_pending

        tracer = trace.get_tracer(__name__)
        status = "completed"
        error = None
        activity_output = None
        result = None

        with tracer.start_as_current_span(
            f"activity.{info.activity_type}",
            attributes={
                "temporal.workflow_id": info.workflow_id,
                "temporal.activity_id": info.activity_id,
            },
        ) as span:
            self._span_processor.register_trace(
                span.get_span_context().trace_id,
                info.workflow_id,
                info.activity_id,
            )

            try:
                result = await self.next.execute_activity(input)
                activity_output = _serialize_value(result)
            except GovernanceBlockedError as e:
                status = "failed"
                error = {
                    "type": "GovernanceBlockedError",
                    "message": str(e),
                    "verdict": e.verdict,
                    "url": e.url,
                }
                self._handle_hook_governance_error(e, info)
            except Exception as e:
                status = "failed"
                error = {"type": type(e).__name__, "message": str(e)}
                raise

        end_time = time.time()
        return result, status, error, activity_output, end_time

    def _handle_hook_governance_error(self, e: GovernanceBlockedError, info) -> None:
        """Handle GovernanceBlockedError from hook-level governance."""
        from .hitl import should_skip_hitl, raise_approval_pending
        from temporalio.exceptions import ApplicationError

        # REQUIRE_APPROVAL → retryable
        if e.verdict.requires_approval() and not should_skip_hitl(
            info.activity_type,
            hitl_enabled=self._config.hitl_enabled,
            skip_types=self._config.skip_hitl_activity_types,
        ):
            buffer = self._span_processor.get_buffer(info.workflow_id)
            if buffer:
                buffer.pending_approval = True
                activity.logger.info(
                    f"Hook REQUIRE_APPROVAL: pending approval for {info.activity_type} "
                    f"(resource: {e.url})"
                )
            raise_approval_pending(f"Approval required: {e.reason}")

        # BLOCK/HALT → non-retryable
        error_type = (
            "GovernanceHalt" if e.verdict == Verdict.HALT else "GovernanceBlock"
        )
        raise ApplicationError(
            f"Hook governance {e.verdict.value}: {e.reason}",
            type=error_type,
            non_retryable=True,
        )

    # ─── Post-execution handling ──────────────────────────────────────────

    async def _handle_completion(
        self, info, status, error, start_time, end_time,
        activity_input, activity_output, result,
    ) -> Any:
        """Send ActivityCompleted, enforce verdict, apply output redaction."""
        # Check abort/halt flags
        was_aborted = (
            self._span_processor.get_activity_abort(
                info.workflow_id, info.activity_id
            )
            is not None
        )
        halt_reason = self._span_processor.get_halt_requested(
            info.workflow_id, info.activity_id
        )
        if halt_reason:
            self._span_processor.clear_halt_requested(
                info.workflow_id, info.activity_id
            )
            await _terminate_workflow_for_halt(info.workflow_id, halt_reason)

        # Cleanup
        self._span_processor.clear_activity_abort(
            info.workflow_id, info.activity_id
        )
        self._span_processor.clear_activity_context(
            info.workflow_id, info.activity_id
        )

        # Send ActivityCompleted event (unless aborted by hook governance)
        completed_verdict = None
        if was_aborted:
            activity.logger.info(
                "Skipping ActivityCompleted event — activity aborted by hook governance"
            )
        else:
            completed_verdict = await self._send_activity_event(
                info,
                WorkflowEventType.ACTIVITY_COMPLETED.value,
                status=status,
                start_time=start_time,
                end_time=end_time,
                duration_ms=(end_time - start_time) * 1000,
                span_count=0,
                spans=[],
                activity_input=activity_input,
                activity_output=activity_output,
                error=error,
            )

        # Enforce completed verdict
        if completed_verdict:
            await self._enforce_verdict(completed_verdict, info, "activity_end")

        # Apply output redaction
        return self._apply_output_redaction(completed_verdict, result)

    # ─── Event sending ────────────────────────────────────────────────────

    async def _send_activity_event(
        self, info, event_type: str, **extra
    ) -> Optional[GovernanceVerdictResponse]:
        """Send activity event via GovernanceClient."""
        serialized_extra = {}
        for key, value in extra.items():
            try:
                serialized_extra[key] = _serialize_value(value)
            except Exception as e:
                activity.logger.warning(f"Failed to serialize {key}: {e}")
                serialized_extra[key] = str(value) if value is not None else None

        payload = {
            "source": "workflow-telemetry",
            "event_type": event_type,
            "workflow_id": info.workflow_id,
            "run_id": info.workflow_run_id,
            "workflow_type": info.workflow_type,
            "activity_id": info.activity_id,
            "activity_type": info.activity_type,
            "task_queue": info.task_queue,
            "attempt": info.attempt,
            "timestamp": _rfc3339_now(),
            **serialized_extra,
        }

        # Final safety check - ensure payload is JSON serializable
        try:
            json.dumps(payload)
        except TypeError as e:
            activity.logger.warning(f"Payload not JSON serializable, cleaning: {e}")
            payload = json.loads(json.dumps(payload, default=str))

        return await self._client.evaluate_event(payload)
