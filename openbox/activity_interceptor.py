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
    """
    Recursively update a dataclass object's fields from a dict.
    Preserves the original object types while updating values.
    """
    if not is_dataclass(obj) or isinstance(obj, type):
        return

    for field in fields(obj):
        if field.name not in data:
            continue

        new_value = data[field.name]
        current_value = getattr(obj, field.name)

        # If current field is a dataclass and new value is a dict, recurse
        if is_dataclass(current_value) and not isinstance(current_value, type) and isinstance(new_value, dict):
            _deep_update_dataclass(current_value, new_value, _logger)
        # If current field is a list of dataclasses and new value is a list of dicts
        elif isinstance(current_value, list) and isinstance(new_value, list):
            for i, (curr_item, new_item) in enumerate(zip(current_value, new_value)):
                if is_dataclass(curr_item) and not isinstance(curr_item, type) and isinstance(new_item, dict):
                    _deep_update_dataclass(curr_item, new_item, _logger)
                elif i < len(current_value):
                    current_value[i] = new_item
        else:
            # Simple value - just update
            if _logger:
                _logger.info(f"_deep_update: Setting {type(obj).__name__}.{field.name} = {new_value}")
            setattr(obj, field.name, new_value)

from temporalio import activity
from temporalio.worker import (
    Interceptor,
    ActivityInboundInterceptor,
    ExecuteActivityInput,
)
from opentelemetry import trace

from .span_processor import WorkflowSpanProcessor
from .config import GovernanceConfig
from .types import WorkflowEventType, WorkflowSpanBuffer, GovernanceVerdictResponse, Verdict, GovernanceBlockedError
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
        # Try to decode bytes as UTF-8, fallback to base64
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
    # Handle Temporal Payload objects
    if hasattr(value, 'data') and hasattr(value, 'metadata'):
        # This is likely a Temporal Payload - try to decode it
        try:
            payload_data = value.data
            if isinstance(payload_data, bytes):
                return json.loads(payload_data.decode('utf-8'))
            return str(payload_data)
        except Exception:
            return f"<Payload: {len(value.data) if hasattr(value, 'data') else '?'} bytes>"
    # Try to convert to string for other types
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
        # Use provided client or create one internally (backward compat)
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
        # Use provided GovernanceClient for HTTP calls (or create one for direct instantiation)
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

        # Check if workflow has a pending "stop" verdict from signal governance
        # This allows signal handlers to block subsequent activities
        buffer = self._span_processor.get_buffer(info.workflow_id)

        # If buffer exists but run_id doesn't match, it's from a previous workflow run - clear it
        if buffer and buffer.run_id != info.workflow_run_id:
            activity.logger.info(f"Clearing stale buffer for workflow {info.workflow_id} (old run_id={buffer.run_id}, new run_id={info.workflow_run_id})")
            self._span_processor.unregister_workflow(info.workflow_id)
            buffer = None

        # Check for pending verdict (stored by workflow interceptor for SignalReceived stop)
        # This is checked BEFORE buffer.verdict because buffer may not exist yet
        pending_verdict = self._span_processor.get_verdict(info.workflow_id)

        # Clear stale verdict from previous workflow run
        if pending_verdict and pending_verdict.get("run_id") != info.workflow_run_id:
            activity.logger.info(f"Clearing stale verdict for workflow {info.workflow_id} (old run_id={pending_verdict.get('run_id')}, new run_id={info.workflow_run_id})")
            self._span_processor.clear_verdict(info.workflow_id)
            pending_verdict = None

        activity.logger.info(f"Checking verdict for workflow {info.workflow_id}: buffer={buffer is not None}, buffer.verdict={buffer.verdict if buffer else None}, pending_verdict={pending_verdict}")

        if pending_verdict and pending_verdict.get("verdict") and Verdict.from_string(pending_verdict.get("verdict")).should_stop():
            pending_v = Verdict.from_string(pending_verdict.get("verdict"))
            reason = pending_verdict.get("reason") or "Workflow blocked by governance"
            activity.logger.info(f"Activity blocked by prior governance verdict (from signal): {reason}")
            if pending_v == Verdict.HALT:
                await _terminate_workflow_for_halt(info.workflow_id, reason)
            else:
                from temporalio.exceptions import ApplicationError
                raise ApplicationError(
                    f"Governance blocked: {reason}",
                    type="GovernanceBlock",
                    non_retryable=True,
                )

        if buffer and buffer.verdict and buffer.verdict.should_stop():
            reason = buffer.verdict_reason or "Workflow blocked by governance"
            activity.logger.info(f"Activity blocked by prior governance verdict (from buffer): {reason}")
            if buffer.verdict == Verdict.HALT:
                await _terminate_workflow_for_halt(info.workflow_id, reason)
            else:
                from temporalio.exceptions import ApplicationError
                raise ApplicationError(
                    f"Governance blocked: {reason}",
                    type="GovernanceBlock",
                    non_retryable=True,
                )

        # ═══ Check for pending approval on retry ═══
        # If there's a pending approval, poll OpenBox Core for status
        from .hitl import handle_approval_response, should_skip_hitl, raise_approval_pending
        approval_granted = False
        if not should_skip_hitl(
            info.activity_type,
            hitl_enabled=self._config.hitl_enabled,
            skip_types=self._config.skip_hitl_activity_types,
        ):
            buffer = self._span_processor.get_buffer(info.workflow_id)
            if buffer and buffer.pending_approval:
                activity.logger.info(f"Polling approval status for workflow_id={info.workflow_id}, activity_id={info.activity_id}")

                # Poll OpenBox Core for approval status, delegate response handling
                approval_response = await self._client.poll_approval(
                    info.workflow_id, info.workflow_run_id, info.activity_id
                )

                activity.logger.info(f"Processing approval response: expired={approval_response.get('expired') if approval_response else None}, verdict={approval_response.get('verdict') if approval_response else None}")

                # Raises ApplicationError for expired/rejected/pending; returns True for ALLOW
                approved = handle_approval_response(
                    approval_response,
                    info.activity_type,
                    info.workflow_id,
                    info.workflow_run_id,
                    info.activity_id,
                )
                if approved:
                    activity.logger.info(f"Approval granted for workflow_id={info.workflow_id}, activity_id={info.activity_id}")
                    buffer.pending_approval = False
                    approval_granted = True

        # Clear any stale abort flag from previous attempt
        self._span_processor.clear_activity_abort(info.workflow_id, info.activity_id)

        # Register a fresh buffer for this activity attempt.
        # Always create new instead of clearing — avoids race with in-flight hooks
        # that may still be writing to the old buffer from a previous attempt.
        buffer = WorkflowSpanBuffer(
            workflow_id=info.workflow_id,
            run_id=info.workflow_run_id,
            workflow_type=info.workflow_type,
            task_queue=info.task_queue,
        )
        self._span_processor.register_workflow(info.workflow_id, buffer)

        tracer = trace.get_tracer(__name__)

        # Serialize activity input arguments
        # input.args is a Sequence[Any] containing the activity arguments
        # For class methods, self is already bound - args contains only the actual arguments
        activity_input = []
        try:
            # Convert to list and serialize each argument
            args_list = list(input.args) if input.args is not None else []
            if args_list:
                activity_input = _serialize_value(args_list)
            # Debug: log what we're capturing
            activity.logger.info(f"Activity {info.activity_type} input: {len(args_list)} args, types: {[type(a).__name__ for a in args_list]}")
        except Exception as e:
            activity.logger.warning(f"Failed to serialize activity input: {e}")
            try:
                activity_input = [str(arg) for arg in input.args] if input.args else []
            except Exception:
                activity_input = []

        # Track governance verdict (may include redacted input)
        governance_verdict: Optional[GovernanceVerdictResponse] = None

        # Optional: Send ActivityStarted event (with input)
        if self._config.send_activity_start_event:
            governance_verdict = await self._send_activity_event(
                info,
                WorkflowEventType.ACTIVITY_STARTED.value,
                activity_input=activity_input,
            )

        # Buffer activity context for hook-level governance (OTel request hooks)
        _activity_event_context = {
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
        }
        self._span_processor.set_activity_context(
            info.workflow_id, info.activity_id, _activity_event_context
        )

        # ═══ Enforce ActivityStarted governance verdict ═══
        if governance_verdict:
            try:
                verdict_result = enforce_verdict(governance_verdict, "activity_start")
                if (
                    verdict_result.requires_hitl
                    and not should_skip_hitl(
                        info.activity_type,
                        hitl_enabled=self._config.hitl_enabled,
                        skip_types=self._config.skip_hitl_activity_types,
                    )
                ):
                    buffer = self._span_processor.get_buffer(info.workflow_id)
                    if buffer:
                        buffer.pending_approval = True
                        activity.logger.info(
                            f"Pending approval stored: workflow_id={info.workflow_id}, run_id={info.workflow_run_id}"
                        )
                    raise_approval_pending(
                        f"Approval required: {governance_verdict.reason or 'Activity requires human approval'}"
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

        # Debug: Log governance verdict details
        if governance_verdict:
            activity.logger.info(
                f"Governance verdict: verdict={governance_verdict.verdict.value}, "
                f"has_guardrails_result={governance_verdict.guardrails_result is not None}"
            )
            if governance_verdict.guardrails_result:
                activity.logger.info(
                    f"Guardrails result: input_type={governance_verdict.guardrails_result.input_type}, "
                    f"redacted_input_type={type(governance_verdict.guardrails_result.redacted_input).__name__}"
                )

        # Apply guardrails redaction if present
        if (
            governance_verdict
            and governance_verdict.guardrails_result
            and governance_verdict.guardrails_result.input_type == "activity_input"
        ):
            redacted = governance_verdict.guardrails_result.redacted_input
            activity.logger.info(f"Applying guardrails redaction to activity input")
            activity.logger.debug(f"Redacted input type: {type(redacted).__name__}")

            # Normalize redacted_input to a list (matching original args structure)
            if isinstance(redacted, dict):
                # API returned a single dict, wrap in list to match args structure
                activity.logger.info("Wrapping dict in list")
                redacted = [redacted]

            if isinstance(redacted, list):
                original_args = list(input.args) if input.args else []
                activity.logger.info(f"Original args count: {len(original_args)}, redacted count: {len(redacted)}")

                for i, redacted_item in enumerate(redacted):
                    activity.logger.info(f"Processing arg {i}: redacted_item type={type(redacted_item).__name__}")
                    if i < len(original_args) and isinstance(redacted_item, dict):
                        original_arg = original_args[i]
                        activity.logger.info(f"Original arg {i} type: {type(original_arg).__name__}, is_dataclass: {is_dataclass(original_arg)}")
                        # If original is a dataclass, update its fields in place (preserves types)
                        if is_dataclass(original_arg) and not isinstance(original_arg, type):
                            _deep_update_dataclass(original_arg, redacted_item, activity.logger)
                            activity.logger.info(f"Updated {type(original_arg).__name__} fields with redacted values")
                            # Verify the update
                            if hasattr(original_arg, 'prompt'):
                                activity.logger.debug(f"After update, prompt redacted")
                        else:
                            # Non-dataclass: replace directly
                            original_args[i] = redacted_item
                            activity.logger.info(f"Replaced arg {i} directly (non-dataclass)")

                # Update activity_input for the completed event (shows redacted values)
                activity_input = _serialize_value(original_args)
                activity.logger.info(f"Updated activity_input for completed event")
            else:
                activity.logger.warning(
                    f"Unexpected redacted_input type: {type(redacted).__name__}, expected list or dict"
                )

        # Debug: Log the actual input that will be passed to activity
        if input.args:
            first_arg = input.args[0]
            if hasattr(first_arg, 'prompt'):
                activity.logger.debug(f"BEFORE ACTIVITY EXECUTION - prompt field present")

        status = "completed"
        error = None
        activity_output = None

        with tracer.start_as_current_span(
            f"activity.{info.activity_type}",
            attributes={
                "temporal.workflow_id": info.workflow_id,
                "temporal.activity_id": info.activity_id,
            },
        ) as span:
            # Register trace_id -> workflow_id + activity_id mapping so child spans
            # (HTTP calls) can be associated with this activity even without attributes
            self._span_processor.register_trace(
                span.get_span_context().trace_id,
                info.workflow_id,
                info.activity_id,
            )

            try:
                result = await self.next.execute_activity(input)
                # Serialize activity output on success
                activity_output = _serialize_value(result)
            except GovernanceBlockedError as e:
                status = "failed"
                error = {"type": "GovernanceBlockedError", "message": str(e), "verdict": e.verdict, "url": e.url}
                from temporalio.exceptions import ApplicationError

                # REQUIRE_APPROVAL → retryable, reuse HITL polling flow
                if (
                    e.verdict.requires_approval()
                    and not should_skip_hitl(
                        info.activity_type,
                        hitl_enabled=self._config.hitl_enabled,
                        skip_types=self._config.skip_hitl_activity_types,
                    )
                ):
                    buffer = self._span_processor.get_buffer(info.workflow_id)
                    if buffer:
                        buffer.pending_approval = True
                        activity.logger.info(
                            f"Hook REQUIRE_APPROVAL: pending approval for {info.activity_type} "
                            f"(resource: {e.url})"
                        )
                    raise_approval_pending(f"Approval required: {e.reason}")

                # Hook-level BLOCK/HALT → raise non-retryable error to stop activity.
                # Preserve verdict in error type for the activity interceptor to
                # differentiate later (e.g. terminate workflow for HALT).
                error_type = "GovernanceHalt" if e.verdict == Verdict.HALT else "GovernanceBlock"
                raise ApplicationError(
                    f"Hook governance {e.verdict.value}: {e.reason}",
                    type=error_type,
                    non_retryable=True,
                )
            except Exception as e:
                status = "failed"
                error = {"type": type(e).__name__, "message": str(e)}
                raise
            finally:
                end_time = time.time()

                # Check abort flag before clearing (determines if we skip ActivityCompleted)
                was_aborted = self._span_processor.get_activity_abort(info.workflow_id, info.activity_id) is not None
                # Check if hook requested HALT → call terminate() here in async context
                halt_reason = self._span_processor.get_halt_requested(info.workflow_id, info.activity_id)
                if halt_reason:
                    self._span_processor.clear_halt_requested(info.workflow_id, info.activity_id)
                    await _terminate_workflow_for_halt(info.workflow_id, halt_reason)
                # Clear abort flag and buffered activity context
                self._span_processor.clear_activity_abort(info.workflow_id, info.activity_id)
                self._span_processor.clear_activity_context(info.workflow_id, info.activity_id)

                # OTel spans not collected — hook-level governance evaluates each
                # operation individually, so bundling spans is redundant.

                # Skip ActivityCompleted when activity was aborted by hook governance
                # (e.g., require_approval) — activity didn't actually run, will retry
                completed_verdict = None
                if was_aborted:
                    activity.logger.info(
                        f"Skipping ActivityCompleted event — activity aborted by hook governance"
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

                # ═══ Enforce ActivityCompleted governance verdict ═══
                if completed_verdict:
                    try:
                        completed_result = enforce_verdict(completed_verdict, "activity_end")
                        if (
                            completed_result.requires_hitl
                            and not should_skip_hitl(
                                info.activity_type,
                                hitl_enabled=self._config.hitl_enabled,
                                skip_types=self._config.skip_hitl_activity_types,
                            )
                        ):
                            buffer = self._span_processor.get_buffer(info.workflow_id)
                            if buffer:
                                buffer.pending_approval = True
                                activity.logger.info(
                                    f"Pending approval stored (post-execution): workflow_id={info.workflow_id}, run_id={info.workflow_run_id}"
                                )
                            raise_approval_pending(
                                f"Approval required for output: {completed_verdict.reason or 'Activity output requires human approval'}"
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
                        activity.logger.info(f"Guardrails output validation failed: {e}")
                        raise ApplicationError(
                            f"Guardrails validation failed: {e}",
                            type="GuardrailsValidationFailed",
                            non_retryable=True,
                        )

                # Apply output redaction if governance returned guardrails_result for activity_output
                if (
                    completed_verdict
                    and completed_verdict.guardrails_result
                    and completed_verdict.guardrails_result.input_type == "activity_output"
                ):
                    redacted_output = completed_verdict.guardrails_result.redacted_input
                    activity.logger.info(f"Applying guardrails redaction to activity output")

                    if redacted_output is not None:
                        # If result is a dataclass, update fields in place
                        if is_dataclass(result) and not isinstance(result, type) and isinstance(redacted_output, dict):
                            _deep_update_dataclass(result, redacted_output)
                            activity.logger.info(f"Updated {type(result).__name__} output fields with redacted values")
                        else:
                            # Replace result directly (dict, primitive, etc.)
                            result = redacted_output
                            activity.logger.info(f"Replaced activity output with redacted value")

        return result

    async def _send_activity_event(self, info, event_type: str, **extra) -> Optional[GovernanceVerdictResponse]:
        """Send activity event via GovernanceClient.

        Builds Temporal-specific payload (using activity.info() fields) and
        delegates HTTP to self._client.evaluate_event().

        Returns:
            GovernanceVerdictResponse on success, None on fail-open error.
        """
        # Serialize extra fields to ensure no Payload objects slip through
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

