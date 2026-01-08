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


def _rfc3339_now() -> str:
    """Return current UTC time in RFC3339 format (UTC+0)."""
    # Lazy import to avoid Temporal sandbox restrictions
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'


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
from .types import WorkflowEventType, WorkflowSpanBuffer, GovernanceVerdictResponse


def _serialize_value(value: Any) -> Any:
    """Convert a value to JSON-serializable format."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool, int)):
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
    ):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.span_processor = span_processor
        self.config = config or GovernanceConfig()

    def intercept_activity(
        self, next_interceptor: ActivityInboundInterceptor
    ) -> ActivityInboundInterceptor:
        return _ActivityInterceptor(
            next_interceptor,
            self.api_url,
            self.api_key,
            self.span_processor,
            self.config,
        )


class _ActivityInterceptor(ActivityInboundInterceptor):
    def __init__(
        self,
        next_interceptor: ActivityInboundInterceptor,
        api_url: str,
        api_key: str,
        span_processor: WorkflowSpanProcessor,
        config: GovernanceConfig,
    ):
        super().__init__(next_interceptor)
        self._api_url = api_url
        self._api_key = api_key
        self._span_processor = span_processor
        self._config = config

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

        if pending_verdict and pending_verdict.get("verdict") == "stop":
            from temporalio.exceptions import ApplicationError
            reason = pending_verdict.get("reason") or "Workflow blocked by governance"
            activity.logger.info(f"Activity blocked by prior governance verdict (from signal): {reason}")
            raise ApplicationError(
                f"Governance blocked: {reason}",
                type="GovernanceStop",
                non_retryable=True,
            )

        if buffer and buffer.verdict == "stop":
            from temporalio.exceptions import ApplicationError
            reason = buffer.verdict_reason or "Workflow blocked by governance"
            activity.logger.info(f"Activity blocked by prior governance verdict (from buffer): {reason}")
            raise ApplicationError(
                f"Governance blocked: {reason}",
                type="GovernanceStop",
                non_retryable=True,
            )

        # ═══ Check for pending approval on retry ═══
        # If there's a pending approval, poll OpenBox Core for status
        approval_granted = False
        if self._config.hitl_enabled and info.activity_type not in self._config.skip_hitl_activity_types:
            buffer = self._span_processor.get_buffer(info.workflow_id)
            if buffer and buffer.pending_approval_governance_event_id:
                governance_event_id = buffer.pending_approval_governance_event_id
                activity.logger.info(f"Polling approval status for governance_event_id={governance_event_id}")

                # Poll OpenBox Core for approval status
                approval_response = await self._poll_approval_status(governance_event_id)

                if approval_response:
                    from temporalio.exceptions import ApplicationError
                    action = approval_response.get("action")

                    if action == "continue":
                        # Approved - clear pending and proceed
                        activity.logger.info(f"Approval granted for governance_event_id={governance_event_id}")
                        buffer.pending_approval_governance_event_id = None
                        approval_granted = True
                    elif action == "stop":
                        # Rejected - clear pending and fail
                        buffer.pending_approval_governance_event_id = None
                        reason = approval_response.get("reason", "Activity rejected")
                        raise ApplicationError(
                            f"Activity rejected: {reason}",
                            type="ApprovalRejected",
                            non_retryable=True,
                        )
                    else:  # action == "require-approval" (still pending)
                        raise ApplicationError(
                            f"Awaiting approval for activity {info.activity_type}",
                            type="ApprovalPending",
                            non_retryable=False,
                        )
                else:
                    # Failed to poll - raise retryable error
                    raise ApplicationError(
                        f"Failed to check approval status, retrying...",
                        type="ApprovalPending",
                        non_retryable=False,
                    )

        # Ensure buffer is registered for this workflow (needed for span collection)
        # The span processor's on_end() will add spans to this buffer
        if not self._span_processor.get_buffer(info.workflow_id):
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
        # Skip if approval granted - already approved, no need to check governance again
        if self._config.send_activity_start_event and not approval_granted:
            governance_verdict = await self._send_activity_event(
                info,
                WorkflowEventType.ACTIVITY_STARTED.value,
                activity_input=activity_input,
            )

        # If governance returned "stop", fail the activity before it runs
        if governance_verdict and governance_verdict.action == "stop":
            from temporalio.exceptions import ApplicationError
            raise ApplicationError(
                f"Governance blocked: {governance_verdict.reason or 'No reason provided'}",
                type="GovernanceStop",
                non_retryable=True,
            )

        # Check guardrails validation_passed - if False, stop the activity
        if (
            governance_verdict
            and governance_verdict.guardrails_result
            and not governance_verdict.guardrails_result.validation_passed
        ):
            from temporalio.exceptions import ApplicationError
            reasons = governance_verdict.guardrails_result.get_reason_strings()
            reason_str = "; ".join(reasons) if reasons else "Guardrails validation failed"
            activity.logger.info(f"Guardrails validation failed: {reason_str}")
            raise ApplicationError(
                f"Guardrails validation failed: {reason_str}",
                type="GuardrailsValidationFailed",
                non_retryable=True,
            )

        # ═══ Handle require-approval verdict (pre-execution) ═══
        if (
            self._config.hitl_enabled
            and governance_verdict
            and governance_verdict.action in ("require-approval", "request-approval")
            and info.activity_type not in self._config.skip_hitl_activity_types
        ):
            from temporalio.exceptions import ApplicationError

            # Store governance_event_id in span buffer for approval tracking
            # Buffer already has workflow_id and run_id
            buffer = self._span_processor.get_buffer(info.workflow_id)
            if buffer:
                buffer.pending_approval_governance_event_id = governance_verdict.governance_event_id
                activity.logger.info(
                    f"Pending approval stored: governance_event_id={governance_verdict.governance_event_id}, "
                    f"workflow_id={info.workflow_id}, run_id={info.workflow_run_id}"
                )

            # Raise retryable error - Temporal will retry the activity
            raise ApplicationError(
                f"Approval required: {governance_verdict.reason or 'Activity requires human approval'}",
                type="ApprovalPending",
                non_retryable=False,  # Retryable!
            )

        # Debug: Log governance verdict details
        if governance_verdict:
            activity.logger.info(
                f"Governance verdict: action={governance_verdict.action}, "
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
            activity.logger.info(f"Redacted input type: {type(redacted).__name__}, value preview: {str(redacted)[:200]}")

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
                                activity.logger.info(f"After update, prompt = {getattr(original_arg, 'prompt', 'N/A')}")
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
                activity.logger.info(f"BEFORE ACTIVITY EXECUTION - input.args[0].prompt = {first_arg.prompt}")

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
            except Exception as e:
                status = "failed"
                error = {"type": type(e).__name__, "message": str(e)}
                raise
            finally:
                end_time = time.time()

                # Get activity spans from buffer
                # Filter by activity_id (stored in span_data by span_processor)
                buffer = self._span_processor.get_buffer(info.workflow_id)
                spans = []

                # Get pending body data from span processor
                # This data is stored by patched httpx.send but not yet merged to spans
                # (because the activity span hasn't ended yet)
                activity_span_id = span.get_span_context().span_id
                pending_body = self._span_processor.get_pending_body(activity_span_id)

                if buffer:
                    for s in buffer.spans:
                        # Check if this span belongs to this activity
                        if (s.get("activity_id") == info.activity_id
                            or s.get("attributes", {}).get("temporal.activity_id") == info.activity_id):
                            spans.append(s)

                    # If we have pending body/header data, propagate to child HTTP spans
                    if pending_body:
                        for s in spans:
                            if "request_body" not in s and pending_body.get("request_body"):
                                s["request_body"] = pending_body["request_body"]
                            if "response_body" not in s and pending_body.get("response_body"):
                                s["response_body"] = pending_body["response_body"]
                            if "request_headers" not in s and pending_body.get("request_headers"):
                                s["request_headers"] = pending_body["request_headers"]
                            if "response_headers" not in s and pending_body.get("response_headers"):
                                s["response_headers"] = pending_body["response_headers"]

                # Send ActivityCompleted event (with input and output)
                # Skip if approval granted - already approved, no need to check governance again
                completed_verdict = None
                if not approval_granted:
                    completed_verdict = await self._send_activity_event(
                        info,
                        WorkflowEventType.ACTIVITY_COMPLETED.value,
                        status=status,
                        start_time=start_time,
                        end_time=end_time,
                        duration_ms=(end_time - start_time) * 1000,
                        span_count=len(spans),
                        spans=spans,
                        activity_input=activity_input,
                        activity_output=activity_output,
                        error=error,
                    )

                    # If governance returned "stop", fail the activity after it completes
                    if completed_verdict and completed_verdict.action == "stop":
                        from temporalio.exceptions import ApplicationError
                        raise ApplicationError(
                            f"Governance blocked: {completed_verdict.reason or 'No reason provided'}",
                            type="GovernanceStop",
                            non_retryable=True,
                        )

                    # Check guardrails validation_passed for output - if False, stop
                    if (
                        completed_verdict
                        and completed_verdict.guardrails_result
                        and not completed_verdict.guardrails_result.validation_passed
                    ):
                        from temporalio.exceptions import ApplicationError
                        reasons = completed_verdict.guardrails_result.get_reason_strings()
                        reason_str = "; ".join(reasons) if reasons else "Guardrails output validation failed"
                        activity.logger.info(f"Guardrails output validation failed: {reason_str}")
                        raise ApplicationError(
                            f"Guardrails validation failed: {reason_str}",
                            type="GuardrailsValidationFailed",
                            non_retryable=True,
                        )

                    # ═══ Handle require-approval verdict (post-execution) ═══
                    if (
                        self._config.hitl_enabled
                        and completed_verdict
                        and completed_verdict.action in ("require-approval", "request-approval")
                        and info.activity_type not in self._config.skip_hitl_activity_types
                    ):
                        from temporalio.exceptions import ApplicationError

                        # Store governance_event_id in span buffer for approval tracking
                        buffer = self._span_processor.get_buffer(info.workflow_id)
                        if buffer:
                            buffer.pending_approval_governance_event_id = completed_verdict.governance_event_id
                            activity.logger.info(
                                f"Pending approval stored (post-execution): governance_event_id={completed_verdict.governance_event_id}, "
                                f"workflow_id={info.workflow_id}, run_id={info.workflow_run_id}"
                            )

                        # Raise retryable error - Temporal will retry the activity
                        raise ApplicationError(
                            f"Approval required for output: {completed_verdict.reason or 'Activity output requires human approval'}",
                            type="ApprovalPending",
                            non_retryable=False,  # Retryable!
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
                else:
                    activity.logger.info(f"Approval granted - skipping ActivityCompleted governance check")

        return result

    async def _send_activity_event(self, info, event_type: str, **extra) -> Optional[GovernanceVerdictResponse]:
        """Send activity event directly via HTTP (allowed in activity context).

        Returns:
            GovernanceVerdictResponse with action, reason, and optional guardrails_result
            None if API fails and policy is fail_open

        NOTE: httpx and datetime are imported lazily here to avoid loading them
        at module level. Module-level httpx import triggers Temporal sandbox
        restrictions because httpx uses os.stat internally.
        """
        # Lazy imports - only loaded when activity interceptor actually runs
        # (in activity context, not workflow context)
        import httpx

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
            # Fallback: convert entire payload using default=str
            payload = json.loads(json.dumps(payload, default=str))

        try:
            async with httpx.AsyncClient(timeout=self._config.api_timeout) as client:
                response = await client.post(
                    f"{self._api_url}/api/v1/governance/evaluate",
                    json=payload,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )
                # Check for HTTP errors
                if response.status_code >= 400:
                    error_msg = f"HTTP {response.status_code}"
                    activity.logger.warning(f"Governance API error: {error_msg}")

                    # Respect on_api_error policy
                    if self._config.on_api_error == "fail_closed":
                        return GovernanceVerdictResponse(
                            action="stop",
                            reason=f"Governance API error: {error_msg}",
                        )

                    return None  # Fail-open: allow activity to proceed

                # Parse governance response
                if response.status_code == 200:
                    try:
                        data = response.json()
                        # Debug: Log raw response
                        activity.logger.info(f"Raw governance response: {data}")
                        activity.logger.info(f"guardrails_result in response: {'guardrails_result' in data}, value: {data.get('guardrails_result')}")

                        verdict = GovernanceVerdictResponse.from_dict(data)

                        if verdict.action == "stop":
                            activity.logger.info(
                                f"Governance blocked activity: {verdict.reason} (policy: {verdict.policy_id})"
                            )

                        if verdict.guardrails_result:
                            activity.logger.info(
                                f"Guardrails redaction applied: input_type={verdict.guardrails_result.input_type}"
                            )

                        return verdict
                    except Exception as e:
                        activity.logger.warning(f"Failed to parse governance response: {e}")

                return None  # Allow activity to proceed

        except Exception as e:
            error_msg = str(e) if str(e) else repr(e)
            activity.logger.warning(f"Governance API error ({type(e).__name__}): {error_msg}")

            # Respect on_api_error policy
            if self._config.on_api_error == "fail_closed":
                return GovernanceVerdictResponse(
                    action="stop",
                    reason=f"Governance API error: {error_msg}",
                )

            return None  # Fail-open: allow activity to proceed

    async def _poll_approval_status(self, governance_event_id: str) -> Optional[dict]:
        """Poll OpenBox Core for approval status.

        Args:
            governance_event_id: The governance event ID to check

        Returns:
            Dict with action ("continue", "stop", "require-approval") and optional reason
            None if API call fails
        """
        import httpx

        try:
            async with httpx.AsyncClient(timeout=self._config.api_timeout) as client:
                response = await client.get(
                    f"{self._api_url}/api/v1/governance/approval/{governance_event_id}",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )

                if response.status_code == 200:
                    data = response.json()
                    activity.logger.info(f"Approval status response: {data}")
                    return data
                else:
                    activity.logger.warning(f"Failed to get approval status: HTTP {response.status_code}")
                    return None

        except Exception as e:
            activity.logger.warning(f"Failed to poll approval status: {e}")
            return None