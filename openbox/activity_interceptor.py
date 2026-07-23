"""
Temporal activity interceptor for activity-boundary governance.

ActivityGovernanceInterceptor: Factory that creates ActivityInboundInterceptor

Captures 2 activity-level events:
4. ActivityStarted (execute_activity entry)
5. ActivityCompleted (execute_activity exit)

IMPORTANT: Activities CAN use datetime/time and make HTTP calls directly.
This is different from workflow interceptors which must maintain determinism.
"""

import dataclasses
import json
import time
from dataclasses import asdict, fields, is_dataclass
from typing import Any, List, NoReturn, Optional

from .types import rfc3339_now as _rfc3339_now


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


from opentelemetry import trace
from temporalio import activity
from temporalio.worker import (
    ActivityInboundInterceptor,
    ExecuteActivityInput,
    Interceptor,
)

from .activities import _terminate_workflow_for_halt
from .client import GovernanceClient
from .config import GovernanceConfig
from .core_adapter import core_activity_scope, get_core_context_store
from .errors import (
    GOVERNANCE_PATCH_ERROR_TYPE,
    GovernanceBlockedError,
    GovernanceHaltError,
    GuardrailsValidationError,
)
from .governance_state import TemporalGovernanceState
from .multi_agent import read_session_from_header
from .patch import PatchRequest, patch_request
from .types import GovernanceVerdictResponse, Verdict, WorkflowEventType
from .verdict_handler import enforce_verdict


def _raise_patch(req: PatchRequest) -> NoReturn:
    """Raise the stable, versioned ``GovernancePatch`` ApplicationError for a
    valid BLOCK-with-patch directive. Non-retryable — the workflow interceptor
    catches this stable type and Continue-As-News with the replacement input."""
    from temporalio.exceptions import ApplicationError

    raise ApplicationError(
        "Governance requested workflow restart",
        req.to_dict(),
        type=GOVERNANCE_PATCH_ERROR_TYPE,
        non_retryable=True,
    )


def _serialize_value(value: Any) -> Any:
    """Convert a value to JSON-serializable format."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return _serialize_bytes(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump(mode="json")
        except Exception:
            pass
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
        state: TemporalGovernanceState,
        config: Optional[GovernanceConfig] = None,
        client: Optional[GovernanceClient] = None,
    ):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.state = state
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
            self.state,
            self.config,
            self._client,
        )


class _ActivityInterceptor(ActivityInboundInterceptor):
    def __init__(
        self,
        next_interceptor: ActivityInboundInterceptor,
        api_url: str,
        api_key: str,
        state: TemporalGovernanceState,
        config: GovernanceConfig,
        client: Optional[GovernanceClient] = None,
    ):
        super().__init__(next_interceptor)
        self._api_url = api_url
        self._api_key = api_key
        self._state = state
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

        if info.activity_type in self._config.skip_activity_types:
            return await self.next.execute_activity(input)

        # Multi-agent session id from the header stamped by the workflow outbound
        # interceptor. Request-local — never stored on self / a module global, since
        # one worker serves many sessions concurrently.
        session_id = read_session_from_header(
            input.headers, activity.payload_converter()
        )

        await self._check_pending_verdicts(info)

        await self._check_pending_approval(info)

        # Clear any stale within-activity abort flag from a prior run reusing the
        # same activity key (base ContextStore abort is not run-scoped).
        get_core_context_store().clear_activity_aborted(
            info.workflow_id, info.activity_id
        )

        activity_input = self._serialize_input(input, info)

        governance_verdict: Optional[GovernanceVerdictResponse] = None
        if self._config.send_activity_start_event:
            governance_verdict = await self._send_activity_event(
                info,
                WorkflowEventType.ACTIVITY_STARTED.value,
                multi_agent_session_id=session_id,
                activity_input=activity_input,
            )

        if governance_verdict:
            await self._enforce_verdict(governance_verdict, info, "activity_start")

        activity_input = self._apply_input_redaction(
            governance_verdict, input, activity_input
        )

        try:
            result, status, error, activity_output, end_time = await self._run_activity(
                input, info, activity_input=activity_input, session_id=session_id
            )
        except Exception:
            # The activity (or a hook) raised, so _handle_completion is skipped. A
            # completed-hook stop recorded during user code must still be enforced:
            # HALT reaches the terminate path and raises GovernanceHalt, and a
            # patch request raises GovernancePatch — either REPLACES the original
            # exception (completed-hook priority HALT > patch > original
            # exception). A plain completed BLOCK is a no-op, so the
            # `raise` below re-propagates the original exception unchanged. Consume
            # the completed-stop either way (also clears it, so it can never
            # strand/leak on the exception path).
            await self._consume_completed_halt(info)
            raise

        result = await self._handle_completion(
            info,
            status,
            error,
            start_time,
            end_time,
            activity_input,
            activity_output,
            result,
            session_id,
        )

        return result

    async def _consume_completed_halt(self, info) -> None:
        """Exception-path safety net: a completed-hook stop recorded during user
        code must still be enforced even though the activity itself raised, so
        _handle_completion (the success path) never runs. Priority: HALT raises
        GovernanceHalt (terminating the workflow and REPLACING the original
        exception); a patch request raises GovernancePatch (also replacing the
        original exception with the restart request); a plain completed BLOCK is
        a no-op here, so the caller's re-raise propagates the original exception
        unchanged. Clears the completed-stop and the base abort flag so nothing
        strands."""
        stop = self._state.take_completed_stop(
            info.workflow_id, info.workflow_run_id, info.activity_id
        )
        get_core_context_store().clear_activity_aborted(
            info.workflow_id, info.activity_id
        )
        if stop is None:
            return
        if stop.verdict is Verdict.HALT:
            await _terminate_workflow_for_halt(
                info.workflow_id, stop.reason or "Governance halt"
            )
            return
        if stop.request is not None:
            _raise_patch(stop.request)

    async def _check_pending_verdicts(self, info) -> None:
        """Enforce a SignalReceived BLOCK/HALT recorded for this run by the
        workflow interceptor. Run-scoped: a stale verdict from a prior run with
        the same workflow_id is ignored and cleared inside the state lookup."""
        entry = self._state.get_signal_verdict(info.workflow_id, info.workflow_run_id)
        if entry is None:
            return
        verdict, reason = entry
        if verdict.should_stop():
            await self._enforce_stop_verdict(
                verdict,
                reason or "Workflow blocked by governance",
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

    async def _check_pending_approval(self, info) -> bool:
        """Poll for pending HITL approval on retry. Returns True if approved."""
        from .hitl import handle_approval_response, should_skip_hitl

        if should_skip_hitl(
            info.activity_type,
            hitl_enabled=self._config.hitl_enabled,
            skip_types=self._config.skip_hitl_activity_types,
        ):
            return False

        if not self._state.has_pending_approval(
            info.workflow_id, info.workflow_run_id, info.activity_id
        ):
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
            activity.logger.info(f"Approval granted for workflow_id={info.workflow_id}")
            self._state.clear_pending_approval(
                info.workflow_id, info.workflow_run_id, info.activity_id
            )
            return True
        return False

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

    async def _enforce_verdict(
        self, verdict_response: GovernanceVerdictResponse, info, context: str
    ) -> None:
        """Enforce a governance verdict (HITL, BLOCK, HALT, guardrails)."""
        from .hitl import raise_approval_pending, should_skip_hitl

        # An exact BLOCK with a valid patch requests a workflow restart and
        # takes priority over the generic BLOCK/HALT/guardrails mapping below —
        # checked before enforce_verdict() so it is never first converted to a
        # plain GovernanceBlockedError. These are REAL activity-lifecycle events
        # (not hooks), so hook_trigger stays the default False.
        event_type = (
            WorkflowEventType.ACTIVITY_STARTED.value
            if context == "activity_start"
            else WorkflowEventType.ACTIVITY_COMPLETED.value
        )
        req = patch_request(verdict_response, event_type=event_type)
        if req is not None:
            _raise_patch(req)

        try:
            verdict_result = enforce_verdict(verdict_response, context)
            if verdict_result.requires_hitl and not should_skip_hitl(
                info.activity_type,
                hitl_enabled=self._config.hitl_enabled,
                skip_types=self._config.skip_hitl_activity_types,
            ):
                self._state.mark_pending_approval(
                    info.workflow_id, info.workflow_run_id, info.activity_id
                )
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

    async def _run_activity(
        self,
        input: ExecuteActivityInput,
        info,
        activity_input=None,
        session_id=None,
    ):
        """Execute the activity inside the base-SDK hook context.

        Base instrumentation fires hooks during user code; a BLOCK/HALT/approval
        verdict is raised as a Temporal-native ApplicationError/ApprovalPending
        BY THE ADAPTER, so it propagates here and fails/retries the activity —
        this interceptor never interprets hook verdicts itself.
        """
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
            trace_id = span.get_span_context().trace_id

            with core_activity_scope(
                info,
                activity_input,
                trace_id=trace_id,
                multi_agent_session_id=session_id,
            ):
                try:
                    result = await self.next.execute_activity(input)
                    activity_output = _serialize_value(result)
                except Exception as e:
                    status = "failed"
                    error = {"type": type(e).__name__, "message": str(e)}
                    raise

        end_time = time.time()
        return result, status, error, activity_output, end_time

    async def _handle_completion(
        self,
        info,
        status,
        error,
        start_time,
        end_time,
        activity_input,
        activity_output,
        result,
        multi_agent_session_id=None,
    ) -> Any:
        """Send ActivityCompleted, enforce verdict, apply output redaction."""
        store = get_core_context_store()

        # Completed-hook stop recorded run-scoped by the adapter (BLOCK/HALT), plus
        # the base within-activity abort flag (a started-hook BLOCK the user code
        # swallowed). Either means the operation was governed-stopped.
        completed_stop = self._state.take_completed_stop(
            info.workflow_id, info.workflow_run_id, info.activity_id
        )
        base_aborted = store.is_activity_aborted(info.workflow_id, info.activity_id)
        store.clear_activity_aborted(info.workflow_id, info.activity_id)

        if completed_stop is not None:
            if completed_stop.verdict is Verdict.HALT:
                await _terminate_workflow_for_halt(
                    info.workflow_id, completed_stop.reason or "Governance halt"
                )
            elif completed_stop.request is not None:
                # A patch request outranks the plain skip-completed-event
                # behavior below — raise the restart request now that user code
                # has already returned.
                _raise_patch(completed_stop.request)

        was_aborted = base_aborted or completed_stop is not None

        completed_verdict = None
        if was_aborted:
            activity.logger.info(
                "Skipping ActivityCompleted event — operation aborted by hook governance"
            )
        else:
            completed_verdict = await self._send_activity_event(
                info,
                WorkflowEventType.ACTIVITY_COMPLETED.value,
                multi_agent_session_id=multi_agent_session_id,
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

        if completed_verdict:
            await self._enforce_verdict(completed_verdict, info, "activity_end")

        return self._apply_output_redaction(completed_verdict, result)

    async def _send_activity_event(
        self, info, event_type: str, multi_agent_session_id=None, **extra
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
            # App-supplied multi-agent session id, propagated from the workflow
            # header. Omitted entirely when absent (never a null key).
            **(
                {"multi_agent_session_id": multi_agent_session_id}
                if multi_agent_session_id
                else {}
            ),
            **serialized_extra,
        }

        try:
            json.dumps(payload)
        except TypeError as e:
            activity.logger.warning(f"Payload not JSON serializable, cleaning: {e}")
            payload = json.loads(json.dumps(payload, default=str))

        return await self._client.evaluate_event(payload)
