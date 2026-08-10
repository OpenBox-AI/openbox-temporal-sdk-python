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

import asyncio
import json
import time
from dataclasses import asdict, fields, is_dataclass
from typing import Any, Literal, NoReturn

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


from datetime import UTC

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
from .core_adapter import get_core_context_store
from .errors import (
    GOVERNANCE_PATCH_ERROR_TYPE,
    GovernanceBlockedError,
    GovernanceHaltError,
    GuardrailsValidationError,
)
from .governance_state import TemporalGovernanceState
from .multi_agent import read_session_from_header
from .patch import PatchRequest, patch_request
from .sandbox.adapter import TemporalSandboxConfig, activity_result
from .sandbox.profiles import CommandResultValidationError
from .sandbox.types import (
    GOVERNED_COMMAND_ACTIVITY_TYPE,
    GovernedCommandInputError,
    GovernedCommandRequest,
    GovernedCommandTypedResult,
)
from .span_processor import WorkflowSpanBuffer, WorkflowSpanProcessor
from .types import (
    GovernanceBlockedError,
    GovernanceVerdictResponse,
    Verdict,
    WorkflowEventType,
)
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


def _extract_trace_context(headers: Any) -> Any:
    """Extract the workflow trace context from the activity task headers.

    The Temporal TracingInterceptor carries the W3C context in its
    ``_tracer-data`` header as a payload-encoded carrier dict; decode it with
    the default payload converter and extract with the tracecontext propagator.
    """
    from opentelemetry.trace.propagation.tracecontext import (
        TraceContextTextMapPropagator,
    )

    if headers is None:
        return None
    try:
        payload = headers.get("_tracer-data")
        if payload is None:
            return None
        from temporalio.converter import DataConverter

        carrier = DataConverter.default.payload_converter.from_payloads([payload])[0]
        if not isinstance(carrier, dict) or not carrier:
            return None
        return TraceContextTextMapPropagator().extract(carrier)
    except Exception:
        return None


class ActivityGovernanceInterceptor(Interceptor):
    """Factory for activity interceptor. Events sent directly (activities can do HTTP)."""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        span_processor: WorkflowSpanProcessor,
        config: GovernanceConfig | None = None,
        client: GovernanceClient | None = None,
        sandbox: TemporalSandboxConfig | None = None,
        state: TemporalGovernanceState | None = None,
    ):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.span_processor = span_processor
        self.config = config or GovernanceConfig()
        self._sandbox = sandbox
        self._state = state or TemporalGovernanceState()
        disconnected_command_worker = sandbox is not None and (
            sandbox.receipt_verifier is not None or sandbox.trust_application_agent
        )
        # Disconnected workers install only the command-dispatch branch. They
        # deliberately have no Core client or usable decision path.
        self._client = (
            None
            if disconnected_command_worker
            else client
            or GovernanceClient(
                api_url=api_url,
                api_key=api_key,
                timeout=self.config.api_timeout,
                on_api_error=self.config.on_api_error,
            )
        )

    def intercept_activity(
        self, next_interceptor: ActivityInboundInterceptor
    ) -> ActivityInboundInterceptor:
        return _ActivityInterceptor(
            next_interceptor,
            self.api_url,
            self.api_key,
            self.span_processor,
            self._state,
            self.config,
            self._client,
            self._sandbox,
        )


class _ActivityInterceptor(ActivityInboundInterceptor):
    def __init__(
        self,
        next_interceptor: ActivityInboundInterceptor,
        api_url: str,
        api_key: str,
        span_processor: WorkflowSpanProcessor,
        state: TemporalGovernanceState,
        config: GovernanceConfig,
        client: GovernanceClient | None = None,
        sandbox: TemporalSandboxConfig | None = None,
    ):
        super().__init__(next_interceptor)
        self._api_url = api_url
        self._api_key = api_key
        self._span_processor = span_processor
        self._state = state
        self._config = config
        self._sandbox = sandbox
        disconnected_command_worker = sandbox is not None and (
            sandbox.receipt_verifier is not None or sandbox.trust_application_agent
        )
        self._client = (
            None
            if disconnected_command_worker
            else client
            or GovernanceClient(
                api_url=api_url,
                api_key=api_key,
                timeout=config.api_timeout,
                on_api_error=config.on_api_error,
            )
        )

    async def execute_activity(self, input: ExecuteActivityInput) -> Any:
        info = activity.info()
        start_time = time.time()

        # Registered commands are intercepted before every skip, HITL, hook,
        # redaction, and downstream-interceptor path.
        if info.activity_type == GOVERNED_COMMAND_ACTIVITY_TYPE:
            return await self._execute_governed_command(input, info, start_time)

        # Disconnected command-only modes bypass Core governance for ordinary
        # activities as well.
        if self._sandbox is not None and (
            self._sandbox.receipt_verifier is not None
            or self._sandbox.trust_application_agent
        ):
            return await self.next.execute_activity(input)

        # Skip if configured (e.g., send_governance_event to avoid loops)
        if info.activity_type in self._config.skip_activity_types:
            return await self.next.execute_activity(input)

        # Check for blocking verdicts from prior governance (signal or buffer)
        await self._check_pending_verdicts(info)

        # Check for pending approval on retry (HITL polling)
        await self._check_pending_approval(info)

        # Multi-agent session id from the header stamped by the workflow outbound
        # interceptor. Request-local — never stored on self / a module global, since
        # one worker serves many sessions concurrently.
        session_id = read_session_from_header(
            input.headers, activity.payload_converter()
        )

        # Temporal's type hints permit absent metadata, but an executing
        # Activity always has these identifiers. Normalize defensively.
        workflow_id = info.workflow_id or ""
        activity_id = info.activity_id or ""
        workflow_run_id = info.workflow_run_id or ""
        workflow_type = info.workflow_type or ""
        task_queue = info.task_queue or ""

        # Clear stale state and register fresh buffer
        self._span_processor.clear_activity_abort(workflow_id, activity_id)
        buffer = WorkflowSpanBuffer(
            workflow_id=workflow_id,
            run_id=workflow_run_id,
            workflow_type=workflow_type,
            task_queue=task_queue,
        )
        self._span_processor.register_workflow(workflow_id, buffer)

        # Serialize activity input
        activity_input = self._serialize_input(input, info)

        # Send ActivityStarted event (optional)
        governance_verdict: GovernanceVerdictResponse | None = None
        if self._config.send_activity_start_event:
            governance_verdict = await self._send_activity_event(
                info,
                WorkflowEventType.ACTIVITY_STARTED.value,
                activity_input=activity_input,
            )

        # Buffer activity context for hook-level governance
        self._span_processor.set_activity_context(
            workflow_id,
            activity_id,
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

        # Send ActivityCompleted + enforce verdict + apply output redaction
        result = await self._handle_completion(
            info,
            status,
            error,
            start_time,
            end_time,
            activity_input,
            activity_output,
            result,
        )

        return result

    async def _execute_governed_command(
        self, input: ExecuteActivityInput, info, start_time: float
    ) -> Any:
        """Execute a registered command without entering ``self.next``."""
        from temporalio.exceptions import ApplicationError

        if self._sandbox is None:
            raise ApplicationError(
                "Governed command support is not configured",
                type="GovernedCommandConfigurationRequired",
                non_retryable=True,
            )
        if info.attempt != 1:
            raise ApplicationError(
                "Governed commands permit exactly one Activity attempt",
                type="GovernedCommandAttemptRejected",
                non_retryable=True,
            )
        try:
            args = list(input.args) if input.args is not None else []
            if len(args) != 1:
                raise GovernedCommandInputError("governed command input rejected")
            request = GovernedCommandRequest.from_value(args[0])
            if self._sandbox.trust_application_agent and request.receipt is not None:
                raise GovernedCommandInputError("governed command input rejected")
            argv = self._sandbox.profiles.derive(request)
            self._sandbox.profiles.profile_fingerprint(request.profile_id)
        except (GovernedCommandInputError, TypeError, ValueError) as error:
            raise ApplicationError(
                "Governed command input rejected",
                type="GovernedCommandInvalid",
                non_retryable=True,
            ) from error

        authorization_id: str | None = None
        if self._sandbox.receipt_verifier is not None:
            try:
                dispatcher_config = self._sandbox.dispatcher._config
                execution_config = dispatcher_config.sandbox
                asset_bundle = execution_config.asset_bundle
                authorization_id = self._sandbox.receipt_verifier.verify(
                    request,
                    expected_workflow_id=info.workflow_id,
                    command_argv=argv,
                    asset_bundle=asset_bundle,
                    profile_fingerprint=self._sandbox.profiles.profile_fingerprint(
                        request.profile_id
                    ),
                )
            except Exception as error:
                raise ApplicationError(
                    "Governed command receipt rejected",
                    type="GovernedCommandUnauthorized",
                    non_retryable=True,
                ) from error

        from openbox_sandbox.dispatcher import (
            Directive,
            Disposition,
            GovernanceDecision,
            GovernedCommand,
        )

        command = GovernedCommand(
            workflow_id=info.workflow_id,
            run_id=info.workflow_run_id,
            activity_id=info.activity_id,
            argv=argv,
            profile_id=request.profile_id,
            timeout_seconds=self._sandbox.timeout_seconds,
            workflow_type=info.workflow_type,
            task_queue=info.task_queue,
            attempt=info.attempt,
            arguments={item.name: item.value for item in request.arguments},
        )
        telemetry_owner = None
        if self._sandbox.otel_bridge is not None:
            try:
                # Join the workflow's W3C trace from the activity task headers
                # (the Temporal TracingInterceptor's "_tracer-data" payload),
                # so the governed span shares the workflow trace even when the
                # tracing interceptor is not outermost in the activity chain.
                from opentelemetry import context as otel_context

                propagated = _extract_trace_context(input.headers)
                token = (
                    otel_context.attach(propagated)
                    if propagated is not None
                    else None
                )
                try:
                    telemetry_owner = self._sandbox.otel_bridge.begin(
                    workflow_id=info.workflow_id,
                    run_id=info.workflow_run_id,
                    activity_id=info.activity_id,
                    attempt=info.attempt,
                    profile_id=request.profile_id,
                        workflow_type=info.workflow_type,
                        task_queue=info.task_queue,
                    )
                finally:
                    if token is not None:
                        otel_context.detach(token)
            except Exception:
                telemetry_owner = None
        dispatch_result = None
        typed_result: GovernedCommandTypedResult | None = None
        terminal_error: BaseException | None = None
        try:
            with self._sandbox.heartbeat_sink.bind(
                activity.heartbeat,
                workflow_id=info.workflow_id,
                run_id=info.workflow_run_id,
                activity_id=info.activity_id,
                attempt=info.attempt,
                profile_id=request.profile_id,
                telemetry_owner=telemetry_owner,
            ):
                heartbeat_task = asyncio.create_task(
                    self._heartbeat_periodically(
                        self._sandbox.heartbeat_interval_seconds
                    )
                )
                if request.governance is not None:
                    decision = GovernanceDecision.parse(request.governance)
                    dispatch_operation = self._sandbox.dispatcher.dispatch_with_decision(
                        command, decision
                    )
                elif (
                    self._sandbox.evaluate_at_interceptor
                    and self._client is not None
                    and request.receipt is None
                ):
                    # Single-client convergence: the plugin's governance
                    # client (wrapping the one shared Core runtime) evaluates
                    # the governed command at activity time; the dispatcher
                    # executes the resulting verdict without a second client.
                    from datetime import datetime

                    now = datetime.now(UTC).isoformat()
                    event = {
                        "source": "governed-dispatcher",
                        "event_type": "ActivityStarted",
                        "workflow_id": info.workflow_id,
                        "run_id": info.workflow_run_id,
                        "workflow_type": info.workflow_type,
                        "task_queue": info.task_queue,
                        "timestamp": now,
                        "activity_id": info.activity_id,
                        "activity_type": GOVERNED_COMMAND_ACTIVITY_TYPE,
                        "attempt": info.attempt,
                        "profile_id": request.profile_id,
                        "activity_input": [{"argv": list(argv)}],
                        "operation": {
                            "profile_id": request.profile_id,
                            "arguments": {
                                item.name: item.value for item in request.arguments
                            },
                        },
                    }
                    response = await self._client.evaluate_event(event)
                    if response is None:
                        raise ApplicationError(
                            "Governed command evaluation unavailable",
                            type="GovernedCommandNotExecuted",
                            non_retryable=True,
                        )
                    import uuid as _uuid

                    decision_value: dict[str, Any] = {
                        "governance_event_id": str(_uuid.uuid4()),
                        "verdict": response.verdict.value,
                        "risk_score": response.risk_score,
                        "action": response.verdict.value,
                        "fallback_used": False,
                        "behavioral_violations": list(
                            response.behavioral_violations or []
                        ),
                    }
                    if response.constraints is not None:
                        decision_value["constraints"] = list(response.constraints)
                    if response.policy_id is not None:
                        decision_value["policy_id"] = response.policy_id
                    if response.reason is not None:
                        decision_value["reason"] = response.reason
                    decision = GovernanceDecision.parse(decision_value)
                    dispatch_operation = self._sandbox.dispatcher.dispatch_with_decision(
                        command, decision
                    )
                elif self._sandbox.trust_application_agent:
                    dispatch_operation = (
                        self._sandbox.dispatcher.dispatch_trusted_constrain(command)
                    )
                elif authorization_id is not None:
                    dispatch_operation = (
                        self._sandbox.dispatcher.dispatch_authorized_constrain(
                            command, authorization_id=authorization_id
                        )
                    )
                else:
                    dispatch_operation = self._sandbox.dispatcher.dispatch(command)
                dispatch_task = asyncio.create_task(dispatch_operation)
                cancellation_task = asyncio.create_task(
                    self._wait_for_temporal_cancellation()
                )
                try:
                    done, _ = await asyncio.wait(
                        {dispatch_task, cancellation_task, heartbeat_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if heartbeat_task in done:
                        # Re-raise the exact heartbeat failure. The surrounding
                        # finally still cancels dispatch and waits for cleanup.
                        await heartbeat_task
                    if cancellation_task in done:
                        dispatch_task.cancel()
                        try:
                            await dispatch_task
                        except asyncio.CancelledError:
                            pass
                        except Exception:
                            # The runtime client maps cancellation to a typed
                            # transport result and the dispatcher performs owned
                            # cleanup before returning or raising.
                            pass
                        raise asyncio.CancelledError()
                    dispatch_result = await dispatch_task
                finally:
                    if not dispatch_task.done():
                        dispatch_task.cancel()
                        try:
                            await dispatch_task
                        except asyncio.CancelledError:
                            pass
                        except Exception:
                            # A typed cancellation/transport result may surface
                            # after the dispatcher has completed owned cleanup.
                            pass
                    heartbeat_task.cancel()
                    cancellation_task.cancel()
                    for task in (heartbeat_task, cancellation_task):
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
            activity.heartbeat(
                {
                    "phase": "governed_dispatch_terminal",
                    "workflow_id": info.workflow_id,
                    "run_id": info.workflow_run_id,
                    "activity_id": info.activity_id,
                    "attempt": info.attempt,
                    "profile_id": request.profile_id,
                    "disposition": dispatch_result.disposition.value,
                }
            )
            if dispatch_result.disposition in (
                Disposition.EXECUTED_ON_HOST,
                Disposition.EXECUTED_IN_SANDBOX,
            ):
                execution = dispatch_result.execution
                if execution is None:
                    raise CommandResultValidationError()
                typed_result = self._sandbox.profiles.parse_result(
                    request.profile_id, execution.stdout
                )
        except asyncio.CancelledError as error:
            terminal_error = error
            try:
                await self._send_registered_completion(
                    info,
                    request,
                    start_time,
                    status="cancelled",
                    dispatch_result=None,
                    error_code="cancelled",
                )
            except BaseException:
                pass
            raise
        except CommandResultValidationError as error:
            terminal_error = error
            await self._send_registered_completion(
                info,
                request,
                start_time,
                status="failed",
                dispatch_result=dispatch_result,
                error_code="typed_result_invalid",
            )
            raise ApplicationError(
                "Governed command typed result rejected",
                type="GovernedCommandResultInvalid",
                non_retryable=True,
            ) from error
        except Exception as error:
            terminal_error = error
            await self._send_registered_completion(
                info,
                request,
                start_time,
                status="failed",
                dispatch_result=None,
                error_code="dispatcher_failure",
            )
            raise ApplicationError(
                "Governed dispatcher failed",
                type="GovernedDispatcherFailure",
                non_retryable=True,
            ) from error
        except BaseException as error:
            terminal_error = error
            raise
        finally:
            if self._sandbox.otel_bridge is not None:
                try:
                    self._sandbox.otel_bridge.finalize(
                        telemetry_owner,
                        dispatch_result=dispatch_result,
                        error=terminal_error,
                    )
                except Exception:
                    pass

        await self._send_registered_completion(
            info,
            request,
            start_time,
            status=(
                "completed"
                if dispatch_result.error is None
                and dispatch_result.disposition
                in (Disposition.EXECUTED_ON_HOST, Disposition.EXECUTED_IN_SANDBOX)
                else "failed"
            ),
            dispatch_result=dispatch_result,
            error_code=(
                None
                if dispatch_result.error is None
                else dispatch_result.error.code.value
            ),
        )
        if dispatch_result.directive is Directive.HALT:
            await _terminate_workflow_for_halt(
                info.workflow_id, "governance halt directive"
            )
        # Fail-closed: a post-execution governance error (e.g. ActivityCompleted
        # transport failure) must not be reported as a successful sandbox run.
        if dispatch_result.error is not None:
            error_code = dispatch_result.error.code.value
            error_type = (
                "GovernedCommandExecutionIndeterminate"
                if dispatch_result.disposition is Disposition.EXECUTION_INDETERMINATE
                else "GovernedCommandNotExecuted"
            )
            raise ApplicationError(
                f"Governed command terminal outcome: {error_code}",
                type=error_type,
                non_retryable=True,
            )
        if dispatch_result.disposition in (
            Disposition.EXECUTED_ON_HOST,
            Disposition.EXECUTED_IN_SANDBOX,
        ):
            return activity_result(
                request.profile_id,
                dispatch_result,
                typed_result=typed_result,
            )
        error_code = (
            "governed_command_not_executed"
            if dispatch_result.error is None
            else dispatch_result.error.code.value
        )
        error_type = (
            "GovernedCommandExecutionIndeterminate"
            if dispatch_result.disposition is Disposition.EXECUTION_INDETERMINATE
            else "GovernedCommandNotExecuted"
        )
        raise ApplicationError(
            f"Governed command terminal outcome: {error_code}",
            type=error_type,
            non_retryable=True,
        )

    @staticmethod

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
        if info.attempt != 1:
            raise ApplicationError(
                "Governed commands permit exactly one Activity attempt",
                type="GovernedCommandAttemptRejected",
                non_retryable=True,
            )
        try:
            args = list(input.args) if input.args is not None else []
            if len(args) != 1:
                raise GovernedCommandInputError("governed command input rejected")
            request = GovernedCommandRequest.from_value(args[0])
            if self._sandbox.trust_application_agent and request.receipt is not None:
                raise GovernedCommandInputError("governed command input rejected")
            argv = self._sandbox.profiles.derive(request)
            self._sandbox.profiles.profile_fingerprint(request.profile_id)
        except (GovernedCommandInputError, TypeError, ValueError) as error:
            raise ApplicationError(
                "Governed command input rejected",
                type="GovernedCommandInvalid",
                non_retryable=True,
            ) from error

        authorization_id: str | None = None
        if self._sandbox.receipt_verifier is not None:
            try:
                dispatcher_config = self._sandbox.dispatcher._config
                execution_config = dispatcher_config.sandbox
                asset_bundle = execution_config.asset_bundle
                authorization_id = self._sandbox.receipt_verifier.verify(
                    request,
                    expected_workflow_id=info.workflow_id,
                    command_argv=argv,
                    asset_bundle=asset_bundle,
                    profile_fingerprint=self._sandbox.profiles.profile_fingerprint(
                        request.profile_id
                    ),
                )
            except Exception as error:
                raise ApplicationError(
                    "Governed command receipt rejected",
                    type="GovernedCommandUnauthorized",
                    non_retryable=True,
                ) from error

        from openbox_sandbox.dispatcher import (
            Directive,
            Disposition,
            GovernanceDecision,
            GovernedCommand,
        )

        command = GovernedCommand(
            workflow_id=info.workflow_id,
            run_id=info.workflow_run_id,
            activity_id=info.activity_id,
            argv=argv,
            profile_id=request.profile_id,
            timeout_seconds=self._sandbox.timeout_seconds,
            workflow_type=info.workflow_type,
            task_queue=info.task_queue,
            attempt=info.attempt,
            arguments={item.name: item.value for item in request.arguments},
        )
        telemetry_owner = None
        if self._sandbox.otel_bridge is not None:
            try:
                # Join the workflow's W3C trace from the activity task headers
                # (the Temporal TracingInterceptor's "_tracer-data" payload),
                # so the governed span shares the workflow trace even when the
                # tracing interceptor is not outermost in the activity chain.
                from opentelemetry import context as otel_context

                propagated = _extract_trace_context(input.headers)
                token = (
                    otel_context.attach(propagated)
                    if propagated is not None
                    else None
                )
                try:
                    telemetry_owner = self._sandbox.otel_bridge.begin(
                    workflow_id=info.workflow_id,
                    run_id=info.workflow_run_id,
                    activity_id=info.activity_id,
                    attempt=info.attempt,
                    profile_id=request.profile_id,
                        workflow_type=info.workflow_type,
                        task_queue=info.task_queue,
                    )
                finally:
                    if token is not None:
                        otel_context.detach(token)
            except Exception:
                telemetry_owner = None
        dispatch_result = None
        typed_result: GovernedCommandTypedResult | None = None
        terminal_error: BaseException | None = None
        try:
            with self._sandbox.heartbeat_sink.bind(
                activity.heartbeat,
                workflow_id=info.workflow_id,
                run_id=info.workflow_run_id,
                activity_id=info.activity_id,
                attempt=info.attempt,
                profile_id=request.profile_id,
                telemetry_owner=telemetry_owner,
            ):
                heartbeat_task = asyncio.create_task(
                    self._heartbeat_periodically(
                        self._sandbox.heartbeat_interval_seconds
                    )
                )
                if request.governance is not None:
                    decision = GovernanceDecision.parse(request.governance)
                    dispatch_operation = self._sandbox.dispatcher.dispatch_with_decision(
                        command, decision
                    )
                elif (
                    self._sandbox.evaluate_at_interceptor
                    and self._client is not None
                    and request.receipt is None
                ):
                    # Single-client convergence: the plugin's governance
                    # client (wrapping the one shared Core runtime) evaluates
                    # the governed command at activity time; the dispatcher
                    # executes the resulting verdict without a second client.
                    from datetime import datetime

                    now = datetime.now(UTC).isoformat()
                    event = {
                        "source": "governed-dispatcher",
                        "event_type": "ActivityStarted",
                        "workflow_id": info.workflow_id,
                        "run_id": info.workflow_run_id,
                        "workflow_type": info.workflow_type,
                        "task_queue": info.task_queue,
                        "timestamp": now,
                        "activity_id": info.activity_id,
                        "activity_type": GOVERNED_COMMAND_ACTIVITY_TYPE,
                        "attempt": info.attempt,
                        "profile_id": request.profile_id,
                        "activity_input": [{"argv": list(argv)}],
                        "operation": {
                            "profile_id": request.profile_id,
                            "arguments": {
                                item.name: item.value for item in request.arguments
                            },
                        },
                    }
                    response = await self._client.evaluate_event(event)
                    if response is None:
                        raise ApplicationError(
                            "Governed command evaluation unavailable",
                            type="GovernedCommandNotExecuted",
                            non_retryable=True,
                        )
                    import uuid as _uuid

                    decision_value: dict[str, Any] = {
                        "governance_event_id": str(_uuid.uuid4()),
                        "verdict": response.verdict.value,
                        "risk_score": response.risk_score,
                        "action": response.verdict.value,
                        "fallback_used": False,
                        "behavioral_violations": list(
                            response.behavioral_violations or []
                        ),
                    }
                    if response.constraints is not None:
                        decision_value["constraints"] = list(response.constraints)
                    if response.policy_id is not None:
                        decision_value["policy_id"] = response.policy_id
                    if response.reason is not None:
                        decision_value["reason"] = response.reason
                    decision = GovernanceDecision.parse(decision_value)
                    dispatch_operation = self._sandbox.dispatcher.dispatch_with_decision(
                        command, decision
                    )
                elif self._sandbox.trust_application_agent:
                    dispatch_operation = (
                        self._sandbox.dispatcher.dispatch_trusted_constrain(command)
                    )
                elif authorization_id is not None:
                    dispatch_operation = (
                        self._sandbox.dispatcher.dispatch_authorized_constrain(
                            command, authorization_id=authorization_id
                        )
                    )
                else:
                    dispatch_operation = self._sandbox.dispatcher.dispatch(command)
                dispatch_task = asyncio.create_task(dispatch_operation)
                cancellation_task = asyncio.create_task(
                    self._wait_for_temporal_cancellation()
                )
                try:
                    done, _ = await asyncio.wait(
                        {dispatch_task, cancellation_task, heartbeat_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if heartbeat_task in done:
                        # Re-raise the exact heartbeat failure. The surrounding
                        # finally still cancels dispatch and waits for cleanup.
                        await heartbeat_task
                    if cancellation_task in done:
                        dispatch_task.cancel()
                        try:
                            await dispatch_task
                        except asyncio.CancelledError:
                            pass
                        except Exception:
                            # The runtime client maps cancellation to a typed
                            # transport result and the dispatcher performs owned
                            # cleanup before returning or raising.
                            pass
                        raise asyncio.CancelledError()
                    dispatch_result = await dispatch_task
                finally:
                    if not dispatch_task.done():
                        dispatch_task.cancel()
                        try:
                            await dispatch_task
                        except asyncio.CancelledError:
                            pass
                        except Exception:
                            # A typed cancellation/transport result may surface
                            # after the dispatcher has completed owned cleanup.
                            pass
                    heartbeat_task.cancel()
                    cancellation_task.cancel()
                    for task in (heartbeat_task, cancellation_task):
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
            activity.heartbeat(
                {
                    "phase": "governed_dispatch_terminal",
                    "workflow_id": info.workflow_id,
                    "run_id": info.workflow_run_id,
                    "activity_id": info.activity_id,
                    "attempt": info.attempt,
                    "profile_id": request.profile_id,
                    "disposition": dispatch_result.disposition.value,
                }
            )
            if dispatch_result.disposition in (
                Disposition.EXECUTED_ON_HOST,
                Disposition.EXECUTED_IN_SANDBOX,
            ):
                execution = dispatch_result.execution
                if execution is None:
                    raise CommandResultValidationError()
                typed_result = self._sandbox.profiles.parse_result(
                    request.profile_id, execution.stdout
                )
        except asyncio.CancelledError as error:
            terminal_error = error
            try:
                await self._send_registered_completion(
                    info,
                    request,
                    start_time,
                    status="cancelled",
                    dispatch_result=None,
                    error_code="cancelled",
                )
            except BaseException:
                pass
            raise
        except CommandResultValidationError as error:
            terminal_error = error
            await self._send_registered_completion(
                info,
                request,
                start_time,
                status="failed",
                dispatch_result=dispatch_result,
                error_code="typed_result_invalid",
            )
            raise ApplicationError(
                "Governed command typed result rejected",
                type="GovernedCommandResultInvalid",
                non_retryable=True,
            ) from error
        except Exception as error:
            terminal_error = error
            await self._send_registered_completion(
                info,
                request,
                start_time,
                status="failed",
                dispatch_result=None,
                error_code="dispatcher_failure",
            )
            raise ApplicationError(
                "Governed dispatcher failed",
                type="GovernedDispatcherFailure",
                non_retryable=True,
            ) from error
        except BaseException as error:
            terminal_error = error
            raise
        finally:
            if self._sandbox.otel_bridge is not None:
                try:
                    self._sandbox.otel_bridge.finalize(
                        telemetry_owner,
                        dispatch_result=dispatch_result,
                        error=terminal_error,
                    )
                except Exception:
                    pass

        await self._send_registered_completion(
            info,
            request,
            start_time,
            status=(
                "completed"
                if dispatch_result.error is None
                and dispatch_result.disposition
                in (Disposition.EXECUTED_ON_HOST, Disposition.EXECUTED_IN_SANDBOX)
                else "failed"
            ),
            dispatch_result=dispatch_result,
            error_code=(
                None
                if dispatch_result.error is None
                else dispatch_result.error.code.value
            ),
        )
        if dispatch_result.directive is Directive.HALT:
            await _terminate_workflow_for_halt(
                info.workflow_id, "governance halt directive"
            )
        # Fail-closed: a post-execution governance error (e.g. ActivityCompleted
        # transport failure) must not be reported as a successful sandbox run.
        if dispatch_result.error is not None:
            error_code = dispatch_result.error.code.value
            error_type = (
                "GovernedCommandExecutionIndeterminate"
                if dispatch_result.disposition is Disposition.EXECUTION_INDETERMINATE
                else "GovernedCommandNotExecuted"
            )
            raise ApplicationError(
                f"Governed command terminal outcome: {error_code}",
                type=error_type,
                non_retryable=True,
            )
        if dispatch_result.disposition in (
            Disposition.EXECUTED_ON_HOST,
            Disposition.EXECUTED_IN_SANDBOX,
        ):
            return activity_result(
                request.profile_id,
                dispatch_result,
                typed_result=typed_result,
            )
        error_code = (
            "governed_command_not_executed"
            if dispatch_result.error is None
            else dispatch_result.error.code.value
        )
        error_type = (
            "GovernedCommandExecutionIndeterminate"
            if dispatch_result.disposition is Disposition.EXECUTION_INDETERMINATE
            else "GovernedCommandNotExecuted"
        )
        raise ApplicationError(
            f"Governed command terminal outcome: {error_code}",
            type=error_type,
            non_retryable=True,
        )

    @staticmethod
    async def _wait_for_temporal_cancellation() -> None:
        waiting = activity.wait_for_cancelled()
        if hasattr(waiting, "__await__"):
            await waiting
        else:
            # Unit tests replace the activity module with a plain mock. A real
            # Temporal Activity always returns an awaitable here.
            await asyncio.Future()

    async def _heartbeat_periodically(self, interval_seconds: float) -> None:
        assert self._sandbox is not None
        while True:
            await asyncio.sleep(interval_seconds)
            self._sandbox.heartbeat_sink.heartbeat_latest()

    async def _send_registered_completion(
        self,
        info,
        request: GovernedCommandRequest,
        start_time: float,
        *,
        status: str,
        dispatch_result: Any,
        error_code: str | None,
    ) -> None:
        """Post ActivityCompleted when the dispatcher did not already own it.

        Connected natural sandbox success path: GovernedDispatcher posts the
        completed sandbox hook and span-free ActivityCompleted with the original
        signer — the Temporal wrapper must not duplicate those calls.

        Host allow path and other non-sandbox terminals still need this wrapper
        to post ActivityCompleted, because `_dispatch_host` only emits local
        telemetry.
        """
        if self._sandbox is None or not self._sandbox.completion_events:
            return
        dispatcher_config = getattr(self._sandbox.dispatcher, "_config", None)
        disposition_value = getattr(
            getattr(dispatch_result, "disposition", None), "value", None
        )
        if (
            not self._sandbox.trust_application_agent
            and self._sandbox.receipt_verifier is None
            and getattr(dispatcher_config, "governance", None) is not None
            and dispatch_result is not None
            and disposition_value == "executed_in_sandbox"
            and getattr(dispatch_result, "error", None) is None
        ):
            return
        output = None
        if dispatch_result is not None:
            execution = dispatch_result.execution
            output = {
                "profile_id": request.profile_id,
                "disposition": dispatch_result.disposition.value,
                "directive": dispatch_result.directive.value,
                "exit_code": None if execution is None else execution.exit_code,
                "timeout_status": (
                    None if execution is None else execution.timeout_status.value
                ),
                "cleanup_status": (
                    None if execution is None else execution.cleanup_status.value
                ),
                "stdout_bytes": 0 if execution is None else len(execution.stdout),
                "stderr_bytes": 0 if execution is None else len(execution.stderr),
            }
        end_time = time.time()
        try:
            await self._send_activity_event(
                info,
                WorkflowEventType.ACTIVITY_COMPLETED.value,
                status=status,
                start_time=start_time,
                end_time=end_time,
                duration_ms=(end_time - start_time) * 1000,
                span_count=0,
                spans=[],
                activity_input=[{"profile_id": request.profile_id}],
                activity_output=output,
                error=(
                    None
                    if error_code is None
                    else {"type": error_code, "non_retryable": True}
                ),
            )
        except Exception:
            activity.logger.warning("Governed command completion telemetry failed")

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
        assert self._client is not None
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
        self,
        verdict_response: GovernanceVerdictResponse,
        info,
        context: Literal["activity_start", "activity_end", "workflow_event"],
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
            if verdict_response.verdict == Verdict.CONSTRAIN:
                from temporalio.exceptions import ApplicationError

                raise ApplicationError(
                    "CONSTRAIN is supported only by a registered governed command",
                    type="GovernanceConstrainUnsupported",
                    non_retryable=True,
                )
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
        verdict: GovernanceVerdictResponse | None,
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
        self, verdict: GovernanceVerdictResponse | None, result: Any
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

    async def _run_activity(
        self,
        input: ExecuteActivityInput,
        info,
        activity_input=None,
        session_id=None,
    ):
        """Execute the activity inside the base-SDK hook context."""

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
        from temporalio.exceptions import ApplicationError

        from .hitl import raise_approval_pending, should_skip_hitl

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
        self,
        info,
        status,
        error,
        start_time,
        end_time,
        activity_input,
        activity_output,
        result,
    ) -> Any:
        """Send ActivityCompleted, enforce verdict, apply output redaction."""
        # Completed-hook stop recorded run-scoped by the adapter (BLOCK/HALT), plus
        # the base within-activity abort flag (a started-hook BLOCK the user code
        # swallowed). Either means the operation was governed-stopped.
        completed_stop = self._state.take_completed_stop(
            info.workflow_id, info.workflow_run_id, info.activity_id
        )
        store = get_core_context_store()
        base_aborted = store.is_activity_aborted(
            info.workflow_id, info.activity_id
        )
        store.clear_activity_aborted(info.workflow_id, info.activity_id)

        halt_reason = self._span_processor.get_halt_requested(
            info.workflow_id, info.activity_id
        )
        if halt_reason:
            self._span_processor.clear_halt_requested(
                info.workflow_id, info.activity_id
            )
            await _terminate_workflow_for_halt(info.workflow_id, halt_reason)

        # Cleanup
        self._span_processor.clear_activity_abort(info.workflow_id, info.activity_id)
        self._span_processor.clear_activity_context(info.workflow_id, info.activity_id)
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
    ) -> GovernanceVerdictResponse | None:
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

        if self._client is None:
            return None
        return await self._client.evaluate_event(payload)
