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

import asyncio
import json
from dataclasses import asdict, is_dataclass
from datetime import timedelta
from typing import Any

from openbox_core.contracts import events as _events
from temporalio import workflow
from temporalio.exceptions import ApplicationError
from temporalio.worker import (
    ExecuteWorkflowInput,
    HandleSignalInput,
    Interceptor,
    StartActivityInput,
    StartLocalActivityInput,
    WorkflowInboundInterceptor,
    WorkflowInterceptorClassInput,
    WorkflowOutboundInterceptor,
)

from .errors import (
    GOVERNANCE_API_ERROR_TYPE,
    GOVERNANCE_BLOCK_ERROR_TYPE,
    GOVERNANCE_HALT_ERROR_TYPE,
    GOVERNANCE_PATCH_ERROR_TYPE,
    GOVERNANCE_RETRYABLE_BLOCK_ERROR_TYPE,
    GOVERNANCE_STOP_ERROR_TYPE,
    GovernanceHaltError,
)
from .multi_agent import inject_session_header, read_session_from_memo
from .patch import PatchRequest, extract_patch_request
from .patch_coordinator import (
    PatchControl,
    PatchCoordinator,
    bind_coordinator,
    next_restart_memo,
    unbind_coordinator,
)
from .types import Verdict

# Temporal patch marker (workflow.patched()) gating ALL BLOCK-with-patch restart
# workflow behavior (coordinator wrap + Continue-As-New). This identifier keeps
# its pre-rename name deliberately: it is Temporal's OWN patched-version marker,
# a distinct concept from the governance "patch" directive, and renaming it to
# use "patch" terminology would conflate the two. An old history predating this
# marker replays the exact pre-feature control flow (patched(...) → False),
# taking no new branch.
_RETRYABLE_BLOCK_PATCH = "openbox-retryable-block-v1"

# Patch marker gating inclusion of the workflow arguments in the WorkflowStarted
# payload (as ``activity_input``). This changes the send_governance_event activity
# input, so a history predating this marker replays the exact pre-feature payload
# (patched(...) → False) and never attaches the field — replay stays deterministic.
_WORKFLOW_START_INPUT_PATCH = "openbox-workflow-start-input-v1"


def _application_error_type(exc: BaseException) -> str | None:
    """Walk exception chain and return the ApplicationError.type if present.

    Temporal wraps activity failures as ActivityError(cause=ApplicationError).
    We walk cause/__cause__/__context__ to find the first ApplicationError and
    return its `type` field. Matching on this field is stable across message
    reformatting, locale changes, and nested wrapping.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
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


def _safe_error_type(exc) -> str | None:
    """Extract error type string from an exception, sanitized for JSON."""
    t = getattr(exc, "type", None)
    if isinstance(t, str) and len(t) < 200:
        return t
    return None


def _extract_cause_info(exc) -> dict | None:
    """Extract cause info dict from an exception's cause chain."""
    cause = (
        getattr(exc, "cause", None)
        or getattr(exc, "__cause__", None)
        or getattr(exc, "__context__", None)
    )
    if not cause:
        return None

    info = {"type": type(cause).__name__, "message": str(cause)[:500]}
    cause_type = _safe_error_type(cause)
    if cause_type:
        info["error_type"] = cause_type
    if hasattr(cause, "non_retryable"):
        info["non_retryable"] = cause.non_retryable
    return info


def _extract_root_cause_info(exc) -> dict | None:
    """Extract root cause info from an exception's deeper cause chain."""
    cause = (
        getattr(exc, "cause", None)
        or getattr(exc, "__cause__", None)
        or getattr(exc, "__context__", None)
    )
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
    try:
        return json.loads(json.dumps(value, default=str))
    except Exception:
        return str(value)



def _legacy_block_degrade(payload: dict) -> dict | None:
    """Pre-feature BLOCK result shape, keyed by event origin.

    A BLOCK with patch degrades HERE — when the caller has the feature disabled
    (an old unpatched history) OR its envelope failed extraction — never to ALLOW.
    Signals read ``result.get(...)`` so they need a dict that still BLOCKS;
    lifecycle/handoff legacy mapping is ``None`` (== the pre-feature
    GovernanceBlock→None conversion).
    """
    if payload.get("event_type") == "SignalReceived":
        return {"success": True, "verdict": "block", "reason": "Governance blocked"}
    return None


async def _send_governance_event(
    payload: dict,
    timeout: float,
    on_api_error: str = "fail_open",
    *,
    patch_enabled: bool = False,
) -> Any:
    """
    Send governance event via activity.

    Args:
        on_api_error: "fail_open" (default) = continue on error
                      "fail_closed" = halt workflow if governance API fails
        patch_enabled: when True (set by callers running inside the
            patch-restart-patched path), a raised ``GovernancePatch`` (or the
            legacy ``GovernanceRetryableBlock`` alias, still possible on an
            in-flight restart chain started before this rename) is converted to
            a ``PatchRequest`` return value; when False (legacy/unpatched
            callers) it degrades to the exact pre-feature result shape by
            event_type, so an old open history never receives the new type or
            crashes.

    Returns ``dict | None`` for the existing verdict paths, or a
    ``PatchRequest`` when a valid BLOCK with patch is surfaced on an enabled
    caller. Callers disambiguate with ``isinstance(result, PatchRequest)``.

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

        if app_error_type in (
            GOVERNANCE_HALT_ERROR_TYPE,
            GOVERNANCE_STOP_ERROR_TYPE,
        ):
            raise GovernanceHaltError(str(e)) from None

        if app_error_type in (
            GOVERNANCE_PATCH_ERROR_TYPE,
            GOVERNANCE_RETRYABLE_BLOCK_ERROR_TYPE,
        ):
            if patch_enabled:
                req = extract_patch_request(e)
                if req is not None:
                    return req
                # Extraction failed (missing/invalid schema / malformed details).
                # The verdict was still BLOCK, so fail safe as plain BLOCK via the
                # same event-specific degrade — NOT None, which a patched signal
                # would read as ALLOW and run the user handler.
            return _legacy_block_degrade(payload)

        if app_error_type == GOVERNANCE_BLOCK_ERROR_TYPE:
            return None

        if app_error_type == GOVERNANCE_API_ERROR_TYPE:
            raise GovernanceHaltError(str(e)) from None

        return None


def _is_halt(exc: BaseException | None) -> bool:
    """True when the exception chain carries a governance HALT.

    Walks cause / __cause__ / __context__ (mirrors ``_application_error_type``)
    and matches a typed ``GovernanceHaltError`` or an ``ApplicationError`` whose
    ``type`` is HALT/STOP. A ``None`` exception → False. Used for HALT dominance
    over a concurrent coordinator patch request (priority: HALT > BLOCK with patch).
    """
    if exc is None:
        return False
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, GovernanceHaltError):
            return True
        if isinstance(current, ApplicationError) and getattr(current, "type", None) in (
            GOVERNANCE_HALT_ERROR_TYPE,
            GOVERNANCE_STOP_ERROR_TYPE,
        ):
            return True
        current = (
            getattr(current, "cause", None)
            or getattr(current, "__cause__", None)
            or getattr(current, "__context__", None)
        )
    return False


def _dispose_user_task(user_task: "asyncio.Task") -> None:
    """Retire the user workflow task when the coordinator wins and the run is about
    to be replaced by Continue-As-New.

    If the task already finished, retrieve its exception so asyncio does not warn
    about an unretrieved exception we are intentionally superseding; otherwise
    cancel it (best-effort, deterministic). Continue-As-New ends the run
    regardless, so this is cleanliness, not correctness.
    """
    if user_task.done():
        if not user_task.cancelled():
            user_task.exception()  # mark retrieved (returns exception or None)
    else:
        user_task.cancel()


async def _continue_as_new(
    req: PatchRequest, current_args: list, max_restarts: int
) -> Any:
    """The ONLY place that calls ``workflow.continue_as_new``.

    Enforces the restart budget first (raises ``GovernancePatchLimitExceeded`` at
    the cap), then maps ``new_input``: ``None`` reuses the current run's args
    exactly; any other value is passed as ONE workflow argument.
    ``continue_as_new`` raises ``ContinueAsNewError`` (control flow) — never caught
    or suppressed by the interceptor.
    """
    memo = next_restart_memo(max_restarts)
    if req.new_input is None:
        workflow.continue_as_new(args=current_args, memo=memo)
    else:
        workflow.continue_as_new(req.new_input, memo=memo)


def _workflow_started_payload(
    info, sid: str | None, input: ExecuteWorkflowInput
) -> dict:
    """Build the WorkflowStarted governance payload, shared by both execute paths.

    Under the workflow-start-input patch (new histories), the Temporal workflow
    arguments are serialized with the sandbox-safe ``_serialize_value`` and
    attached as ``activity_input`` so OpenBox stores/displays them through the
    generic input column. The outer argument list is preserved verbatim:
    ``workflow(a, b)`` → ``[a, b]`` and ``workflow([1, 2])`` → ``[[1, 2]]``; no
    arguments → ``[]``. ``input.args`` is copied, never mutated.

    Histories predating the patch reproduce the exact pre-feature payload with no
    ``activity_input`` key, keeping the send_governance_event activity input
    replay-deterministic.
    """
    extra = None
    if workflow.patched(_WORKFLOW_START_INPUT_PATCH):
        args = list(input.args) if input.args is not None else []
        extra = {"activity_input": _serialize_value(args)}
    return _events.workflow_started(
        workflow_id=info.workflow_id,
        run_id=info.run_id,
        workflow_type=info.workflow_type,
        task_queue=info.task_queue,
        multi_agent_session_id=sid,
        extra=extra,
    ).to_payload_dict()


class GovernanceInterceptor(Interceptor):
    """Factory for workflow interceptor. Events sent via activity for determinism."""

    def __init__(
        self,
        api_url: str = "",
        api_key: str = "",
        state=None,  # TemporalGovernanceState — signal verdict bridge to activities
        config=None,  # Optional GovernanceConfig
    ):
        self.api_url = api_url.rstrip("/") if api_url else ""
        self.api_key = api_key
        self.state = state
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
        self.max_patch_restarts = (
            getattr(config, "max_patch_restarts", 3) if config else 3
        )

    def workflow_interceptor_class(
        self, input: WorkflowInterceptorClassInput
    ) -> type[WorkflowInboundInterceptor] | None:
        state = self.state
        timeout = self.api_timeout
        on_error = self.on_api_error
        send_start = self.send_start_event
        skip_types = self.skip_workflow_types
        skip_sigs = self.skip_signals
        max_restarts = self.max_patch_restarts

        class _Outbound(WorkflowOutboundInterceptor):
            """Stamps the multi-agent session id onto every scheduled activity."""

            def start_activity(self, input: StartActivityInput):
                sid = read_session_from_memo()
                if sid:
                    input.headers = inject_session_header(
                        input.headers, sid, workflow.payload_converter()
                    )
                return super().start_activity(input)

            def start_local_activity(self, input: StartLocalActivityInput):
                sid = read_session_from_memo()
                if sid:
                    input.headers = inject_session_header(
                        input.headers, sid, workflow.payload_converter()
                    )
                return super().start_local_activity(input)

        class _Inbound(WorkflowInboundInterceptor):
            # Run-local coordinator, set at the top of the patch-restart execute
            # path and read by handle_signal (same instance). None until bound /
            # on the legacy path.
            _coordinator: PatchCoordinator | None = None

            def init(self, outbound: WorkflowOutboundInterceptor) -> None:
                super().init(_Outbound(outbound))

            async def execute_workflow(self, input: ExecuteWorkflowInput) -> Any:
                info = workflow.info()

                if info.workflow_type in skip_types:
                    return await super().execute_workflow(input)

                if workflow.patched(_RETRYABLE_BLOCK_PATCH):
                    return await self._execute_workflow_patch(info, input)
                return await self._execute_workflow_legacy(info, input)

            async def _execute_workflow_legacy(
                self, info, input: ExecuteWorkflowInput
            ) -> Any:
                """Exact pre-feature control flow. Taken when replaying a history
                that predates the patch-restart marker — no coordinator,
                no Continue-As-New."""
                sid = read_session_from_memo()

                if send_start and workflow.patched("openbox-v2-start"):
                    await _send_governance_event(
                        _workflow_started_payload(info, sid, input),
                        timeout,
                        on_error,
                    )

                error = None
                try:
                    result = await super().execute_workflow(input)

                    if workflow.patched("openbox-v2-complete"):
                        workflow_output = None
                        try:
                            workflow_output = _serialize_value(result)
                        except Exception:
                            workflow_output = (
                                str(result) if result is not None else None
                            )

                        await _send_governance_event(
                            _events.workflow_completed(
                                workflow_id=info.workflow_id,
                                run_id=info.run_id,
                                workflow_type=info.workflow_type,
                                multi_agent_session_id=sid,
                                extra={"workflow_output": workflow_output},
                            ).to_payload_dict(),
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
                                _events.workflow_failed(
                                    workflow_id=info.workflow_id,
                                    run_id=info.run_id,
                                    workflow_type=info.workflow_type,
                                    # Base event types `error` as str|None; the
                                    # SDK sends the structured error dict (runtime
                                    # contract). Pre-existing shape, unchanged.
                                    error=error,  # type: ignore[arg-type]
                                    multi_agent_session_id=sid,
                                ).to_payload_dict(),
                                timeout,
                                on_error,
                            )
                        except Exception:
                            pass

                    raise

            async def _execute_workflow_patch(
                self, info, input: ExecuteWorkflowInput
            ) -> Any:
                """Patch-restart control flow: own the run-local coordinator, race
                the user workflow against the coordinator wake condition, and
                Continue-As-New on a valid patch request from any origin. This is
                the ONLY path that calls ``workflow.continue_as_new`` (via
                ``_continue_as_new``)."""
                sid = read_session_from_memo()
                coordinator = PatchCoordinator()
                self._coordinator = coordinator  # signals read this instance field
                token = bind_coordinator(
                    coordinator
                )  # emit_handoff reads the ContextVar
                current_args = list(input.args) if input.args is not None else []

                try:
                    # WorkflowStarted — a BLOCK with patch restarts before user code.
                    if send_start:
                        started = await _send_governance_event(
                            _workflow_started_payload(info, sid, input),
                            timeout,
                            on_error,
                            patch_enabled=True,
                        )
                        if isinstance(started, PatchRequest):
                            return await _continue_as_new(
                                started, current_args, max_restarts
                            )

                    # Race the user workflow against the coordinator wake condition.
                    # No workflow.create_task in temporalio; asyncio.create_task runs
                    # on the sandbox's deterministic loop.
                    user_task = asyncio.create_task(super().execute_workflow(input))
                    await workflow.wait_condition(
                        lambda: coordinator.has_request() or user_task.done()
                    )

                    coordinator_request = coordinator.get_request()
                    if coordinator_request is not None:
                        # A signal / handoff / activity-origin patch arrived mid-run.
                        await workflow.wait_condition(workflow.all_handlers_finished)
                        # HALT dominates a concurrent patch (priority: HALT > BLOCK).
                        if user_task.done() and _is_halt(user_task.exception()):
                            raise user_task.exception()  # type: ignore[misc]
                        _dispose_user_task(user_task)
                        return await _continue_as_new(
                            coordinator_request, current_args, max_restarts
                        )

                    # The user task finished first.
                    try:
                        result = user_task.result()
                    except PatchControl:
                        # A handoff unwound user code AND submitted to the coordinator.
                        await workflow.wait_condition(workflow.all_handlers_finished)
                        control_request = coordinator.get_request()
                        if control_request is not None:
                            return await _continue_as_new(
                                control_request, current_args, max_restarts
                            )
                        raise  # unreachable: the control signal always follows a submit
                    except Exception as e:
                        # A genuine HALT surfacing through user code (e.g. a signal
                        # or completed-hook HALT enforced by an activity) dominates:
                        # never restart it (priority HALT > BLOCK with patch, §3.3;
                        # acceptance: HALT never restarts). Mirrors the coordinator-
                        # win path's HALT-dominance guard above.
                        if _is_halt(e):
                            raise
                        # Activity / hook origin surfaced as an ActivityError chain.
                        req = extract_patch_request(e)
                        if req is not None:
                            return await _continue_as_new(
                                req, current_args, max_restarts
                            )
                        # Genuine failure: evaluate WorkflowFailed. A valid patch
                        # there overrides the original failure with a restart;
                        # otherwise the original exception is rethrown unchanged.
                        error = _build_error_dict(e)
                        failed_outcome = None
                        try:
                            failed_outcome = await _send_governance_event(
                                _events.workflow_failed(
                                    workflow_id=info.workflow_id,
                                    run_id=info.run_id,
                                    workflow_type=info.workflow_type,
                                    # Base event types `error` as str|None; the
                                    # SDK sends the structured error dict (runtime
                                    # contract). Pre-existing shape, unchanged.
                                    error=error,  # type: ignore[arg-type]
                                    multi_agent_session_id=sid,
                                ).to_payload_dict(),
                                timeout,
                                on_error,
                                patch_enabled=True,
                            )
                        except Exception:
                            # Swallow ALL failed-reporting errors (including a
                            # fail_closed governance outage mapped to
                            # GovernanceHaltError) so they never shadow the original
                            # workflow exception. A genuine policy HALT still
                            # terminates the workflow via the reporting activity's
                            # client.terminate() before this point, so HALT remains
                            # honored; a BLOCK with patch is returned (not raised) by
                            # the dispatcher, so it is unaffected by this swallow.
                            pass
                        if isinstance(failed_outcome, PatchRequest):
                            return await _continue_as_new(
                                failed_outcome, current_args, max_restarts
                            )
                        raise

                    # WorkflowCompleted — a BLOCK with patch restarts instead of
                    # returning the user result.
                    workflow_output = None
                    try:
                        workflow_output = _serialize_value(result)
                    except Exception:
                        workflow_output = str(result) if result is not None else None

                    completed = await _send_governance_event(
                        _events.workflow_completed(
                            workflow_id=info.workflow_id,
                            run_id=info.run_id,
                            workflow_type=info.workflow_type,
                            multi_agent_session_id=sid,
                            extra={"workflow_output": workflow_output},
                        ).to_payload_dict(),
                        timeout,
                        on_error,
                        patch_enabled=True,
                    )
                    if isinstance(completed, PatchRequest):
                        return await _continue_as_new(
                            completed, current_args, max_restarts
                        )
                    return result
                finally:
                    # Always reset the ContextVar so no coordinator leaks into a
                    # sequential run on the same worker event loop.
                    unbind_coordinator(token)

            async def handle_signal(self, input: HandleSignalInput) -> None:
                info = workflow.info()

                if input.signal in skip_sigs or info.workflow_type in skip_types:
                    return await super().handle_signal(input)

                if workflow.patched("openbox-v2-signal"):
                    sid = read_session_from_memo()
                    patch_enabled = workflow.patched(_RETRYABLE_BLOCK_PATCH)
                    result = await _send_governance_event(
                        _events.signal_received(
                            workflow_id=info.workflow_id,
                            run_id=info.run_id,
                            workflow_type=info.workflow_type,
                            task_queue=info.task_queue,
                            signal_name=input.signal,
                            multi_agent_session_id=sid,
                            extra={"signal_args": input.args},
                        ).to_payload_dict(),
                        timeout,
                        on_error,
                        patch_enabled=patch_enabled,
                    )

                    if patch_enabled and isinstance(result, PatchRequest):
                        # Never Continue-As-New inside the handler. Submit to the
                        # run-local coordinator; the main execute path drains
                        # handlers then Continue-As-News. Skip the user handler.
                        if self._coordinator is not None:
                            self._coordinator.submit(result)  # first-wins
                        elif state:
                            # Coordinator unbound (should not happen in a patched
                            # run): never silently drop the patch — enforce it as a
                            # plain block via the activity bridge (fail safe).
                            state.set_signal_verdict(
                                info.workflow_id,
                                info.run_id,
                                Verdict.BLOCK,
                                result.reason,
                            )
                        return

                    # Legacy/plain path: result is a dict (or None). Unpatched runs
                    # reach here too (patch_enabled=False → dispatcher returns the
                    # legacy dict shape), so a signal carrying a patch still BLOCKS.
                    verdict = (
                        Verdict.from_string(
                            result.get("verdict") or result.get("action")
                        )
                        if result
                        else Verdict.ALLOW
                    )
                    if verdict.should_stop() and state:
                        # Run-scoped: the next activity in THIS run enforces it.
                        state.set_signal_verdict(
                            info.workflow_id,
                            info.run_id,
                            verdict,
                            result.get("reason"),
                        )

                await super().handle_signal(input)

        return _Inbound
