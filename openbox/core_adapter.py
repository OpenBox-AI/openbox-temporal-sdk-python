# openbox/core_adapter.py
"""Temporal FrameworkAdapter + core ActivityContext binding for the base SDK.

This is the Temporal side of the ``openbox_core`` adapter seam:

- ``TemporalFrameworkAdapter`` maps base-SDK verdicts to Temporal-native
  effects: BLOCK/HALT -> non-retryable ``ApplicationError`` with the existing
  ``GovernanceBlock``/``GovernanceHalt`` types; REQUIRE_APPROVAL -> the
  retryable ``ApprovalPending`` error driving Temporal's native HITL
  retry loop; completed-hook verdicts -> abort state for FUTURE execution.
- ``core_activity_scope`` binds the shared ``ActivityContext`` (and trace
  registration) around activity execution with a GUARANTEED try/finally
  reset — context can no longer leak when an activity raises.

NOT sandbox-safe — imports temporalio.exceptions and the base-SDK runtime
modules. Do NOT import from workflow_interceptor.py or other workflow-context
code (guarded by tests/test_workflow_sandbox_import_safety.py).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, NoReturn, Optional

from openbox_core.context import ContextStore, activity_scope
from openbox_core.contracts.context import ActivityContext
from openbox_core.contracts.results import EvaluationResult, Verdict

from .errors import (
    GOVERNANCE_BLOCK_ERROR_TYPE,
    GOVERNANCE_HALT_ERROR_TYPE,
)

__all__ = [
    "TemporalFrameworkAdapter",
    "get_core_context_store",
    "build_core_activity_context",
    "core_activity_scope",
]

# One process-wide store, mirroring the SDK's global-config model. The worker
# owns install/uninstall; interceptors bind per-activity scopes into it.
_core_context_store = ContextStore()


def get_core_context_store() -> ContextStore:
    return _core_context_store


class TemporalFrameworkAdapter:
    """Maps base-SDK governance outcomes onto Temporal-native behavior.

    Args:
        span_processor: Legacy ``WorkflowSpanProcessor`` — when provided,
            REQUIRE_APPROVAL marks ``buffer.pending_approval`` so the retry
            attempt POLLS approval status instead of re-evaluating from
            scratch (the legacy HITL loop keys off that flag).
        hitl_enabled / skip_hitl_activity_types: mirror the legacy config;
            REQUIRE_APPROVAL with HITL unavailable degrades to a
            non-retryable block (fail safe), matching the legacy hook path.
    """

    name = "temporal"

    def __init__(
        self,
        span_processor: Any = None,
        *,
        hitl_enabled: bool = True,
        skip_hitl_activity_types: Optional[set] = None,
        context_store: Optional[ContextStore] = None,
    ):
        self._span_processor = span_processor
        self._hitl_enabled = hitl_enabled
        self._skip_hitl_activity_types = skip_hitl_activity_types or set()
        self._store = context_store if context_store is not None else _core_context_store

    async def handle_approval(self, result: EvaluationResult) -> None:
        """REQUIRE_APPROVAL -> Temporal's retry-based HITL loop.

        Temporal drives approval by failing the activity with a RETRYABLE
        ``ApprovalPending`` error; the interceptor polls approval status on
        the next attempt. The base runtime treats a raise here as
        \"not approved yet\" — exactly the Temporal semantics.
        """
        self._pending_approval_or_block(result)

    def handle_approval_sync(self, result: EvaluationResult) -> None:
        """Sync hook seam — same retry-based flow (never polls inline; an
        inline wait would wedge the activity thread Temporal is retrying)."""
        self._pending_approval_or_block(result)

    def _pending_approval_or_block(self, result: EvaluationResult) -> None:
        from .hitl import raise_approval_pending, should_skip_hitl

        ctx = self._store.current_activity_context()
        activity_type = ctx.activity_type if ctx else ""
        if should_skip_hitl(
            activity_type,
            hitl_enabled=self._hitl_enabled,
            skip_types=self._skip_hitl_activity_types,
        ):
            # HITL unavailable for this activity: approval can never resolve,
            # so degrade to a non-retryable block (legacy hook behavior).
            self._raise_application_error(result)
        # Flag the legacy buffer BEFORE raising: the retry attempt only polls
        # approval status when ``pending_approval`` is set — without it every
        # retry re-runs started-hook evaluation from scratch.
        if self._span_processor is not None and ctx is not None:
            buffer = self._span_processor.get_buffer(ctx.workflow_id)
            if buffer is not None:
                buffer.pending_approval = True
        raise_approval_pending(result.reason or "Approval required")

    def raise_lifecycle_blocked(self, result: EvaluationResult) -> NoReturn:
        self._raise_application_error(result)

    def raise_hook_blocked(self, result: EvaluationResult) -> NoReturn:
        self._raise_application_error(result)

    def on_completed_hook_result(self, result: EvaluationResult) -> None:
        """Completed verdicts affect FUTURE execution only (the operation
        already ran). The hook runtime has already marked the abort/halt flags
        in the core ContextStore; nothing Temporal-specific remains to do
        until the legacy span-processor abort state is retired."""
        return None

    @staticmethod
    def _raise_application_error(result: EvaluationResult) -> NoReturn:
        from temporalio.exceptions import ApplicationError

        reason = result.reason or "Blocked by governance policy"
        if result.verdict is Verdict.HALT:
            raise ApplicationError(
                f"Governance halt: {reason}",
                type=GOVERNANCE_HALT_ERROR_TYPE,
                non_retryable=True,
            )
        raise ApplicationError(
            f"Governance block: {reason}",
            type=GOVERNANCE_BLOCK_ERROR_TYPE,
            non_retryable=True,
        )


def build_core_activity_context(
    info: Any,
    activity_input: Any = None,
    multi_agent_session_id: Optional[str] = None,
) -> ActivityContext:
    """Shared ActivityContext from a ``temporalio.activity.Info``.

    Temporal-specific extras (``attempt``) ride in ``metadata`` — they have no
    first-class field on the shared context.
    """
    return ActivityContext(
        workflow_id=info.workflow_id,
        run_id=info.workflow_run_id,
        workflow_type=info.workflow_type,
        task_queue=info.task_queue,
        activity_id=info.activity_id,
        activity_type=info.activity_type,
        activity_input=activity_input,
        multi_agent_session_id=multi_agent_session_id,
        metadata={"attempt": info.attempt, "source": "workflow-telemetry"},
    )


@contextmanager
def core_activity_scope(
    info: Any,
    activity_input: Any = None,
    *,
    trace_id: Optional[int] = None,
    multi_agent_session_id: Optional[str] = None,
) -> Iterator[ActivityContext]:
    """Bind the shared context around activity execution (try/finally reset).

    Fixes the historical leak where context reset lived in
    ``_handle_completion()`` and was skipped when the activity raised.
    """
    ctx = build_core_activity_context(info, activity_input, multi_agent_session_id)
    with activity_scope(ctx, trace_id=trace_id, store=_core_context_store) as bound:
        yield bound


def create_core_runtime(
    api_url: str,
    api_key: str,
    *,
    timeout_seconds: float = 30.0,
    on_api_error: str = "fail_open",
    agent_did: Optional[str] = None,
    agent_private_key: Optional[str] = None,
    span_processor: Any = None,
    hitl_enabled: bool = True,
    skip_hitl_activity_types: Optional[set] = None,
    **instrumentation_toggles: Any,
):
    """Build an ``OpenBoxRuntime`` wired for Temporal (OPT-IN, worker scope).

    Constructs the base-SDK runtime with the ``TemporalFrameworkAdapter`` and
    the shared context store. Call from worker/plugin initialization — NEVER
    from workflow sandbox paths.

    NOTE: the legacy in-repo hook instrumentation remains the default on this
    branch. Installing core instrumentation (``runtime.install_instrumentation()``)
    while legacy hooks are active would double-govern operations — the flip
    happens per operation type once hook-payload parity is proven (HTTP first,
    then DB/file/function; Redis/Mongo stay on legacy hooks until the base SDK
    scopes them).
    """
    from openbox_core.config import InstrumentationConfig, OpenBoxConfig
    from openbox_core.runtime import OpenBoxRuntime

    config = OpenBoxConfig(
        api_url=api_url,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        on_api_error=on_api_error,
        agent_did=agent_did,
        agent_private_key=agent_private_key,
        instrumentation=InstrumentationConfig(**instrumentation_toggles),
    ).normalized()
    return OpenBoxRuntime(
        config,
        TemporalFrameworkAdapter(
            span_processor,
            hitl_enabled=hitl_enabled,
            skip_hitl_activity_types=skip_hitl_activity_types,
            context_store=_core_context_store,
        ),
        context_store=_core_context_store,
    )
