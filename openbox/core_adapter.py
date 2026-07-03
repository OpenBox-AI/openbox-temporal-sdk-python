# openbox/core_adapter.py
"""Temporal FrameworkAdapter + core ActivityContext binding for the base SDK.

This is the Temporal side of the ``openbox_core`` adapter seam. Governance is
owned by the base runtime; this module maps base verdicts to Temporal-native
effects and never builds hook payloads or evaluates hook events itself:

- ``TemporalFrameworkAdapter`` — BLOCK/HALT (started hook or lifecycle) ->
  non-retryable ``ApplicationError``; REQUIRE_APPROVAL -> retryable
  ``ApprovalPending`` (Temporal's native HITL retry loop) + a pending marker in
  ``TemporalGovernanceState``; completed-hook BLOCK/HALT -> recorded in
  ``TemporalGovernanceState`` (run-scoped) for the activity interceptor to
  surface after user code returns.
- ``core_activity_scope`` binds the shared ``ActivityContext`` around activity
  execution with a GUARANTEED try/finally reset.

NOT sandbox-safe — imports temporalio.exceptions and the base-SDK runtime
modules. Do NOT import from workflow_interceptor.py or other workflow-context
code (guarded by tests/test_workflow_sandbox_import_safety.py).
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator, NoReturn, Optional

from openbox_core.context import ContextStore, activity_scope
from openbox_core.contracts.context import ActivityContext
from openbox_core.contracts.results import EvaluationResult, Verdict

from .errors import (
    GOVERNANCE_BLOCK_ERROR_TYPE,
    GOVERNANCE_HALT_ERROR_TYPE,
)
from .governance_state import TemporalGovernanceState

logger = logging.getLogger(__name__)

__all__ = [
    "TemporalFrameworkAdapter",
    "get_core_context_store",
    "build_core_activity_context",
    "core_activity_scope",
    "create_core_runtime",
]

# One process-wide store, mirroring the SDK's global-config model. The worker
# owns install/uninstall; interceptors bind per-activity scopes into it.
_core_context_store = ContextStore()


def get_core_context_store() -> ContextStore:
    return _core_context_store


class TemporalFrameworkAdapter:
    """Maps base-SDK governance outcomes onto Temporal-native behavior.

    Args:
        state: ``TemporalGovernanceState`` — REQUIRE_APPROVAL marks a pending
            approval so the retry attempt polls status; completed-hook BLOCK/HALT
            is recorded run-scoped for the activity interceptor.
        hitl_enabled / skip_hitl_activity_types: mirror the config; a
            REQUIRE_APPROVAL where HITL is unavailable degrades to a non-retryable
            block (fail safe).
    """

    name = "temporal"

    def __init__(
        self,
        state: TemporalGovernanceState,
        *,
        hitl_enabled: bool = True,
        skip_hitl_activity_types: Optional[set] = None,
        context_store: Optional[ContextStore] = None,
    ):
        self._state = state
        self._hitl_enabled = hitl_enabled
        self._skip_hitl_activity_types = skip_hitl_activity_types or set()
        self._store = context_store if context_store is not None else _core_context_store

    async def handle_approval(self, result: EvaluationResult) -> None:
        """Async hook REQUIRE_APPROVAL -> Temporal's retry-based HITL loop.

        Runs in the activity's own task, so the ambient ContextVar resolves the
        activity context. Raises the retryable pending error; the interceptor
        polls approval status on the next attempt."""
        self._pending_approval_or_block(result)

    def handle_approval_sync(
        self, result: EvaluationResult, context: Optional[ActivityContext] = None
    ) -> None:
        """Sync hook seam — same retry-based flow with the span-resolved context
        (ambient lookup can miss in user-spawned threads)."""
        self._pending_approval_or_block(result, context)

    def _pending_approval_or_block(
        self, result: EvaluationResult, context: Optional[ActivityContext] = None
    ) -> None:
        from .hitl import raise_approval_pending, should_skip_hitl

        ctx = context if context is not None else self._store.current_activity_context()
        activity_type = (ctx.activity_type or "") if ctx else ""
        if should_skip_hitl(
            activity_type,
            hitl_enabled=self._hitl_enabled,
            skip_types=self._skip_hitl_activity_types,
        ):
            # HITL unavailable for this activity: approval can never resolve, so
            # degrade to a non-retryable block (fail safe).
            self._raise_application_error(result)
        if ctx is not None:
            # Mark BEFORE raising: the retry attempt only POLLS approval status
            # when the marker is set; otherwise it would re-evaluate from scratch.
            # Bound activity contexts always carry these keys; coerce for typing.
            self._state.mark_pending_approval(
                ctx.workflow_id or "", ctx.run_id or "", ctx.activity_id or ""
            )
        else:
            logger.warning(
                "core REQUIRE_APPROVAL without a resolved activity context: the "
                "retry attempt will re-evaluate instead of polling approval status"
            )
        raise_approval_pending(result.reason or "Approval required")

    def raise_lifecycle_blocked(self, result: EvaluationResult) -> NoReturn:
        self._raise_application_error(result)

    def raise_hook_blocked(self, result: EvaluationResult) -> NoReturn:
        self._raise_application_error(result)

    def on_completed_hook_result(
        self, result: EvaluationResult, context: Optional[ActivityContext] = None
    ) -> None:
        """Completed verdicts affect FUTURE execution only (the operation already
        ran). Record a BLOCK/HALT run-scoped so the activity interceptor can skip
        a duplicate completed event (BLOCK) or reach the terminate path (HALT)
        after user code returns. Without a resolved context there is no run to key
        on — the base ContextStore still carries the within-activity abort flag."""
        if not result.verdict.should_stop() or context is None:
            return
        self._state.record_completed_stop(
            context.workflow_id or "",
            context.run_id or "",
            context.activity_id or "",
            result.verdict,
            result.reason,
        )

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

    The ONLY hook-context bridge: base instrumentation resolves the activity
    context from the store this binds into (ambient ContextVar, or the trace map
    for hook code running where ContextVars do not propagate).
    """
    ctx = build_core_activity_context(info, activity_input, multi_agent_session_id)
    with activity_scope(ctx, trace_id=trace_id, store=_core_context_store) as bound:
        yield bound


def create_core_runtime(
    *,
    api_url: str,
    api_key: str,
    state: TemporalGovernanceState,
    timeout_seconds: float = 30.0,
    on_api_error: str = "fail_open",
    agent_did: Optional[str] = None,
    agent_private_key: Optional[str] = None,
    hitl_enabled: bool = True,
    skip_hitl_activity_types: Optional[set] = None,
    skip_workflow_types: Optional[set] = None,
    skip_activity_types: Optional[set] = None,
    skip_signals: Optional[set] = None,
    send_start_event: bool = True,
    send_activity_start_event: bool = True,
    instrument_databases: bool = True,
    instrument_file_io: bool = True,
    max_body_size: int = 65536,
) -> Any:
    """Build the ``OpenBoxRuntime`` that OWNS all hook instrumentation for a
    Temporal worker/plugin. Call from worker/plugin init — NEVER from workflow
    sandbox paths. The caller stores the returned runtime and calls
    ``runtime.install_instrumentation()`` / ``uninstall_instrumentation()``.

    The base InstrumentationManager always ignores ``config.api_url`` so the
    evaluate call never governs itself.
    """
    from openbox_core.config import (
        GateConfig,
        HitlConfig,
        InstrumentationConfig,
        OpenBoxConfig,
        PrivacyConfig,
    )
    from openbox_core.runtime import OpenBoxRuntime

    config = OpenBoxConfig(
        api_url=api_url,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        on_api_error=on_api_error,
        agent_did=agent_did,
        agent_private_key=agent_private_key,
        hitl=HitlConfig(
            enabled=hitl_enabled,
            skip_activity_types=(skip_hitl_activity_types or {"send_governance_event"}),
        ),
        gate=GateConfig(
            skip_workflow_types=skip_workflow_types or set(),
            skip_activity_types=(skip_activity_types or {"send_governance_event"}),
            skip_signals=skip_signals or set(),
            send_start_event=send_start_event,
            send_activity_start_event=send_activity_start_event,
        ),
        instrumentation=InstrumentationConfig(
            db_enabled=instrument_databases,
            file_enabled=instrument_file_io,
        ),
        privacy=PrivacyConfig(max_body_size=max_body_size),
    ).normalized()

    adapter = TemporalFrameworkAdapter(
        state,
        hitl_enabled=hitl_enabled,
        skip_hitl_activity_types=skip_hitl_activity_types,
        context_store=_core_context_store,
    )
    return OpenBoxRuntime(config, adapter, context_store=_core_context_store)
