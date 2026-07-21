"""Run-local restart coordinator, restart-budget helper, and the internal
control-flow signal for Continue-As-New.

These are the sandbox-safe, deterministic primitives the workflow boundary uses to
honor a retryable BLOCK. Sandbox-safe: at module top level this imports ONLY stdlib
(``contextvars``), ``temporalio.workflow``, ``temporalio.exceptions.ApplicationError``
(both pure/sandbox-safe — the workflow interceptor imports them the same way), and
the pure envelope module. No ``httpx`` / ``logging`` / crypto. Guarded by
``tests/test_workflow_sandbox_import_safety.py``.

The pure request envelope + normalizer live in the sibling ``openbox/retryable_block.py``
(which has no ``temporalio`` dependency); this module adds the workflow-runtime
pieces on top of it.
"""

from __future__ import annotations

import contextvars
from typing import Optional

from temporalio import workflow
from temporalio.exceptions import ApplicationError

from .errors import GOVERNANCE_RETRY_LIMIT_EXCEEDED_ERROR_TYPE
from .retryable_block import RetryableBlockRequest

__all__ = [
    "RetryableBlockControl",
    "RetryableBlockCoordinator",
    "bind_coordinator",
    "unbind_coordinator",
    "get_coordinator",
    "MEMO_RESTART_COUNT_KEY",
    "next_restart_memo",
]


class RetryableBlockControl(BaseException):
    """Internal control-flow signal: a retry request was submitted; unwind the user
    workflow task so no post-discovery user statement runs before Continue-As-New.

    Subclasses ``BaseException`` (not ``Exception``) so an ordinary
    ``except Exception`` in application code does not accidentally swallow it —
    mirroring how Temporal's own cancellation / Continue-As-New control errors evade
    broad handlers. Owned by the SDK; application code must not catch/suppress it.
    Continue-As-New stays owned by the inbound interceptor, which catches this and
    reads the coordinator.
    """


class RetryableBlockCoordinator:
    """Run-local, first-wins store for a single retry request.

    Workflow code is single-threaded and deterministic, so no lock is needed. Holds
    no wall-clock or process-global state. Created once per workflow run by the
    inbound interceptor; reachable from ``handle_signal`` (interceptor instance
    field) and from ``emit_handoff`` (the module ContextVar below).
    """

    def __init__(self) -> None:
        self._request: Optional[RetryableBlockRequest] = None

    def submit(self, req: RetryableBlockRequest) -> None:
        """Store the first request; later submits are ignored (first-wins)."""
        if self._request is None:
            self._request = req

    def has_request(self) -> bool:
        return self._request is not None

    def get_request(self) -> Optional[RetryableBlockRequest]:
        return self._request


# Run-scoped accessor for emit_handoff (user code, no interceptor reference). Bound
# with a token at execute_workflow entry and ALWAYS reset in a finally, so a
# coordinator never leaks into a sequential run (or an error path) on the same
# worker event loop. Defaults to None between runs.
_coordinator_var: contextvars.ContextVar[Optional[RetryableBlockCoordinator]] = (
    contextvars.ContextVar("openbox_retry_coordinator", default=None)
)


def bind_coordinator(
    coordinator: RetryableBlockCoordinator,
) -> "contextvars.Token[Optional[RetryableBlockCoordinator]]":
    """Bind the run-local coordinator; returns a token for ``unbind_coordinator``."""
    return _coordinator_var.set(coordinator)


def unbind_coordinator(
    token: "contextvars.Token[Optional[RetryableBlockCoordinator]]",
) -> None:
    """Reset the ContextVar (call in a finally) so no coordinator leaks across runs."""
    _coordinator_var.reset(token)


def get_coordinator() -> Optional[RetryableBlockCoordinator]:
    """The run-local coordinator bound for the current workflow run, or None."""
    return _coordinator_var.get()


# Namespaced workflow-memo key holding the restart count across the whole chain.
MEMO_RESTART_COUNT_KEY = "openbox_retryable_block_restart_count"


def next_restart_memo(max_restarts: int) -> dict:
    """Read the memo counter, validate + increment, enforce the cap, and return the
    FULL memo dict to hand to ``continue_as_new(memo=...)``.

    Copying the whole memo preserves application metadata and the OpenBox multi-agent
    session id. Deterministic / replay-safe (``workflow.memo`` / ``workflow.memo_value``
    — never ``workflow.info().memo``, which is undecoded raw Payloads).

    Raises ``ApplicationError(type=GovernanceRetryLimitExceeded, non_retryable=True)``
    when the stored counter is not a non-negative int, or when the next restart would
    exceed ``max_restarts``.
    """
    memo = dict(workflow.memo())  # copy → preserves other keys (incl. session id)
    raw = workflow.memo_value(MEMO_RESTART_COUNT_KEY, 0)
    # bool is an int subclass — reject it and any non-int / negative counter.
    if not isinstance(raw, int) or isinstance(raw, bool) or raw < 0:
        raise ApplicationError(
            "Invalid retryable-block restart counter in workflow memo",
            type=GOVERNANCE_RETRY_LIMIT_EXCEEDED_ERROR_TYPE,
            non_retryable=True,
        )
    nxt = raw + 1
    if nxt > max_restarts:
        raise ApplicationError(
            f"Retryable-block restart limit ({max_restarts}) exceeded",
            type=GOVERNANCE_RETRY_LIMIT_EXCEEDED_ERROR_TYPE,
            non_retryable=True,
        )
    memo[MEMO_RESTART_COUNT_KEY] = nxt
    return memo
