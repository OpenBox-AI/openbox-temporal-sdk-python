"""Phase 4 — restart coordinator, run-scoped accessor, and restart-budget helper.

The coordinator/accessor/budget are sandbox-safe deterministic primitives. The
budget helper reads the workflow memo, so its tests monkeypatch ``workflow.memo`` /
``workflow.memo_value`` (the full Temporal-runtime visibility of the ContextVar
accessor across sequential/concurrent runs is validated end-to-end in Phase 7).
"""

from __future__ import annotations

import pytest
from temporalio import workflow
from temporalio.exceptions import ApplicationError

from openbox.errors import GOVERNANCE_PATCH_LIMIT_EXCEEDED_ERROR_TYPE
from openbox.patch import PatchRequest
from openbox.patch_coordinator import (
    MEMO_RESTART_COUNT_KEY,
    PatchControl,
    PatchCoordinator,
    bind_coordinator,
    get_coordinator,
    next_restart_memo,
    unbind_coordinator,
)


def _req(new_input="x", event_type="SignalReceived") -> PatchRequest:
    return PatchRequest(
        schema_version=1,
        new_input=new_input,
        governance_event_id=None,
        reason=None,
        event_type=event_type,
        hook_trigger=False,
        hook_stage=None,
    )


# --------------------------------------------------------------------------- #
# Coordinator (first-wins)
# --------------------------------------------------------------------------- #


def test_coordinator_starts_empty():
    coord = PatchCoordinator()
    assert coord.has_request() is False
    assert coord.get_request() is None


def test_coordinator_first_submit_wins():
    coord = PatchCoordinator()
    first = _req(new_input="first")
    second = _req(new_input="second")
    coord.submit(first)
    coord.submit(second)  # ignored
    assert coord.has_request() is True
    assert coord.get_request() is first


def test_control_signal_is_base_exception_not_exception():
    # Must evade a plain `except Exception` in user code, like cancellation.
    assert issubclass(PatchControl, BaseException)
    assert not issubclass(PatchControl, Exception)


# --------------------------------------------------------------------------- #
# Run-scoped accessor (bind / get / unbind)
# --------------------------------------------------------------------------- #


def test_accessor_default_none_and_bind_unbind_cycle():
    assert get_coordinator() is None
    coord = PatchCoordinator()
    token = bind_coordinator(coord)
    try:
        assert get_coordinator() is coord
    finally:
        unbind_coordinator(token)
    assert get_coordinator() is None  # reset restores the pre-bind (None) state


def test_accessor_unbind_restores_previous_binding():
    outer = PatchCoordinator()
    inner = PatchCoordinator()
    outer_token = bind_coordinator(outer)
    try:
        inner_token = bind_coordinator(inner)
        assert get_coordinator() is inner
        unbind_coordinator(inner_token)
        assert get_coordinator() is outer  # no leak across nested runs
    finally:
        unbind_coordinator(outer_token)
    assert get_coordinator() is None


# --------------------------------------------------------------------------- #
# Restart budget (memo counter)
# --------------------------------------------------------------------------- #


def _patch_memo(monkeypatch, memo: dict) -> None:
    monkeypatch.setattr(workflow, "memo", lambda: dict(memo))
    monkeypatch.setattr(
        workflow, "memo_value", lambda key, default=None, **_: memo.get(key, default)
    )


def test_budget_missing_counter_starts_at_one(monkeypatch):
    _patch_memo(monkeypatch, {"openbox_multi_agent_session_id": "sess"})
    out = next_restart_memo(3)
    assert out[MEMO_RESTART_COUNT_KEY] == 1
    # Full memo is copied — application metadata / session id preserved.
    assert out["openbox_multi_agent_session_id"] == "sess"


def test_budget_increments_existing_counter(monkeypatch):
    _patch_memo(monkeypatch, {MEMO_RESTART_COUNT_KEY: 1, "k": "v"})
    out = next_restart_memo(3)
    assert out[MEMO_RESTART_COUNT_KEY] == 2
    assert out["k"] == "v"


def test_budget_at_cap_raises_limit_exceeded(monkeypatch):
    _patch_memo(monkeypatch, {MEMO_RESTART_COUNT_KEY: 3})
    with pytest.raises(ApplicationError) as exc:
        next_restart_memo(3)
    assert exc.value.type == GOVERNANCE_PATCH_LIMIT_EXCEEDED_ERROR_TYPE
    assert exc.value.non_retryable is True


@pytest.mark.parametrize("bad", ["1", 1.5, None, {"x": 1}])
def test_budget_non_int_counter_raises(monkeypatch, bad):
    _patch_memo(monkeypatch, {MEMO_RESTART_COUNT_KEY: bad})
    with pytest.raises(ApplicationError) as exc:
        next_restart_memo(3)
    assert exc.value.type == GOVERNANCE_PATCH_LIMIT_EXCEEDED_ERROR_TYPE


def test_budget_bool_counter_rejected(monkeypatch):
    # bool is an int subclass; it must NOT be accepted as a count.
    _patch_memo(monkeypatch, {MEMO_RESTART_COUNT_KEY: True})
    with pytest.raises(ApplicationError):
        next_restart_memo(3)


def test_budget_negative_counter_raises(monkeypatch):
    _patch_memo(monkeypatch, {MEMO_RESTART_COUNT_KEY: -1})
    with pytest.raises(ApplicationError):
        next_restart_memo(3)
