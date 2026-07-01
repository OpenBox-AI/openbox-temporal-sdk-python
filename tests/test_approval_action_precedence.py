"""Regression gate for the approval decision-source precedence change.

BEHAVIOR CHANGE (deliberate, base-SDK adoption): approval poll responses now
parse via ``openbox_core.contracts.results.ApprovalResult``, which prefers
``action`` over ``verdict`` when both are present. The pre-migration code
preferred ``verdict`` (``Verdict.from_string(response.get("verdict") or
response.get("action"))``). Additionally, a response with NEITHER field is now
PENDING (retry) instead of the old implicit ALLOW from ``from_string(None)``.

These tests pin the new semantics explicitly so any future flip is loud.
"""

from __future__ import annotations

import pytest
from temporalio.exceptions import ApplicationError

from openbox.hitl import handle_approval_response

ARGS = dict(
    activity_type="charge_card",
    workflow_id="wf-1",
    run_id="run-1",
    activity_id="act-1",
)


def call(response):
    return handle_approval_response(response, **ARGS)


class TestActionPrecedence:
    def test_action_allow_wins_over_verdict_block(self):
        """OLD behavior: verdict-first -> BLOCK -> ApprovalRejected.
        NEW behavior: action-first -> ALLOW -> approved."""
        assert call({"verdict": "block", "action": "continue"}) is True

    def test_action_block_wins_over_verdict_allow(self):
        with pytest.raises(ApplicationError) as exc_info:
            call({"verdict": "allow", "action": "block", "reason": "human said no"})
        assert exc_info.value.type == "ApprovalRejected"
        assert exc_info.value.non_retryable is True

    def test_verdict_still_used_when_no_action(self):
        assert call({"verdict": "allow"}) is True
        with pytest.raises(ApplicationError) as exc_info:
            call({"verdict": "halt"})
        assert exc_info.value.type == "ApprovalRejected"


class TestAbsentDecisionIsPending:
    def test_empty_response_keeps_polling_never_auto_allows(self):
        """OLD behavior: from_string(None) -> ALLOW (approval granted!).
        NEW behavior: pending -> retryable ApprovalPending."""
        with pytest.raises(ApplicationError) as exc_info:
            call({})
        assert exc_info.value.type == "ApprovalPending"
        assert exc_info.value.non_retryable is False


class TestUnchangedSemantics:
    def test_none_response_still_pending(self):
        with pytest.raises(ApplicationError) as exc_info:
            call(None)
        assert exc_info.value.type == "ApprovalPending"

    def test_expired_still_checked_before_verdict(self):
        with pytest.raises(ApplicationError) as exc_info:
            call({"expired": True, "verdict": "allow"})
        assert exc_info.value.type == "ApprovalExpired"
        assert exc_info.value.non_retryable is True

    def test_require_approval_keeps_polling(self):
        with pytest.raises(ApplicationError) as exc_info:
            call({"verdict": "require_approval"})
        assert exc_info.value.type == "ApprovalPending"

    def test_legacy_id_field_normalized(self):
        from openbox_core.contracts.results import ApprovalResult

        assert ApprovalResult.from_dict({"id": "legacy"}).approval_id == "legacy"
