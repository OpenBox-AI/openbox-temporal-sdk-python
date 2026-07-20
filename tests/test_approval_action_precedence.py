"""Regression gate for approval decision-source precedence.

Approval poll responses parse via ``ApprovalResult``, which prefers ``action``
over ``verdict`` when both are present. A response with neither field remains
pending and must never auto-approve.
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


class TestStrictDecisionVocabularyGate:
    """C1 hardening gate: unparseable decisions NEVER auto-approve.

    The evaluate-path leniency (unknown verdict -> ALLOW, legacy parity)
    must not exist at the human-approval boundary: empty ``action`` strings
    no longer shadow the verdict, and unknown vocabulary keeps polling.
    """

    def test_empty_action_does_not_shadow_blocking_verdict(self):
        with pytest.raises(ApplicationError) as exc_info:
            call({"verdict": "block", "action": "", "reason": "denied"})
        assert exc_info.value.type == "ApprovalRejected"

    def test_empty_action_with_pending_verdict_keeps_polling(self):
        with pytest.raises(ApplicationError) as exc_info:
            call({"verdict": "require_approval", "action": ""})
        assert exc_info.value.type == "ApprovalPending"

    def test_unknown_action_vocabulary_keeps_polling_never_approves(self):
        for junk in ("denied", "pending", "approved-maybe"):
            with pytest.raises(ApplicationError) as exc_info:
                call({"action": junk})
            assert exc_info.value.type == "ApprovalPending", junk

    def test_unknown_verdict_vocabulary_keeps_polling(self):
        with pytest.raises(ApplicationError) as exc_info:
            call({"verdict": "banana"})
        assert exc_info.value.type == "ApprovalPending"
