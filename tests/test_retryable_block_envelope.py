"""Phase 1 — request envelope, normalizer, and error-transport extractor.

Covers the base-contract matrix (proposal §13.1) and the Temporal transport
matrix (§13.2). The verdict matrix itself lives in the base SDK
(``handle_retryable_block``); here we assert the Temporal wrapper delegates to it
correctly and that the versioned detail payload round-trips + extracts safely.
"""

from __future__ import annotations

from openbox_core.contracts.results import ApprovalResult
from temporalio.exceptions import ApplicationError

from openbox.errors import GOVERNANCE_RETRYABLE_BLOCK_ERROR_TYPE
from openbox.retryable_block import (
    GOVERNANCE_RETRYABLE_BLOCK_SCHEMA_VERSION,
    RetryableBlockRequest,
    extract_retryable_block_request,
    retryable_block_request,
)
from openbox.types import GovernanceVerdictResponse


def _eval(data: dict) -> GovernanceVerdictResponse:
    return GovernanceVerdictResponse.from_dict(data)


# --------------------------------------------------------------------------- #
# Base-contract matrix (§13.1) — delegation to handle_retryable_block
# --------------------------------------------------------------------------- #


def test_block_with_valid_plan_produces_request():
    result = _eval(
        {
            "verdict": "block",
            "reason": "retry with corrected input",
            "governance_event_id": "evt_123",
            "retry_plan": {"new_input": {"query": "corrected"}},
        }
    )
    req = retryable_block_request(result, event_type="WorkflowStarted")
    assert req is not None
    assert req.schema_version == GOVERNANCE_RETRYABLE_BLOCK_SCHEMA_VERSION
    assert req.new_input == {"query": "corrected"}
    assert req.governance_event_id == "evt_123"
    assert req.reason == "retry with corrected input"
    assert req.event_type == "WorkflowStarted"
    assert req.hook_trigger is False
    assert req.hook_stage is None


def test_new_input_none_is_valid_and_distinct_from_absent_plan():
    with_plan = _eval({"verdict": "block", "retry_plan": {"new_input": None}})
    req = retryable_block_request(with_plan, event_type="ActivityCompleted")
    assert req is not None
    assert req.new_input is None  # valid "reuse current input" directive

    without_plan = _eval({"verdict": "block", "reason": "plain block"})
    assert retryable_block_request(without_plan, event_type="ActivityCompleted") is None


def test_hook_origin_metadata_is_carried():
    result = _eval({"verdict": "block", "retry_plan": {"new_input": 42}})
    req = retryable_block_request(
        result, event_type="ActivityStarted", hook_trigger=True, hook_stage="completed"
    )
    assert req is not None
    assert req.event_type == "ActivityStarted"
    assert req.hook_trigger is True
    assert req.hook_stage == "completed"
    assert req.new_input == 42


def test_plain_block_without_plan_returns_none():
    assert (
        retryable_block_request(
            _eval({"verdict": "block", "reason": "nope"}), event_type="WorkflowFailed"
        )
        is None
    )


def test_malformed_plans_return_none():
    # Extra key beyond new_input.
    extra = _eval({"verdict": "block", "retry_plan": {"new_input": 1, "x": 2}})
    assert retryable_block_request(extra, event_type="ActivityStarted") is None
    # Boolean new_input is rejected by the base contract.
    boolean = _eval({"verdict": "block", "retry_plan": {"new_input": True}})
    assert retryable_block_request(boolean, event_type="ActivityStarted") is None
    # Unsafe integer beyond JS-safe range.
    unsafe = _eval({"verdict": "block", "retry_plan": {"new_input": 2**53}})
    assert retryable_block_request(unsafe, event_type="ActivityStarted") is None


def test_non_block_verdicts_return_none():
    for verdict in ("allow", "constrain", "require_approval"):
        result = _eval({"verdict": verdict, "retry_plan": {"new_input": "x"}})
        assert retryable_block_request(result, event_type="WorkflowStarted") is None


def test_halt_never_produces_request_even_with_plan():
    result = _eval({"verdict": "halt", "retry_plan": {"new_input": "x"}})
    assert retryable_block_request(result, event_type="WorkflowFailed") is None


def test_approval_expired_and_pending_return_none():
    expired = ApprovalResult.from_dict(
        {"verdict": "block", "expired": True, "retry_plan": {"new_input": "x"}}
    )
    assert retryable_block_request(expired, event_type="ActivityStarted") is None

    pending = ApprovalResult.from_dict({"retry_plan": {"new_input": "x"}})
    assert retryable_block_request(pending, event_type="ActivityStarted") is None


def test_approval_non_expired_block_with_plan_produces_request():
    approval = ApprovalResult.from_dict(
        {
            "verdict": "block",
            "reason": "admin retry",
            "id": "evt_a",
            "retry_plan": {"new_input": {"k": "v"}},
        }
    )
    req = retryable_block_request(approval, event_type="ActivityStarted")
    assert req is not None
    assert req.new_input == {"k": "v"}
    assert req.governance_event_id == "evt_a"


# --------------------------------------------------------------------------- #
# Transport matrix (§13.2) — to_dict / from_dict / extractor
# --------------------------------------------------------------------------- #


def _make_request(**over) -> RetryableBlockRequest:
    base = dict(
        schema_version=GOVERNANCE_RETRYABLE_BLOCK_SCHEMA_VERSION,
        new_input={"query": "x"},
        governance_event_id="evt_1",
        reason="retry",
        event_type="ActivityCompleted",
        hook_trigger=False,
        hook_stage=None,
    )
    base.update(over)
    return RetryableBlockRequest(**base)


def test_to_from_dict_round_trip():
    req = _make_request()
    assert RetryableBlockRequest.from_dict(req.to_dict()) == req


def test_from_dict_rejects_unknown_schema_version():
    d = _make_request().to_dict()
    d["schema_version"] = 999
    assert RetryableBlockRequest.from_dict(d) is None


def test_from_dict_absent_new_input_key_fails_but_present_none_preserved():
    d = _make_request().to_dict()
    del d["new_input"]
    assert RetryableBlockRequest.from_dict(d) is None

    d2 = _make_request(new_input=None).to_dict()
    assert "new_input" in d2
    rebuilt = RetryableBlockRequest.from_dict(d2)
    assert rebuilt is not None and rebuilt.new_input is None


def test_from_dict_rejects_bad_field_types():
    # Non-bool hook_trigger (including 0/1 ints).
    assert (
        RetryableBlockRequest.from_dict(_make_request().to_dict() | {"hook_trigger": 1})
        is None
    )
    # Empty / non-str event_type.
    assert (
        RetryableBlockRequest.from_dict(_make_request().to_dict() | {"event_type": ""})
        is None
    )
    assert (
        RetryableBlockRequest.from_dict(_make_request().to_dict() | {"event_type": 5})
        is None
    )
    # Wrong-typed optional fields.
    assert (
        RetryableBlockRequest.from_dict(_make_request().to_dict() | {"hook_stage": 3})
        is None
    )
    assert (
        RetryableBlockRequest.from_dict(
            _make_request().to_dict() | {"governance_event_id": 7}
        )
        is None
    )
    assert (
        RetryableBlockRequest.from_dict(_make_request().to_dict() | {"reason": 9})
        is None
    )
    # Not a dict at all.
    assert RetryableBlockRequest.from_dict(["not", "a", "dict"]) is None


def _app_error(req: RetryableBlockRequest) -> ApplicationError:
    return ApplicationError(
        "Governance requested workflow restart",
        req.to_dict(),
        type=GOVERNANCE_RETRYABLE_BLOCK_ERROR_TYPE,
        non_retryable=True,
    )


class _FakeActivityError(Exception):
    """Stand-in for temporalio ActivityError, which exposes a ``.cause`` property."""

    def __init__(self, cause: BaseException):
        super().__init__("activity failed")
        self.cause = cause


def test_extractor_finds_request_through_cause_chain():
    req = _make_request()
    wrapped = _FakeActivityError(_app_error(req))
    found = extract_retryable_block_request(wrapped)
    assert found == req


def test_extractor_finds_request_through_dunder_cause_chain():
    req = _make_request(new_input=None)
    try:
        try:
            raise _app_error(req)
        except ApplicationError as inner:
            raise RuntimeError("boundary") from inner
    except RuntimeError as outer:
        found = extract_retryable_block_request(outer)
    assert found == req


def test_extractor_matches_type_not_message():
    # Same message text but a different (plain-BLOCK) type → not extracted.
    err = ApplicationError(
        "Governance requested workflow restart",
        _make_request().to_dict(),
        type="GovernanceBlock",
        non_retryable=True,
    )
    assert extract_retryable_block_request(err) is None


def test_extractor_returns_none_on_bad_detail_shapes():
    req = _make_request()
    # Zero details.
    assert (
        extract_retryable_block_request(
            ApplicationError("x", type=GOVERNANCE_RETRYABLE_BLOCK_ERROR_TYPE)
        )
        is None
    )
    # Two details.
    assert (
        extract_retryable_block_request(
            ApplicationError(
                "x",
                req.to_dict(),
                req.to_dict(),
                type=GOVERNANCE_RETRYABLE_BLOCK_ERROR_TYPE,
            )
        )
        is None
    )
    # Non-dict detail.
    assert (
        extract_retryable_block_request(
            ApplicationError(
                "x", "not-a-dict", type=GOVERNANCE_RETRYABLE_BLOCK_ERROR_TYPE
            )
        )
        is None
    )
    # Right type, unknown schema version in the detail.
    bad_schema = req.to_dict()
    bad_schema["schema_version"] = 999
    assert (
        extract_retryable_block_request(
            ApplicationError(
                "x", bad_schema, type=GOVERNANCE_RETRYABLE_BLOCK_ERROR_TYPE
            )
        )
        is None
    )


def test_extractor_returns_none_when_no_matching_error():
    assert extract_retryable_block_request(RuntimeError("nothing here")) is None
