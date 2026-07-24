"""Tests for openbox.multi_agent: handoff payloads, emit_handoff, session headers."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from temporalio.converter import default as _default_converter

from openbox import emit_handoff
from openbox.multi_agent import (
    HEADER_KEY,
    MEMO_KEY,
    build_handoff_payload,
    inject_session_header,
    read_session_from_header,
)
from openbox.types import WorkflowEventType


def _converter():
    return _default_converter().payload_converter


def test_handoff_event_type_value():
    assert WorkflowEventType.HANDOFF == "Handoff"
    assert WorkflowEventType.HANDOFF.value == "Handoff"


def test_build_handoff_payload_minimal():
    payload = build_handoff_payload("did:aip:sender", "sess-123")
    assert payload == {
        "source": "workflow-telemetry",
        "event_type": "Handoff",
        "from_agent_did": "did:aip:sender",
        "multi_agent_session_id": "sess-123",
    }
    # Receiver is derived server-side — never sent.
    assert "agent_did" not in payload


def test_build_handoff_payload_with_timestamp():
    ts = datetime(2026, 5, 27, 12, 0, 0, 500000, tzinfo=timezone.utc)
    payload = build_handoff_payload("did:aip:sender", "sess-123", ts)
    assert payload["timestamp"] == "2026-05-27T12:00:00.500Z"


@pytest.mark.parametrize(
    "from_did, session_id",
    [
        ("", "sess"),
        ("   ", "sess"),
        ("did:aip:sender", ""),
        ("did:aip:sender", "   "),
        (None, "sess"),
        ("did:aip:sender", None),
    ],
)
def test_build_handoff_payload_rejects_empty(from_did, session_id):
    with pytest.raises(ValueError):
        build_handoff_payload(from_did, session_id)


@pytest.mark.asyncio
async def test_emit_handoff_routes_through_activity_dispatcher():
    sender = AsyncMock(return_value={"verdict": "allow"})
    with (
        patch("openbox.workflow_interceptor._send_governance_event", sender),
        patch("temporalio.workflow.patched", return_value=False),
    ):
        result = await emit_handoff(
            multi_agent_session_id="sess-123",
            from_agent_did="did:aip:sender",
            timeout=12.0,
            on_api_error="fail_closed",
        )

    assert result == {"verdict": "allow"}
    sender.assert_awaited_once()
    payload, timeout, on_api_error = sender.await_args.args
    assert payload["event_type"] == "Handoff"
    assert payload["from_agent_did"] == "did:aip:sender"
    assert payload["multi_agent_session_id"] == "sess-123"
    assert timeout == 12.0
    assert on_api_error == "fail_closed"


def _handoff_request(new_input="fixed"):
    from openbox.patch import PatchRequest

    return PatchRequest(
        schema_version=1,
        new_input=new_input,
        governance_event_id="evt",
        reason="patch",
        event_type="Handoff",
        hook_trigger=False,
        hook_stage=None,
    )


@pytest.mark.asyncio
async def test_emit_handoff_patch_submits_to_coordinator_and_raises_control():
    """A BLOCK-with-patch handoff submits to the run-local coordinator and unwinds
    user code via PatchControl (CAN stays owned by the interceptor)."""
    from openbox.patch_coordinator import (
        PatchControl,
        PatchCoordinator,
        bind_coordinator,
        unbind_coordinator,
    )

    req = _handoff_request("via-handoff")
    sender = AsyncMock(return_value=req)
    coord = PatchCoordinator()
    token = bind_coordinator(coord)
    try:
        with (
            patch("openbox.workflow_interceptor._send_governance_event", sender),
            patch("temporalio.workflow.patched", return_value=True),
        ):
            with pytest.raises(PatchControl):
                await emit_handoff(
                    multi_agent_session_id="s", from_agent_did="did:aip:x"
                )
    finally:
        unbind_coordinator(token)

    assert coord.get_request() == req


@pytest.mark.asyncio
async def test_emit_handoff_patch_without_coordinator_fails_safe_as_block():
    """No coordinator bound (should not happen in a patched run) → fail safe as a
    plain governance block, never a silent continuation."""
    from temporalio.exceptions import ApplicationError

    from openbox.errors import GOVERNANCE_BLOCK_ERROR_TYPE
    from openbox.patch_coordinator import get_coordinator

    assert get_coordinator() is None  # no active binding
    sender = AsyncMock(return_value=_handoff_request())
    with (
        patch("openbox.workflow_interceptor._send_governance_event", sender),
        patch("temporalio.workflow.patched", return_value=True),
    ):
        with pytest.raises(ApplicationError) as exc:
            await emit_handoff(multi_agent_session_id="s", from_agent_did="did:aip:x")
    assert exc.value.type == GOVERNANCE_BLOCK_ERROR_TYPE


@pytest.mark.asyncio
async def test_emit_handoff_allow_unchanged_on_patched_path():
    """ALLOW returns the dict unchanged even when the patch-restart path is enabled."""
    sender = AsyncMock(return_value={"verdict": "allow"})
    with (
        patch("openbox.workflow_interceptor._send_governance_event", sender),
        patch("temporalio.workflow.patched", return_value=True),
    ):
        result = await emit_handoff(
            multi_agent_session_id="s", from_agent_did="did:aip:x"
        )
    assert result == {"verdict": "allow"}


@pytest.mark.asyncio
async def test_emit_handoff_validates_before_network():
    sender = AsyncMock()
    with patch("openbox.workflow_interceptor._send_governance_event", sender):
        with pytest.raises(ValueError):
            await emit_handoff(multi_agent_session_id="", from_agent_did="did:aip:x")
    sender.assert_not_awaited()


def test_inject_and_read_session_header_round_trip():
    conv = _converter()
    headers = inject_session_header({}, "sess-xyz", conv)
    assert HEADER_KEY in headers
    assert read_session_from_header(headers, conv) == "sess-xyz"


def test_inject_session_header_preserves_existing():
    conv = _converter()
    existing = {"other": conv.to_payload("keep")}
    headers = inject_session_header(existing, "sess", conv)
    assert "other" in headers
    assert HEADER_KEY in headers


def test_read_session_from_header_none_and_empty():
    conv = _converter()
    assert read_session_from_header(None, conv) is None
    assert read_session_from_header({}, conv) is None
    assert read_session_from_header({"unrelated": conv.to_payload("x")}, conv) is None


def test_memo_and_header_keys_are_distinct():
    # Memo key is app-facing; header key is internal wire propagation.
    assert MEMO_KEY == "openbox_multi_agent_session_id"
    assert HEADER_KEY == "openbox-multi-agent-session-id"
    assert MEMO_KEY != HEADER_KEY
