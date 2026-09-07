from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from openbox_core.contracts.results import EvaluationResult
from openbox_core.errors import GovernanceBlockedError

from openbox import hook_governance


def _configured_hook(monkeypatch, result):
    span = Mock()
    span.get_span_context.return_value = SimpleNamespace(
        trace_id=123, span_id=0x00F067AA0BA902B7
    )

    processor = Mock()
    processor.get_activity_abort.return_value = None
    processor.get_activity_context_by_trace.return_value = {
        "workflow_id": "wf-1",
        "run_id": "run-1",
        "activity_id": "act-1",
        "event_type": "ActivityStarted",
    }

    client = Mock()
    client.evaluate.return_value = result
    client.aevaluate = AsyncMock(return_value=result)

    handler = Mock()
    handler.handle_constrain_sync = Mock()
    handler.handle_constrain = AsyncMock()

    monkeypatch.setattr(hook_governance, "_api_url", "https://core.test")
    monkeypatch.setattr(hook_governance, "_span_processor", processor)
    monkeypatch.setattr(hook_governance, "_evaluation_client", client)
    monkeypatch.setattr(hook_governance, "_constrain_handler", handler, raising=False)
    return span, handler


def _constrain_result():
    return EvaluationResult.from_dict(
        {
            "verdict": "constrain",
            "reason": "behavior rule matched",
            "age_result": {"profile_id": "post-batch"},
        }
    )


@pytest.mark.asyncio
async def test_async_started_hook_forwards_constrain_to_framework_handler(monkeypatch):
    result = _constrain_result()
    span, handler = _configured_hook(monkeypatch, result)

    with pytest.raises(GovernanceBlockedError) as exc_info:
        await hook_governance.evaluate_async(
            span,
            identifier="https://example.com",
            span_data={"stage": "started", "hook_type": "http_request"},
        )

    assert exc_info.value.verdict.value == "constrain"
    handler.handle_constrain.assert_awaited_once()
    forwarded_result, context = handler.handle_constrain.await_args.args
    assert forwarded_result is result
    assert context.metadata["openbox.trigger_span_id"] == "00f067aa0ba902b7"
    hook_governance._span_processor.set_activity_abort.assert_called_once_with(
        "wf-1", "act-1", "behavior rule matched"
    )


def test_sync_started_hook_forwards_constrain_to_framework_handler(monkeypatch):
    result = _constrain_result()
    span, handler = _configured_hook(monkeypatch, result)

    with pytest.raises(GovernanceBlockedError) as exc_info:
        hook_governance.evaluate_sync(
            span,
            identifier="https://example.com",
            span_data={"stage": "started", "hook_type": "http_request"},
        )

    assert exc_info.value.verdict.value == "constrain"
    handler.handle_constrain_sync.assert_called_once()
    forwarded_result, context = handler.handle_constrain_sync.call_args.args
    assert forwarded_result is result
    assert context.metadata["openbox.trigger_span_id"] == "00f067aa0ba902b7"
    hook_governance._span_processor.set_activity_abort.assert_called_once_with(
        "wf-1", "act-1", "behavior rule matched"
    )


@pytest.mark.asyncio
async def test_completed_hook_does_not_dispatch_constrain(monkeypatch):
    result = _constrain_result()
    span, handler = _configured_hook(monkeypatch, result)

    await hook_governance.evaluate_async(
        span,
        identifier="https://example.com",
        span_data={"stage": "completed", "hook_type": "http_request"},
    )

    handler.handle_constrain.assert_not_awaited()
