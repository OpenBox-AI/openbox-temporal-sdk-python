"""Deterministic Temporal-context hook parity (no live OpenAI credentials).

Proves that when base instrumentation is installed through the Temporal
runtime/adapter and a Temporal-shaped ActivityContext is bound, an OpenAI-shaped
chat/completions HTTP call emits the FLAT hook interface Core consumes:

- top-level ``hook_type == "http_request"``, ``http_url``, ``http_method``
- NOT a nested shape where the useful OTel data lives only under ``data.otel``

``semantic_type`` (e.g. ``llm_completion``) is intentionally NOT set by the SDK
— the base wire contract computes it at Core from this flat span. This test
asserts the flat inputs to that classification; the optional live-OpenAI smoke
test is separate.
"""

import httpx
import pytest
from openbox_core.config import OpenBoxConfig
from openbox_core.conformance.fake_core import FakeCore, fake_client
from openbox_core.conformance.instrumentation import LocalCountingServer
from openbox_core.context import activity_scope
from openbox_core.contracts.context import ActivityContext
from openbox_core.contracts.results import Verdict
from openbox_core.instrumentation.manager import InstrumentationManager
from openbox_core.runtime import OpenBoxRuntime
from temporalio.exceptions import ApplicationError

from openbox.core_adapter import TemporalFrameworkAdapter, get_core_context_store
from openbox.governance_state import TemporalGovernanceState


@pytest.fixture(scope="module")
def server():
    srv = LocalCountingServer()  # loopback server, responds to any path
    yield srv
    srv.stop()


def _chat_url(server) -> str:
    """OpenAI-shaped path on the loopback server (base url ends in /echo)."""
    return server.url.replace("/echo", "/v1/chat/completions")


def _temporal_runtime(
    fake_core: FakeCore, state: TemporalGovernanceState | None = None
) -> tuple[OpenBoxRuntime, InstrumentationManager, TemporalGovernanceState]:
    """Runtime wired exactly like create_core_runtime, but with a fake Core client
    so the emitted hook payload can be inspected without a live network. Returns
    the TemporalGovernanceState too so completed-hook bridging can be asserted
    (a caller may also pass one in to inspect it after the request)."""
    store = get_core_context_store()
    state = state if state is not None else TemporalGovernanceState()
    config = OpenBoxConfig(
        api_url="https://core.test", api_key="obx_test_llm_parity"
    ).normalized()
    adapter = TemporalFrameworkAdapter(state, context_store=store)
    runtime = OpenBoxRuntime(
        config, adapter, client=fake_client(fake_core), context_store=store
    )
    manager = InstrumentationManager(runtime)
    runtime._instrumentation_manager = manager
    return runtime, manager, state


class _FakeActivityInfo:
    workflow_id = "wf-llm"
    workflow_run_id = "run-llm"
    workflow_type = "OpenAIWorkflow"
    task_queue = "llm-queue"
    activity_id = "act-llm"
    activity_type = "call_openai"
    attempt = 1


def test_openai_chat_completion_emits_flat_http_hook(server):
    chat_url = _chat_url(server)
    fake_core = FakeCore({"verdict": "allow"}, {"verdict": "allow"})
    runtime, manager, _state = _temporal_runtime(fake_core)
    manager.install()
    try:
        ctx = ActivityContext(
            workflow_id=_FakeActivityInfo.workflow_id,
            run_id=_FakeActivityInfo.workflow_run_id,
            workflow_type=_FakeActivityInfo.workflow_type,
            task_queue=_FakeActivityInfo.task_queue,
            activity_id=_FakeActivityInfo.activity_id,
            activity_type=_FakeActivityInfo.activity_type,
        )
        with activity_scope(ctx, store=runtime.context_store):
            resp = httpx.post(
                chat_url, json={"model": "gpt-4", "messages": []}, timeout=5
            )
            assert resp.status_code == 200
    finally:
        manager.uninstall()

    assert fake_core.started_payloads, "no started hook payload captured"
    started = fake_core.started_payloads[-1]["spans"][0]

    # FLAT, hook-oriented interface (top-level fields, not nested under data.otel).
    assert started["hook_type"] == "http_request"
    assert started["http_method"] == "POST"
    assert started["http_url"].endswith("/v1/chat/completions")
    assert started["stage"] == "started"
    # The nested OTel shape must NOT be how data is carried.
    assert "data" not in started or "otel" not in (started.get("data") or {})

    # Completed stage carries the same flat contract.
    assert fake_core.completed_payloads, "no completed hook payload captured"
    completed = fake_core.completed_payloads[-1]["spans"][0]
    assert completed["hook_type"] == "http_request"
    assert completed["http_url"].endswith("/v1/chat/completions")
    assert completed["stage"] == "completed"


def test_started_block_prevents_openai_call(server):
    """A BLOCK verdict on the started hook stops the request before it is sent."""
    from openbox_core.errors import GovernanceBlockedError

    chat_url = _chat_url(server)
    fake_core = FakeCore({"verdict": "block", "reason": "no external llm"})
    runtime, manager, _state = _temporal_runtime(fake_core)
    manager.install()
    try:
        ctx = ActivityContext(
            workflow_id="wf-b",
            run_id="run-b",
            workflow_type="W",
            task_queue="q",
            activity_id="act-b",
            activity_type="call_openai",
        )
        before = server.hits
        with activity_scope(ctx, store=runtime.context_store):
            # Base adapter raises a Temporal-native ApplicationError; the base
            # defense-in-depth surfaces GovernanceBlockedError if it were to slip.
            with pytest.raises(Exception) as exc:
                httpx.post(chat_url, json={"model": "gpt-4"}, timeout=5)
            # It must be a governance stop, not a transport error.
            name = type(exc.value).__name__
            assert (
                "Governance" in name
                or "ApplicationError" in name
                or isinstance(exc.value, GovernanceBlockedError)
            )
        assert server.hits == before  # request never reached the server
    finally:
        manager.uninstall()


def test_started_block_with_retry_plan_raises_governance_retryable_block(server):
    """A BLOCK verdict with a valid retry plan on the started hook requests a
    workflow restart (GovernanceRetryableBlock) instead of the plain
    GovernanceBlock — checked before the generic BLOCK/HALT mapping. Started
    hooks are wire-tagged event_type=ActivityStarted + hook_trigger=True with
    hook_stage="started"."""
    chat_url = _chat_url(server)
    fake_core = FakeCore(
        {
            "verdict": "block",
            "reason": "rerouting to an approved model",
            "governance_event_id": "evt_hook",
            "retry_plan": {"new_input": {"model": "gpt-4o-mini"}},
        }
    )
    runtime, manager, _state = _temporal_runtime(fake_core)
    manager.install()
    try:
        ctx = ActivityContext(
            workflow_id="wf-retry",
            run_id="run-retry",
            workflow_type="W",
            task_queue="q",
            activity_id="act-retry",
            activity_type="call_openai",
        )
        before = server.hits
        with activity_scope(ctx, store=runtime.context_store):
            with pytest.raises(ApplicationError) as exc_info:
                httpx.post(chat_url, json={"model": "gpt-4"}, timeout=5)
        assert server.hits == before  # request never reached the server
    finally:
        manager.uninstall()

    assert exc_info.value.type == "GovernanceRetryableBlock"
    assert exc_info.value.non_retryable is True
    details = exc_info.value.details[0]
    assert details["new_input"] == {"model": "gpt-4o-mini"}
    assert details["governance_event_id"] == "evt_hook"
    assert details["event_type"] == "ActivityStarted"
    assert details["hook_trigger"] is True
    assert details["hook_stage"] == "started"


def test_completed_hook_block_with_retry_plan_records_retryable_request(server):
    """A BLOCK verdict with a valid retry plan on the COMPLETED hook never
    raises — completed telemetry never undoes the operation. It records the
    full retryable request in TemporalGovernanceState (hook_stage="completed")
    for the activity interceptor to raise after user code returns."""
    chat_url = _chat_url(server)
    fake_core = FakeCore(
        {"verdict": "allow"},  # started hook: proceed
        {
            "verdict": "block",
            "reason": "flagged after the fact",
            "retry_plan": {"new_input": {"model": "gpt-4o-mini"}},
        },  # completed hook: retryable block
    )
    state = TemporalGovernanceState()
    runtime, manager, _ = _temporal_runtime(fake_core, state=state)
    manager.install()
    try:
        ctx = ActivityContext(
            workflow_id="wf-completed-retry",
            run_id="run-completed-retry",
            workflow_type="W",
            task_queue="q",
            activity_id="act-completed-retry",
            activity_type="call_openai",
        )
        with activity_scope(ctx, store=runtime.context_store):
            resp = httpx.post(chat_url, json={"model": "gpt-4"}, timeout=5)
            assert resp.status_code == 200  # operation ran; never undone
    finally:
        manager.uninstall()

    stop = state.take_completed_stop(
        "wf-completed-retry", "run-completed-retry", "act-completed-retry"
    )
    assert stop is not None
    assert stop.verdict is Verdict.BLOCK
    assert stop.request is not None
    assert stop.request.new_input == {"model": "gpt-4o-mini"}
    assert stop.request.event_type == "ActivityStarted"
    assert stop.request.hook_trigger is True
    assert stop.request.hook_stage == "completed"
