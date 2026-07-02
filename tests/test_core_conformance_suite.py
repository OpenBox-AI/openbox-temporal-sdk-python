"""Run the base-SDK conformance kit with the TEMPORAL FrameworkAdapter.

Proves the Temporal adapter drives core instrumentation with native
semantics: started BLOCK/HALT surface as non-retryable ApplicationErrors with
the existing ``GovernanceBlock``/``GovernanceHalt`` types BEFORE the real
operation runs; REQUIRE_APPROVAL surfaces as the retryable ``ApprovalPending``
error powering Temporal's HITL retry loop; completed verdicts only mark
future execution. This is the Phase 8 kit imported from an external repo —
exactly what later framework migrations will do.
"""

from __future__ import annotations

import pytest
import requests
from temporalio.exceptions import ApplicationError

from openbox.core_adapter import TemporalFrameworkAdapter, build_core_activity_context
from openbox_core.conformance.fake_core import FakeCore, assert_hook_wire_shape
from openbox_core.conformance.instrumentation import (
    LocalCountingServer,
    installed_conformance_runtime,
)
from openbox_core.context import ContextStore, activity_scope


class _FakeActivityInfo:
    workflow_id = "wf-conf"
    workflow_run_id = "run-conf"
    workflow_type = "ConfWorkflow"
    task_queue = "conf-queue"
    activity_id = "act-conf"
    activity_type = "conf_activity"
    attempt = 1


@pytest.fixture(scope="module")
def server():
    server = LocalCountingServer()
    yield server
    server.stop()


def temporal_context():
    return build_core_activity_context(_FakeActivityInfo(), activity_input=[1, 2])


class TestTemporalAdapterConformance:
    def test_http_block_raises_governance_block_application_error(self, server):
        fake_core = FakeCore({"verdict": "block", "reason": "policy"})
        adapter, store = TemporalFrameworkAdapter(), ContextStore()
        with installed_conformance_runtime(fake_core, adapter, store):
            with activity_scope(temporal_context(), store=store):
                before = server.hits
                with pytest.raises(ApplicationError) as exc_info:
                    requests.get(server.url, timeout=5)
                assert server.hits == before  # blocked BEFORE the request
        assert exc_info.value.type == "GovernanceBlock"
        assert exc_info.value.non_retryable is True
        assert_hook_wire_shape(fake_core.started_payloads[0])

    def test_http_halt_raises_governance_halt_and_sets_halt_flag(self, server):
        fake_core = FakeCore({"verdict": "halt", "reason": "emergency"})
        adapter, store = TemporalFrameworkAdapter(), ContextStore()
        with installed_conformance_runtime(fake_core, adapter, store):
            with activity_scope(temporal_context(), store=store):
                before = server.hits
                with pytest.raises(ApplicationError) as exc_info:
                    requests.get(server.url, timeout=5)
                assert server.hits == before
                assert store.halt_requested  # adapter/worker decides terminate()
        assert exc_info.value.type == "GovernanceHalt"
        assert exc_info.value.non_retryable is True

    async def test_require_approval_surfaces_retryable_approval_pending(self, server):
        """Async path: the adapter maps approval to Temporal's HITL retry loop
        (retryable ApprovalPending), so the operation does NOT run now."""
        import httpx

        fake_core = FakeCore({"verdict": "require_approval", "approval_id": "app-1"})
        adapter, store = TemporalFrameworkAdapter(), ContextStore()
        with installed_conformance_runtime(fake_core, adapter, store):
            with activity_scope(temporal_context(), store=store):
                before = server.hits
                async with httpx.AsyncClient() as client:
                    with pytest.raises(ApplicationError) as exc_info:
                        await client.get(server.url)
                assert server.hits == before
        assert exc_info.value.type == "ApprovalPending"
        assert exc_info.value.non_retryable is False  # Temporal retries -> polls

    def test_completed_block_marks_future_only(self, server):
        fake_core = FakeCore(
            {"verdict": "allow"},
            {"verdict": "block", "reason": "post-hoc"},
        )
        adapter, store = TemporalFrameworkAdapter(), ContextStore()
        ctx = temporal_context()
        with installed_conformance_runtime(fake_core, adapter, store):
            with activity_scope(ctx, store=store):
                response = requests.get(server.url, timeout=5)  # runs; NOT undone
                assert response.status_code == 200
        assert store.is_activity_aborted(ctx.workflow_id, ctx.activity_id)

    def test_activity_context_carries_temporal_fields(self):
        ctx = temporal_context()
        payload = ctx.to_payload_fields()
        assert payload["workflow_id"] == "wf-conf"
        assert payload["run_id"] == "run-conf"
        assert payload["activity_id"] == "act-conf"
        assert payload["activity_type"] == "conf_activity"
        assert payload["attempt"] == 1  # metadata merged at top level
        assert payload["source"] == "workflow-telemetry"

    def test_create_core_runtime_builds_wired_runtime(self):
        from openbox.core_adapter import create_core_runtime, get_core_context_store

        runtime = create_core_runtime("https://core.test", "obx_test_x")
        assert runtime.adapter.name == "temporal"
        assert runtime.context_store is get_core_context_store()
        runtime.close()


class _FakeSpanProcessor:
    """Minimal legacy span-processor stand-in: one buffer, keyed by workflow."""

    def __init__(self, buffer):
        self._buffer = buffer

    def get_buffer(self, workflow_id):
        return self._buffer if workflow_id == self._buffer.workflow_id else None


def _legacy_buffer():
    from openbox.types import WorkflowSpanBuffer

    return WorkflowSpanBuffer(
        workflow_id="wf-conf",
        run_id="run-conf",
        workflow_type="ConfWorkflow",
        task_queue="conf-queue",
    )


class TestApprovalBridgesLegacyHitlBuffer:
    """Core REQUIRE_APPROVAL must arm the legacy retry-poll loop: without
    ``buffer.pending_approval`` the next attempt re-evaluates from scratch
    instead of polling approval status."""

    def test_sync_hook_approval_raises_retryable_pending_and_flags_buffer(self, server):
        fake_core = FakeCore({"verdict": "require_approval", "approval_id": "app-2"})
        buffer = _legacy_buffer()
        store = ContextStore()
        adapter = TemporalFrameworkAdapter(_FakeSpanProcessor(buffer), context_store=store)
        with installed_conformance_runtime(fake_core, adapter, store):
            with activity_scope(temporal_context(), store=store):
                before = server.hits
                with pytest.raises(ApplicationError) as exc_info:
                    requests.get(server.url, timeout=5)
                assert server.hits == before
        assert exc_info.value.type == "ApprovalPending"
        assert exc_info.value.non_retryable is False
        assert buffer.pending_approval is True
        # Temporal HITL is retry-driven: the core inline poller must not run.
        assert fake_core.approval_requests == []

    async def test_async_approval_flags_buffer_too(self):
        from openbox_core.contracts.results import EvaluationResult, Verdict

        buffer = _legacy_buffer()
        adapter = TemporalFrameworkAdapter(_FakeSpanProcessor(buffer))
        store = adapter._store
        with activity_scope(temporal_context(), store=store):
            with pytest.raises(ApplicationError) as exc_info:
                await adapter.handle_approval(
                    EvaluationResult(verdict=Verdict.REQUIRE_APPROVAL, approval_id="a")
                )
        assert exc_info.value.type == "ApprovalPending"
        assert buffer.pending_approval is True

    def test_hitl_disabled_degrades_to_non_retryable_block(self):
        from openbox_core.contracts.results import EvaluationResult, Verdict

        adapter = TemporalFrameworkAdapter(hitl_enabled=False)
        with activity_scope(temporal_context(), store=adapter._store):
            with pytest.raises(ApplicationError) as exc_info:
                adapter.handle_approval_sync(
                    EvaluationResult(verdict=Verdict.REQUIRE_APPROVAL, approval_id="a")
                )
        assert exc_info.value.type == "GovernanceBlock"
        assert exc_info.value.non_retryable is True

    def test_skip_hitl_activity_type_degrades_to_block(self):
        from openbox_core.contracts.results import EvaluationResult, Verdict

        adapter = TemporalFrameworkAdapter(
            hitl_enabled=True, skip_hitl_activity_types={"conf_activity"}
        )
        with activity_scope(temporal_context(), store=adapter._store):
            with pytest.raises(ApplicationError) as exc_info:
                adapter.handle_approval_sync(
                    EvaluationResult(verdict=Verdict.REQUIRE_APPROVAL, approval_id="a")
                )
        assert exc_info.value.type == "GovernanceBlock"


class TestFailClosedMapsToTemporalHalt:
    """Fail-closed evaluation failure = non-retryable GovernanceHalt (legacy
    parity) — never a generic retryable activity error."""

    def test_unreachable_core_fail_closed_halts_before_operation(self, server):
        import httpx

        from openbox_core.client import EvaluationClient
        from openbox_core.config import OpenBoxConfig
        from openbox_core.instrumentation.manager import InstrumentationManager
        from openbox_core.runtime import OpenBoxRuntime

        def down(request):
            raise httpx.ConnectError("core is down")

        transport = httpx.MockTransport(down)
        client = EvaluationClient(
            "https://core.test",
            "obx_test_conformance",
            on_api_error="fail_closed",
            transport=transport,
            async_transport=transport,
        )
        store = ContextStore()
        runtime = OpenBoxRuntime(
            OpenBoxConfig(api_url="https://core.test", api_key="obx_test_conformance"),
            TemporalFrameworkAdapter(),
            client=client,
            context_store=store,
        )
        manager = InstrumentationManager(runtime)
        manager.install()
        try:
            with activity_scope(temporal_context(), store=store):
                before = server.hits
                with pytest.raises(ApplicationError) as exc_info:
                    requests.get(server.url, timeout=5)
                assert server.hits == before  # halted BEFORE the request
        finally:
            manager.uninstall()
        assert exc_info.value.type == "GovernanceHalt"
        assert exc_info.value.non_retryable is True
