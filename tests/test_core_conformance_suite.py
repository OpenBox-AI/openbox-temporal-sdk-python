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
