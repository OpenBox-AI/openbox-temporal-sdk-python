"""End-to-end tests for BLOCK-with-Patch Workflow Restart.

These drive the REAL governance stack — the workflow interceptor, activity
interceptor, base core runtime, and the real ``send_governance_event`` activity —
against a fake HTTP Core (localhost) on an ephemeral Temporal server. Only
``validate_api_key`` is patched (to skip the network key check at plugin init);
everything under test runs for real, so these tests genuinely exercise the
``asyncio.create_task`` race + ``workflow.wait_condition`` + real
``ContinueAsNewError`` propagation + memo round-trip across an actual restart.

Assertions cover: same Workflow ID, new Run ID, prior run ``ContinuedAsNew``,
replacement ``new_input`` delivered (and ``new_input=None`` reuses args), the
blocked governance activity is not auto-retried, and the restart budget
terminates the chain with ``GovernancePatchLimitExceeded``.

The declared floor is ``temporalio>=1.23.0``; the APIs the feature relies on
(``workflow.patched``, ``workflow.memo``/``memo_value``,
``workflow.all_handlers_finished``, in-sandbox ``asyncio.create_task``,
``workflow.continue_as_new(memo=...)``) all predate that floor (the newest,
``all_handlers_finished``, landed in temporalio 1.6.0). Confirm in a CI matrix leg.
"""

import json
import threading
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

import pytest
from temporalio import activity, workflow
from temporalio.client import WorkflowExecutionStatus, WorkflowFailureError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from openbox.errors import GOVERNANCE_PATCH_LIMIT_EXCEEDED_ERROR_TYPE

PATCH_BASE = "openbox.plugin"


def _BLOCK(new_input):
    """A BLOCK-with-patch governance response carrying a valid patch."""
    return {"verdict": "block", "patch": {"new_input": new_input}}


_ALLOW = {"verdict": "allow"}


def _start_fake_core(script):
    """Start a fake OpenBox Core on a background thread.

    ``script``: ``{event_type: [response, ...]}`` — responses are consumed by
    per-event call index (the last entry repeats). Unlisted event types return
    ALLOW. Serves POST ``/api/v1/governance/evaluate`` (BaseHTTPRequestHandler
    dispatches POST to ``do_POST``) and GET ``/api/v1/auth/validate``.

    Returns ``(server, counts)`` where ``counts[event_type]`` is the number of
    evaluate calls seen for that event — used to prove the blocked activity is
    not auto-retried.
    """
    counts: dict[str, int] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 (BaseHTTPRequestHandler dispatch name)
            self._respond({"valid": True})

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            try:
                data = json.loads(self.rfile.read(length)) if length else {}
            except json.JSONDecodeError:
                data = {}
            # The evaluate request carries the event fields at the TOP level
            # (send_governance_event posts {**event_payload, "timestamp": ...});
            # fall back to a nested "payload" only defensively.
            event_type = data.get("event_type") or data.get("payload", {}).get(
                "event_type", "unknown"
            )
            counts[event_type] = counts.get(event_type, 0) + 1
            responses = script.get(event_type)
            if responses:
                resp = responses[min(counts[event_type] - 1, len(responses) - 1)]
            else:
                resp = _ALLOW
            self._respond(resp)

        def _respond(self, body):
            payload = json.dumps(body).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):  # silence server logging
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, counts


def _has_error_type(exc: BaseException, error_type: str) -> bool:
    """True if any ApplicationError in the cause chain has ``type == error_type``."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if getattr(current, "type", None) == error_type:
            return True
        current = (
            getattr(current, "cause", None)
            or getattr(current, "__cause__", None)
            or getattr(current, "__context__", None)
        )
    return False


@activity.defn
async def echo_activity(value: str) -> str:
    return f"processed:{value}"


@workflow.defn
class EchoWorkflow:
    """Runs one activity and returns its result — the arg is what a patch
    replaces, so the final result reveals which input the last run received."""

    @workflow.run
    async def run(self, value: str) -> str:
        return await workflow.execute_activity(
            echo_activity, value, start_to_close_timeout=timedelta(seconds=10)
        )


def _real_plugin(core_url: str, **overrides):
    """Build a REAL OpenBoxPlugin pointed at the fake Core (only the network key
    check is skipped)."""
    from openbox.plugin import OpenBoxPlugin

    with patch(f"{PATCH_BASE}.validate_api_key"):
        return OpenBoxPlugin(
            openbox_url=core_url, openbox_api_key="obx_test_real", **overrides
        )


class TestPatchRealContinueAsNew:
    @pytest.fixture
    async def env(self):
        async with await WorkflowEnvironment.start_local() as env:
            yield env

    async def _first_and_latest(self, env, handle):
        """(first_run_id, latest_run_id, first_run_status) for the workflow chain."""
        first_run_id = handle.first_execution_run_id
        latest_desc = await env.client.get_workflow_handle(handle.id).describe()
        first_desc = await env.client.get_workflow_handle(
            handle.id, run_id=first_run_id
        ).describe()
        return first_run_id, latest_desc.run_id, first_desc.status

    async def test_workflow_started_block_patch_restarts_with_new_input(self, env):
        """WorkflowStarted BLOCK+patch → real Continue-As-New before user code;
        the next run receives the replacement input; the blocked activity is not
        auto-retried."""
        server, counts = _start_fake_core(
            {"WorkflowStarted": [_BLOCK("replaced"), _ALLOW]}
        )
        url = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            async with Worker(
                env.client,
                task_queue="tq-started",
                workflows=[EchoWorkflow],
                activities=[echo_activity],
                plugins=[_real_plugin(url)],
            ):
                handle = await env.client.start_workflow(
                    EchoWorkflow.run,
                    "original",
                    id="wf-started-can",
                    task_queue="tq-started",
                )
                result = await handle.result()

            # new_input replaced the original arg — only possible via a real CAN
            assert result == "processed:replaced"
            first_id, latest_id, first_status = await self._first_and_latest(
                env, handle
            )
            assert latest_id != first_id  # new Run ID, same Workflow ID
            assert first_status == WorkflowExecutionStatus.CONTINUED_AS_NEW
            # exactly one evaluate per run (blocked non-retryable activity not retried)
            assert counts["WorkflowStarted"] == 2
        finally:
            server.shutdown()

    async def test_new_input_none_reuses_current_args(self, env):
        """WorkflowStarted BLOCK+patch with new_input=None reuses the current run's
        args exactly on restart."""
        server, _ = _start_fake_core({"WorkflowStarted": [_BLOCK(None), _ALLOW]})
        url = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            async with Worker(
                env.client,
                task_queue="tq-none",
                workflows=[EchoWorkflow],
                activities=[echo_activity],
                plugins=[_real_plugin(url)],
            ):
                result = await env.client.execute_workflow(
                    EchoWorkflow.run,
                    "keep-me",
                    id="wf-none-reuse",
                    task_queue="tq-none",
                )
            assert result == "processed:keep-me"  # None → original args reused
        finally:
            server.shutdown()

    async def test_workflow_completed_block_patch_restarts(self, env):
        """WorkflowCompleted BLOCK+patch → Continue-As-New instead of returning the
        user result."""
        server, _ = _start_fake_core({"WorkflowCompleted": [_BLOCK("after"), _ALLOW]})
        url = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            async with Worker(
                env.client,
                task_queue="tq-completed",
                workflows=[EchoWorkflow],
                activities=[echo_activity],
                plugins=[_real_plugin(url)],
            ):
                result = await env.client.execute_workflow(
                    EchoWorkflow.run,
                    "first",
                    id="wf-completed-can",
                    task_queue="tq-completed",
                )
            assert result == "processed:after"
        finally:
            server.shutdown()

    async def test_restart_limit_terminates_chain(self, env):
        """An always-BLOCK policy is bounded: the chain stops at the configured cap
        with a non-retryable GovernancePatchLimitExceeded, not an unbounded loop."""
        server, counts = _start_fake_core({"WorkflowStarted": [_BLOCK("loop")]})
        url = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            async with Worker(
                env.client,
                task_queue="tq-limit",
                workflows=[EchoWorkflow],
                activities=[echo_activity],
                plugins=[_real_plugin(url, max_patch_restarts=2)],
            ):
                handle = await env.client.start_workflow(
                    EchoWorkflow.run, "x", id="wf-limit", task_queue="tq-limit"
                )
                with pytest.raises(WorkflowFailureError) as exc_info:
                    await handle.result()

            assert _has_error_type(
                exc_info.value, GOVERNANCE_PATCH_LIMIT_EXCEEDED_ERROR_TYPE
            )
            # cap=2: runs with memo counts 0,1,2 evaluate WorkflowStarted; the 3rd
            # increment (→3) exceeds the cap and fails. Bounded, not runaway.
            assert counts["WorkflowStarted"] == 3
        finally:
            server.shutdown()

    async def test_allow_completes_normally_through_real_transport(self, env):
        """Sanity: with no patch, the real transport lets the workflow run and
        complete unchanged (no restart)."""
        server, counts = _start_fake_core({})  # all ALLOW
        url = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            async with Worker(
                env.client,
                task_queue="tq-allow",
                workflows=[EchoWorkflow],
                activities=[echo_activity],
                plugins=[_real_plugin(url, max_patch_restarts=3)],
            ):
                handle = await env.client.start_workflow(
                    EchoWorkflow.run, "plain", id="wf-allow", task_queue="tq-allow"
                )
                result = await handle.result()
            assert result == "processed:plain"
            _, latest_id, first_status = await self._first_and_latest(env, handle)
            assert first_status == WorkflowExecutionStatus.COMPLETED  # no CAN
        finally:
            server.shutdown()
