"""Replay determinism tests for BLOCK-with-Patch Workflow Restart.

These record REAL histories (real governance interceptors + real
send_governance_event activity against a fake HTTP Core) and replay them on a
worker carrying the full feature, asserting no non-determinism error. Critically,
one test records a history that actually contains the ``openbox-retryable-block-v1``
patch marker AND a Continue-As-New command, and replays it — the guarantee that
the restructured ``execute_workflow`` (asyncio task race + CAN) is replay-safe.

A separate test records a history with NO OpenBox interceptor at all (a
"pre-feature" / upgrade history) and replays it on the full-feature worker: the
patch markers are absent, so the legacy path is taken and replay stays clean.
"""

import json
import threading
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

import pytest
from temporalio import activity, workflow
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer, Worker

PATCH_BASE = "openbox.plugin"


def _BLOCK(new_input):
    return {"verdict": "block", "patch": {"new_input": new_input}}


_ALLOW = {"verdict": "allow"}


def _start_fake_core(script):
    """Fake Core serving POST /evaluate + GET /auth/validate. ``script`` maps an
    event_type to a list of responses consumed by call index (last repeats)."""
    counts: dict[str, int] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self._respond({"valid": True})

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            try:
                data = json.loads(self.rfile.read(length)) if length else {}
            except json.JSONDecodeError:
                data = {}
            event_type = data.get("event_type") or data.get("payload", {}).get(
                "event_type", "unknown"
            )
            counts[event_type] = counts.get(event_type, 0) + 1
            responses = script.get(event_type)
            resp = (
                responses[min(counts[event_type] - 1, len(responses) - 1)]
                if responses
                else _ALLOW
            )
            self._respond(resp)

        def _respond(self, body):
            payload = json.dumps(body).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _real_plugin(core_url, **overrides):
    from openbox.plugin import OpenBoxPlugin

    with patch(f"{PATCH_BASE}.validate_api_key"):
        return OpenBoxPlugin(
            openbox_url=core_url, openbox_api_key="obx_test_real", **overrides
        )


@activity.defn
async def simple_activity(value: str) -> str:
    return f"processed:{value}"


@workflow.defn
class SimpleWorkflow:
    @workflow.run
    async def run(self, value: str) -> str:
        return await workflow.execute_activity(
            simple_activity, value, start_to_close_timeout=timedelta(seconds=10)
        )


class TestPatchReplay:
    @pytest.fixture
    async def env(self):
        async with await WorkflowEnvironment.start_local() as env:
            yield env

    async def test_replay_real_governed_allow_history_clean(self, env):
        """A real governed run (marker recorded, governance activities present)
        replays cleanly on the full-feature worker."""
        server = _start_fake_core({})  # all ALLOW
        url = f"http://127.0.0.1:{server.server_address[1]}"
        plugin = _real_plugin(url)
        try:
            async with Worker(
                env.client,
                task_queue="tq-replay-allow",
                workflows=[SimpleWorkflow],
                activities=[simple_activity],
                plugins=[plugin],
            ):
                handle = await env.client.start_workflow(
                    SimpleWorkflow.run,
                    "hello",
                    id="wf-replay-allow",
                    task_queue="tq-replay-allow",
                )
                assert await handle.result() == "processed:hello"
                history = await handle.fetch_history()

            await Replayer(
                workflows=[SimpleWorkflow], plugins=[plugin]
            ).replay_workflow(history)
        finally:
            server.shutdown()

    async def test_replay_continue_as_new_history_clean(self, env):
        """THE determinism guarantee: a first-run history that records the patch
        marker AND a Continue-As-New (WorkflowStarted BLOCK+patch) replays with no
        non-determinism error."""
        server = _start_fake_core({"WorkflowStarted": [_BLOCK("v2"), _ALLOW]})
        url = f"http://127.0.0.1:{server.server_address[1]}"
        plugin = _real_plugin(url)
        try:
            async with Worker(
                env.client,
                task_queue="tq-replay-can",
                workflows=[SimpleWorkflow],
                activities=[simple_activity],
                plugins=[plugin],
            ):
                handle = await env.client.start_workflow(
                    SimpleWorkflow.run,
                    "v1",
                    id="wf-replay-can",
                    task_queue="tq-replay-can",
                )
                assert await handle.result() == "processed:v2"  # CAN happened
                # The handle points at the FIRST run, whose history ends in the
                # ContinueAsNew command (and carries the patch marker).
                first_run_history = await handle.fetch_history()

            await Replayer(
                workflows=[SimpleWorkflow], plugins=[plugin]
            ).replay_workflow(first_run_history)
        finally:
            server.shutdown()

    async def test_replay_pre_feature_history_on_new_worker_clean(self, env):
        """Upgrade path: a history recorded with NO OpenBox interceptor (marker
        absent) replays cleanly on the full-feature worker — the legacy path is
        taken (all patched(...) return False), so no new commands are emitted."""
        # Record with a BARE worker (no governance interceptors / no marker).
        async with Worker(
            env.client,
            task_queue="tq-replay-prefeature",
            workflows=[SimpleWorkflow],
            activities=[simple_activity],
        ):
            handle = await env.client.start_workflow(
                SimpleWorkflow.run,
                "legacy",
                id="wf-replay-prefeature",
                task_queue="tq-replay-prefeature",
            )
            assert await handle.result() == "processed:legacy"
            history = await handle.fetch_history()

        # Replay on the FULL-feature worker; the missing marker must not crash or
        # trigger Continue-As-New.
        server = _start_fake_core({})
        url = f"http://127.0.0.1:{server.server_address[1]}"
        plugin = _real_plugin(url)
        try:
            await Replayer(
                workflows=[SimpleWorkflow], plugins=[plugin]
            ).replay_workflow(history)
        finally:
            server.shutdown()

    async def test_replay_can_history_deterministic_across_repeats(self, env):
        """Replaying the Continue-As-New history repeatedly is stable."""
        server = _start_fake_core({"WorkflowStarted": [_BLOCK("v2"), _ALLOW]})
        url = f"http://127.0.0.1:{server.server_address[1]}"
        plugin = _real_plugin(url)
        try:
            async with Worker(
                env.client,
                task_queue="tq-replay-repeat",
                workflows=[SimpleWorkflow],
                activities=[simple_activity],
                plugins=[plugin],
            ):
                handle = await env.client.start_workflow(
                    SimpleWorkflow.run,
                    "v1",
                    id="wf-replay-repeat",
                    task_queue="tq-replay-repeat",
                )
                await handle.result()
                history = await handle.fetch_history()

            replayer = Replayer(workflows=[SimpleWorkflow], plugins=[plugin])
            for _ in range(3):
                await replayer.replay_workflow(history)
        finally:
            server.shutdown()
