from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openbox_sandbox import (
    GOVERNED_COMMAND_ACTIVITY_TYPE,
    SandboxCommandRequest,
    SandboxReceipt,
)
from temporalio.exceptions import ApplicationError

from openbox.sandbox.interceptor import (
    GovernedCommandInterceptor,
    _GovernedCommandActivityInterceptor,
)

from .sandbox_test_support import sandbox_config


def info(*, activity_type: str = GOVERNED_COMMAND_ACTIVITY_TYPE, attempt: int = 1):
    return SimpleNamespace(
        workflow_id="workflow-1",
        workflow_run_id="run-1",
        workflow_type="ProofWorkflow",
        task_queue="sandbox-queue",
        activity_id="activity-1",
        activity_type=activity_type,
        attempt=attempt,
    )


def receipt(profile_fingerprint: str) -> SandboxReceipt:
    return SandboxReceipt(
        1,
        "receipt-1",
        "nonce-1",
        "workflow-1",
        "constrain",
        "proof",
        "a" * 64,
        "b" * 64,
        "c" * 64,
        profile_fingerprint,
        "2026-07-22T00:00:00Z",
        "2026-07-22T00:05:00Z",
        "key-1",
        "signature",
    )


@pytest.mark.asyncio
async def test_governed_activity_executes_only_in_shared_engine() -> None:
    config, runtime = sandbox_config()
    next_interceptor = SimpleNamespace(execute_activity=AsyncMock())
    interceptor = _GovernedCommandActivityInterceptor(next_interceptor, config)
    heartbeats = []
    cancelled = asyncio.Future()
    input_value = SimpleNamespace(
        args=[SandboxCommandRequest("proof", {}).to_history_value()]
    )

    with (
        patch("openbox.sandbox.interceptor.activity.info", return_value=info()),
        patch(
            "openbox.sandbox.interceptor.activity.heartbeat",
            side_effect=lambda value: heartbeats.append(value),
        ),
        patch(
            "openbox.sandbox.interceptor.activity.wait_for_cancelled",
            new=lambda: cancelled,
        ),
    ):
        result = await interceptor.execute_activity(input_value)

    assert result.disposition == "executed_in_sandbox"
    assert result.profile_id == "proof"
    assert runtime.calls == ["create", "wait_ready", "exec", "delete", "wait_deleted"]
    next_interceptor.execute_activity.assert_not_awaited()
    assert heartbeats[0]["phase"] == "governed_dispatch_started"
    assert heartbeats[-1]["phase"] == "governed_dispatch_terminal"


@pytest.mark.asyncio
async def test_temporal_cancellation_waits_for_engine_cleanup() -> None:
    config, runtime = sandbox_config()
    interceptor = _GovernedCommandActivityInterceptor(
        SimpleNamespace(execute_activity=AsyncMock()), config
    )
    ready_started = asyncio.Event()
    never_ready = asyncio.Event()

    async def wait_ready(*args):
        runtime.calls.append("wait_ready")
        ready_started.set()
        await never_ready.wait()

    runtime.wait_ready = wait_ready
    cancellation = asyncio.Future()
    with (
        patch("openbox.sandbox.interceptor.activity.info", return_value=info()),
        patch("openbox.sandbox.interceptor.activity.heartbeat"),
        patch(
            "openbox.sandbox.interceptor.activity.wait_for_cancelled",
            new=lambda: cancellation,
        ),
    ):
        task = asyncio.create_task(
            interceptor.execute_activity(
                SimpleNamespace(
                    args=[SandboxCommandRequest("proof", {}).to_history_value()]
                )
            )
        )
        await ready_started.wait()
        cancellation.set_result(None)
        with pytest.raises(asyncio.CancelledError):
            await task
    assert runtime.calls == ["create", "wait_ready", "delete", "wait_deleted"]


@pytest.mark.asyncio
async def test_governed_activity_returns_only_profile_typed_values() -> None:
    config, _ = sandbox_config(typed_result=True)
    interceptor = _GovernedCommandActivityInterceptor(
        SimpleNamespace(execute_activity=AsyncMock()), config
    )
    cancelled = asyncio.Future()
    with (
        patch("openbox.sandbox.interceptor.activity.info", return_value=info()),
        patch("openbox.sandbox.interceptor.activity.heartbeat"),
        patch(
            "openbox.sandbox.interceptor.activity.wait_for_cancelled",
            new=lambda: cancelled,
        ),
    ):
        result = await interceptor.execute_activity(
            SimpleNamespace(
                args=[SandboxCommandRequest("proof", {}).to_history_value()]
            )
        )
    assert result.typed_result is not None
    assert result.typed_result.schema_name == "proof-v1"
    assert [(item.name, item.value) for item in result.typed_result.values] == [
        ("status", "ok"),
        ("count", 2),
    ]
    assert not hasattr(result, "stdout")
    assert result.stdout_bytes == len(b'{"count":2,"status":"ok"}')


@pytest.mark.asyncio
async def test_retry_attempt_is_rejected_before_sandbox() -> None:
    config, runtime = sandbox_config()
    interceptor = _GovernedCommandActivityInterceptor(
        SimpleNamespace(execute_activity=AsyncMock()), config
    )
    with patch(
        "openbox.sandbox.interceptor.activity.info",
        return_value=info(attempt=2),
    ):
        with pytest.raises(ApplicationError) as exc_info:
            await interceptor.execute_activity(
                SimpleNamespace(
                    args=[SandboxCommandRequest("proof", {}).to_history_value()]
                )
            )
    assert exc_info.value.type == "GovernedCommandAttemptRejected"
    assert runtime.calls == []


@pytest.mark.asyncio
async def test_false_mode_without_receipt_fails_before_sandbox() -> None:
    config, runtime = sandbox_config(trust_application_agent=False)
    interceptor = _GovernedCommandActivityInterceptor(
        SimpleNamespace(execute_activity=AsyncMock()), config
    )
    with patch("openbox.sandbox.interceptor.activity.info", return_value=info()):
        with pytest.raises(ApplicationError) as exc_info:
            await interceptor.execute_activity(
                SimpleNamespace(
                    args=[SandboxCommandRequest("proof", {}).to_history_value()]
                )
            )
    assert exc_info.value.type == "GovernedCommandUnauthorized"
    assert runtime.calls == []


@pytest.mark.asyncio
async def test_trusted_mode_rejects_receipt_bearing_history_input() -> None:
    config, runtime = sandbox_config(trust_application_agent=True)
    interceptor = _GovernedCommandActivityInterceptor(
        SimpleNamespace(execute_activity=AsyncMock()), config
    )
    request = SandboxCommandRequest(
        "proof",
        {},
        receipt(config.profiles.profile_fingerprint("proof")),
    )
    with patch("openbox.sandbox.interceptor.activity.info", return_value=info()):
        with pytest.raises(ApplicationError) as exc_info:
            await interceptor.execute_activity(
                SimpleNamespace(args=[request.to_history_value()])
            )
    assert exc_info.value.type == "GovernedCommandInvalid"
    assert runtime.calls == []


@pytest.mark.asyncio
async def test_receipt_mode_uses_verifier_binding_before_sandbox() -> None:
    verifier = SimpleNamespace(verify=MagicMock(return_value="receipt-1"))
    config, runtime = sandbox_config(
        trust_application_agent=False,
        receipt_verifier=verifier,
    )
    interceptor = _GovernedCommandActivityInterceptor(
        SimpleNamespace(execute_activity=AsyncMock()), config
    )
    cancelled = asyncio.Future()
    with (
        patch("openbox.sandbox.interceptor.activity.info", return_value=info()),
        patch("openbox.sandbox.interceptor.activity.heartbeat"),
        patch(
            "openbox.sandbox.interceptor.activity.wait_for_cancelled",
            new=lambda: cancelled,
        ),
    ):
        authorization_receipt = receipt(config.profiles.profile_fingerprint("proof"))
        await interceptor.execute_activity(
            SimpleNamespace(
                args=[
                    SandboxCommandRequest(
                        "proof", {}, authorization_receipt
                    ).to_history_value()
                ]
            )
        )
    assert runtime.calls == ["create", "wait_ready", "exec", "delete", "wait_deleted"]
    verify_kwargs = verifier.verify.call_args.kwargs
    assert verify_kwargs["expected_workflow_id"] == "workflow-1"
    assert verify_kwargs["command_argv"] == ("/bin/echo",)
    assert verify_kwargs["asset_bundle"] is config.engine.asset_bundle
    assert verify_kwargs["profile_fingerprint"] == config.profiles.profile_fingerprint(
        "proof"
    )


@pytest.mark.asyncio
async def test_non_governed_activity_passes_through_unchanged() -> None:
    config, runtime = sandbox_config()
    next_interceptor = SimpleNamespace(
        execute_activity=AsyncMock(return_value="native")
    )
    interceptor = _GovernedCommandActivityInterceptor(next_interceptor, config)
    value = SimpleNamespace(args=[])
    with patch(
        "openbox.sandbox.interceptor.activity.info",
        return_value=info(activity_type="ordinary"),
    ):
        assert await interceptor.execute_activity(value) == "native"
    next_interceptor.execute_activity.assert_awaited_once_with(value)
    assert runtime.calls == []


def test_factory_requires_typed_config() -> None:
    config, _ = sandbox_config()
    assert isinstance(GovernedCommandInterceptor(config), GovernedCommandInterceptor)
    with pytest.raises(TypeError):
        GovernedCommandInterceptor(object())  # type: ignore[arg-type]
