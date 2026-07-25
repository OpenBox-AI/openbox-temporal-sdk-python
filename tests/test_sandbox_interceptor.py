from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openbox_sandbox import (
    GOVERNED_COMMAND_ACTIVITY_TYPE,
    SandboxAuthorization,
    SandboxCommandRequest,
    SandboxReceipt,
)
from temporalio.exceptions import ApplicationError

from openbox.sandbox.adapter import TemporalSandboxConfigurationError
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


def dispatch_result(
    stdout: bytes = b"ok",
    *,
    stderr: bytes = b"err",
    disposition: str = "executed_in_sandbox",
    exit_code: int | None = 0,
    execution: bool = True,
    error: object | None = None,
):
    metadata = (
        SimpleNamespace(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            timeout_status=SimpleNamespace(value="not_observed"),
            cleanup_status=SimpleNamespace(value="deleted"),
        )
        if execution
        else None
    )
    return SimpleNamespace(
        disposition=SimpleNamespace(value=disposition),
        execution=metadata,
        error=error,
    )


def governed_command_factory() -> MagicMock:
    return MagicMock(return_value=SimpleNamespace(factory_product=True))


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


@pytest.mark.parametrize(
    ("dispatcher", "factory"),
    [
        (None, None),
        (SimpleNamespace(dispatch=AsyncMock()), None),
        (None, governed_command_factory()),
    ],
)
def test_incomplete_natural_mode_fails_closed_at_configuration(
    dispatcher: object | None,
    factory: object | None,
) -> None:
    with pytest.raises(TemporalSandboxConfigurationError):
        sandbox_config(
            trust_application_agent=False,
            dispatcher=dispatcher,
            governed_command_factory=factory,
        )


@pytest.mark.asyncio
async def test_natural_mode_dispatches_once_with_genuine_temporal_identity() -> None:
    dispatcher = SimpleNamespace(
        dispatch=AsyncMock(return_value=dispatch_result()),
        authorize=MagicMock(),
        execute=AsyncMock(),
        verify=MagicMock(),
    )
    factory = governed_command_factory()
    config, runtime = sandbox_config(
        trust_application_agent=False,
        dispatcher=dispatcher,
        governed_command_factory=factory,
    )
    interceptor = _GovernedCommandActivityInterceptor(
        SimpleNamespace(execute_activity=AsyncMock()), config
    )
    cancelled = asyncio.Future()
    heartbeats: list[dict[str, object]] = []
    with (
        patch("openbox.sandbox.interceptor.activity.info", return_value=info()),
        patch(
            "openbox.sandbox.interceptor.activity.heartbeat",
            side_effect=heartbeats.append,
        ),
        patch(
            "openbox.sandbox.interceptor.activity.wait_for_cancelled",
            new=lambda: cancelled,
        ),
        patch.object(
            SandboxAuthorization, "trusted_application"
        ) as trusted_application,
        patch.object(SandboxAuthorization, "verified_receipt") as verified_receipt,
    ):
        result = await interceptor.execute_activity(
            SimpleNamespace(
                args=[SandboxCommandRequest("proof", {}).to_history_value()]
            )
        )

    factory.assert_called_once_with(
        workflow_id="workflow-1",
        run_id="run-1",
        activity_id="activity-1",
        argv=("/bin/echo",),
        profile_id="proof",
        timeout_seconds=30,
        workflow_type="ProofWorkflow",
        task_queue="sandbox-queue",
        attempt=1,
    )
    dispatcher.dispatch.assert_awaited_once_with(factory.return_value)
    assert config.engine is None
    assert runtime.calls == []
    trusted_application.assert_not_called()
    verified_receipt.assert_not_called()
    dispatcher.authorize.assert_not_called()
    dispatcher.execute.assert_not_awaited()
    dispatcher.verify.assert_not_called()
    assert not hasattr(result, "stdout")
    assert result.disposition == "executed_in_sandbox"
    assert result.exit_code == 0
    assert result.timeout_status == "not_observed"
    assert result.cleanup_status == "deleted"
    assert result.stdout_bytes == 2
    assert result.stderr_bytes == 3
    assert heartbeats[-1]["disposition"] == "executed_in_sandbox"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "dispatcher_result",
    [
        object(),
        dispatch_result(disposition="executed_on_host"),
        dispatch_result(execution=False),
        dispatch_result(exit_code=None),
        dispatch_result(exit_code=-1),
        dispatch_result(exit_code=2**31),
        dispatch_result(error=object()),
        dispatch_result(b"x" * (1024 * 1024 + 1)),
        dispatch_result(stderr=b"x" * (1024 * 1024 + 1)),
    ],
)
async def test_natural_mode_rejects_invalid_host_nonterminal_or_unbounded_result(
    dispatcher_result: object,
) -> None:
    dispatcher = SimpleNamespace(dispatch=AsyncMock(return_value=dispatcher_result))
    config, _ = sandbox_config(
        trust_application_agent=False,
        dispatcher=dispatcher,
        governed_command_factory=governed_command_factory(),
        typed_result=True,
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
        with pytest.raises(ApplicationError) as exc_info:
            await interceptor.execute_activity(
                SimpleNamespace(
                    args=[SandboxCommandRequest("proof", {}).to_history_value()]
                )
            )
    assert exc_info.value.type == "GovernedCommandEngineFailure"
    assert exc_info.value.non_retryable is True
    dispatcher.dispatch.assert_awaited_once()


@pytest.mark.asyncio
async def test_natural_mode_cancellation_waits_for_dispatcher_cleanup() -> None:
    dispatch_started = asyncio.Event()
    cleanup_finished = asyncio.Event()

    async def dispatch(command):
        dispatch_started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            await asyncio.sleep(0)
            cleanup_finished.set()
            raise

    dispatcher = SimpleNamespace(dispatch=dispatch)
    config, _ = sandbox_config(
        trust_application_agent=False,
        dispatcher=dispatcher,
        governed_command_factory=governed_command_factory(),
    )
    interceptor = _GovernedCommandActivityInterceptor(
        SimpleNamespace(execute_activity=AsyncMock()), config
    )
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
        await dispatch_started.wait()
        cancellation.set_result(None)
        with pytest.raises(asyncio.CancelledError):
            await task
    assert cleanup_finished.is_set()


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
async def test_natural_mode_rejects_compatibility_receipt_before_dispatch() -> None:
    dispatcher = SimpleNamespace(dispatch=AsyncMock(return_value=dispatch_result()))
    config, runtime = sandbox_config(
        trust_application_agent=False,
        dispatcher=dispatcher,
        governed_command_factory=governed_command_factory(),
    )
    request = SandboxCommandRequest(
        "proof",
        {},
        receipt(config.profiles.profile_fingerprint("proof")),
    )
    interceptor = _GovernedCommandActivityInterceptor(
        SimpleNamespace(execute_activity=AsyncMock()), config
    )
    with patch("openbox.sandbox.interceptor.activity.info", return_value=info()):
        with pytest.raises(ApplicationError) as exc_info:
            await interceptor.execute_activity(
                SimpleNamespace(args=[request.to_history_value()])
            )
    assert exc_info.value.type == "GovernedCommandInvalid"
    dispatcher.dispatch.assert_not_awaited()
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
    assert config.engine is not None
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
