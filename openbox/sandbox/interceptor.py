"""Temporal Activity interception for governed sandbox commands."""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any

from openbox_sandbox import (
    GOVERNED_COMMAND_ACTIVITY_TYPE,
    Disposition,
    SandboxAuthorization,
    SandboxCommand,
    SandboxCommandRequest,
    SandboxExecutionResult,
    SandboxInputError,
)
from openbox_sandbox.command_profiles import CommandResultValidationError
from temporalio import activity
from temporalio.exceptions import ApplicationError
from temporalio.worker import (
    ActivityInboundInterceptor,
    ExecuteActivityInput,
    Interceptor,
)

from ..core_adapter import build_core_activity_context
from .adapter import (
    TemporalSandboxConfig,
    TemporalSandboxConfigurationError,
    activity_result,
    dispatch_activity_result,
    dispatch_execution,
)


class GovernedCommandInterceptor(Interceptor):
    """Intercept only the registered governed-command Activity type."""

    def __init__(self, config: TemporalSandboxConfig):
        if not isinstance(config, TemporalSandboxConfig):
            raise TypeError("TemporalSandboxConfig required")
        self._config = config

    def intercept_activity(
        self, next_interceptor: ActivityInboundInterceptor
    ) -> ActivityInboundInterceptor:
        return _GovernedCommandActivityInterceptor(next_interceptor, self._config)


class _GovernedCommandActivityInterceptor(ActivityInboundInterceptor):
    def __init__(
        self,
        next_interceptor: ActivityInboundInterceptor,
        config: TemporalSandboxConfig,
    ) -> None:
        super().__init__(next_interceptor)
        self._config = config

    async def execute_activity(self, input: ExecuteActivityInput) -> Any:
        info = activity.info()
        if info.activity_type != GOVERNED_COMMAND_ACTIVITY_TYPE:
            return await self.next.execute_activity(input)
        if type(info.attempt) is not int or info.attempt != 1:
            raise ApplicationError(
                "Governed commands permit exactly one Activity attempt",
                type="GovernedCommandAttemptRejected",
                non_retryable=True,
            )

        workflow_id = self._identifier(info.workflow_id)
        run_id = self._identifier(info.workflow_run_id)
        activity_id = self._identifier(info.activity_id)
        workflow_type = self._identifier(info.workflow_type)
        task_queue = self._identifier(info.task_queue)
        request, argv = self._request(input)
        natural_mode = self._config.dispatcher is not None
        authorization = None
        if natural_mode:
            factory = self._config.governed_command_factory
            assert factory is not None
            try:
                command = factory(
                    workflow_id=workflow_id,
                    run_id=run_id,
                    activity_id=activity_id,
                    argv=argv,
                    profile_id=request.profile_id,
                    timeout_seconds=self._config.timeout_seconds,
                    workflow_type=workflow_type,
                    task_queue=task_queue,
                    attempt=info.attempt,
                )
            except Exception as error:
                raise ApplicationError(
                    "Governed command construction failed",
                    type="GovernedCommandInvalid",
                    non_retryable=True,
                ) from error
        else:
            authorization = self._authorization(
                request,
                argv,
                workflow_id=workflow_id,
                run_id=run_id,
                activity_id=activity_id,
            )
            context = build_core_activity_context(
                info,
                activity_input=request.to_history_value(),
            )
            command = SandboxCommand(
                context=context,
                argv=argv,
                profile_id=request.profile_id,
                timeout_seconds=self._config.timeout_seconds,
            )

        with self._config.heartbeat_sink.bind(
            activity.heartbeat,
            workflow_id=workflow_id,
            run_id=run_id,
            activity_id=activity_id,
            attempt=info.attempt,
            profile_id=request.profile_id,
        ):
            if natural_mode:
                dispatcher = self._config.dispatcher
                assert dispatcher is not None
                execution_task = asyncio.create_task(dispatcher.dispatch(command))
            else:
                # Compatibility modes alone construct trusted/receipt authority.
                engine = self._config.engine
                assert engine is not None and authorization is not None
                execution_task = asyncio.create_task(
                    engine.execute(command, authorization)
                )
            cancellation_task = asyncio.create_task(self._wait_for_cancellation())
            heartbeat_task = asyncio.create_task(self._heartbeat_periodically())
            try:
                done, _ = await asyncio.wait(
                    {execution_task, cancellation_task, heartbeat_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if heartbeat_task in done:
                    await heartbeat_task
                if cancellation_task in done:
                    execution_task.cancel()
                    await self._await_cleanup(execution_task)
                    raise asyncio.CancelledError()
                result = await execution_task
            finally:
                if not execution_task.done():
                    execution_task.cancel()
                    await self._await_cleanup(execution_task)
                for task in (heartbeat_task, cancellation_task):
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

        if natural_mode:
            try:
                execution = dispatch_execution(result)
                typed_result = self._config.profiles.parse_result(
                    request.profile_id, execution.stdout
                )
                mapped_result = dispatch_activity_result(
                    request.profile_id,
                    execution,
                    typed_result=typed_result,
                )
            except CommandResultValidationError as error:
                raise ApplicationError(
                    "Governed command typed result rejected",
                    type="GovernedCommandResultInvalid",
                    non_retryable=True,
                ) from error
            except TemporalSandboxConfigurationError as error:
                raise ApplicationError(
                    "Governed dispatcher result rejected",
                    type="GovernedCommandEngineFailure",
                    non_retryable=True,
                ) from error
            self._terminal_heartbeat(
                workflow_id,
                run_id,
                activity_id,
                info.attempt,
                request.profile_id,
                mapped_result.disposition,
            )
            return mapped_result

        if not isinstance(result, SandboxExecutionResult):
            raise ApplicationError(
                "Sandbox engine returned an invalid result",
                type="GovernedCommandEngineFailure",
                non_retryable=True,
            )
        self._terminal_heartbeat(
            workflow_id,
            run_id,
            activity_id,
            info.attempt,
            request.profile_id,
            result.disposition.value,
        )
        if result.disposition is Disposition.EXECUTED_IN_SANDBOX:
            if result.execution is None:
                raise ApplicationError(
                    "Sandbox engine omitted execution metadata",
                    type="GovernedCommandEngineFailure",
                    non_retryable=True,
                )
            try:
                typed_result = self._config.profiles.parse_result(
                    request.profile_id, result.execution.stdout
                )
                return activity_result(
                    request.profile_id,
                    result,
                    typed_result=typed_result,
                )
            except CommandResultValidationError as error:
                raise ApplicationError(
                    "Governed command typed result rejected",
                    type="GovernedCommandResultInvalid",
                    non_retryable=True,
                ) from error
            except (SandboxInputError, TemporalSandboxConfigurationError) as error:
                raise ApplicationError(
                    "Governed command result rejected",
                    type="GovernedCommandEngineFailure",
                    non_retryable=True,
                ) from error

        error_code = (
            "governed_command_not_executed"
            if result.error is None
            else result.error.code.value
        )
        error_type = (
            "GovernedCommandExecutionIndeterminate"
            if result.disposition is Disposition.EXECUTION_INDETERMINATE
            else "GovernedCommandNotExecuted"
        )
        raise ApplicationError(
            f"Governed command terminal outcome: {error_code}",
            type=error_type,
            non_retryable=True,
        )

    def _request(
        self, input: ExecuteActivityInput
    ) -> tuple[SandboxCommandRequest, tuple[str, ...]]:
        try:
            args = list(input.args) if input.args is not None else []
            if len(args) != 1:
                raise SandboxInputError("governed command input rejected")
            request = SandboxCommandRequest.from_value(args[0])
            if (
                self._config.trust_application_agent
                or self._config.dispatcher is not None
            ) and request.receipt is not None:
                raise SandboxInputError("governed command input rejected")
            argv = self._config.profiles.derive(request)
            self._config.profiles.profile_fingerprint(request.profile_id)
            return request, argv
        except (SandboxInputError, TypeError, ValueError) as error:
            raise ApplicationError(
                "Governed command input rejected",
                type="GovernedCommandInvalid",
                non_retryable=True,
            ) from error

    def _authorization(
        self,
        request: SandboxCommandRequest,
        argv: tuple[str, ...],
        *,
        workflow_id: str,
        run_id: str,
        activity_id: str,
    ) -> SandboxAuthorization:
        if self._config.trust_application_agent:
            binding = "\0".join((workflow_id, run_id, activity_id)).encode("utf-8")
            return SandboxAuthorization.trusted_application(
                f"trusted:{hashlib.sha256(binding).hexdigest()}"
            )
        engine = self._config.engine
        verifier = self._config.receipt_verifier
        if verifier is None or engine is None:
            raise ApplicationError(
                "Governed command authorization is not configured",
                type="GovernedCommandUnauthorized",
                non_retryable=True,
            )
        try:
            authorization_id = verifier.verify(
                request,
                expected_workflow_id=workflow_id,
                command_argv=argv,
                asset_bundle=engine.asset_bundle,
                profile_fingerprint=self._config.profiles.profile_fingerprint(
                    request.profile_id
                ),
            )
        except Exception as error:
            raise ApplicationError(
                "Governed command receipt rejected",
                type="GovernedCommandUnauthorized",
                non_retryable=True,
            ) from error
        return SandboxAuthorization.verified_receipt(authorization_id)

    @staticmethod
    def _identifier(value: object) -> str:
        if not isinstance(value, str) or not value:
            raise ApplicationError(
                "Governed command Activity identity rejected",
                type="GovernedCommandInvalid",
                non_retryable=True,
            )
        return value

    @staticmethod
    def _terminal_heartbeat(
        workflow_id: str,
        run_id: str,
        activity_id: str,
        attempt: int,
        profile_id: str,
        disposition: str,
    ) -> None:
        activity.heartbeat(
            {
                "phase": "governed_dispatch_terminal",
                "workflow_id": workflow_id,
                "run_id": run_id,
                "activity_id": activity_id,
                "attempt": attempt,
                "profile_id": profile_id,
                "disposition": disposition,
            }
        )

    async def _heartbeat_periodically(self) -> None:
        while True:
            await asyncio.sleep(self._config.heartbeat_interval_seconds)
            self._config.heartbeat_sink.heartbeat_latest()

    @staticmethod
    async def _wait_for_cancellation() -> None:
        waiting = activity.wait_for_cancelled()
        if hasattr(waiting, "__await__"):
            await waiting
        else:
            await asyncio.Future()

    @staticmethod
    async def _await_cleanup(task: asyncio.Task[Any]) -> None:
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            # Engine terminal errors can surface only after its cleanup boundary.
            pass
