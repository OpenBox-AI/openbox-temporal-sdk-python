"""Temporal-only configuration and result mapping for sandbox dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from openbox_sandbox import (
    SandboxActivityResult,
    SandboxExecutionEngine,
    SandboxExecutionResult,
    SandboxInputError,
    SandboxTypedResult,
    StructuredCommandProfileBundle,
)

from .heartbeat import TemporalHeartbeatSink


class TemporalSandboxConfigurationError(ValueError):
    pass


class GovernedCommandDispatcher(Protocol):
    """Structural seam for ``governed_dispatcher.GovernedDispatcher``."""

    async def dispatch(self, command: Any) -> Any: ...


class GovernedCommandFactory(Protocol):
    """Structural seam for ``governed_dispatcher.GovernedCommand``."""

    def __call__(
        self,
        *,
        workflow_id: str,
        run_id: str,
        activity_id: str,
        argv: tuple[str, ...],
        profile_id: str,
        timeout_seconds: int,
        workflow_type: str,
        task_queue: str,
        attempt: int,
    ) -> Any: ...


@dataclass(frozen=True)
class TemporalSandboxConfig:
    engine: SandboxExecutionEngine | None
    profiles: StructuredCommandProfileBundle
    heartbeat_sink: TemporalHeartbeatSink
    timeout_seconds: int = 30
    heartbeat_interval_seconds: float = 10.0
    receipt_verifier: Any = None
    trust_application_agent: bool = False
    dispatcher: GovernedCommandDispatcher | None = None
    governed_command_factory: GovernedCommandFactory | None = None

    def __post_init__(self) -> None:
        natural_requested = (
            self.dispatcher is not None or self.governed_command_factory is not None
        )
        natural_valid = (
            self.dispatcher is not None
            and callable(getattr(self.dispatcher, "dispatch", None))
            and callable(self.governed_command_factory)
            and self.engine is None
            and self.receipt_verifier is None
            and not self.trust_application_agent
        )
        compatibility_valid = (
            not natural_requested
            and isinstance(self.engine, SandboxExecutionEngine)
            and (self.trust_application_agent != (self.receipt_verifier is not None))
            and (
                self.receipt_verifier is None
                or callable(getattr(self.receipt_verifier, "verify", None))
            )
        )
        profiles_valid = isinstance(self.profiles, StructuredCommandProfileBundle)
        engine_profiles_valid = self.engine is None or (
            isinstance(self.engine, SandboxExecutionEngine)
            and profiles_valid
            and self.engine.telemetry_sink is self.heartbeat_sink
            and self.engine.profiles.fingerprint == self.profiles.fingerprint
            and self.engine.profiles.bundle_version == self.profiles.bundle_version
        )
        if (
            not profiles_valid
            or not isinstance(self.heartbeat_sink, TemporalHeartbeatSink)
            or not engine_profiles_valid
            or isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, int)
            or not 1 <= self.timeout_seconds <= 300
            or isinstance(self.heartbeat_interval_seconds, bool)
            or not isinstance(self.heartbeat_interval_seconds, (int, float))
            or not 0.1 <= self.heartbeat_interval_seconds <= 60
            or type(self.trust_application_agent) is not bool
            or not (natural_valid or compatibility_valid)
        ):
            raise TemporalSandboxConfigurationError(
                "Temporal sandbox configuration rejected"
            )


def activity_result(
    profile_id: str,
    execution_result: SandboxExecutionResult,
    *,
    typed_result: SandboxTypedResult | None = None,
) -> SandboxActivityResult:
    """Map the concrete compatibility-engine result."""
    execution = execution_result.execution
    if execution is None or execution.exit_code is None:
        raise TemporalSandboxConfigurationError(
            "sandbox engine omitted terminal execution"
        )
    return SandboxActivityResult(
        profile_id=profile_id,
        disposition=execution_result.disposition.value,
        exit_code=execution.exit_code,
        timeout_status=execution.timeout_status.value,
        cleanup_status=execution.cleanup_status.value,
        stdout_bytes=len(execution.stdout),
        stderr_bytes=len(execution.stderr),
        typed_result=typed_result,
    )


@dataclass(frozen=True)
class DispatchExecution:
    """Validated structural view of governed-dispatcher execution metadata."""

    exit_code: int
    stdout: bytes
    stderr: bytes
    timeout_status: str
    cleanup_status: str


def dispatch_execution(result: Any) -> DispatchExecution:
    """Validate the structural ``DispatchResult``/``ExecutionMetadata`` shape."""
    try:
        disposition = _enum_value(result.disposition)
        execution = result.execution
        error = result.error
        if (
            disposition != "executed_in_sandbox"
            or execution is None
            or error is not None
        ):
            raise TemporalSandboxConfigurationError(
                "dispatcher did not return terminal sandbox execution"
            )
        exit_code = execution.exit_code
        stdout = execution.stdout
        stderr = execution.stderr
        timeout_status = _enum_value(execution.timeout_status)
        cleanup_status = _enum_value(execution.cleanup_status)
        if (
            type(exit_code) is not int
            or not 0 <= exit_code <= 2**31 - 1
            or not isinstance(stdout, bytes)
            or not isinstance(stderr, bytes)
            or len(stdout) > 1024 * 1024
            or len(stderr) > 1024 * 1024
            or len(stdout) + len(stderr) > 2 * 1024 * 1024
            or timeout_status
            not in {"not_observed", "confirmed_timeout", "possible_timeout"}
            or cleanup_status not in {"deleted", "failed"}
        ):
            raise TemporalSandboxConfigurationError(
                "dispatcher returned invalid execution metadata"
            )
        return DispatchExecution(
            exit_code,
            stdout,
            stderr,
            timeout_status,
            cleanup_status,
        )
    except (AttributeError, TypeError) as error:
        raise TemporalSandboxConfigurationError(
            "dispatcher returned invalid result"
        ) from error


def dispatch_activity_result(
    profile_id: str,
    execution: DispatchExecution,
    *,
    typed_result: SandboxTypedResult | None = None,
) -> SandboxActivityResult:
    """Map validated dispatcher metadata into the bounded history contract."""
    try:
        return SandboxActivityResult(
            profile_id=profile_id,
            disposition="executed_in_sandbox",
            exit_code=execution.exit_code,
            timeout_status=execution.timeout_status,
            cleanup_status=execution.cleanup_status,
            stdout_bytes=len(execution.stdout),
            stderr_bytes=len(execution.stderr),
            typed_result=typed_result,
        )
    except SandboxInputError as error:
        raise TemporalSandboxConfigurationError(
            "dispatcher execution metadata exceeded bounds"
        ) from error


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)
