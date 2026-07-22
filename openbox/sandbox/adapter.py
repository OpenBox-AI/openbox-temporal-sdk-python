"""Temporal-only configuration and result mapping for openbox_sandbox."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openbox_sandbox import (
    SandboxActivityResult,
    SandboxExecutionEngine,
    SandboxExecutionResult,
    SandboxTypedResult,
    StructuredCommandProfileBundle,
)

from .heartbeat import TemporalHeartbeatSink


class TemporalSandboxConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class TemporalSandboxConfig:
    engine: SandboxExecutionEngine
    profiles: StructuredCommandProfileBundle
    heartbeat_sink: TemporalHeartbeatSink
    timeout_seconds: int = 30
    heartbeat_interval_seconds: float = 10.0
    receipt_verifier: Any = None
    trust_application_agent: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.engine, SandboxExecutionEngine)
            or not isinstance(self.profiles, StructuredCommandProfileBundle)
            or not isinstance(self.heartbeat_sink, TemporalHeartbeatSink)
            or self.engine.telemetry_sink is not self.heartbeat_sink
            or self.engine.profiles.fingerprint != self.profiles.fingerprint
            or self.engine.profiles.bundle_version != self.profiles.bundle_version
            or isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, int)
            or not 1 <= self.timeout_seconds <= 300
            or isinstance(self.heartbeat_interval_seconds, bool)
            or not isinstance(self.heartbeat_interval_seconds, (int, float))
            or not 0.1 <= self.heartbeat_interval_seconds <= 60
            or type(self.trust_application_agent) is not bool
            or (
                self.receipt_verifier is not None
                and (
                    self.trust_application_agent
                    or not callable(getattr(self.receipt_verifier, "verify", None))
                )
            )
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
