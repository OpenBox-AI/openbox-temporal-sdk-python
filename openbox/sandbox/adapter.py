"""Activity-side adapter to the proven provider-neutral dispatcher."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .heartbeat import TemporalHeartbeatSink
from .profiles import TemporalCommandProfileBundle
from .types import GovernedCommandActivityResult, GovernedCommandTypedResult


class TemporalSandboxConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class TemporalSandboxConfig:
    dispatcher: Any
    profiles: TemporalCommandProfileBundle
    heartbeat_sink: TemporalHeartbeatSink
    timeout_seconds: int = 30
    heartbeat_interval_seconds: float = 10.0
    completion_events: bool = True
    otel_bridge: Any = None
    evaluate_at_interceptor: bool = False

    def __post_init__(self) -> None:
        try:
            from openbox_sandbox.dispatcher import GovernedDispatcher
        except ImportError as error:
            raise TemporalSandboxConfigurationError(
                "the private governed-dispatcher package is required"
            ) from error
        if (
            not isinstance(self.dispatcher, GovernedDispatcher)
            or not isinstance(self.profiles, TemporalCommandProfileBundle)
            or not isinstance(self.heartbeat_sink, TemporalHeartbeatSink)
            or getattr(self.dispatcher, "_telemetry", None) is not self.heartbeat_sink
            or isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, int)
            or not 1 <= self.timeout_seconds <= 300
            or isinstance(self.heartbeat_interval_seconds, bool)
            or not isinstance(self.heartbeat_interval_seconds, (int, float))
            or not 0.1 <= self.heartbeat_interval_seconds <= 60
            or type(self.completion_events) is not bool
            or (
                self.otel_bridge is not None
                and getattr(self.heartbeat_sink, "_otel_bridge", None)
                is not self.otel_bridge
            )
        ):
            raise TemporalSandboxConfigurationError(
                "Temporal sandbox configuration rejected"
            )


def require_matching_governance_signing(
    sandbox: TemporalSandboxConfig | None, agent_did: str | None
) -> None:
    """Fail startup when completion and dispatcher preflight signing differ."""
    if sandbox is None:
        return
    dispatcher_did = getattr(sandbox.dispatcher, "governance_signer_did", None)
    if dispatcher_did != agent_did:
        raise TemporalSandboxConfigurationError(
            "Temporal and dispatcher governance signing must match"
        )


def activity_result(
    profile_id: str,
    dispatch_result: Any,
    *,
    typed_result: GovernedCommandTypedResult | None = None,
) -> GovernedCommandActivityResult:
    execution = dispatch_result.execution
    if execution is None or execution.exit_code is None:
        raise TemporalSandboxConfigurationError("dispatcher omitted terminal execution")
    return GovernedCommandActivityResult(
        profile_id=profile_id,
        disposition=dispatch_result.disposition.value,
        exit_code=execution.exit_code,
        timeout_status=execution.timeout_status.value,
        cleanup_status=execution.cleanup_status.value,
        stdout_bytes=len(execution.stdout),
        stderr_bytes=len(execution.stderr),
        typed_result=typed_result,
    )
