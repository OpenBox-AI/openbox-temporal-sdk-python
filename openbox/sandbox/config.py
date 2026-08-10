"""Simplified sandbox configuration for the OpenBoxPlugin."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from openbox_sandbox import GovernedCommandRegistry, TelemetrySink


@dataclass(frozen=True)
class SandboxConfig:
    """One-stop sandbox configuration for the OpenBoxPlugin.

    Minimal usage::

        worker = Worker(..., plugins=[OpenBoxPlugin(..., sandbox=SandboxConfig(registry=my_registry))])

    The registry is the only required field — it defines which commands the
    application is authorized to execute. Everything else is auto-discovered
    from the approved release and deployment manifest.
    """

    # Application-layer (REQUIRED — user defines their commands)
    registry: GovernedCommandRegistry

    # Infrastructure — auto-discovered with sensible defaults
    deployment_manifest: Path | None = None
    release_path: Path | None = None

    # Transport — auto-detected from manifest
    transport: Literal["auto", "uds_agent", "direct_tls"] = "auto"
    socket_path: Path | None = None

    # Execution tuning — sensible defaults
    timeout_seconds: int = 30
    heartbeat_interval_seconds: float = 10.0

    # Trust model
    trust_application_agent: bool = True

    # Advanced (rarely needed)
    receipt_verifier: Any = None
    dispatcher: Any = None
    governed_command_factory: Any = None
    telemetry: TelemetrySink | None = None

    def __post_init__(self) -> None:
        from openbox_sandbox import GovernedCommandRegistry as _Registry

        if not isinstance(self.registry, _Registry):
            raise TypeError("SandboxConfig.registry must be a GovernedCommandRegistry")
        if self.deployment_manifest is not None and not isinstance(
            self.deployment_manifest, Path
        ):
            raise TypeError("SandboxConfig.deployment_manifest must be a Path or None")
        if self.release_path is not None and not isinstance(self.release_path, Path):
            raise TypeError("SandboxConfig.release_path must be a Path or None")
        if self.transport not in ("auto", "uds_agent", "direct_tls"):
            raise ValueError(f"Invalid transport: {self.transport!r}")
        if self.socket_path is not None and not isinstance(self.socket_path, Path):
            raise TypeError("SandboxConfig.socket_path must be a Path or None")
        if isinstance(self.timeout_seconds, bool) or not isinstance(
            self.timeout_seconds, int
        ):
            raise TypeError("SandboxConfig.timeout_seconds must be an int")
        if not 1 <= self.timeout_seconds <= 300:
            raise ValueError("SandboxConfig.timeout_seconds must be 1-300")
        if isinstance(self.heartbeat_interval_seconds, bool) or not isinstance(
            self.heartbeat_interval_seconds, (int, float)
        ):
            raise TypeError("SandboxConfig.heartbeat_interval_seconds must be numeric")
        if not 0.1 <= self.heartbeat_interval_seconds <= 60:
            raise ValueError(
                "SandboxConfig.heartbeat_interval_seconds must be 0.1-60"
            )
        if type(self.trust_application_agent) is not bool:
            raise TypeError("SandboxConfig.trust_application_agent must be a bool")
