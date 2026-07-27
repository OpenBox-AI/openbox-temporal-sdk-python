"""Resolve a simplified SandboxConfig into a fully-wired TemporalSandboxConfig."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from openbox_sandbox import (
    SandboxEngineConfig as _SandboxEngineConfig,
    SandboxExecutionEngine as _SandboxExecutionEngine,
    StructuredCommandProfileBundle as _StructuredCommandProfileBundle,
)
from openbox_sandbox.deployment import load_sandbox_deployment
from openbox_sandbox.engine import UnixAgentExecutionConfig
from openbox_sandbox.registry import GovernedCommandRegistry
from openbox_sandbox.release import (
    load_approved_sandbox_release,
    materialize_approved_sandbox_release,
)
from openbox_sandbox.runtime import (
    OutputLimits,
    PolicyDocument,
    UnixAgentRuntimeClient,
    UnixAgentRuntimeClientConfig,
)

from .adapter import TemporalSandboxConfig
from .config import SandboxConfig
from .heartbeat import TemporalHeartbeatSink

if TYPE_CHECKING:
    from openbox_sandbox import TelemetrySink

# Well-known paths (relative to project root or absolute)
_WELL_KNOWN_RELEASE_PATHS = [
    Path("openbox-sandbox-release.json"),
    Path(".openbox/sandbox-release.json"),
]

_WELL_KNOWN_MANIFEST_PATHS = [
    Path("openbox-sandbox-deployment.json"),
    Path(".openbox/sandbox-deployment.json"),
]


def _find_well_known_path(candidates: list[Path]) -> Path | None:
    """Check well-known paths relative to CWD."""
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _load_release(release_path: Path | None) -> None:
    """Load the approved sandbox release."""
    if release_path is not None:
        load_approved_sandbox_release(release_path)
        return

    # Try well-known paths
    found = _find_well_known_path(_WELL_KNOWN_RELEASE_PATHS)
    if found is not None:
        load_approved_sandbox_release(found)
        return

    # Release may already be installed (e.g., by a previous call or app setup)
    # If not, materialize_approved_sandbox_release() will fail with a clear error


def _build_engine_from_release(
    registry: GovernedCommandRegistry,
    *,
    socket_path: Path | None = None,
    telemetry: TelemetrySink | None = None,
) -> tuple[_SandboxExecutionEngine, _StructuredCommandProfileBundle]:
    """Build a sandbox engine directly from the installed release (no manifest)."""
    material = materialize_approved_sandbox_release()
    profiles = registry.admission_profile_bundle()
    structured_profiles = registry.structured_profile_bundle()

    resolved_socket = socket_path or Path(
        f"/tmp/openbox-sandbox-{registry.fingerprint[:16]}.sock"
    )

    sandbox_config = UnixAgentExecutionConfig(
        socket_path=resolved_socket,
        registry_fingerprint=registry.fingerprint,
        asset_bundle=material.asset_bundle,
        policy_document=material.policy_document,
        output_limits=OutputLimits(
            stdout_bytes=1024 * 1024,
            stderr_bytes=1024 * 1024,
            combined_bytes=2 * 1024 * 1024,
            chunk_bytes=4 * 1024 * 1024,
        ),
        enabled=True,
    )

    runtime = UnixAgentRuntimeClient(
        UnixAgentRuntimeClientConfig(
            socket_path=resolved_socket,
            asset_bundle=material.asset_bundle,
            registry_fingerprint=registry.fingerprint,
        )
    )

    engine_config = _SandboxEngineConfig(
        profiles=profiles,
        sandbox=sandbox_config,
        telemetry=telemetry,
    )

    engine = _SandboxExecutionEngine._from_components(
        engine_config,
        sandbox=runtime,
        clock=lambda: datetime.now(timezone.utc),
        sandbox_id=lambda: f"sbx-{uuid.uuid4()}",
    )

    return engine, structured_profiles


def resolve_sandbox_config(
    config: SandboxConfig,
) -> TemporalSandboxConfig:
    """Resolve a simplified SandboxConfig into a fully-wired TemporalSandboxConfig.

    This is the core of the simplified sandbox API. It handles:
    1. Loading the approved release
    2. Loading or building the deployment
    3. Wiring everything into a TemporalSandboxConfig
    """
    # 1. Load release
    _load_release(config.release_path)

    # 2. Build or load deployment
    heartbeat_sink = TemporalHeartbeatSink()

    if config.deployment_manifest is not None:
        # Use manifest-based deployment
        deployment = load_sandbox_deployment(
            config.deployment_manifest,
            registry=config.registry,
            telemetry=config.telemetry or heartbeat_sink,
        )
        engine = deployment.engine
        structured_profiles = deployment.structured_profiles

        # Verify engine telemetry is the same object we're passing
        if config.telemetry is None and engine.telemetry_sink is not heartbeat_sink:
            # Rebuild engine with correct telemetry reference
            engine_config = _SandboxEngineConfig(
                profiles=deployment.profiles,
                sandbox=deployment.config.sandbox,
                telemetry=heartbeat_sink,
                cleanup_backlog=deployment.cleanup_backlog,
            )
            engine = _SandboxExecutionEngine._from_components(
                engine_config,
                sandbox=deployment._runtime,
                clock=lambda: datetime.now(timezone.utc),
                sandbox_id=lambda: f"sbx-{uuid.uuid4()}",
            )
    else:
        # Build from release directly
        engine, structured_profiles = _build_engine_from_release(
            config.registry,
            socket_path=config.socket_path,
            telemetry=heartbeat_sink,
        )

    # 3. Determine trust model
    if config.dispatcher is not None:
        # Natural mode — no engine, no receipt verifier
        return TemporalSandboxConfig(
            engine=None,
            profiles=structured_profiles,
            heartbeat_sink=heartbeat_sink,
            timeout_seconds=config.timeout_seconds,
            heartbeat_interval_seconds=config.heartbeat_interval_seconds,
            dispatcher=config.dispatcher,
            governed_command_factory=config.governed_command_factory,
        )

    # Compatibility mode — engine + trust or receipt
    return TemporalSandboxConfig(
        engine=engine,
        profiles=structured_profiles,
        heartbeat_sink=heartbeat_sink,
        timeout_seconds=config.timeout_seconds,
        heartbeat_interval_seconds=config.heartbeat_interval_seconds,
        trust_application_agent=config.trust_application_agent,
        receipt_verifier=config.receipt_verifier,
    )
