"""Simplified plugin sandbox configuration built from an application registry.

The worker declares only the typed command registry (plus the trusted service
boundary and execution tuning); the plugin resolves it into the fully wired
``TemporalSandboxConfig`` — building the governed dispatcher, both profile
bundles, the heartbeat sink, the output limits, the policy document, and the
asset bundle internally.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MAX_JSON_BYTES = 1024 * 1024
_DEFAULT_OUTPUT_LIMITS = (
    1024 * 1024,
    1024 * 1024,
    2 * 1024 * 1024,
    4 * 1024 * 1024,
)


@dataclass(frozen=True)
class SandboxConfig:
    """One-stop sandbox configuration for :class:`openbox.OpenBoxPlugin`.

    Minimal usage::

        sandbox = SandboxConfig(
            registry=my_registry,
            service_config=Path("sandbox-service-config.json"),
            policy=Path("sandbox-policy.yaml"),
            socket_path=Path("/run/openbox/agent.sock"),
        )

        worker = Worker(
            client, task_queue="q", workflows=[MyWorkflow],
            activities=[my_activity],
            plugins=[OpenBoxPlugin(..., sandbox=sandbox)],
        )

    The registry is the only required field — it defines which commands the
    application is authorized to execute and how structured activity input
    maps onto command argv. The plugin builds the governed dispatcher, the
    profile bundles, the heartbeat sink, the output limits, the policy
    document, and the asset bundle internally from these fields.
    """

    # Application layer (REQUIRED — the user defines their commands).
    registry: Any  # GovernedCommandRegistry

    # Trusted service boundary. Either a UDS agent socket or the direct-TLS
    # sandbox service described by the service config document.
    service_config: Path | None = None
    policy: Path | None = None
    policy_registry_dir: Path | None = None
    socket_path: Path | None = None
    ca: Path | None = None
    certificate: Path | None = None
    private_key: Path | None = None
    host_workdir: Path | None = None
    registry_fingerprint: str | None = None

    # Execution tuning (sensible defaults when omitted).
    timeout_seconds: int = 30
    heartbeat_interval_seconds: float = 10.0
    completion_events: bool = True
    # When True, the plugin also gives the governed dispatcher its own Core
    # governance client (the plugin's shared identity); when False (default,
    # single-client convergence) the plugin's interceptor evaluates and
    # reports lifecycle events and the dispatcher only executes the verdict.
    dispatcher_governance: bool = False
    stdout_bytes: int | None = None
    stderr_bytes: int | None = None
    create_deadline_ms: int | None = None
    readiness_deadline_ms: int | None = None
    exec_deadline_ms: int | None = None
    delete_deadline_ms: int | None = None
    wait_deleted_deadline_ms: int | None = None

    # Optional observers/advanced wiring. When omitted the plugin owns a
    # private heartbeat sink and no telemetry bridge.
    telemetry: Any | None = None
    otel_bridge: Any = None

    def __post_init__(self) -> None:
        from openbox.sandbox.registry import GovernedCommandRegistry

        if not isinstance(self.registry, GovernedCommandRegistry):
            raise TypeError("SandboxConfig.registry must be a GovernedCommandRegistry")
        for name in (
            "service_config",
            "policy",
            "policy_registry_dir",
            "socket_path",
            "ca",
            "certificate",
            "private_key",
            "host_workdir",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Path):
                raise TypeError(f"SandboxConfig.{name} must be a Path or None")
        if isinstance(self.timeout_seconds, bool) or not isinstance(
            self.timeout_seconds, int
        ):
            raise TypeError("SandboxConfig.timeout_seconds must be an int")
        if not 1 <= self.timeout_seconds <= 300:
            raise ValueError("SandboxConfig.timeout_seconds must be 1-300")
        if isinstance(self.heartbeat_interval_seconds, bool) or not isinstance(
            self.heartbeat_interval_seconds, (int, float)
        ):
            raise TypeError(
                "SandboxConfig.heartbeat_interval_seconds must be numeric"
            )
        if not 0.1 <= self.heartbeat_interval_seconds <= 60:
            raise ValueError(
                "SandboxConfig.heartbeat_interval_seconds must be 0.1-60"
            )
        if type(self.completion_events) is not bool:
            raise TypeError("SandboxConfig.completion_events must be a bool")
        if type(self.dispatcher_governance) is not bool:
            raise TypeError("SandboxConfig.dispatcher_governance must be a bool")
        for name in (
            "stdout_bytes",
            "stderr_bytes",
            "create_deadline_ms",
            "readiness_deadline_ms",
            "exec_deadline_ms",
            "delete_deadline_ms",
            "wait_deleted_deadline_ms",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"SandboxConfig.{name} must be a positive int")
        if (
            self.registry_fingerprint is not None
            and _SHA256.fullmatch(self.registry_fingerprint) is None
        ):
            raise ValueError("SandboxConfig.registry_fingerprint must be a sha256")

    @property
    def command_ids(self) -> tuple[str, ...]:
        return self.registry.command_ids


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON object keys (strict parser callback)."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key rejected")
        result[key] = value
    return result


def _parse_json(body: bytes) -> dict[str, Any]:
    try:
        value = json.loads(body.decode("utf-8"), object_pairs_hook=_strict_object)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("service configuration rejected") from None
    if not isinstance(value, dict):
        raise ValueError("service configuration rejected")
    return value


def _trusted_file_bytes(path: Path, *, maximum: int) -> bytes:
    if not path.is_absolute() or not path.is_file():
        raise ValueError("trusted file rejected")
    try:
        body = path.read_bytes()
    except OSError:
        raise ValueError("trusted file rejected") from None
    if not 1 <= len(body) <= maximum:
        raise ValueError("trusted file rejected")
    return body


def _policy_document(policy: Path) -> Any:
    from openbox_sandbox.runtime_client import PolicyDocument

    body = _trusted_file_bytes(policy, maximum=256 * 1024)
    return PolicyDocument("application/yaml", body)


def _policy_registry_resolver(registry_dir: Path) -> Callable[[str], Any]:
    """Resolve ``policy-<id>.yaml`` from the shipped policy registry.

    Resolution fails closed: an unknown id, a missing file, or an unreadable
    document raises, and the dispatch terminates without execution.
    """
    if not registry_dir.is_dir():
        raise RuntimeError("policy registry directory missing")

    def resolve(policy_id: str) -> Any:
        from openbox_sandbox.runtime_client import PolicyDocument

        if not isinstance(policy_id, str) or not re.fullmatch(
            r"[a-z0-9][a-z0-9-]{0,63}", policy_id
        ):
            raise RuntimeError(f"policy id rejected: {policy_id!r}")
        candidate = registry_dir / f"policy-{policy_id}.yaml"
        if not candidate.is_file():
            raise RuntimeError(f"policy not in registry: {policy_id!r}")
        return PolicyDocument("application/yaml", candidate.read_bytes())

    return resolve


def _asset_bundle(document: Mapping[str, Any]) -> Any:
    from openbox_sandbox.runtime_client import AssetBundleIdentity, PolicyIdentity

    value = document.get("asset_bundle")
    if not isinstance(value, dict):
        raise ValueError("sandbox asset bundle rejected")
    policy = value.get("policy")
    if not isinstance(policy, dict):
        raise ValueError("sandbox policy identity rejected")
    for key in (
        "runtime_contract_version",
        "adapter_build_sha256",
        "template",
        "compatibility_id",
    ):
        if key not in value:
            raise ValueError("sandbox asset bundle rejected")
    for key in ("id", "version", "sha256"):
        if key not in policy:
            raise ValueError("sandbox policy identity rejected")
    try:
        return AssetBundleIdentity(
            value["runtime_contract_version"],
            value["adapter_build_sha256"],
            value["template"],
            PolicyIdentity(policy["id"], policy["version"], policy["sha256"]),
            value["compatibility_id"],
        )
    except (TypeError, ValueError):
        raise ValueError("sandbox asset bundle rejected") from None


def _service_boundary(sandbox: SandboxConfig) -> tuple[str, int]:
    if sandbox.service_config is None:
        return "localhost", 0
    document = _parse_json(
        _trusted_file_bytes(sandbox.service_config, maximum=_MAX_JSON_BYTES)
    )
    bind_address = document.get("bind_address")
    if not isinstance(bind_address, str) or ":" not in bind_address:
        raise ValueError("sandbox service bind address rejected")
    host, raw_port = bind_address.rsplit(":", 1)
    if (
        not host
        or not raw_port.isascii()
        or not raw_port.isdecimal()
        or not 1 <= int(raw_port) <= 65535
    ):
        raise ValueError("sandbox service bind address rejected")
    return host, int(raw_port)


def _output_limits(sandbox: SandboxConfig) -> Any:
    from openbox_sandbox.runtime_client import OutputLimits

    stdout_bytes = sandbox.stdout_bytes or _DEFAULT_OUTPUT_LIMITS[0]
    stderr_bytes = sandbox.stderr_bytes or _DEFAULT_OUTPUT_LIMITS[1]
    combined_bytes = stdout_bytes + stderr_bytes
    chunk_bytes = _DEFAULT_OUTPUT_LIMITS[3]
    if (
        combined_bytes > _DEFAULT_OUTPUT_LIMITS[2]
        or chunk_bytes < combined_bytes
    ):
        raise ValueError("sandbox output limits rejected")
    return OutputLimits(
        stdout_bytes, stderr_bytes, combined_bytes, chunk_bytes
    )


def _deadlines(
    sandbox: SandboxConfig,
) -> dict[str, int]:
    # Defaults must stay within the core SDK's validation caps
    # (SandboxExecutionConfig: create<=60s, readiness<=120s, exec<=45s,
    # delete<=60s, wait_deleted<=60s) or every construct is rejected.
    return {
        "create_deadline_ms": sandbox.create_deadline_ms or 60_000,
        "readiness_deadline_ms": sandbox.readiness_deadline_ms or 120_000,
        "exec_deadline_ms": sandbox.exec_deadline_ms or 45_000,
        "delete_deadline_ms": sandbox.delete_deadline_ms or 60_000,
        "wait_deleted_deadline_ms": sandbox.wait_deleted_deadline_ms or 60_000,
    }


def resolve_sandbox_config(
    sandbox: SandboxConfig,
    *,
    openbox_url: str,
    openbox_api_key: str,
    sdk_version: str,
    core_ca_path: str | None = None,
    request_signer: Any | None = None,
) -> Any:
    """Resolve a simplified :class:`SandboxConfig` into a fully wired
    ``TemporalSandboxConfig``.

    Builds the profile bundles from the registry, the heartbeat sink (or uses
    the configured observer), the governed dispatcher (optionally with the
    plugin's shared Core governance client), the sandbox service boundary,
    and the bounded execution limits.
    """
    from openbox_sandbox.dispatcher import (
        DispatcherConfig,
        GovernedDispatcher,
        SandboxExecutionConfig,
        UnixAgentExecutionConfig,
    )
    from openbox_sandbox.dispatcher.governance import GovernanceClientConfig

    from .adapter import TemporalSandboxConfig
    from .heartbeat import TemporalHeartbeatSink

    if not isinstance(sandbox, SandboxConfig):
        raise TypeError("resolve_sandbox_config requires SandboxConfig")

    if sandbox.host_workdir is not None and not sandbox.host_workdir.is_absolute():
        raise ValueError("SandboxConfig.host_workdir must be absolute")
    if sandbox.socket_path is not None and not sandbox.socket_path.is_absolute():
        raise ValueError("SandboxConfig.socket_path must be absolute")

    sink = sandbox.telemetry if sandbox.telemetry is not None else TemporalHeartbeatSink()
    dispatcher_profiles = sandbox.registry.dispatcher_profile_bundle()
    temporal_profiles = sandbox.registry.temporal_profile_bundle()

    if sandbox.policy is not None:
        policy_document = _policy_document(sandbox.policy)
        policy_resolver = None
    elif sandbox.policy_registry_dir is not None:
        policy_resolver = _policy_registry_resolver(sandbox.policy_registry_dir)
        policy_document = policy_resolver("temporal-activity-worker-dev")
    else:
        raise ValueError(
            "SandboxConfig requires either policy or policy_registry_dir"
        )

    if sandbox.service_config is not None:
        service_document = _parse_json(
            _trusted_file_bytes(sandbox.service_config, maximum=_MAX_JSON_BYTES)
        )
        asset = _asset_bundle(service_document)
        host, port = _service_boundary(sandbox)
    else:
        asset = None
        host, port = "localhost", 0

    if sandbox.socket_path is not None:
        if asset is None:
            raise ValueError("SandboxConfig requires service_config with socket_path")
        sandbox_execution = UnixAgentExecutionConfig(
            socket_path=sandbox.socket_path,
            registry_fingerprint=(
                sandbox.registry_fingerprint or sandbox.registry.fingerprint
            ),
            asset_bundle=asset,
            policy_document=policy_document,
            policy_resolver=policy_resolver,
            output_limits=_output_limits(sandbox),
            **_deadlines(sandbox),
        )
    else:
        if (
            sandbox.ca is None
            or sandbox.certificate is None
            or sandbox.private_key is None
            or asset is None
            or not port
        ):
            raise ValueError(
                "SandboxConfig requires ca, certificate, private_key, and "
                "service_config for the direct-TLS boundary"
            )
        sandbox_execution = SandboxExecutionConfig(
            host,
            port,
            "localhost",
            sandbox.ca,
            sandbox.certificate,
            sandbox.private_key,
            asset,
            policy_document,
            _output_limits(sandbox),
            **_deadlines(sandbox),
        )

    governance = (
        GovernanceClientConfig(
            base_url=openbox_url,
            bearer_token=openbox_api_key,
            sdk_version=sdk_version,
            ca_path=None if core_ca_path is None else Path(core_ca_path),
            request_signer=request_signer,
        )
        if sandbox.dispatcher_governance
        else None
    )
    dispatcher = GovernedDispatcher(
        DispatcherConfig(
            governance=governance,
            profiles=dispatcher_profiles,
            sandbox=sandbox_execution,
            host_workdir=(
                sandbox.host_workdir if sandbox.host_workdir is not None else Path.cwd()
            ),
            telemetry=sink,
        )
    )
    return TemporalSandboxConfig(
        dispatcher=dispatcher,
        profiles=temporal_profiles,
        heartbeat_sink=sink,
        timeout_seconds=sandbox.timeout_seconds,
        heartbeat_interval_seconds=sandbox.heartbeat_interval_seconds,
        completion_events=sandbox.completion_events,
        otel_bridge=sandbox.otel_bridge,
    )
