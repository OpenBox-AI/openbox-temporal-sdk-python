"""Strict deployment factory for governed-command Temporal workers.

This module performs file and environment I/O and is intentionally loaded lazily
by :mod:`openbox.sandbox`. Workflow code must continue to import only
``openbox.workflow_commands`` or ``openbox.sandbox.types``.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import os
import re
import stat
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit

from openbox_sandbox.dispatcher import (
    CleanupBacklog,
    CleanupReconciliationResult,
    CommandProfileBundle,
    DispatcherConfig,
    GovernanceClientConfig,
    GovernedDispatcher,
    SandboxExecutionConfig,
)
from openbox_sandbox.runtime_client import (
    AssetBundleIdentity,
    OutputLimits,
    PolicyDocument,
    PolicyIdentity,
)
from temporalio.client import Client
from temporalio.common import VersioningBehavior, WorkerDeploymentVersion
from temporalio.service import TLSConfig
from temporalio.worker import Worker, WorkerDeploymentConfig

from openbox.errors import OpenBoxConfigError
from openbox.worker import create_openbox_worker

from .adapter import TemporalSandboxConfig
from .errors import GovernedCommandDeploymentError
from .heartbeat import TemporalHeartbeatSink
from .profiles import TemporalCommandProfileBundle
from .signing import AipEd25519RequestSigner

DEPLOYMENT_ENV = "OPENBOX_GOVERNED_COMMAND_DEPLOYMENT"
_MAX_JSON_BYTES = 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}\Z")
_IMAGE = re.compile(r"[^\s]+@sha256:[0-9a-f]{64}\Z")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GovernedCommandDeploymentError()
        result[key] = value
    return result


def _reject_constant(_: str) -> None:
    raise GovernedCommandDeploymentError()


def _trusted_file_bytes(
    path: Path,
    *,
    maximum: int | None = None,
    private: bool = False,
    read_data: bool = True,
) -> bytes:
    """Open or read one owner-controlled regular file via a verified descriptor.

    ``O_NOFOLLOW`` closes the symlink race where supported. The lstat/inode check
    is a best-effort fallback on platforms without it; mounts still must be
    immutable and owner-controlled because sibling path replacement cannot be
    made impossible portably.
    """
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    before = None
    try:
        if nofollow:
            flags |= nofollow
        else:
            before = os.lstat(path)
            if stat.S_ISLNK(before.st_mode):
                raise GovernedCommandDeploymentError()
        descriptor = os.open(path, flags)
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.getuid()
                or metadata.st_mode & (0o077 if private else 0o022)
                or metadata.st_size <= 0
                or (maximum is not None and metadata.st_size > maximum)
                or (
                    before is not None
                    and (before.st_dev, before.st_ino)
                    != (metadata.st_dev, metadata.st_ino)
                )
            ):
                raise GovernedCommandDeploymentError()
            if not read_data:
                return b""
            limit = metadata.st_size if maximum is None else maximum
            chunks: list[bytes] = []
            remaining = limit + 1
            while remaining:
                chunk = os.read(descriptor, min(remaining, 64 * 1024))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            value = b"".join(chunks)
            if (
                not value
                or len(value) != metadata.st_size
                or (maximum is not None and len(value) > maximum)
            ):
                raise GovernedCommandDeploymentError()
            return value
        finally:
            os.close(descriptor)
    except GovernedCommandDeploymentError:
        raise
    except (OSError, ValueError):
        raise GovernedCommandDeploymentError() from None


def _validate_trusted_file(path: Path, *, private: bool = False) -> None:
    _trusted_file_bytes(path, private=private, read_data=False)


def _parse_json(body: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            body,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (
        json.JSONDecodeError,
        UnicodeDecodeError,
        GovernedCommandDeploymentError,
    ):
        raise GovernedCommandDeploymentError() from None
    if not isinstance(value, dict):
        raise GovernedCommandDeploymentError()
    return value


def _load_json(path: Path) -> dict[str, Any]:
    return _parse_json(_trusted_file_bytes(path, maximum=_MAX_JSON_BYTES))


def _exact(value: object, fields: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise GovernedCommandDeploymentError()
    return value


def _string(value: object, *, maximum: int = 4096) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(ord(character) < 32 for character in value)
        or len(value.encode("utf-8")) > maximum
    ):
        raise GovernedCommandDeploymentError()
    return value


def _integer(value: object, minimum: int, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise GovernedCommandDeploymentError()
    return value


def _number(value: object, minimum: float, maximum: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not minimum <= value <= maximum
    ):
        raise GovernedCommandDeploymentError()
    return float(value)


def _boolean(value: object) -> bool:
    if type(value) is not bool:
        raise GovernedCommandDeploymentError()
    return value


def _absolute_path(value: object) -> Path:
    path = Path(_string(value))
    if not path.is_absolute():
        raise GovernedCommandDeploymentError()
    return path


def _env_name(value: object) -> str:
    name = _string(value, maximum=128)
    if _ENV_NAME.fullmatch(name) is None:
        raise GovernedCommandDeploymentError()
    return name


def _secret(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name)
    if not isinstance(value, str) or not value or "\r" in value or "\n" in value:
        raise GovernedCommandDeploymentError()
    return value


def _read_bytes(path: Path) -> bytes:
    return _trusted_file_bytes(path, maximum=_MAX_JSON_BYTES)


def _parse_bind_address(value: object) -> tuple[str, int]:
    text = _string(value, maximum=128)
    try:
        parsed = urlsplit("//" + text)
        host = parsed.hostname
        port = parsed.port
        address = ipaddress.ip_address(host or "")
    except ValueError:
        raise GovernedCommandDeploymentError() from None
    if (
        host is None
        or port is None
        or parsed.path
        or parsed.query
        or parsed.fragment
        or not address.is_loopback
        or not 1 <= port <= 65535
    ):
        raise GovernedCommandDeploymentError()
    return str(address), port


def _parse_service_config(path: Path) -> tuple[str, int, AssetBundleIdentity]:
    required_fields = {
        "bind_address",
        "server_certificate_path",
        "server_private_key_path",
        "client_ca_path",
        "authorized_callers",
        "state_directory",
        "asset_bundle",
        "runtime_endpoint",
        "runtime_mtls_directory",
        "runtime_connect_timeout_ms",
        "runtime_poll_interval_ms",
        "reconcile_delete_deadline_ms",
        "reconcile_wait_deadline_ms",
        "maximum_connections",
        "drain_timeout_ms",
    }
    # Optional non-prod field consumed only by sandbox-service. The Worker never
    # acts on it, but must accept it when the shared service config enables the
    # degraded Landlock tier (macOS libkrun guests have no Landlock LSM).
    optional_fields = {"allow_degraded_landlock"}
    raw_root = _load_json(path)
    if not isinstance(raw_root, dict):
        raise GovernedCommandDeploymentError()
    present = set(raw_root)
    if present - required_fields - optional_fields or required_fields - present:
        raise GovernedCommandDeploymentError()
    if "allow_degraded_landlock" in raw_root:
        _boolean(raw_root["allow_degraded_landlock"])
    root = raw_root
    host, port = _parse_bind_address(root["bind_address"])
    # These paths belong to sandbox-service and its runtime identity. Validate
    # their absolute shape from the trusted config, but never open them from the
    # Worker process: in a least-privilege deployment the service private key is
    # intentionally unreadable by the Temporal Worker UID.
    for name in (
        "server_certificate_path",
        "server_private_key_path",
        "client_ca_path",
        "state_directory",
        "runtime_mtls_directory",
    ):
        _absolute_path(root[name])
    runtime_endpoint = _string(root["runtime_endpoint"])
    endpoint = urlsplit(runtime_endpoint)
    try:
        _ = endpoint.port
    except ValueError:
        raise GovernedCommandDeploymentError() from None
    if (
        endpoint.scheme != "https"
        or not endpoint.hostname
        or endpoint.username is not None
        or endpoint.password is not None
        or endpoint.query
        or endpoint.fragment
        or endpoint.path not in {"", "/"}
    ):
        raise GovernedCommandDeploymentError()
    for name, maximum in (
        ("runtime_connect_timeout_ms", 120_000),
        ("runtime_poll_interval_ms", 60_000),
        ("reconcile_delete_deadline_ms", 120_000),
        ("reconcile_wait_deadline_ms", 120_000),
        ("drain_timeout_ms", 120_000),
    ):
        _integer(root[name], 1, maximum)
    _integer(root["maximum_connections"], 1, 65_536)
    callers = root["authorized_callers"]
    if not isinstance(callers, list) or not callers or len(callers) > 1024:
        raise GovernedCommandDeploymentError()
    fingerprints: set[str] = set()
    for raw in callers:
        caller = _exact(raw, {"certificate_sha256", "role"})
        fingerprint = _string(caller["certificate_sha256"], maximum=64)
        if (
            _SHA256.fullmatch(fingerprint) is None
            or fingerprint in fingerprints
            or caller["role"] not in {"runtime", "administrator"}
        ):
            raise GovernedCommandDeploymentError()
        fingerprints.add(fingerprint)

    raw_bundle = _exact(
        root["asset_bundle"],
        {
            "runtime_contract_version",
            "adapter_build_sha256",
            "template",
            "policy",
            "compatibility_id",
        },
    )
    raw_policy = _exact(raw_bundle["policy"], {"id", "version", "sha256"})
    template = _string(raw_bundle["template"])
    adapter_hash = _string(raw_bundle["adapter_build_sha256"], maximum=64)
    policy_hash = _string(raw_policy["sha256"], maximum=64)
    if (
        _IMAGE.fullmatch(template) is None
        or _SHA256.fullmatch(adapter_hash) is None
        or _SHA256.fullmatch(policy_hash) is None
    ):
        raise GovernedCommandDeploymentError()
    try:
        bundle = AssetBundleIdentity(
            runtime_contract_version=_integer(
                raw_bundle["runtime_contract_version"], 1, 2**31 - 1
            ),
            adapter_build_sha256=adapter_hash,
            template=template,
            policy=PolicyIdentity(
                id=_string(raw_policy["id"], maximum=128),
                version=_integer(raw_policy["version"], 1, 2**31 - 1),
                sha256=policy_hash,
            ),
            compatibility_id=_string(raw_bundle["compatibility_id"], maximum=128),
        )
    except (TypeError, ValueError):
        raise GovernedCommandDeploymentError() from None
    return host, port, bundle


@dataclass(frozen=True, slots=True)
class GovernedCommandWorkerVersioning:
    """Validated modern Temporal Worker Deployment settings."""

    deployment_name: str
    build_id: str
    use_worker_versioning: bool
    default_behavior: str


@dataclass(slots=True, repr=False)
class GovernedCommandDeployment:
    """A fully validated governed-command deployment awaiting reconciliation."""

    deployment_id: str
    manifest_path: Path
    core_base_url: str
    core_ca_path: Path
    core_sdk_version: str
    core_timeout_seconds: float
    schema_version: int
    temporal_target: str
    temporal_namespace: str | None
    temporal_tls_required: bool | None
    task_queue: str
    graceful_shutdown_seconds: int | None
    worker_versioning: GovernedCommandWorkerVersioning | None
    profile_bundle_version: str
    profile_key_id: str
    profile_ids: tuple[str, ...]
    sandbox_enabled: bool
    sandbox: TemporalSandboxConfig
    _bearer_token: str = field(repr=False)
    _agent_did: str | None = field(repr=False)
    _agent_private_key: str | None = field(repr=False)
    _state: str = field(default="new", init=False, repr=False)
    _state_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    @property
    def prepared(self) -> bool:
        with self._state_lock:
            return self._state == "prepared"

    async def prepare(self) -> CleanupReconciliationResult:
        """Reconcile cleanup exactly once before one Worker construction attempt."""
        with self._state_lock:
            if self._state not in {"new", "failed"}:
                raise GovernedCommandDeploymentError()
            self._state = "preparing"
        try:
            result = await self.sandbox.dispatcher.reconcile_cleanup()
            if (
                not isinstance(result, CleanupReconciliationResult)
                or any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                    for value in (result.attempted, result.deleted, result.remaining)
                )
                or result.deleted > result.attempted
                or result.remaining > result.attempted
                or result.deleted + result.remaining != result.attempted
                or result.remaining != 0
            ):
                raise GovernedCommandDeploymentError()
        except BaseException as error:
            with self._state_lock:
                if self._state == "preparing":
                    self._state = "failed"
            if isinstance(error, Exception):
                raise GovernedCommandDeploymentError() from None
            raise
        with self._state_lock:
            if self._state != "preparing":
                self._state = "failed"
                raise GovernedCommandDeploymentError()
            self._state = "prepared"
        return result

    def reserve_for_worker(self) -> None:
        """Reserve this deployment for automatic Worker lifecycle preparation."""
        with self._state_lock:
            if self._state != "new":
                raise GovernedCommandDeploymentError()
            self._state = "reserved"

    async def prepare_reserved_worker(self) -> CleanupReconciliationResult:
        """Prepare one reserved automatic Worker before its first poll."""
        with self._state_lock:
            if self._state != "reserved":
                raise GovernedCommandDeploymentError()
            self._state = "preparing"
        try:
            result = await self.sandbox.dispatcher.reconcile_cleanup()
            if (
                not isinstance(result, CleanupReconciliationResult)
                or any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                    for value in (result.attempted, result.deleted, result.remaining)
                )
                or result.deleted > result.attempted
                or result.remaining > result.attempted
                or result.deleted + result.remaining != result.attempted
                or result.remaining != 0
            ):
                raise GovernedCommandDeploymentError()
        except BaseException as error:
            with self._state_lock:
                self._state = "failed"
            if isinstance(error, Exception):
                raise GovernedCommandDeploymentError() from None
            raise
        with self._state_lock:
            if self._state != "preparing":
                self._state = "failed"
                raise GovernedCommandDeploymentError()
            self._state = "consumed"
        return result

    def worker_deployment_config(self) -> WorkerDeploymentConfig | None:
        """Build the validated Temporal Worker Deployment configuration."""
        if self.worker_versioning is None:
            return None
        default_behavior = {
            "pinned": VersioningBehavior.PINNED,
            "auto_upgrade": VersioningBehavior.AUTO_UPGRADE,
        }[self.worker_versioning.default_behavior]
        return WorkerDeploymentConfig(
            version=WorkerDeploymentVersion(
                deployment_name=self.worker_versioning.deployment_name,
                build_id=self.worker_versioning.build_id,
            ),
            use_worker_versioning=self.worker_versioning.use_worker_versioning,
            default_versioning_behavior=default_behavior,
        )

    def create_worker(
        self,
        client: Client,
        *,
        workflows: Sequence[type[Any]] = (),
        activities: Sequence[Callable[..., Any]] = (),
    ) -> Worker:
        """Create the fixed fail-closed worker after :meth:`prepare` succeeds."""
        try:
            client_config = client.service_client.config
            target_host = client_config.target_host
            if self.schema_version >= 2:
                namespace = client.namespace
                tls = client_config.tls
        except (AttributeError, TypeError):
            raise GovernedCommandDeploymentError() from None
        if not isinstance(target_host, str) or target_host != self.temporal_target:
            raise GovernedCommandDeploymentError()
        if self.schema_version >= 2:
            if not isinstance(namespace, str) or namespace != self.temporal_namespace:
                raise GovernedCommandDeploymentError()
            if self.temporal_tls_required:
                if tls is not True and not isinstance(tls, TLSConfig):
                    raise GovernedCommandDeploymentError()
            elif tls is not False:
                raise GovernedCommandDeploymentError()

        deployment_config = self.worker_deployment_config()

        with self._state_lock:
            if self._state != "prepared":
                raise GovernedCommandDeploymentError()
            self._state = "consumed"
        try:
            return create_openbox_worker(
                client=client,
                task_queue=self.task_queue,
                workflows=workflows,
                activities=activities,
                openbox_url=self.core_base_url,
                openbox_api_key=self._bearer_token,
                agent_did=self._agent_did,
                agent_private_key=self._agent_private_key,
                core_ca_path=str(self.core_ca_path),
                governance_timeout=self.core_timeout_seconds,
                governance_policy="fail_closed",
                sandbox=self.sandbox,
                instrument_http=False,
                instrument_databases=False,
                instrument_file_io=False,
                enable_trace_propagation=False,
                graceful_shutdown_timeout=timedelta(
                    seconds=self.graceful_shutdown_seconds or 0
                ),
                deployment_config=deployment_config,
            )
        except BaseException:
            with self._state_lock:
                self._state = "failed"
            raise

    def __repr__(self) -> str:
        return (
            f"GovernedCommandDeployment(deployment_id={self.deployment_id!r}, "
            f"temporal_target={self.temporal_target!r}, task_queue={self.task_queue!r}, "
            f"profile_bundle_version={self.profile_bundle_version!r}, "
            f"profile_ids={self.profile_ids!r}, sandbox_enabled="
            f"{self.sandbox_enabled}, credentials=<redacted>, "
            f"prepared={self.prepared})"
        )


def _load_governed_command_deployment(
    *,
    environment: Mapping[str, str] | None = None,
    now: datetime | None = None,
    expected_manifest_sha256: str | None = None,
) -> GovernedCommandDeployment:
    """Internal loader; the public wrapper below normalizes every failure."""
    env = os.environ if environment is None else environment
    selected = env.get(DEPLOYMENT_ENV)
    if not isinstance(selected, str) or not selected:
        raise GovernedCommandDeploymentError()
    manifest_path = _absolute_path(selected)
    manifest_body = _trusted_file_bytes(manifest_path, maximum=_MAX_JSON_BYTES)
    if (
        expected_manifest_sha256 is not None
        and hashlib.sha256(manifest_body).hexdigest() != expected_manifest_sha256
    ):
        raise GovernedCommandDeploymentError()
    raw_root = _parse_json(manifest_body)
    schema_version = raw_root.get("schema_version")
    if type(schema_version) is not int or schema_version not in {1, 2, 3}:
        raise GovernedCommandDeploymentError()
    root_fields = {
        "schema_version",
        "deployment_id",
        "core",
        "temporal",
        "sandbox_service",
        "policy",
        "profiles",
        "host_workdir",
        "cleanup_backlog_directory",
        "timeouts",
        "output_limits",
        "sandbox_enabled",
        "completion_events",
    }
    if schema_version >= 2:
        root_fields.add("worker")
    if schema_version == 3:
        root_fields.add("otel")
    root = _exact(raw_root, root_fields)
    deployment_id = _string(root["deployment_id"], maximum=128)
    if _IDENTIFIER.fullmatch(deployment_id) is None:
        raise GovernedCommandDeploymentError()

    core = _exact(
        root["core"],
        {
            "base_url",
            "ca_path",
            "timeout_seconds",
            "sdk_version",
            "bearer_token_env",
            "aip_signer",
        },
    )
    base_url = _string(core["base_url"])
    parsed_url = urlsplit(base_url)
    if (
        parsed_url.scheme != "https"
        or not parsed_url.hostname
        or parsed_url.username is not None
        or parsed_url.password is not None
        or parsed_url.query
        or parsed_url.fragment
        or parsed_url.path not in {"", "/"}
    ):
        raise GovernedCommandDeploymentError()
    core_ca_path = _absolute_path(core["ca_path"])
    _validate_trusted_file(core_ca_path)
    core_timeout = _number(core["timeout_seconds"], 0.1, 30)
    sdk_version = _string(core["sdk_version"], maximum=128)
    bearer_token = _secret(env, _env_name(core["bearer_token_env"]))

    agent_did: str | None = None
    agent_private_key: str | None = None
    signer: AipEd25519RequestSigner | None = None
    signer_config = core["aip_signer"]
    if signer_config is not None:
        signer_fields = _exact(
            signer_config, {"agent_did_env", "agent_private_key_env"}
        )
        agent_did = _secret(env, _env_name(signer_fields["agent_did_env"]))
        agent_private_key = _secret(
            env, _env_name(signer_fields["agent_private_key_env"])
        )
        try:
            signer = AipEd25519RequestSigner.from_base64_seed(
                agent_did, agent_private_key
            )
        except (OpenBoxConfigError, TypeError, ValueError):
            raise GovernedCommandDeploymentError() from None

    temporal_namespace: str | None = None
    temporal_tls_required: bool | None = None
    graceful_shutdown_seconds: int | None = None
    worker_versioning: GovernedCommandWorkerVersioning | None = None
    if schema_version == 1:
        temporal = _exact(root["temporal"], {"target", "task_queue"})
    else:
        temporal = _exact(
            root["temporal"], {"target", "namespace", "task_queue", "tls_required"}
        )
        temporal_namespace = _string(temporal["namespace"], maximum=256)
        temporal_tls_required = _boolean(temporal["tls_required"])

        worker = _exact(root["worker"], {"graceful_shutdown_seconds", "versioning"})
        graceful_shutdown_seconds = _integer(
            worker["graceful_shutdown_seconds"], 1, 600
        )
        raw_versioning = worker["versioning"]
        if raw_versioning is not None:
            versioning = _exact(
                raw_versioning,
                {
                    "deployment_name",
                    "build_id",
                    "use_worker_versioning",
                    "default_behavior",
                },
            )
            deployment_name = _string(versioning["deployment_name"], maximum=128)
            build_id = _string(versioning["build_id"], maximum=128)
            default_behavior = versioning["default_behavior"]
            if (
                _IDENTIFIER.fullmatch(deployment_name) is None
                or _IDENTIFIER.fullmatch(build_id) is None
                or default_behavior not in {"pinned", "auto_upgrade"}
            ):
                raise GovernedCommandDeploymentError()
            use_worker_versioning = _boolean(versioning["use_worker_versioning"])
            if not use_worker_versioning:
                raise GovernedCommandDeploymentError()
            worker_versioning = GovernedCommandWorkerVersioning(
                deployment_name=deployment_name,
                build_id=build_id,
                use_worker_versioning=use_worker_versioning,
                default_behavior=default_behavior,
            )

    if schema_version == 3:
        otel = _exact(root["otel"], {"export_mode"})
        if otel["export_mode"] != "local":
            raise GovernedCommandDeploymentError()

    temporal_target = _string(temporal["target"], maximum=256)
    task_queue = _string(temporal["task_queue"], maximum=256)

    service = _exact(
        root["sandbox_service"],
        {
            "config_path",
            "client_ca_path",
            "client_certificate_path",
            "client_private_key_path",
            "server_name",
        },
    )
    service_config_path = _absolute_path(service["config_path"])
    host, port, asset_bundle = _parse_service_config(service_config_path)
    client_ca_path = _absolute_path(service["client_ca_path"])
    client_certificate_path = _absolute_path(service["client_certificate_path"])
    client_private_key_path = _absolute_path(service["client_private_key_path"])
    _validate_trusted_file(client_ca_path)
    _validate_trusted_file(client_certificate_path)
    _validate_trusted_file(client_private_key_path, private=True)
    server_name = _string(service["server_name"], maximum=253)

    policy = _exact(root["policy"], {"path", "media_type"})
    policy_path = _absolute_path(policy["path"])
    policy_bytes = _read_bytes(policy_path)
    if hashlib.sha256(policy_bytes).hexdigest() != asset_bundle.policy.sha256:
        raise GovernedCommandDeploymentError()
    policy_document = PolicyDocument(
        _string(policy["media_type"], maximum=128), policy_bytes
    )

    profiles = _exact(
        root["profiles"],
        {"dispatcher_path", "temporal_path", "key_id", "hmac_secret_env"},
    )
    dispatcher_profile_path = _absolute_path(profiles["dispatcher_path"])
    temporal_profile_path = _absolute_path(profiles["temporal_path"])
    profile_key_id = _string(profiles["key_id"], maximum=128)
    profile_secret = _secret(env, _env_name(profiles["hmac_secret_env"])).encode()
    try:
        dispatcher_profiles = CommandProfileBundle.load(
            _read_bytes(dispatcher_profile_path),
            secret=profile_secret,
            expected_key_id=profile_key_id,
            now=now,
        )
        temporal_profiles = TemporalCommandProfileBundle.load(
            _read_bytes(temporal_profile_path),
            secret=profile_secret,
            expected_key_id=profile_key_id,
            now=now,
        )
    except (TypeError, ValueError):
        raise GovernedCommandDeploymentError() from None
    metadata_matches = (
        dispatcher_profiles.key_id == temporal_profiles.key_id == profile_key_id
        and dispatcher_profiles.bundle_version == temporal_profiles.bundle_version
        and dispatcher_profiles.issued_at == temporal_profiles.issued_at
        and dispatcher_profiles.expires_at == temporal_profiles.expires_at
        and dispatcher_profiles.profile_ids == temporal_profiles.profile_ids
    )
    if not metadata_matches:
        raise GovernedCommandDeploymentError()

    timeouts = _exact(
        root["timeouts"],
        {
            "command_seconds",
            "heartbeat_interval_seconds",
            "create_deadline_ms",
            "readiness_deadline_ms",
            "exec_deadline_ms",
            "delete_deadline_ms",
            "wait_deleted_deadline_ms",
        },
    )
    command_seconds = _integer(timeouts["command_seconds"], 1, 300)
    heartbeat_interval = _number(timeouts["heartbeat_interval_seconds"], 0.1, 60)
    create_deadline = _integer(timeouts["create_deadline_ms"], 1, 60_000)
    readiness_deadline = _integer(timeouts["readiness_deadline_ms"], 1, 120_000)
    exec_deadline = _integer(timeouts["exec_deadline_ms"], 1, 45_000)
    delete_deadline = _integer(timeouts["delete_deadline_ms"], 1, 60_000)
    wait_deleted_deadline = _integer(timeouts["wait_deleted_deadline_ms"], 1, 60_000)
    if schema_version >= 2:
        minimum_graceful_shutdown = (
            command_seconds
            + (delete_deadline + 999) // 1000
            + (wait_deleted_deadline + 999) // 1000
        )
        if (
            graceful_shutdown_seconds is None
            or graceful_shutdown_seconds < minimum_graceful_shutdown
        ):
            raise GovernedCommandDeploymentError()

    limits = _exact(
        root["output_limits"],
        {"stdout_bytes", "stderr_bytes", "combined_bytes", "chunk_bytes"},
    )
    stdout_bytes = _integer(limits["stdout_bytes"], 1, 4 * 1024 * 1024)
    stderr_bytes = _integer(limits["stderr_bytes"], 1, 4 * 1024 * 1024)
    combined_bytes = _integer(limits["combined_bytes"], 1, 4 * 1024 * 1024)
    chunk_bytes = _integer(limits["chunk_bytes"], 1, 4 * 1024 * 1024)
    if combined_bytes < max(stdout_bytes, stderr_bytes):
        raise GovernedCommandDeploymentError()
    output_limits = OutputLimits(
        stdout_bytes=stdout_bytes,
        stderr_bytes=stderr_bytes,
        combined_bytes=combined_bytes,
        chunk_bytes=chunk_bytes,
    )

    heartbeat_sink = TemporalHeartbeatSink()
    otel_bridge = None
    if schema_version == 3:
        from .otel_telemetry import GovernedCommandTelemetryBridge

        otel_bridge = GovernedCommandTelemetryBridge(asset_bundle.template)
        heartbeat_sink.attach_otel_bridge(otel_bridge)
    sandbox_execution = SandboxExecutionConfig(
        host=host,
        port=port,
        server_name=server_name,
        ca_path=client_ca_path,
        certificate_path=client_certificate_path,
        private_key_path=client_private_key_path,
        asset_bundle=asset_bundle,
        policy_document=policy_document,
        output_limits=output_limits,
        create_deadline_ms=create_deadline,
        readiness_deadline_ms=readiness_deadline,
        exec_deadline_ms=exec_deadline,
        delete_deadline_ms=delete_deadline,
        wait_deleted_deadline_ms=wait_deleted_deadline,
        enabled=_boolean(root["sandbox_enabled"]),
    )
    governance = GovernanceClientConfig(
        base_url=base_url,
        bearer_token=bearer_token,
        sdk_version=sdk_version,
        ca_path=core_ca_path,
        timeout_seconds=core_timeout,
        request_signer=signer,
    )
    cleanup_backlog = CleanupBacklog(
        directory=_absolute_path(root["cleanup_backlog_directory"]),
        compatibility_id=asset_bundle.compatibility_id,
    )
    dispatcher_config = DispatcherConfig(
        governance=governance,
        profiles=dispatcher_profiles,
        sandbox=sandbox_execution,
        host_workdir=_absolute_path(root["host_workdir"]),
        telemetry=heartbeat_sink,
        cleanup_backlog=cleanup_backlog,
    )
    try:
        dispatcher = GovernedDispatcher(dispatcher_config)
        temporal_sandbox = TemporalSandboxConfig(
            dispatcher=dispatcher,
            profiles=temporal_profiles,
            heartbeat_sink=heartbeat_sink,
            timeout_seconds=command_seconds,
            heartbeat_interval_seconds=heartbeat_interval,
            completion_events=_boolean(root["completion_events"]),
            otel_bridge=otel_bridge,
        )
    except (OSError, TypeError, ValueError):
        raise GovernedCommandDeploymentError() from None
    return GovernedCommandDeployment(
        deployment_id=deployment_id,
        manifest_path=manifest_path,
        core_base_url=base_url,
        core_ca_path=core_ca_path,
        core_sdk_version=sdk_version,
        core_timeout_seconds=core_timeout,
        schema_version=schema_version,
        temporal_target=temporal_target,
        temporal_namespace=temporal_namespace,
        temporal_tls_required=temporal_tls_required,
        task_queue=task_queue,
        graceful_shutdown_seconds=graceful_shutdown_seconds,
        worker_versioning=worker_versioning,
        profile_bundle_version=dispatcher_profiles.bundle_version,
        profile_key_id=profile_key_id,
        profile_ids=dispatcher_profiles.profile_ids,
        sandbox_enabled=sandbox_execution.enabled,
        sandbox=temporal_sandbox,
        _bearer_token=bearer_token,
        _agent_did=agent_did,
        _agent_private_key=agent_private_key,
    )


def load_governed_command_deployment(
    *,
    environment: Mapping[str, str] | None = None,
    now: datetime | None = None,
) -> GovernedCommandDeployment:
    """Load one deployment while exposing only the constant public error."""
    try:
        return _load_governed_command_deployment(environment=environment, now=now)
    except GovernedCommandDeploymentError:
        raise GovernedCommandDeploymentError() from None
    except Exception:
        raise GovernedCommandDeploymentError() from None
