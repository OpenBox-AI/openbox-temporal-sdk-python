from __future__ import annotations

import base64
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openbox_sandbox import (
    IdentifierResultField,
    IntegerResultField,
    SandboxCommandDefinition,
    SandboxEngineConfig,
    SandboxExecutionConfig,
    SandboxExecutionEngine,
    TypedJsonResultSchema,
    sandbox_command_registry,
)
from openbox_sandbox.runtime import (
    AssetBundleIdentity,
    OutputLimits,
    PolicyDocument,
    PolicyIdentity,
    ServiceResponse,
)

from openbox.sandbox.adapter import TemporalSandboxConfig
from openbox.sandbox.heartbeat import TemporalHeartbeatSink

NOW = datetime(2026, 7, 22, tzinfo=timezone.utc)
SANDBOX_ID = "sbx-550e8400-e29b-41d4-a716-446655440000"


class FakeSandboxRuntime:
    def __init__(self) -> None:
        asset = asset_bundle()
        self.calls: list[str] = []
        self.values = {
            "create": ServiceResponse(
                "created",
                {
                    "request_id": SANDBOX_ID,
                    "lifecycle_token": "550e8400-e29b-41d4-a716-446655440001",
                },
            ),
            "wait_ready": ServiceResponse(
                "ready",
                {
                    "request_id": SANDBOX_ID,
                    "lifecycle_token": "550e8400-e29b-41d4-a716-446655440002",
                    "active_policy": asset.policy.to_wire(),
                },
            ),
            "exec": ServiceResponse(
                "executed",
                {
                    "result": {
                        "exit_code": 0,
                        "stdout_base64": base64.b64encode(b"ok").decode(),
                        "stderr_base64": "",
                        "timeout": "not_observed",
                    }
                },
            ),
            "delete": ServiceResponse("deleted", {"outcome": "deleted"}),
            "wait_deleted": ServiceResponse("terminally_absent", {}),
        }

    async def _call(self, name: str, *args: Any) -> Any:
        self.calls.append(name)
        return self.values[name]

    async def create(self, *args: Any) -> Any:
        return await self._call("create", *args)

    async def wait_ready(self, *args: Any) -> Any:
        return await self._call("wait_ready", *args)

    async def exec(self, *args: Any) -> Any:
        return await self._call("exec", *args)

    async def delete(self, *args: Any) -> Any:
        return await self._call("delete", *args)

    async def wait_deleted(self, *args: Any) -> Any:
        return await self._call("wait_deleted", *args)


def asset_bundle() -> AssetBundleIdentity:
    return AssetBundleIdentity(
        1,
        "a" * 64,
        "registry.invalid/sandbox@sha256:" + "c" * 64,
        PolicyIdentity("deny-network", 1, "b" * 64),
        "test-v1",
    )


def sandbox_config(
    *,
    trust_application_agent: bool = True,
    dispatcher: Any = None,
    governed_command_factory: Any = None,
    receipt_verifier: Any = None,
    typed_result: bool = False,
) -> tuple[TemporalSandboxConfig, FakeSandboxRuntime]:
    result_schema = (
        TypedJsonResultSchema(
            "proof-v1",
            (
                IdentifierResultField("status", max_bytes=16),
                IntegerResultField("count", 0, 100),
            ),
        )
        if typed_result
        else None
    )
    registry = sandbox_command_registry(
        SandboxCommandDefinition("proof", "/bin/echo", result_schema=result_schema)
    )
    heartbeat = TemporalHeartbeatSink()
    runtime = FakeSandboxRuntime()
    if typed_result:
        runtime.values["exec"] = ServiceResponse(
            "executed",
            {
                "result": {
                    "exit_code": 0,
                    "stdout_base64": base64.b64encode(
                        b'{"count":2,"status":"ok"}'
                    ).decode(),
                    "stderr_base64": "",
                    "timeout": "not_observed",
                }
            },
        )
    engine = SandboxExecutionEngine._from_components(
        SandboxEngineConfig(
            profiles=registry.admission_profile_bundle(),
            sandbox=SandboxExecutionConfig(
                host="127.0.0.1",
                port=7443,
                server_name="sandbox.invalid",
                ca_path=Path("/ca"),
                certificate_path=Path("/cert"),
                private_key_path=Path("/key"),
                asset_bundle=asset_bundle(),
                policy_document=PolicyDocument("application/yaml", b"version: 1\n"),
                output_limits=OutputLimits(1024, 1024, 1536, 4096),
            ),
            telemetry=heartbeat,
        ),
        sandbox=runtime,
        clock=lambda: NOW,
        sandbox_id=lambda: SANDBOX_ID,
    )
    return (
        TemporalSandboxConfig(
            engine=(
                None
                if dispatcher is not None or governed_command_factory is not None
                else engine
            ),
            profiles=registry.structured_profile_bundle(),
            heartbeat_sink=heartbeat,
            heartbeat_interval_seconds=60,
            dispatcher=dispatcher,
            governed_command_factory=governed_command_factory,
            receipt_verifier=receipt_verifier,
            trust_application_agent=trust_application_agent,
        ),
        runtime,
    )
