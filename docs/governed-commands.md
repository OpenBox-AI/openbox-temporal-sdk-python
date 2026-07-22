# Governed sandbox commands

Install the optional integration with:

```bash
pip install "openbox-temporal-sdk-python[sandbox]"
```

The Temporal package is a thin framework wrapper over `openbox-sandbox-sdk-python`. It owns deterministic history conversion, one-attempt Activity scheduling, heartbeat/cancellation behavior, Worker/plugin registration, and Temporal error/result mapping. The shared sandbox package owns profiles, runtime transport, sandbox lifecycle, telemetry, bounded results, and cleanup.

## Authorization order

OpenBox Core authorizes, but does not start or clean up the local sandbox.

1. The application agent evaluates the proposed operation with Core.
2. A successful, strict Core response must be `CONSTRAIN` before the application starts the Workflow.
3. The application starts the Workflow with bounded business input.
4. Workflow code schedules `openbox_governed_command` exactly once.
5. The Activity interceptor maps the request into an already-authorized sandbox execution.

`ALLOW` remains the application's native framework execution path and must not enter `SandboxExecutionEngine`. An uncertain exec is never retried.

## Workflow code

Workflow code imports only the deterministic helper and bounded history contracts:

```python
from datetime import timedelta

from openbox.workflow_commands import execute_governed_command
from openbox_sandbox import SandboxCommandRequest
from temporalio import workflow


@workflow.defn
class ReconciliationWorkflow:
    @workflow.run
    async def run(self, batch_id: str):
        return await execute_governed_command(
            SandboxCommandRequest("reconcile", {"batch_id": batch_id}),
            start_to_close_timeout=timedelta(minutes=6),
            heartbeat_timeout=timedelta(minutes=2),
        )
```

The helper sets `RetryPolicy(maximum_attempts=1)` and `WAIT_CANCELLATION_COMPLETED`. Raw argv, credentials, policy documents, and sandbox output do not enter Workflow history. A profile may return bounded typed JSON values; otherwise the durable result contains metadata only.

## Dedicated sandbox Worker

Build one typed registry and derive both bundles from it:

```python
from openbox.sandbox import (
    TemporalHeartbeatSink,
    TemporalSandboxConfig,
    create_sandbox_worker,
)
from openbox_sandbox import (
    IdentifierArgument,
    SandboxCommandDefinition,
    SandboxEngineConfig,
    SandboxExecutionEngine,
    sandbox_command_registry,
)

registry = sandbox_command_registry(
    SandboxCommandDefinition(
        "reconcile",
        "/usr/local/bin/reconcile",
        (IdentifierArgument("batch_id"),),
    )
)
heartbeat = TemporalHeartbeatSink()
engine = SandboxExecutionEngine(
    SandboxEngineConfig(
        profiles=registry.admission_profile_bundle(),
        sandbox=sandbox_execution_config,
        telemetry=heartbeat,
        cleanup_backlog=cleanup_backlog,
    )
)

sandbox = TemporalSandboxConfig(
    engine=engine,
    profiles=registry.structured_profile_bundle(),
    heartbeat_sink=heartbeat,
    trust_application_agent=True,
)
worker = create_sandbox_worker(
    temporal_client,
    "openbox-sandbox-activities",
    sandbox=sandbox,
)
```

`create_sandbox_worker()` is intentionally command-only: it registers no Workflows and only the defensive governed-command Activity. Use `OpenBoxSandboxPlugin` when adding the command components to an existing Worker.

`trust_application_agent` defaults to `False`. Set it to `True` only when the application agent and Worker are one explicitly owned trust domain and the application has already enforced the strict `CONSTRAIN` boundary. Trusted mode rejects receipt-bearing inputs and constructs no Worker-side Core client.

With the default `False`, configure a receipt verifier and include the signed receipt in `SandboxCommandRequest`. Without either explicit same-domain trust or a valid receipt, the Activity fails closed before sandbox creation. Receipt mode is incompatible with `trust_application_agent=True`.

## Cleanup and cancellation

The engine owns deletion after any create that may have succeeded. Temporal cancellation waits for that cleanup boundary. Success requires deletion followed by terminal absence; failed absence confirmation can be persisted in `CleanupBacklog` and retried by the process owner through `reconcile_cleanup()`.
