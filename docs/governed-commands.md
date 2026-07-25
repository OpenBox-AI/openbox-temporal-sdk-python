# Governed sandbox commands

Install the optional Temporal contracts and compatibility engine with:

```bash
pip install "openbox-temporal-sdk-python[sandbox]"
```

The Temporal package owns bounded history conversion, one-attempt Activity scheduling, Temporal identity binding, heartbeat/cancellation behavior, Worker/plugin registration, and bounded result mapping. In production, the injected governed dispatcher owns governance and execution. `openbox-sandbox-sdk-python` continues to provide profile/history contracts and the pre-authorized compatibility engine.

## Production dispatch flow

The governed-dispatcher package is not a dependency of this package. The application injects both its dispatcher and its real `GovernedCommand` class as a factory:

```python
from governed_dispatcher import GovernedCommand, GovernedDispatcher
from openbox.sandbox import TemporalSandboxConfig

sandbox = TemporalSandboxConfig(
    engine=None,
    profiles=registry.structured_profile_bundle(),
    heartbeat_sink=heartbeat,
    dispatcher=dispatcher,  # a GovernedDispatcher
    governed_command_factory=GovernedCommand,
)
```

For each `openbox_governed_command` Activity, the adapter:

1. accepts only the registered Activity type and rejects any attempt other than the first;
2. reads Workflow ID, run ID, Activity ID, Workflow type, task queue, and attempt from `temporalio.activity.info()`;
3. derives bounded `argv` and `profile_id` from the authenticated profile bundle;
4. calls the injected factory exactly as:

   ```python
   governed_command_factory(
       workflow_id=info.workflow_id,
       run_id=info.workflow_run_id,
       activity_id=info.activity_id,
       argv=derived_argv,
       profile_id=request.profile_id,
       timeout_seconds=config.timeout_seconds,
       workflow_type=info.workflow_type,
       task_queue=info.task_queue,
       attempt=info.attempt,
   )
   ```

5. invokes only `await dispatcher.dispatch(command)` on the dispatcher; and
6. structurally validates the returned governed-dispatcher `DispatchResult` and `ExecutionMetadata` before mapping it to `SandboxActivityResult`.

The host dispatcher is also the sole Core caller for this Activity. It reuses its preflight signer/agent identity to attach a completed `sandbox_execution` hook span to the existing `ActivityStarted` event, then sends a separate span-free `ActivityCompleted` on the normal path. The Temporal sandbox worker installs no Core transport or OTLP exporter and never duplicates those calls.

The result must report `executed_in_sandbox`, carry no error, include terminal execution metadata, byte-valued `stdout`/`stderr`, an exit code from 0 through 2³¹−1, an accepted terminal timeout status, and `deleted` or `failed` cleanup status. Before typed-result parsing, stdout and stderr are each limited to 1 MiB and their combined size to 2 MiB. Host results, missing/nonterminal execution, malformed metadata, and oversized output fail closed. Raw output never enters Workflow history.

**Host-result rejection happens after dispatch.** A real `GovernedDispatcher` may already have run an `ALLOW` command as a host subprocess before returning `executed_on_host`; Temporal then rejects that result, but it did not prevent the host attempt. Preventing host execution belongs to Core policy and dispatcher deployment. A zero-host deployment must ensure the applicable Core decision is exactly `CONSTRAIN` and deploy the dispatcher so its host path is unavailable or otherwise disabled. Stock Temporal result validation is not host-path enforcement, and this integration does not claim a stock dispatcher disable flag.

Natural mode also has a two-bundle trust assumption. Temporal's `StructuredCommandProfileBundle` derives `argv`, while the dispatcher's `CommandProfileBundle` independently re-admits it. Both bundles must come from equivalent command definitions and bundle versions; disagreement fails closed rather than authorizing profile drift.

Natural mode is valid only when `trust_application_agent=False`, `receipt_verifier=None`, `engine=None`, and both `dispatcher` and `governed_command_factory` are supplied. A missing half of the injection seam fails during configuration.

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

The helper sets `RetryPolicy(maximum_attempts=1)` and `WAIT_CANCELLATION_COMPLETED`. Raw argv, credentials, policy documents, authority data, and raw command output do not enter Workflow history. A profile may return bounded typed JSON values; otherwise the durable result contains execution metadata only.

An authorization receipt records permission to execute; it does not show that execution occurred. The bounded Activity result reports terminal disposition, exit code, timeout and cleanup status, and output byte counts correlated with Temporal lifecycle signals. It is not a portable signed runtime receipt or independent proof of execution.

## Dedicated Worker

```python
from openbox.sandbox import create_sandbox_worker

worker = create_sandbox_worker(
    temporal_client,
    "openbox-sandbox-activities",
    sandbox=sandbox,
)
```

`create_sandbox_worker()` is intentionally command-only: it registers no Workflows and only the defensive governed-command Activity. Use `OpenBoxSandboxPlugin` when adding the command components to an existing Worker.

## Compatibility modes

These paths are preserved for existing deployments. They bypass natural governed dispatch and cannot be combined with a dispatcher or governed-command factory. Both require a valid `SandboxExecutionEngine` in `engine`.

### Pre-authorized trusted-agent compatibility

Set `trust_application_agent=True` only when the application agent and Worker are one explicitly owned trust domain and the application has already enforced the strict `CONSTRAIN` boundary. This compatibility mode calls the sandbox engine with internally derived trusted authorization. It rejects receipt-bearing inputs and constructs no Worker-side Core client.

### Authorization-receipt compatibility

Set `trust_application_agent=False`, provide `receipt_verifier`, and include the signed authorization receipt in `SandboxCommandRequest`. This compatibility mode verifies the receipt before calling the sandbox engine. A valid receipt proves authorization only; do not present it as evidence that the command ran.

## Cleanup and cancellation

The natural dispatcher must own deletion after any create that may have succeeded and must not finish cancellation before that cleanup boundary. The compatibility engine provides the same guarantee. Temporal cancellation cancels the in-flight dispatch/engine task and waits for it to finish cleanup. Result mapping requires accepted terminal execution metadata, and cleanup status remains explicit in that reported metadata. Compatibility cleanup failures may be persisted in `CleanupBacklog` and retried by the process owner through `reconcile_cleanup()`.
