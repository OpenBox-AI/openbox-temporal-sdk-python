# Governed sandbox commands

The optional sandbox integration makes a `CONSTRAIN` governance verdict route
the user's own activity into the sandbox transparently. Workflow code never
imports OpenBox; the plugin intercepts the activity at the worker boundary:

- an `ALLOW` verdict runs the activity on the host process as usual; and
- a `CONSTRAIN` verdict derives the command from the activity input through the
  sandbox profile bundle and executes it through the injected governed
  dispatcher, returning the bounded result to the caller.

## Worker composition

The governed command runs inside the native Worker. The plugin registers no
additional Activities of its own:

```python
from temporalio.worker import Worker
from openbox import OpenBoxPlugin

worker = Worker(
    temporal_client,
    task_queue="my-queue",
    workflows=[MyWorkflow],
    plugins=[OpenBoxPlugin(
        openbox_url=...,
        openbox_api_key=...,
        sandbox=sandbox,
    )],
)
```

The application injects its own `GovernedDispatcher` instance and the profile
bundle that maps structured activity input onto command argv:

```python
from openbox.sandbox import TemporalCommandProfileBundle, TemporalSandboxConfig

sandbox = TemporalSandboxConfig(
    dispatcher=dispatcher,   # a GovernedDispatcher
    profiles=profile_bundle, # a TemporalCommandProfileBundle
    heartbeat_sink=heartbeat,
)
```

`TemporalSandboxConfig` validates the composition at construction: the
dispatcher must be a real `GovernedDispatcher`, the profile bundle must be a
`TemporalCommandProfileBundle`, the heartbeat sink must be the dispatcher's own
telemetry sink, and the timeout/heartbeat intervals must be in range.

## Workflow code

Workflow code never imports OpenBox. It calls its own business activity and
passes a structured request (`profile_id` + named arguments) as its input:

```python
from datetime import timedelta
from temporalio import workflow


@workflow.defn
class ReconciliationWorkflow:
    @workflow.run
    async def run(self, batch_id: str):
        return await workflow.execute_activity(
            reconcile,
            {"profile_id": "reconcile", "arguments": {"batch_id": batch_id}},
            start_to_close_timeout=timedelta(minutes=6),
            heartbeat_timeout=timedelta(minutes=2),
        )
```

The plugin intercepts the activity. The governance verdict decides the
execution path: `ALLOW` runs it on the host process; `CONSTRAIN` routes it
into the sandbox automatically. The workflow stays clean. Raw argv,
credentials, policy documents, authority data, and raw command output do not
enter Workflow history.

## Dispatch flow

For each `CONSTRAIN` verdict on a user activity, the interceptor:

1. reads Workflow ID, run ID, Activity ID, Workflow type, task queue, and
   attempt from `temporalio.activity.info()`;
2. validates the single structured activity input (`profile_id` + named
   arguments; a workflow-supplied pre-evaluated `governance` decision is
   honored when present);
3. derives bounded `argv` and `profile_id` from the authenticated profile
   bundle — the input never carries executable text;
4. builds a `GovernedCommand` from genuine `activity.info()` identity and the
   derived argv;
5. maps the `CONSTRAIN` verdict from the ActivityStarted event onto the
   dispatcher decision shape and calls
   `await dispatcher.dispatch_with_decision(command, decision)` — the
   dispatcher executes the verdict without a second governance client;
6. waits on dispatch, heartbeat, and Temporal cancellation together, so
   cancellation cancels the in-flight dispatch and waits for its owned
   cleanup to finish;
7. structurally validates the returned `DispatchResult` and maps it to a
   bounded `GovernedCommandActivityResult` (terminal disposition, exit code,
   timeout and cleanup status, stdout/stderr byte counts, and profile-admitted
   typed result values when the profile declares a result schema).

The host dispatcher is the sole Core caller for this Activity. When its config
carries a governance client it owns the completed sandbox hook and span-free
`ActivityCompleted`; otherwise the interceptor wrapper posts `ActivityCompleted`
with the mapped bounded output. The bounded result is not a portable signed
runtime receipt or independent proof of execution.

**Host results are rejected after dispatch.** A real `GovernedDispatcher` may
already have run an `ALLOW` command as a host subprocess before returning
`executed_on_host`; Temporal then rejects that result, but it did not prevent
the host attempt. Preventing host execution belongs to Core policy: a zero-host
deployment must ensure the applicable Core decision is exactly `CONSTRAIN` and
deploy the dispatcher so its host path is unavailable or otherwise disabled.
The dispatcher's `HALT` directive terminates the workflow after the sandbox run.

Natural mode has a two-bundle trust assumption. Temporal's
`TemporalCommandProfileBundle` derives `argv`, while the dispatcher's
`CommandProfileBundle` independently re-admits it. Both bundles must come from
equivalent command definitions and bundle versions; disagreement fails closed
rather than authorizing profile drift.

## Result limits

Before typed-result parsing, stdout and stderr are each limited to 1 MiB and
their combined size to 2 MiB. Host results, missing/nonterminal execution,
malformed metadata, and oversized output fail closed. Raw output never enters
Workflow history.

## Cleanup and cancellation

The natural dispatcher must own deletion after any create that may have
succeeded and must not finish cancellation before that cleanup boundary.
Temporal cancellation cancels the in-flight dispatch task and waits for it to
finish cleanup. Result mapping requires accepted terminal execution metadata,
and cleanup status remains explicit in that reported metadata. Cleanup failures
may be persisted in the dispatcher's `CleanupBacklog` and retried by the
process owner through `reconcile_cleanup()`.
